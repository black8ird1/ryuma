"""Live model catalog — a new model must never cost a code edit or a restart.

A hard-coded model tuple is a chore with a release schedule: the provider ships
something, somebody edits the list, redeploys, and restarts every bot. This module
deletes that job. The catalog is fetched from the provider's own `/v1/models`
endpoint, cached on disk with a TTL, refreshed on a background thread, and consulted
per turn — so a model released an hour ago appears in `/model` on a bot that has been
running for a week, untouched.

Three properties every caller depends on:

* **Never blocks a turn.** A cold cache answers instantly from the seed list and the
  network fetch happens on a daemon thread.
* **Never empty.** Network down, token expired, endpoint moved → last good cache,
  else the compiled-in seed. A gateway with no internet still starts.
* **Self-correcting.** A model the locally installed CLI is too old to run is
  quarantined AGAINST THAT CLI VERSION, so alias resolution walks down to one that
  works — and the moment the CLI is upgraded the quarantine stops matching and the
  new model comes back on its own. Nothing to un-do by hand.

Aliases are the other half of the deal. Pin `CLAUDE_MODEL=latest` and the gateway
tracks the newest model of a preferred family forever; `latest-sonnet` tracks that
family; a concrete id still pins exactly.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]

# Vendor words that lead a model id without naming its family ("claude-opus-5-5").
_VENDOR_PREFIXES = ("claude", "anthropic", "openai")

# Families `latest` is willing to pick, best first. Fable is deliberately absent:
# it is newer than some Opus releases but is not the coding model, and `latest` on a
# coding gateway must not drift into it. `latest-fable` still selects it explicitly,
# and `latest-any` means the literal newest thing the provider lists.
DEFAULT_PREFERRED_FAMILIES = ("opus", "sonnet", "haiku")

_ALIAS_HEADS = ("latest", "newest", "auto")

# A quarantine with no CLI version attached expires on the clock instead.
QUARANTINE_TTL_SEC = 7 * 24 * 3600


@dataclass(frozen=True)
class ModelInfo:
    """One model as the provider describes it. `created_at` drives ordering."""

    id: str
    display_name: str = ""
    created_at: str = ""

    @property
    def family(self) -> str:
        return model_family(self.id)


def model_family(model_id: str) -> str:
    """'claude-opus-5-5' -> 'opus'; 'gpt-5.6-sol' -> 'gpt'. The family is the first
    token that isn't a vendor name, so a provider renaming its prefix doesn't break
    alias resolution."""
    parts = [p for p in re.split(r"[-_/]", model_id.strip().lower()) if p]
    for part in parts:
        if part not in _VENDOR_PREFIXES:
            return part
    return parts[0] if parts else ""


def parse_alias(name: str) -> tuple[bool, str]:
    """('latest-opus') -> (True, 'opus'); ('latest') -> (True, ''); a concrete id ->
    (False, ''). 'latest-any' asks for the literal newest, encoded as family 'any'."""
    token = (name or "").strip().lower()
    if not token:
        return False, ""
    for head in _ALIAS_HEADS:
        if token == head:
            return True, ""
        if token.startswith(head + "-"):
            return True, token[len(head) + 1 :]
    return False, ""


def _anthropic_auth() -> tuple[str, str] | None:
    """(header, value) for whichever credential this host actually has. Claude Code
    logins carry an OAuth token, not an API key, and the models endpoint accepts it
    as a bearer token — so a gateway that can run `claude` can also list models."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return "x-api-key", key
    for name in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            return "Authorization", f"Bearer {token}"
    return None


def fetch_anthropic_models(timeout: float = 8.0) -> list[ModelInfo]:
    """GET /v1/models. Raises on any failure — the caller decides what to fall back
    to, because a fetch failure must never be mistaken for 'no models exist'."""
    auth = _anthropic_auth()
    if auth is None:
        raise RuntimeError("no Anthropic credential in env")
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    req = urllib.request.Request(
        f"{base}/v1/models?limit=100",
        headers={auth[0]: auth[1], "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed host
        payload = json.loads(resp.read().decode("utf-8"))
    out = []
    for row in payload.get("data") or []:
        mid = str(row.get("id") or "").strip()
        if mid:
            out.append(ModelInfo(mid, str(row.get("display_name") or ""), str(row.get("created_at") or "")))
    if not out:
        raise RuntimeError("models endpoint returned an empty list")
    return out


def claude_cli_version(claude_bin: str = "claude") -> str:
    """'2.1.280' from `claude --version`, or '' if it can't be read. Used as the
    quarantine scope, so upgrading the CLI silently re-admits every model that was
    only ever rejected for being too new."""
    try:
        proc = subprocess.run([claude_bin, "--version"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    match = re.search(r"\d+\.\d+\.\d+", proc.stdout or "")
    return match.group(0) if match else ""


class ModelCatalog:
    """Disk-cached, background-refreshed list of the models this account can run.

    Thread-safe: the Telegram thread reads it to draw `/model` while a backend thread
    resolves an alias mid-turn and a refresh thread replaces the contents.
    """

    def __init__(
        self,
        *,
        seed: Sequence[str] = (),
        cache_path: str | Path | None = None,
        ttl_sec: float | None = None,
        fetcher: Callable[[], list[ModelInfo]] | None = None,
        preferred: Sequence[str] | None = None,
        auto_refresh: bool | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._seed = tuple(dict.fromkeys(seed))
        self._fetcher = fetcher or fetch_anthropic_models
        self._cache_path = Path(
            cache_path
            or os.environ.get("AGENT_GATEWAY_MODEL_CATALOG", str(ROOT / "state" / "agent-gateway" / "model-catalog.json"))
        )
        self._ttl = float(ttl_sec if ttl_sec is not None else os.environ.get("AGENT_GATEWAY_MODEL_CATALOG_TTL", "21600"))
        env_pref = os.environ.get("AGENT_GATEWAY_MODEL_PREFER", "")
        self._preferred = tuple(
            p.strip().lower() for p in (preferred or (env_pref.split(",") if env_pref else DEFAULT_PREFERRED_FAMILIES)) if p.strip()
        )
        self._auto = (
            auto_refresh
            if auto_refresh is not None
            else os.environ.get("AGENT_GATEWAY_MODEL_CATALOG_REFRESH", "1") != "0"
        )
        self._models: tuple[ModelInfo, ...] = ()
        self._quarantine: dict[str, dict] = {}
        self._fetched_at = 0.0
        self._refreshing = False
        self._thread: threading.Thread | None = None
        self._load_cache()

    # ---------------------------------------------------------------- persistence

    def _load_cache(self) -> None:
        try:
            data = json.loads(self._cache_path.read_text())
        except (OSError, ValueError):
            return
        with self._lock:
            self._models = tuple(
                ModelInfo(str(m.get("id")), str(m.get("display_name") or ""), str(m.get("created_at") or ""))
                for m in (data.get("models") or [])
                if m.get("id")
            )
            self._quarantine = {str(k): v for k, v in (data.get("quarantine") or {}).items() if isinstance(v, dict)}
            self._fetched_at = float(data.get("fetched_at") or 0)

    def _save_cache(self) -> None:
        """Atomic write. A truncated catalog would be read back as 'no models' and
        silently downgrade every bot on the box, so never write in place."""
        with self._lock:
            payload = {
                "fetched_at": self._fetched_at,
                "models": [{"id": m.id, "display_name": m.display_name, "created_at": m.created_at} for m in self._models],
                "quarantine": self._quarantine,
            }
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self._cache_path)
        except OSError:
            pass

    # ------------------------------------------------------------------- refresh

    def _stale(self) -> bool:
        return (time.time() - self._fetched_at) > self._ttl

    def refresh(self, *, force: bool = False) -> bool:
        """Blocking fetch. Returns True when the catalog actually changed."""
        if not force and not self._stale():
            return False
        try:
            models = self._fetcher()
        except Exception as exc:  # noqa: BLE001 - any failure keeps the cache
            print(f"model-catalog: refresh failed ({type(exc).__name__}: {exc}); keeping cache", flush=True)
            return False
        with self._lock:
            changed = tuple(m.id for m in models) != tuple(m.id for m in self._models)
            self._models = tuple(models)
            self._fetched_at = time.time()
        self._save_cache()
        if changed:
            print(f"model-catalog: {len(models)} models, newest={models[0].id if models else '-'}", flush=True)
        return changed

    def _refresh_async(self) -> None:
        """Kick one background refresh if the cache is stale and none is in flight.
        Callers are on the turn path — this must return in microseconds."""
        if not self._auto:
            return
        with self._lock:
            if self._refreshing or not self._stale():
                return
            self._refreshing = True

        def run() -> None:
            try:
                self.refresh(force=True)
            finally:
                with self._lock:
                    self._refreshing = False

        self._thread = threading.Thread(target=run, name="model-catalog", daemon=True)
        self._thread.start()

    def start_auto_refresh(self) -> None:
        """Long-lived ticker: re-fetch every TTL so a week-old process still knows
        about this morning's release without anyone touching it."""
        if not self._auto:
            return

        def loop() -> None:
            while True:
                try:
                    self.refresh(force=True)
                except Exception:  # noqa: BLE001 - a ticker must never die
                    pass
                time.sleep(max(60.0, self._ttl) * random.uniform(0.9, 1.1))

        threading.Thread(target=loop, name="model-catalog-ticker", daemon=True).start()

    # ---------------------------------------------------------------- quarantine

    def quarantine(self, model_id: str, *, reason: str = "", scope: str = "") -> None:
        """Mark a model unusable HERE. `scope` should be the local CLI version: the
        entry only applies while that version is installed, so an upgrade is the
        un-quarantine and nobody has to remember this ever happened."""
        if not model_id:
            return
        with self._lock:
            self._quarantine[model_id] = {"reason": reason, "scope": scope, "at": time.time()}
        self._save_cache()
        print(f"model-catalog: quarantined {model_id} (scope={scope or 'time'}): {reason}", flush=True)

    def is_quarantined(self, model_id: str, *, scope: str = "") -> bool:
        with self._lock:
            entry = self._quarantine.get(model_id)
        if not entry:
            return False
        entry_scope = str(entry.get("scope") or "")
        if entry_scope:
            return entry_scope == scope
        return (time.time() - float(entry.get("at") or 0)) < QUARANTINE_TTL_SEC

    def has_quarantine(self) -> bool:
        """Whether any model is currently shunned. Callers use this to skip working
        out a scope at all — computing one costs a subprocess, and with an empty
        quarantine the scope cannot change a single answer."""
        with self._lock:
            return bool(self._quarantine)

    def quarantine_reason(self, model_id: str) -> str:
        with self._lock:
            return str((self._quarantine.get(model_id) or {}).get("reason") or "")

    # ------------------------------------------------------------------- reading

    def models(self, *, scope: str = "", include_quarantined: bool = False) -> tuple[ModelInfo, ...]:
        """Newest first. Seeds fill in for any known-good id the provider didn't
        return (or that we've never fetched), so the list is never shorter than the
        compiled-in one."""
        self._refresh_async()
        with self._lock:
            live = list(self._models)
        known = {m.id for m in live}
        live.sort(key=lambda m: m.created_at, reverse=True)
        tail = [ModelInfo(mid) for mid in self._seed if mid not in known]
        out = live + tail
        if include_quarantined:
            return tuple(out)
        return tuple(m for m in out if not self.is_quarantined(m.id, scope=scope))

    def ids(self, *, scope: str = "") -> tuple[str, ...]:
        return tuple(m.id for m in self.models(scope=scope))

    def suggestions(self, *, limit: int = 8, scope: str = "") -> tuple[str, ...]:
        """What `/model` draws as tap-buttons: the newest of EVERY family first, then
        the rest by recency, capped so the keyboard stays thumb-sized on a phone.

        Family-champions-first matters — ranking purely by preference buries whole
        families (six Opus releases would push Sonnet, Haiku and Fable off an 8-slot
        keyboard), and the picker's job is to show the operator every kind of model
        they own, newest of each."""
        models = self.models(scope=scope)
        if not models:
            return ()
        recency = {m.family: i for i, m in reversed(list(enumerate(models)))}  # first (newest) wins
        families = sorted(
            recency,
            key=lambda f: (self._preferred.index(f) if f in self._preferred else len(self._preferred), recency[f]),
        )
        order = [next(m.id for m in models if m.family == f) for f in families]
        order += [m.id for m in models if m.id not in order]
        return tuple(order)[:limit]

    def resolve(self, name: str, *, scope: str = "") -> str:
        """Alias or concrete id -> the id to actually run.

        A concrete id passes through untouched unless it is quarantined here, in
        which case we fall to the newest usable model of the SAME family — the bot
        keeps answering on a slightly older model instead of erroring every turn.
        """
        requested = (name or "").strip()
        is_alias, family = parse_alias(requested)
        usable = self.models(scope=scope)
        if not usable:
            return "" if is_alias else requested
        if is_alias:
            if family and family != "any":
                match = next((m for m in usable if m.family == family), None)
                return match.id if match else ""
            if family == "any":
                return usable[0].id
            preferred = [m for m in usable if m.family in self._preferred]
            return (preferred or list(usable))[0].id
        if not self.is_quarantined(requested, scope=scope):
            return requested
        want = model_family(requested)
        match = next((m for m in usable if m.family == want), None) or usable[0]
        return match.id


# Matches the CLI's own refusal: "API Error: 400 Claude Code 2.1.257 does not support
# this model; version 2.1.280 or newer is required." The required version is the
# actionable half — it is what the operator has to install — so capture it.
_TOO_OLD = re.compile(
    r"Claude Code\s+(?P<have>\d+\.\d+\.\d+)\s+does not support this model;\s*version\s+(?P<need>\d+\.\d+\.\d+)",
    re.IGNORECASE,
)
_UNRECOGNIZED = re.compile(r"unrecognized_model|does not support this model", re.IGNORECASE)


def cli_too_old(text: str) -> tuple[str, str] | None:
    """(have, need) when the reply is the CLI refusing a model it's too old to run;
    None for every other failure. ('', '') when the shape is recognised but the
    versions aren't quoted — still a model rejection, still worth downgrading."""
    if not text:
        return None
    match = _TOO_OLD.search(text)
    if match:
        return match.group("have"), match.group("need")
    if _UNRECOGNIZED.search(text):
        return "", ""
    return None


def upgrade_claude_cli(command: str = "") -> tuple[bool, str]:
    """Run the CLI's own upgrade. Opt-in (AGENT_GATEWAY_CLAUDE_AUTO_UPDATE=1): this
    installs software as whatever user the gateway runs as, which must be the
    operator's explicit choice, not a default. Safe for running bots — replacing the
    binary on disk does not touch processes that already exec'd it; the next worker
    picks up the new one."""
    cmd = command or os.environ.get(
        "AGENT_GATEWAY_CLAUDE_UPDATE_CMD", "npm install -g @anthropic-ai/claude-code@latest"
    )
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()
    return proc.returncode == 0, (tail[-1] if tail else "")


def default_catalog(seed: Iterable[str] = ()) -> ModelCatalog:
    return ModelCatalog(seed=tuple(seed))
