"""Simple one-bot profile files for the agent gateway.

The gateway can still be configured through AGENT_GATEWAY_* env vars, but the
operator path is intentionally smaller:

    # users.env
    ALLOWED_USER_IDS=111

    # codex.env
    TELEGRAM_TOKEN=123:abc
    AGENT=codex

Run with:

    python3 scripts/agent_gateway_bot.py --profile codex
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIR = ROOT / "state" / "agent-gateway" / "profiles"
SHARED_PROFILE_NAME = "users.env"

ALIASES = {
    "TELEGRAM_TOKEN": "AGENT_GATEWAY_TELEGRAM_TOKEN",
    "BOT_TOKEN": "AGENT_GATEWAY_TELEGRAM_TOKEN",
    "ALLOWED_USER_IDS": "AGENT_GATEWAY_ALLOWED_USER_IDS",
    "TELEGRAM_ALLOWED_USER_IDS": "AGENT_GATEWAY_ALLOWED_USER_IDS",
    "USER_ID": "AGENT_GATEWAY_ALLOWED_USER_IDS",
    "BOT_USERNAME": "AGENT_GATEWAY_BOT_USERNAME",
    "USERNAME": "AGENT_GATEWAY_BOT_USERNAME",
    "AGENT": "AGENT_GATEWAY_DEFAULT_BACKEND",
    "BACKEND": "AGENT_GATEWAY_DEFAULT_BACKEND",
    "BACKENDS": "AGENT_GATEWAY_BACKENDS",
    "FIXED_BACKEND": "AGENT_GATEWAY_FIXED_BACKEND",
    "AUTO_WORKTREE": "AGENT_GATEWAY_AUTO_WORKTREE",
    "WORKTREE_ROOT": "AGENT_GATEWAY_WORKTREE_ROOT",
    "AUTO_COMMIT": "AGENT_GATEWAY_AUTO_COMMIT",
    "ALLOW_PUSH": "AGENT_GATEWAY_ALLOW_PUSH",
    "ALLOW_MERGE": "AGENT_GATEWAY_ALLOW_MERGE",
    "CODEX_BIN": "AGENT_GATEWAY_CODEX_BIN",
    "CODEX_MODEL": "AGENT_GATEWAY_CODEX_MODEL",
    "CODEX_MAX_SESSION_TURNS": "AGENT_GATEWAY_CODEX_MAX_SESSION_TURNS",
    "CLAUDE_BIN": "AGENT_GATEWAY_CLAUDE_BIN",
    "CLAUDE_MODEL": "AGENT_GATEWAY_CLAUDE_MODEL",
    "CLAUDE_MAX_SESSION_TURNS": "AGENT_GATEWAY_CLAUDE_MAX_SESSION_TURNS",
    "CLAUDE_THINKING_TOKENS": "AGENT_GATEWAY_CLAUDE_THINKING_TOKENS",
}


@dataclass(frozen=True)
class ProfileLoadResult:
    name: str
    path: Path
    values: dict[str, str]


def profile_path(name_or_path: str, *, profiles_dir: Path = DEFAULT_PROFILE_DIR) -> Path:
    raw = name_or_path.strip()
    if not raw:
        raise ValueError("profile name is required")
    candidate = Path(raw)
    if candidate.suffix == ".env" or "/" in raw:
        return candidate if candidate.is_absolute() else (ROOT / candidate)
    return profiles_dir / f"{raw}.env"


def parse_profile_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"{path}:{lineno}: expected KEY=value")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{lineno}: empty key")
        values[_normalize_key(key)] = _parse_value(value.strip())
    return values


def apply_profile(
    name_or_path: str,
    *,
    environ: MutableMapping[str, str] | None = None,
    profiles_dir: Path = DEFAULT_PROFILE_DIR,
) -> ProfileLoadResult:
    env = environ if environ is not None else os.environ
    path = profile_path(name_or_path, profiles_dir=profiles_dir)
    if not path.exists():
        raise FileNotFoundError(f"profile not found: {path}")
    values: dict[str, str] = {}
    shared_path = path.parent / SHARED_PROFILE_NAME
    if shared_path.exists():
        values.update(parse_profile_file(shared_path))
    values.update(parse_profile_file(path))
    _apply_profile_defaults(values, profile_name=path.stem)
    for key, value in values.items():
        env[key] = value
    return ProfileLoadResult(name=path.stem, path=path, values=values)


def write_shared_profile_template(
    *,
    profiles_dir: Path = DEFAULT_PROFILE_DIR,
) -> Path:
    path = profiles_dir / SHARED_PROFILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            """# Shared Telegram allowlist for every Agent Gateway bot.
# Fill this once with your numeric Telegram user ID.
# Multiple users: comma-separated, e.g. ALLOWED_USER_IDS=111,222
ALLOWED_USER_IDS=
"""
        )
        path.chmod(0o600)
    return path


def write_profile_template(
    name: str,
    *,
    backend: str | None = None,
    token: str = "",
    overwrite: bool = False,
    profiles_dir: Path = DEFAULT_PROFILE_DIR,
) -> Path:
    path = profile_path(name, profiles_dir=profiles_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_shared_profile_template(profiles_dir=path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"profile already exists: {path}")
    backend = (backend or name).strip() or "codex"
    text = f"""# One Telegram bot = one token + its DEFAULT agent. Everything else (agent config,
# git safety, Live view, allowlist) is shared once in users.env. Paste the BotFather
# token, then start it: scripts/agent_gateway_start.sh {name}
TELEGRAM_TOKEN={token}

# Default agent this bot opens on: mock, codex, claude, or a future backend.
# Every bot runs them all — switch live from the bot via /model.
AGENT={backend}
"""
    path.write_text(text)
    path.chmod(0o600)
    return path


def write_users_env(
    user_ids: "list[str | int]",
    *,
    profiles_dir: Path = DEFAULT_PROFILE_DIR,
) -> Path:
    """Write the shared allowlist with concrete ids (onboarding auto-capture path).

    Unlike write_shared_profile_template (which seeds an EMPTY allowlist for hand
    editing), this stamps the ids the wizard captured so the operator never types
    a numeric Telegram id by hand."""
    path = profiles_dir / SHARED_PROFILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = ",".join(str(u).strip() for u in user_ids if str(u).strip())
    path.write_text(
        "# Shared Telegram allowlist for every Agent Gateway bot (written by onboarding).\n"
        "# Add more operators with a comma, e.g. ALLOWED_USER_IDS=111,222\n"
        f"ALLOWED_USER_IDS={ids}\n"
    )
    path.chmod(0o600)
    return path


def _normalize_key(key: str) -> str:
    upper = key.strip().upper()
    if upper.startswith("AGENT_GATEWAY_"):
        return upper
    return ALIASES.get(upper, upper)


def _parse_value(value: str) -> str:
    if not value:
        return ""
    try:
        parts = shlex.split(value, comments=True, posix=True)
    except ValueError:
        return value.strip().strip("\"'")
    if len(parts) == 1:
        return parts[0]
    return value.strip().strip("\"'")


def _apply_profile_defaults(values: dict[str, str], *, profile_name: str) -> None:
    backend = values.get("AGENT_GATEWAY_DEFAULT_BACKEND", "").strip()
    values.setdefault("AGENT_GATEWAY_STATE_DIR", str(ROOT / "state" / "agent-gateway" / "runtime" / profile_name))
    # Every bot runs ALL agents and lets you switch live from the bot (/model). The
    # profile's AGENT is only the DEFAULT; FIXED_BACKEND=0 unlocks switching. (users.env
    # usually sets both, so these are just the fallback when a profile stands alone.)
    if backend and "AGENT_GATEWAY_BACKENDS" not in values:
        values["AGENT_GATEWAY_BACKENDS"] = "mock" if backend == "mock" else "mock,codex,claude"
    values.setdefault("AGENT_GATEWAY_FIXED_BACKEND", "0")
    if backend != "mock":
        values.setdefault("AGENT_GATEWAY_AUTO_WORKTREE", "1")
        values.setdefault("AGENT_GATEWAY_AUTO_COMMIT", "1")
        values.setdefault("AGENT_GATEWAY_ALLOW_PUSH", "0")
        values.setdefault("AGENT_GATEWAY_ALLOW_MERGE", "0")
