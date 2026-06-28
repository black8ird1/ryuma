"""First-run onboarding wizard for the Agent Gateway.

One interactive flow takes a brand-new operator from nothing to a running bot:

  1. preflight — git, python, and WHICH agent CLIs are installed (claude/codex)
  2. create a Telegram bot in BotFather → paste the token (verified live via getMe)
  3. AUTO-CAPTURE the operator's Telegram user id: just message the bot once, we
     read it off getUpdates. No hunting for a numeric id — the one step that
     trips up every other bot's setup.
  4. pick the default agent
  5. write users.env + <agent>.env (0600)
  6. offer to start now

Everything pure — token shape, getMe / getUpdates parsing, agent detection — is a
module-level function so it unit-tests with no network and no TTY. The interactive
run() wires those to real stdin and the Telegram HTTP API.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable

from agent_gateway.profiles import write_profile_template, write_users_env
from agent_gateway.project import brand_emoji, brand_name

# BotFather tokens look like 123456789:AA... (numeric id, colon, 35-char secret).
TOKEN_RE = re.compile(r"^\d{6,}:[A-Za-z0-9_-]{30,}$")

# Agents we know how to drive, in the order we offer them. mock needs no CLI.
KNOWN_AGENTS = ("claude", "codex")

HttpFn = Callable[[str, str, dict], dict]  # (method, token, params) -> parsed JSON


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no I/O)
# --------------------------------------------------------------------------- #
def validate_token_shape(token: str) -> bool:
    """Cheap local shape check before we spend a network round-trip on getMe."""
    return bool(TOKEN_RE.match(token.strip()))


def parse_getme(payload: dict) -> str | None:
    """Bot username from a getMe response, or None if the call wasn't ok."""
    if not isinstance(payload, dict) or not payload.get("ok"):
        return None
    result = payload.get("result") or {}
    username = result.get("username")
    return str(username) if username else None


def first_user_id(updates_payload: dict) -> int | None:
    """First human's numeric id from a getUpdates response.

    Scans message / edited_message / channel_post shapes so any kind of inbound
    update captures the operator. Returns None when nothing usable arrived yet."""
    if not isinstance(updates_payload, dict) or not updates_payload.get("ok"):
        return None
    for update in updates_payload.get("result") or []:
        for key in ("message", "edited_message", "channel_post", "callback_query"):
            container = update.get(key) or {}
            sender = container.get("from") or {}
            uid = sender.get("id")
            if isinstance(uid, int):
                return uid
    return None


def detect_agents(which: Callable[[str], str | None] = shutil.which) -> list[str]:
    """Which known agent CLIs are on PATH. mock is always appended (no CLI)."""
    found = [a for a in KNOWN_AGENTS if which(a)]
    found.append("mock")
    return found


# --------------------------------------------------------------------------- #
# Live Telegram HTTP (injectable for tests)
# --------------------------------------------------------------------------- #
def telegram_http(method: str, token: str, params: dict) -> dict:
    """Minimal Bot API call over stdlib urllib. Returns parsed JSON (or {ok:False})."""
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(params).encode() if params else None
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:  # 401/404 still carry a JSON body
        try:
            return json.loads(exc.read().decode())
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001 - network/timeouts must not crash the wizard
        return {"ok": False, "error": str(exc)}


def capture_user_id(
    token: str,
    *,
    http: HttpFn = telegram_http,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = 40,
    interval: float = 1.5,
) -> int | None:
    """Poll getUpdates until the operator's message lands; return their id.

    Drains any stale backlog first (offset = last+1) so we only capture a FRESH
    message sent during onboarding — not an old one from a previous run."""
    drained = http("getUpdates", token, {"timeout": 0})
    offset = 0
    for upd in (drained.get("result") or []) if isinstance(drained, dict) else []:
        offset = max(offset, int(upd.get("update_id", 0)) + 1)
    for _ in range(attempts):
        payload = http("getUpdates", token, {"timeout": 0, "offset": offset})
        uid = first_user_id(payload)
        if uid is not None:
            return uid
        if isinstance(payload, dict):
            for upd in payload.get("result") or []:
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
        sleep(interval)
    return None


# --------------------------------------------------------------------------- #
# Interactive flow
# --------------------------------------------------------------------------- #
@dataclass
class OnboardResult:
    profile: str
    agent: str
    user_id: int
    username: str | None


BOTFATHER_STEPS = (
    "Let's create your Telegram bot (takes ~30 seconds):\n"
    "  1. Open Telegram and message  @BotFather\n"
    "  2. Send  /newbot  and follow the prompts (name + username)\n"
    "  3. BotFather replies with a token like  123456789:AAE...\n"
)


def run(
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
    http: HttpFn = telegram_http,
    which: Callable[[str], str | None] = shutil.which,
    sleep: Callable[[float], None] = time.sleep,
) -> OnboardResult | None:
    """Interactive onboarding. Returns the result, or None if the user aborts."""
    p = print_fn
    p(f"\n{brand_emoji()}  {brand_name()} — setup\n" + "=" * 32)

    # 1. preflight ----------------------------------------------------------- #
    agents = detect_agents(which)
    cli_agents = [a for a in agents if a != "mock"]
    if cli_agents:
        p(f"✓ Found agent CLI(s): {', '.join(cli_agents)}")
    else:
        p("⚠ No agent CLI found on PATH (claude / codex).")
        p("  You can still finish setup with the 'mock' agent and install one later.")

    # 2. token --------------------------------------------------------------- #
    p("\n" + BOTFATHER_STEPS)
    token = ""
    username = None
    while True:
        token = input_fn("Paste your bot token (or 'q' to quit): ").strip()
        if token.lower() in {"q", "quit", ""}:
            p("Aborted. Run setup again any time.")
            return None
        if not validate_token_shape(token):
            p("✗ That doesn't look like a BotFather token. Try again.")
            continue
        username = parse_getme(http("getMe", token, {}))
        if username:
            p(f"✓ Connected to @{username}")
            break
        p("✗ Telegram rejected that token. Double-check you copied all of it.")

    # 3. auto-capture user id ------------------------------------------------ #
    p(
        f"\nNow open Telegram, find  @{username}, and send it any message "
        "(e.g. 'hi').\nThis is how we learn your account id — no need to look it up."
    )
    input_fn("Press Enter once you've sent the message... ")
    p("Listening for your message...")
    user_id = capture_user_id(token, http=http, sleep=sleep)
    while user_id is None:
        p("✗ Didn't see a message yet.")
        retry = input_fn("Send one and press Enter to retry, or paste your numeric id: ").strip()
        if retry.isdigit():
            user_id = int(retry)
            break
        user_id = capture_user_id(token, http=http, sleep=sleep)
    p(f"✓ Captured your id: {user_id}")

    # 4. pick agent ---------------------------------------------------------- #
    default_agent = cli_agents[0] if cli_agents else "mock"
    p(f"\nDefault agent for this bot? {agents}  [{default_agent}]")
    choice = input_fn(f"Agent [{default_agent}]: ").strip().lower()
    agent = choice if choice in agents else default_agent

    # 5. write profiles ------------------------------------------------------ #
    write_users_env([user_id])
    profile_name = agent
    profile_path = write_profile_template(
        profile_name, backend=agent, token=token, overwrite=True
    )
    p(f"\n✓ Wrote allowlist + profile: {profile_path}")

    # 6. start? -------------------------------------------------------------- #
    p("\nSetup complete. 🎉")
    p(f"  Start now:   scripts/agent_gateway_start.sh {profile_name}")
    p(f"  Then message @{username} on Telegram and just talk to it.")
    return OnboardResult(profile=profile_name, agent=agent, user_id=user_id, username=username)


def _join(items: Iterable[str]) -> str:
    return ", ".join(items)
