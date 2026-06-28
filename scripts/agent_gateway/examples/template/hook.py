"""TEMPLATE brain hook — a worked example of the AugmentationHook seam.

This file is NOT part of the bot engine. It's the handler side of the pluggable
hook: the engine calls it at turn boundaries so you can give every agent shared
project context and record what they did. Your knowledge system stays entirely
outside the (publishable) engine — this is where a private edge would live.

Enable it from a profile:
    AGENT_GATEWAY_HOOK=scripts/agent_gateway/examples/template/hook.py:make_hook

Contract (all four are optional; a missing method just does nothing):
    before_turn(turn)            -> str   context injected into the prompt
    after_turn(turn, final_text) -> None  record the completed turn (side effects)
    final_footer(turn)           -> str   signature appended to the answer
    start_background(send)        -> None  spawn a boot-time loop (alerts/digests)

Every method MUST degrade quietly — a slow or broken hook must never break a turn,
so we wrap all I/O and return empty on failure.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    """The repo the bot operates on. Prefer the explicit env, else the git
    top-level, else the current directory — never hardcode a path."""
    env = os.environ.get("AGENT_GATEWAY_REPO_ROOT", "").strip()
    if env:
        return Path(env)
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=4, check=False,
        ).stdout.strip()
        if out:
            return Path(out)
    except Exception:  # noqa: BLE001 - fall through to cwd
        pass
    return Path.cwd()


# Files we inject as "current state" if they exist at the repo root. Rename these
# to whatever your project keeps its live notes in.
STATE_FILES = ("STATUS.md", "NOTES.md", "NOW.md")


class TemplateHook:
    def before_turn(self, turn) -> str:
        """Inject current project state so every agent starts with shared context.
        READ-ONLY — safe to run for many parallel agents at once."""
        repo = _repo_root()
        for name in STATE_FILES:
            path = repo / name
            if path.is_file():
                try:
                    return f"{name} (current state):\n" + path.read_text(errors="replace")[:1500]
                except OSError:
                    return ""
        return ""  # nothing to inject — the bot runs fine without it

    def after_turn(self, turn, final_text: str) -> None:
        """Record the completed turn. APPEND-ONLY, so parallel agents don't collide.
        Here we just log a line; a real brain might append to an event store/DB."""
        summary = " ".join((getattr(turn, "text", "") or "").split())[:120]
        if not summary:
            return
        state_dir = Path(os.environ.get("AGENT_GATEWAY_STATE_DIR", _repo_root() / "state"))
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            with (state_dir / "turns.log").open("a", encoding="utf-8") as fh:
                fh.write(f"{getattr(turn, 'backend', 'agent')}\t{summary}\n")
        except OSError:
            pass  # logging is best-effort; never fail a turn over it

    def final_footer(self, turn) -> str:
        """Optional one-line signature under each answer. Return "" for none.
        The engine adds nothing by default; this opts in."""
        agent = str(getattr(turn, "backend", "") or "agent")
        model = str(getattr(turn, "model", "") or "").strip()
        return f"— {agent}" + (f" · {model}" if model else "")

    def start_background(self, send) -> None:
        """Optional boot-time loop for proactive messages (alerts, a daily digest).
        `send(text)` delivers to the operator. Must NOT block — spawn a daemon
        thread and return. Left as a no-op here; uncomment to use.

            import threading, time
            def _digest():
                while True:
                    time.sleep(24 * 3600)
                    send("daily digest: ...")   # your roundup here
            threading.Thread(target=_digest, daemon=True).start()
        """
        return None


def make_hook() -> TemplateHook:
    return TemplateHook()
