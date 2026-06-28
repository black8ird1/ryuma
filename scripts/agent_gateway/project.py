"""Project decoupling — the bot core ships project-neutral.

A project supplies its own system prompt and brand via config (an env var, or a
file), so the same bot serves any repository. Any profile under examples/ is just
one such project. The FUNCTIONAL contract (SUGGEST / POSTTURN markers) is how the
bot itself works and is always present, independent of the project.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_BRAND_NAME = "Ryuma"
DEFAULT_BRAND_EMOJI = "🐉"

GENERIC_SYSTEM_PROMPT = (
    "You are a coding agent operated through a Telegram bot, working in the given "
    "repository. Keep context efficient: read specific files on demand rather than "
    "bulk-loading docs. Report what you changed and what you verified."
)

# Always appended — these markers are part of the bot's contract, not any project.
FUNCTIONAL_CONTRACT = (
    "When a task has obvious next moves, you may end with one line: "
    "SUGGEST: action | action | action — it is rendered as tap-to-send buttons. "
    "For write tasks, you may request scoped end-of-turn git automation with a final "
    "one-line JSON marker: "
    'POSTTURN: {"commit":{"message":"...","paths":["file"]},"push":false,"merge":false} '
    "— include only files you intentionally touched; push/merge obey gateway policy."
)


def brand_name() -> str:
    return os.environ.get("AGENT_GATEWAY_BRAND_NAME", DEFAULT_BRAND_NAME).strip() or DEFAULT_BRAND_NAME


def brand_emoji() -> str:
    return os.environ.get("AGENT_GATEWAY_BRAND_EMOJI", DEFAULT_BRAND_EMOJI).strip() or DEFAULT_BRAND_EMOJI


def project_system_prompt() -> str:
    """The project's own instructions. Inline env wins, then a file, then generic."""
    inline = os.environ.get("AGENT_GATEWAY_SYSTEM_PROMPT", "").strip()
    if inline:
        return inline
    path = os.environ.get("AGENT_GATEWAY_SYSTEM_PROMPT_FILE", "").strip()
    if path:
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            pass
    return GENERIC_SYSTEM_PROMPT


def full_system_prompt() -> str:
    return f"{project_system_prompt()}\n\n{FUNCTIONAL_CONTRACT}"
