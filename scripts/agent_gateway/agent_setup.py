"""Agent setup metadata + availability detection.

Single source of truth for "how do I add Claude / Codex?" shared by the
onboarding wizard (onboard.py) and the in-bot /agents command (telegram.py), so
both describe the same install + login steps. The gateway always *runs* every
backend; the only thing that varies per machine is which agent CLI is actually
installed + signed in. This module answers that, and tells the operator how to
fix a missing one.

Pure: the only I/O is shutil.which (injectable), so it unit-tests with no
network and no TTY.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Callable

WhichFn = Callable[[str], str | None]


@dataclass(frozen=True)
class AgentSpec:
    name: str  # backend key the gateway uses (claude / codex)
    label: str  # human-facing name
    bin: str  # CLI binary we look for on PATH
    install: str  # one-line install command
    login: str  # one-line auth command
    docs: str  # docs URL


# Order = the order we offer agents in the wizard + the /agents board.
AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec(
        name="claude",
        label="Claude Code",
        bin="claude",
        install="npm install -g @anthropic-ai/claude-code",
        login="claude  (sign in once)  ·  headless: claude setup-token",
        docs="https://docs.claude.com/en/docs/claude-code",
    ),
    AgentSpec(
        name="codex",
        label="Codex",
        bin="codex",
        install="npm install -g @openai/codex",
        login="codex login",
        docs="https://developers.openai.com/codex/cli",
    ),
)

# Names of the real (non-mock) agents, in offer order.
KNOWN_AGENTS: tuple[str, ...] = tuple(spec.name for spec in AGENT_SPECS)


def spec_for(name: str) -> AgentSpec | None:
    """The spec for a backend name, or None for unknown / mock."""
    key = name.strip().lower()
    for spec in AGENT_SPECS:
        if spec.name == key:
            return spec
    return None


def agent_ready(spec: AgentSpec, which: WhichFn = shutil.which) -> bool:
    """True when the agent's CLI is on PATH (installed). Sign-in state can't be
    probed cheaply, so 'ready' means installed — the login hint covers the rest."""
    return bool(which(spec.bin))


def agent_status(which: WhichFn = shutil.which) -> list[tuple[AgentSpec, bool]]:
    """Every known agent paired with whether its CLI is installed."""
    return [(spec, agent_ready(spec, which)) for spec in AGENT_SPECS]


def ready_agents(which: WhichFn = shutil.which) -> list[str]:
    """Backend names whose CLI is installed (mock is always available separately)."""
    return [spec.name for spec, ok in agent_status(which) if ok]


def setup_lines(spec: AgentSpec) -> list[str]:
    """Copy-paste setup steps for one agent (install → login → docs)."""
    return [
        f"  1. install:  {spec.install}",
        f"  2. sign in:  {spec.login}",
        f"  docs: {spec.docs}",
    ]
