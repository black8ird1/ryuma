"""Pluggable augmentation hooks — the seam between the generic bot and a private
"brain". This is the symmetric twin of the backend adapter: backends plug in
MODELS, hooks plug in CONTEXT/MEMORY. The bot core knows nothing about any
business; it just calls two methods at turn boundaries.

The core ships with a NoOpHook (a generic bot has no brain). An operator points
AGENT_GATEWAY_HOOK at their own module/file to:
  * before_turn  -> inject business context (project state, relevant memories)
  * after_turn   -> record what happened (e.g. append to their event log)

Their knowledge system stays entirely OUTSIDE this (publishable) engine. A broken
or slow private hook must never break a turn — callers wrap every hook call.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from .core import AgentTurn


@runtime_checkable
class AugmentationHook(Protocol):
    def before_turn(self, turn: AgentTurn) -> str:
        """Extra context to inject into the prompt. Empty string = inject nothing."""
        ...

    def after_turn(self, turn: AgentTurn, final_text: str) -> None:
        """Record the completed turn. Side effects only."""
        ...

    def final_footer(self, turn: AgentTurn) -> str:
        """Optional signature appended to the final message (e.g. model · effort ·
        agent). The neutral bot adds nothing; a personal brain opts in. Empty = none."""
        ...

    def start_background(self, send) -> None:
        """Optional: start a personal background loop at boot — scheduled alerts,
        digests, watchers. `send(text)` delivers a message to the operator(s). This
        is the SCHEDULE seam: the engine runs nothing; the brain fills it (e.g. a
        product-event watcher). Must not block — spawn a daemon thread and return."""
        ...


class NoOpHook:
    """Default: a generic bot has no brain. Injects nothing, records nothing, and
    imposes no footer — the operator's final message is theirs to structure."""

    def before_turn(self, turn: AgentTurn) -> str:
        return ""

    def after_turn(self, turn: AgentTurn, final_text: str) -> None:
        return None

    def final_footer(self, turn: AgentTurn) -> str:
        return ""

    def start_background(self, send) -> None:
        return None


def _load_factory(spec: str):
    """Resolve 'module_or_path:factory'. Supports a dotted module on sys.path or a
    .py file path, so a private hook can live anywhere outside this repo."""
    target, _, attr = spec.partition(":")
    attr = attr or "make_hook"
    target = target.strip()
    if target.endswith(".py") or "/" in target:
        path = Path(target)
        mod_spec = importlib.util.spec_from_file_location(f"_ryuma_hook_{path.stem}", str(path))
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"cannot load hook file: {target}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, attr)


def load_hook() -> AugmentationHook:
    """Load AGENT_GATEWAY_HOOK ('module_or_path:factory'), or NoOpHook if unset or
    broken. A bad private hook degrades to no-op — it never breaks the bot."""
    spec = os.environ.get("AGENT_GATEWAY_HOOK", "").strip()
    if not spec:
        return NoOpHook()
    try:
        factory = _load_factory(spec)
        hook = factory()
        if hasattr(hook, "before_turn") and hasattr(hook, "after_turn"):
            return hook
    except Exception:  # noqa: BLE001 - a broken brain must not stop the engine.
        pass
    return NoOpHook()
