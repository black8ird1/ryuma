"""Shared Telegram agent gateway foundation.

This package is intentionally separate from the existing Codex and Claude bots.
It provides a transport/runtime/backend split so multiple agent runners can use
one Telegram cockpit without forcing one runner's session model onto another.
"""

from .core import (
    AgentEvent,
    AgentTurn,
    BackendCapabilities,
    GatewayRuntime,
    InjectionBuffer,
    LiveCard,
    ReplyContext,
    SubmitResult,
    build_interface_report,
    extract_suggestions,
    is_status_frame,
)
from .post_turn import CommitRequest, PostTurnPolicy, PostTurnRequest, PostTurnRunner, extract_post_turn_request
from .worktrees import WorktreeAssignment, WorktreeBroker, WorktreePolicy

__all__ = [
    "AgentEvent",
    "AgentTurn",
    "BackendCapabilities",
    "CommitRequest",
    "GatewayRuntime",
    "InjectionBuffer",
    "LiveCard",
    "PostTurnPolicy",
    "PostTurnRequest",
    "PostTurnRunner",
    "ReplyContext",
    "SubmitResult",
    "WorktreeAssignment",
    "WorktreeBroker",
    "WorktreePolicy",
    "build_interface_report",
    "extract_post_turn_request",
    "extract_suggestions",
    "is_status_frame",
]
