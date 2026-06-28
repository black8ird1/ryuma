"""Provider-neutral core for a unified Telegram agent cockpit.

The important split:
  - Telegram transport owns auth, routing, buttons, reply context and live UI.
  - GatewayRuntime owns queueing, /stop, /new, chat prefs and injection.
  - Backends own model-specific process/API details.

That lets Codex, Claude Code, OpenCode, GLM or a mock backend share the same bot
interface while keeping their different execution mechanics behind adapters.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol


ROOT = Path(__file__).resolve().parents[2]

Mode = Literal["plan", "write"]  # write = raw, model judges; plan = read-only lock
EventKind = Literal[
    "started",
    "thinking",
    "tool",
    "status",
    "injected",
    "final",
    "error",
    "usage",
]
SubmitResult = Literal["started", "queued", "steered", "rejected"]


STATUS_FRAME_MARKERS = (
    "Codex live",
    "Agent live",
    "/stop cancels",
    "still working",
    "text queues next turn",
    "Usage:",
    "queue:",
)

_SUGGEST_RE = re.compile(r"^\s*SUGGEST:\s*(?P<items>[^\n]+?)\s*$", re.MULTILINE)
MAX_SUGGESTIONS = 3


def is_status_frame(text: str | None) -> bool:
    """Return true for transient bot UI that should not be quoted as context."""
    if not text:
        return False
    return any(marker in text for marker in STATUS_FRAME_MARKERS)


def extract_suggestions(text: str) -> tuple[str, list[str]]:
    """Strip a final SUGGEST marker and return up to three next-prompt buttons."""
    suggestions: list[str] = []

    def _take(match: re.Match[str]) -> str:
        if not suggestions:
            raw_items = match.group("items")
            suggestions.extend(item.strip() for item in raw_items.split("|") if item.strip())
        return ""

    cleaned = _SUGGEST_RE.sub(_take, text).strip()
    return cleaned, suggestions[:MAX_SUGGESTIONS]


@dataclass(frozen=True)
class ReplyContext:
    """A previous Telegram message intentionally carried into the next prompt."""

    author: str
    text: str

    def render(self) -> str:
        return f"[Context - replying to {self.author}]:\n{self.text.strip()}"


@dataclass(frozen=True)
class Attachment:
    path: Path
    kind: str = "file"
    caption: str = ""


@dataclass(frozen=True)
class AgentTurn:
    chat_id: int
    user_id: int
    text: str
    backend: str
    mode: Mode = "write"
    reply_context: ReplyContext | None = None
    attachments: tuple[Attachment, ...] = ()
    workdir: Path | None = None
    augment: str = ""  # optional project/brain context injected by an AugmentationHook
    model: str = ""    # per-chat model override (free-text, passed through to the backend)
    effort: str = ""   # per-chat effort override (free-text, interpreted by the backend)
    created_at: float = field(default_factory=time.time)

    def prompt_text(self) -> str:
        parts: list[str] = []
        if self.reply_context is not None:
            parts.append(self.reply_context.render())
        parts.append(self.text.strip())
        if self.attachments:
            lines = [f"- {a.kind}: {a.path}" + (f" ({a.caption})" if a.caption else "") for a in self.attachments]
            parts.append("Attachments:\n" + "\n".join(lines))
        return "\n\n".join(p for p in parts if p).strip()


@dataclass(frozen=True)
class AgentEvent:
    kind: EventKind
    text: str = ""
    detail: str = ""
    backend: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    label: str
    streams: bool = False
    live_thinking: bool = False
    live_tools: bool = False
    steering: bool = False
    persistent_sessions: bool = False
    skills: bool = False
    images: bool = False
    voice: bool = False
    write_access: bool = False
    compact_final: bool = True
    model_suggestions: tuple[str, ...] = ()  # common models, shown as /model tap-buttons (free-text still works)

    def summary(self) -> str:
        flags = []
        if self.streams:
            flags.append("streams")
        if self.live_thinking:
            flags.append("thinking")
        if self.live_tools:
            flags.append("tools")
        if self.steering:
            flags.append("inject")
        if self.persistent_sessions:
            flags.append("sessions")
        if self.skills:
            flags.append("skills")
        if self.write_access:
            flags.append("write")
        return f"{self.label}: " + (", ".join(flags) if flags else "basic")


class InjectionBuffer:
    """Thread-safe mid-run message buffer.

    Claude-style backends can poll or wait on this and fold new Telegram messages
    into the active turn. Non-steering backends ignore it and the runtime queues.
    """

    def __init__(self) -> None:
        self._items: queue.Queue[str] = queue.Queue()
        self._event = threading.Event()

    def push(self, text: str) -> None:
        self._items.put(text)
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def drain(self) -> list[str]:
        items: list[str] = []
        while True:
            try:
                items.append(self._items.get_nowait())
            except queue.Empty:
                break
        self._event.clear()
        return items


Emit = Callable[[AgentEvent], None]


class AgentBackend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    def run(
        self,
        turn: AgentTurn,
        emit: Emit,
        stop_event: threading.Event,
        injections: InjectionBuffer,
    ) -> str: ...

    def reset(self, chat_id: int) -> None: ...


@dataclass
class ChatPrefs:
    backend: str
    mode: Mode = "write"
    model: str = ""
    effort: str = ""


@dataclass
class _ActiveRun:
    thread: threading.Thread
    stop: threading.Event
    injections: InjectionBuffer
    backend: str


class GatewayRuntime:
    """Small synchronous runtime with one active run per chat."""

    def __init__(
        self,
        backends: dict[str, AgentBackend],
        *,
        default_backend: str = "mock",
        max_queue_depth: int = 5,
    ) -> None:
        if default_backend not in backends:
            raise ValueError(f"default backend {default_backend!r} is not registered")
        self.backends = backends
        self.default_backend = default_backend
        self.max_queue_depth = max_queue_depth
        self._prefs: dict[int, ChatPrefs] = {}
        self._queues: dict[int, list[AgentTurn]] = {}
        self._active: dict[int, _ActiveRun] = {}
        self._lock = threading.RLock()

    def backend_names(self) -> list[str]:
        return sorted(self.backends)

    def backend_for_chat(self, chat_id: int) -> str:
        return self._prefs.get(chat_id, ChatPrefs(self.default_backend)).backend

    def mode_for_chat(self, chat_id: int) -> Mode:
        return self._prefs.get(chat_id, ChatPrefs(self.default_backend)).mode

    def model_for_chat(self, chat_id: int) -> str:
        return self._prefs.get(chat_id, ChatPrefs(self.default_backend)).model

    def effort_for_chat(self, chat_id: int) -> str:
        return self._prefs.get(chat_id, ChatPrefs(self.default_backend)).effort

    def select_backend(self, chat_id: int, backend: str) -> None:
        if backend not in self.backends:
            raise ValueError(f"unknown backend: {backend}")
        prefs = self._prefs.setdefault(chat_id, ChatPrefs(self.default_backend))
        prefs.backend = backend

    def set_mode(self, chat_id: int, mode: Mode) -> None:
        if mode not in ("plan", "write"):
            raise ValueError(f"unknown mode: {mode}")
        prefs = self._prefs.setdefault(chat_id, ChatPrefs(self.default_backend))
        prefs.mode = mode

    def try_steer(self, turn: AgentTurn) -> bool:
        """Atomically fold a message into a running steering-capable turn. Returns
        True if folded (caller shows NO new card — it lands in the original turn's
        card); False if there's nothing to steer (caller starts a fresh turn)."""
        with self._lock:
            active = self._active.get(turn.chat_id)
            if active is None:
                return False
            backend = self.backends.get(active.backend)
            if backend is None or not backend.capabilities.steering:
                return False
            active.injections.push(turn.prompt_text())
            return True

    def set_model(self, chat_id: int, model: str) -> None:
        prefs = self._prefs.setdefault(chat_id, ChatPrefs(self.default_backend))
        prefs.model = model.strip()

    def set_effort(self, chat_id: int, effort: str) -> None:
        prefs = self._prefs.setdefault(chat_id, ChatPrefs(self.default_backend))
        prefs.effort = effort.strip()

    def status(self, chat_id: int) -> dict[str, Any]:
        with self._lock:
            active = self._active.get(chat_id)
            return {
                "backend": self.backend_for_chat(chat_id),
                "mode": self.mode_for_chat(chat_id),
                "model": self.model_for_chat(chat_id) or "(backend default)",
                "effort": self.effort_for_chat(chat_id) or "(backend default)",
                "active": active.backend if active else None,
                "queued": len(self._queues.get(chat_id, [])),
                "backends": self.backend_names(),
            }

    def reset_chat(self, chat_id: int) -> int:
        with self._lock:
            queued = len(self._queues.pop(chat_id, []))
            active = self._active.get(chat_id)
            if active:
                active.stop.set()
            for backend in self.backends.values():
                backend.reset(chat_id)
            return queued

    def stop(self, chat_id: int) -> bool:
        with self._lock:
            self._queues.pop(chat_id, None)
            active = self._active.get(chat_id)
            if not active:
                return False
            active.stop.set()
            return True

    def submit(self, turn: AgentTurn, emit: Emit) -> SubmitResult:
        backend = self.backends.get(turn.backend)
        if backend is None:
            emit(AgentEvent("error", f"Unknown backend: {turn.backend}", backend=turn.backend))
            return "rejected"

        with self._lock:
            active = self._active.get(turn.chat_id)
            if active is not None:
                active_backend = self.backends[active.backend]
                if active_backend.capabilities.steering:
                    active.injections.push(turn.prompt_text())
                    emit(AgentEvent("injected", "Injected into the active turn.", backend=active.backend))
                    return "steered"
                q = self._queues.setdefault(turn.chat_id, [])
                if len(q) >= self.max_queue_depth:
                    emit(AgentEvent("error", f"Queue full ({len(q)}/{self.max_queue_depth}).", backend=turn.backend))
                    return "rejected"
                q.append(turn)
                emit(AgentEvent("status", f"Queued for next turn ({len(q)}/{self.max_queue_depth}).", backend=turn.backend))
                return "queued"

            stop = threading.Event()
            injections = InjectionBuffer()
            thread = threading.Thread(
                target=self._run_loop,
                args=(turn, emit, stop, injections),
                daemon=True,
                name=f"agent-gateway-{turn.chat_id}",
            )
            self._active[turn.chat_id] = _ActiveRun(thread, stop, injections, turn.backend)
            thread.start()
            return "started"

    def wait_for_idle(self, chat_id: int, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                active = self._active.get(chat_id)
            if active is None:
                return True
            active.thread.join(timeout=0.05)
        return False

    def _run_loop(
        self,
        first: AgentTurn,
        emit: Emit,
        stop: threading.Event,
        injections: InjectionBuffer,
    ) -> None:
        current: AgentTurn | None = first
        while current is not None:
            backend = self.backends[current.backend]
            emit(AgentEvent("started", f"{backend.capabilities.label} started.", backend=current.backend))
            final = ""
            try:
                final = backend.run(current, emit, stop, injections)
                if final:
                    emit(AgentEvent("final", final, backend=current.backend))
            except Exception as exc:  # noqa: BLE001 - backend failures must not kill the bot.
                emit(AgentEvent("error", f"{type(exc).__name__}: {exc}", backend=current.backend))
            with self._lock:
                if stop.is_set():
                    self._queues.pop(first.chat_id, None)
                    current = None
                else:
                    q = self._queues.get(first.chat_id) or []
                    current = q.pop(0) if q else None
                    if current is not None:
                        self._queues[first.chat_id] = q
                if current is None:
                    self._active.pop(first.chat_id, None)
                else:
                    # Keep the same active stop/injection objects for the chat loop.
                    self._active[first.chat_id] = _ActiveRun(
                        threading.current_thread(), stop, injections, current.backend
                    )


_PULSE_FRAMES = ("◐", "◓", "◑", "◒")


def phase_icon(phase: str) -> str:
    """Map a free-text phase to one icon. Backend-neutral: any model's phase
    strings flow through here, so the card looks identical everywhere."""
    low = phase.lower()
    if "test" in low:
        return "🧪"
    if "check" in low or "lint" in low or "type" in low:
        return "🔎"
    if "git" in low or "commit" in low or "merge" in low:
        return "🌿"
    if "edit" in low or "file" in low or "patch" in low or "writ" in low:
        return "✍️"
    if "search" in low or "grep" in low or "glob" in low:
        return "🔭"
    if "shell" in low or "command" in low or "read" in low or "bash" in low:
        return "⌨️"
    if "final" in low or "answer" in low or "done" in low:
        return "✅"
    if "think" in low or "reason" in low:
        return "🧠"
    if "session" in low or "start" in low:
        return "🟢"
    if "steer" in low or "inject" in low:
        return "📨"
    if "error" in low or "fail" in low:
        return "⚠"
    return "⚙"


def result_icon(result: str) -> str:
    low = result.lower()
    if "fail" in low or "error" in low:
        return "❌"
    if "pass" in low or "success" in low or "ok" in low or "clean" in low:
        return "✅"
    return "📌"


class LiveCard:
    """Rich, model-agnostic progress card shared by every backend.

    Fed ONLY by AgentEvents. Backends never format Telegram text — they emit
    events; this card decides how it looks. That is what makes Codex, Claude and
    any future backend render an identical live experience.
    """

    _CATEGORY_ATTR = {"cmd": "cmd_count", "file": "file_count", "tool": "tool_count"}
    _ACTION_ICONS = {"cmd": "⚡", "file": "✏️", "tool": "🔧"}

    def __init__(
        self,
        *,
        backend: str,
        mode: Mode,
        label: str | None = None,
        brand: str = "🤖",
        queue_depth: int = 0,
        steering: bool = False,
        model: str = "",
        effort: str = "",
    ) -> None:
        self.backend = backend
        self.label = label or backend
        self.mode = mode
        self.brand = brand
        self.model = model
        self.effort = effort
        self.queue_depth = queue_depth
        self.steering = steering
        self.started_at = time.time()
        self.last_event_at = self.started_at
        self._pulse = 0
        self.phase = "starting"
        self.current = "warming up"
        self.shell = ""
        self.diff = ""
        self.cmd_count = 0
        self.tool_count = 0
        self.file_count = 0
        self.result = ""
        self.last = ""
        self.feed = ""  # combined stream (chat card tail)
        self.timeline: list[dict] = []  # ordered stream for the Mini App: reason/answer prose + action waypoints, interleaved
        self.reasoning = ""  # live reasoning (thinking_delta) — Mini App shows it dim
        self.answer = ""  # the answer (text_delta) — Mini App shows it bright
        self.stream_chars = 0  # size of streamed output so far (an always-live gauge)
        self.usage = ""
        self.final_text = ""
        self.error_text = ""

    def _timeline_push(self, kind: str, text: str) -> None:
        """Append to the Mini App's flowing stream. Consecutive prose of the same
        kind (reason/answer) merges into one paragraph so the stream reads
        continuously; an action waypoint between them breaks the rhythm — that
        interleaving is what gives the live view its sense of progress."""
        if not text:
            return
        if kind in ("reason", "answer") and self.timeline and self.timeline[-1]["t"] == kind:
            self.timeline[-1]["text"] += text
        else:
            self.timeline.append({"t": kind, "text": text})
        if len(self.timeline) > 160:  # bound the polled payload
            self.timeline = self.timeline[-160:]

    def _feed_mark(self, icon: str, text: str) -> None:
        """Put an action on its own line in the feed, deduping consecutive repeats
        so a chatty backend can't spam the same line."""
        line = f"{icon} {text}"
        last = self.feed.rstrip("\n").rsplit("\n", 1)[-1] if self.feed else ""
        if last == line:
            return
        sep = "" if (not self.feed or self.feed.endswith("\n")) else "\n"
        self.feed += f"{sep}{line}\n"

    def _feed_tail(self, limit: int = 2600, max_lines: int = 34) -> str:
        """The live stream window: grows downward as it writes (you read it flowing,
        like the specialized bot) and only tails once it gets long — never unbounded."""
        feed = self.feed
        if len(feed) > limit:
            feed = feed[-limit:]
            nl = feed.find("\n")
            if nl != -1:
                feed = feed[nl + 1:]  # drop the partial leading line
        return "\n".join(feed.splitlines()[-max_lines:]).strip()

    def update(self, event: AgentEvent) -> None:
        self.last_event_at = event.ts
        data = event.data or {}
        text = (event.text or event.detail or "").strip()
        if event.kind == "final":
            self.final_text = event.text
            return
        if event.kind == "usage":
            if text:
                self.usage = text
            return
        if event.kind == "error":
            self.error_text = text
            self.phase = "error"
            if text:
                self.current = text[:140]
            return
        phase = str(data.get("phase") or "").strip()
        if not phase:
            phase = {
                "started": "starting",
                "thinking": "thinking",
                "tool": "working",
                "status": "working",
                "injected": "steering",
            }.get(event.kind, "working")
        self.phase = phase
        category = str(data.get("category") or "").strip()
        attr = self._CATEGORY_ATTR.get(category)
        if attr:
            setattr(self, attr, getattr(self, attr) + 1)
        elif event.kind == "tool":
            self.tool_count += 1
        # THE SEPARATION: the feed is valuable thinking only (streamed prose). All
        # PROCESS — tool actions, status, injections — feeds the data panel via
        # `current`, never the thought stream.
        if event.kind == "thinking" and data.get("stream"):
            self.stream_chars += len(event.text)
            if data.get("phase") == "writing":
                if not self.answer and self.reasoning.strip():
                    # Mark the think→answer transition. The Mini App carries this
                    # with dim→bright; the chat feed has no styling, so a glyph on
                    # its own block signals "done reasoning, here's the answer".
                    self.feed = self.feed.rstrip() + "\n\n💬 "
                self.answer += event.text
                self._timeline_push("answer", event.text)
            else:
                self.reasoning += event.text
                self._timeline_push("reason", event.text)
            self.feed += event.text
        elif event.kind == "tool" and text:
            self.current = text[:120]
            self._timeline_push("act", text[:120])  # waypoint in the flowing stream
            # Prose-streaming backends (Claude) fill the feed with live thinking, so
            # actions stay in the data panel (`current`) — the separation above. But a
            # backend that streams NO reasoning prose (Codex app-server only emits its
            # agent message at the very end) would otherwise show a frozen "…thinking"
            # body for minutes while it actually works. Until prose starts flowing,
            # mirror each action into the feed so the live card visibly progresses.
            if self.stream_chars == 0:
                self._feed_mark(self._ACTION_ICONS.get(category, "➡️"), text[:120])
        elif event.kind == "injected" and text:
            self.current = f"📨 {text}"
            self._timeline_push("act", f"📨 {text}"[:120])
        # status events update only the phase — never the now-line (no path/"session
        # active" noise) and never the feed.

    def _latest_thought(self, limit: int = 180) -> str:
        """The single most recent line of streamed thinking — a glanceable pulse, NOT
        the full stream. The chat stays a calm record; the whole thought stream lives
        in the Live view app. Collapses newlines so it never grows the card."""
        src = " ".join((self.answer or self.reasoning or "").split())
        if not src:
            return ""
        if len(src) <= limit:
            return src
        tail = src[-limit:]
        sp = tail.find(" ")
        return "…" + (tail[sp + 1:] if 0 <= sp < 48 else tail)

    def render_live(self) -> str:
        elapsed = int(time.time() - self.started_at)
        clock = f"{elapsed // 60}:{elapsed % 60:02d}"
        idle = int(time.time() - self.last_event_at)
        self._pulse += 1
        pulse = _PULSE_FRAMES[self._pulse % len(_PULSE_FRAMES)]
        # IN-CHAT THOUGHT STREAM (the Mini App is retired): the chat IS the live view,
        # like the specialized bot — one slim header + status line, then the continuous
        # thought stream gets the room. Simpler, all in one place, nothing to tap.
        lines = [f"{pulse} {self.brand} {self.label} · {phase_icon(self.phase)} {self.phase} · ⏱ {clock}"]
        status: list[str] = []
        if self.current and self.current != "warming up":
            status.append(f"➡️ {self.current}")  # the single live-action line (the "data panel")
        if self.diff:
            status.append(f"📊 {self.diff}")
        if self.queue_depth:
            status.append(f"📦 q{self.queue_depth}")
        if idle >= 20:
            status.append(f"⏳ {idle}s")
        if status:
            lines.append("  ·  ".join(status))
        lines.append("")
        lines.append(self._feed_tail(limit=3000, max_lines=44) or "…thinking")
        lines.append("")
        foot = "🛑 /stop"
        if self.steering:
            foot = "✎ reply to steer · " + foot
        lines.append(foot)
        return "\n".join(lines)

    def render_final(self) -> str:
        # The neutral gateway never imposes a footer on the answer — whoever runs an
        # agent owns its final message format. Signing (model/effort/agent) is an
        # opt-in concern of the personal brain (see AugmentationHook.final_footer).
        if self.error_text and not self.final_text:
            return f"⚠ {self.error_text}"
        # Tokens in/out are deliberately not shown — no universal limits datum across
        # backends, so the card stays lean (founder's call). Signing (model/effort/
        # agent) is the personal brain's job via AugmentationHook.final_footer.
        return self.final_text.strip() or "Done."

    def snapshot(self) -> dict:
        """JSON-able live state for the Mini App. Published on every event, so the
        web view can poll faster than the throttled Telegram card edits."""
        elapsed = int(time.time() - self.started_at)
        return {
            "label": self.label,
            "brand": self.brand,
            "model": self.model,
            "effort": self.effort,
            "mode": self.mode,
            "clock": f"{elapsed // 60}:{elapsed % 60:02d}",
            "phase": self.phase,
            "phase_icon": phase_icon(self.phase),
            "now": "" if self.current == "warming up" else self.current,
            "diff": self.diff,
            "cmd": self.cmd_count,
            "tool": self.tool_count,
            "file": self.file_count,
            "chars": self.stream_chars,
            "started_at": self.started_at,  # stable per-turn id so the web view can reset on a new turn
            "timeline": self.timeline,      # the flowing stream: reason/answer prose + action waypoints, in order
            "reasoning": self.reasoning[-4000:],
            "answer": self.answer[-9000:],
            "usage": self.usage,
            "feed": self.feed[-8000:],
            "steering": self.steering,
            "queue": self.queue_depth,
            "idle": int(time.time() - self.last_event_at),
            "final": self.final_text,
            "error": self.error_text,
            "done": bool(self.final_text or self.error_text),
            "ts": time.time(),
        }


def build_interface_report() -> str:
    """Human-readable design summary: best features extracted from both bots."""
    return "\n".join(
        [
            "Ideal shared interface:",
            "- Reply injection: quote real prior messages, ignore transient status frames.",
            "- Live progress: stream thinking/tool/status events into one edited card.",
            "- Steering: backends that support it fold mid-run messages into the active turn.",
            "- Queue fallback: non-steering backends queue safely instead of losing messages.",
            "- Compact finals: final answer stays concise; usage/model/effort live in a footer.",
            "- Continuation buttons: final `SUGGEST: a | b | c` lines become three tap-to-send prompts.",
            "- Post-turn hook: write backends can request scoped commit/push/merge through a stripped `POSTTURN: {...}` line.",
            "- Worktree broker: write turns can run in isolated branch worktrees automatically.",
            "- Merge checkpoint: queued merge requests can be listed, inspected, approved or rejected before landing.",
            "- Skills menu: discover skills dynamically; activation becomes a backend-neutral turn.",
            "- Attachments: normalize photo albums, voice transcripts and documents as turn inputs.",
            "- Group routing: explicit @mention wins; reply routing only when no other bot is tagged.",
            "- Session controls: /new resets backend session, /stop cancels, /agent switches backend.",
            "- Self-heal hooks: adapters classify auth/rate/session drift without polluting Telegram UI.",
            "",
            "Backend-specific work stays behind adapters: Codex compact CLI runs, Claude warm-worker streaming, OpenCode provider runs, GLM API chat, Hermes if it exposes a stable headless API.",
        ]
    )
