"""Backend adapters for the shared agent gateway."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .core import AgentEvent, AgentTurn, BackendCapabilities, Emit, InjectionBuffer, ROOT
from .project import full_system_prompt


def _gateway_prompt(turn: AgentTurn, *, include_system: bool = True) -> str:
    # No mode, no directive — pure raw capability. `include_system=False` skips the
    # static system prompt for backends that already carry it out-of-band (Claude's
    # --append-system-prompt), so it isn't re-sent and re-billed every turn.
    workdir_line = f"Working directory: {turn.workdir}\n" if turn.workdir else ""
    augment_block = f"\nProject context:\n{turn.augment}\n" if turn.augment else ""
    head = f"{full_system_prompt()}\n" if include_system else ""
    # Plan mode = read-only (no worktree is created for it): propose, don't execute.
    plan_block = ""
    if turn.mode == "plan":
        plan_block = (
            "\nPLAN MODE — read-only. Investigate and produce a clear, actionable plan: "
            "concise numbered steps naming the files/commands you WOULD touch and why. "
            "Do NOT edit/create files or run mutating commands — propose, don't execute. "
            "The operator reviews your plan and taps to build it.\n"
        )
    return (
        f"{head}"
        f"{workdir_line}"
        f"{plan_block}"
        f"{augment_block}"
        f"\nUser request:\n{turn.prompt_text()}"
    )


class MockBackend:
    """Deterministic backend for dry-runs and tests."""

    capabilities = BackendCapabilities(
        name="mock",
        label="Mock Agent",
        streams=True,
        live_thinking=True,
        live_tools=True,
        steering=True,
        persistent_sessions=True,
        skills=True,
        images=True,
        voice=True,
        write_access=False,
    )

    def __init__(self, *, delay: float = 0.05) -> None:
        self.delay = delay
        self.reset_calls: list[int] = []

    def run(
        self,
        turn: AgentTurn,
        emit: Emit,
        stop_event: threading.Event,
        injections: InjectionBuffer,
    ) -> str:
        emit(AgentEvent("thinking", "Reading shared context and normalizing the turn.", backend="mock"))
        time.sleep(self.delay)
        if stop_event.is_set():
            emit(AgentEvent("status", "Stopped before work began.", backend="mock"))
            return "Stopped."
        emit(AgentEvent("tool", "mock.inspect_repo", "No files changed.", backend="mock"))
        injections.wait(self.delay)
        injected = injections.drain()
        if injected:
            emit(AgentEvent("injected", f"Folded {len(injected)} injected message(s).", backend="mock"))
        suffix = f"\n\nInjected:\n" + "\n\n".join(injected) if injected else ""
        return f"Mock final for `{turn.mode}` on: {turn.text.strip()}{suffix}"

    def reset(self, chat_id: int) -> None:
        self.reset_calls.append(chat_id)


@dataclass
class CodexExecBackend:
    """Codex CLI backend with bounded persistent thread support.

    This keeps the gateway independent from the existing Codex bot while
    preserving the important behavior: write turns resume a bounded Codex thread
    until /new or the turn cap clears it.
    """

    codex_bin: str = "codex"
    model: str = os.environ.get("CODEX_MODEL", os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    sandbox: str = "workspace-write"
    timeout_sec: int = 900
    workdir: Path = ROOT
    runs_dir: Path = ROOT / "state" / "agent-gateway" / "runs"
    effort: str = os.environ.get("AGENT_GATEWAY_CODEX_EFFORT", os.environ.get("CODEX_REASONING_EFFORT", "medium"))
    max_session_turns: int = int(os.environ.get("AGENT_GATEWAY_CODEX_MAX_SESSION_TURNS", "8"))
    persist_write_sessions: bool = os.environ.get("AGENT_GATEWAY_CODEX_PERSIST", "1").lower() not in {"0", "false", "no"}

    def __post_init__(self) -> None:
        self.sessions: dict[int, dict[str, object]] = {}

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="codex",
            label="Codex CLI",
            streams=True,
            live_tools=True,
            steering=False,
            persistent_sessions=True,
            skills=False,
            images=True,
            voice=True,
            write_access=self.sandbox != "read-only",
            compact_final=True,
            model_suggestions=("gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"),
        )

    def run(
        self,
        turn: AgentTurn,
        emit: Emit,
        stop_event: threading.Event,
        injections: InjectionBuffer,
    ) -> str:
        prompt = _gateway_prompt(turn)
        workdir = turn.workdir or self.workdir
        out_path = self._out_path(turn)
        session = self.sessions.get(turn.chat_id) if self.persist_write_sessions and turn.mode == "write" else None
        resume_id = str(session.get("thread_id")) if session and session.get("thread_id") else None
        turns = int(session.get("turns") or 0) if session else 0
        if turns >= self.max_session_turns:
            resume_id = None
            self.sessions.pop(turn.chat_id, None)
            emit(AgentEvent("status", "Codex session turn cap reached; starting fresh.", backend="codex"))

        thread_holder: dict[str, str] = {}
        cmd = self.build_cmd(
            prompt, out_path=out_path, resume_id=resume_id, persistent=turn.mode == "write", workdir=workdir,
            model=turn.model or self.model, effort=turn.effort or self.effort,
        )

        def parser(line: str):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {}
            if isinstance(obj, dict) and obj.get("type") == "thread.started":
                tid = str(obj.get("thread_id") or "").strip()
                if tid:
                    thread_holder["thread_id"] = tid
            return _parse_codex_event(line)

        emit(AgentEvent("status", "Spawning codex exec" + (" resume." if resume_id else "."), backend="codex"))
        fallback = _run_streaming_process(
            cmd,
            emit,
            stop_event,
            cwd=workdir,
            timeout_sec=self.timeout_sec,
            parser=parser,
        )
        reply = _read_text_if_exists(out_path) or fallback
        thread_id = thread_holder.get("thread_id") or resume_id
        if self.persist_write_sessions and turn.mode == "write" and thread_id and not stop_event.is_set():
            prior_turns = int((self.sessions.get(turn.chat_id) or {}).get("turns") or 0)
            self.sessions[turn.chat_id] = {"thread_id": thread_id, "turns": prior_turns + 1, "updated_ts": int(time.time())}
        return reply

    def build_cmd(self, prompt: str, *, out_path: Path, resume_id: str | None, persistent: bool, workdir: Path | None = None, model: str | None = None, effort: str | None = None) -> list[str]:
        run_dir = workdir or self.workdir
        model = (model or self.model)
        effort = _normalize_codex_effort(effort or self.effort)
        effort_config = f'model_reasoning_effort="{effort}"'
        if resume_id:
            return [
                self.codex_bin,
                "exec",
                "resume",
                "-c",
                effort_config,
                "--json",
                "--model",
                model,
                "--output-last-message",
                str(out_path),
                resume_id,
                prompt,
            ]
        cmd = [
            self.codex_bin,
            "exec",
            "-c",
            effort_config,
            "--json",
            "--model",
            model,
            "--sandbox",
            self.sandbox,
            "--cd",
            str(run_dir),
            "--output-last-message",
            str(out_path),
        ]
        if not persistent:
            cmd.append("--ephemeral")
        cmd.append(prompt)
        return cmd

    def _out_path(self, turn: AgentTurn) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        return self.runs_dir / f"codex-{turn.chat_id}-{stamp}.txt"

    def reset(self, chat_id: int) -> None:
        self.sessions.pop(chat_id, None)


class _AppServerWorker:
    """A warm `codex app-server` process speaking line-delimited JSON-RPC over stdio.

    One per chat (mirrors the Claude warm worker): a reader thread fans stdout into
    response futures (by id) and a notification queue (by method). This is what gives
    Codex true mid-turn steering (`turn/steer`), interrupt (`turn/interrupt`) and a
    real token-delta stream (`item/agentMessage/delta`) — none of which `codex exec`
    could do."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self.notifs: queue.Queue[dict] = queue.Queue()
        self._resp: dict[int, queue.Queue] = {}
        self._id = 0
        self._lock = threading.Lock()
        self.thread_id: str | None = None
        self.stderr: list[str] = []
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in obj and ("result" in obj or "error" in obj):
                box = self._resp.get(obj["id"])
                if box is not None:
                    box.put(obj)
            elif obj.get("method"):
                self.notifs.put(obj)
        self.notifs.put({"method": "_eof"})  # stream closed → unblock any waiter

    def _read_stderr(self) -> None:
        if self.proc.stderr is None:
            return
        for line in self.proc.stderr:
            self.stderr.append(line)
            if len(self.stderr) > 200:
                del self.stderr[:100]

    def request(self, method: str, params: dict, timeout: float = 60.0) -> dict:
        with self._lock:
            self._id += 1
            rid = self._id
            box: queue.Queue = queue.Queue(maxsize=1)
            self._resp[rid] = box
        try:
            self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise _AppServerDown(str(exc))
        try:
            return box.get(timeout=timeout)
        except queue.Empty:
            raise _AppServerDown(f"{method} timed out")
        finally:
            self._resp.pop(rid, None)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def terminate(self) -> None:
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                pipe and pipe.close()
            except OSError:
                pass
        try:
            self.proc.terminate()
        except OSError:
            pass


class _AppServerDown(Exception):
    """The app-server worker died or stopped responding — caller heals or surfaces."""


@dataclass
class CodexAppServerBackend:
    """Codex via the persistent `app-server` JSON-RPC protocol — the steering-capable
    twin of the Claude warm worker. Folds mid-turn messages into the live turn with
    `turn/steer`, streams token deltas, and interrupts on /stop. Falls back to the
    `exec` backend's behavior contract (same .run signature)."""

    codex_bin: str = "codex"
    model: str = os.environ.get("AGENT_GATEWAY_CODEX_MODEL", os.environ.get("CODEX_MODEL", "gpt-5.5"))
    sandbox: str = os.environ.get("AGENT_GATEWAY_CODEX_SANDBOX", "workspace-write")
    timeout_sec: int = 900
    workdir: Path = ROOT
    effort: str = os.environ.get("AGENT_GATEWAY_CODEX_EFFORT", os.environ.get("CODEX_REASONING_EFFORT", "medium"))
    max_session_turns: int = int(os.environ.get("AGENT_GATEWAY_CODEX_MAX_SESSION_TURNS", "8"))

    def __post_init__(self) -> None:
        self._workers: dict[int, _AppServerWorker] = {}
        self._turns: dict[int, int] = {}

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="codex",
            label="Codex CLI",
            streams=True,
            live_thinking=True,
            live_tools=True,
            steering=True,  # the headline: app-server turn/steer folds mid-turn messages
            persistent_sessions=True,
            skills=False,
            images=True,
            voice=True,
            write_access=self.sandbox != "read-only",
            compact_final=True,
            model_suggestions=("gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"),
        )

    def _sandbox_policy(self) -> dict:
        policy = {
            "read-only": {"type": "readOnly"},
            "workspace-write": {"type": "workspaceWrite"},
            "danger-full-access": {"type": "dangerFullAccess"},
        }.get(self.sandbox, {"type": "workspaceWrite"})
        # A write turn runs in an ISOLATED worktree, so workspace-write only grants
        # writes UNDER that worktree. A path in the MAIN checkout (e.g. a shared
        # state/brain dir an operator's after-turn ritual records into) stays
        # read-only and the write fails ("sandbox can't write …"). Opt in by listing
        # absolute paths in AGENT_GATEWAY_CODEX_WRITABLE_ROOTS — only widens the
        # workspace-write sandbox, never read-only. Field is the app-server's
        # camelCase `writableRoots` (verified against codex-cli 0.137).
        if policy.get("type") == "workspaceWrite":
            roots = [p.strip() for p in os.environ.get("AGENT_GATEWAY_CODEX_WRITABLE_ROOTS", "").split(",") if p.strip()]
            if roots:
                policy = {**policy, "writableRoots": roots}
        return policy

    def _ensure_worker(self, chat_id: int, emit: Emit) -> _AppServerWorker:
        worker = self._workers.get(chat_id)
        if worker is not None and worker.alive():
            return worker
        proc = subprocess.Popen(
            [self.codex_bin, "app-server", "--stdio"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        worker = _AppServerWorker(proc)
        worker.request("initialize", {"clientInfo": {"name": "ryuma-gateway", "version": "0.1"}}, timeout=20)
        self._workers[chat_id] = worker
        emit(AgentEvent("status", "Codex app-server worker active.", backend="codex", data={"phase": "session"}))
        return worker

    def _ensure_thread(self, worker: _AppServerWorker, workdir: Path) -> str:
        if worker.thread_id:
            return worker.thread_id
        resp = worker.request("thread/start", {
            "cwd": str(workdir),
            "approvalPolicy": "never",
            "sandboxPolicy": self._sandbox_policy(),
            "baseInstructions": full_system_prompt(),
        }, timeout=30)
        if resp.get("error"):
            raise _AppServerDown(str(resp["error"]))
        worker.thread_id = str(resp["result"]["thread"]["id"])
        return worker.thread_id

    def run(self, turn: AgentTurn, emit: Emit, stop_event: threading.Event, injections: InjectionBuffer) -> str:
        if self._turns.get(turn.chat_id, 0) >= self.max_session_turns:
            self.reset(turn.chat_id)
            emit(AgentEvent("status", "Codex session turn cap reached; starting fresh.", backend="codex", data={"phase": "session"}))
        try:
            return self._run_once(turn, emit, stop_event, injections)
        except _AppServerDown as exc:
            self.reset(turn.chat_id)  # drop the dead worker; next turn respawns fresh
            tail = "".join(self._workers.get(turn.chat_id).stderr[-6:]) if self._workers.get(turn.chat_id) else ""
            emit(AgentEvent("error", f"Codex app-server: {exc}" + (f"\n{tail}" if tail else ""), backend="codex"))
            return ""

    def _run_once(self, turn: AgentTurn, emit: Emit, stop_event: threading.Event, injections: InjectionBuffer) -> str:
        workdir = turn.workdir or self.workdir
        worker = self._ensure_worker(turn.chat_id, emit)
        self._ensure_thread(worker, workdir)
        start = worker.request("turn/start", {
            "threadId": worker.thread_id,
            # cwd + sandbox + approval MUST be set per-TURN, not just at thread/start.
            # Empirically, thread-level sandbox/approval leaves the workspace read-only
            # ("writing is blocked by the read-only sandbox"); the SAME policy on
            # turn/start actually grants writes. And cwd re-points the reused thread at
            # this turn's worktree.
            "cwd": str(workdir),
            "sandboxPolicy": self._sandbox_policy(),
            "approvalPolicy": "never",
            "input": [{"type": "text", "text": _gateway_prompt(turn, include_system=False)}],
            "model": turn.model or self.model,
            "effort": _normalize_codex_effort(turn.effort or self.effort),
        }, timeout=30)
        if start.get("error"):
            raise _AppServerDown(str(start["error"]))
        # Turn-level objects nest the id at result.turn.id (item-level notifications use
        # a flat turnId). turn/steer's expectedTurnId precondition needs THIS id.
        turn_id = str(((start.get("result") or {}).get("turn") or {}).get("id") or "")
        answer_parts: list[str] = []
        started = time.time()
        while True:
            if stop_event.is_set():
                self._safe(worker, "turn/interrupt", {"threadId": worker.thread_id, "turnId": turn_id})
                return "Stopped."
            if time.time() - started > self.timeout_sec:
                self._safe(worker, "turn/interrupt", {"threadId": worker.thread_id, "turnId": turn_id})
                return f"Codex turn timed out after {self.timeout_sec}s."
            # Fold any mid-turn messages into the LIVE turn — the feature codex exec
            # could never do. turn/steer requires the active turn id as a precondition.
            for msg in injections.drain():
                resp = self._safe(worker, "turn/steer", {
                    "threadId": worker.thread_id, "expectedTurnId": turn_id,
                    "input": [{"type": "text", "text": msg}],
                })
                if resp and resp.get("error"):
                    # Turn was already closing when the steer landed — too late to fold.
                    # Tell the user honestly rather than dropping it silently.
                    emit(AgentEvent("status", "couldn't fold — turn was already finishing; resend it", backend="codex"))
                else:
                    emit(AgentEvent("injected", "folding your message into this reply", backend="codex"))
            try:
                obj = worker.notifs.get(timeout=0.2)
            except queue.Empty:
                if not worker.alive():
                    raise _AppServerDown("worker exited mid-turn")
                continue
            done = self._handle_notif(obj, turn_id, answer_parts, emit)
            if done:
                self._turns[turn.chat_id] = self._turns.get(turn.chat_id, 0) + 1
                return "".join(answer_parts).strip() or "Codex completed with no text."

    def _handle_notif(self, obj: dict, turn_id: str, answer_parts: list[str], emit: Emit) -> bool:
        method = obj.get("method", "")
        p = obj.get("params") or {}
        if turn_id and p.get("turnId") and p["turnId"] != turn_id:
            return False  # a stray notification from another turn — ignore
        if method == "item/agentMessage/delta":
            delta = str(p.get("delta") or "")
            if delta:
                answer_parts.append(delta)
                emit(AgentEvent("thinking", delta, backend="codex", data={"phase": "writing", "stream": True}))
        elif method in ("item/reasoning/delta", "item/reasoningSummary/delta"):
            delta = str(p.get("delta") or "")
            if delta:
                emit(AgentEvent("thinking", delta, backend="codex", data={"phase": "thinking", "stream": True}))
        elif method == "item/started":
            label = _codex_item_label(p.get("item") or {})
            if label:
                emit(AgentEvent("tool", label, backend="codex", data={"category": "cmd", "phase": "tool"}))
        elif method == "turn/completed":
            return True
        elif method in ("turn/failed", "turn/aborted"):
            emit(AgentEvent("error", str(p.get("error") or method), backend="codex"))
            return True
        elif method == "_eof":
            raise _AppServerDown("app-server closed the stream")
        return False

    def _safe(self, worker: _AppServerWorker, method: str, params: dict) -> None:
        try:
            worker.request(method, params, timeout=10)
        except _AppServerDown:
            pass  # interrupt/steer best-effort — never crash the turn loop on it

    def reset(self, chat_id: int) -> None:
        worker = self._workers.pop(chat_id, None)
        if worker is not None:
            worker.terminate()
        self._turns.pop(chat_id, None)


def _codex_item_label(item: dict) -> str:
    """Human label for a started thread item (command/file/tool) for the live card."""
    cmd = item.get("command") or item.get("parsedCmd") or item.get("commandLine")
    if isinstance(cmd, list):
        cmd = " ".join(str(x) for x in cmd)
    if cmd:
        return f"running: {str(cmd)[:80]}"
    path = item.get("path") or item.get("file")
    if path:
        return f"editing {Path(str(path)).name}"
    itype = str(item.get("itemType") or item.get("type") or "").replace("_", " ").strip()
    return itype if itype and itype != "agent message" else ""


@dataclass
class _ClaudeWorker:
    proc: subprocess.Popen[str]
    lines: queue.Queue[str | None]
    stderr: list[str]
    session_id: str
    resumed: bool


@dataclass
class ClaudePrintBackend:
    """Warm-worker Claude CLI backend with stdin steering.

    The existing Claude bot remains untouched. This adapter ports the important
    worker model into the shared gateway: one process per chat, user turns sent
    as stream-json stdin messages, and injected Telegram replies written into
    the active turn when the gateway supports steering.
    """

    claude_bin: str = "claude"
    model: str = os.environ.get("CLAUDE_MODEL", "")
    fallback_model: str = os.environ.get("AGENT_GATEWAY_CLAUDE_FALLBACK_MODEL", os.environ.get("CLAUDE_FALLBACK_MODEL", ""))
    # 'dontAsk' = autonomous + works as ROOT. 'bypassPermissions'/--dangerously-skip-
    # permissions are REJECTED under root/sudo (the gateway runs as root via systemd),
    # and bare 'default' would block on a permission prompt with no TTY to answer. So a
    # headless root bot must default to dontAsk — the same mode the specialized bot uses.
    permission_mode: str = os.environ.get("CLAUDE_PERMISSION_MODE", "dontAsk")
    # dontAsk denies any tool NOT in this list, so an empty list = read-only (the
    # "read-only diagnostic mode" bug). Grant the core write tools, same as the proven
    # specialized bot, so the gateway can actually build.
    allowed_tools: str = os.environ.get("AGENT_GATEWAY_CLAUDE_ALLOWED_TOOLS", os.environ.get("CLAUDE_ALLOWED_TOOLS", "Read,Edit,Write,Bash,Glob,Grep"))
    timeout_sec: int = 900
    workdir: Path = ROOT
    home: str = os.environ.get("AGENT_GATEWAY_CLAUDE_HOME", os.environ.get("HOME", "/root"))
    thinking_tokens: int = int(os.environ.get("AGENT_GATEWAY_CLAUDE_THINKING_TOKENS", os.environ.get("MAX_THINKING_TOKENS", "4000") or "0"))
    max_session_turns: int = int(os.environ.get("AGENT_GATEWAY_CLAUDE_MAX_SESSION_TURNS", "8"))
    # Orphan-drain windows (ports the specialized bot's CONTINUOUS_READER_FIX): a
    # steered message can emit a SECOND result. Wait `quiet` for the first orphan
    # byte; once seen, keep reading up to `drain` even through long silent thinking
    # (e.g. Fable) so the orphan is absorbed into THIS reply, not leaked to the next.
    orphan_quiet_sec: float = float(os.environ.get("AGENT_GATEWAY_CLAUDE_ORPHAN_QUIET_SEC", "3"))
    orphan_drain_sec: float = float(os.environ.get("AGENT_GATEWAY_CLAUDE_ORPHAN_DRAIN_SEC", "60"))

    def __post_init__(self) -> None:
        self._workers: dict[int, _ClaudeWorker] = {}
        self._sessions: dict[int, str] = {}
        self._started: set[int] = set()
        self._turns: dict[int, int] = {}
        self._worker_sig: dict[int, tuple[str, int, str]] = {}
        self._pending_sig: tuple[str, int, str] = (self.model, self.thinking_tokens, self.permission_mode)
        self._lock = threading.RLock()

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="claude",
            label="Claude Code",
            streams=True,
            live_thinking=True,
            live_tools=True,
            steering=True,
            persistent_sessions=True,
            skills=True,
            images=True,
            voice=True,
            write_access=True,
            compact_final=False,
            model_suggestions=("claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"),
        )

    def run(
        self,
        turn: AgentTurn,
        emit: Emit,
        stop_event: threading.Event,
        injections: InjectionBuffer,
    ) -> str:
        if self._turns.get(turn.chat_id, 0) >= self.max_session_turns:
            self._reset_session(turn.chat_id)
            emit(AgentEvent("status", "Claude session turn cap reached; starting fresh.", backend="claude", data={"phase": "session"}))
        # Per-chat model/effort/MODE: if any changed, respawn the warm worker (keeping
        # the session via --resume) so the new flags take effect. A plan turn locks
        # the worker to Claude's native read-only `plan` permission mode; the next
        # write turn flips back to the build default — so we're ALWAYS back in write
        # after a one-shot /plan (different sig → respawn back). No session lost.
        permission = "plan" if turn.mode == "plan" else self.permission_mode
        desired = (turn.model or self.model, _claude_thinking_tokens(turn.effort, self.thinking_tokens), permission)
        if turn.chat_id in self._workers and self._worker_sig.get(turn.chat_id) != desired:
            self._terminate_worker(turn.chat_id)
            emit(AgentEvent("status", f"Claude reconfigured ({desired[0] or 'default'}); restarting worker.", backend="claude", data={"phase": "session"}))
        self._pending_sig = desired
        # Self-heal: if a resumed session was lost server-side ("no conversation
        # found") or its id collided ("already in use"), drop it and retry ONCE
        # from a fresh session, replaying this turn — instead of failing the user.
        last_error = ""
        for attempt in range(2):
            final, drift, error = self._run_once(turn, emit, stop_event, injections)
            if not drift:
                if error:
                    emit(AgentEvent("error", error, backend="claude"))
                    return ""
                return final
            self._reset_session(turn.chat_id)
            last_error = error
            if attempt == 0:
                emit(AgentEvent("status", "Claude session drift — restarting fresh", backend="claude", data={"phase": "session"}))
        emit(AgentEvent("error", last_error or "Claude session could not recover.", backend="claude"))
        return ""

    def _run_once(
        self,
        turn: AgentTurn,
        emit: Emit,
        stop_event: threading.Event,
        injections: InjectionBuffer,
    ) -> tuple[str, bool, str]:
        """One turn attempt. Returns (final_text, session_drift, error_text).
        Never emits the terminal error itself — run() decides heal-or-surface."""
        worker = self._ensure_worker(turn.chat_id, emit, workdir=turn.workdir)
        final_chunks: list[str] = []
        streamed_answer: list[str] = []  # the answer as it streamed (text_delta) — a fallback
        injected = False
        started = time.time()
        # Flush any output still sitting in the pipe from a prior turn's late/orphan
        # result, so it can't be misattributed to this turn.
        _drain_queue(worker.lines)
        try:
            # System prompt is already in --append-system-prompt; don't re-bill it.
            self._send_user_message(worker, _gateway_prompt(turn, include_system=False))
        except (BrokenPipeError, OSError):
            err = self._dead_worker_stderr(worker, turn.chat_id, "Claude worker exited before accepting input.")
            return "", _is_session_drift(err), err
        while True:
            if stop_event.is_set():
                self._terminate_worker(turn.chat_id)
                return "Stopped.", False, ""
            if time.time() - started > self.timeout_sec:
                self._terminate_worker(turn.chat_id)
                return "", False, f"Claude turn timed out after {self.timeout_sec}s."
            # Steer: fold mid-turn messages into the live turn immediately. Drain is
            # non-blocking, so this adds no per-line latency to streaming output.
            batch = injections.drain()
            if batch:
                injected = True
                try:
                    self._send_user_message(worker, "\n\n".join(batch))
                    emit(AgentEvent("injected", "folding your message into this reply", backend="claude"))
                except (BrokenPipeError, OSError):
                    err = self._dead_worker_stderr(worker, turn.chat_id, "Claude worker died while steering.")
                    return "", _is_session_drift(err), err
            try:
                line = worker.lines.get(timeout=0.1)
            except queue.Empty:
                if worker.proc.poll() is not None:
                    err = self._dead_worker_stderr(worker, turn.chat_id, "Claude worker exited.")
                    return "", _is_session_drift(err), err
                continue
            if line is None:
                err = self._dead_worker_stderr(worker, turn.chat_id, "Claude worker closed stdout.")
                return "", _is_session_drift(err), err
            parsed = _parse_claude_event(line)
            if parsed is None:
                continue
            event, final = parsed
            if event is not None:
                emit(event)
                # Keep the streamed ANSWER (text_delta → phase "writing") as a fallback:
                # the structured final (assistant text / result field) is occasionally
                # empty (tool-only final message, odd result subtype, worktree pulled
                # mid-turn), but the user already watched the answer stream — don't throw
                # it away and show "completed with no text".
                if event.kind == "thinking" and event.data.get("stream") and event.data.get("phase") == "writing":
                    streamed_answer.append(event.text)
            if final:
                final_chunks.append(final)
            if _is_result_event(line):
                if injected:
                    # A steered message can emit a SECOND result after this one;
                    # absorb it into THIS reply (off-by-one protection), tolerating
                    # long silent thinking before it appears.
                    final_chunks.extend(self._drain_claude_trailing(worker, emit, deadline=time.time() + self.orphan_drain_sec))
                self._started.add(turn.chat_id)
                self._turns[turn.chat_id] = self._turns.get(turn.chat_id, 0) + 1
                text = "\n\n".join(_dedupe_text_chunks(final_chunks)).strip()
                if not text:
                    text = "".join(streamed_answer).strip()  # fall back to what already streamed
                    # Log WHY the structured final was empty so the irregular cause is
                    # confirmable next time (result subtype / whether any text streamed).
                    try:
                        subtype = str((json.loads(line) or {}).get("subtype") or "?")
                    except Exception:  # noqa: BLE001
                        subtype = "?"
                    print(f"claude empty-final fallback: subtype={subtype} recovered={len(text)}c "
                          f"streamed={bool(streamed_answer)} chunks={len(final_chunks)}", flush=True)
                return text or "Claude completed with no text.", False, ""

    def _dead_worker_stderr(self, worker: "_ClaudeWorker", chat_id: int, fallback: str) -> str:
        # Give the stderr pipe a beat to drain so drift detection sees the message.
        time.sleep(0.1)
        err = "".join(worker.stderr)[-1800:] or fallback
        self._drop_worker(chat_id)
        return err

    def _reset_session(self, chat_id: int) -> None:
        self._terminate_worker(chat_id)
        self._sessions.pop(chat_id, None)
        self._started.discard(chat_id)
        self._turns.pop(chat_id, None)

    def build_cmd(self, chat_id: int, model: str, permission: str = "") -> tuple[list[str], str, bool]:
        session_id = self._sessions.get(chat_id) or str(uuid.uuid4())
        resume = chat_id in self._started
        self._sessions[chat_id] = session_id
        cmd = [
            self.claude_bin,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--resume" if resume else "--session-id",
            session_id,
            "--append-system-prompt",
            _gateway_system_context(),
            "--permission-mode",
            permission or self.permission_mode,
        ]
        if self.allowed_tools:
            cmd += ["--allowed-tools", self.allowed_tools]
        if model:
            cmd += ["--model", model]
        if self.fallback_model:
            cmd += ["--fallback-model", self.fallback_model]
        return cmd, session_id, resume

    def _ensure_worker(self, chat_id: int, emit: Emit, workdir: Path | None = None) -> _ClaudeWorker:
        with self._lock:
            existing = self._workers.get(chat_id)
            if existing is not None and existing.proc.poll() is None:
                return existing
            run_dir = workdir or self.workdir
            model, thinking, permission = getattr(self, "_pending_sig", (self.model, self.thinking_tokens, self.permission_mode))
            cmd, session_id, resume = self.build_cmd(chat_id, model, permission)
            env = os.environ.copy()
            env.setdefault("TERM", "xterm-256color")
            env["HOME"] = self.home
            if thinking > 0:
                env["MAX_THINKING_TOKENS"] = str(thinking)
            self._worker_sig[chat_id] = (model, thinking, permission)
            proc = subprocess.Popen(
                cmd,
                cwd=str(run_dir),
                env=env,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            lines: queue.Queue[str | None] = queue.Queue()
            stderr: list[str] = []
            assert proc.stdout is not None
            assert proc.stderr is not None
            threading.Thread(target=_pipe_lines, args=(proc.stdout, lines), daemon=True).start()
            threading.Thread(target=_pipe_stderr, args=(proc.stderr, stderr), daemon=True).start()
            worker = _ClaudeWorker(proc=proc, lines=lines, stderr=stderr, session_id=session_id, resumed=resume)
            self._workers[chat_id] = worker
            emit(AgentEvent("status", "Claude warm worker active" + (" (resume)." if resume else "."), backend="claude"))
            return worker

    @staticmethod
    def _send_user_message(worker: _ClaudeWorker, text: str) -> None:
        if worker.proc.stdin is None:
            raise RuntimeError("Claude worker stdin is closed")
        worker.proc.stdin.write(_claude_user_msg(text))
        worker.proc.stdin.flush()

    def _drain_claude_trailing(self, worker: _ClaudeWorker, emit: Emit, *, deadline: float) -> list[str]:
        """Two-phase orphan drain (CONTINUOUS_READER_FIX port). Phase 1: wait only
        a short quiet window for the FIRST orphan byte — cheap exit if none comes.
        Phase 2: once any orphan byte is seen, keep reading up to `deadline` even
        through long silent thinking, until the orphan's own result arrives."""
        chunks: list[str] = []
        saw_orphan = False
        quiet_deadline = time.time() + self.orphan_quiet_sec
        while True:
            now = time.time()
            # Phase 1 waits only to the quiet window; phase 2 (orphan seen) to the
            # full deadline. The get timeout is bounded by the active phase so the
            # quiet-window exit is responsive instead of stuck on a long read.
            effective = deadline if saw_orphan else quiet_deadline
            remaining = effective - now
            if remaining <= 0:
                return chunks
            try:
                line = worker.lines.get(timeout=max(0.02, min(remaining, 0.5)))
            except queue.Empty:
                continue
            if not line:
                return chunks
            parsed = _parse_claude_event(line)
            if parsed is None:
                continue
            saw_orphan = True  # phase 2: tolerate long silence until this orphan finishes
            event, final = parsed
            if event is not None:
                emit(event)
            if final:
                chunks.append(final)
            if _is_result_event(line):
                return chunks
        return chunks

    def _terminate_worker(self, chat_id: int) -> bool:
        worker = self._workers.pop(chat_id, None)
        if worker is None:
            return False
        if worker.proc.poll() is None:
            try:
                os.killpg(worker.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                worker.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(worker.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    worker.proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            _close_proc_pipes(worker.proc)
            return True
        _close_proc_pipes(worker.proc)
        return False

    def _drop_worker(self, chat_id: int) -> None:
        worker = self._workers.pop(chat_id, None)
        if worker is not None:
            _close_proc_pipes(worker.proc)

    def reset(self, chat_id: int) -> None:
        self._reset_session(chat_id)


def build_backends_from_env() -> dict[str, object]:
    backends: dict[str, object] = {"mock": MockBackend()}
    enabled = {
        item.strip().lower()
        for item in os.environ.get("AGENT_GATEWAY_BACKENDS", "mock,codex,claude").split(",")
        if item.strip()
    }
    if "codex" in enabled:
        # Default to the app-server backend: it adds mid-turn steering (turn/steer),
        # interrupt, and a real token-delta stream that `codex exec` can't do. Set
        # AGENT_GATEWAY_CODEX_MODE=exec to fall back to the one-shot exec backend.
        codex_mode = os.environ.get("AGENT_GATEWAY_CODEX_MODE", "app-server").strip().lower()
        if codex_mode == "exec":
            backends["codex"] = CodexExecBackend(
                codex_bin=os.environ.get("AGENT_GATEWAY_CODEX_BIN", "codex"),
                model=os.environ.get("AGENT_GATEWAY_CODEX_MODEL", os.environ.get("CODEX_MODEL", "gpt-5.5")),
                sandbox=os.environ.get("AGENT_GATEWAY_CODEX_SANDBOX", "workspace-write"),
                timeout_sec=int(os.environ.get("AGENT_GATEWAY_TIMEOUT", "900")),
                effort=os.environ.get("AGENT_GATEWAY_CODEX_EFFORT", os.environ.get("CODEX_REASONING_EFFORT", "medium")),
                max_session_turns=int(os.environ.get("AGENT_GATEWAY_CODEX_MAX_SESSION_TURNS", "8")),
            )
        else:
            backends["codex"] = CodexAppServerBackend(
                codex_bin=os.environ.get("AGENT_GATEWAY_CODEX_BIN", "codex"),
                model=os.environ.get("AGENT_GATEWAY_CODEX_MODEL", os.environ.get("CODEX_MODEL", "gpt-5.5")),
                sandbox=os.environ.get("AGENT_GATEWAY_CODEX_SANDBOX", "workspace-write"),
                timeout_sec=int(os.environ.get("AGENT_GATEWAY_TIMEOUT", "900")),
                effort=os.environ.get("AGENT_GATEWAY_CODEX_EFFORT", os.environ.get("CODEX_REASONING_EFFORT", "medium")),
                max_session_turns=int(os.environ.get("AGENT_GATEWAY_CODEX_MAX_SESSION_TURNS", "8")),
            )
    if "claude" in enabled:
        backends["claude"] = ClaudePrintBackend(
            claude_bin=os.environ.get("AGENT_GATEWAY_CLAUDE_BIN", "claude"),
            model=os.environ.get("AGENT_GATEWAY_CLAUDE_MODEL", os.environ.get("CLAUDE_MODEL", "")),
            fallback_model=os.environ.get("AGENT_GATEWAY_CLAUDE_FALLBACK_MODEL", os.environ.get("CLAUDE_FALLBACK_MODEL", "")),
            permission_mode=os.environ.get("AGENT_GATEWAY_CLAUDE_PERMISSION_MODE", "dontAsk"),  # works as root; see ClaudePrintBackend
            allowed_tools=os.environ.get("AGENT_GATEWAY_CLAUDE_ALLOWED_TOOLS", os.environ.get("CLAUDE_ALLOWED_TOOLS", "Read,Edit,Write,Bash,Glob,Grep")),
            timeout_sec=int(os.environ.get("AGENT_GATEWAY_TIMEOUT", "900")),
            thinking_tokens=int(os.environ.get("AGENT_GATEWAY_CLAUDE_THINKING_TOKENS", os.environ.get("MAX_THINKING_TOKENS", "4000") or "0")),
            max_session_turns=int(os.environ.get("AGENT_GATEWAY_CLAUDE_MAX_SESSION_TURNS", "8")),
        )
    return backends


def _run_streaming_process(
    cmd: list[str],
    emit: Emit,
    stop_event: threading.Event,
    *,
    cwd: Path,
    timeout_sec: int,
    parser,
) -> str:
    env = os.environ.copy()
    env.setdefault("TERM", "xterm-256color")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.time()
    final_chunks: list[str] = []
    tail: list[str] = []
    lines: queue.Queue[str | None] = queue.Queue()
    assert proc.stdout is not None

    def read_stdout() -> None:
        try:
            for stdout_line in proc.stdout or []:
                lines.put(stdout_line)
        finally:
            lines.put(None)

    threading.Thread(target=read_stdout, daemon=True).start()

    while True:
        if stop_event.is_set():
            _terminate_process(proc)
            return "Stopped."
        if time.time() - started > timeout_sec:
            _terminate_process(proc)
            emit(AgentEvent("error", f"Timed out after {timeout_sec}s."))
            return ""
        try:
            line = lines.get(timeout=0.1)
        except queue.Empty:
            if proc.poll() is not None:
                break
            continue
        if line is None:
            break
        tail.append(line)
        tail[:] = tail[-80:]
        parsed = parser(line)
        if parsed is None:
            continue
        event, final = parsed
        if event is not None:
            emit(event)
        if final:
            final_chunks.append(final)
    while True:
        try:
            line = lines.get_nowait()
        except queue.Empty:
            break
        if line:
            tail.append(line)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _terminate_process(proc)
        emit(AgentEvent("error", "Process stdout closed but the process did not exit cleanly."))
        return ""
    _close_proc_pipes(proc)
    rc = proc.returncode or 0
    if rc != 0:
        emit(AgentEvent("error", f"process exited {rc}\n{''.join(tail)[-1800:]}"))
        return ""
    return "\n\n".join(chunk.strip() for chunk in final_chunks if chunk.strip()) or "".join(tail)[-4000:].strip()


def _pipe_lines(stream, out: queue.Queue[str | None]) -> None:
    try:
        for line in stream:
            out.put(line)
    finally:
        out.put(None)


def _pipe_stderr(stream, out: list[str]) -> None:
    for line in stream:
        out.append(line)
        if len(out) > 80:
            del out[:40]


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    _close_proc_pipes(proc)


def _close_proc_pipes(proc: subprocess.Popen[str]) -> None:
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except Exception:
            pass


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        return ""


def _normalize_codex_effort(value: str) -> str:
    effort = (value or "").strip().lower()
    aliases = {"med": "medium", "hi": "high", "x": "xhigh", "xh": "xhigh", "max": "xhigh"}
    effort = aliases.get(effort, effort)
    return effort if effort in {"low", "medium", "high", "xhigh"} else "medium"


def _claude_user_msg(text: str) -> str:
    obj = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}
    return json.dumps(obj) + "\n"


def _gateway_system_context() -> str:
    return full_system_prompt()


def _drain_queue(q: "queue.Queue") -> list:
    """Non-blocking: pull and discard everything currently in a queue."""
    drained: list = []
    while True:
        try:
            drained.append(q.get_nowait())
        except queue.Empty:
            break
    return drained


def _claude_thinking_tokens(effort: str, default: int) -> int:
    """Interpret a free-text effort for Claude as a thinking-token budget. Numbers
    pass through; common words map; anything else keeps the default. This lives in
    the backend (model-specific knowledge), so the bot core stays model-agnostic
    and new effort values just work without touching the core."""
    effort = (effort or "").strip().lower()
    if not effort:
        return default
    if effort.isdigit():
        return int(effort)
    return {
        "off": 0, "none": 0, "low": 2000, "medium": 4000, "med": 4000,
        "high": 10000, "max": 32000, "ultra": 32000,
    }.get(effort, default)


def _is_session_drift(text: str) -> bool:
    """Recoverable session-state errors: the resumed session is gone or its id
    collided. These heal by starting fresh, so we retry instead of failing."""
    low = text.lower()
    return (
        "no conversation found" in low
        or "already in use" in low
        or ("session" in low and "not found" in low)
    )


def _is_result_event(line: str) -> bool:
    try:
        return json.loads(line).get("type") == "result"
    except Exception:
        return False


def _dedupe_text_chunks(chunks: list[str]) -> list[str]:
    out: list[str] = []
    for chunk in chunks:
        clean = chunk.strip()
        if not clean:
            continue
        if out and clean == out[-1]:
            continue
        out.append(clean)
    return out


def _cmd_to_str(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value).strip()
    return str(value or "").strip()


def _cmd_kind(cmd: str) -> str:
    """Coarse phase label from a shell command. Shared meaning across backends."""
    low = cmd.lower()
    if any(t in low for t in ("pytest", "unittest", "jest", "vitest", " test")):
        return "tests"
    if any(t in low for t in ("lint", "eslint", "ruff", "mypy", "tsc", "typecheck")):
        return "checks"
    if low.startswith("git ") or " git " in low:
        return "git"
    if low.split(" ", 1)[0] in {"cat", "ls", "grep", "rg", "find", "head", "tail", "sed", "less"}:
        return "reading"
    return "shell"


def _fmt_tokens(value: object) -> str | None:
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _fmt_codex_usage(usage: dict) -> str:
    parts: list[str] = []
    inp = _fmt_tokens(usage.get("input_tokens"))
    out = _fmt_tokens(usage.get("output_tokens"))
    if inp:
        parts.append(f"in {inp}")
    if out:
        parts.append(f"out {out}")
    return " · ".join(parts)


def _parse_codex_event(line: str) -> tuple[AgentEvent | None, str | None] | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    typ = str(obj.get("type") or "")
    if typ == "thread.started":
        tid = str(obj.get("thread_id") or "")
        return AgentEvent("status", "thread active", backend="codex", data={"thread_id": tid, "phase": "session"}), None
    if typ in {"turn.started", "task_started"}:
        return AgentEvent("thinking", "thinking", backend="codex", data={"phase": "thinking"}), None
    if typ in {"agent_reasoning", "reasoning"}:
        # Codex emits reasoning mostly as a phase marker; if a build carries reasoning
        # text, stream it into the feed (graceful — many carry none, then phase-only).
        rtext = str(obj.get("text") or obj.get("reasoning") or obj.get("delta") or "").strip()
        if rtext:
            return AgentEvent("thinking", rtext, backend="codex", data={"phase": "thinking", "stream": True}), None
        return AgentEvent("thinking", "reasoning", backend="codex", data={"phase": "thinking"}), None
    if typ in {"exec_command_begin", "tool_call_begin"}:
        cmd = _cmd_to_str(obj.get("command") or obj.get("cmd") or obj.get("argv"))
        if cmd:
            kind = _cmd_kind(cmd)
            return AgentEvent("tool", f"running: {cmd[:80]}", backend="codex", data={"category": "cmd", "cmd": cmd, "phase": kind}), None
        name = str(obj.get("name") or obj.get("tool_name") or "tool")
        return AgentEvent("tool", f"using {name}", backend="codex", data={"category": "tool", "phase": "tool"}), None
    if typ in {"patch_apply_begin", "file_change", "apply_patch"}:
        path = str(obj.get("path") or obj.get("file") or "").strip()
        return AgentEvent("tool", f"editing {path or 'files'}", backend="codex", data={"category": "file", "phase": "editing"}), None
    if typ in {"item.started", "item.completed"}:
        item = obj.get("item") or {}
        if not isinstance(item, dict):
            return None
        itype = str(item.get("type") or "")
        if itype == "agent_message":
            text = str(item.get("text") or "")
            # Codex sends the answer whole (no token deltas like Claude), so stream it
            # into the feed on completion — the card shows the answer building, then
            # render_final shows it clean. Skip the started-event to avoid doubling.
            if typ == "item.completed" and text:
                return AgentEvent("thinking", text, backend="codex", data={"phase": "writing", "stream": True}), text
            return None, text
        finished = typ == "item.completed"
        if itype == "command_execution":
            cmd = _cmd_to_str(item.get("command") or item.get("cmd"))
            kind = _cmd_kind(cmd) if cmd else "shell"
            label = f"{'ran' if finished else 'running'}: {cmd[:80]}" if cmd else f"{'ran' if finished else 'running'} command"
            return AgentEvent("tool", label, backend="codex", data={"category": "cmd", "cmd": cmd, "phase": f"{kind} done" if finished else kind}), None
        if itype in {"file_change", "patch"}:
            return AgentEvent("tool", "editing files", backend="codex", data={"category": "file", "phase": "editing"}), None
        if itype in {"tool_call", "function_call"}:
            name = str(item.get("name") or item.get("tool_name") or "tool")
            return AgentEvent("tool", f"{'used' if finished else 'using'} {name}", backend="codex", data={"category": "tool", "phase": "tool"}), None
    if typ == "turn.completed":
        usage = obj.get("usage") or (obj.get("payload") or {}).get("usage")
        if isinstance(usage, dict):
            strip = _fmt_codex_usage(usage)
            if strip:
                return AgentEvent("usage", strip, backend="codex"), None
    return None


_CLAUDE_TOOL_MAP = {
    "bash": ("cmd", "shell"),
    "bashoutput": ("cmd", "shell"),
    "edit": ("file", "editing"),
    "write": ("file", "editing"),
    "multiedit": ("file", "editing"),
    "notebookedit": ("file", "editing"),
    "read": ("tool", "reading"),
    "grep": ("tool", "search"),
    "glob": ("tool", "search"),
    "task": ("tool", "subagent"),
    "webfetch": ("tool", "web"),
    "websearch": ("tool", "search"),
    "todowrite": ("tool", "planning"),
}


def _claude_tool_meta(name: str) -> dict[str, str]:
    category, phase = _CLAUDE_TOOL_MAP.get(name.lower(), ("tool", "tool"))
    return {"category": category, "phase": phase}


def _claude_tool_label(name: str, inp: dict) -> str:
    """Human, terminal-style action for the live feed — the real command/file,
    not just the tool name (e.g. '$ git status', '✎ core.py', 'read foo.py')."""
    low = name.lower()
    if low in {"bash", "bashoutput"}:
        cmd = str(inp.get("command") or "").strip()
        return f"$ {cmd[:80]}" if cmd else "bash"
    if low in {"edit", "write", "multiedit", "notebookedit"}:
        fp = str(inp.get("file_path") or inp.get("notebook_path") or "").strip()
        return f"✎ {fp.rsplit('/', 1)[-1]}" if fp else low
    if low == "read":
        fp = str(inp.get("file_path") or "").strip()
        return f"read {fp.rsplit('/', 1)[-1]}" if fp else "read"
    if low in {"grep", "glob"}:
        pat = str(inp.get("pattern") or "").strip()
        return f"{low} {pat[:40]}" if pat else low
    if low == "task":
        desc = str(inp.get("description") or "").strip()
        return f"subagent: {desc[:50]}" if desc else "subagent"
    if low == "todowrite":
        return "planning"
    return name


def _fmt_claude_usage(usage: dict) -> str:
    parts: list[str] = []
    inp = _fmt_tokens(usage.get("input_tokens"))
    out = _fmt_tokens(usage.get("output_tokens"))
    if inp:
        parts.append(f"in {inp}")
    if out:
        parts.append(f"out {out}")
    return " · ".join(parts)


def _parse_claude_event(line: str) -> tuple[AgentEvent | None, str | None] | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    typ = str(obj.get("type") or "")
    if typ == "system":
        return AgentEvent("status", "session active", backend="claude", data={"phase": "session"}), None
    if typ == "assistant":
        message = obj.get("message") or {}
        texts: list[str] = []
        tools: list[str] = []
        names: list[str] = []
        for block in message.get("content") or []:
            if block.get("type") == "text" and block.get("text"):
                texts.append(str(block["text"]))
            elif block.get("type") == "tool_use":
                name = str(block.get("name") or "tool")
                names.append(name)
                tools.append(_claude_tool_label(name, block.get("input") or {}))
        if tools:
            meta = _claude_tool_meta(names[0])
            return AgentEvent("tool", " · ".join(tools), backend="claude", data=meta), ("\n\n".join(texts) if texts else None)
        if texts:
            return AgentEvent("thinking", texts[-1][:140], backend="claude", data={"phase": "thinking"}), "\n\n".join(texts)
    if typ == "stream_event":
        ev = obj.get("event") or {}
        if ev.get("type") == "content_block_delta":
            delta = ev.get("delta") or {}
            # Stream BOTH the reasoning (thinking_delta) and the answer (text_delta)
            # into the feed, so the user watches the model actually THINK, not just
            # see the answer appear. The clean answer is rebuilt for render_final.
            if delta.get("type") == "thinking_delta" and delta.get("thinking"):
                return AgentEvent("thinking", str(delta.get("thinking")), backend="claude", data={"phase": "thinking", "stream": True}), None
            if delta.get("type") == "text_delta" and delta.get("text"):
                return AgentEvent("thinking", str(delta.get("text")), backend="claude", data={"phase": "writing", "stream": True}), None
    if typ == "result":
        usage = obj.get("usage")
        if isinstance(usage, dict):
            strip = _fmt_claude_usage(usage)
            return (AgentEvent("usage", strip, backend="claude") if strip else None), str(obj.get("result") or "")
        return None, str(obj.get("result") or "")
    return None
