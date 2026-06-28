"""Raw Telegram transport for the shared agent gateway."""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


# Force IPv4 for the Telegram API host. On some VPSes the IPv6 (AAAA) lookup for
# api.telegram.org intermittently fails ("Name or service not known"), causing
# dropped poll cycles; the A record is reliable. Mirrors the specialized bots.
_ORIGINAL_GETADDRINFO = socket.getaddrinfo


def _telegram_ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host == "api.telegram.org":
        family = socket.AF_INET
    return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)


socket.getaddrinfo = _telegram_ipv4_getaddrinfo

from .backends import build_backends_from_env
from .core import (
    AgentEvent,
    AgentTurn,
    GatewayRuntime,
    LiveCard,
    ReplyContext,
    extract_suggestions,
    is_status_frame,
)
from dataclasses import replace

from .core import Attachment
from .formatting import md_to_html, split_message
from .hooks import load_hook
from .livestore import LiveStore, bot_id_from_token
from .media import AlbumBuffer, attachment_for, extract_media, save_path, transcribe, voice_enabled
from .merge_gate import MergeGate
from .post_turn import PostTurnRunner, extract_post_turn_request
from .project import brand_emoji, brand_name
from .skills import activation_prompt, discover_skills
from .worktrees import WorktreeBroker


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = ROOT / "state" / "agent-gateway"
# Shared snapshot store: every bot writes here under its bot id; the one standalone
# Mini App server reads it. Lives outside per-profile STATE_DIR so all bots share it.
LIVE_STORE_DIR = DEFAULT_STATE_DIR / "live"
TELEGRAM_LIMIT = 3600
EDIT_THROTTLE_SEC = float(os.environ.get("AGENT_GATEWAY_EDIT_THROTTLE_SEC", "1.5"))
# Re-render the live card on this cadence even with NO new events, so the elapsed
# clock never looks frozen during quiet phases (Claude warm-up, Codex's delta-less
# thinking). Backs off whenever a real event edited recently — never fights the stream.
HEARTBEAT_SEC = float(os.environ.get("AGENT_GATEWAY_HEARTBEAT_SEC", "4.0"))

# Registered with Telegram so the bot shows the "Menu" / slash autocomplete
# (the burger menu) instead of forcing the user to type /help.
BOT_COMMANDS: list[tuple[str, str]] = [
    ("plan", "Plan a task read-only → tap ⚡ to build it"),
    ("new", "Reset the session"),
    ("stop", "Cancel the active run"),
    ("status", "Backend, model, effort"),
    ("model", "Pick agent + model"),
    ("effort", "Set reasoning effort"),
    ("help", "Show help"),
]


@dataclass(frozen=True)
class TelegramGatewayConfig:
    token: str
    allowed_user_ids: set[int]
    bot_username: str = ""
    bot_id: int = 0
    fixed_backend: bool = False


def load_config() -> TelegramGatewayConfig:
    token = os.environ.get("AGENT_GATEWAY_TELEGRAM_TOKEN", "").strip()
    allowed_raw = os.environ.get("AGENT_GATEWAY_ALLOWED_USER_IDS", "").strip()
    if not token:
        raise RuntimeError("AGENT_GATEWAY_TELEGRAM_TOKEN is required")
    if not allowed_raw:
        raise RuntimeError("AGENT_GATEWAY_ALLOWED_USER_IDS is required")
    allowed = {int(x) for x in allowed_raw.replace(" ", "").split(",") if x}
    bot_id = 0
    try:
        bot_id = int(token.split(":", 1)[0])
    except Exception:
        bot_id = 0
    username = os.environ.get("AGENT_GATEWAY_BOT_USERNAME", "").strip().lstrip("@")
    if not username:
        try:
            data = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=15).json()
            username = str((data.get("result") or {}).get("username") or "")
        except Exception:
            username = ""
    fixed_backend = os.environ.get("AGENT_GATEWAY_FIXED_BACKEND", "").strip().lower() in {"1", "true", "yes", "on"}
    return TelegramGatewayConfig(
        token=token,
        allowed_user_ids=allowed,
        bot_username=username,
        bot_id=bot_id,
        fixed_backend=fixed_backend,
    )


class TelegramClient:
    def __init__(self, token: str) -> None:
        self.token = token
        self.base = f"https://api.telegram.org/bot{token}"

    def api(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        res = requests.post(f"{self.base}/{method}", json=payload or {}, timeout=35)
        res.raise_for_status()
        return res.json()

    def send(self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None) -> int | None:
        # Split long output on natural boundaries; render each chunk as HTML so
        # **bold**/*italic*/`code`/links show properly. Markup rides the last chunk.
        chunks = split_message(text) or [""]
        first_id: int | None = None
        for idx, chunk in enumerate(chunks):
            payload: dict[str, Any] = {"chat_id": chat_id, "text": md_to_html(chunk), "parse_mode": "HTML"}
            if reply_markup and idx == len(chunks) - 1:
                payload["reply_markup"] = reply_markup
            try:
                data = self.api("sendMessage", payload)
            except Exception:  # noqa: BLE001 - a bad HTML render must never drop the message.
                payload.pop("parse_mode", None)
                payload["text"] = chunk[:TELEGRAM_LIMIT]
                data = self.api("sendMessage", payload)
            mid = int((data.get("result") or {}).get("message_id") or 0) or None
            if first_id is None:
                first_id = mid
        return first_id

    def edit(self, chat_id: int, message_id: int, text: str, reply_markup: dict[str, Any] | None = None) -> None:
        # The live card is short; if a final answer is too long it is sent
        # separately (see the final handler), so here we keep to the first chunk.
        rendered = md_to_html(split_message(text)[0])
        payload: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": rendered, "parse_mode": "HTML"}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        try:
            self.api("editMessageText", payload)
        except Exception:  # noqa: BLE001 - fall back to plain text on any HTML/parse error.
            payload.pop("parse_mode", None)
            payload["text"] = text[:TELEGRAM_LIMIT]
            self.api("editMessageText", payload)

    def chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Show Telegram's native '… is typing' indicator. Expires after ~5s, so it
        must be refreshed on a loop while the agent works (see _typing_loop)."""
        try:
            self.api("sendChatAction", {"chat_id": chat_id, "action": action})
        except Exception:  # noqa: BLE001 - a cosmetic indicator must never break a turn.
            pass

    def answer_callback(self, callback_id: str) -> None:
        self.api("answerCallbackQuery", {"callback_query_id": callback_id})

    def set_my_commands(self, commands: list[tuple[str, str]]) -> None:
        self.api("setMyCommands", {"commands": [{"command": c, "description": d} for c, d in commands]})

    def get_updates(self, offset: int) -> list[dict[str, Any]]:
        data = self.api("getUpdates", {"offset": offset, "timeout": 25, "allowed_updates": ["message", "callback_query"]})
        return list(data.get("result") or [])

    def file_path(self, file_id: str) -> str:
        data = self.api("getFile", {"file_id": file_id})
        return str((data.get("result") or {}).get("file_path") or "")

    def download(self, file_path: str, dest: Path) -> None:
        url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
        res = requests.get(url, timeout=60)
        res.raise_for_status()
        dest.write_bytes(res.content)


class TelegramGatewayApp:
    def __init__(self, config: TelegramGatewayConfig, runtime: GatewayRuntime) -> None:
        self.config = config
        self.client = TelegramClient(config.token)
        self.runtime = runtime
        self.suggestions: dict[str, str] = {}
        self._suggestion_serial = 0
        self.worktrees = WorktreeBroker()
        self.merge_gate = MergeGate(repo=self.worktrees.policy.repo)
        self.albums = AlbumBuffer(
            self._flush_album,
            debounce_sec=float(os.environ.get("AGENT_GATEWAY_ALBUM_DEBOUNCE_SEC", "2.0")),
        )
        self.auto_land = _truthy(os.environ.get("AGENT_GATEWAY_AUTO_LAND"))
        # Smart-merge (default ON): auto-land a clean branch to the base, tap-gate
        # only when a conflict is likely. Set AGENT_GATEWAY_SMART_MERGE=0 to always gate.
        self.smart_merge = os.environ.get("AGENT_GATEWAY_SMART_MERGE", "1").strip().lower() not in {"0", "false", "no", "off"}
        self._startup_worktrees: list[dict[str, object]] = []  # parallel branches awaiting land, surfaced on restart
        self._land_tokens: dict[str, str] = {}
        self._land_serial = 0
        self._discard_tokens: dict[str, str] = {}  # token -> branch, for the 🗑 Discard escape hatch
        self._build_tokens: dict[str, str] = {}  # plan text keyed by token for the ⚡ Build this button
        self._build_serial = 0
        self._skill_prompts: dict[str, str] = {}
        self._skill_serial = 0
        self.hook = load_hook()  # private brain augmentation; NoOp by default
        # Universal live Mini App: this bot is a WRITER. It drops each snapshot into
        # the shared on-disk store keyed by its bot id; ONE standalone server (see
        # agent_gateway_webapp.py) reads the store and serves the Mini App for every
        # bot off one port + one tunnel — so N bots need no extra ports/routes/secrets.
        self.live_states: dict[int, dict] = {}  # in-memory mirror (legacy in-process server)
        self.webapp_enabled = _truthy(os.environ.get("AGENT_GATEWAY_WEBAPP"))
        self.webapp_port = int(os.environ.get("AGENT_GATEWAY_WEBAPP_PORT", "8847"))
        self.webapp_token = os.environ.get("AGENT_GATEWAY_WEBAPP_TOKEN", "")
        self.webapp_url = os.environ.get("AGENT_GATEWAY_WEBAPP_URL", "").strip()
        # Run the server in-process only if explicitly asked (legacy single-bot). The
        # default is the standalone server, so bots don't fight over the port.
        self.webapp_inprocess = _truthy(os.environ.get("AGENT_GATEWAY_WEBAPP_INPROCESS"))
        self.state_dir = Path(os.environ.get("AGENT_GATEWAY_STATE_DIR", DEFAULT_STATE_DIR))
        self.bot_key = bot_id_from_token(self.config.token, fallback=os.environ.get("AGENT_GATEWAY_PROFILE", "bot"))
        self.livestore = LiveStore(LIVE_STORE_DIR)
        self.offset_file = self.state_dir / "offset"

    def _publish(self, chat_id: int, snapshot: dict) -> None:
        """Keep the in-memory snapshot mirror (used by /status). The Mini App / disk
        store is retired — the live view is now the in-chat thought stream."""
        self.live_states[chat_id] = snapshot

    def _webapp_button(self, chat_id: int) -> dict[str, Any] | None:
        # Mini App retired: the chat IS the live view now. No Live view button.
        return None

    def run_forever(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Worktree hygiene on every start: reclaim landed/empty worktrees (kills the
        # restart leak), and surface any that still hold unlanded work so the operator
        # sees parallel branches the moment the bot comes back.
        try:
            self._startup_worktrees = self.worktrees.sweep_startup(self.merge_gate.base)
        except Exception as exc:  # noqa: BLE001 - cleanup must never block startup.
            print(f"worktree sweep failed: {exc}", flush=True)
            self._startup_worktrees = []
        if self._startup_worktrees:
            print(f"startup: {len(self._startup_worktrees)} parallel worktree(s) awaiting land", flush=True)
        if self.webapp_enabled and self.webapp_inprocess:
            # Legacy single-bot path: serve the Mini App from inside this process.
            # The universal path runs ONE standalone server (agent_gateway_webapp.py)
            # reading the shared store, so multiple bots never fight over the port.
            try:
                from .webserver import start_webapp_server
                start_webapp_server(port=self.webapp_port, token=self.webapp_token, snapshot_for=self.live_states.get)
                print(f"webapp live server (in-process) on 127.0.0.1:{self.webapp_port}", flush=True)
            except Exception as exc:  # noqa: BLE001 - the bot must run even if the web view can't.
                print(f"webapp server failed: {exc}", flush=True)
        elif self.webapp_enabled:
            print(f"webapp: writer mode (bot {self.bot_key}); standalone server serves the Mini App", flush=True)
        try:
            self.client.set_my_commands(BOT_COMMANDS)  # the burger menu / slash autocomplete
        except Exception as exc:  # noqa: BLE001 - menu registration must not block polling.
            print(f"set_my_commands failed: {_redact_token(str(exc), self.config.token)}", flush=True)
        if _truthy(os.environ.get("AGENT_GATEWAY_SEND_ONLINE_MESSAGE")):
            self._send_online_message()
        # Schedule seam: let the personal brain start background work (alerts/digests).
        try:
            if getattr(self.hook, "start_background", None):
                self.hook.start_background(self._broadcast_to_operators)
        except Exception as exc:  # noqa: BLE001 - a broken brain must never block polling.
            print(f"hook start_background failed: {_redact_token(str(exc), self.config.token)}", flush=True)
        offset = int(self.offset_file.read_text().strip() or "0") if self.offset_file.exists() else 0
        while True:
            try:
                updates = self.client.get_updates(offset)
                for update in updates:
                    offset = max(offset, int(update.get("update_id") or 0) + 1)
                    self.offset_file.write_text(str(offset))
                    self.handle_update(update)
            except Exception as exc:  # noqa: BLE001
                time.sleep(3)
                print(f"gateway poll error: {_redact_token(str(exc), self.config.token)}", flush=True)

    def _send_or_edit(self, chat_id: int, text: str, markup, message_id) -> None:
        if message_id:
            try:
                self.client.edit(chat_id, message_id, text, reply_markup=markup)
                return
            except Exception:  # noqa: BLE001 - fall back to a fresh message.
                pass
        self.client.send(chat_id, text, reply_markup=markup)

    def _setup_picker(self, chat_id: int, message_id=None) -> None:
        """Unified agent→model picker (BotFather-style). Many backends → pick the
        agent first; one (fixed) → jump straight to its models."""
        backends = self.runtime.backend_names()
        if len(backends) <= 1:
            self._show_models(chat_id, backends[0] if backends else self.runtime.backend_for_chat(chat_id), message_id)
            return
        cur = self.runtime.backend_for_chat(chat_id)
        rows = [[{"text": ("✓ " if b == cur else "") + b, "callback_data": f"pa:{b}"}] for b in backends]
        self._send_or_edit(chat_id, "Pick an agent:", {"inline_keyboard": rows}, message_id)

    def _show_models(self, chat_id: int, backend: str, message_id=None) -> None:
        obj = self.runtime.backends.get(backend)
        models = list(getattr(obj.capabilities, "model_suggestions", ()) if obj else ())
        cur_model = self.runtime.model_for_chat(chat_id)
        rows = [[{"text": ("✓ " if m == cur_model else "") + m, "callback_data": f"pm:{backend}:{m}"}] for m in models]
        if len(self.runtime.backend_names()) > 1 and not self.config.fixed_backend:
            rows.append([{"text": "← agents", "callback_data": "pa:"}])
        label = f"{backend} · pick a model:" if models else f"{backend} · type /model <name> (no suggestions)"
        self._send_or_edit(chat_id, label, {"inline_keyboard": rows} if rows else None, message_id)

    def _broadcast_to_operators(self, text: str) -> None:
        """Send a message to every allowed operator — used by the brain's background
        loop (scheduled alerts/digests) via the start_background seam."""
        for chat_id in sorted(self.config.allowed_user_ids):
            try:
                self.client.send(chat_id, text)
            except Exception:  # noqa: BLE001 - one failed deliver must not stop the loop.
                pass

    def _send_online_message(self) -> None:
        profile = os.environ.get("AGENT_GATEWAY_PROFILE", "").strip()
        backend = self.runtime.backend_for_chat(0)
        label = profile or backend
        text = (
            f"{brand_emoji()} {brand_name()} · {label} is live.\n"
            "Just talk to it — describe a task and watch it work."
        )
        reply_markup = None
        if self._startup_worktrees:
            text += f"\n\n📝 {len(self._startup_worktrees)} unsaved change-set(s) from before — tap to keep:"
            rows: list[list[dict[str, str]]] = []
            for idx, wt in enumerate(self._startup_worktrees[:8], start=1):
                branch = str(wt.get("branch"))
                self._land_serial += 1
                key = f"startup.{self._land_serial}"
                self._land_tokens[key] = branch
                rows.append([{"text": f"✅ Keep change-set {idx} ({wt.get('ahead')})", "callback_data": f"land:{key}"}])
            while len(self._land_tokens) > 60:
                self._land_tokens.pop(next(iter(self._land_tokens)))
            reply_markup = {"inline_keyboard": rows}
        for chat_id in sorted(self.config.allowed_user_ids):
            try:
                self.client.send(chat_id, text, reply_markup=reply_markup)
                print(f"startup message sent to Telegram user {chat_id}", flush=True)
            except Exception as exc:  # noqa: BLE001 - startup proof must not kill polling.
                clean = _redact_token(str(exc), self.config.token)
                print(f"startup message failed for {chat_id}: {clean}", flush=True)

    def handle_update(self, update: dict[str, Any]) -> None:
        if update.get("callback_query"):
            self._handle_callback(update["callback_query"])
            return
        msg = update.get("message") or {}
        chat = msg.get("chat") or {}
        user = msg.get("from") or {}
        chat_id = int(chat.get("id") or 0)
        user_id = int(user.get("id") or 0)
        if user_id not in self.config.allowed_user_ids:
            self.client.send(chat_id, "Not authorized.")
            return
        refs = extract_media(msg)
        if refs:
            caption = str(msg.get("caption") or "").strip()
            if not self._is_for_me(chat, msg, caption):
                return
            group_id = str(msg.get("media_group_id") or "")
            if group_id:
                attachments, _ = self._download_refs(chat_id, refs)
                self.albums.add(group_id, chat_id, user_id, attachments, caption)
                return
            self._handle_media(chat_id, user_id, refs, caption)
            return
        text = str(msg.get("text") or "").strip()
        if not text:
            return
        if not self._is_for_me(chat, msg, text):
            return
        text = self._strip_mention(text)
        if text.startswith("/"):
            self._handle_command(chat_id, user_id, text)
            return
        reply_context = self._reply_context(msg)
        self._submit(chat_id, user_id, text, reply_context)

    def _handle_callback(self, callback: dict[str, Any]) -> None:
        user = callback.get("from") or {}
        msg = callback.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = int(chat.get("id") or 0)
        user_id = int(user.get("id") or 0)
        callback_id = str(callback.get("id") or "")
        if user_id not in self.config.allowed_user_ids:
            self.client.answer_callback(callback_id)
            self.client.send(chat_id, "Not authorized.")
            return
        self.client.answer_callback(callback_id)
        data = str(callback.get("data") or "")
        mid = int(msg.get("message_id") or 0) or None  # edit the picker in place (BotFather-style drill-down)
        if data.startswith("land:"):
            self._handle_land(chat_id, data[5:])
            return
        if data.startswith("disc:"):  # 🗑 Discard — first tap: ask to confirm (mobile dialogs get suppressed)
            key = data[5:]
            branch = self._discard_tokens.get(key)
            if not branch:
                self.client.send(chat_id, "That expired — your changes are still here if you need them.")
                return
            self.client.send(
                chat_id,
                "🗑 Throw away these changes? They can be recovered for a while if you "
                "change your mind. Tap to confirm:",
                reply_markup={"inline_keyboard": [[{"text": "🗑 Yes, throw away", "callback_data": f"dscok:{key}"}]]},
            )
            return
        if data.startswith("dscok:"):  # 🗑 Discard — second tap: actually destroy the worktree + branch
            self._handle_discard(chat_id, data[6:])
            return
        if data.startswith("build:"):
            self._handle_build(chat_id, user_id, data[6:])
            return
        if data.startswith("pa:"):  # picked an agent (or '←' back) in the agent→model tree
            backend = data[3:]
            if backend:
                self._show_models(chat_id, backend, mid)
            else:
                self._setup_picker(chat_id, mid)
            return
        if data.startswith("pm:"):  # picked backend:model — set both, confirm in place
            _, backend, model = data.split(":", 2)
            if not self.config.fixed_backend:
                try:
                    self.runtime.select_backend(chat_id, backend)
                except ValueError:
                    pass
            self.runtime.set_model(chat_id, model)
            self._send_or_edit(chat_id, f"✓ {backend} · {model}", None, mid)
            return
        if data.startswith("skill:"):
            prompt = self._skill_prompts.get(data[6:])
            if not prompt:
                self.client.send(chat_id, "Skill menu expired — send /skills again.")
                return
            self._submit(chat_id, user_id, prompt, None)
            return
        if data.startswith("sugg:"):
            prompt = self.suggestions.get(data[5:])
            if not prompt:
                self.client.send(chat_id, "Suggestion expired. Type the next prompt.")
                return
            self._submit(chat_id, user_id, prompt, None)

    def _handle_build(self, chat_id: int, user_id: int, key: str) -> None:
        """The ⚡ Build this tap: execute the plan from a prior /plan turn as a real
        build (worktree, edits, merge gate)."""
        plan = self._build_tokens.pop(key, None)
        if not plan:
            self.client.send(chat_id, "Build action expired — send /plan again, or just describe the task to build it directly.")
            return
        self._submit(chat_id, user_id, f"Execute this plan now:\n\n{plan}", None, mode_override="write")

    def _handle_land(self, chat_id: int, key: str) -> None:
        branch = self._land_tokens.pop(key, None)
        if not branch:
            self.client.send(chat_id, "That save expired — just send the task again.")
            return
        result = self.merge_gate.land(branch)
        if result.ok:
            self.worktrees.remove_branch(branch)  # landed → reclaim the worktree
            self.client.send(chat_id, "✅ Saved.")
            return
        extra = ("\nConflicts: " + ", ".join(result.conflicts)) if result.conflicts else ""
        self.client.send(chat_id, f"⚠ Couldn't save — {result.message}{extra}\nYour changes are untouched.")

    def _handle_discard(self, chat_id: int, key: str) -> None:
        branch = self._discard_tokens.pop(key, None)
        self._land_tokens.pop(key, None)  # the LAND option for the same branch is now moot
        if not branch:
            self.client.send(chat_id, "That expired — your changes are still here if you need them.")
            return
        self.worktrees.discard_branch(branch)
        self.client.send(chat_id, "🗑 Thrown away.")

    def _download_refs(self, chat_id: int, refs: list) -> tuple[list[Attachment], str]:
        """Download each media ref; transcribe voice when possible. Returns
        (attachments, transcript). Project-neutral — no product prompts here."""
        attachments: list[Attachment] = []
        transcript = ""
        now_ms = int(time.time() * 1000)
        for index, ref in enumerate(refs):
            try:
                remote = self.client.file_path(ref.file_id)
                dest = save_path(chat_id, ref, now_ms + index)
                self.client.download(remote, dest)
            except Exception as exc:  # noqa: BLE001 - a bad fetch must not kill polling.
                self.client.send(chat_id, f"Could not fetch attachment: {type(exc).__name__}")
                continue
            if ref.kind == "voice" and voice_enabled():
                try:
                    text = transcribe(dest)
                except Exception:  # noqa: BLE001 - fall back to attaching the audio file.
                    text = None
                if text:
                    transcript = f"{transcript}\n{text}".strip()
                    # The single user-facing echo (clipped 🎤) is sent once in
                    # _handle_media — don't echo per-ref here too (was double-showing).
                    continue
            attachments.append(attachment_for(ref, dest))
        return attachments, transcript

    def _typing_loop(self, chat_id: int, stop: threading.Event) -> None:
        """Keep Telegram's native 'typing…' indicator alive while a turn runs — it
        expires after ~5s, so refresh every 4s. Universal: every backend gets it,
        which matters most for backends (Codex) that don't stream a live feed."""
        while not stop.is_set():
            self.client.chat_action(chat_id, "typing")
            stop.wait(4.0)

    def _handle_media(self, chat_id: int, user_id: int, refs: list, caption: str) -> None:
        """Download inbound media and submit one turn carrying the attachments."""
        attachments, transcript = self._download_refs(chat_id, refs)
        text = transcript or caption
        if not text and attachments:
            text = "Read the attached file(s) and respond."
        if not text and not attachments:
            return
        if transcript:
            # Echo what we heard so a voice note is visibly confirmed (display is
            # clipped to stay under Telegram's 4096 cap; the FULL transcript still
            # goes to the agent). Mirrors the specialized bots, but backend-neutral.
            if len(transcript) <= 3800:
                echo = f"🎤 “{transcript}”"
            else:
                echo = f"🎤 “{transcript[:3800]}…” (clipped for display — full text sent to the agent)"
            try:
                self.client.send(chat_id, echo)
            except Exception:  # noqa: BLE001 - the echo must never block the turn.
                pass
        self._submit(chat_id, user_id, text, None, attachments=tuple(attachments))

    def _stage_attachments(self, workdir: Path, attachments: tuple[Attachment, ...]) -> tuple[Attachment, ...]:
        """Copy inbound files INTO the agent's worktree so its sandbox (which only
        reads its own checkout) can actually open them — /tmp is unreadable."""
        if not attachments or workdir is None:
            return attachments
        staged: list[Attachment] = []
        dest_dir = workdir / ".ryuma-inbound"
        for att in attachments:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / att.path.name
                shutil.copy2(att.path, dest)
                staged.append(Attachment(path=dest, kind=att.kind, caption=att.caption))
            except Exception:  # noqa: BLE001 - fall back to the original path.
                staged.append(att)
        return tuple(staged)

    def _flush_album(self, chat_id: int, user_id: int, attachments: tuple[Attachment, ...], caption: str) -> None:
        """One turn for a whole album, so the model sees every image together."""
        if not attachments:
            return
        text = caption or "Read the attached files and respond."
        self._submit(chat_id, user_id, text, None, attachments=attachments)

    def _is_for_me(self, chat: dict[str, Any], msg: dict[str, Any], text: str) -> bool:
        if chat.get("type") not in {"group", "supergroup"}:
            return True
        if (msg.get("from") or {}).get("is_bot"):
            return False
        uname = self.config.bot_username.lower()
        if uname and f"@{uname}" in text.lower():
            return True
        if re.search(r"@\w+", text):
            return False
        rt = msg.get("reply_to_message") or {}
        return bool(self.config.bot_id and (rt.get("from") or {}).get("id") == self.config.bot_id)

    def _strip_mention(self, text: str) -> str:
        if not self.config.bot_username:
            return text.strip()
        return re.sub(rf"@{re.escape(self.config.bot_username)}\b", "", text, flags=re.IGNORECASE).strip()

    def _reply_context(self, msg: dict[str, Any]) -> ReplyContext | None:
        rt = msg.get("reply_to_message") or {}
        text = str(rt.get("text") or rt.get("caption") or "").strip()
        if not text or is_status_frame(text):
            return None
        who = str((rt.get("from") or {}).get("first_name") or "previous message")
        return ReplyContext(author=who, text=text)

    def _handle_command(self, chat_id: int, user_id: int, text: str) -> None:
        cmd, _, rest = text.partition(" ")
        cmd = cmd.split("@", 1)[0].lower()
        rest = rest.strip()
        if cmd in {"/start", "/help"}:
            self.client.send(chat_id, self._help())
            return
        if cmd in {"/model", "/agent"}:
            # Unified agent→model picker. No arg → the tap-tree; an arg still sets
            # directly (/model <name> any model; /agent <name> switch backend).
            if not rest:
                self._setup_picker(chat_id)
                return
            if cmd == "/model":
                self.runtime.set_model(chat_id, rest)
                self.client.send(chat_id, f"Model → {rest}")
                return
            if self.config.fixed_backend:
                self.client.send(chat_id, f"Fixed agent: {self.runtime.backend_for_chat(chat_id)}")
                return
            try:
                self.runtime.select_backend(chat_id, rest)
                self.client.send(chat_id, f"Agent → {rest}")
            except ValueError as exc:
                self.client.send(chat_id, str(exc))
            return
        if cmd == "/effort":
            if rest:
                self.runtime.set_effort(chat_id, rest)
                self.client.send(chat_id, f"Effort → {rest}")
            else:
                cur = self.runtime.effort_for_chat(chat_id) or "(backend default)"
                self.client.send(chat_id, f"Effort: {cur}\nUsage: /effort <value> — e.g. /effort high, or a number. Interpreted by the current agent.")
            return
        if cmd == "/plan":
            if not rest:
                self.client.send(chat_id, "Usage: /plan <task> — a read-only planning turn (no edits, no worktree). "
                                          "The plan comes back with a ⚡ Build this button. Default (no command) builds directly.")
                return
            self._submit(chat_id, user_id, rest, None, mode_override="plan")
            return
        if cmd == "/new":
            dropped = self.runtime.reset_chat(chat_id)
            self.worktrees.reset(chat_id)
            self.client.send(chat_id, f"Fresh boundary set. Dropped {dropped} queued turn(s).")
            return
        if cmd == "/stop":
            stopped = self.runtime.stop(chat_id)
            self.client.send(chat_id, "Stopped." if stopped else "Nothing running.")
            return
        if cmd == "/status":
            st = self.runtime.status(chat_id)
            self.client.send(
                chat_id,
                f"agent {st['backend']} · model {st['model']} · effort {st['effort']} · active {st['active'] or 'no'} · queued {st['queued']}",
            )
            return
        self._submit(chat_id, user_id, text, None)

    def _submit(self, chat_id: int, user_id: int, text: str, reply_context: ReplyContext | None, attachments: tuple[Attachment, ...] = (), mode_override: "Mode | None" = None) -> None:
        backend = self.runtime.backend_for_chat(chat_id)
        # If a steering-capable turn is already running, FOLD this message into it
        # with NO new card — it lands in the original turn's live message.
        fold = AgentTurn(chat_id=chat_id, user_id=user_id, text=text, backend=backend,
                         reply_context=reply_context, attachments=attachments)
        if self.runtime.try_steer(fold):
            return
        mode = mode_override or self.runtime.mode_for_chat(chat_id)
        backend_obj = self.runtime.backends.get(backend)
        label = backend_obj.capabilities.label if backend_obj is not None else backend
        steering = bool(backend_obj.capabilities.steering) if backend_obj is not None else False
        card = LiveCard(
            backend=backend,
            mode=mode,
            label=label,
            brand=brand_emoji(),
            steering=steering,
            model=self.runtime.model_for_chat(chat_id) or getattr(backend_obj, "model", ""),
            effort=self.runtime.effort_for_chat(chat_id) or getattr(backend_obj, "effort", ""),
        )
        self._publish(chat_id, card.snapshot())
        message_id = self.client.send(chat_id, card.render_live(), reply_markup=self._webapp_button(chat_id))
        last_edit = 0.0
        edit_lock = threading.Lock()
        turn_workdir: Path | None = None
        turn_branch: str | None = None

        def _redraw() -> None:
            """Re-render the live card (thread-safe). Shared by stream events and the
            heartbeat, so the elapsed clock keeps advancing during quiet phases when no
            event would otherwise trigger an edit."""
            nonlocal last_edit
            if not message_id:
                return
            with edit_lock:
                try:
                    card.queue_depth = self.runtime.status(chat_id).get("queued", 0)
                    if turn_workdir:
                        card.diff = _worktree_diff(turn_workdir) or card.diff
                    self.client.edit(chat_id, message_id, card.render_live(), reply_markup=self._webapp_button(chat_id))
                    last_edit = time.time()
                except Exception:
                    pass

        # Native 'typing…' indicator for the whole turn — refreshed on a thread,
        # stopped at the final/error event / any early exit below.
        typing_stop = threading.Event()
        threading.Thread(target=self._typing_loop, args=(chat_id, typing_stop), daemon=True).start()

        def _heartbeat() -> None:
            # Tick the card so the clock never looks frozen; skip when a real event
            # edited within the window, so the heartbeat never fights the live stream.
            while not typing_stop.wait(HEARTBEAT_SEC):
                if time.time() - last_edit >= HEARTBEAT_SEC:
                    _redraw()

        threading.Thread(target=_heartbeat, daemon=True).start()

        def emit(event: AgentEvent) -> None:
            if event.kind == "final":
                typing_stop.set()
                final_text, reply_markup = self._final_text_and_markup(chat_id, mode, event.text, cwd=turn_workdir)
                if mode == "write" and turn_branch:
                    note, row = self._land_controls(chat_id, turn_branch)
                    if note:
                        final_text = f"{final_text}\n\n{note}".strip()
                    if row:
                        reply_markup = _prepend_keyboard_row(reply_markup, row)
                elif mode == "plan" and (event.text or "").strip():
                    # Plan turn → the primary action is build/refine, so show ONLY
                    # ⚡ Build this and SUPPRESS the generic SUGGEST continuations
                    # (the SUGGEST line was already stripped from the text above).
                    self._build_serial += 1
                    key = str(self._build_serial)
                    self._build_tokens[key] = (event.text or "").strip()[:6000]
                    reply_markup = {"inline_keyboard": [[{"text": "⚡ Build this", "callback_data": f"build:{key}"}]]}
                card.update(AgentEvent("final", final_text, backend=event.backend, data=event.data))
                self._publish(chat_id, card.snapshot())
                rendered = card.render_final()
                # Personal-brain signature (model · effort · agent) — opt-in, never
                # imposed by the neutral gateway. A broken hook just yields no footer.
                try:
                    footer = (getattr(self.hook, "final_footer", None) and self.hook.final_footer(turn)) or ""
                except Exception:  # noqa: BLE001 - a broken brain must never break the reply.
                    footer = ""
                if footer:
                    rendered = f"{rendered}\n\n{footer}".strip()
                # Resolve the live card IN PLACE: it morphs into the answer, and any
                # overflow spills into bare continuation messages with the buttons on
                # the last chunk. The data card renders exactly once per turn — it
                # never folds and never doubles (the old '✅ done' stub is gone).
                self._resolve_in_place(chat_id, message_id, rendered, reply_markup)
                try:
                    self.hook.after_turn(turn, event.text)
                except Exception:  # noqa: BLE001 - recording must never break the reply.
                    pass
                return
            if event.kind == "error":
                # Terminal: the backend gave up on this turn and emits NO final event,
                # so nothing else will stop the typing indicator or collapse the card.
                # Do both here — otherwise the user sees a phantom 'typing…' over a
                # frozen 'starting' card while the Mini App shows the error.
                typing_stop.set()
                card.update(event)
                self._publish(chat_id, card.snapshot())
                if message_id:
                    with edit_lock:
                        try:
                            self.client.edit(chat_id, message_id, card.render_final(), reply_markup=self._webapp_button(chat_id))
                        except Exception:
                            try:
                                self.client.send(chat_id, card.render_final())
                            except Exception:
                                pass
                return
            card.update(event)
            self._publish(chat_id, card.snapshot())  # publish live (web view polls faster than chat edits)
            if message_id and time.time() - last_edit > EDIT_THROTTLE_SEC:
                _redraw()

        try:
            pending_turn = AgentTurn(
                chat_id=chat_id,
                user_id=user_id,
                text=text,
                backend=backend,
                mode=mode,
                reply_context=reply_context,
                attachments=attachments,
            )
            assignment = self.worktrees.prepare(pending_turn)
            if assignment is not None:
                turn_workdir = assignment.path
                turn_branch = assignment.branch
                attachments = self._stage_attachments(turn_workdir, attachments)
                # worktree readiness is internal plumbing — not shown on the card.
            turn = AgentTurn(
                chat_id=chat_id,
                user_id=user_id,
                text=text,
                backend=backend,
                mode=mode,
                reply_context=reply_context,
                workdir=turn_workdir,
                attachments=attachments,
                model=self.runtime.model_for_chat(chat_id),
                effort=self.runtime.effort_for_chat(chat_id),
            )
            try:
                augment = self.hook.before_turn(turn) or ""
            except Exception:  # noqa: BLE001 - a broken brain must never break a turn.
                augment = ""
            if augment:
                turn = replace(turn, augment=augment)
        except Exception as exc:  # noqa: BLE001
            typing_stop.set()
            card.update(AgentEvent("error", f"worktree setup failed: {type(exc).__name__}: {exc}", backend=backend))
            if message_id:
                try:
                    self.client.edit(chat_id, message_id, card.render_final())
                except Exception:
                    self.client.send(chat_id, card.render_final())
            return

        result = self.runtime.submit(turn, emit)
        if result in {"queued", "steered", "rejected"}:
            typing_stop.set()  # no active stream in this card — drop the indicator
            try:
                self.client.edit(chat_id, message_id or 0, card.render_live())
            except Exception:
                pass

    def _resolve_in_place(self, chat_id: int, message_id: int | None, rendered: str, reply_markup: dict[str, Any] | None) -> None:
        """Turn the live card into the final answer with NO duplicate card.

        Short answer → the card message is edited into the answer, buttons attached.
        Long answer → the card becomes chunk 1 *in place*, the overflow rides bare
        continuation messages, and the buttons sit on the LAST chunk only. So the
        data card renders exactly once per turn — it never folds and never doubles
        (this replaces the old '✅ done' stub + separately-sent answer, which read
        as two cards).
        """
        chunks = split_message(rendered) or [rendered]
        last = len(chunks) - 1
        edited = False
        if message_id:
            try:
                # markup rides the card only when chunk 0 IS the last chunk
                self.client.edit(chat_id, message_id, chunks[0], reply_markup=reply_markup if last == 0 else None)
                edited = True
            except Exception:  # noqa: BLE001 - edit can fail if the card was deleted; fall through to a fresh send.
                edited = False
        for idx in range(1 if edited else 0, len(chunks)):
            self.client.send(chat_id, chunks[idx], reply_markup=reply_markup if idx == last else None)

    def _final_text_and_markup(self, chat_id: int, mode: str, text: str, *, cwd: Path | None = None) -> tuple[str, dict[str, Any] | None]:
        clean, request = extract_post_turn_request(text)
        post_lines: list[str] = []
        if request is not None:
            if mode == "write":
                try:
                    result = PostTurnRunner(cwd=cwd or ROOT).run(request)
                    post_lines.extend(result.lines)
                except Exception as exc:  # noqa: BLE001 - post-turn failures should not hide the model answer.
                    post_lines.append(f"post-turn failed: {type(exc).__name__}: {exc}")
            else:
                post_lines.append("post-turn skipped: only write mode can mutate git")
        clean, suggestions = extract_suggestions(clean)
        if post_lines:
            clean = f"{clean}\n\nPost-turn:\n" + "\n".join(f"- {line}" for line in post_lines)
        return clean, self._suggestion_keyboard(chat_id, suggestions)

    def _suggestion_keyboard(self, chat_id: int, suggestions: list[str]) -> dict[str, Any] | None:
        rows: list[list[dict[str, str]]] = []
        for suggestion in suggestions[:3]:
            self._suggestion_serial += 1
            key = f"{chat_id}.{self._suggestion_serial}"
            self.suggestions[key] = suggestion
            while len(self.suggestions) > 60:
                self.suggestions.pop(next(iter(self.suggestions)))
            label = suggestion if len(suggestion) <= 55 else suggestion[:52].rstrip() + "..."
            rows.append([{"text": f"💡 {label}", "callback_data": f"sugg:{key}"}])
        return {"inline_keyboard": rows} if rows else None

    def _land_controls(self, chat_id: int, branch: str) -> tuple[str | None, list[dict[str, str]] | None]:
        """Tap-to-confirm merge gate, every time a merge is needed.

        Returns (note, button_row). The note is appended to the final card; the
        row, if present, is the one-tap LAND button. When auto-land is enabled and
        the merge is a provably-clean fast-forward, it lands immediately instead.
        """
        try:
            assessment = self.merge_gate.assess(branch)
        except Exception:  # noqa: BLE001 - the model answer must never be hidden by a git probe.
            return None, None
        if not assessment.needed:
            return None, None
        # SMART-MERGE: a clean branch (landable, no likely conflict) auto-lands to the
        # base and its worktree is reclaimed. land() is atomic — it aborts on any real
        # conflict — so even if the conflict guess is wrong, this stays safe.
        if self.smart_merge and assessment.landable and not assessment.likely_conflict:
            result = self.merge_gate.land(branch)
            if result.ok:
                self.worktrees.remove_branch(branch)  # landed → drop the worktree
                n = assessment.ahead
                return f"✅ Saved · {n} update{'' if n == 1 else 's'}", None
            # land surprised us (a real conflict the guess missed) → fall through to the gate
        if not assessment.landable:
            return f"⚠ Couldn't save yet — {assessment.summary()}", None
        self._land_serial += 1
        key = f"{chat_id}.{self._land_serial}"
        self._land_tokens[key] = branch
        self._discard_tokens[key] = branch
        while len(self._land_tokens) > 60:
            self._land_tokens.pop(next(iter(self._land_tokens)))
        while len(self._discard_tokens) > 60:
            self._discard_tokens.pop(next(iter(self._discard_tokens)))
        warn = " ⚠ may conflict" if assessment.likely_conflict else ""
        # Keep = merge the work → base + reclaim the worktree; Throw away = discard the
        # branch (its own throwaway branch only — safe, and recoverable from reflog).
        row = [
            {"text": f"✅ Keep ({assessment.ahead}){warn}", "callback_data": f"land:{key}"},
            {"text": "🗑 Throw away", "callback_data": f"disc:{key}"},
        ]
        return "Changes ready — keep them or throw them away:", row

    def _skills_menu(self, chat_id: int) -> None:
        skills = discover_skills()
        if not skills:
            self.client.send(chat_id, "No skills found. Set AGENT_GATEWAY_SKILLS_DIR or add a SKILL.md.")
            return
        rows: list[list[dict[str, str]]] = []
        for skill in skills[:20]:
            self._skill_serial += 1
            key = f"{chat_id}.{self._skill_serial}"
            self._skill_prompts[key] = activation_prompt(skill)
            while len(self._skill_prompts) > 80:
                self._skill_prompts.pop(next(iter(self._skill_prompts)))
            rows.append([{"text": f"✨ {skill.name}", "callback_data": f"skill:{key}"}])
        self.client.send(chat_id, "Tap a skill to run it:", reply_markup={"inline_keyboard": rows})

    def _help(self) -> str:
        agent = self.runtime.backend_for_chat(0)
        switch = "" if self.config.fixed_backend else "/agent <name> — switch backend\n"
        return (
            f"{brand_emoji()} {brand_name()} — {agent} in your pocket.\n"
            "Just talk to it: describe a task and watch it work live (this builds directly). "
            "Send more mid-task to steer; tap LAND to merge.\n\n"
            "/plan <task> — read-only: returns a plan, then tap ⚡ Build this to execute it\n"
            "/new — fresh session\n"
            "/stop — cancel the run\n"
            "/status — agent · model · effort\n"
            "/model — pick model (tap or type)\n"
            "/effort <level> — set effort (e.g. ultra)\n"
            f"{switch}"
            "\nNo /commit or /merge ceremony — it proposes scoped git, you tap to land."
        )


def build_app_from_env() -> TelegramGatewayApp:
    backends = build_backends_from_env()
    default = os.environ.get("AGENT_GATEWAY_DEFAULT_BACKEND", "mock").strip() or "mock"
    runtime = GatewayRuntime(backends, default_backend=default)
    return TelegramGatewayApp(load_config(), runtime)


def _redact_token(text: str, token: str) -> str:
    return text.replace(token, "<telegram-token>") if token else text


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _worktree_diff(path: Path) -> str:
    """Live churn in an agent's worktree, compacted: '+40 −8 3f'. '' on any error."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "diff", "--shortstat", "HEAD"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=3,
        )
    except Exception:  # noqa: BLE001 - a live stat must never break the turn.
        return ""
    s = proc.stdout
    ins = re.search(r"(\d+) insertion", s)
    dele = re.search(r"(\d+) deletion", s)
    files = re.search(r"(\d+) file", s)
    parts = []
    if ins:
        parts.append(f"+{ins.group(1)}")
    if dele:
        parts.append(f"−{dele.group(1)}")
    if files:
        parts.append(f"{files.group(1)}f")
    return " ".join(parts)


def _prepend_keyboard_row(markup: dict[str, Any] | None, row: list[dict[str, str]]) -> dict[str, Any]:
    if not markup or "inline_keyboard" not in markup:
        return {"inline_keyboard": [row]}
    return {"inline_keyboard": [row, *markup["inline_keyboard"]]}
