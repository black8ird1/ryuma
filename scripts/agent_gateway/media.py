"""Inbound media intake for the shared transport — images and voice.

Deliberately project-neutral: this layer never injects product-specific prompts.
An inbound image is saved and attached by path (the backend's model reads it with
its own file tools); a voice note is transcribed to text when GROQ_API_KEY is set,
otherwise attached as an audio file. Same behaviour for every backend, which is
the whole point of the universal layer.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

from .core import Attachment


def tmp_dir() -> Path:
    return Path(os.environ.get("AGENT_GATEWAY_TMP_DIR", "/tmp"))


@dataclass(frozen=True)
class MediaRef:
    file_id: str
    kind: str  # "image" | "voice"
    suffix: str
    caption: str = ""


def extract_media(msg: dict[str, Any]) -> list[MediaRef]:
    """Pure: pull downloadable media references from a Telegram message dict."""
    refs: list[MediaRef] = []
    caption = str(msg.get("caption") or "").strip()
    photos = msg.get("photo")
    if isinstance(photos, list) and photos:
        largest = photos[-1]  # Telegram sorts photo sizes ascending; last = largest
        if largest.get("file_id"):
            refs.append(MediaRef(str(largest["file_id"]), "image", ".jpg", caption))
    doc = msg.get("document")
    if isinstance(doc, dict) and str(doc.get("mime_type") or "").lower().startswith("image/") and doc.get("file_id"):
        name = str(doc.get("file_name") or "")
        suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ".img"
        refs.append(MediaRef(str(doc["file_id"]), "image", suffix, caption))
    voice = msg.get("voice") or msg.get("audio")
    if isinstance(voice, dict) and voice.get("file_id"):
        refs.append(MediaRef(str(voice["file_id"]), "voice", ".ogg", caption))
    return refs


def voice_enabled() -> bool:
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def save_path(chat_id: int, ref: MediaRef, now_ms: int) -> Path:
    return tmp_dir() / f"agentgw_{ref.kind}_{chat_id}_{now_ms}{ref.suffix}"


def transcribe(path: Path) -> str | None:
    """Transcribe a voice note via Groq Whisper (OpenAI-compatible). None if no key."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
    with open(path, "rb") as fh:
        resp = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            data={"model": model, "response_format": "text"},
            files={"file": (path.name, fh, "audio/ogg")},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.text.strip()


def attachment_for(ref: MediaRef, path: Path) -> Attachment:
    return Attachment(path=path, kind=ref.kind, caption=ref.caption)


def _default_timer(delay: float, fn: Callable[[], None]) -> threading.Timer:
    timer = threading.Timer(delay, fn)
    timer.daemon = True
    return timer


@dataclass
class _Album:
    chat_id: int
    user_id: int
    attachments: list[Attachment] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    timer: Any = None


class AlbumBuffer:
    """Debounce-collect Telegram album messages (same media_group_id) and flush
    them as ONE batch, so a multi-image send becomes a single turn instead of
    one queued turn per photo.

    The timer is injectable so the accumulation logic is unit-testable without
    real sleeps. on_flush(chat_id, user_id, attachments, caption) is called once
    per album, `debounce_sec` after the last photo arrives.
    """

    def __init__(
        self,
        on_flush: Callable[[int, int, tuple[Attachment, ...], str], None],
        *,
        debounce_sec: float = 2.0,
        timer_factory: Callable[[float, Callable[[], None]], Any] = _default_timer,
    ) -> None:
        self._on_flush = on_flush
        self._debounce = debounce_sec
        self._timer_factory = timer_factory
        self._groups: dict[str, _Album] = {}
        self._lock = threading.Lock()

    def add(self, group_id: str, chat_id: int, user_id: int, attachments: list[Attachment], caption: str) -> None:
        with self._lock:
            album = self._groups.get(group_id)
            if album is None:
                album = _Album(chat_id=chat_id, user_id=user_id)
                self._groups[group_id] = album
            album.attachments.extend(attachments)
            if caption:
                album.captions.append(caption)
            if album.timer is not None:
                album.timer.cancel()
            album.timer = self._timer_factory(self._debounce, lambda: self.flush(group_id))
            album.timer.start()

    def flush(self, group_id: str) -> None:
        with self._lock:
            album = self._groups.pop(group_id, None)
        if album is None:
            return
        caption = " ".join(c for c in album.captions if c).strip()
        self._on_flush(album.chat_id, album.user_id, tuple(album.attachments), caption)
