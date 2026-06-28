"""Shared on-disk live-snapshot store — the backbone of the universal Live view.

Every bot is a WRITER: on each turn it drops its live snapshot at
``<state>/live/<bot_id>/<chat_id>.json``. One standalone web server (see
``webserver.py`` + ``agent_gateway_webapp.py``) is the only READER, serving the
Mini App for *every* bot off one port behind one tunnel. That is what lets you
spawn N bots per backend with zero per-bot ports, routes, or web secrets.

Writes are atomic (temp file + ``os.replace``) so a poll never reads a half-written
JSON, and debounced per (bot, chat) so a fast token stream can't thrash the disk —
terminal snapshots (done/error) always flush so the final state is never dropped.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

# A Telegram bot token is "<numeric-id>:<secret>"; the numeric id is a stable,
# non-secret handle for the bot. We key the store by it so the public Mini App URL
# carries ?bot=<id> without ever exposing the token.
_TOKEN_ID = re.compile(r"^\s*(\d+):")
_SAFE = re.compile(r"[^0-9A-Za-z_-]")


def bot_id_from_token(token: str, *, fallback: str = "") -> str:
    """Derive a stable, non-secret bot id from a Telegram token (its numeric
    prefix). Falls back to a sanitized profile name when the token is malformed."""
    m = _TOKEN_ID.match(token or "")
    if m:
        return m.group(1)
    clean = _SAFE.sub("-", (fallback or "bot").strip()) or "bot"
    return clean


class LiveStore:
    """Disk-backed map of (bot_id, chat_id) -> latest snapshot dict."""

    def __init__(self, root: Path, *, debounce_sec: float = 0.4) -> None:
        self.root = Path(root)
        self.debounce_sec = debounce_sec
        self._last_write: dict[tuple[str, int], float] = {}
        self._lock = threading.Lock()

    def _path(self, bot_id: str, chat_id: int) -> Path:
        safe_bot = _SAFE.sub("-", str(bot_id)) or "bot"
        return self.root / safe_bot / f"{int(chat_id)}.json"

    def write(self, bot_id: str, chat_id: int, snapshot: dict, *, force: bool = False) -> None:
        """Persist a snapshot. Debounced unless ``force`` or the snapshot is terminal
        (done/error) — those must never be lost to debouncing."""
        terminal = bool(snapshot.get("done") or snapshot.get("error"))
        key = (str(bot_id), int(chat_id))
        now = time.time()
        with self._lock:
            if not (force or terminal):
                if now - self._last_write.get(key, 0.0) < self.debounce_sec:
                    return
            self._last_write[key] = now
        path = self._path(bot_id, chat_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(snapshot), encoding="utf-8")
            os.replace(tmp, path)  # atomic — readers never see a partial write
        except OSError:
            pass  # the live view is best-effort; a write failure must never break a turn

    def read(self, bot_id: str, chat_id: int) -> dict:
        try:
            return json.loads(self._path(bot_id, chat_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
