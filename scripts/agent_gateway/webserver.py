"""Optional in-process web server for the Ryuma live Mini App.

Serves the glass-cockpit web view + a token-gated /state JSON of the live run,
read by the Telegram Mini App. Runs in a daemon thread and is entirely optional —
the bot is unaffected if it never starts. HTML and JSON come from one origin (no
CORS); /state is token-gated because the server is exposed via a public tunnel.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

WEBAPP_HTML = Path(__file__).resolve().parent / "webapp" / "index.html"


def start_webapp_server(
    *,
    port: int,
    token: str,
    snapshot_for: Callable[[int], dict] | None = None,
    store_read: Callable[[str, int], dict] | None = None,
    host: str = "127.0.0.1",
) -> ThreadingHTTPServer:
    """Serve the Mini App. Two modes:

    * ``snapshot_for(chat)`` — single in-process bot (legacy).
    * ``store_read(bot, chat)`` — the universal standalone server: one origin reads
      the disk store for ANY bot, so N bots share one port + one tunnel. The Mini
      App URL carries ``?bot=<id>&chat=<id>``; ``bot`` defaults to empty for the
      legacy single-bot path.
    """
    html = WEBAPP_HTML.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: A003 - silence default stderr logging
            return

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/app", "/index.html"):
                self._send(200, html, "text/html; charset=utf-8")
                return
            if u.path == "/state":
                q = parse_qs(u.query)
                if token and q.get("key", [""])[0] != token:
                    self._send(403, b'{"error":"forbidden"}', "application/json")
                    return
                try:
                    cid = int(q.get("chat", [""])[0])
                except (ValueError, TypeError, IndexError):
                    cid = 0
                bot = q.get("bot", [""])[0]
                if store_read is not None:
                    snap = store_read(bot, cid) or {}
                elif snapshot_for is not None:
                    snap = snapshot_for(cid) or {}
                else:
                    snap = {}
                self._send(200, json.dumps(snap).encode("utf-8"), "application/json")
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
