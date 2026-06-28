#!/usr/bin/env python3
"""Standalone Live-view server — the single Mini App frontend for ALL gateway bots.

Every bot writes its live snapshot to the shared on-disk store; this one process
reads that store and serves the Mini App. Run it once (one port, behind the one
tunnel you already have); spawn as many bots as you like and they all show up here
keyed by ?bot=<id>&chat=<id>. No per-bot ports, routes, or web secrets.

    AGENT_GATEWAY_WEBAPP_PORT=8847 AGENT_GATEWAY_WEBAPP_TOKEN=… \
        python3 scripts/agent_gateway_webapp.py

Reads users.env for the shared token if --profile-dir is given.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from agent_gateway.livestore import LiveStore
from agent_gateway.profiles import DEFAULT_PROFILE_DIR, SHARED_PROFILE_NAME, parse_profile_file
from agent_gateway.webserver import start_webapp_server

ROOT = Path(__file__).resolve().parents[1]
LIVE_STORE_DIR = ROOT / "state" / "agent-gateway" / "live"


def _shared_token() -> str:
    """Pull AGENT_GATEWAY_WEBAPP_TOKEN from env, else from profiles/users.env so the
    server and the bots agree on one gate without duplicating the secret."""
    token = os.environ.get("AGENT_GATEWAY_WEBAPP_TOKEN", "").strip()
    if token:
        return token
    shared = DEFAULT_PROFILE_DIR / SHARED_PROFILE_NAME
    if shared.exists():
        try:
            return parse_profile_file(shared).get("AGENT_GATEWAY_WEBAPP_TOKEN", "").strip()
        except Exception:  # noqa: BLE001 - missing/garbled token just means an open dev server.
            pass
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone Live-view server for all gateway bots")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENT_GATEWAY_WEBAPP_PORT", "8847")))
    parser.add_argument("--host", default=os.environ.get("AGENT_GATEWAY_WEBAPP_HOST", "127.0.0.1"))
    args = parser.parse_args(argv)

    store = LiveStore(LIVE_STORE_DIR)
    token = _shared_token()
    start_webapp_server(port=args.port, token=token, store_read=store.read, host=args.host)
    print(
        f"agent-gateway live-view server on {args.host}:{args.port} "
        f"(store={LIVE_STORE_DIR}, gated={'yes' if token else 'no'})",
        flush=True,
    )
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    raise SystemExit(main())
