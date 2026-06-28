#!/usr/bin/env python3
"""Standalone launcher for the experimental shared Telegram agent gateway.

This does not replace or modify the existing Codex/Claude bots. Use `--demo`
first to verify the runtime without a Telegram token:

    python3 scripts/agent_gateway_bot.py --demo "hello"

For the simple multi-bot operator path, create one profile per Telegram bot:

    python3 scripts/agent_gateway_bot.py --init-profile codex
    $EDITOR state/agent-gateway/profiles/codex.env
    python3 scripts/agent_gateway_bot.py --profile codex
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from agent_gateway.backends import build_backends_from_env
from agent_gateway.core import AgentEvent, AgentTurn, GatewayRuntime, LiveCard, build_interface_report
from agent_gateway.profiles import apply_profile, write_profile_template
from agent_gateway.telegram import build_app_from_env


def run_demo(text: str, *, backend: str, mode: str) -> int:
    backends = build_backends_from_env()
    if backend not in backends:
        print(f"unknown backend {backend!r}; available: {', '.join(sorted(backends))}", file=sys.stderr)
        return 2
    runtime = GatewayRuntime(backends, default_backend=backend)
    card = LiveCard(backend=backend, mode=mode)  # type: ignore[arg-type]

    def emit(event: AgentEvent) -> None:
        card.update(event)
        print(card.render_live() if event.kind != "final" else card.render_final())
        print("---")

    turn = AgentTurn(chat_id=1, user_id=1, text=text, backend=backend, mode=mode)  # type: ignore[arg-type]
    runtime.submit(turn, emit)
    runtime.wait_for_idle(1, timeout=30)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ryuma — multi-agent Telegram gateway")
    parser.add_argument("--profile", help="Load state/agent-gateway/profiles/<name>.env or an explicit .env path")
    parser.add_argument("--init-profile", metavar="AGENT", help="Create a simple editable profile for that fixed agent")
    parser.add_argument("--demo", metavar="TEXT", help="Run one local demo turn without Telegram polling")
    parser.add_argument("--backend", help="Demo backend; defaults to the profile agent or mock")
    parser.add_argument("--mode", default="write", choices=["ask", "plan", "write"], help="Demo mode")
    parser.add_argument("--interface", action="store_true", help="Print the shared interface design")
    parser.add_argument(
        "--onboard",
        "--setup",
        dest="onboard",
        action="store_true",
        help="Interactive first-run wizard: paste a token, message the bot, done",
    )
    args = parser.parse_args(argv)

    if args.onboard:
        from agent_gateway.onboard import run as run_onboarding

        result = run_onboarding()
        return 0 if result else 1

    if args.init_profile:
        # One command to spawn another bot: name + which backend. Paste the token,
        # start it — Live view works automatically (shared store + standalone server).
        path = write_profile_template(args.init_profile, backend=args.backend)
        backend = (args.backend or args.init_profile)
        print(f"created {path}  (agent={backend})")
        print(f"users.env already holds ALLOWED_USER_IDS + the shared Live-view keys.")
        print(f"1. paste this bot's token into {path}  (TELEGRAM_TOKEN=…)")
        print(f"2. start it:  scripts/agent_gateway_start.sh {args.init_profile}")
        return 0

    if args.profile:
        loaded = apply_profile(args.profile)
        print(f"loaded profile {loaded.name}: {loaded.path}", flush=True)

    if args.interface:
        print(build_interface_report())
        return 0
    if args.demo is not None:
        backend = args.backend or os.environ.get("AGENT_GATEWAY_DEFAULT_BACKEND", "mock")
        return run_demo(args.demo, backend=backend, mode=args.mode)

    app = build_app_from_env()
    app.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
