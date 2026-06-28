<p align="center">
  <img src="assets/logo.png" alt="Ryuma" width="170">
</p>

<h1 align="center">Ryuma</h1>

<p align="center">
  Run any coding agent — Claude Code, Codex, and more — straight from a Telegram chat.<br>
  Live progress, an isolated git worktree per task, auto-merge of clean work, voice + image input.
</p>

---

## Why

Your coding agent lives on a server. You live on your phone. Ryuma is the bridge:
message a Telegram bot, watch the agent think and edit in real time, and get clean
work merged back automatically — no SSH, no laptop required.

## Features

- 🤖 **Any agent** — Claude Code, Codex, or the built-in `mock`. One gateway, swap the backend.
- 📲 **Telegram-native** — chat, voice notes, and images in; a live streaming thought feed out.
- 🌿 **Isolated worktrees** — every write turn runs in its own git worktree and auto-merges when clean. You're only asked on a real conflict.
- 🔒 **Allowlisted** — only the Telegram ids you authorize can drive it.
- 🧩 **Hooks** — inject your own project context per turn (NoOp by default).

## Install

```bash
./install.sh
```

That checks prerequisites and launches the setup wizard, which:

1. walks you through creating a bot in @BotFather (paste the token),
2. learns your Telegram id automatically — you just message your bot once,
3. writes the config and tells you how to start.

## Start

```bash
scripts/agent_gateway_start.sh <agent>     # e.g. claude or codex
```

Then open your bot in Telegram and talk to it.

## Agents

You only need **one** agent set up to start. The bot runs every backend, so you
can add the other later **from your phone** — no re-running setup:

- Send **`/agents`** in the chat to see which agents are installed, get the exact
  install + sign-in commands for any that aren't, and tap to switch.

Setting up an agent CLI (once, on the host):

| Agent | Install | Sign in |
|-------|---------|---------|
| Claude Code | `npm install -g @anthropic-ai/claude-code` | `claude` (or `claude setup-token` for headless) |
| Codex | `npm install -g @openai/codex` | `codex login` |

## Requirements

- Python 3.10+
- git
- At least one agent CLI on PATH: `claude` or `codex` (or use the built-in `mock`).
  Don't have one yet? The wizard sets up with `mock`; add a real agent later with `/agents`.
- Optional: `GROQ_API_KEY` for voice transcription

## How it works

Each write turn runs in an isolated git worktree and auto-merges back when clean
(smart-merge). You only get asked when there's a real conflict. Point it at your
repo with `AGENT_GATEWAY_REPO_ROOT`, or run it from inside the repo.

## License

[Apache-2.0](LICENSE).
