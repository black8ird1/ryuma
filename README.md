# 🐉 Ryuma

Run any coding agent (Claude Code, Codex, …) from a Telegram chat. Live progress,
isolated git worktrees per task, auto-merge of clean work, voice + image input.

## Install

```bash
./install.sh
```

That checks prerequisites and launches the setup wizard. The wizard:

1. walks you through creating a bot in @BotFather (paste the token),
2. learns your Telegram id automatically — you just message your bot once,
3. writes the config, and tells you how to start.

## Start

```bash
scripts/agent_gateway_start.sh <agent>     # e.g. claude or codex
```

Then open your bot in Telegram and talk to it.

## Requirements

- Python 3.10+
- git
- At least one agent CLI on PATH: `claude` or `codex` (or use the built-in `mock`)
- Optional: `GROQ_API_KEY` for voice transcription

## How it works

Each write turn runs in an isolated git worktree and auto-merges back when clean
(smart-merge). You only get asked when there's a real conflict. Point it at your
repo with `AGENT_GATEWAY_REPO_ROOT`, or run it from inside the repo.

## License

[Apache-2.0](LICENSE).
