# Ryuma

Experimental shared Telegram cockpit foundation. It does not replace the current
Codex bot or Claude bot.

The simple operator model is:

> one Telegram bot token + one small profile file + one fixed backend

So three parallel bots are just three profiles and three running processes:

- `codex.env` -> Codex bot
- `claude.env` -> Claude bot
- `glm.env` -> future GLM/OpenCode/etc bot

## What It Extracts

- Reply-context injection with status-frame filtering.
- One live progress card for thinking, tools, injected messages, errors, usage.
- Steering for backends that support mid-run injection.
- Queue fallback for one-shot backends.
- Shared `/agent`, `/mode`, `/new`, `/stop`, `/status`, `/interface`.
- Universal continuation buttons from a final `SUGGEST: a | b | c` marker.
- Hidden post-turn automation from a final `POSTTURN: {...}` marker, so there is
  no extra `/commit` or `/merge` command surface.
- Automatic write-turn worktree brokerage, so backends can run in isolated
  `agent/<backend>/<chat-task>` branches instead of sharing a dirty checkout.
- Backend adapters for `mock`, `codex`, and `claude`.

## Local Smoke

```bash
python3 scripts/agent_gateway_bot.py --interface
python3 scripts/agent_gateway_bot.py --demo "build the universal bot foundation"
```

## Simple Multi-Bot Activation

Create one shared user file plus one profile file for each BotFather token:

```bash
python3 scripts/agent_gateway_bot.py --init-profile codex
nano state/agent-gateway/profiles/users.env
nano state/agent-gateway/profiles/codex.env
```

The shared allowlist is defined once:

```env
ALLOWED_USER_IDS=123456789
```

Each bot profile stays token + fixed agent:

```env
TELEGRAM_TOKEN=123:token-from-botfather
AGENT=codex

AUTO_WORKTREE=1
AUTO_COMMIT=1
ALLOW_PUSH=0
ALLOW_MERGE=0
FIXED_BACKEND=1
```

Run that one bot:

```bash
python3 scripts/agent_gateway_bot.py --profile codex
```

On the VPS, start the normal Codex + Claude gateway pair and send private-chat
test messages to the shared `ALLOWED_USER_IDS` with:

```bash
scripts/agent_gateway_start.sh codex claude
scripts/agent_gateway_start.sh --status
```

On a systemd host, the starter runs each profile as an independent transient
unit (`agent-gateway-<profile>.service`; the prefix is set by
`AGENT_GATEWAY_UNIT_PREFIX`) instead of a child of the launching process. That
keeps gateway bots alive across launcher restarts.
The runtime process sends its own startup proof message, so a delivered "online"
message verifies the actual long-running bot has token, network, and allowlist
access. Non-systemd environments fall back to `nohup` for local smoke only.

Run three parallel bots by opening three services/processes, each with a different
profile:

```bash
python3 scripts/agent_gateway_bot.py --profile codex
python3 scripts/agent_gateway_bot.py --profile claude
python3 scripts/agent_gateway_bot.py --profile glm
```

Profile aliases are accepted so the files stay readable:

```env
TELEGRAM_TOKEN=...
AGENT=claude
CODEX_MAX_SESSION_TURNS=8
CLAUDE_THINKING_TOKENS=4000
```

Current backend behavior:

- `mock` is deterministic and safe for UX testing.
- `codex` uses `codex exec` with bounded persistent write sessions and captures
  the output file from `--output-last-message`.
- `claude` uses a warm worker per chat, sends turns through stream-json stdin,
  and supports injected follow-up messages while a turn is active.

Context budget guardrails:

- Gateway prompts point agents at canonical files and tell them to read only
  specific files on demand, not bulk-load docs or archives.
- Codex and Claude gateway sessions are both bounded by max-session-turn env
  vars, defaulting to 8 turns. `/new` also resets a chat session manually.
- The shared brain status is compact by design; detailed history stays in files
  and JSONL logs instead of being pasted into every model turn.

The next production-grade follow-up is to activate the shared gateway behind a
separate Telegram token and run a live mock/Codex/Claude shakedown before any
existing bot path is replaced.

## Advanced Raw Env Activation

Profiles are just a small wrapper around the underlying env vars. This is still
supported for systemd or debugging:

```bash
export AGENT_GATEWAY_TELEGRAM_TOKEN="123:token"
export AGENT_GATEWAY_ALLOWED_USER_IDS="123456789"
export AGENT_GATEWAY_DEFAULT_BACKEND="mock"
export AGENT_GATEWAY_FIXED_BACKEND="1"
python3 scripts/agent_gateway_bot.py
```

Worktree broker:

- Write-mode turns create or reuse one worktree per chat/backend task.
- Default branch shape: `agent/<backend>/<chat>-<task-hash>`.
- Default root: a worktrees dir under your home; override with
  `AGENT_GATEWAY_WORKTREE_ROOT`.
- Disable with `AGENT_GATEWAY_AUTO_WORKTREE=0`.
- `/new` clears the chat assignment so the next write request starts a fresh
  task worktree.

End-of-turn automation:

- Backends can end a final answer with
  `POSTTURN: {"commit":{"message":"...","paths":["file"]},"push":false,"merge":false}`.
- The gateway strips the marker before Telegram sees it, commits only the listed
  pathspecs, and leaves unrelated dirty files untouched.
- `AGENT_GATEWAY_AUTO_COMMIT=1` is the default. Push and direct merge are off by
  default; enable only with `AGENT_GATEWAY_ALLOW_PUSH=1` or
  `AGENT_GATEWAY_ALLOW_MERGE=1`.
- When direct merge is disabled, `merge:true` writes a pending request under
  `state/telegram-agent/merge-requests/` for a later checkpoint.
- Checkpoints are handled through `python3 scripts/agent-brain.py merge ...`:
  `list`, `show`, `approve` and `reject`.
- `merge approve` performs a local clean-checkout, ff-only merge. It does not
  push, deploy or mutate production.

Continuation buttons:

- Backends can end a final answer with `SUGGEST: run tests | commit it | queue merge`.
- The gateway strips that line and renders up to three tap-to-send buttons.
- Tapping a suggestion submits that text as the next normal prompt.
