# Project template

Copy this folder to point the gateway at your own project. Three optional pieces,
each a seam the engine knows nothing about:

| File | What it does | Required? |
|------|--------------|-----------|
| `profile.env.example` | Brand, repo, branch, merge, and hook settings | no — all have defaults |
| `system-prompt.md` | Your project's standing instructions (read every turn) | recommended |
| `hook.py` | Inject live state + record turns (your "brain") | no — NoOp if unset |

## Use it

```bash
cp -r scripts/agent_gateway/examples/template scripts/agent_gateway/examples/myproject
# 1. fill in system-prompt.md with your project's facts
# 2. (optional) edit hook.py to inject your live state / record turns
# 3. set the env from myproject/profile.env in your bot profile
```

The agent then starts every turn with your system prompt (and, if you enable the
hook, your injected state), and you've leaked none of it into the public engine.
