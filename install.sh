#!/usr/bin/env bash
# One-command install for Ryuma.
set -euo pipefail
cd "$(dirname "$0")"

echo "🐉 Ryuma — install"
command -v git >/dev/null  || { echo "✗ git is required"; exit 1; }
PY=$(command -v python3 || true)
[ -n "$PY" ] || { echo "✗ python3 (3.10+) is required"; exit 1; }
"$PY" - <<'EOF'
import sys
assert sys.version_info >= (3, 10), "Python 3.10+ required, found %s" % sys.version.split()[0]
EOF

chmod +x scripts/agent_gateway_start.sh 2>/dev/null || true
echo "✓ Prerequisites OK — launching setup..."
exec "$PY" scripts/agent_gateway_bot.py --onboard
