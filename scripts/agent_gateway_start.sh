#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
UNIT_PREFIX="${AGENT_GATEWAY_UNIT_PREFIX:-ryuma}"
SYSTEMD_MODE="${AGENT_GATEWAY_SYSTEMD:-auto}"

mkdir -p scripts/logs state/agent-gateway/runtime

unit_name() {
  printf '%s-%s' "$UNIT_PREFIX" "$1"
}

unit_service() {
  printf '%s.service' "$(unit_name "$1")"
}

use_systemd() {
  if [ "$SYSTEMD_MODE" = "0" ] || [ "$SYSTEMD_MODE" = "false" ]; then
    return 1
  fi
  command -v systemd-run >/dev/null 2>&1 && [ -d /run/systemd/system ]
}

status_profile() {
  profile="$1"
  runtime_dir="state/agent-gateway/runtime/${profile}"
  pidfile="${runtime_dir}/pid"
  unit="$(unit_service "$profile")"
  logfile="scripts/logs/agent-gateway-${profile}.log"

  if use_systemd && systemctl show "$unit" >/dev/null 2>&1; then
    active="$(systemctl is-active "$unit" 2>/dev/null || true)"
    pid="$(systemctl show "$unit" -p MainPID --value 2>/dev/null || true)"
    printf '%s: systemd %s unit %s pid %s -> %s\n' "$profile" "${active:-unknown}" "$unit" "${pid:-unknown}" "$logfile"
    return 0
  fi

  if [ -s "$pidfile" ]; then
    oldpid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
      printf '%s: running pid %s -> %s\n' "$profile" "$oldpid" "$logfile"
      return 0
    fi
  fi
  printf '%s: not running -> %s\n' "$profile" "$logfile"
}

start_with_systemd() {
  profile="$1"
  runtime_dir="state/agent-gateway/runtime/${profile}"
  pidfile="${runtime_dir}/pid"
  logfile="scripts/logs/agent-gateway-${profile}.log"
  mkdir -p "$runtime_dir"
  unit="$(unit_service "$profile")"
  unit_base="$(unit_name "$profile")"

  systemctl stop "$unit" >/dev/null 2>&1 || true
  systemctl reset-failed "$unit" >/dev/null 2>&1 || true
  rm -f "$pidfile"

  systemd-run \
    --unit="$unit_base" \
    --description="Ryuma ${profile}" \
    --working-directory="$ROOT" \
    --setenv=PYTHONUNBUFFERED=1 \
    --setenv=AGENT_GATEWAY_PROFILE="$profile" \
    --setenv=AGENT_GATEWAY_SEND_ONLINE_MESSAGE=1 \
    --property=Restart=always \
    --property=RestartSec=3 \
    --property=StandardOutput="append:${ROOT}/${logfile}" \
    --property=StandardError="append:${ROOT}/${logfile}" \
    "$PYTHON_BIN" "$ROOT/scripts/agent_gateway_bot.py" --profile "$profile" >/dev/null

  for _ in 1 2 3 4 5; do
    active="$(systemctl is-active "$unit" 2>/dev/null || true)"
    [ "$active" = "active" ] && break
    sleep 0.5
  done
  pid="$(systemctl show "$unit" -p MainPID --value 2>/dev/null || true)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" != "0" ]; then
    echo "$pid" > "$pidfile"
  fi
  printf '%s: systemd %s unit %s pid %s -> %s\n' "$profile" "${active:-unknown}" "$unit" "${pid:-unknown}" "$logfile"
  if [ "${active:-}" != "active" ]; then
    systemctl status "$unit" --no-pager -l 2>/dev/null | tail -n 20 || true
    return 1
  fi
}

start_with_nohup() {
  profile="$1"
  runtime_dir="state/agent-gateway/runtime/${profile}"
  pidfile="${runtime_dir}/pid"
  logfile="scripts/logs/agent-gateway-${profile}.log"
  mkdir -p "$runtime_dir"
  if [ -s "$pidfile" ]; then
    oldpid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
      kill "$oldpid" 2>/dev/null || true
      sleep 1
    fi
  fi

  AGENT_GATEWAY_PROFILE="$profile" AGENT_GATEWAY_SEND_ONLINE_MESSAGE=1 \
    nohup "$PYTHON_BIN" scripts/agent_gateway_bot.py --profile "$profile" >> "$logfile" 2>&1 &
  echo "$!" > "$pidfile"
  printf '%s: nohup pid %s -> %s\n' "$profile" "$(cat "$pidfile")" "$logfile"
  printf '%s: warning: nohup fallback is tied to the caller process tree; use systemd on the VPS.\n' "$profile"
}

# The shared Live-view server: ONE process, ONE port, behind the ONE tunnel —
# serves the Mini App for every bot off the shared on-disk store. Started as its
# own unit ("webapp") so spawning more bots never needs another port or route.
start_webapp_unit() {
  logfile="scripts/logs/agent-gateway-webapp.log"
  unit="$(unit_service webapp)"
  unit_base="$(unit_name webapp)"
  if ! use_systemd; then
    AGENT_GATEWAY_PROFILE=webapp nohup "$PYTHON_BIN" scripts/agent_gateway_webapp.py >> "$logfile" 2>&1 &
    printf 'webapp: nohup pid %s -> %s\n' "$!" "$logfile"
    return 0
  fi
  systemctl stop "$unit" >/dev/null 2>&1 || true
  systemctl reset-failed "$unit" >/dev/null 2>&1 || true
  systemd-run \
    --unit="$unit_base" \
    --description="Ryuma live-view server" \
    --working-directory="$ROOT" \
    --setenv=PYTHONUNBUFFERED=1 \
    --property=Restart=always \
    --property=RestartSec=3 \
    --property=StandardOutput="append:${ROOT}/${logfile}" \
    --property=StandardError="append:${ROOT}/${logfile}" \
    "$PYTHON_BIN" "$ROOT/scripts/agent_gateway_webapp.py" >/dev/null
  active="$(systemctl is-active "$unit" 2>/dev/null || true)"
  printf 'webapp: systemd %s unit %s -> %s\n' "${active:-unknown}" "$unit" "$logfile"
}

start_profile() {
  profile="$1"
  if [ "$profile" = "webapp" ]; then
    start_webapp_unit
    return $?
  fi
  if use_systemd; then
    start_with_systemd "$profile"
  else
    start_with_nohup "$profile"
  fi
}

if [ "${1:-}" = "--status" ] || [ "${1:-}" = "status" ]; then
  shift
  if [ "$#" -eq 0 ]; then
    set -- codex claude mock
  fi
  for profile in "$@"; do
    status_profile "$profile"
  done
  exit 0
fi

if [ "$#" -eq 0 ]; then
  set -- codex claude
fi

for profile in "$@"; do
  start_profile "$profile"
done
