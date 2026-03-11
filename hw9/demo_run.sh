#!/usr/bin/env bash
set -euo pipefail

APP_LOG="artifacts/app.log"
APP_PID="artifacts/app.pid"

mkdir -p artifacts

cleanup() {
  if [[ -f "$APP_PID" ]]; then
    PID="$(cat "$APP_PID")"
    if kill -0 "$PID" >/dev/null 2>&1; then
      kill "$PID" >/dev/null 2>&1 || true
    fi
    rm -f "$APP_PID"
  fi
}

trap cleanup EXIT

python app.py > "$APP_LOG" 2>&1 &
echo $! > "$APP_PID"

for _ in $(seq 1 30); do
  if curl -s http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt":"hello"}' >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

python demo_probe.py

echo "Saved app log to $APP_LOG"