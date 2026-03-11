#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
APP_LOG="artifacts/app_final.log"
RUN_LOG="artifacts/promptfoo_redteam.log"
APP_PID_FILE="artifacts/app_final.pid"

mkdir -p artifacts

cleanup() {
  if [[ -f "$APP_PID_FILE" ]]; then
    PID="$(cat "$APP_PID_FILE")"
    if kill -0 "$PID" >/dev/null 2>&1; then
      kill "$PID" >/dev/null 2>&1 || true
    fi
    rm -f "$APP_PID_FILE"
  fi
}

trap cleanup EXIT

python app.py > "$APP_LOG" 2>&1 &
echo $! > "$APP_PID_FILE"

for _ in $(seq 1 30); do
  if curl -s http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt":"hello"}' >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Application is up. Starting Promptfoo..."
./run_promptfoo.sh run | tee "$RUN_LOG"

echo "Saved Promptfoo run log to $RUN_LOG"
echo "Saved Flask app log to $APP_LOG"

if [[ "$MODE" == "both" ]]; then
  echo "Opening Promptfoo report..."
  ./run_promptfoo.sh report
else
  echo "Run './run_promptfoo.sh report' when you are ready to open the report."
fi