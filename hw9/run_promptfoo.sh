#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-both}"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js 20+ is required for Promptfoo."
  exit 1
fi

if [[ -x "./node_modules/.bin/promptfoo" ]]; then
  PROMPTFOO="./node_modules/.bin/promptfoo"
else
  PROMPTFOO="npx promptfoo@latest"
fi

case "$MODE" in
  run)
    eval "$PROMPTFOO redteam run"
    ;;
  report)
    eval "$PROMPTFOO redteam report"
    ;;
  both)
    eval "$PROMPTFOO redteam run"
    eval "$PROMPTFOO redteam report"
    ;;
  *)
    echo "Usage: ./run_promptfoo.sh [run|report|both]"
    exit 1
    ;;
esac