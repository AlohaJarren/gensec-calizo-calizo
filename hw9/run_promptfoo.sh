#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-both}"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js 20+ is required for Promptfoo."
  exit 1
fi

case "$MODE" in
  run)
    npx promptfoo@latest redteam run
    ;;
  report)
    npx promptfoo@latest redteam report
    ;;
  both)
    npx promptfoo@latest redteam run
    npx promptfoo@latest redteam report
    ;;
  *)
    echo "Usage: ./run_promptfoo.sh [run|report|both]"
    exit 1
    ;;
esac