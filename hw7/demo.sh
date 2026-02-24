#!/usr/bin/env bash
set -euo pipefail

# HW7 demo runner (rate-limit friendly)

SLEEP_BETWEEN="${DEMO_SLEEP_SECONDS:-15}"
RETRIES="${DEMO_RETRIES:-3}"

if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "ERROR: Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running."
  exit 2
fi

echo "[1/3] Deterministic scan"
uv run python hw7/app.py scan --target ssh --out hw7/out/report.md --json >/dev/null
echo "  Wrote: hw7/out/report.md"

echo "[2/3] Agent question: identify ssh server"
uv run python hw7/app.py agent "What is the ssh server being run on this machine?" >/dev/null
echo "  Wrote: hw7/out/agent_answer.md"
echo "  Wrote: hw7/out/agent_trace.json"

echo "  Sleeping ${SLEEP_BETWEEN}s to avoid API rate limits..."
sleep "${SLEEP_BETWEEN}"

echo "[3/3] Agent question: vulnerabilities"
for i in $(seq 1 "${RETRIES}"); do
  if uv run python hw7/app.py agent "Does the current ssh server being run have any vulnerabilities associated with it?" >/dev/null; then
    echo "  Updated: hw7/out/agent_answer.md"
    echo "  Updated: hw7/out/agent_trace.json"
    echo ""
    echo "Done."
    echo "Artifacts:"
    echo "  - hw7/out/report.md"
    echo "  - hw7/out/agent_answer.md"
    echo "  - hw7/out/agent_trace.json"
    echo "  - hw7/out/mcp_server_stderr.log"
    exit 0
  fi

  echo "  Attempt ${i}/${RETRIES} failed (possible 429). Sleeping 20s and retrying..."
  sleep 20
done

echo "ERROR: demo failed after ${RETRIES} retries (Gemini API rate limit / quota)."
exit 3
