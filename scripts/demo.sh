#!/usr/bin/env bash
# Live demo (≈ 2 min): a request -> an alert -> a rollback -> back to normal.
# Isolated incident registry (mlflow_incident.db): the serving champion is the version
# retrained on the corrupted weather backfill (see docs/incident.md).
#
#   bash scripts/demo.sh --prepare   # before the presentation (≈ 1 min): puts the registry back in the "incident" state
#   bash scripts/demo.sh             # the demo, pausing between steps (Enter)
#   bash scripts/demo.sh --no-pause  # in one go (rehearsal, test)
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-8000}
API="http://127.0.0.1:${PORT}"
export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow_incident.db"
export PREDICTIONS_LOG="$(pwd)/data/monitoring/demo/predictions.parquet"
export PGA_MONITORING_DIR="$(pwd)/data/monitoring/demo"   # outputs outside reports/ (not versioned)
export PGA_TRIGGER_DIR="$(pwd)/data/monitoring/demo"
PAUSE=1
[[ "${1:-}" == "--no-pause" ]] && PAUSE=0

if [[ "${1:-}" == "--prepare" ]]; then
  uv run python scripts/incident.py setup
  uv run python scripts/incident.py corrupt
  echo "ready: the champion is the version retrained on the corrupted backfill"
  exit 0
fi

step() {
  echo
  echo "────────────────────────────────────────────────────────────────"
  echo "▶ $1"
  echo "────────────────────────────────────────────────────────────────"
  if [[ $PAUSE == 1 ]]; then read -r -p "  (Enter) " _; fi
}
quiet() { "$@" 2>&1 | grep -vE "INFO|WARNING|warnings.warn|^\s*$" || true; }

rm -f "$PREDICTIONS_LOG"
uv run uvicorn pga.api:app --port "$PORT" --log-level warning > /tmp/pga_demo_api.log 2>&1 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do curl -s "$API/health" > /dev/null && break; sleep 0.5; done

step "1. The API serves the current champion"
curl -s "$API/health"; echo

step "2. Production: the March-April 2024 tournaments sent to /predict"
quiet uv run python scripts/send_requests.py --from 2024-03-01 --to 2024-04-21 --api-url "$API"

step "3. Monitoring on the API prediction journal (actual scores known)"
quiet uv run python -m pga.monitor check || true

step "4. The trigger decides and acts (rollback + API reload)"
quiet uv run python -m pga.trigger --as-of 2024-04-22 --api-url "$API" || true

step "5. The API serves the healthy version"
curl -s "$API/health"; echo

step "6. Same requests, then monitoring: back within thresholds"
quiet uv run python scripts/send_requests.py --from 2024-03-01 --to 2024-04-21 --api-url "$API"
quiet uv run python -m pga.monitor check || true

step "7. The same backfill today: promotion refused"
quiet uv run python scripts/incident.py retry

echo
echo "end of demo — 'bash scripts/demo.sh --prepare' to replay it"
