#!/usr/bin/env bash
# Démo en direct (≈ 2 min) : une requête -> une alerte -> un rollback -> retour à la normale.
# Registry isolé de l'incident (mlflow_incident.db) : le champion en service est la version
# réentraînée sur le backfill météo corrompu (voir docs/incident.md).
#
#   bash scripts/demo.sh --prepare   # avant l'oral (≈ 1 min) : remet le registry dans l'état « incident »
#   bash scripts/demo.sh             # la démo, avec une pause entre chaque étape (Entrée)
#   bash scripts/demo.sh --no-pause  # d'une traite (répétition, test)
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-8000}
API="http://127.0.0.1:${PORT}"
export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow_incident.db"
export PREDICTIONS_LOG="$(pwd)/data/monitoring/demo/predictions.parquet"
export PGA_MONITORING_DIR="$(pwd)/data/monitoring/demo"   # sorties hors de reports/ (non versionné)
export PGA_TRIGGER_DIR="$(pwd)/data/monitoring/demo"
PAUSE=1
[[ "${1:-}" == "--no-pause" ]] && PAUSE=0

if [[ "${1:-}" == "--prepare" ]]; then
  uv run python scripts/incident.py setup
  uv run python scripts/incident.py corrupt
  echo "prêt : le champion est la version réentraînée sur le backfill corrompu"
  exit 0
fi

step() {
  echo
  echo "────────────────────────────────────────────────────────────────"
  echo "▶ $1"
  echo "────────────────────────────────────────────────────────────────"
  if [[ $PAUSE == 1 ]]; then read -r -p "  (Entrée) " _; fi
}
quiet() { "$@" 2>&1 | grep -vE "INFO|WARNING|warnings.warn|^\s*$" || true; }

rm -f "$PREDICTIONS_LOG"
uv run uvicorn pga.api:app --port "$PORT" --log-level warning > /tmp/pga_demo_api.log 2>&1 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do curl -s "$API/health" > /dev/null && break; sleep 0.5; done

step "1. L'API sert le champion actuel"
curl -s "$API/health"; echo

step "2. Production : les tournois de mars-avril 2024 envoyés à /predict"
quiet uv run python scripts/send_requests.py --from 2024-03-01 --to 2024-04-21 --api-url "$API"

step "3. Monitoring sur le journal de l'API (scores réels connus)"
quiet uv run python -m pga.monitor check || true

step "4. Le déclencheur décide et agit (rollback + rechargement de l'API)"
quiet uv run python -m pga.trigger --as-of 2024-04-22 --api-url "$API" || true

step "5. L'API sert la version saine"
curl -s "$API/health"; echo

step "6. Mêmes requêtes, puis monitoring : retour dans les seuils"
quiet uv run python scripts/send_requests.py --from 2024-03-01 --to 2024-04-21 --api-url "$API"
quiet uv run python -m pga.monitor check || true

echo
echo "fin de la démo — 'bash scripts/demo.sh --prepare' pour la rejouer"
