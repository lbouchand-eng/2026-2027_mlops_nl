#!/usr/bin/env bash
# Live demo on the visible Compose stack: same steps as demo.sh, but against the API (:8000)
# and registry served by Compose. Each step says what to look at: Grafana, MLflow, Swagger.
# Commands run inside the `api` container: only Docker is needed on the host.
#
#   docker compose --profile observability up -d --build   # API, MLflow, Prometheus, Pushgateway, Grafana
#   bash scripts/demo_live.sh --prepare    # before the presentation (≈ 2 min): stack registry in the "incident" state
#   bash scripts/demo_live.sh              # the demo, pausing between steps (Enter)
#   bash scripts/demo_live.sh --no-pause   # in one go (rehearsal, test)
#
# --prepare **replaces** the stack registry (healthy champion v1, then v2 retrained on the
# corrupted backfill and promoted) and clears the prediction journal. To get the standard
# registry back: docker compose down -v. Do not run the `scheduler` service at the same time.
set -euo pipefail
cd "$(dirname "$0")/.."

API="http://127.0.0.1:${API_PORT:-8000}"
MLFLOW="http://127.0.0.1:${MLFLOW_PORT:-5001}"
GRAFANA="http://127.0.0.1:${GRAFANA_PORT:-3000}"
PUSHGATEWAY="http://127.0.0.1:${PUSHGATEWAY_PORT:-9091}"
DASHBOARD="$GRAFANA/d/pga-mlops/pga-mlops?refresh=5s&from=now-15m&to=now"
MODEL="$MLFLOW/#/models/pga-score-model"
SWAGGER="$API/docs"
IN_API="http://127.0.0.1:8000"                       # the API as seen from its own container
JOURNAL="/app/data/monitoring/predictions.parquet"   # the stack's PREDICTIONS_LOG
PAUSE=1
[[ "${1:-}" == "--no-pause" ]] && PAUSE=0

compose() { docker compose --progress quiet --profile observability "$@"; }
# stdin closed: otherwise `exec` reads the terminal and swallows the "Enter" of the next pauses.
in_api() {
  compose exec -T -e PUSHGATEWAY_URL=http://pushgateway:9091 -e MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false api "$@" < /dev/null
}
quiet() { "$@" 2>&1 | grep -vE "INFO|WARNING|warnings.warn|^\s*$" || true; }
wait_api() {
  for _ in $(seq 1 90); do
    curl -fs "$API/health" 2>/dev/null | grep -q '"status":"ok"' && return 0
    sleep 2
  done
  echo "the API is not responding on $API (docker compose --profile observability logs api)" >&2
  exit 1
}
clear_pushgateway() { curl -fs -X DELETE "$PUSHGATEWAY/metrics/job/pga_monitor" > /dev/null || true; }
# Direct link to an MLflow experiment (experiment list if it does not exist yet).
experiment() {
  local id
  id=$(curl -s "$MLFLOW/api/2.0/mlflow/experiments/get-by-name?experiment_name=$1" \
       | grep -oE '"experiment_id": ?"[0-9]+"' | grep -oE '[0-9]+' | head -1 || true)
  echo "$MLFLOW/#/experiments/${id}"
}

if [[ "${1:-}" == "--prepare" ]]; then
  echo "== stack: images built, services started"
  compose up -d --build
  wait_api
  echo "== stack registry -> \"incident\" state (≈ 1 min)"
  # SQLite: the API and MLflow are stopped while the database is recreated.
  compose stop api mlflow
  log=$(mktemp)
  if ! compose run --rm --no-deps -T -e PGA_INCIDENT_DB=/mlflow/mlflow.db -e PGA_INCIDENT_ARTIFACTS=/app/mlartifacts api \
       bash -c "python scripts/incident.py setup && python scripts/incident.py corrupt && rm -rf /app/data/monitoring/*" \
       < /dev/null > "$log" 2>&1; then
    tail -30 "$log" >&2
    echo "preparation failed: the API and MLflow remain stopped (docker compose --profile observability up -d)" >&2
    exit 1
  fi
  grep -E "^== " "$log" || true
  rm -f "$log"
  compose up -d
  wait_api
  clear_pushgateway
  echo "ready: the API serves $(curl -s "$API/health" | grep -oE '"model_version":"[0-9]+"'), retrained on the corrupted backfill"
  echo "open before the presentation: $DASHBOARD"
  echo "                             $MODEL"
  echo "                             $SWAGGER"
  exit 0
fi

step() {
  echo
  echo "────────────────────────────────────────────────────────────────"
  echo "▶ $1"
  echo "────────────────────────────────────────────────────────────────"
  if [[ $PAUSE == 1 ]]; then read -r -p "  (Enter) " _; fi
}
look() { printf '  👀 %-8s %s\n' "$1" "$2"; }

if ! curl -fs "$API/health" > /dev/null 2>&1; then
  echo "no stack on $API: docker compose --profile observability up -d --build, then bash scripts/demo_live.sh --prepare" >&2
  exit 1
fi
curl -fs "$GRAFANA/api/health" > /dev/null 2>&1 || echo "  (no Grafana on $GRAFANA: start the stack with --profile observability)"

in_api rm -f "$JOURNAL"   # monitoring only sees this demo's requests
clear_pushgateway

echo "Demo on the stack: Grafana $DASHBOARD"
echo "                   MLflow  $MODEL"
echo "                   Swagger $SWAGGER"

step "1. The API serves the current champion"
curl -s "$API/health"; echo
look Grafana "\"Served version\": the version retrained on the corrupted backfill"
look MLflow "$MODEL  (champion alias on this version, previous on v1)"
look Swagger "$SWAGGER  (GET /health)"

step "2. Production: the March-April 2024 tournaments sent to /predict"
quiet in_api python scripts/send_requests.py --from 2024-03-01 --to 2024-04-21 --api-url "$IN_API" --batch 100
look Grafana "requests/s on /predict, p50/p95 latency, rows predicted by the corrupted version"

step "3. Monitoring on the API prediction journal (actual scores known)"
quiet in_api python -m pga.monitor check || true
look Grafana "drift panels: ALERT status, bias and z-score above thresholds, CUSUM"
look MLflow "$(experiment pga-monitoring)  (\"check\" run)"

step "4. The trigger decides and acts (rollback + API reload)"
quiet in_api python -m pga.trigger --as-of 2024-04-22 --api-url "$IN_API" || true
look MLflow "$MODEL  (champion back on v1, previous on the corrupted version)"
look MLflow "$(experiment pga-operations)  (record of the decision and the rollback)"
look Grafana "\"Served version\" changes, \"Reloads\" +1"

step "5. The API serves the healthy version"
curl -s "$API/health"; echo
look Swagger "$SWAGGER  (GET /health)"

step "6. Same requests, then monitoring: back within thresholds"
quiet in_api python scripts/send_requests.py --from 2024-03-01 --to 2024-04-21 --api-url "$IN_API" --batch 100
quiet in_api python -m pga.monitor check || true
look Grafana "predicted rows: v1's share rises; drift status back to OK"
look MLflow "$(experiment pga-monitoring)  (new \"check\" run, status ok)"

step "7. The same backfill today: promotion refused"
quiet in_api env PGA_INCIDENT_DB=/mlflow/mlflow.db PGA_INCIDENT_ARTIFACTS=/app/mlartifacts python scripts/incident.py retry
look MLflow "$MODEL  (new version registered, no alias: not promoted)"

echo
echo "end of demo — 'bash scripts/demo_live.sh --prepare' to replay it"
