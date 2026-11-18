#!/usr/bin/env bash
# Démo en direct sur la stack Compose visible : mêmes étapes que demo.sh, mais sur l'API (:8000)
# et le registry servis par Compose. Chaque étape indique quoi regarder : Grafana, MLflow, Swagger.
# Les commandes tournent dans le conteneur `api` : seul Docker est nécessaire sur la machine.
#
#   docker compose --profile observability up -d --build   # API, MLflow, Prometheus, Pushgateway, Grafana
#   bash scripts/demo_live.sh --prepare    # avant l'oral (≈ 2 min) : registry de la stack dans l'état « incident »
#   bash scripts/demo_live.sh              # la démo, avec une pause entre chaque étape (Entrée)
#   bash scripts/demo_live.sh --no-pause   # d'une traite (répétition, test)
#
# --prepare **remplace** le registry de la stack (champion sain v1, puis v2 réentraînée sur le
# backfill corrompu et promue) et vide le journal des prédictions. Pour revenir au registry
# standard : docker compose down -v. Ne pas faire tourner le service `scheduler` en même temps.
set -euo pipefail
cd "$(dirname "$0")/.."

API="http://127.0.0.1:${API_PORT:-8000}"
MLFLOW="http://127.0.0.1:${MLFLOW_PORT:-5001}"
GRAFANA="http://127.0.0.1:${GRAFANA_PORT:-3000}"
PUSHGATEWAY="http://127.0.0.1:${PUSHGATEWAY_PORT:-9091}"
DASHBOARD="$GRAFANA/d/pga-mlops/pga-mlops?refresh=5s&from=now-15m&to=now"
MODEL="$MLFLOW/#/models/pga-score-model"
SWAGGER="$API/docs"
IN_API="http://127.0.0.1:8000"                       # l'API vue depuis son propre conteneur
JOURNAL="/app/data/monitoring/predictions.parquet"   # PREDICTIONS_LOG de la stack
PAUSE=1
[[ "${1:-}" == "--no-pause" ]] && PAUSE=0

compose() { docker compose --progress quiet --profile observability "$@"; }
# stdin fermé : sinon `exec` lit le terminal et avale les « Entrée » des pauses suivantes.
in_api() {
  compose exec -T -e PUSHGATEWAY_URL=http://pushgateway:9091 -e MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false api "$@" < /dev/null
}
quiet() { "$@" 2>&1 | grep -vE "INFO|WARNING|warnings.warn|^\s*$" || true; }
wait_api() {
  for _ in $(seq 1 90); do
    curl -fs "$API/health" 2>/dev/null | grep -q '"status":"ok"' && return 0
    sleep 2
  done
  echo "l'API ne répond pas sur $API (docker compose --profile observability logs api)" >&2
  exit 1
}
clear_pushgateway() { curl -fs -X DELETE "$PUSHGATEWAY/metrics/job/pga_monitor" > /dev/null || true; }
# Lien direct vers une expérience MLflow (liste des expériences si elle n'existe pas encore).
experiment() {
  local id
  id=$(curl -s "$MLFLOW/api/2.0/mlflow/experiments/get-by-name?experiment_name=$1" \
       | grep -oE '"experiment_id": ?"[0-9]+"' | grep -oE '[0-9]+' | head -1 || true)
  echo "$MLFLOW/#/experiments/${id}"
}

if [[ "${1:-}" == "--prepare" ]]; then
  echo "== stack : images construites, services démarrés"
  compose up -d --build
  wait_api
  echo "== registry de la stack -> état « incident » (≈ 1 min)"
  # SQLite : l'API et MLflow sont arrêtés pendant que la base est recréée.
  compose stop api mlflow
  log=$(mktemp)
  if ! compose run --rm --no-deps -T -e PGA_INCIDENT_DB=/mlflow/mlflow.db -e PGA_INCIDENT_ARTIFACTS=/app/mlartifacts api \
       bash -c "python scripts/incident.py setup && python scripts/incident.py corrupt && rm -rf /app/data/monitoring/*" \
       < /dev/null > "$log" 2>&1; then
    tail -30 "$log" >&2
    echo "échec de la préparation : l'API et MLflow restent arrêtés (docker compose --profile observability up -d)" >&2
    exit 1
  fi
  grep -E "^== " "$log" || true
  rm -f "$log"
  compose up -d
  wait_api
  clear_pushgateway
  echo "prêt : l'API sert $(curl -s "$API/health" | grep -oE '"model_version":"[0-9]+"'), réentraînée sur le backfill corrompu"
  echo "ouvrir avant l'oral : $DASHBOARD"
  echo "                      $MODEL"
  echo "                      $SWAGGER"
  exit 0
fi

step() {
  echo
  echo "────────────────────────────────────────────────────────────────"
  echo "▶ $1"
  echo "────────────────────────────────────────────────────────────────"
  if [[ $PAUSE == 1 ]]; then read -r -p "  (Entrée) " _; fi
}
look() { printf '  👀 %-8s %s\n' "$1" "$2"; }

if ! curl -fs "$API/health" > /dev/null 2>&1; then
  echo "stack absente sur $API : docker compose --profile observability up -d --build, puis bash scripts/demo_live.sh --prepare" >&2
  exit 1
fi
curl -fs "$GRAFANA/api/health" > /dev/null 2>&1 || echo "  (Grafana absent sur $GRAFANA : lancer la stack avec --profile observability)"

in_api rm -f "$JOURNAL"   # le monitoring ne voit que les requêtes de cette démo
clear_pushgateway

echo "Démo sur la stack : Grafana $DASHBOARD"
echo "                    MLflow  $MODEL"
echo "                    Swagger $SWAGGER"

step "1. L'API sert le champion actuel"
curl -s "$API/health"; echo
look Grafana "« Version servie » : la version réentraînée sur le backfill corrompu"
look MLflow "$MODEL  (alias champion sur cette version, previous sur v1)"
look Swagger "$SWAGGER  (GET /health)"

step "2. Production : les tournois de mars-avril 2024 envoyés à /predict"
quiet in_api python scripts/send_requests.py --from 2024-03-01 --to 2024-04-21 --api-url "$IN_API" --batch 100
look Grafana "requêtes/s sur /predict, latence p50/p95, lignes prédites par la version corrompue"

step "3. Monitoring sur le journal de l'API (scores réels connus)"
quiet in_api python -m pga.monitor check || true
look Grafana "panneaux drift : statut ALERT, biais et z-score au-dessus des seuils, CUSUM"
look MLflow "$(experiment pga-monitoring)  (run « check »)"

step "4. Le déclencheur décide et agit (rollback + rechargement de l'API)"
quiet in_api python -m pga.trigger --as-of 2024-04-22 --api-url "$IN_API" || true
look MLflow "$MODEL  (champion revenu sur v1, previous sur la version corrompue)"
look MLflow "$(experiment pga-operations)  (trace de la décision et du rollback)"
look Grafana "« Version servie » change, « Rechargements » +1"

step "5. L'API sert la version saine"
curl -s "$API/health"; echo
look Swagger "$SWAGGER  (GET /health)"

step "6. Mêmes requêtes, puis monitoring : retour dans les seuils"
quiet in_api python scripts/send_requests.py --from 2024-03-01 --to 2024-04-21 --api-url "$IN_API" --batch 100
quiet in_api python -m pga.monitor check || true
look Grafana "lignes prédites : la part de v1 monte ; statut du drift repasse à OK"
look MLflow "$(experiment pga-monitoring)  (nouveau run « check », statut ok)"

step "7. Le même backfill aujourd'hui : refusé à la promotion"
quiet in_api env PGA_INCIDENT_DB=/mlflow/mlflow.db PGA_INCIDENT_ARTIFACTS=/app/mlartifacts python scripts/incident.py retry
look MLflow "$MODEL  (nouvelle version enregistrée, sans alias : non promue)"

echo
echo "fin de la démo — 'bash scripts/demo_live.sh --prepare' pour la rejouer"
