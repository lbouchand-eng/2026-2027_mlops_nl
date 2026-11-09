#!/usr/bin/env bash
# Lance le déclencheur à intervalle fixe sur le registry de la stack (service Compose `scheduler`).
# Codes 0, 10, 20, 30 : décisions normales. Tout autre code : le déclencheur a échoué.
set -uo pipefail
cd "$(dirname "$0")/.."

while true; do
  python -m pga.trigger --api-url "${API_URL:-http://api:8000}"
  code=$?
  case $code in
    0|10|20|30) ;;
    *)
      echo "déclencheur en échec (code $code)" >&2
      python -c "import sys; from pga.alerting import notify; notify(sys.argv[1])" "[PGA] déclencheur en échec (code $code)"
      ;;
  esac
  sleep "${TRIGGER_INTERVAL_SECONDS:-604800}"
done
