#!/usr/bin/env bash
# Runs the trigger at a fixed interval against the stack registry (Compose `scheduler` service).
# Codes 0, 10, 20, 30: normal decisions. Any other code: the trigger failed.
set -uo pipefail
cd "$(dirname "$0")/.."

while true; do
  python -m pga.trigger --api-url "${API_URL:-http://api:8000}"
  code=$?
  case $code in
    0|10|20|30) ;;
    *)
      echo "trigger failed (code $code)" >&2
      python -c "import sys; from pga.alerting import notify; notify(sys.argv[1])" "[PGA] trigger failed (code $code)"
      ;;
  esac
  sleep "${TRIGGER_INTERVAL_SECONDS:-604800}"
done
