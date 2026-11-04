#!/usr/bin/env bash
# Prépare un Registry démontrable sur un clone neuf : un `champion` et un `previous`
# (sinon le rollback n'a pas de cible).
#   - version « historique » : entraînée sur 2021 seulement (MAE validation ≈ 2,41)
#   - version « standard » (params.yaml, 2018-2021, ≈ 2,38) : promue si elle bat le champion
# Si un champion existait déjà (dvc repro), la version historique devient `previous`.
# Usage : bash scripts/demo_setup.sh   (depuis n'importe où)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f data/processed/features.parquet ]]; then
  echo "features absentes : lancer d'abord 'uv run dvc pull' (stockage partagé) ou 'uv run dvc repro'." >&2
  exit 1
fi

if uv run python scripts/registry_status.py --has-previous 2>/dev/null; then
  echo "Registry déjà prêt (alias previous présent) : rien à faire."
  uv run python scripts/registry_status.py
  exit 0
fi

echo "== version historique (entraînement 2021 seul, moins bonne)"
uv run python -m pga.train --train-years 2021 2021 --refit-years 2021 2022 --run-name "v-historique"
echo "== version standard (params.yaml), promue si elle bat le champion"
uv run python -m pga.train --run-name "v-standard"
uv run python scripts/registry_status.py --ensure-previous
