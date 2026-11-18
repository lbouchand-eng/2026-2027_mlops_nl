#!/usr/bin/env bash
# Prepares a demo-ready Registry on a fresh clone: a `champion` and a `previous`
# (otherwise the rollback has no target).
#   - "historical" version: trained on 2021 only (validation MAE ≈ 2.41)
#   - "standard" version (params.yaml, 2018-2021, ≈ 2.38): promoted if it beats the champion
# If a champion already existed (dvc repro), the historical version becomes `previous`.
# Usage: bash scripts/demo_setup.sh   (from anywhere)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f data/processed/features.parquet ]]; then
  echo "features missing: run 'uv run dvc pull' (shared storage) or 'uv run dvc repro' first." >&2
  exit 1
fi

if uv run python scripts/registry_status.py --has-previous 2>/dev/null; then
  echo "Registry already ready (previous alias present): nothing to do."
  uv run python scripts/registry_status.py
  exit 0
fi

echo "== historical version (trained on 2021 only, worse)"
uv run python -m pga.train --train-years 2021 2021 --refit-years 2021 2022 --run-name "v-historique"
echo "== standard version (params.yaml), promoted if it beats the champion"
uv run python -m pga.train --run-name "v-standard"
uv run python scripts/registry_status.py --ensure-previous
