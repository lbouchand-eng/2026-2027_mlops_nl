"""État du Registry (alias champion / previous) ; utilisé par scripts/demo_setup.sh.

  uv run python scripts/registry_status.py                    # affiche les versions et alias
  uv run python scripts/registry_status.py --has-previous     # code 0 si l'alias previous existe
  uv run python scripts/registry_status.py --ensure-previous  # pose previous sur la dernière version « v-historique »
"""
from __future__ import annotations

import argparse
import os
import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from pga.config import MLFLOW_URI, params


def alias(c: MlflowClient, name: str, a: str) -> str | None:
    try:
        return str(c.get_model_version_by_alias(name, a).version)
    except MlflowException:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--has-previous", action="store_true")
    ap.add_argument("--ensure-previous", action="store_true")
    args = ap.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))
    name = params()["train"]["registered_model_name"]
    c = MlflowClient()
    if args.has_previous:
        return 0 if alias(c, name, "previous") else 1

    if args.ensure_previous and not alias(c, name, "previous"):
        champ = alias(c, name, "champion")
        hist = [int(v.version) for v in c.search_model_versions(f"name='{name}'")
                if c.get_run(v.run_id).info.run_name == "v-historique" and str(v.version) != champ]
        if not hist:
            print("aucune version historique distincte du champion", file=sys.stderr)
            return 1
        c.set_registered_model_alias(name, "previous", str(max(hist)))
        c.set_model_version_tag(name, str(max(hist)), "status", "previous")

    versions = sorted(c.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))
    for v in versions:
        v = c.get_model_version(name, v.version)  # la recherche ne renvoie pas les alias
        tags = v.tags
        print(f"v{v.version:<3} {','.join(v.aliases) or '-':<10} MAE valid {tags.get('valid_mae', '?'):<9} "
              f"run {c.get_run(v.run_id).info.run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
