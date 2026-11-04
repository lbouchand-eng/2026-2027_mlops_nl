"""Rollback du modèle servi (procédure : docs/runbook.md).

`champion` ← `previous` (ou la version donnée par --to) et `previous` ← ancien champion :
relancer la commande annule donc le rollback. Chaque rollback pose des tags sur la version
remise en service et laisse un run de trace dans l'expérience MLflow `pga-operations`.
Avec --api-url, l'API recharge le modèle (`POST /reload`) et on vérifie `/health`.

Usage : uv run python -m pga.rollback [--to VERSION] [--reason "..."] [--api-url http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from pga.config import MLFLOW_URI, ROOT, params

EXPERIMENT = "pga-operations"
CHAMPION, PREVIOUS = "champion", "previous"


class RollbackError(RuntimeError):
    pass


@dataclass
class RollbackResult:
    name: str
    from_version: str
    to_version: str
    reason: str


def _alias(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def rollback(client: MlflowClient, name: str, to: str | None = None, reason: str = "") -> RollbackResult:
    current = _alias(client, name, CHAMPION)
    if current is None:
        raise RollbackError(f"aucun champion pour {name} : rien à annuler")
    target = str(to) if to else _alias(client, name, PREVIOUS)
    if target is None:
        raise RollbackError(f"pas d'alias `{PREVIOUS}` pour {name} : préciser --to VERSION")
    if target == current:
        raise RollbackError(f"v{target} est déjà le champion")
    try:
        client.get_model_version(name, target)
    except MlflowException as exc:
        raise RollbackError(f"version v{target} introuvable pour {name}") from exc

    client.set_registered_model_alias(name, CHAMPION, target)
    client.set_registered_model_alias(name, PREVIOUS, current)
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    for k, v in {"rolled_back_from": current, "rolled_back_at": now, "rollback_reason": reason or "non précisée",
                 "status": "champion"}.items():
        client.set_model_version_tag(name, target, k, v)
    client.set_model_version_tag(name, current, "status", "previous")
    return RollbackResult(name, current, target, reason)


def log_operation(res: RollbackResult) -> str:
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        from pga.config import MLARTIFACTS

        mlflow.create_experiment(EXPERIMENT, artifact_location=MLARTIFACTS.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"rollback v{res.from_version}->v{res.to_version}") as run:
        mlflow.set_tags({"operation": "rollback", "model_name": res.name, "from_version": res.from_version,
                         "to_version": res.to_version, "reason": res.reason or "non précisée"})
        return run.info.run_id


def reload_api(api_url: str, expected: str, timeout: float = 30) -> dict:
    """Demande à l'API de recharger le champion et vérifie la version servie."""
    r = requests.post(f"{api_url.rstrip('/')}/reload", timeout=timeout)
    r.raise_for_status()
    health = requests.get(f"{api_url.rstrip('/')}/health", timeout=timeout).json()
    if str(health.get("model_version")) != expected:
        raise RollbackError(f"l'API sert v{health.get('model_version')} au lieu de v{expected}")
    return health


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Rollback du modèle servi (champion <-> previous)")
    ap.add_argument("--to", help="version à remettre en service (défaut : alias previous)")
    ap.add_argument("--reason", default="", help="motif (ex. alerte monitoring du 2024-03-28)")
    ap.add_argument("--api-url", help="URL de l'API à recharger (ex. http://127.0.0.1:8000)")
    args = ap.parse_args(argv)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))
    name = params()["train"]["registered_model_name"]
    try:
        res = rollback(MlflowClient(), name, args.to, args.reason)
    except RollbackError as exc:
        print(f"rollback refusé : {exc}", file=sys.stderr)
        return 1
    run_id = log_operation(res)
    print(f"{name} : champion v{res.from_version} -> v{res.to_version} (previous = v{res.from_version}), trace MLflow {run_id}")
    if args.api_url:
        try:
            health = reload_api(args.api_url, res.to_version)
        except (requests.RequestException, RollbackError) as exc:
            print(f"  ! registry mis à jour mais l'API n'a pas rechargé : {exc}", file=sys.stderr)
            print(f"    relancer : curl -X POST {args.api_url.rstrip('/')}/reload", file=sys.stderr)
            return 2
        print(f"  API rechargée : sert v{health['model_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
