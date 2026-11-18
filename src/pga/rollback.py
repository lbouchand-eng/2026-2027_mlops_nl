"""Rollback of the served model (procedure: docs/runbook.md).

`champion` ← `previous` (or the version given by --to) and `previous` ← former champion:
running the command again therefore undoes the rollback. Each rollback sets tags on the
redeployed version and leaves a trace run in the MLflow experiment `pga-operations`.
With --api-url, the API reloads the model (`POST /reload`) and `/health` is checked.

Usage: uv run python -m pga.rollback [--to VERSION] [--reason "..."] [--api-url http://127.0.0.1:8000]
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
from pga.train import set_champion

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
        raise RollbackError(f"no champion for {name}: nothing to undo")
    target = str(to) if to else _alias(client, name, PREVIOUS)
    if target is None:
        raise RollbackError(f"no `{PREVIOUS}` alias for {name}: specify --to VERSION")
    if target == current:
        raise RollbackError(f"v{target} is already the champion")
    try:
        client.get_model_version(name, target)
    except MlflowException as exc:
        raise RollbackError(f"version v{target} not found for {name}") from exc

    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    set_champion(client, name, target, promoted_at=now)  # champion <- target, previous <- former
    for k, v in {"rolled_back_from": current, "rolled_back_at": now, "rollback_reason": reason or "unspecified"}.items():
        client.set_model_version_tag(name, target, k, v)
    return RollbackResult(name, current, target, reason)


def log_operation(res: RollbackResult) -> str:
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        from pga.config import MLARTIFACTS

        mlflow.create_experiment(EXPERIMENT, artifact_location=MLARTIFACTS.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"rollback v{res.from_version}->v{res.to_version}") as run:
        mlflow.set_tags({"operation": "rollback", "model_name": res.name, "from_version": res.from_version,
                         "to_version": res.to_version, "reason": res.reason or "unspecified"})
        return run.info.run_id


def reload_api(api_url: str, expected: str, timeout: float = 30) -> dict:
    """Asks the API to reload the champion and checks the served version."""
    key = os.environ.get("PGA_API_KEY")
    r = requests.post(f"{api_url.rstrip('/')}/reload", headers={"X-API-Key": key} if key else None, timeout=timeout)
    r.raise_for_status()
    health = requests.get(f"{api_url.rstrip('/')}/health", timeout=timeout).json()
    if str(health.get("model_version")) != expected:
        raise RollbackError(f"the API serves v{health.get('model_version')} instead of v{expected}")
    return health


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Rollback of the served model (champion <-> previous)")
    ap.add_argument("--to", help="version to redeploy (default: previous alias)")
    ap.add_argument("--reason", default="", help="reason (e.g. monitoring alert of 2024-03-28)")
    ap.add_argument("--api-url", help="URL of the API to reload (e.g. http://127.0.0.1:8000)")
    args = ap.parse_args(argv)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))
    name = params()["train"]["registered_model_name"]
    try:
        res = rollback(MlflowClient(), name, args.to, args.reason)
    except RollbackError as exc:
        print(f"rollback refused: {exc}", file=sys.stderr)
        return 1
    run_id = log_operation(res)
    print(f"{name}: champion v{res.from_version} -> v{res.to_version} (previous = v{res.from_version}), MLflow trace {run_id}")
    if args.api_url:
        try:
            health = reload_api(args.api_url, res.to_version)
        except (requests.RequestException, RollbackError) as exc:
            print(f"  ! registry updated but the API did not reload: {exc}", file=sys.stderr)
            print(f"    retry: curl -X POST {args.api_url.rstrip('/')}/reload", file=sys.stderr)
            return 2
        print(f"  API reloaded: serving v{health['model_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
