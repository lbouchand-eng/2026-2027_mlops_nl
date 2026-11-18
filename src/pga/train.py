"""Stage 7: training, MLflow tracking and registration in the Model Registry.

Protocol (docs/metric.md): fit on `train_years`, evaluate on `valid_years`
(promotion decision), then refit on `refit_years` (full reference period) with the
same hyperparameters: that is the model that gets registered. Drift years are
evaluated for information only.

Each MLflow run holds: hyperparameters and year windows (params), MAE / RMSE /
bias / skill (metrics), git commit and data hash (tags), the model and its signature.

Promotion: the `champion` alias moves to the new version only if its validation skill
is > 0 and its validation MAE beats the current champion's; the former champion
gets the `previous` alias (used by the rollback procedure).

Usage: uv run python -m pga.train [--train-years 2021 2021] [--valid-years 2022 2022]
        [--refit-years 2021 2022] [--run-name NAME] [--no-register]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor

from pga.config import FEATURES_PATH, METRICS_PATH, MLARTIFACTS, MLFLOW_URI, ROOT, params
from pga.features import FEATURES, TARGET
from pga.profile import PROFILE_ARTIFACT, build_profile

EXPERIMENT = "pga-score"
CHAMPION, PREVIOUS = "champion", "previous"
# skops serialization (MLflow 3 default): internal HGB class to declare as trusted.
PIPELINE_OUTPUTS = ("dvc.lock", "reports")
SKOPS_TRUSTED = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


# --- data and metrics ----------------------------------------------------------------

def select(df: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    return df[df["year"].between(years[0], years[1])]


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].astype(float), df[TARGET].astype(float)


def baseline(df: pd.DataFrame) -> np.ndarray:
    """Naive baseline from docs/metric.md: historical course difficulty, 0 if unknown."""
    return df["course_diff_mean"].fillna(0.0).to_numpy()


def evaluate(y, pred, base, rounds) -> dict[str, float]:
    y, pred, base, rounds = map(np.asarray, (y, pred, base, rounds))
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    mae_base = float(np.mean(np.abs(base - y)))
    out = {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "mae_baseline": mae_base,
        "skill": 1.0 - mae / mae_base if mae_base > 0 else 0.0,
        "n": float(len(y)),
    }
    for r in range(1, 5):
        m = rounds == r
        if m.any():
            out[f"mae_r{r}"] = float(np.mean(np.abs(err[m])))
    return out


def event_bias_sd(event_ids, err) -> float:
    """Standard deviation of the mean bias per tournament: monitoring's reference noise.

    The whole field of a tournament shares the weather and the course setup: errors
    are correlated within a tournament, the independent unit is the tournament.
    """
    by_event = pd.Series(np.asarray(err, dtype=float)).groupby(np.asarray(event_ids)).mean()
    return float(by_event.std(ddof=1)) if len(by_event) > 1 else float("nan")


def make_model(p: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="absolute_error", random_state=p["random_state"], **p["model"])


def train(df: pd.DataFrame, p: dict, train_years, valid_years, refit_years, eval_years=()):
    """Returns (refit model, metrics {prefix_name: value})."""
    tr, va = select(df, train_years), select(df, valid_years)
    if tr.empty or va.empty:
        raise ValueError(f"empty window: train={train_years} ({len(tr)}), valid={valid_years} ({len(va)})")
    model = make_model(p).fit(*to_xy(tr))
    valid_pred = model.predict(to_xy(va)[0])
    metrics = {f"valid_{k}": v for k, v in evaluate(va[TARGET], valid_pred, baseline(va), va["round"]).items()}
    if "event_id" in va:
        metrics["valid_event_bias_sd"] = event_bias_sd(va["event_id"], valid_pred - va[TARGET].to_numpy())

    final = make_model(p).fit(*to_xy(select(df, refit_years)))
    for year in eval_years:
        ev = select(df, (year, year))
        if len(ev):
            m = evaluate(ev[TARGET], final.predict(to_xy(ev)[0]), baseline(ev), ev["round"])
            metrics |= {f"eval_{year}_{k}": v for k, v in m.items()}
    return final, metrics


# --- traceability -----------------------------------------------------------------------

def data_hash(path: Path) -> str:
    """md5 of the file: identical to the md5 recorded by DVC in dvc.lock."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dvc_lock_md5(path: Path, root: Path = ROOT) -> str | None:
    lock = root / "dvc.lock"
    if not lock.exists() or not path.resolve().is_relative_to(root):
        return None
    rel = path.resolve().relative_to(root).as_posix()
    for stage in (yaml.safe_load(lock.read_text()).get("stages") or {}).values():
        for out in stage.get("outs") or []:
            if out.get("path") == rel:
                return out.get("md5")
    return None


def git_info(root: Path = ROOT) -> dict[str, str]:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    # Files written by `dvc repro` itself: they don't make the code "dirty"
    # (data is tracked separately by data_hash).
    status = run("status", "--porcelain", "--untracked-files=no", "--", ".", *(f":!{p}" for p in PIPELINE_OUTPUTS))
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(status) and status != "unknown").lower(),
    }


# --- registry ----------------------------------------------------------------------------

def set_champion(client: MlflowClient, name: str, version: str, promoted_at: str | None = None) -> str | None:
    """`champion` -> `version`, the former champion -> `previous`. Returns the former champion.

    `promoted_at`: deployment date ("as-of" simulation); default: now.
    """
    try:
        old = str(client.get_model_version_by_alias(name, CHAMPION).version)
    except mlflow.exceptions.MlflowException:
        old = None
    if old == str(version):
        return old
    if old:
        client.set_registered_model_alias(name, PREVIOUS, old)
        client.set_model_version_tag(name, old, "status", "previous")
    client.set_registered_model_alias(name, CHAMPION, str(version))
    client.set_model_version_tag(name, str(version), "status", "champion")
    client.set_model_version_tag(name, str(version), "promoted_at",
                                 promoted_at or pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
    return old


def promote_if_better(client: MlflowClient, name: str, version: str, valid_mae: float, valid_skill: float) -> tuple[bool, str]:
    """Applies the promotion rule; returns (promoted, reason)."""
    client.set_model_version_tag(name, version, "valid_mae", f"{valid_mae:.6f}")
    client.set_model_version_tag(name, version, "valid_skill", f"{valid_skill:.6f}")
    if valid_skill <= 0:
        reason = f"skill {valid_skill:.4f} <= 0: does not beat the baseline"
    else:
        try:
            champ = client.get_model_version_by_alias(name, CHAMPION)
        except mlflow.exceptions.MlflowException:
            champ = None
        champ_mae = float(champ.tags.get("valid_mae", "inf")) if champ else float("inf")
        if valid_mae < champ_mae:
            set_champion(client, name, version)
            reason = f"MAE {valid_mae:.4f} < champion {champ_mae:.4f}" if champ else "first champion"
            return True, reason
        reason = f"MAE {valid_mae:.4f} >= champion v{champ.version} {champ_mae:.4f}"
    client.set_model_version_tag(name, version, "status", "candidate")
    return False, reason


def setup_mlflow(uri: str, artifacts: Path = MLARTIFACTS, experiment: str = EXPERIMENT) -> None:
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifacts.resolve().as_uri())
    mlflow.set_experiment(experiment)


def log_model_run(model, X_example: pd.DataFrame, params_: dict, metrics: dict, data_path: Path,
                  name: str, register: bool, fit_frame: pd.DataFrame, tags: dict | None = None) -> tuple[str, str | None]:
    """Logs a trained model in the active MLflow run (params, metrics, traceability).

    `fit_frame`: training data, summarized in the profile read back by monitoring.

    Returns (run_id, registered version or None).
    """
    run = mlflow.active_run()
    mlflow.log_params(params_)
    mlflow.set_tags(
        {
            **git_info(),
            "data_path": str(data_path.resolve().relative_to(ROOT)) if data_path.resolve().is_relative_to(ROOT) else str(data_path),
            "data_hash": data_hash(data_path),
            "dvc_features_md5": dvc_lock_md5(data_path) or "absent",
            **(tags or {}),
        }
    )
    mlflow.log_metrics(metrics)
    mlflow.log_dict(build_profile(fit_frame, FEATURES, params()["monitor"]["psi_bins"]), PROFILE_ARTIFACT)
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=infer_signature(X_example, model.predict(X_example)),
        input_example=X_example,
        skops_trusted_types=SKOPS_TRUSTED,
        registered_model_name=name if register else None,
    )
    return run.info.run_id, (str(info.registered_model_version) if register else None)


def model_params(p: dict) -> dict:
    return {f"model_{k}": v for k, v in p["model"].items()} | {
        "random_state": p["random_state"], "n_features": len(FEATURES), "features": ",".join(FEATURES)}


def run(df: pd.DataFrame, p: dict, years: dict, data_path: Path, register: bool = True, run_name: str | None = None) -> dict:
    model, metrics = train(df, p, years["train"], years["valid"], years["refit"], p.get("eval_years", ()))
    name = p["registered_model_name"]
    with mlflow.start_run(run_name=run_name):
        params_ = model_params(p) | {f"{k}_years": f"{v[0]}-{v[1]}" for k, v in years.items()}
        refit = select(df, years["refit"])
        run_id, version = log_model_run(model, to_xy(refit)[0].head(5), params_, metrics, data_path, name, register, refit)
        result = {"run_id": run_id, "metrics": metrics, "promoted": False, "version": version}
        if register:
            client = MlflowClient()
            if "valid_event_bias_sd" in metrics:
                client.set_model_version_tag(name, version, "valid_event_bias_sd", f"{metrics['valid_event_bias_sd']:.6f}")
            promoted, reason = promote_if_better(client, name, version, metrics["valid_mae"], metrics["valid_skill"])
            mlflow.set_tags({"registered_version": version, "promoted": str(promoted).lower(), "promotion_reason": reason})
            result |= {"promoted": promoted, "reason": reason}
    return result


def main() -> None:
    load_dotenv(ROOT / ".env")
    p = params()["train"]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("train", "valid", "refit"):
        ap.add_argument(f"--{k}-years", nargs=2, type=int, default=p[f"{k}_years"], metavar=("START", "END"))
    ap.add_argument("--run-name")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--data", type=Path, default=FEATURES_PATH)
    args = ap.parse_args()

    setup_mlflow(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))
    df = pd.read_parquet(args.data)
    years = {"train": args.train_years, "valid": args.valid_years, "refit": args.refit_years}
    res = run(df, p, years, args.data, register=not args.no_register, run_name=args.run_name)

    m = res["metrics"]
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({k: round(v, 4) for k, v in m.items()}, indent=2) + "\n")
    print(f"run {res['run_id']}")
    print(f"validation {years['valid']}: MAE {m['valid_mae']:.3f} | baseline {m['valid_mae_baseline']:.3f} "
          f"| skill {m['valid_skill']:+.3%} | bias {m['valid_bias']:+.3f}")
    for k in sorted(k for k in m if k.endswith("_mae") and k.startswith("eval_")):
        y = k.split("_")[1]
        print(f"  {y}: MAE {m[k]:.3f} | baseline {m[f'eval_{y}_mae_baseline']:.3f} | bias {m[f'eval_{y}_bias']:+.3f}")
    if res["version"]:
        outcome = "PROMOTED champion" if res["promoted"] else "not promoted"
        print(f"{p['registered_model_name']} v{res['version']}: {outcome} ({res['reason']})")


if __name__ == "__main__":
    main()
