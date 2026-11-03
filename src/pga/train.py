"""Étape 7 : entraînement, suivi MLflow et enregistrement dans le Model Registry.

Protocole (docs/metric.md) : fit sur `train_years`, évaluation sur `valid_years`
(décision de promotion), puis refit sur `refit_years` (période de référence complète) avec
les mêmes hyperparamètres : c'est ce modèle qui est enregistré. Les années de drift sont
évaluées à titre informatif.

Chaque run MLflow contient : hyperparamètres et fenêtres d'années (params), MAE / RMSE /
biais / skill (metrics), commit git et hash des données (tags), le modèle et sa signature.

Promotion : l'alias `champion` passe sur la nouvelle version seulement si son skill de
validation est > 0 et sa MAE de validation meilleure que celle du champion actuel ; l'ancien
champion reçoit l'alias `previous` (utilisé par la procédure de rollback).

Usage : uv run python -m pga.train [--train-years 2021 2021] [--valid-years 2022 2022]
        [--refit-years 2021 2022] [--run-name NOM] [--no-register]
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

EXPERIMENT = "pga-score"
CHAMPION, PREVIOUS = "champion", "previous"
# Sérialisation skops (défaut MLflow 3) : classe interne du HGB à déclarer comme sûre.
SKOPS_TRUSTED = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


# --- données et métriques -------------------------------------------------------------

def select(df: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    return df[df["year"].between(years[0], years[1])]


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].astype(float), df[TARGET].astype(float)


def baseline(df: pd.DataFrame) -> np.ndarray:
    """Baseline naïve de docs/metric.md : difficulté historique du parcours, 0 si inconnue."""
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


def make_model(p: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="absolute_error", random_state=p["random_state"], **p["model"])


def train(df: pd.DataFrame, p: dict, train_years, valid_years, refit_years, eval_years=()):
    """Renvoie (modèle refit, métriques {préfixe_nom: valeur})."""
    tr, va = select(df, train_years), select(df, valid_years)
    if tr.empty or va.empty:
        raise ValueError(f"fenêtre vide : train={train_years} ({len(tr)}), valid={valid_years} ({len(va)})")
    model = make_model(p).fit(*to_xy(tr))
    metrics = {f"valid_{k}": v for k, v in evaluate(va[TARGET], model.predict(to_xy(va)[0]), baseline(va), va["round"]).items()}

    final = make_model(p).fit(*to_xy(select(df, refit_years)))
    for year in eval_years:
        ev = select(df, (year, year))
        if len(ev):
            m = evaluate(ev[TARGET], final.predict(to_xy(ev)[0]), baseline(ev), ev["round"])
            metrics |= {f"eval_{year}_{k}": v for k, v in m.items()}
    return final, metrics


# --- traçabilité -------------------------------------------------------------------------

def data_hash(path: Path) -> str:
    """md5 du fichier : identique au md5 enregistré par DVC dans dvc.lock."""
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

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(status) and status != "unknown").lower(),
    }


# --- registry ----------------------------------------------------------------------------

def promote_if_better(client: MlflowClient, name: str, version: str, valid_mae: float, valid_skill: float) -> tuple[bool, str]:
    """Applique la règle de promotion ; renvoie (promu, raison)."""
    client.set_model_version_tag(name, version, "valid_mae", f"{valid_mae:.6f}")
    client.set_model_version_tag(name, version, "valid_skill", f"{valid_skill:.6f}")
    if valid_skill <= 0:
        reason = f"skill {valid_skill:.4f} <= 0 : ne bat pas la baseline"
    else:
        try:
            champ = client.get_model_version_by_alias(name, CHAMPION)
        except mlflow.exceptions.MlflowException:
            champ = None
        champ_mae = float(champ.tags.get("valid_mae", "inf")) if champ else float("inf")
        if valid_mae < champ_mae:
            if champ:
                client.set_registered_model_alias(name, PREVIOUS, champ.version)
            client.set_registered_model_alias(name, CHAMPION, version)
            reason = f"MAE {valid_mae:.4f} < champion {champ_mae:.4f}" if champ else "premier champion"
            client.set_model_version_tag(name, version, "status", "champion")
            return True, reason
        reason = f"MAE {valid_mae:.4f} >= champion v{champ.version} {champ_mae:.4f}"
    client.set_model_version_tag(name, version, "status", "candidate")
    return False, reason


def setup_mlflow(uri: str, artifacts: Path = MLARTIFACTS) -> None:
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=artifacts.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def run(df: pd.DataFrame, p: dict, years: dict, data_path: Path, register: bool = True, run_name: str | None = None) -> dict:
    model, metrics = train(df, p, years["train"], years["valid"], years["refit"], p.get("eval_years", ()))
    name = p["registered_model_name"]
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.log_params({f"model_{k}": v for k, v in p["model"].items()})
        mlflow.log_params(
            {
                "random_state": p["random_state"],
                "train_years": f"{years['train'][0]}-{years['train'][1]}",
                "valid_years": f"{years['valid'][0]}-{years['valid'][1]}",
                "refit_years": f"{years['refit'][0]}-{years['refit'][1]}",
                "n_features": len(FEATURES),
                "features": ",".join(FEATURES),
            }
        )
        mlflow.set_tags(
            {
                **git_info(),
                "data_path": str(data_path.resolve().relative_to(ROOT)) if data_path.resolve().is_relative_to(ROOT) else str(data_path),
                "data_hash": data_hash(data_path),
                "dvc_features_md5": dvc_lock_md5(data_path) or "absent",
            }
        )
        mlflow.log_metrics(metrics)
        X_ex = to_xy(select(df, years["refit"]))[0].head(5)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_ex, model.predict(X_ex)),
            input_example=X_ex,
            skops_trusted_types=SKOPS_TRUSTED,
            registered_model_name=name if register else None,
        )
        result = {"run_id": active.info.run_id, "metrics": metrics, "promoted": False, "version": None}
        if register:
            version = str(info.registered_model_version)
            promoted, reason = promote_if_better(MlflowClient(), name, version, metrics["valid_mae"], metrics["valid_skill"])
            mlflow.set_tags({"registered_version": version, "promoted": str(promoted).lower(), "promotion_reason": reason})
            result |= {"promoted": promoted, "version": version, "reason": reason}
    return result


def main() -> None:
    load_dotenv(ROOT / ".env")
    p = params()["train"]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("train", "valid", "refit"):
        ap.add_argument(f"--{k}-years", nargs=2, type=int, default=p[f"{k}_years"], metavar=("DEBUT", "FIN"))
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
    print(f"validation {years['valid']} : MAE {m['valid_mae']:.3f} | baseline {m['valid_mae_baseline']:.3f} "
          f"| skill {m['valid_skill']:+.3%} | biais {m['valid_bias']:+.3f}")
    for k in sorted(k for k in m if k.endswith("_mae") and k.startswith("eval_")):
        y = k.split("_")[1]
        print(f"  {y} : MAE {m[k]:.3f} | baseline {m[f'eval_{y}_mae_baseline']:.3f} | biais {m[f'eval_{y}_bias']:+.3f}")
    if res["version"]:
        print(f"{p['registered_model_name']} v{res['version']} : {'PROMU champion' if res['promoted'] else 'non promu'} ({res['reason']})")


if __name__ == "__main__":
    main()
