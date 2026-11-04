"""Déclencheur : décide, à une date donnée, entre ne rien faire, réentraîner ou revenir en arrière.

Le monitoring est recalculé « as-of » (uniquement avec les tournois joués avant la date)
sur les `consecutive_checks` derniers contrôles, puis :

| Constat                                                       | Action          | Code |
|---------------------------------------------------------------|-----------------|------|
| rien d'anormal                                                | rien            | 0    |
| biais/MAE dégradés, champion en service depuis longtemps      | réentraînement  | 10   |
| alerte PSI sur une entrée                                     | incident de données : pas de réentraînement (on apprendrait le bug) | 20 |
| biais/MAE dégradés, champion mis en service récemment         | rollback        | 30   |

« Dégradés » : alerte au dernier contrôle, ou warning/alert sur tous les derniers contrôles.

Réentraînement sans fuite et comparable : la fenêtre de décision est formée des
`window_events` derniers tournois avant la date ; le challenger est entraîné sur tout ce
qui précède ; **champion et challenger sont évalués sur cette même fenêtre**. Le challenger
devient champion si sa MAE est meilleure et son |biais| pas pire.

Usage : uv run python -m pga.trigger [--as-of 2024-02-08] [--dry-run] [--api-url http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from pga import monitor as M
from pga import rollback as R
from pga.config import FEATURES_PATH, MLFLOW_URI, REPORTS, ROOT, params
from pga.features import FEATURES, TARGET
from pga.train import baseline, evaluate, log_model_run, make_model, model_params, set_champion, setup_mlflow, to_xy

OPERATIONS = "pga-operations"
CODES = {"none": 0, "retrain": 10, "data_incident": 20, "rollback": 30}
TRIGGER_DIR = Path(os.environ.get("PGA_TRIGGER_DIR", REPORTS / "trigger"))  # surchargeable (démo)


@dataclass
class Decision:
    action: str
    reason: str
    as_of: str
    champion: str
    last_status: str
    perf_levels: list[str]


# --- décision -------------------------------------------------------------------------------

def perf_level(row: pd.Series, pm: dict) -> str:
    z, ratio = row.get("bias_z"), row.get("mae_ratio")
    if z is None or pd.isna(z):
        return "ok"
    if abs(z) > pm["bias_z_alert"] or (pd.notna(ratio) and ratio > pm["mae_ratio_alert"]):
        return "alert"
    return "warning" if abs(z) > pm["bias_z_warning"] else "ok"


def psi_alert(row: pd.Series) -> bool:
    return "[alert] PSI" in str(row.get("reasons", ""))


def decide(timeline: pd.DataFrame, as_of: pd.Timestamp, promoted_at: pd.Timestamp | None,
           champion: str, pm: dict, pt: dict) -> Decision:
    last = timeline.iloc[-1]
    levels = [perf_level(r, pm) for _, r in timeline.tail(pt["consecutive_checks"]).iterrows()]
    base = dict(as_of=str(as_of.date()), champion=champion, last_status=last["status"], perf_levels=levels)
    if psi_alert(last):
        return Decision("data_incident", f"alerte PSI : {last['reasons']}", **base)
    degraded = levels[-1] == "alert" or (len(levels) >= pt["consecutive_checks"] and "ok" not in levels)
    if not degraded:
        return Decision("none", "performance dans les seuils", **base)
    detail = f"biais {last['bias']:+.2f} coup (z {last['bias_z']:+.1f}), MAE ratio {last['mae_ratio']:.2f}, contrôles {levels}"
    if promoted_at is not None and 0 <= (as_of - promoted_at).days <= pt["recent_promotion_days"]:
        return Decision("rollback", f"dégradation juste après la mise en service de v{champion} "
                                    f"({promoted_at.date()}) : {detail}", **base)
    return Decision("retrain", f"dérive confirmée : {detail}", **base)


# --- monitoring as-of ---------------------------------------------------------------------------

def event_order(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    return df[df["event_start"] <= as_of].groupby("event_id")["event_start"].min().sort_values()


def monitoring_as_of(df: pd.DataFrame, as_of: pd.Timestamp, champ: M.Champion, pm: dict, k: int) -> pd.DataFrame:
    """Les `k` derniers contrôles du monitoring avec les seuls tournois joués avant `as_of`."""
    order = event_order(df, as_of)
    needed = order.index[-(pm["window_events"] + k - 1):]
    cur = df[df["event_id"].isin(needed)].copy()
    cur["prediction"] = M.predict_champion(champ, cur)
    lo, hi = pm["reference_years"]
    ref = M.Reference(df[df["year"].between(lo, hi)], pm["psi_features"], pm["psi_bins"], M.load_thresholds())
    return M.rolling_checks(cur, ref, champ, pm, emit_from=order.iloc[-k])


# --- réentraînement -----------------------------------------------------------------------------

def decision_window(df: pd.DataFrame, as_of: pd.Timestamp, n_events: int) -> tuple[pd.Index, pd.Timestamp]:
    order = event_order(df, as_of)
    ids = order.index[-n_events:]
    return ids, order.loc[ids].min()


def retrain(df: pd.DataFrame, as_of: pd.Timestamp, champ: M.Champion, p_train: dict, pm: dict, pt: dict,
            data_path: Path, transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
            transform_label: str = "aucune", promote: bool = True) -> dict:
    """Entraîne un challenger sur les données antérieures à la fenêtre de décision et le compare
    au champion sur cette fenêtre. `transform` : altération des données d'entraînement et de
    décision (backfill), utilisée par l'incident simulé.
    """
    ids, start = decision_window(df, as_of, pm["window_events"])
    data = transform(df) if transform else df
    fit = data[(data["year"] >= pt["train_start_year"]) & (data["event_start"] < start)]
    dec = data[data["event_id"].isin(ids)]
    if fit.empty or dec.empty:
        raise ValueError(f"données insuffisantes au {as_of.date()} : fit={len(fit)}, décision={len(dec)}")

    challenger = make_model(p_train).fit(*to_xy(fit))
    X_dec, y_dec = to_xy(dec)
    m_ch = evaluate(y_dec, challenger.predict(X_dec), baseline(dec), dec["round"])
    m_champ = evaluate(y_dec, M.predict_champion(champ, dec), baseline(dec), dec["round"])
    better = m_ch["mae"] < m_champ["mae"] and abs(m_ch["bias"]) <= abs(m_champ["bias"]) + pt["bias_tolerance"]

    name = p_train["registered_model_name"]
    metrics = {f"decision_{k}": v for k, v in m_ch.items()} | {f"decision_champion_{k}": v for k, v in m_champ.items()}
    params_ = model_params(p_train) | {"fit_from": str(fit["event_start"].min().date()), "fit_until": str(start.date()),
                                        "decision_events": len(ids), "as_of": str(as_of.date())}
    tags = {"trigger": "retrain", "as_of": str(as_of.date()), "compared_to": f"v{champ.version}",
            "data_transform": transform_label}
    with mlflow.start_run(run_name=f"retrain as-of {as_of.date()}"):
        run_id, version = log_model_run(challenger, X_dec.head(5), params_, metrics, data_path, name, True, tags)
        client = MlflowClient()
        # Seuils de monitoring de la nouvelle version : MAE mesurée sur la fenêtre de décision,
        # bruit du biais par tournoi hérité du champion (propriété du problème, pas du modèle).
        for k, v in {"valid_mae": m_ch["mae"], "valid_skill": m_ch["skill"],
                     "valid_event_bias_sd": champ.event_bias_sd, "as_of": str(as_of.date()),
                     "data_transform": transform_label}.items():
            client.set_model_version_tag(name, version, k, f"{v:.6f}" if isinstance(v, float) else str(v))
        promoted = bool(better and promote)
        if promoted:
            set_champion(client, name, version, promoted_at=str(as_of.date()))
        else:
            client.set_model_version_tag(name, version, "status", "candidate")
        reason = (f"challenger MAE {m_ch['mae']:.3f} / biais {m_ch['bias']:+.3f} vs champion v{champ.version} "
                  f"MAE {m_champ['mae']:.3f} / biais {m_champ['bias']:+.3f} sur {len(ids)} tournois")
        mlflow.set_tags({"registered_version": version, "promoted": str(promoted).lower(), "promotion_reason": reason})
    return {"version": version, "promoted": promoted, "reason": reason, "run_id": run_id,
            "challenger": m_ch, "champion": m_champ, "fit_until": str(start.date())}


# --- exécution -------------------------------------------------------------------------------

def parse_ts(raw: str | None) -> pd.Timestamp | None:
    if not raw:
        return None
    ts = pd.Timestamp(raw)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def log_operation(decision: Decision, outcome: dict) -> str:
    setup_mlflow(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI), experiment=OPERATIONS)
    with mlflow.start_run(run_name=f"trigger {decision.action} as-of {decision.as_of}") as run:
        mlflow.set_tags({"operation": "trigger", "action": decision.action, "as_of": decision.as_of,
                         "champion": decision.champion, "reason": decision.reason[:5000],
                         **{f"outcome_{k}": str(v) for k, v in outcome.items() if not isinstance(v, dict)}})
        for side in ("challenger", "champion"):
            if isinstance(outcome.get(side), dict):
                mlflow.log_metrics({f"{side}_{k}": v for k, v in outcome[side].items()})
        return run.info.run_id


def run_trigger(as_of: pd.Timestamp | None = None, dry_run: bool = False, api_url: str | None = None,
                df: pd.DataFrame | None = None, data_path: Path | None = None) -> tuple[Decision, dict]:
    all_p = params()
    pm, pt, p_train = all_p["monitor"], all_p["trigger"], all_p["train"]
    data_path = data_path or FEATURES_PATH
    df = pd.read_parquet(data_path) if df is None else df
    as_of = as_of or df["event_start"].max()
    client = MlflowClient()
    champ = M.champion_info(client, p_train["registered_model_name"], pm["fallback_event_bias_sd"])
    promoted_at = parse_ts(client.get_model_version(champ.name, champ.version).tags.get("promoted_at"))

    setup_mlflow(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI), experiment=M.EXPERIMENT)
    timeline = monitoring_as_of(df, as_of, champ, pm, pt["consecutive_checks"])
    decision = decide(timeline, as_of, promoted_at, champ.version, pm, pt)

    outcome: dict = {"executed": False}
    if not dry_run and decision.action == "retrain":
        setup_mlflow(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))
        outcome = {"executed": True} | retrain(df, as_of, champ, p_train, pm, pt, data_path)
    elif not dry_run and decision.action == "rollback":
        res = R.rollback(client, champ.name, reason=decision.reason[:500])
        outcome = {"executed": True, "from_version": res.from_version, "to_version": res.to_version}
        if api_url:
            outcome["api_version"] = R.reload_api(api_url, res.to_version)["model_version"]
    outcome["operation_run_id"] = log_operation(decision, outcome)

    TRIGGER_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"decision": asdict(decision), "outcome": outcome}, indent=2, ensure_ascii=False, default=str) + "\n"
    (TRIGGER_DIR / "latest.json").write_text(text)
    if outcome.get("executed"):  # archive des décisions réellement exécutées
        (TRIGGER_DIR / f"{decision.as_of}_{decision.action}.json").write_text(text)
    return decision, outcome


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Déclencheur réentraînement / rollback")
    ap.add_argument("--as-of", help="date de la décision (AAAA-MM-JJ) ; défaut : dernier tournoi connu")
    ap.add_argument("--dry-run", action="store_true", help="décider sans agir")
    ap.add_argument("--api-url", help="API à recharger après un rollback")
    args = ap.parse_args(argv)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))
    as_of = pd.Timestamp(args.as_of, tz="UTC") + pd.Timedelta(hours=23, minutes=59) if args.as_of else None
    decision, outcome = run_trigger(as_of, args.dry_run, args.api_url)
    print(f"[{decision.as_of}] champion v{decision.champion} -> action : {decision.action.upper()}"
          f"{' (simulation, rien exécuté)' if args.dry_run else ''}")
    print(f"  {decision.reason}")
    if outcome.get("version"):
        print(f"  challenger v{outcome['version']} : {'PROMU champion' if outcome['promoted'] else 'non promu'} — {outcome['reason']}")
    if outcome.get("to_version"):
        print(f"  rollback v{outcome['from_version']} -> v{outcome['to_version']}")
    return CODES[decision.action]


if __name__ == "__main__":
    sys.exit(main())
