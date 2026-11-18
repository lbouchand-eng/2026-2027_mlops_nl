"""Trigger: decides, at a given date, between doing nothing, retraining or rolling back.

Monitoring is recomputed "as-of" (only with the tournaments finished at that date)
over the last `consecutive_checks` checks, then:

| Finding                                                       | Action          | Code |
|---------------------------------------------------------------|-----------------|------|
| nothing abnormal                                              | nothing         | 0    |
| degraded bias/MAE, champion deployed long ago                 | retraining      | 10   |
| PSI alert on an input                                         | data incident: no retraining (it would learn the bug) | 20 |
| degraded bias/MAE, champion deployed recently                 | rollback        | 30   |

"Degraded": alert on the last check, or warning/alert on all of the last checks.

Leak-free, comparable retraining: the decision window is made of the last
`window_events` tournaments before the date; the challenger is trained on everything
before it; **champion and challenger are evaluated on that same window**. The challenger
becomes champion if its MAE is better and its |bias| is not worse.

This window is read from the **API prediction journal** as soon as the journal covers it: these
are the inputs actually seen in production, which a backfill cannot rewrite. Otherwise, fall back
to the feature table, recorded by the `decision_source` tag.

Usage: uv run python -m pga.trigger [--as-of 2024-02-08] [--dry-run] [--api-url http://127.0.0.1:8000]
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
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from pga import monitor as M
from pga import rollback as R
from pga.alerting import notify
from pga.config import FEATURES_PATH, MLFLOW_URI, PREDICTIONS_LOG, REPORTS, ROOT, params
from pga.features import TARGET
from pga.train import baseline, evaluate, log_model_run, make_model, model_params, set_champion, setup_mlflow, to_xy
from pga.weather import wind_median_problem

OPERATIONS = "pga-operations"
CODES = {"none": 0, "retrain": 10, "data_incident": 20, "rollback": 30}
TRIGGER_DIR = Path(os.environ.get("PGA_TRIGGER_DIR", REPORTS / "trigger"))  # overridable (demo)
ROUND_HOURS = 6   # a round lasts ≈ 5 h: score known ≈ 6 h after the last tee time
EVENT_DAYS = 4    # end of a tournament without tee times: start + 4 days


@dataclass
class Decision:
    action: str
    reason: str
    as_of: str
    champion: str
    last_status: str
    perf_levels: list[str]


# --- decision -------------------------------------------------------------------------------

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
    recent = promoted_at is not None and 0 <= (as_of - promoted_at).days <= pt["recent_promotion_days"]
    if psi_alert(last):
        if recent:  # the last deployment is suspect: its training data differs from production
            return Decision("rollback", f"PSI alert right after the deployment of v{champion} "
                                        f"({promoted_at.date()}): {last['reasons']}", **base)
        return Decision("data_incident", f"PSI alert: {last['reasons']}", **base)
    degraded = levels[-1] == "alert" or (len(levels) >= pt["consecutive_checks"] and "ok" not in levels)
    if not degraded:
        return Decision("none", "performance within thresholds", **base)
    detail = f"bias {last['bias']:+.2f} strokes (z {last['bias_z']:+.1f}), MAE ratio {last['mae_ratio']:.2f}, checks {levels}"
    if recent:
        return Decision("rollback", f"degradation right after the deployment of v{champion} "
                                    f"({promoted_at.date()}): {detail}", **base)
    return Decision("retrain", f"drift confirmed: {detail}", **base)


# --- monitoring as-of ---------------------------------------------------------------------------

def event_end(df: pd.DataFrame) -> pd.Series:
    """End of each tournament: last tee time + duration of a round (start + 4 days if tee times unknown)."""
    start = df.groupby("event_id")["event_start"].min()
    fallback = start + pd.Timedelta(days=EVENT_DAYS)
    if "tee_time_utc" not in df:
        return fallback
    return (df.groupby("event_id")["tee_time_utc"].max() + pd.Timedelta(hours=ROUND_HOURS)).fillna(fallback)


def event_order(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Tournaments **finished** at `as_of`, by start date: a tournament in progress would bring
    in the scores of rounds not yet played (leak)."""
    start = df.groupby("event_id")["event_start"].min()
    return start[event_end(df) <= as_of].sort_values()


def monitoring_as_of(df: pd.DataFrame, as_of: pd.Timestamp, champ: M.Champion, pm: dict, k: int) -> pd.DataFrame:
    """The last `k` monitoring checks using only the tournaments played before `as_of`."""
    order = event_order(df, as_of)
    needed = order.index[-(pm["window_events"] + k - 1):]
    cur = df[df["event_id"].isin(needed)].copy()
    cur["prediction"] = M.predict_champion(champ, cur)
    return M.rolling_checks(cur, M.champion_reference(champ, df, pm), champ, pm, emit_from=order.iloc[-k])


# --- retraining -----------------------------------------------------------------------------

def decision_window(df: pd.DataFrame, as_of: pd.Timestamp, n_events: int) -> tuple[pd.Index, pd.Timestamp]:
    order = event_order(df, as_of)
    ids = order.index[-n_events:]
    return ids, order.loc[ids].min()


def load_journal() -> pd.DataFrame | None:
    path = Path(os.environ.get("PREDICTIONS_LOG", PREDICTIONS_LOG))
    return M.load_log(path) if path.exists() else None


def journal_window(journal: pd.DataFrame | None, ids: pd.Index, min_rows: int) -> pd.DataFrame | None:
    """Scored prediction journal rows over the decision window; None if it is not fully covered."""
    if journal is None:
        return None
    scored = journal[journal["event_id"].isin(ids) & journal[TARGET].notna()]
    return scored if scored["event_id"].nunique() == len(ids) and len(scored) >= min_rows else None


def promotion_blocks(fit: pd.DataFrame, ref: M.Reference, pm: dict, wind_range: list[float] | None) -> list[str]:
    """Reasons for refusing a challenger: wind unit, then deviation from the champion's training data."""
    blocks = []
    if wind_range and (problem := wind_median_problem(fit["fc_wind_mean"], wind_range)):
        blocks.append(problem)
    _, reasons = M.status(ref.drift(fit), None, pm, ref.thresholds)
    return blocks + [r.removeprefix("[alert] ") for r in reasons if r.startswith("[alert]")]


def retrain(df: pd.DataFrame, as_of: pd.Timestamp, champ: M.Champion, p_train: dict, pm: dict, pt: dict,
            data_path: Path, transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
            transform_label: str = "none", promote: bool = True, guards: bool = True,
            wind_range: list[float] | None = None, journal: pd.DataFrame | None = None) -> dict:
    """Trains a challenger on the data before the decision window and compares it to the
    champion on that window. `transform`: alteration of the training and decision data
    (backfill), used by the simulated incident.

    `guards`: refuses the promotion if the challenger's data deviates from the champion's
    (disabled only to replay the original incident scenario).

    `journal`: API prediction journal. If it covers the window, the comparison is made on it, without `transform`.
    """
    ids, start = decision_window(df, as_of, pm["window_events"])
    data = transform(df) if transform else df
    fit = data[(data["year"] >= pt["train_start_year"]) & (data["event_start"] < start)]
    logged = journal_window(journal, ids, pm["min_rows"])
    dec = data[data["event_id"].isin(ids)] if logged is None else logged
    source = "backfill" if logged is None else "journal"
    if fit.empty or dec.empty:
        raise ValueError(f"insufficient data as of {as_of.date()}: fit={len(fit)}, decision={len(dec)}")

    challenger = make_model(p_train).fit(*to_xy(fit))
    X_dec, y_dec = to_xy(dec)
    m_ch = evaluate(y_dec, challenger.predict(X_dec), baseline(dec), dec["round"])
    m_champ = evaluate(y_dec, M.predict_champion(champ, dec), baseline(dec), dec["round"])
    better = m_ch["mae"] < m_champ["mae"] and abs(m_ch["bias"]) <= abs(m_champ["bias"]) + pt["bias_tolerance"]
    blocks = promotion_blocks(fit, M.champion_reference(champ, df, pm), pm, wind_range) if guards else []

    name = p_train["registered_model_name"]
    metrics = {f"decision_{k}": v for k, v in m_ch.items()} | {f"decision_champion_{k}": v for k, v in m_champ.items()}
    params_ = model_params(p_train) | {"fit_from": str(fit["event_start"].min().date()), "fit_until": str(start.date()),
                                        "decision_events": len(ids), "as_of": str(as_of.date())}
    tags = {"trigger": "retrain", "as_of": str(as_of.date()), "compared_to": f"v{champ.version}",
            "data_transform": transform_label, "decision_source": source}
    with mlflow.start_run(run_name=f"retrain as-of {as_of.date()}"):
        run_id, version = log_model_run(challenger, X_dec.head(5), params_, metrics, data_path, name, True, fit, tags)
        client = MlflowClient()
        # Monitoring thresholds of the new version: MAE measured on the decision window,
        # per-tournament bias noise inherited from the champion (a property of the problem, not the model).
        for k, v in {"valid_mae": m_ch["mae"], "valid_skill": m_ch["skill"],
                     "valid_event_bias_sd": champ.event_bias_sd, "as_of": str(as_of.date()),
                     "data_transform": transform_label}.items():
            client.set_model_version_tag(name, version, k, f"{v:.6f}" if isinstance(v, float) else str(v))
        promoted = bool(better and promote and not blocks)
        if promoted:
            set_champion(client, name, version, promoted_at=str(as_of.date()))
        else:
            client.set_model_version_tag(name, version, "status", "candidate")
        if blocks:
            client.set_model_version_tag(name, version, "blocked_reason", "; ".join(blocks)[:500])
        reason = (f"challenger MAE {m_ch['mae']:.3f} / bias {m_ch['bias']:+.3f} vs champion v{champ.version} "
                  f"MAE {m_champ['mae']:.3f} / bias {m_champ['bias']:+.3f} on {len(ids)} tournaments")
        if logged is not None:
            reason += " from the API prediction journal"
        if blocks:
            reason += " | promotion blocked: " + "; ".join(blocks)
        mlflow.set_tags({"registered_version": version, "promoted": str(promoted).lower(), "promotion_reason": reason})
    return {"version": version, "promoted": promoted, "reason": reason, "run_id": run_id, "blocked": blocks,
            "challenger": m_ch, "champion": m_champ, "fit_until": str(start.date()), "decision_source": source}


# --- execution -------------------------------------------------------------------------------

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
    as_of = as_of or event_end(df).max()
    client = MlflowClient()
    champ = M.champion_info(client, p_train["registered_model_name"], pm["fallback_event_bias_sd"])
    promoted_at = parse_ts(client.get_model_version(champ.name, champ.version).tags.get("promoted_at"))

    setup_mlflow(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI), experiment=M.EXPERIMENT)
    timeline = monitoring_as_of(df, as_of, champ, pm, pt["consecutive_checks"])
    decision = decide(timeline, as_of, promoted_at, champ.version, pm, pt)

    outcome: dict = {"executed": False}
    if not dry_run and decision.action == "retrain":
        setup_mlflow(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))
        wind_range = all_p.get("weather", {}).get("wind_median_kmh")
        outcome = {"executed": True} | retrain(df, as_of, champ, p_train, pm, pt, data_path, wind_range=wind_range,
                                               journal=load_journal())
    elif not dry_run and decision.action == "rollback":
        res = R.rollback(client, champ.name, reason=decision.reason[:500])
        outcome = {"executed": True, "from_version": res.from_version, "to_version": res.to_version}
        if api_url:
            outcome["api_version"] = R.reload_api(api_url, res.to_version)["model_version"]
    outcome["operation_run_id"] = log_operation(decision, outcome)

    TRIGGER_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"decision": asdict(decision), "outcome": outcome}, indent=2, ensure_ascii=False, default=str) + "\n"
    (TRIGGER_DIR / "latest.json").write_text(text)
    if outcome.get("executed"):  # archive of the decisions actually executed
        (TRIGGER_DIR / f"{decision.as_of}_{decision.action}.json").write_text(text)
    return decision, outcome


def describe(decision: Decision, outcome: dict, dry_run: bool) -> list[str]:
    lines = [f"[{decision.as_of}] champion v{decision.champion} -> action: {decision.action.upper()}"
             f"{' (dry run, nothing executed)' if dry_run else ''}", f"  {decision.reason}"]
    if outcome.get("version"):
        verdict = "PROMOTED to champion" if outcome["promoted"] else "not promoted"
        lines.append(f"  challenger v{outcome['version']}: {verdict} — {outcome['reason']}")
    if outcome.get("to_version"):
        lines.append(f"  rollback v{outcome['from_version']} -> v{outcome['to_version']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Retraining / rollback trigger")
    ap.add_argument("--as-of", help="decision date (YYYY-MM-DD); default: last known tournament")
    ap.add_argument("--dry-run", action="store_true", help="decide without acting")
    ap.add_argument("--api-url", help="API to reload after a rollback")
    args = ap.parse_args(argv)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))
    as_of = pd.Timestamp(args.as_of, tz="UTC") + pd.Timedelta(hours=23, minutes=59) if args.as_of else None
    decision, outcome = run_trigger(as_of, args.dry_run, args.api_url)
    text = "\n".join(describe(decision, outcome, args.dry_run))
    print(text)
    if decision.action != "none":
        notify(f"[PGA] trigger\n{text}")
    return CODES[decision.action]


if __name__ == "__main__":
    sys.exit(main())
