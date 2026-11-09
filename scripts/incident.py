"""Incident simulé : réentraînement sur un backfill météo corrompu (vent en m/s au lieu de km/h).

Scénario (rapport : docs/incident.md) :
  1. setup    : registry vierge, champion sain v1 (protocole standard, données 2018-2022).
  2. corrupt  : le 1er mars 2024, réentraînement de routine sur un backfill météo re-téléchargé
                avec `wind_speed_unit=ms` -> vent et rafales ÷ 3,6 sur tout l'historique. La
                fenêtre de décision est corrompue elle aussi : le challenger gagne la comparaison
                et devient champion (v2).
  3. detect   : en production (données saines), contrôle après chaque tournoi -> première date où
                le déclencheur réagit, et quelle action il choisit.
  4. rollback : le déclencheur exécute le rollback (champion <- v1) ; --api-url recharge l'API.
  5. recover  : monitoring de la même période avec v1 -> retour dans les seuils.
  Chaque étape écrit dans reports/incident/ ; `summary` assemble la chronologie.

Mode --guards (après correctifs, reports/incident_guarded/) : le même backfill est d'abord refusé
à la promotion ; on le force ensuite pour vérifier la seconde ligne de défense, le PSI calculé
sur les données d'entraînement du modèle servi. `retry` retente le backfill et imprime le refus.

Registry isolé (mlflow_incident.db, mlartifacts_incident/) : le scénario est rejouable sans
toucher au registry principal.

Usage : uv run python scripts/incident.py all [--guards] [--api-url http://127.0.0.1:8000]
        uv run python scripts/incident.py setup|corrupt|detect|rollback|recover|summary|retry
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "mlflow_incident.db"
ARTIFACTS = ROOT / "mlartifacts_incident"
OUT = ROOT / "reports" / "incident"
GUARDS = False
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{DB}"  # avant tout import de pga

import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from pga import monitor as M  # noqa: E402
from pga import rollback as R  # noqa: E402
from pga import train as T  # noqa: E402
from pga import trigger as TR  # noqa: E402
from pga.config import FEATURES_PATH, params  # noqa: E402

CORRUPT_AT = pd.Timestamp("2024-03-01 23:59", tz="UTC")
PROD_END = pd.Timestamp("2024-06-30", tz="UTC")
LABEL = "backfill vent en m/s (÷3,6) sur tout l'historique"


def corrupt(df: pd.DataFrame) -> pd.DataFrame:
    """Le backfill a été téléchargé avec wind_speed_unit=ms : vent et rafales ÷ 3,6."""
    out = df.copy()
    for col in ("fc_wind_mean", "fc_gust_max"):
        out[col] = out[col] / 3.6
    return out


def ctx():
    p = params()
    df = pd.read_parquet(FEATURES_PATH)
    return p, df, MlflowClient(), p["train"]["registered_model_name"]


def champion(p, client, name) -> M.Champion:
    return M.champion_info(client, name, p["monitor"]["fallback_event_bias_sd"])


def save(step: str, data: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{step}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"== {step} : " + json.dumps({k: v for k, v in data.items() if not isinstance(v, (dict, list))}, ensure_ascii=False, default=str))
    return data


def prod_timeline(df, champ, p, label, title, before: M.Champion | None = None):
    """Contrôle après chaque tournoi de production (mars -> juin 2024), fenêtre préremplie.

    Chaque tournoi est prédit par le modèle en service à sa date : `before` (s'il est donné)
    pour les tournois antérieurs au 1er mars, `champ` ensuite.
    """
    pm = p["monitor"]
    order = df.groupby("event_id")["event_start"].min().sort_values()
    events = order[(order > CORRUPT_AT) & (order <= PROD_END)]
    warm = order[order <= CORRUPT_AT].index[-(pm["window_events"] - 1):]
    cur = df[df["event_id"].isin(events.index.union(warm))].copy()
    cur["prediction"] = M.predict_champion(champ, cur)
    if before is not None:
        early = cur["event_id"].isin(warm)
        cur.loc[early, "prediction"] = M.predict_champion(before, cur[early])
    ref = M.champion_reference(champ, df, pm) if GUARDS else M.period_reference(df, pm)
    tl = M.rolling_checks(cur, ref, champ, pm, emit_from=events.iloc[0])
    OUT.mkdir(parents=True, exist_ok=True)
    tl.to_csv(OUT / f"timeline_{label}.csv", index=False)
    M.plot_timeline(tl, pm, OUT / f"timeline_{label}.png", title, str(CORRUPT_AT.date()))
    return tl


def step_setup(args):
    if DB.exists():
        DB.unlink()
    shutil.rmtree(ARTIFACTS, ignore_errors=True)
    T.setup_mlflow(os.environ["MLFLOW_TRACKING_URI"], ARTIFACTS)
    p, df, client, name = ctx()
    years = {k: p["train"][f"{k}_years"] for k in ("train", "valid", "refit")}
    res = T.run(df, p["train"], years, FEATURES_PATH, run_name="champion sain")
    client.set_model_version_tag(name, res["version"], "promoted_at", "2023-01-01")
    return save("1_setup", {"champion": res["version"], "valid_mae": res["metrics"]["valid_mae"],
                            "promoted_at": "2023-01-01 (mise en service simulée)"})


def attempt(p, df, champ, guards: bool) -> dict:
    return TR.retrain(df, CORRUPT_AT, champ, p["train"], p["monitor"], p["trigger"], FEATURES_PATH, transform=corrupt,
                      transform_label=LABEL, guards=guards, wind_range=p["weather"]["wind_median_kmh"])


def step_retry(args):
    T.setup_mlflow(os.environ["MLFLOW_TRACKING_URI"], ARTIFACTS)
    p, df, client, name = ctx()
    champ = champion(p, client, name)
    res = attempt(p, df, champ, guards=True)
    print(f"challenger v{res['version']} : {'PROMU' if res['promoted'] else 'REFUSÉ'}, le champion reste v{champ.version}")
    for b in res["blocked"]:
        print(f"  - {b}")
    return res


def step_corrupt(args):
    T.setup_mlflow(os.environ["MLFLOW_TRACKING_URI"], ARTIFACTS)
    p, df, client, name = ctx()
    champ = champion(p, client, name)
    if GUARDS:
        blocked = attempt(p, df, champ, guards=True)
        save("2_blocked", {"date": str(CORRUPT_AT.date()), "challenger": blocked["version"], "promoted": blocked["promoted"],
                           "reason": blocked["reason"], "blocked": blocked["blocked"]})
    res = attempt(p, df, champ, guards=False)
    return save("2_corrupt", {"date": str(CORRUPT_AT.date()), "challenger": res["version"], "promoted": res["promoted"],
                              "reason": res["reason"], "challenger_metrics": res["challenger"],
                              "champion_metrics": res["champion"], "data_transform": LABEL})


def step_detect(args):
    T.setup_mlflow(os.environ["MLFLOW_TRACKING_URI"], ARTIFACTS, experiment=M.EXPERIMENT)
    p, df, client, name = ctx()
    champ = champion(p, client, name)
    promoted_at = TR.parse_ts(client.get_model_version(name, champ.version).tags.get("promoted_at"))
    prev = client.get_model_version_by_alias(name, "previous")
    before = M.Champion(name, str(prev.version), float(prev.tags["valid_mae"]), champ.event_bias_sd, "tag")
    tl = prod_timeline(df, champ, p, "detection", f"Production mars-juin 2024 — v{champ.version} (réentraînée sur backfill corrompu)",
                       before=before)
    first = None
    for i in range(len(tl)):
        as_of = pd.Timestamp(tl.iloc[i]["event_start"]) + pd.Timedelta(days=4)  # lundi suivant le tournoi
        d = TR.decide(tl.iloc[: i + 1], as_of, promoted_at, champ.version, p["monitor"], p["trigger"])
        if d.action != "none":
            first = {"as_of": str(as_of.date()), "event": tl.iloc[i]["event_name"], "action": d.action,
                     "reason": d.reason, "tournaments_after_corruption": i + 1}
            break
    return save("3_detect", {"champion": champ.version, "first_reaction": first,
                             "checks": len(tl), "n_alert": int((tl.status == "alert").sum()),
                             "first_alert": str(tl[tl.status == "alert"].event_start.min()),
                             "psi_alerts": int(tl.reasons.str.contains(r"\[alert\] PSI").sum()),
                             "bias_window_max": round(float(tl.bias.max()), 3)})


def step_rollback(args):
    T.setup_mlflow(os.environ["MLFLOW_TRACKING_URI"], ARTIFACTS)
    p, df, client, name = ctx()
    det = json.loads((OUT / "3_detect.json").read_text())["first_reaction"]
    if not det or det["action"] != "rollback":
        raise SystemExit(f"le déclencheur n'a pas demandé de rollback : {det}")
    res = R.rollback(client, name, reason=f"incident {det['as_of']} : {det['reason'][:300]}")
    out = {"as_of": det["as_of"], "from": res.from_version, "to": res.to_version, "trace": R.log_operation(res)}
    if args.api_url:
        out["api_version"] = R.reload_api(args.api_url, res.to_version)["model_version"]
    return save("4_rollback", out)


def step_recover(args):
    T.setup_mlflow(os.environ["MLFLOW_TRACKING_URI"], ARTIFACTS, experiment=M.EXPERIMENT)
    p, df, client, name = ctx()
    champ = champion(p, client, name)
    tl = prod_timeline(df, champ, p, "apres_rollback", f"Même période après rollback — v{champ.version}")
    return save("5_recover", {"champion": champ.version, "checks": len(tl), "statuses": tl.status.value_counts().to_dict(),
                              "bias_window_max": round(float(tl.bias.max()), 3), "bias_z_max": round(float(tl.bias_z.max()), 2)})


def step_summary(args):
    steps = {f.stem: json.loads(f.read_text()) for f in sorted(OUT.glob("[1-5]_*.json"))}
    if not GUARDS:
        steps.pop("2_blocked", None)
    return save("summary", {"scenario": LABEL, "corrupted_at": str(CORRUPT_AT.date()), "guards": GUARDS, "steps": steps})


STEPS = {"setup": step_setup, "corrupt": step_corrupt, "detect": step_detect, "rollback": step_rollback,
         "recover": step_recover, "summary": step_summary}
ONCE = {"retry": step_retry}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=[*STEPS, *ONCE, "all"])
    ap.add_argument("--api-url")
    ap.add_argument("--guards", action="store_true", help="scénario après correctifs (reports/incident_guarded/)")
    args = ap.parse_args()
    global GUARDS, OUT
    if args.guards:
        GUARDS, OUT = True, ROOT / "reports" / "incident_guarded"
    for name in (STEPS if args.step == "all" else [args.step]):
        (STEPS | ONCE)[name](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
