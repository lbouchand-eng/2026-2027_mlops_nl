"""Monitoring du drift (méthode et seuils : docs/monitoring.md).

Deux familles de signaux, calculées sur une fenêtre glissante des N derniers tournois :
- drift des données : PSI de chaque feature surveillée vs les données d'entraînement du
  modèle servi (disponible dès la prédiction, sans attendre les scores) ;
- drift de performance : biais moyen par tournoi rapporté à son bruit de référence
  (z-score) et MAE rapportée à la MAE de validation du champion (scores nécessaires).
Les seuils de performance sont lus dans les tags de la version **champion** du Registry.

Commandes :
  uv run python -m pga.monitor replay --years 2023 2024 [--inject fc_wind_mean+15 --since 2024-03-01] [--name incident]
  uv run python -m pga.monitor check          # journal de l'API + scores réels
  uv run python -m pga.monitor calibrate      # distribution des signaux sur 2022 (justification des seuils)
  uv run python -m pga.monitor backfill-tags  # pose valid_event_bias_sd sur les versions anciennes
Code de sortie : 0 ok, 1 warning, 2 alert.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from pga.config import FEATURES_PATH, MLFLOW_URI, PREDICTIONS_LOG, REPORTS, ROOT, ROUNDS, params
from pga.features import FEATURES, TARGET
from pga.profile import PROFILE_ARTIFACT, make_edges, shares

EXPERIMENT = "pga-monitoring"
MONITORING = Path(os.environ.get("PGA_MONITORING_DIR", REPORTS / "monitoring"))  # surchargeable (démo)
LEVELS = {"ok": 0, "warning": 1, "alert": 2}
KEYS = ["event_id", "player_id", "round"]


# --- signaux ------------------------------------------------------------------------------

def psi_from_shares(pr: np.ndarray | None, cur, edges: np.ndarray) -> float:
    """PSI de `cur` par rapport à une référence déjà résumée (NaN si l'un des deux est vide)."""
    cur = np.asarray(cur, dtype=float)
    if pr is None or np.isnan(cur).all():
        return float("nan")
    pc = shares(cur, edges)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def psi(ref, cur, bins: int = 10, edges: np.ndarray | None = None) -> float:
    """Population Stability Index de `cur` par rapport à `ref` (NaN ignorés)."""
    ref = np.asarray(ref, dtype=float)
    if np.isnan(ref).all():
        return float("nan")
    edges = make_edges(ref, bins) if edges is None else edges
    return psi_from_shares(shares(ref, edges), cur, edges)


@dataclass
class Reference:
    """Distribution de référence (bins + parts par feature), issue de données ou d'un profil."""

    data: pd.DataFrame | None
    features: list[str]
    bins: int
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    profile: dict | None = None
    edges: dict[str, np.ndarray] = field(init=False)
    ref_shares: dict[str, np.ndarray | None] = field(init=False)

    def __post_init__(self):
        if self.profile is not None:
            known = self.profile["features"]
            self.features = [f for f in self.features if f in known]
            self.edges = {f: np.asarray(known[f]["edges"], dtype=float) for f in self.features}
            self.ref_shares = {f: None if known[f]["shares"] is None else np.asarray(known[f]["shares"], dtype=float)
                               for f in self.features}
            return
        self.edges = {f: make_edges(self.data[f], self.bins) for f in self.features}
        self.ref_shares = {f: None if self.data[f].isna().all() else shares(self.data[f], self.edges[f])
                           for f in self.features}

    def drift(self, cur: pd.DataFrame) -> dict[str, float]:
        return {f: psi_from_shares(self.ref_shares[f], cur[f], self.edges[f]) for f in self.features}


def perf_drift(window: pd.DataFrame, ref_mae: float, ref_bias_sd: float) -> dict[str, float] | None:
    """MAE, biais et z-score du biais sur les lignes dont le score est connu.

    Le biais est d'abord moyenné par tournoi (unité indépendante), puis entre tournois :
    z = biais / (écart-type de référence du biais par tournoi / √n_tournois).
    """
    w = window.dropna(subset=[TARGET, "prediction"])
    if w.empty:
        return None
    err = w["prediction"] - w[TARGET]
    by_event = err.groupby(w["event_id"]).mean()
    n = len(by_event)
    bias = float(by_event.mean())
    mae = float(err.abs().mean())
    return {
        "mae": mae,
        "mae_ratio": mae / ref_mae,
        "bias": bias,
        "bias_z": bias / (ref_bias_sd / np.sqrt(n)),
        "n_events": float(n),
        "n_rows": float(len(w)),
    }


def status(drift: dict[str, float], perf: dict[str, float] | None, p: dict,
           thresholds: dict[str, dict[str, float]] | None = None) -> tuple[str, list[str]]:
    """Statut global (le pire des signaux) et raisons lisibles.

    `thresholds` : seuils PSI par feature (calibration) ; à défaut, seuils globaux de `p`.
    """
    thresholds = thresholds or {}
    level, reasons = "ok", []

    def raise_to(new: str, why: str):
        nonlocal level
        reasons.append(f"[{new}] {why}")
        if LEVELS[new] > LEVELS[level]:
            level = new

    for f, v in sorted(drift.items(), key=lambda kv: -np.nan_to_num(kv[1])):
        if np.isnan(v):
            continue
        t = thresholds.get(f, {"warning": p["psi_warning"], "alert": p["psi_alert"]})
        if v > t["alert"]:
            raise_to("alert", f"PSI {f} = {v:.2f} > {t['alert']:.2f}")
        elif v > t["warning"]:
            raise_to("warning", f"PSI {f} = {v:.2f} > {t['warning']:.2f}")
    if perf:
        z = abs(perf["bias_z"])
        if z > p["bias_z_alert"]:
            raise_to("alert", f"biais {perf['bias']:+.2f} coup (z = {perf['bias_z']:+.1f}, seuil {p['bias_z_alert']})")
        elif z > p["bias_z_warning"]:
            raise_to("warning", f"biais {perf['bias']:+.2f} coup (z = {perf['bias_z']:+.1f}, seuil {p['bias_z_warning']})")
        if perf["mae_ratio"] > p["mae_ratio_alert"]:
            raise_to("alert", f"MAE {perf['mae']:.2f} = {perf['mae_ratio']:.2f} x validation (seuil {p['mae_ratio_alert']})")
    return level, reasons


INJECT_RE = re.compile(r"^(\w+)\s*([+*-])\s*(-?\d+(?:\.\d+)?)$")


def inject(df: pd.DataFrame, spec: str, since: str | None = None) -> pd.DataFrame:
    """Perturbe une feature (`col+val`, `col-val`, `col*val`) à partir de `since` (incident simulé)."""
    m = INJECT_RE.match(spec.strip())
    if not m or m.group(1) not in df:
        raise ValueError(f"injection invalide : {spec!r} (attendu : <feature>+|-|*<nombre>)")
    col, op, val = m.group(1), m.group(2), float(m.group(3))
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    if since:
        mask = out["event_start"] >= pd.Timestamp(since, tz="UTC")
    out.loc[mask, col] = {"+": out.loc[mask, col] + val, "-": out.loc[mask, col] - val, "*": out.loc[mask, col] * val}[op]
    return out


# --- registry ------------------------------------------------------------------------------

@dataclass
class Champion:
    name: str
    version: str
    valid_mae: float
    event_bias_sd: float
    sd_source: str  # "tag" ou "défaut"
    profile: dict | None = None  # données d'entraînement résumées ; None sur les anciennes versions


def load_profile(run_id: str) -> dict | None:
    try:
        return mlflow.artifacts.load_dict(f"runs:/{run_id}/{PROFILE_ARTIFACT}")
    except Exception:  # noqa: BLE001 — version antérieure aux profils : repli sur la période fixe
        return None


def champion_info(client: MlflowClient, name: str, fallback_sd: float) -> Champion:
    mv = client.get_model_version_by_alias(name, "champion")
    sd = mv.tags.get("valid_event_bias_sd")
    return Champion(
        name=name,
        version=str(mv.version),
        valid_mae=float(mv.tags["valid_mae"]),
        event_bias_sd=float(sd) if sd else fallback_sd,
        sd_source="tag" if sd else "défaut",
        profile=load_profile(mv.run_id),
    )


def predict_champion(champ: Champion, df: pd.DataFrame) -> np.ndarray:
    model = mlflow.pyfunc.load_model(f"models:/{champ.name}/{champ.version}")
    return np.asarray(model.predict(df[FEATURES].astype(float)), dtype=float)


# --- fenêtres glissantes ----------------------------------------------------------------------

def rolling_checks(cur: pd.DataFrame, ref: Reference, champ: Champion, p: dict, emit_from=None) -> pd.DataFrame:
    """Un contrôle après chaque tournoi, sur les `window_events` derniers tournois.

    `cur` : lignes avec event_id, event_start, features, prediction et (si connu) to_par.
    Les tournois antérieurs à `emit_from` ne servent qu'à remplir la première fenêtre
    (pas de contrôle émis) ; une fenêtre incomplète n'est jamais contrôlée, sauf si `cur`
    contient moins de `window_events` tournois (mode check sur un journal court).
    """
    order = cur.groupby("event_id")["event_start"].min().sort_values()
    n_win = min(p["window_events"], len(order))
    rows = []
    for i, (eid, start) in enumerate(order.items()):
        if i + 1 < n_win or (emit_from is not None and start < emit_from):
            continue
        window_ids = order.index[i + 1 - n_win : i + 1]
        w = cur[cur["event_id"].isin(window_ids)]
        # Trop peu de lignes (journal API court, scores pas encore connus) : signaux non
        # interprétables, on ne les évalue pas plutôt que de lever de fausses alertes.
        min_rows = p.get("min_rows", 0)
        enough = len(w) >= min_rows
        drift = ref.drift(w) if enough else {f: float("nan") for f in ref.features}
        perf = perf_drift(w, champ.valid_mae, champ.event_bias_sd)
        n_scored = int(perf["n_rows"]) if perf else 0
        if perf and n_scored < min_rows:
            perf = None
        level, reasons = status(drift, perf, p, ref.thresholds)
        if not enough:
            reasons.append(f"[info] PSI non évalué : {len(w)} lignes < {min_rows}")
        if n_scored < min_rows:
            reasons.append(f"[info] performance non évaluée : {n_scored} lignes notées < {min_rows}")
        alert_of = lambda f: ref.thresholds.get(f, {"alert": p["psi_alert"]})["alert"]  # noqa: E731
        worst = max(drift, key=lambda f: np.nan_to_num(drift[f] / alert_of(f), nan=-1))
        rows.append(
            {
                "step": len(rows),
                "event_id": eid,
                "event_name": cur.loc[cur["event_id"] == eid, "event_name"].iloc[0] if "event_name" in cur else eid,
                "event_start": start,
                "status": level,
                "psi_max": drift[worst],
                "psi_max_feature": worst,
                "psi_ratio_max": max(np.nan_to_num(v / ref.thresholds.get(f, {"alert": p["psi_alert"]})["alert"])
                                     for f, v in drift.items()),
                **{f"psi_{f}": v for f, v in drift.items()},
                **(perf or {}),
                "reasons": " | ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


# --- sorties ---------------------------------------------------------------------------------

def setup_mlflow() -> None:
    from pga.train import setup_mlflow as _setup

    _setup(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI), experiment=EXPERIMENT)


def log_timeline(timeline: pd.DataFrame, champ: Champion, tags: dict, p: dict) -> str:
    with mlflow.start_run(run_name=tags.get("mode")) as run:
        mlflow.log_params({k: v for k, v in p.items() if not isinstance(v, list)})
        mlflow.set_tags({**tags, "model_version": champ.version, "event_bias_sd_source": champ.sd_source})
        for r in timeline.itertuples():
            metrics = {"status_level": LEVELS[r.status], "psi_max": r.psi_max, "psi_ratio_max": r.psi_ratio_max}
            for k in ("bias", "bias_z", "mae", "mae_ratio"):
                v = getattr(r, k, np.nan)
                if pd.notna(v):
                    metrics[k] = v
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)}, step=int(r.step))
        last = timeline.iloc[-1]
        mlflow.set_tags({"status": last.status, "n_warning": int((timeline.status == "warning").sum()),
                         "n_alert": int((timeline.status == "alert").sum())})
        return run.info.run_id


def write_latest(timeline: pd.DataFrame, champ: Champion, extra: dict, out: Path | None = None) -> dict:
    out = out or MONITORING
    last = timeline.iloc[-1]
    alerts = timeline[timeline.status == "alert"]
    warnings = timeline[timeline.status == "warning"]
    latest = {
        "status": last.status,
        "reasons": [r for r in str(last.reasons).split(" | ") if r and r != "nan"],
        "last_event": {"event_id": last.event_id, "event_name": last.event_name, "event_start": str(last.event_start)},
        "model": {"name": champ.name, "version": champ.version, "valid_mae": champ.valid_mae,
                  "event_bias_sd": champ.event_bias_sd, "event_bias_sd_source": champ.sd_source},
        "signals": {k: (None if pd.isna(last.get(k)) else round(float(last.get(k)), 4))
                    for k in ("psi_max", "psi_ratio_max", "bias", "bias_z", "mae", "mae_ratio", "n_events")},
        "psi_max_feature": last.psi_max_feature,
        "history": {"n_checks": len(timeline), "n_warning": len(warnings), "n_alert": len(alerts),
                    "first_warning": str(warnings.event_start.min()) if len(warnings) else None,
                    "first_alert": str(alerts.event_start.min()) if len(alerts) else None},
        **extra,
    }
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(latest, indent=2, ensure_ascii=False, default=str) + "\n"
    (out / "latest.json").write_text(text)  # dernier contrôle, lu par le déclencheur
    if extra.get("label"):
        (out / f"summary_{extra['label']}.json").write_text(text)  # trace du scénario
    return latest


def plot_timeline(timeline: pd.DataFrame, p: dict, path: Path, title: str, since: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = pd.to_datetime(timeline["event_start"])
    colors = {"ok": "#ffffff", "warning": "#f6d58b", "alert": "#ee8f8f"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    panels = [
        ("bias_z", "z-score du biais", [(p["bias_z_warning"], "--"), (p["bias_z_alert"], "-"),
                                        (-p["bias_z_warning"], "--"), (-p["bias_z_alert"], "-")]),
        ("mae_ratio", "MAE / MAE validation", [(p["mae_ratio_alert"], "-")]),
        ("psi_ratio_max", "PSI / seuil d'alerte (max)", [(1.0, "-")]),
    ]
    for ax, (col, label, lines) in zip(axes, panels, strict=True):
        for k in range(len(t)):
            left = t.iloc[k]
            right = t.iloc[k + 1] if k + 1 < len(t) else left + pd.Timedelta(days=7)
            ax.axvspan(left, right, color=colors[timeline["status"].iloc[k]], alpha=0.6, lw=0)
        if col in timeline:
            ax.plot(t, timeline[col], marker="o", ms=3, color="#1f3b73")
        for y, style in lines:
            ax.axhline(y, color="#9b1c1c", ls=style, lw=1)
        if since:
            ax.axvline(pd.Timestamp(since, tz="UTC"), color="black", lw=1.2, ls=":")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    if timeline["psi_ratio_max"].max() > 3:  # injection : échelle log pour garder les seuils lisibles
        axes[2].set_yscale("log")
    axes[0].set_title(title)
    fig.text(0.01, 0.005, "fond : blanc = ok, jaune = warning, rouge = alert ; trait = seuil d'alerte, tirets = warning"
             + (" ; pointillé = début de l'injection" if since else ""), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- commandes -------------------------------------------------------------------------------

CALIBRATION = REPORTS / "monitoring" / "calibration.json"


def load_thresholds(path: Path | None = None) -> dict[str, dict[str, float]]:
    path = path or CALIBRATION
    if not path.exists():
        print(f"  ! pas de calibration ({path.name}) : seuils PSI globaux", file=sys.stderr)
        return {}
    return {f: {"warning": v["warning"], "alert": v["alert"]} for f, v in json.loads(path.read_text()).items()}


def calibrated_thresholds(dist: pd.DataFrame, p: dict) -> pd.DataFrame:
    """warning = max(seuil global, quantile) ; alert = max(seuil global, marge x max observé)."""
    out = dist.copy()
    out["warning"] = np.maximum(p["psi_warning"], dist[f"{int(p['calibration_quantile'] * 100)}%"])
    out["alert"] = np.maximum(p["psi_alert"], p["calibration_alert_margin"] * dist["max"])
    return out.round(3)


def period_reference(df: pd.DataFrame, p: dict) -> Reference:
    """Référence sur la période fixe de params.yaml."""
    lo, hi = p["reference_years"]
    return Reference(df[df["year"].between(lo, hi)], p["psi_features"], p["psi_bins"], load_thresholds())


def champion_reference(champ: Champion, df: pd.DataFrame, p: dict) -> Reference:
    """Les données d'entraînement du modèle servi, ou la période fixe s'il n'a pas de profil."""
    if champ.profile is None:
        return period_reference(df, p)
    return Reference(None, p["psi_features"], p["psi_bins"], load_thresholds(), profile=champ.profile)


def load_reference(p: dict, champ: Champion, features_path: Path | None = None) -> tuple[pd.DataFrame, Reference]:
    df = pd.read_parquet(features_path or FEATURES_PATH)
    return df, champion_reference(champ, df, p)


def cmd_replay(args, p: dict, name: str) -> int:
    champ = champion_info(MlflowClient(), name, p["fallback_event_bias_sd"])
    df, ref = load_reference(p, champ)
    y0, y1 = args.years
    emit_from = pd.Timestamp(f"{y0}-01-01", tz="UTC")
    # Préremplissage : les derniers tournois de l'année précédente complètent la 1re fenêtre.
    before = df[df["year"] == y0 - 1]
    warmup = before.groupby("event_id")["event_start"].min().nlargest(p["window_events"] - 1).index
    cur = pd.concat([before[before["event_id"].isin(warmup)], df[df["year"].between(y0, y1)]]).copy()
    if args.inject:
        cur = inject(cur, args.inject, args.since or str(emit_from.date()))
    cur["prediction"] = predict_champion(champ, cur)
    timeline = rolling_checks(cur, ref, champ, p, emit_from=emit_from)

    label = args.name or ("incident" if args.inject else "replay")
    MONITORING.mkdir(parents=True, exist_ok=True)
    timeline.to_csv(MONITORING / f"timeline_{label}.csv", index=False)
    title = f"Monitoring {args.years[0]}-{args.years[1]} — modèle v{champ.version}"
    if args.inject:
        title += f" — injection {args.inject} dès {args.since}"
    plot_timeline(timeline, p, MONITORING / f"timeline_{label}.png", title, args.since if args.inject else None)
    tags = {"mode": f"replay:{label}", "years": f"{args.years[0]}-{args.years[1]}", "inject": args.inject or "none",
            "inject_since": args.since or "none"}
    run_id = log_timeline(timeline, champ, tags, p)
    latest = write_latest(timeline, champ, {"mode": tags["mode"], "label": label, "inject": args.inject, "inject_since": args.since,
                                            "mlflow_run_id": run_id, "timeline": f"reports/monitoring/timeline_{label}.png"})
    report(latest)
    return LEVELS[latest["status"]]


def load_log(path: Path, rounds_path: Path | None = None) -> pd.DataFrame:
    rounds_path = rounds_path or ROUNDS
    """Journal de l'API + score réel quand le tour est joué (sinon to_par = NaN)."""
    if not path.exists():
        raise FileNotFoundError(f"journal des prédictions absent : {path} (lancer l'API et appeler /predict)")
    log = pd.read_parquet(path).sort_values("ts").drop_duplicates(KEYS, keep="last")
    log["round"] = log["round"].astype(int)
    if rounds_path.exists():
        r = pd.read_parquet(rounds_path, columns=KEYS + [TARGET, "event_start", "event_name"])
        log = log.merge(r, on=KEYS, how="left")
    else:
        log[TARGET] = np.nan
    if "event_start" not in log or log["event_start"].isna().all():
        log["event_start"] = log.groupby("event_id")["ts"].transform("min")
    log["event_start"] = log["event_start"].fillna(log.groupby("event_id")["ts"].transform("min"))
    return log


def cmd_check(args, p: dict, name: str) -> int:
    champ = champion_info(MlflowClient(), name, p["fallback_event_bias_sd"])
    _, ref = load_reference(p, champ)
    log = load_log(Path(os.environ.get("PREDICTIONS_LOG", PREDICTIONS_LOG)))
    timeline = rolling_checks(log, ref, champ, p).tail(1)
    run_id = log_timeline(timeline.assign(step=0), champ, {"mode": "check"}, p)
    labelled = int(log[TARGET].notna().sum())
    latest = write_latest(timeline, champ, {"mode": "check", "mlflow_run_id": run_id,
                                            "log_rows": len(log), "log_rows_with_score": labelled})
    report(latest)
    return LEVELS[latest["status"]]


def cmd_calibrate(args, p: dict, name: str) -> int:
    """Distribution naturelle des PSI sur la période de référence → seuils par feature.

    Fenêtres glissantes de `window_events` tournois consécutifs sur toute la référence
    (toutes saisons, fenêtres à cheval sur deux années comprises) ; chaque fenêtre est
    comparée à la référence privée de ses propres années (pas d'auto-comparaison).
    """
    df = pd.read_parquet(FEATURES_PATH)
    lo, hi = p["reference_years"]
    refdf = df[df["year"].between(lo, hi)]
    order = refdf.groupby("event_id")["event_start"].min().sort_values()
    years = refdf.groupby("event_id")["year"].first()
    refs: dict[frozenset, Reference] = {}
    rows = []
    for i in range(p["window_events"] - 1, len(order)):
        ids = order.index[i + 1 - p["window_events"] : i + 1]
        excl = frozenset(years[ids])
        if excl not in refs:
            refs[excl] = Reference(refdf[~refdf["year"].isin(excl)], p["psi_features"], p["psi_bins"])
        rows.append(refs[excl].drift(refdf[refdf["event_id"].isin(ids)]))
    q = p["calibration_quantile"]
    dist = pd.DataFrame(rows).describe(percentiles=[0.5, q]).T[["50%", f"{int(q * 100)}%", "max"]]
    table = calibrated_thresholds(dist, p)
    print(f"PSI naturel sur {len(rows)} fenêtres de {p['window_events']} tournois ({lo}-{hi}, référence hors années de la fenêtre) :")
    print(table.to_string())
    MONITORING.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(table.to_json(orient="index", indent=2) + "\n")
    print(f"-> {CALIBRATION}")
    return 0


def cmd_backfill(args, p: dict, name: str) -> int:
    """Calcule valid_event_bias_sd pour les versions qui ne l'ont pas (si mêmes données)."""
    from pga.train import data_hash, event_bias_sd, make_model, select, to_xy

    client = MlflowClient()
    df = pd.read_parquet(FEATURES_PATH)
    current = data_hash(FEATURES_PATH)
    for mv in client.search_model_versions(f"name='{name}'"):
        if "valid_event_bias_sd" in mv.tags:
            print(f"v{mv.version} : déjà renseigné ({mv.tags['valid_event_bias_sd']})")
            continue
        run = client.get_run(mv.run_id)
        if run.data.tags.get("data_hash") != current:
            print(f"v{mv.version} : entraîné sur d'autres données ({run.data.tags.get('data_hash')}) -> ignoré")
            continue
        rp = run.data.params
        model_p = {k.removeprefix("model_"): yaml.safe_load(v) for k, v in rp.items() if k.startswith("model_")}
        years = {k: tuple(int(x) for x in rp[k].split("-")) for k in ("train_years", "valid_years")}
        m = make_model({"model": model_p, "random_state": int(rp["random_state"])}).fit(*to_xy(select(df, years["train_years"])))
        va = select(df, years["valid_years"])
        sd = event_bias_sd(va["event_id"], m.predict(to_xy(va)[0]) - va[TARGET].to_numpy())
        client.set_model_version_tag(name, mv.version, "valid_event_bias_sd", f"{sd:.6f}")
        print(f"v{mv.version} : valid_event_bias_sd = {sd:.4f}")
    return 0


def report(latest: dict) -> None:
    s = latest["signals"]
    print(f"statut : {latest['status'].upper()}  (modèle v{latest['model']['version']}, dernier tournoi : "
          f"{latest['last_event']['event_name']})")
    print(f"  PSI max {s['psi_max']} ({latest['psi_max_feature']}) | biais {s['bias']} (z {s['bias_z']}) | MAE ratio {s['mae_ratio']}")
    for r in latest["reasons"]:
        print(f"  - {r}")
    h = latest["history"]
    print(f"  historique : {h['n_checks']} contrôles, {h['n_warning']} warning, {h['n_alert']} alert "
          f"(1er warning {h['first_warning']}, 1re alerte {h['first_alert']})")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    all_p = params()
    p, name = all_p["monitor"], all_p["train"]["registered_model_name"]
    ap = argparse.ArgumentParser(description="Monitoring du drift du modèle servi")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay", help="rejoue une période tournoi par tournoi")
    rp.add_argument("--years", nargs=2, type=int, default=[2023, 2024])
    rp.add_argument("--inject", help="ex. fc_wind_mean+15 ou fc_wind_mean*3.6")
    rp.add_argument("--since", help="date de début de l'injection (AAAA-MM-JJ)")
    rp.add_argument("--name", help="suffixe des fichiers de sortie")
    sub.add_parser("check", help="contrôle sur le journal de l'API")
    sub.add_parser("calibrate", help="distribution des PSI en 2022")
    sub.add_parser("backfill-tags", help="renseigne valid_event_bias_sd sur les anciennes versions")
    args = ap.parse_args(argv)

    setup_mlflow()
    cmds = {"replay": cmd_replay, "check": cmd_check, "calibrate": cmd_calibrate, "backfill-tags": cmd_backfill}
    return cmds[args.cmd](args, p, name)


if __name__ == "__main__":
    sys.exit(main())
