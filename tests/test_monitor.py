"""Tests du monitoring : PSI, drift de performance, statut, injection, journal, replay."""
import json
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
from test_train import YEARS, P, make_features

from pga import monitor as M
from pga import train as T
from pga.features import FEATURES, TARGET

MP = {
    "reference_years": [2021, 2022],
    "window_events": 3,
    "min_rows": 20,
    "psi_bins": 5,
    "psi_warning": 0.10,
    "psi_alert": 0.25,
    "calibration_quantile": 0.99,
    "calibration_alert_margin": 1.5,
    "psi_features": ["form_ewm", "fc_wind_mean"],
    "bias_z_warning": 2.0,
    "bias_z_alert": 3.0,
    "mae_ratio_alert": 1.10,
    "fallback_event_bias_sd": 0.85,
}


# --- fonctions pures ----------------------------------------------------------------------

def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    assert M.psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.02


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    assert M.psi(rng.normal(size=5000), rng.normal(loc=1.0, size=5000)) > 0.25


def test_psi_handles_nan_and_constants():
    ref = np.array([1.0, 2.0, np.nan, 3.0, 4.0] * 100)
    assert M.psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(M.psi(ref, [np.nan, np.nan]))
    assert M.psi(np.ones(100), np.ones(50)) == pytest.approx(0.0, abs=1e-9)


def window(bias_per_event: list[float], n_per_event: int = 50) -> pd.DataFrame:
    rows = []
    for e, b in enumerate(bias_per_event):
        y = np.zeros(n_per_event)
        rows.append(pd.DataFrame({"event_id": f"E{e}", TARGET: y, "prediction": y + b}))
    return pd.concat(rows, ignore_index=True)


def test_perf_drift_z_score():
    perf = M.perf_drift(window([0.5, 0.5, 0.5, 0.5]), ref_mae=0.4, ref_bias_sd=0.5)
    assert perf["bias"] == pytest.approx(0.5)
    assert perf["bias_z"] == pytest.approx(0.5 / (0.5 / 2))  # √4 tournois
    assert perf["mae_ratio"] == pytest.approx(0.5 / 0.4)
    assert perf["n_events"] == 4


def test_perf_drift_ignores_rows_without_score():
    w = window([1.0, 1.0])
    w.loc[w.event_id == "E1", TARGET] = np.nan
    perf = M.perf_drift(w, ref_mae=1.0, ref_bias_sd=1.0)
    assert perf["n_events"] == 1
    assert M.perf_drift(w.assign(**{TARGET: np.nan}), 1.0, 1.0) is None


@pytest.mark.parametrize(
    "drift, perf, expected",
    [
        ({"a": 0.05}, None, "ok"),
        ({"a": 0.15}, None, "warning"),
        ({"a": 0.30}, None, "alert"),
        ({"a": 0.05}, {"bias": 0.5, "bias_z": 2.5, "mae": 1, "mae_ratio": 1.0}, "warning"),
        ({"a": 0.05}, {"bias": -0.9, "bias_z": -3.5, "mae": 1, "mae_ratio": 1.0}, "alert"),
        ({"a": 0.05}, {"bias": 0.0, "bias_z": 0.0, "mae": 2, "mae_ratio": 1.2}, "alert"),
    ],
)
def test_status_levels(drift, perf, expected):
    level, reasons = M.status(drift, perf, MP)
    assert level == expected
    assert (expected == "ok") == (reasons == [])


MPC = {**MP, "cusum_k": 0.5, "cusum_h_warning": 3.0, "cusum_h_alert": 4.0}


def test_cusum_accumulates_persistent_shift_only():
    up, down = M.cusum([0.4] * 20, k=0.5)  # en dessous de k : absorbé
    assert up.max() == 0 and down.max() == 0
    up, down = M.cusum([1.5] * 5, k=0.5)  # +1 par tournoi
    assert up.tolist() == [1, 2, 3, 4, 5] and down.max() == 0
    up, down = M.cusum([-1.5, np.nan, -1.5], k=0.5)  # tournoi sans score : sommes inchangées
    assert down.tolist() == [1, 1, 2] and up.max() == 0


def test_event_cusum_follows_event_order_and_scale():
    w = window([0.0, 0.0, 3.0, 3.0, 3.0, 3.0])  # biais de 3 coups, σ_tournoi = 2 -> z = 1,5
    order = pd.Series(range(6), index=[f"E{i}" for i in range(6)])
    cs = M.event_cusum(w, order, event_bias_sd=2.0, k=0.5)
    assert cs["cusum_up"].tolist() == pytest.approx([0, 0, 1, 2, 3, 4])
    reverse = M.event_cusum(w, order[::-1], event_bias_sd=2.0, k=0.5)  # retour au calme : -k par tournoi
    assert reverse["cusum_up"].tolist() == pytest.approx([1, 2, 3, 4, 3.5, 3])


@pytest.mark.parametrize("c, expected", [(2.0, "ok"), (3.5, "warning"), (4.5, "alert")])
def test_status_cusum_levels(c, expected):
    perf = {"bias": 0.0, "bias_z": 0.0, "mae": 1, "mae_ratio": 1.0, "cusum_up": 0.0, "cusum_down": c}
    level, reasons = M.status({}, perf, MPC)
    assert level == expected
    assert expected == "ok" or "trop basses" in reasons[0]


def test_status_uses_per_feature_thresholds():
    thresholds = {"fc_wind_mean": {"warning": 0.8, "alert": 1.5}}
    assert M.status({"fc_wind_mean": 0.5}, None, MP, thresholds)[0] == "ok"
    level, reasons = M.status({"fc_wind_mean": 2.0}, None, MP, thresholds)
    assert level == "alert" and "fc_wind_mean" in reasons[0]


def test_inject_only_target_column_after_since():
    df = pd.DataFrame({
        "event_start": pd.to_datetime(["2024-01-01", "2024-04-01"], utc=True),
        "fc_wind_mean": [10.0, 10.0],
        "fc_gust_max": [20.0, 20.0],
    })
    out = M.inject(df, "fc_wind_mean*3.6", "2024-03-01")
    assert out["fc_wind_mean"].tolist() == [10.0, 36.0]
    assert out["fc_gust_max"].tolist() == [20.0, 20.0]
    assert M.inject(df, "fc_wind_mean+15")["fc_wind_mean"].tolist() == [25.0, 25.0]
    with pytest.raises(ValueError):
        M.inject(df, "colonne_inconnue+1")


def test_calibrated_thresholds_respect_floors():
    dist = pd.DataFrame({"50%": [0.01, 0.5], "99%": [0.05, 0.9], "max": [0.06, 1.2]}, index=["a", "b"])
    t = M.calibrated_thresholds(dist, MP)
    assert t.loc["a", "warning"] == 0.10 and t.loc["a", "alert"] == 0.25
    assert t.loc["b", "warning"] == 0.9 and t.loc["b", "alert"] == pytest.approx(1.8)


# --- journal de l'API -------------------------------------------------------------------

def test_load_log_joins_scores(tmp_path):
    log = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"], utc=True),
        "event_id": ["E1", "E1", "E1"], "player_id": ["A", "B", "A"], "round": [1, 1, 1],
        "model_version": "2", "prediction": [0.0, 1.0, 0.5],
    })
    log.to_parquet(tmp_path / "log.parquet")
    rounds = pd.DataFrame({"event_id": ["E1"], "player_id": ["A"], "round": [1], TARGET: [-2],
                           "event_start": pd.to_datetime(["2024-01-01"], utc=True), "event_name": ["Open"]})
    rounds.to_parquet(tmp_path / "rounds.parquet")
    out = M.load_log(tmp_path / "log.parquet", tmp_path / "rounds.parquet").set_index("player_id")
    assert len(out) == 2  # doublon (E1, A, 1) : on garde la dernière prédiction
    assert out.loc["A", "prediction"] == 0.5 and out.loc["A", TARGET] == -2
    assert np.isnan(out.loc["B", TARGET])  # tour pas encore noté
    with pytest.raises(FileNotFoundError):
        M.load_log(tmp_path / "absent.parquet")


# --- bout en bout ----------------------------------------------------------------------------

def make_events(df: pd.DataFrame, per_event: int = 15) -> pd.DataFrame:
    df = df.copy()
    k = df.groupby("year").cumcount() // per_event
    df["event_id"] = df["year"].astype(str) + "-" + k.astype(str)
    df["event_name"] = "Open " + df["event_id"]
    df["event_start"] = pd.to_datetime(df["year"].astype(str) + "-01-01", utc=True) + pd.to_timedelta(k * 7, unit="D")
    df["fc_wind_mean"] = np.random.default_rng(1).gamma(4, 3, size=len(df))
    return df


@pytest.fixture
def setup(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    T.setup_mlflow(uri, tmp_path / "artifacts")
    df = make_events(make_features())
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    version = T.run(df, P, YEARS, path)["version"]
    out = tmp_path / "monitoring"
    monkeypatch.setattr(M, "FEATURES_PATH", path)
    monkeypatch.setattr(M, "MONITORING", out)
    monkeypatch.setattr(M, "CALIBRATION", out / "calibration.json")
    T.setup_mlflow(uri, tmp_path / "artifacts", experiment=M.EXPERIMENT)
    yield {"version": version, "out": out, "name": P["registered_model_name"]}
    mlflow.set_tracking_uri(None)


def test_champion_info_reads_tags(setup):
    champ = M.champion_info(mlflow.tracking.MlflowClient(), setup["name"], fallback_sd=0.85)
    assert champ.version == setup["version"]
    assert champ.sd_source == "tag" and champ.event_bias_sd > 0  # posé par l'entraînement


def test_replay_detects_injected_drift(setup):
    args = SimpleNamespace(years=[2023, 2023], inject=None, since=None, name="calme")
    M.cmd_calibrate(args, MP, setup["name"])
    assert (setup["out"] / "calibration.json").exists()

    code_calm = M.cmd_replay(args, MP, setup["name"])
    calm = json.loads((setup["out"] / "summary_calme.json").read_text())

    inc = SimpleNamespace(years=[2023, 2023], inject="fc_wind_mean+40", since="2023-01-01", name="incident")
    code = M.cmd_replay(inc, MP, setup["name"])
    latest = json.loads((setup["out"] / "latest.json").read_text())
    assert code == 2 and latest["status"] == "alert"
    assert any("fc_wind_mean" in r for r in latest["reasons"])
    assert latest["model"]["version"] == setup["version"]
    assert code_calm <= code and calm["mode"] == "replay:calme"
    assert (setup["out"] / "timeline_incident.png").exists()
    timeline = pd.read_csv(setup["out"] / "timeline_incident.csv")
    assert len(timeline) == 10  # un contrôle par tournoi de 2023, fenêtre préremplie par 2022

    runs = mlflow.search_runs(experiment_names=[M.EXPERIMENT])
    assert set(runs["tags.status"]) >= {"alert"} and len(runs) == 2


def test_replay_reports_cusum(setup):
    args = SimpleNamespace(years=[2023, 2023], inject=None, since=None, name="cusum")
    M.cmd_replay(args, MPC, setup["name"])
    latest = json.loads((setup["out"] / "latest.json").read_text())
    assert latest["signals"]["cusum_up"] is not None and latest["signals"]["cusum_down"] is not None
    timeline = pd.read_csv(setup["out"] / "timeline_cusum.csv")
    assert {"cusum_up", "cusum_down"} <= set(timeline.columns)
    assert (timeline["cusum_up"] >= 0).all() and (timeline["cusum_down"] >= 0).all()


def test_check_on_api_log(setup, monkeypatch, tmp_path):
    df = make_events(make_features())
    cur = df[df.year == 2023].head(30).assign(player_id=[f"P{i}" for i in range(30)])
    log = cur[["event_id", "player_id", "round"] + [f for f in FEATURES if f != "round"]].copy()
    log["ts"] = pd.Timestamp("2024-01-01", tz="UTC")
    log["model_version"] = setup["version"]
    log["prediction"] = 0.0
    log.to_parquet(tmp_path / "log.parquet")
    monkeypatch.setenv("PREDICTIONS_LOG", str(tmp_path / "log.parquet"))
    monkeypatch.setattr(M, "ROUNDS", tmp_path / "pas_de_scores.parquet")
    code = M.cmd_check(SimpleNamespace(), MP, setup["name"])
    latest = json.loads((setup["out"] / "latest.json").read_text())
    assert latest["mode"] == "check" and latest["log_rows"] == 30
    assert latest["log_rows_with_score"] == 0 and latest["signals"]["bias"] is None  # pas encore de scores
    assert code == M.LEVELS[latest["status"]]

    # journal trop court : PSI non évalué, pas de fausse alerte
    log.head(5).to_parquet(tmp_path / "log.parquet")
    assert M.cmd_check(SimpleNamespace(), MP, setup["name"]) == 0
    latest = json.loads((setup["out"] / "latest.json").read_text())
    assert latest["status"] == "ok" and any("non évalué" in r for r in latest["reasons"])
