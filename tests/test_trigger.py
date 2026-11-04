"""Tests du déclencheur : table de décision, réentraînement sans fuite, promotion, rollback."""
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from pga import monitor as M
from pga import trigger as TR
from pga import train as T
from test_monitor import MP, make_events
from test_train import P, YEARS, make_features

PT = {"consecutive_checks": 3, "recent_promotion_days": 60, "train_start_year": 2021, "bias_tolerance": 0.0}
AS_OF = pd.Timestamp("2023-03-15", tz="UTC")


def timeline(zs: list[float], reasons: str = "", mae_ratio: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"status": "ok", "bias": [z * 0.3 for z in zs], "bias_z": zs,
                         "mae_ratio": mae_ratio, "reasons": reasons})


# --- table de décision -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "zs, reasons, promoted_days_ago, expected",
    [
        ([0.5, 1.0, 1.5], "", None, "none"),                       # tout va bien
        ([2.5, 1.0, 2.5], "", None, "none"),                       # warnings non consécutifs
        ([2.5, 2.2, 2.4], "", None, "retrain"),                    # 3 warnings consécutifs
        ([0.5, 0.5, 3.5], "", None, "retrain"),                    # alerte au dernier contrôle
        ([0.5, 0.5, 3.5], "", 20, "rollback"),                     # ... juste après une mise en service
        ([0.5, 0.5, 3.5], "", 400, "retrain"),                     # mise en service ancienne
        ([0.5, 0.5, 0.5], "[alert] PSI fc_wind_mean = 7 > 1.5", None, "data_incident"),
        ([3.5, 3.5, 3.5], "[alert] PSI fc_wind_mean = 7 > 1.5", 20, "data_incident"),  # jamais de retrain sur données fausses
    ],
)
def test_decision_table(zs, reasons, promoted_days_ago, expected):
    promoted = AS_OF - pd.Timedelta(days=promoted_days_ago) if promoted_days_ago is not None else None
    d = TR.decide(timeline(zs, reasons), AS_OF, promoted, "7", MP, PT)
    assert d.action == expected
    assert d.reason


def test_mae_ratio_alone_triggers():
    assert TR.decide(timeline([0.1, 0.1, 0.1], mae_ratio=1.3), AS_OF, None, "1", MP, PT).action == "retrain"


def test_promotion_in_the_future_is_not_recent():
    future = AS_OF + pd.Timedelta(days=5)
    assert TR.decide(timeline([0.5, 0.5, 3.5]), AS_OF, future, "1", MP, PT).action == "retrain"


# --- réentraînement ------------------------------------------------------------------------------

@pytest.fixture
def setup(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    T.setup_mlflow(uri, tmp_path / "artifacts")
    df = make_events(make_features())
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    T.run(df, P | {"model": P["model"] | {"max_iter": 3}}, YEARS, path)  # champion volontairement faible
    champ = M.champion_info(MlflowClient(), P["registered_model_name"], 0.85)
    yield {"df": df, "path": path, "champ": champ}
    mlflow.set_tracking_uri(None)


def test_decision_window_is_excluded_from_training(setup):
    df = setup["df"]
    ids, start = TR.decision_window(df, AS_OF, MP["window_events"])
    assert len(ids) == MP["window_events"]
    assert (df[df.event_id.isin(ids)].event_start >= start).all()
    assert df[df.event_id.isin(ids)].event_start.max() <= AS_OF
    res = TR.retrain(df, AS_OF, setup["champ"], P, MP, PT, setup["path"])
    assert res["fit_until"] == str(start.date())
    run = mlflow.get_run(res["run_id"])
    assert run.data.params["fit_until"] == str(start.date())
    assert run.data.tags["compared_to"] == f"v{setup['champ'].version}"


def test_event_in_progress_is_not_used(setup):
    """Un tournoi commencé avant `as_of` mais pas terminé n'entre ni dans le monitoring ni
    dans la fenêtre de décision : ses derniers tours ne sont pas encore joués."""
    df = setup["df"]
    starts = df.groupby("event_id")["event_start"].min()
    in_progress, start = starts.idxmax(), starts.max()
    as_of = start + pd.Timedelta(days=1)  # lendemain du 1er tour
    assert in_progress not in TR.event_order(df, as_of).index
    ids, _ = TR.decision_window(df, as_of, MP["window_events"])
    assert in_progress not in ids
    assert in_progress in TR.event_order(df, start + pd.Timedelta(days=4)).index


def test_event_end_uses_last_tee_time():
    df = pd.DataFrame({"event_id": ["E", "E"], "event_start": pd.Timestamp("2024-02-01", tz="UTC"),
                       "tee_time_utc": pd.to_datetime(["2024-02-01 17:00", "2024-02-03 19:00"], utc=True)})
    assert TR.event_end(df)["E"] == pd.Timestamp("2024-02-04 01:00", tz="UTC")
    assert TR.event_order(df, pd.Timestamp("2024-02-02 23:59", tz="UTC")).empty  # cas Pebble Beach 2024
    assert list(TR.event_order(df, pd.Timestamp("2024-02-05", tz="UTC")).index) == ["E"]


def test_better_challenger_is_promoted_with_as_of_date(setup):
    # tolérance large sur le biais : on teste ici la promotion sur la MAE
    res = TR.retrain(setup["df"], AS_OF, setup["champ"], P, MP, PT | {"bias_tolerance": 5.0}, setup["path"])
    assert res["challenger"]["mae"] < res["champion"]["mae"]  # champion faible (3 itérations)
    assert res["promoted"]
    c = MlflowClient()
    v = c.get_model_version_by_alias(P["registered_model_name"], "champion")
    assert str(v.version) == res["version"]
    assert v.tags["promoted_at"] == str(AS_OF.date())
    assert float(v.tags["valid_event_bias_sd"]) == pytest.approx(setup["champ"].event_bias_sd)
    assert str(c.get_model_version_by_alias(P["registered_model_name"], "previous").version) == setup["champ"].version


def test_worse_challenger_is_not_promoted(setup):
    weak = P | {"model": P["model"] | {"max_iter": 1}}  # plus faible que le champion (3 itérations)
    res = TR.retrain(setup["df"], AS_OF, setup["champ"], weak, MP, PT, setup["path"])
    assert not res["promoted"]
    champ = MlflowClient().get_model_version_by_alias(P["registered_model_name"], "champion")
    assert str(champ.version) == setup["champ"].version


def test_corrupted_backfill_fools_the_comparison(setup):
    """Limite assumée (base de l'incident simulé) : si la fenêtre de décision est corrompue
    comme les données d'entraînement, le challenger corrompu paraît meilleur et est promu.
    Seul le monitoring en production, sur données saines, peut le détecter."""
    def corrupt(d):
        d = d.copy()
        d["to_par"] = d["to_par"] + 5
        return d

    res = TR.retrain(setup["df"], AS_OF, setup["champ"], P, MP, PT | {"bias_tolerance": 5.0}, setup["path"],
                     transform=corrupt, transform_label="to_par+5")
    assert res["promoted"]
    v = MlflowClient().get_model_version(P["registered_model_name"], res["version"])
    assert v.tags["data_transform"] == "to_par+5"


def test_data_incident_never_trains(setup, monkeypatch):
    called = []
    monkeypatch.setattr(TR, "retrain", lambda *a, **k: called.append(1))
    monkeypatch.setattr(TR, "monitoring_as_of", lambda *a, **k: timeline([3.5, 3.5, 3.5], "[alert] PSI fc_wind_mean = 7 > 1.5"))
    monkeypatch.setattr(TR, "params", lambda: {"monitor": MP, "trigger": PT, "train": P})
    monkeypatch.setattr(TR, "TRIGGER_DIR", setup["path"].parent / "trigger")
    decision, outcome = TR.run_trigger(AS_OF, df=setup["df"], data_path=setup["path"])
    assert decision.action == "data_incident" and not called and not outcome["executed"]
    assert (setup["path"].parent / "trigger" / "latest.json").exists()
