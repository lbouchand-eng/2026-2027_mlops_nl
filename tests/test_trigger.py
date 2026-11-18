"""Trigger tests: decision table, leak-free retraining, promotion, rollback."""
import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from test_monitor import MP, make_events
from test_train import YEARS, P, make_features

from pga import monitor as M
from pga import train as T
from pga import trigger as TR

PT = {"consecutive_checks": 3, "recent_promotion_days": 60, "train_start_year": 2021, "bias_tolerance": 0.0}
AS_OF = pd.Timestamp("2023-03-15", tz="UTC")


def timeline(zs: list[float], reasons: str = "", mae_ratio: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"status": "ok", "bias": [z * 0.3 for z in zs], "bias_z": zs,
                         "mae_ratio": mae_ratio, "reasons": reasons})


# --- decision table -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "zs, reasons, promoted_days_ago, expected",
    [
        ([0.5, 1.0, 1.5], "", None, "none"),                       # all good
        ([2.5, 1.0, 2.5], "", None, "none"),                       # non-consecutive warnings
        ([2.5, 2.2, 2.4], "", None, "retrain"),                    # 3 consecutive warnings
        ([0.5, 0.5, 3.5], "", None, "retrain"),                    # alert on the last check
        ([0.5, 0.5, 3.5], "", 20, "rollback"),                     # ... right after a deployment
        ([0.5, 0.5, 3.5], "", 400, "retrain"),                     # old deployment
        ([0.5, 0.5, 0.5], "[alert] PSI fc_wind_mean = 7 > 1.5", None, "data_incident"),
        ([3.5, 3.5, 3.5], "[alert] PSI fc_wind_mean = 7 > 1.5", 400, "data_incident"),  # never retrain on wrong data
        ([0.5, 0.5, 0.5], "[alert] PSI fc_wind_mean = 7 > 1.5", 20, "rollback"),  # the last deployment is suspect
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


# --- retraining ------------------------------------------------------------------------------

@pytest.fixture
def setup(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    T.setup_mlflow(uri, tmp_path / "artifacts")
    df = make_events(make_features())
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    T.run(df, P | {"model": P["model"] | {"max_iter": 3}}, YEARS, path)  # deliberately weak champion
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
    """A tournament started before `as_of` but not finished enters neither the monitoring nor
    the decision window: its last rounds have not been played yet."""
    df = setup["df"]
    starts = df.groupby("event_id")["event_start"].min()
    in_progress, start = starts.idxmax(), starts.max()
    as_of = start + pd.Timedelta(days=1)  # day after the 1st round
    assert in_progress not in TR.event_order(df, as_of).index
    ids, _ = TR.decision_window(df, as_of, MP["window_events"])
    assert in_progress not in ids
    assert in_progress in TR.event_order(df, start + pd.Timedelta(days=4)).index


def test_event_end_uses_last_tee_time():
    df = pd.DataFrame({"event_id": ["E", "E"], "event_start": pd.Timestamp("2024-02-01", tz="UTC"),
                       "tee_time_utc": pd.to_datetime(["2024-02-01 17:00", "2024-02-03 19:00"], utc=True)})
    assert TR.event_end(df)["E"] == pd.Timestamp("2024-02-04 01:00", tz="UTC")
    assert TR.event_order(df, pd.Timestamp("2024-02-02 23:59", tz="UTC")).empty  # Pebble Beach 2024 case
    assert list(TR.event_order(df, pd.Timestamp("2024-02-05", tz="UTC")).index) == ["E"]


def test_better_challenger_is_promoted_with_as_of_date(setup):
    # wide bias tolerance: this tests promotion on MAE
    res = TR.retrain(setup["df"], AS_OF, setup["champ"], P, MP, PT | {"bias_tolerance": 5.0}, setup["path"])
    assert res["challenger"]["mae"] < res["champion"]["mae"]  # weak champion (3 iterations)
    assert res["promoted"]
    c = MlflowClient()
    v = c.get_model_version_by_alias(P["registered_model_name"], "champion")
    assert str(v.version) == res["version"]
    assert v.tags["promoted_at"] == str(AS_OF.date())
    assert float(v.tags["valid_event_bias_sd"]) == pytest.approx(setup["champ"].event_bias_sd)
    assert str(c.get_model_version_by_alias(P["registered_model_name"], "previous").version) == setup["champ"].version


def test_worse_challenger_is_not_promoted(setup):
    weak = P | {"model": P["model"] | {"max_iter": 1}}  # weaker than the champion (3 iterations)
    res = TR.retrain(setup["df"], AS_OF, setup["champ"], weak, MP, PT, setup["path"])
    assert not res["promoted"]
    champ = MlflowClient().get_model_version_by_alias(P["registered_model_name"], "champion")
    assert str(champ.version) == setup["champ"].version


def test_corrupted_backfill_fools_the_comparison(setup):
    """Limitation without a prediction journal (basis of the simulated incident): if the decision window is
    corrupted like the training data, the corrupted challenger looks better and is promoted."""
    def corrupt(d):
        d = d.copy()
        d["to_par"] = d["to_par"] + 5
        return d

    res = TR.retrain(setup["df"], AS_OF, setup["champ"], P, MP, PT | {"bias_tolerance": 5.0}, setup["path"],
                     transform=corrupt, transform_label="to_par+5")
    assert res["promoted"]
    v = MlflowClient().get_model_version(P["registered_model_name"], res["version"])
    assert v.tags["data_transform"] == "to_par+5"


def wind_in_ms(d):
    d = d.copy()
    for col in ("fc_wind_mean", "fc_gust_max"):
        d[col] = d[col] / 3.6
    return d


def test_wind_backfill_in_ms_is_blocked(setup):
    """The challenger looks better on a corrupted window, but its data differs from the champion's."""
    res = TR.retrain(setup["df"], AS_OF, setup["champ"], P, MP, PT | {"bias_tolerance": 5.0}, setup["path"],
                     transform=wind_in_ms, transform_label="wind in m/s")
    assert not res["promoted"] and any("fc_wind_mean" in b for b in res["blocked"])
    c = MlflowClient()
    assert str(c.get_model_version_by_alias(P["registered_model_name"], "champion").version) == setup["champ"].version
    assert "fc_wind_mean" in c.get_model_version(P["registered_model_name"], res["version"]).tags["blocked_reason"]


def test_guards_off_replays_the_original_incident(setup):
    res = TR.retrain(setup["df"], AS_OF, setup["champ"], P, MP, PT | {"bias_tolerance": 5.0}, setup["path"],
                     transform=wind_in_ms, transform_label="wind in m/s", guards=False)
    assert res["promoted"] and res["blocked"] == []


def add_five_to_target(d):
    d = d.copy()
    d["to_par"] = d["to_par"] + 5
    return d


def window(df):
    ids, _ = TR.decision_window(df, AS_OF, MP["window_events"])
    return ids, df[df.event_id.isin(ids)]


def test_journal_catches_what_the_backfill_hides(setup):
    """Same backfill as test_corrupted_backfill_fools_the_comparison, judged on what production saw."""
    _, journal = window(setup["df"])
    res = TR.retrain(setup["df"], AS_OF, setup["champ"], P, MP | {"min_rows": 1}, PT | {"bias_tolerance": 5.0}, setup["path"],
                     transform=add_five_to_target, transform_label="to_par+5", journal=journal)
    assert not res["promoted"] and res["decision_source"] == "journal"
    assert mlflow.get_run(res["run_id"]).data.tags["decision_source"] == "journal"


def test_journal_on_clean_data_changes_nothing(setup):
    _, journal = window(setup["df"])
    mp = MP | {"min_rows": 1}
    from_table = TR.retrain(setup["df"], AS_OF, setup["champ"], P, mp, PT, setup["path"])
    from_journal = TR.retrain(setup["df"], AS_OF, setup["champ"], P, mp, PT, setup["path"], journal=journal)
    assert from_table["decision_source"] == "backfill"
    assert from_journal["challenger"]["mae"] == pytest.approx(from_table["challenger"]["mae"])


def test_partial_journal_is_not_used(setup):
    ids, journal = window(setup["df"])
    assert TR.journal_window(journal, ids, 1) is not None
    assert TR.journal_window(journal[journal.event_id != ids[0]], ids, 1) is None   # a tournament is missing
    assert TR.journal_window(journal, ids, len(journal) + 1) is None                # too few rows
    assert TR.journal_window(journal.assign(to_par=float("nan")), ids, 1) is None   # scores not yet known
    assert TR.journal_window(None, ids, 1) is None


def test_data_incident_never_trains(setup, monkeypatch):
    called = []
    monkeypatch.setattr(TR, "retrain", lambda *a, **k: called.append(1))
    monkeypatch.setattr(TR, "monitoring_as_of", lambda *a, **k: timeline([3.5, 3.5, 3.5], "[alert] PSI fc_wind_mean = 7 > 1.5"))
    monkeypatch.setattr(TR, "params", lambda: {"monitor": MP, "trigger": PT, "train": P})
    monkeypatch.setattr(TR, "TRIGGER_DIR", setup["path"].parent / "trigger")
    decision, outcome = TR.run_trigger(AS_OF, df=setup["df"], data_path=setup["path"])
    assert decision.action == "data_incident" and not called and not outcome["executed"]
    assert (setup["path"].parent / "trigger" / "latest.json").exists()
