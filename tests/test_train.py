"""Tests de l'entraînement : métriques, traçabilité MLflow et règle de promotion."""
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from pga import train as T
from pga.features import FEATURES, TARGET

P = {
    "random_state": 0,
    "registered_model_name": "test-model",
    "eval_years": [2023],
    "model": {"learning_rate": 0.1, "max_iter": 30, "max_leaf_nodes": 7, "min_samples_leaf": 10, "l2_regularization": 0.0},
}
YEARS = {"train": [2021, 2021], "valid": [2022, 2022], "refit": [2021, 2022]}


def make_features(n_per_year: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = np.repeat([2021, 2022, 2023], n_per_year)
    df = pd.DataFrame({c: rng.normal(size=len(years)) for c in FEATURES})
    df["round"] = rng.integers(1, 5, size=len(years))
    df["year"] = years
    df["course_diff_mean"] = rng.normal(scale=0.5, size=len(years))
    # signal : la forme et la difficulté expliquent une partie du score
    df[TARGET] = (2 * df["form_ewm"] + df["course_diff_mean"] + rng.normal(scale=1.0, size=len(years))).round()
    return df


@pytest.fixture
def mlflow_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    T.setup_mlflow(f"sqlite:///{tmp_path / 'mlflow.db'}", tmp_path / "artifacts")
    yield MlflowClient()
    mlflow.set_tracking_uri(None)


@pytest.fixture
def data_file(tmp_path):
    path = tmp_path / "features.parquet"
    make_features().to_parquet(path)
    return path


def test_evaluate_known_values():
    m = T.evaluate(y=[0, 2, -2, 4], pred=[1, 2, -2, 2], base=[0, 0, 0, 0], rounds=[1, 1, 2, 2])
    assert m["mae"] == pytest.approx(0.75)
    assert m["bias"] == pytest.approx(-0.25)
    assert m["mae_baseline"] == pytest.approx(2.0)
    assert m["skill"] == pytest.approx(1 - 0.75 / 2.0)
    assert m["mae_r1"] == pytest.approx(0.5) and m["mae_r2"] == pytest.approx(1.0)


def test_train_rejects_empty_window():
    with pytest.raises(ValueError):
        T.train(make_features(), P, [2030, 2030], [2022, 2022], [2021, 2022])


def test_run_logs_traceability_and_registers(mlflow_tmp, data_file):
    df = pd.read_parquet(data_file)
    res = T.run(df, P, YEARS, data_file)
    r = mlflow_tmp.get_run(res["run_id"])
    assert r.data.tags["data_hash"] == T.data_hash(data_file)
    assert r.data.tags["git_commit"]
    assert {"valid_mae", "valid_skill", "valid_bias", "eval_2023_mae"} <= set(r.data.metrics)
    assert r.data.params["refit_years"] == "2021-2022"
    assert res["metrics"]["valid_skill"] > 0  # le signal synthétique bat la baseline
    champ = mlflow_tmp.get_model_version_by_alias("test-model", T.CHAMPION)
    assert str(champ.version) == res["version"] and res["promoted"]
    loaded = mlflow.pyfunc.load_model("models:/test-model@champion")
    assert len(loaded.predict(df[FEATURES].astype(float).head(3))) == 3


def test_promotion_rule(mlflow_tmp, data_file):
    df = pd.read_parquet(data_file)
    first = T.run(df, P, YEARS, data_file)
    assert first["promoted"]

    # Modèle dégradé (1 itération) : ne doit pas détrôner le champion.
    weak_p = P | {"model": P["model"] | {"max_iter": 1}}
    weak = T.run(df, weak_p, YEARS, data_file)
    assert not weak["promoted"]
    assert str(mlflow_tmp.get_model_version_by_alias("test-model", T.CHAMPION).version) == first["version"]

    # Version non promue au run, puis déclarée meilleure : devient champion,
    # l'ancien champion passe en previous.
    v3 = T.run(df, weak_p, YEARS, data_file)["version"]
    promoted, _ = T.promote_if_better(mlflow_tmp, "test-model", v3, valid_mae=0.0, valid_skill=0.9)
    assert promoted
    assert str(mlflow_tmp.get_model_version_by_alias("test-model", T.CHAMPION).version) == v3
    assert str(mlflow_tmp.get_model_version_by_alias("test-model", T.PREVIOUS).version) == first["version"]


def test_negative_skill_never_promoted(mlflow_tmp, data_file):
    v = T.run(pd.read_parquet(data_file), P, YEARS, data_file, register=True)["version"]
    promoted, reason = T.promote_if_better(mlflow_tmp, "test-model", v, valid_mae=0.0, valid_skill=-0.1)
    assert not promoted and "baseline" in reason


def test_git_dirty_ignores_pipeline_outputs(tmp_path):
    import subprocess

    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "code.py").write_text("a = 1\n")
    (tmp_path / "dvc.lock").write_text("v1\n")
    git("add", ".")
    git("commit", "-qm", "init")
    (tmp_path / "dvc.lock").write_text("v2\n")
    assert T.git_info(tmp_path)["git_dirty"] == "false"
    (tmp_path / "code.py").write_text("a = 2\n")
    assert T.git_info(tmp_path)["git_dirty"] == "true"


def test_event_bias_sd_logged_and_tagged(mlflow_tmp, tmp_path):
    df = make_features()
    df["event_id"] = (df.index // 15).astype(str)  # 10 tournois par année
    path = tmp_path / "f.parquet"
    df.to_parquet(path)
    res = T.run(df, P, YEARS, path)
    sd = res["metrics"]["valid_event_bias_sd"]
    assert sd > 0
    v = mlflow_tmp.get_model_version("test-model", res["version"])
    assert float(v.tags["valid_event_bias_sd"]) == pytest.approx(sd, rel=1e-5)


def test_event_bias_sd_value():
    assert T.event_bias_sd(["a", "a", "b", "b"], [1, 1, -1, -1]) == pytest.approx(np.std([1, -1], ddof=1))
