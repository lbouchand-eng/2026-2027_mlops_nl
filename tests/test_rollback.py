"""Tests du rollback : échange champion/previous, erreurs, trace, rechargement de l'API."""
import mlflow
import pytest
from mlflow.tracking import MlflowClient
from test_train import YEARS, P, make_features

from pga import rollback as R
from pga import train as T

NAME = P["registered_model_name"]


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Trois versions : v1 champion puis v3 promue (v1 -> previous), v2 simple candidate."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    T.setup_mlflow(uri, tmp_path / "artifacts")
    df = make_features()
    path = tmp_path / "f.parquet"
    df.to_parquet(path)
    weak = P | {"model": P["model"] | {"max_iter": 1}}
    v1 = T.run(df, weak, YEARS, path)["version"]          # 1er : champion
    v2 = T.run(df, weak, YEARS, path)["version"]          # identique : non promu
    v3 = T.run(df, P, YEARS, path)["version"]             # meilleur : champion, v1 -> previous
    c = MlflowClient()
    yield c, v1, v2, v3
    mlflow.set_tracking_uri(None)


def alias(c, a):
    return str(c.get_model_version_by_alias(NAME, a).version)


def test_rollback_swaps_champion_and_previous(registry):
    c, v1, _, v3 = registry
    assert (alias(c, "champion"), alias(c, "previous")) == (v3, v1)
    res = R.rollback(c, NAME, reason="test")
    assert (res.from_version, res.to_version) == (v3, v1)
    assert (alias(c, "champion"), alias(c, "previous")) == (v1, v3)
    tags = c.get_model_version(NAME, v1).tags
    assert tags["rolled_back_from"] == v3 and tags["rollback_reason"] == "test"


def test_double_rollback_restores_initial_state(registry):
    c, v1, _, v3 = registry
    R.rollback(c, NAME)
    R.rollback(c, NAME)
    assert (alias(c, "champion"), alias(c, "previous")) == (v3, v1)


def test_rollback_to_explicit_version(registry):
    c, _, v2, v3 = registry
    R.rollback(c, NAME, to=v2)
    assert (alias(c, "champion"), alias(c, "previous")) == (v2, v3)


@pytest.mark.parametrize("to, msg", [("999", "introuvable"), (None, "déjà")])
def test_rollback_errors(registry, to, msg):
    c, _, _, v3 = registry
    with pytest.raises(R.RollbackError, match=msg):
        R.rollback(c, NAME, to=to or v3)
    assert alias(c, "champion") == v3  # rien n'a bougé


def test_rollback_without_previous(registry):
    c, *_ = registry
    c.delete_registered_model_alias(NAME, "previous")
    with pytest.raises(R.RollbackError, match="previous"):
        R.rollback(c, NAME)


def test_main_logs_operation_and_reloads_api(registry, monkeypatch):
    c, v1, _, v3 = registry
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri())
    monkeypatch.setattr(R, "params", lambda: {"train": {"registered_model_name": NAME}})
    calls = []

    class Resp:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    monkeypatch.setattr(R.requests, "post", lambda url, timeout: calls.append(url) or Resp({}))
    monkeypatch.setattr(R.requests, "get", lambda url, timeout: Resp({"model_version": v1}))
    assert R.main(["--reason", "alerte", "--api-url", "http://api"]) == 0
    assert calls == ["http://api/reload"]
    runs = mlflow.search_runs(experiment_names=[R.EXPERIMENT])
    assert runs.iloc[0]["tags.to_version"] == v1 and runs.iloc[0]["tags.from_version"] == v3

    # l'API ne suit pas : code 2, le registry reste basculé
    monkeypatch.setattr(R.requests, "get", lambda url, timeout: Resp({"model_version": "42"}))
    assert R.main(["--api-url", "http://api"]) == 2
    assert alias(c, "champion") == v3
