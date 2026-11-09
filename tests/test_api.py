"""Tests d'intégration de l'API : vrai modèle entraîné dans un MLflow temporaire."""
import math

import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from test_train import YEARS, P, make_features

from pga import rollback as R
from pga import train as T
from pga.api import create_app
from pga.config import params
from pga.features import FEATURES

NAME = params()["train"]["registered_model_name"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """MLflow temporaire avec deux versions enregistrées ; v1 = champion."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("PREDICTIONS_LOG", str(tmp_path / "predictions.parquet"))
    T.setup_mlflow(uri, tmp_path / "artifacts")
    df = make_features()
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    p = P | {"registered_model_name": NAME}
    v1 = T.run(df, p, YEARS, path)["version"]
    v2 = T.run(df, p | {"model": p["model"] | {"max_iter": 1}}, YEARS, path)["version"]  # non promue
    yield {"df": df, "v1": v1, "v2": v2, "log": tmp_path / "predictions.parquet"}
    mlflow.set_tracking_uri(None)


@pytest.fixture
def client(env):
    with TestClient(create_app()) as c:  # `with` déclenche le chargement du modèle
        yield c


def rows(df: pd.DataFrame, n: int = 3) -> list[dict]:
    out = []
    for i, r in df.head(n).reset_index(drop=True).iterrows():
        row = {"event_id": f"E{i}", "player_id": f"P{i}", "round": int(r["round"])}
        row |= {f: float(r[f]) for f in FEATURES if f != "round"}
        out.append(row)
    return out


def test_health_reports_champion_version(client, env):
    h = client.get("/health").json()
    assert h == {"status": "ok", "model_name": NAME, "model_alias": "champion", "model_version": env["v1"], "detail": None}


def test_predict_matches_registry_model_and_logs(client, env):
    payload = rows(env["df"])
    res = client.post("/predict", json={"rows": payload})
    assert res.status_code == 200
    body = res.json()
    assert body["model_version"] == env["v1"]
    assert [p["player_id"] for p in body["predictions"]] == ["P0", "P1", "P2"]

    expected = mlflow.pyfunc.load_model(f"models:/{NAME}/{env['v1']}").predict(env["df"][FEATURES].astype(float).head(3))
    assert [p["prediction"] for p in body["predictions"]] == pytest.approx(list(expected))

    log = pd.read_parquet(env["log"])
    assert len(log) == 3 and set(log.model_version) == {env["v1"]}
    assert {"ts", "event_id", "player_id", "round", "prediction", *FEATURES} <= set(log.columns)
    client.post("/predict", json={"rows": payload[:1]})
    assert len(pd.read_parquet(env["log"])) == 4  # le journal s'accumule


def test_missing_features_become_nan(client):
    res = client.post("/predict", json={"rows": [{"event_id": "E", "player_id": "P", "round": 1, "form_ewm": None}]})
    assert res.status_code == 200
    assert math.isfinite(res.json()["predictions"][0]["prediction"])


@pytest.mark.parametrize(
    "row",
    [
        {"event_id": "E", "player_id": "P", "round": 1, "colonne_inconnue": 1.0},
        {"event_id": "E", "player_id": "P", "round": 1, "form_ewm": "abc"},
        {"event_id": "E", "player_id": "P", "round": 7},
        {"player_id": "P", "round": 1},
    ],
)
def test_invalid_input_is_422(client, row):
    assert client.post("/predict", json={"rows": [row]}).status_code == 422


def test_empty_request_is_422(client):
    assert client.post("/predict", json={"rows": []}).status_code == 422


def test_reload_follows_alias_rollback(client, env):
    MlflowClient().set_registered_model_alias(NAME, "champion", env["v2"])
    assert client.get("/health").json()["model_version"] == env["v1"]  # pas encore rechargé
    assert client.post("/reload").json()["model_version"] == env["v2"]
    res = client.post("/predict", json={"rows": rows(env["df"], 1)})
    assert res.json()["model_version"] == env["v2"]


def metric(client, line: str) -> float:
    found = [row for row in client.get("/metrics").text.splitlines() if row.startswith(line + " ")]
    return float(found[0].split()[-1]) if found else 0.0


def test_metrics_follow_traffic_and_reload(client, env):
    client.post("/predict", json={"rows": rows(env["df"], 3)})
    client.post("/predict", json={"rows": [{"event_id": "E", "player_id": "P", "round": 7}]})
    client.get("/introuvable")
    assert metric(client, f'pga_predicted_rows_total{{model_version="{env["v1"]}"}}') == 3
    assert metric(client, 'pga_http_requests_total{path="/predict",status="200"}') == 1
    assert metric(client, 'pga_http_requests_total{path="/predict",status="422"}') == 1
    assert metric(client, 'pga_http_requests_total{path="inconnu",status="404"}') == 1
    assert metric(client, "pga_predict_latency_seconds_count") == 1
    assert metric(client, "pga_served_model_version") == float(env["v1"])

    MlflowClient().set_registered_model_alias(NAME, "champion", env["v2"])
    client.post("/reload")
    assert metric(client, "pga_served_model_version") == float(env["v2"])
    assert metric(client, 'pga_reloads_total{result="ok"}') == 1


@pytest.mark.parametrize("headers, expected", [({}, 401), ({"X-API-Key": "mauvaise"}, 401), ({"X-API-Key": "s3cret"}, 200)])
def test_reload_requires_the_key_when_configured(client, monkeypatch, headers, expected):
    monkeypatch.setenv("PGA_API_KEY", "s3cret")
    assert client.post("/reload", headers=headers).status_code == expected


def test_predict_stays_open_when_a_key_is_configured(client, env, monkeypatch):
    monkeypatch.setenv("PGA_API_KEY", "s3cret")
    assert client.post("/predict", json={"rows": rows(env["df"], 1)}).status_code == 200


def test_rollback_sends_the_key(client, env, monkeypatch):
    """Le rollback recharge une API protégée."""
    monkeypatch.setenv("PGA_API_KEY", "s3cret")
    monkeypatch.setattr(R.requests, "post", lambda url, headers, timeout: client.post("/reload", headers=headers))
    monkeypatch.setattr(R.requests, "get", lambda url, timeout: client.get("/health"))
    assert R.reload_api("http://api", env["v1"])["model_version"] == env["v1"]


def test_no_model_is_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'vide.db'}")
    with TestClient(create_app()) as c:
        assert c.get("/health").json()["status"] == "no_model"
        assert c.post("/predict", json={"rows": [{"event_id": "E", "player_id": "P", "round": 1}]}).status_code == 503
        assert c.post("/reload").status_code == 503
    mlflow.set_tracking_uri(None)
