"""FastAPI prediction service (contract: docs/architecture.md).

- GET  /health  : status and version of the served model;
- POST /predict : `to_par` prediction for a list of player-rounds;
- POST /reload  : re-reads the `champion` alias from the Registry (after rollback or promotion);
                  protected by the `X-API-Key` header if `PGA_API_KEY` is set;
- GET  /metrics : Prometheus metrics (requests, latency, predicted rows, served version).

The model is loaded **by version number** (resolved from the alias): the announced
version is always the one that predicts. Every prediction is appended to the
`PREDICTIONS_LOG` prediction journal, read back by monitoring.

Run: uv run uvicorn pga.api:app --reload
"""
from __future__ import annotations

import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, create_model

from pga.config import MLFLOW_URI, PREDICTIONS_LOG, ROOT, params
from pga.features import FEATURES

ALIAS = "champion"
KEY_FIELDS = {"event_id": (str, ...), "player_id": (str, ...), "round": (int, Field(..., ge=1, le=4))}

# One row = keys + each feature (number or null -> NaN). Unknown column -> 422.
Row = create_model(
    "Row",
    __config__=ConfigDict(extra="forbid"),
    **KEY_FIELDS,
    **{f: (float | None, None) for f in FEATURES if f not in KEY_FIELDS},
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[Row] = Field(..., min_length=1)  # type: ignore[valid-type]


class Prediction(BaseModel):
    event_id: str
    player_id: str
    round: int
    prediction: float


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class Health(BaseModel):
    status: str
    model_name: str
    model_alias: str
    model_version: str | None
    detail: str | None = None


class ModelService:
    """Served model + its version number; hot-reloadable."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.version: str | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Loads the version the alias points to. On failure, keeps the current model."""
        try:
            mv = MlflowClient().get_model_version_by_alias(self.name, ALIAS)
            model = mlflow.pyfunc.load_model(f"models:/{self.name}/{mv.version}")
        except Exception as exc:  # noqa: BLE001 — the service starts even without a model
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        with self._lock:
            self.model, self.version, self.error = model, str(mv.version), None
        return True

    def health(self) -> Health:
        ok = self.model is not None
        return Health(
            status="ok" if ok else "no_model",
            model_name=self.name,
            model_alias=ALIAS,
            model_version=self.version,
            detail=None if ok else self.error,
        )

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, str]:
        with self._lock:
            if self.model is None:
                raise HTTPException(503, f"no model loaded: {self.error}")
            return np.asarray(self.model.predict(X), dtype=float), self.version  # type: ignore[return-value]


def to_frame(rows: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(keys, float features in FEATURES order)."""
    df = pd.DataFrame([r.model_dump() for r in rows])
    X = df.reindex(columns=FEATURES).astype(float)
    return df[list(KEY_FIELDS)], X


def append_log(path: Path, keys: pd.DataFrame, X: pd.DataFrame, pred: np.ndarray, version: str) -> None:
    """Appends the predictions to the parquet prediction journal (monitoring input)."""
    entry = keys.assign(ts=pd.Timestamp.now(tz="UTC"), model_version=version, prediction=pred)
    entry = pd.concat([entry, X.drop(columns=[c for c in KEY_FIELDS if c in X])], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        entry = pd.concat([pd.read_parquet(path), entry], ignore_index=True)
    entry.to_parquet(path, index=False)


class Metrics:
    """Registry specific to each app: `create_app` can be called several times."""

    def __init__(self, service: ModelService):
        self.registry = CollectorRegistry()
        self.requests = Counter("pga_http_requests", "HTTP requests", ["path", "status"], registry=self.registry)
        self.rows = Counter("pga_predicted_rows", "Predicted rows", ["model_version"], registry=self.registry)
        self.latency = Histogram("pga_predict_latency_seconds", "Inference duration", registry=self.registry)
        self.reloads = Counter("pga_reloads", "Model reloads", ["result"], registry=self.registry)
        served = Gauge("pga_served_model_version", "Served model version (0: none)", registry=self.registry)
        served.set_function(lambda: int(service.version or 0))


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Without `PGA_API_KEY`, no check: the Compose stack only listens on the local machine."""
    expected = os.environ.get("PGA_API_KEY")
    if expected and not secrets.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(401, "missing or invalid API key (X-API-Key header)")


def create_app() -> FastAPI:
    load_dotenv(ROOT / ".env")
    service = ModelService(params()["train"]["registered_model_name"])
    metrics = Metrics(service)
    log_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))
        service.load()
        yield

    app = FastAPI(title="PGA Tour — score vs par", version="1.0", lifespan=lifespan)
    app.state.service = service

    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        response = await call_next(request)
        route = request.scope.get("route")  # declared path: an unknown URL doesn't create a series
        metrics.requests.labels(route.path if route else "unknown", str(response.status_code)).inc()
        return response

    @app.get("/health", response_model=Health)
    def health():
        return service.health()

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        keys, X = to_frame(req.rows)
        with metrics.latency.time():
            pred, version = service.predict(X)
        metrics.rows.labels(version).inc(len(pred))
        with log_lock:
            append_log(Path(os.environ.get("PREDICTIONS_LOG", PREDICTIONS_LOG)), keys, X, pred, version)
        return PredictResponse(
            model_version=version,
            predictions=[Prediction(**k, prediction=float(p)) for k, p in zip(keys.to_dict("records"), pred, strict=True)],
        )

    @app.post("/reload", response_model=Health, dependencies=[Depends(require_api_key)])
    def reload():
        loaded = service.load()
        metrics.reloads.labels("ok" if loaded else "error").inc()
        if not loaded:
            # the previous model (if any) keeps being served
            raise HTTPException(503, f"reload failed: {service.error}")
        return service.health()

    @app.get("/metrics")
    def prometheus():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
