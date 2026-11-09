"""Service de prédiction FastAPI (contrat : docs/architecture.md).

- GET  /health  : état et version du modèle servi ;
- POST /predict : prédiction de `to_par` pour une liste de joueur-tours ;
- POST /reload  : relit l'alias `champion` du Registry (après rollback ou promotion).

Le modèle est chargé **par numéro de version** (résolu depuis l'alias) : la version
annoncée est toujours celle qui prédit. Chaque prédiction est ajoutée au journal
`PREDICTIONS_LOG`, relu par le monitoring.

Lancement : uv run uvicorn pga.api:app --reload
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field, create_model

from pga.config import MLFLOW_URI, PREDICTIONS_LOG, ROOT, params
from pga.features import FEATURES

ALIAS = "champion"
KEY_FIELDS = {"event_id": (str, ...), "player_id": (str, ...), "round": (int, Field(..., ge=1, le=4))}

# Une ligne = clés + chaque feature (nombre ou null -> NaN). Colonne inconnue -> 422.
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
    """Modèle servi + son numéro de version ; rechargeable à chaud."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.version: str | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Charge la version pointée par l'alias. En cas d'échec, garde le modèle courant."""
        try:
            mv = MlflowClient().get_model_version_by_alias(self.name, ALIAS)
            model = mlflow.pyfunc.load_model(f"models:/{self.name}/{mv.version}")
        except Exception as exc:  # noqa: BLE001 — le service démarre même sans modèle
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
                raise HTTPException(503, f"aucun modèle chargé : {self.error}")
            return np.asarray(self.model.predict(X), dtype=float), self.version  # type: ignore[return-value]


def to_frame(rows: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(clés, features float dans l'ordre de FEATURES)."""
    df = pd.DataFrame([r.model_dump() for r in rows])
    X = df.reindex(columns=FEATURES).astype(float)
    return df[list(KEY_FIELDS)], X


def append_log(path: Path, keys: pd.DataFrame, X: pd.DataFrame, pred: np.ndarray, version: str) -> None:
    """Ajoute les prédictions au journal parquet (entrée du monitoring)."""
    entry = keys.assign(ts=pd.Timestamp.now(tz="UTC"), model_version=version, prediction=pred)
    entry = pd.concat([entry, X.drop(columns=[c for c in KEY_FIELDS if c in X])], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        entry = pd.concat([pd.read_parquet(path), entry], ignore_index=True)
    entry.to_parquet(path, index=False)


def create_app() -> FastAPI:
    load_dotenv(ROOT / ".env")
    service = ModelService(params()["train"]["registered_model_name"])
    log_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))
        service.load()
        yield

    app = FastAPI(title="PGA Tour — score vs par", version="1.0", lifespan=lifespan)
    app.state.service = service

    @app.get("/health", response_model=Health)
    def health():
        return service.health()

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        keys, X = to_frame(req.rows)
        pred, version = service.predict(X)
        with log_lock:
            append_log(Path(os.environ.get("PREDICTIONS_LOG", PREDICTIONS_LOG)), keys, X, pred, version)
        return PredictResponse(
            model_version=version,
            predictions=[Prediction(**k, prediction=float(p)) for k, p in zip(keys.to_dict("records"), pred, strict=True)],
        )

    @app.post("/reload", response_model=Health)
    def reload():
        if not service.load():
            # le modèle précédent (s'il existe) continue d'être servi
            raise HTTPException(503, f"rechargement impossible : {service.error}")
        return service.health()

    return app


app = create_app()
