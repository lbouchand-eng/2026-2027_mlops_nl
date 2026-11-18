# Architecture, division of work and API ↔ monitoring contract

## Overview

```
 ESPN / Open-Meteo / OSM
          │  dvc repro (calendar → leaderboards → rounds → courses → weather)
          ▼
 data/processed/features.parquet ──────────────┐   (DVC, repo Release data-v1)
          │  pga.train                          │
          ▼                                     │ reference 2018-2022
 MLflow (runs + Model Registry)                 │
   pga-score-model @champion / @previous        │
          │  mlflow.pyfunc.load_model           │
          ▼                                     ▼
 API FastAPI (pga.api) ──► data/monitoring/predictions.parquet ──► pga.monitor
   /health /predict /reload                                        (PSI + MAE/bias)
          ▲                                                              │ alert
          │ /reload                                                      ▼
          └──────────── pga.rollback (champion ← previous)  or  retraining (pga.train)
```

## Division of work

| Block | Owner | Files |
|---|---|---|
| FastAPI API, integration tests, prediction journal | **Nicolas** | `src/pga/api.py`, `tests/test_api.py` |
| Drift monitoring | **Léo** | `src/pga/monitor.py`, `tests/test_monitor.py` |
| Rollback and runbook | **Léo** | `src/pga/rollback.py`, `docs/runbook.md` |
| Retraining trigger, simulated incident, "fresh clone" test, slides | **together** | `src/pga/trigger.py`, `docs/incident.md` |

## API contract

**Served model**: `mlflow.pyfunc.load_model("models:/pga-score-model@champion")`.
Tracking URI: `MLFLOW_TRACKING_URI` variable (`.env`), otherwise `pga.config.MLFLOW_URI`.
The input columns are exactly `pga.features.FEATURES` (single schema, do not
copy the list).

### `GET /health`

```json
{"status": "ok", "model_name": "pga-score-model", "model_alias": "champion", "model_version": "2"}
```

### `POST /predict`

Input: a list of player-rounds. `event_id`, `player_id`, `round` are required;
each feature of `FEATURES` is a number or `null` (→ NaN, handled by the model). Unknown
column or invalid type → **422**.

```json
{"rows": [{"event_id": "401580344", "player_id": "9478", "round": 1,
           "form_ewm": -0.8, "course_diff_mean": 0.4, "fc_wind_mean": 14.2, "...": null}]}
```

Output:

```json
{"model_version": "2",
 "predictions": [{"event_id": "401580344", "player_id": "9478", "round": 1, "prediction": -0.63}]}
```

### `POST /reload`

Re-reads the `champion` alias in the Registry and returns the same response as `/health`.
Called after a rollback or a promotion, without restarting the service.
If `PGA_API_KEY` is set, the `X-API-Key` header is required (401 otherwise).

### `GET /metrics`

Prometheus format: `pga_http_requests_total{path,status}`, `pga_predict_latency_seconds`,
`pga_predicted_rows_total{model_version}`, `pga_reloads_total{result}`, `pga_served_model_version`.
Scraped every 5 s by Prometheus (Compose profile `observability`); `pga.monitor check`
adds its drift signals through the Pushgateway (`docs/monitoring.md`, "Dashboard").

## Prediction journal (monitoring input)

Each call to `/predict` appends its rows to `data/monitoring/predictions.parquet`
(path `pga.config.PREDICTIONS_LOG`, ignored by git):

| Column | Type |
|---|---|
| `ts` | UTC timestamp of the prediction |
| `event_id`, `player_id`, `round` | player-round keys |
| `model_version` | Registry version that predicted |
| `prediction` | predicted `to_par` |
| `FEATURES` columns | received values |

`pga.monitor` reads this file back, joins it to the actual scores once the rounds are played
(`rounds.parquet`) and computes data drift (PSI per feature) and performance drift
(MAE, bias, skill).

## Working rules

- One branch and one PR per block, reviewed and approved by the other before merging.
- `uv run pytest` green before any PR.
- Secrets only in `.env` (never in git); data only under DVC.
- Every new command is documented in the README.
