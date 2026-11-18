# PGA Tour — score per round (MLOps project)

[![CI](https://github.com/lbouchand-eng/2026-2027_mlops_nl/actions/workflows/ci.yml/badge.svg)](https://github.com/lbouchand-eng/2026-2027_mlops_nl/actions/workflows/ci.yml)

Predict the **vs par** score of every PGA Tour player in every round, from:
the **forecast** weather over their tee-time window, their recent form relative to the field,
their history on the course and the course difficulty (past editions only).

| Period     | Role                                         |
|------------|----------------------------------------------|
| 2016-2017  | history used to build the features           |
| 2018-2022  | training / reference (validation: 2022)      |
| 2023-2024  | drift period                                 |

Architecture, division of work and API ↔ monitoring contract: [`docs/architecture.md`](docs/architecture.md).

## Metric

**MAE on `to_par` (in strokes)**, compared with a naive baseline (historical course
average) through a skill score `1 − MAE/MAE_baseline`. Rationale, secondary
metrics and protocol: [`docs/metric.md`](docs/metric.md).

## Quick start (fresh clone)

Prerequisites: git, [uv](https://docs.astral.sh/uv/), [gh](https://cli.github.com/)
(authenticated with an account that has access to the repo) and Docker. No API key, no secret.

```bash
git clone https://github.com/lbouchand-eng/2026-2027_mlops_nl.git && cd 2026-2027_mlops_nl
gh release download data-v1 -p pga-dvcstore.tar.gz -D ..    # versioned data (DVC store)
tar xzf ../pga-dvcstore.tar.gz -C ..                        # next to the repo: default DVC remote
uv sync && uv run dvc pull                                  # data/ at the exact hashes of dvc.lock
docker compose up --build                                   # registry (champion + previous), API, MLflow
```

The first run trains both model versions (≈ 1 min). Then:

| Service | Address |
|---|---|
| API (Swagger) | http://127.0.0.1:8000/docs |
| MLflow UI | http://127.0.0.1:5001 |

```bash
# One prediction, then a rollback on the running service
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"rows":[{"event_id":"401580344","player_id":"9478","round":1,"form_ewm":-0.8,"fc_wind_mean":14.2}]}'
docker compose exec api python -m pga.rollback --reason "test" --api-url http://127.0.0.1:8000
curl -s localhost:8000/health

# Full incident demo: requests -> alert -> rollback -> back to normal
docker compose run --rm --build demo                        # pause at each step (Enter)
docker compose run --rm --build -e DEMO_ARGS=--no-pause demo   # in one go

docker compose down -v                                      # stop and reset the registry
```

### Demo on the visible stack (Grafana, MLflow, Swagger)

```bash
docker compose --profile observability up -d --build       # + Prometheus, Pushgateway, Grafana
bash scripts/demo_live.sh --prepare                         # before the presentation (≈ 2 min): stack registry -> "incident"
bash scripts/demo_live.sh                                   # same scenario as demo.sh, pause at each step
```

| Interface | Address | What it shows during the demo |
|---|---|---|
| Grafana, "PGA MLOps" dashboard | http://127.0.0.1:3000 | requests/s, p50/p95 latency, predicted rows per version, served version, drift status and signals |
| MLflow, model `pga-score-model` | http://127.0.0.1:5001/#/models/pga-score-model | `champion` / `previous` aliases that switch on rollback, monitoring and operations runs |
| Swagger | http://127.0.0.1:8000/docs | `/health` before and after the rollback |
| Prometheus / Pushgateway | http://127.0.0.1:9090 / :9091 | raw dashboard data |

`demo_live.sh` runs each command in the `api` container (only Docker is required) and prints at
each step the URL to look at. `--prepare` **replaces** the stack registry with the incident one
(healthy v1, v2 retrained on the corrupted backfill and served) and empties the prediction
journal; `docker compose down -v` goes back to the standard registry. Grafana allows anonymous
access without a password (port bound to the local machine). `demo.sh` and the `demo` service, on an
isolated registry, are unchanged.

Without Docker, after the `dvc pull`: `uv run pytest` (119 tests, no data required),
`bash scripts/demo_setup.sh` then `uv run uvicorn pga.api:app`. To rebuild everything
from the public APIs rather than from the Release: `uv run dvc repro` (≈ 40 min).

Every pipeline stage is idempotent: raw API responses are kept under
`data/raw/` and reused, so an interrupted `dvc repro` restarts without loss.
`uv run python scripts/registry_status.py` shows the model versions and their aliases.

CI (`.github/workflows/ci.yml`): on every PR and push to `main`, `uv sync --locked`,
`ruff check` then `pytest` (without DVC data or local MLflow), and Docker image build
with a check that the service starts.

## Data pipeline (`dvc.yaml`)

| Stage          | Script                    | Output                                         |
|----------------|---------------------------|------------------------------------------------|
| `calendar`     | `src/pga/calendar.py`     | `data/interim/events.parquet` (1 call/season)  |
| `leaderboards` | `src/pga/leaderboards.py` | `data/raw/espn/leaderboard/{event_id}.json`    |
| `rounds`       | `src/pga/rounds.py`       | `data/processed/rounds.parquet`                |
| `courses`      | `src/pga/geocode.py`      | `data/interim/courses.parquet`                 |
| `weather`      | `src/pga/weather.py`      | `data/processed/weather_rounds.parquet`        |
| `features`     | `src/pga/features.py`     | `data/processed/features.parquet`              |
| `train`        | `src/pga/train.py`        | MLflow run + Registry version, `reports/metrics.json` |

Sources (free, no key):
- ESPN `site.api.espn.com/.../golf/pga/scoreboard?dates=` (calendar) and
  `site.web.api.espn.com/.../golf/leaderboard?event=` (scores per round, teeTime, course);
- Nominatim / OpenStreetMap (geocoding city → lat/lon, corrections in
  `data/courses_override.csv`);
- Open-Meteo `historical-forecast-api` (archived forecast, model input) and
  `archive-api` ERA5 (observed, to measure the observed − forecast gap).

Cleaning choices (`rounds.py`): individual stroke play only (match play and
team events excluded), playoff rounds excluded, last round of a WD/DQ player removed,
`to_par` taken from ESPN's `displayValue` (correct even for multi-course tournaments,
flagged by `multi_course`). Parameters in `params.yaml`.

## Features (`features.py`)

One row per player-round; single `FEATURES` / `TARGET` schema importable
(`from pga.features import FEATURES, TARGET`) by training, the API and monitoring.

| Group | Features |
|---|---|
| Form relative to the field | `form_rel_short`, `form_rel_long`, `form_rel_xlong` (150 rounds, baseline level), `form_ewm`, `form_ewm_slow`, `n_rounds_hist`, `days_since_last`, `is_new_player` |
| Course history / difficulty | `course_rel_mean`, `course_n_rounds`, `course_diff_mean`, `course_diff_std`, `is_new_course` |
| **Forecast** weather | `fc_temp_mean`, `fc_humidity_mean`, `fc_precip_sum`, `fc_wind_mean`, `fc_gust_max` |
| Context | `round`, `major`, `multi_course`, `yards`, `host_par` |

Anti-leakage: `rel = to_par − field mean (tournament, round)`; form only uses the
player's earlier rounds, course history and difficulty only previous
editions; observed weather (`obs_*`, `diff_*`) is reserved for drift monitoring.
Checked by `tests/test_features.py`.

## Training and MLflow (`train.py`)

```bash
uv run dvc repro train                       # or: uv run python -m pga.train [--refit-years 2021 2023]
uv run dvc metrics show                      # validation MAE / skill
uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db"   # http://127.0.0.1:5000
```

- Model: `HistGradientBoostingRegressor(loss="absolute_error")`, which directly optimizes the MAE
  and handles missing values. Hyperparameters in `params.yaml → train`.
- Protocol: fit 2018-2021, validation 2022 (decision), refit 2018-2022 (registered model),
  informative evaluation on 2023 and 2024 (drift).
- Each run logs: hyperparameters and year windows (params); MAE, RMSE, bias,
  skill, MAE per round (metrics); `git_commit`, `git_dirty`, `data_hash` (md5 of
  `features.parquet`, identical to `dvc.lock`) (tags); the model with its signature.
- Registry: model `pga-score-model`. Alias **`champion`** = served model, **`previous`** =
  previous champion (rollback target). Promotion only if skill > 0 and validation MAE
  better than the champion's (`docs/metric.md`).
- Local storage: `mlflow.db` (runs + registry) and `mlartifacts/` (models), ignored by git;
  URI can be overridden via `MLFLOW_TRACKING_URI` in `.env` (see `.env.example`).
  Loading: `mlflow.pyfunc.load_model("models:/pga-score-model@champion")`.

## API (`api.py`)

```bash
uv run uvicorn pga.api:app --reload          # http://127.0.0.1:8000/docs (Swagger)
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"rows":[{"event_id":"401580344","player_id":"9478","round":1,"form_ewm":-0.8,"course_diff_mean":0.4,"fc_wind_mean":14.2}]}'
curl -s -X POST localhost:8000/reload        # after a rollback or a promotion
```

- Serves `models:/pga-score-model@champion`, loaded **by version number**: the version
  returned by `/health` and `/predict` is the one that predicted.
- Inputs validated by Pydantic from `pga.features.FEATURES`: missing feature or
  `null` → NaN; unknown column, invalid type, round outside 1-4 → 422; no model → 503.
- `/reload` re-reads the alias without restarting; on failure, the old model keeps being served.
- Every prediction is appended to `data/monitoring/predictions.parquet` (variable
  `PREDICTIONS_LOG`), read back by monitoring. Full contract: `docs/architecture.md`.
- `/reload` requires the `X-API-Key` header if `PGA_API_KEY` is set in `.env`; the rollback and
  the trigger send it. Without this variable, no check (ports bound to the local machine).
- `GET /metrics` (Prometheus format): requests per route and status, inference latency,
  predicted rows per version, reloads, served version. Scraped every 5 s by the
  Prometheus of the `observability` Compose profile, shown in the "PGA MLOps" Grafana dashboard
  (`observability/`).
- Integration tests (`tests/test_api.py`): real model trained in a temporary
  MLflow, checks of the served version, the journal, the 422/503 and
  reloading after the alias moves.

## Monitoring and rollback

```bash
uv run python -m pga.monitor replay --years 2023 2024                 # natural drift -> reports/monitoring/timeline_replay.png
uv run python -m pga.monitor replay --years 2023 2024 --inject fc_wind_mean+15 --since 2024-03-01   # simulated incident
uv run python -m pga.monitor check                                    # production: API journal
uv run python -m pga.rollback --reason "..." --api-url http://127.0.0.1:8000   # champion <-> previous
```

- Signals over the last 8 tournaments: PSI per feature (thresholds calibrated on 2018-2022),
  z-score of the per-tournament bias and MAE ratio (thresholds read from the champion's tags);
  plus a CUSUM of the per-tournament bias, accumulated without a window, against small persistent drifts.
  Status `ok` / `warning` / `alert`, with exit code 0 / 1 / 2.
- With `PUSHGATEWAY_URL`, `monitor check` also pushes its status and signals to the
  Pushgateway (`pga_drift_status`, `pga_bias`, `pga_bias_z`, `pga_mae_ratio`, `pga_psi_max`,
  `pga_cusum`), read by Prometheus and Grafana. Without the variable, nothing changes.
- Natural drift 2023-2024: 79 ok, 9 warning (bias of +0.65 strokes in early 2024), 1 alert.
  "Wind +15 km/h" incident: alert from the 1st affected week thanks to the CUSUM (4 tournaments without it). The PSI is computed against the
  served model's training data (`reference_profile.json`).
- Method, thresholds and results: [`docs/monitoring.md`](docs/monitoring.md); what to do
  on an alert: [`docs/runbook.md`](docs/runbook.md).

## Retraining trigger (`trigger.py`)

```bash
uv run python -m pga.trigger --dry-run                 # nothing (0) / retraining (10) / data incident (20) / rollback (30)
uv run python -m pga.trigger --as-of 2024-02-05        # simulation: retraining after the January 2024 drift
```

Champion and challenger are compared on the same 8 recent tournaments, read from the API
journal when it covers them (inputs actually served, which a backfill cannot rewrite). A degradation or
a PSI alert shortly after a deployment triggers a rollback; a PSI alert never
triggers retraining. A challenger whose data departs from the
champion's is not promoted. Details and results: [`docs/runbook.md`](docs/runbook.md).

Scheduling:
- `docker compose --profile ops up -d` adds the `scheduler` service, which runs the trigger
  every week and **executes** its decisions on the stack registry;
- the `trigger.yml` workflow runs every Monday at 6:00 UTC and **decides without acting** (ephemeral
  runner registry): decision in the job summary, `latest.json` as an artifact.

With `PGA_ALERT_WEBHOOK` (`.env`, or a GitHub secret for the workflow), `monitor check` alerts
and any trigger action other than "nothing" are sent to this webhook.

## Simulated incident and demo

```bash
uv run python scripts/incident.py all       # full scenario -> reports/incident/ (isolated registry mlflow_incident.db)
uv run python scripts/incident.py all --guards   # same failure after the fixes -> reports/incident_guarded/
bash scripts/demo.sh --prepare               # before the presentation: champion = corrupted version
bash scripts/demo.sh                         # live demo, pause between each step
bash scripts/demo_live.sh --prepare && bash scripts/demo_live.sh   # the same, on the Compose stack + Grafana
```

Scenario: on March 1, 2024, a routine retraining runs on a weather backfill
downloaded in m/s instead of km/h (wind ÷ 3.6). Since the decision window is corrupted
too, the challenger wins the comparison and becomes champion. In production, on healthy
data, the PSI does not move, but the bias rises to +0.88 strokes in 4 tournaments. The
trigger chooses the **rollback** on March 25 (degradation right after a
deployment), and monitoring goes back to ok (19 checks out of 19).

The demo (`scripts/demo.sh`, ≈ 10 s without the pauses) runs: the API serves the corrupted
version, 3,442 predictions from March-April 2024, `monitor check` → alert (bias +1.79,
z = 5.9), `trigger` → rollback and API reload, same requests → ok. Report:
[`docs/incident.md`](docs/incident.md).

After the fixes, the same backfill is refused at promotion. If forced, the PSI alerts from
the first tournament and the rollback happens on March 11 instead of the 25th (19 PSI alerts out of 19
checks, versus 0 before). Compared on the API journal, the corrupted challenger also loses
on its own (MAE 2.652 versus 2.268). The last step of the demo shows the refusal.

## Project report

Full presentation of the project in English (problem, system, monitoring, incident, choices and
limitations): [`report/project.qmd`](report/project.qmd), rendered as a single self-contained HTML page.

```bash
uv pip install quarto-cli                         # Quarto in .venv (or Quarto installed separately)
uv run python report/make_figures.py              # figures drawn from reports/ and params.yaml
uv run quarto render report/project.qmd           # -> report/project.html
```

## DVC remote (local storage)

The default remote `localstore` is a folder **next to the repo**:
`../pga-dvcstore` (relative path, identical for each member).
Google Drive was ruled out: DVC's OAuth app is blocked by Google and
student accounts cannot create their own OAuth client.

Two ways to get the data on a new machine:

1. **Rebuild from the APIs** (no key, ≈ 30-40 min): `uv run dvc repro`.
   The sources are free and the stages idempotent. If an API has changed its
   data in the meantime, `uv run dvc status` reports it (hash differs from `dvc.lock`).
2. **Fetch the storage**: the `pga-dvcstore.tar.gz` archive from the repo's `data-v1` Release
   (private like the repo), to extract next to the repo, then `uv run dvc pull`.

To use another location without touching the repo:
`uv run dvc remote modify --local localstore url /path/to/pga-dvcstore`.

`.dvc/config.local` and `.env` are ignored by git: no secret in the repo.
