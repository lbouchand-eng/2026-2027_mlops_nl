"""Paths and parameters shared by the pipeline stages."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_SCOREBOARD = RAW / "espn" / "scoreboard"
RAW_LEADERBOARD = RAW / "espn" / "leaderboard"
RAW_GEOCODE = RAW / "geocode"
RAW_METEO_FC = RAW / "meteo" / "forecast"
RAW_METEO_OBS = RAW / "meteo" / "archive"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

EVENTS = INTERIM / "events.parquet"
COURSES = INTERIM / "courses.parquet"
ROUNDS = PROCESSED / "rounds.parquet"
WEATHER_ROUNDS = PROCESSED / "weather_rounds.parquet"
FEATURES_PATH = PROCESSED / "features.parquet"
COURSES_OVERRIDE = DATA / "courses_override.csv"


def params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text())

REPORTS = ROOT / "reports"
METRICS_PATH = REPORTS / "metrics.json"
MLARTIFACTS = ROOT / "mlartifacts"
# Overridable by the MLFLOW_TRACKING_URI environment variable (.env).
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
# API prediction journal (monitoring input), overridable by PREDICTIONS_LOG.
PREDICTIONS_LOG = DATA / "monitoring" / "predictions.parquet"
