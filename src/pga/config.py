"""Chemins et paramètres partagés par les étapes du pipeline."""
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
COURSES_OVERRIDE = DATA / "courses_override.csv"


def params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text())
