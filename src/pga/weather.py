"""Stage 5: weather over each round's playing window (forecast and observed).

Per tournament, two Open-Meteo calls on the host course (UTC hours):
- historical-forecast-api: the archived forecast (what would have been known before the round)
  -> used as a model input (no information leakage);
- archive-api (ERA5): the observed weather -> used to measure the observed - forecast gap.
The archived forecast does not exist for the oldest years (≈ before 2018):
those rows have fc_* = NaN, which is fine since 2016-2020 is only used for history.

Aggregation over [teeTime, teeTime + window_hours[ for each player-round.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pga.config import COURSES, RAW_METEO_FC, RAW_METEO_OBS, ROUNDS, WEATHER_ROUNDS, params
from pga.http import get_json

SOURCES = {
    "fc": ("https://historical-forecast-api.open-meteo.com/v1/forecast", RAW_METEO_FC),
    "obs": ("https://archive-api.open-meteo.com/v1/archive", RAW_METEO_OBS),
}
# hourly variable -> (aggregated name, aggregation function)
AGG = {
    "temperature_2m": ("temp_mean", np.nanmean),
    "relative_humidity_2m": ("humidity_mean", np.nanmean),
    "precipitation": ("precip_sum", np.nansum),
    "wind_speed_10m": ("wind_mean", np.nanmean),
    "wind_gusts_10m": ("gust_max", np.nanmax),
}


def check_wind_units(units: dict) -> None:
    """Rejects a response whose wind is not in km/h (cause of the incident: docs/incident.md)."""
    for col in ("wind_speed_10m", "wind_gusts_10m"):
        if units.get(col) != "km/h":
            raise ValueError(f"unexpected unit for {col}: {units.get(col)!r} (expected: km/h)")


def wind_median_problem(wind: pd.Series, bounds: list[float]) -> str | None:
    """Rejection reason if the forecast wind median falls outside the historical range, else None."""
    median = float(wind.median())
    if bounds[0] <= median <= bounds[1]:
        return None
    return f"wind median {median:.1f} km/h outside the range [{bounds[0]:g}, {bounds[1]:g}]"


def fetch_hourly(src: str, event_id: str, lat: float, lon: float, start, end, hourly: list[str], p: dict) -> pd.DataFrame:
    url, raw_dir = SOURCES[src]
    data = get_json(
        url,
        raw_dir / f"{event_id}.json",
        {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": ",".join(hourly),
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        },
        timeout=p["timeout_s"],
        interval=p["request_interval_s"],
    )
    check_wind_units(data.get("hourly_units", {}))
    h = pd.DataFrame(data["hourly"])
    h["time"] = pd.to_datetime(h["time"], utc=True)
    return h.set_index("time").astype(float)


def aggregate_window(hourly: pd.DataFrame, tee_times: pd.Series, window_h: int, prefix: str) -> pd.DataFrame:
    """One row per tee time: hourly aggregates over the playing window.

    Players in the same group share their tee time: compute once per unique
    tee time, then redistribute.
    """
    idx = hourly.index.tz_convert(None).to_numpy()
    uniq = pd.Series(tee_times.dropna().unique())
    out = []
    for t in uniq:
        start = np.datetime64(t.floor("h").tz_convert(None))
        win = hourly[(idx >= start) & (idx < start + np.timedelta64(window_h, "h"))]
        row = {"tee_time_utc": t}
        for var, (name, fn) in AGG.items():
            vals = win[var].to_numpy(dtype=float) if var in win else np.array([])
            vals = vals[~np.isnan(vals)]
            row[f"{prefix}_{name}"] = float(fn(vals)) if len(vals) else np.nan
        out.append(row)
    cols = ["tee_time_utc"] + [f"{prefix}_{n}" for n, _ in AGG.values()]
    per_tee = pd.DataFrame(out, columns=cols)
    merged = tee_times.rename("tee_time_utc").to_frame().merge(per_tee, on="tee_time_utc", how="left")
    merged.index = tee_times.index
    return merged.drop(columns="tee_time_utc")


def main() -> None:
    p_all = params()
    p, w = p_all["collect"], p_all["weather"]
    rounds = pd.read_parquet(ROUNDS)
    courses = pd.read_parquet(COURSES).set_index("course_id")

    frames, failed = [], []
    for i, (eid, g) in enumerate(rounds.groupby("event_id", sort=False), 1):
        course = courses.loc[g.course_id.iloc[0]]
        tees = g.tee_time_utc.dropna()
        base = g.event_start.iloc[0]
        start = tees.min() if len(tees) else base
        end = (tees.max() if len(tees) else base + pd.Timedelta(days=4)) + pd.Timedelta(hours=w["window_hours"] + 1)
        parts = [g[["event_id", "player_id", "round"]]]
        for src in SOURCES:
            try:
                hourly = fetch_hourly(src, eid, course.lat, course.lon, start, end, w["hourly"], p)
            except Exception as exc:  # noqa: BLE001
                failed.append((eid, src, str(exc)))
                print(f"  ! {eid} {src}: {exc}", file=sys.stderr)
                hourly = pd.DataFrame(columns=list(AGG), dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
            parts.append(aggregate_window(hourly, g.tee_time_utc, w["window_hours"], src))
        frames.append(pd.concat(parts, axis=1))
        if i % 25 == 0:
            print(f"[{i}/{rounds.event_id.nunique()}] {g.event_name.iloc[0]} {base:%Y}")

    df = pd.concat(frames, ignore_index=True)
    for name, _ in AGG.values():
        df[f"diff_{name}"] = df[f"obs_{name}"] - df[f"fc_{name}"]

    problem = wind_median_problem(df["fc_wind_mean"], w["wind_median_kmh"])
    if problem:
        sys.exit(f"weather backfill rejected: {problem}")
    WEATHER_ROUNDS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WEATHER_ROUNDS, index=False)
    year = rounds.set_index(["event_id", "player_id", "round"]).year
    cov = df.set_index(["event_id", "player_id", "round"]).join(year).groupby("year")[
        ["fc_wind_mean", "obs_wind_mean"]
    ].agg(lambda s: round(100 * s.notna().mean(), 1))
    print(f"{len(df):,} rows -> {WEATHER_ROUNDS}")
    print("% of rounds with weather (forecast / observed):")
    print(cov.to_string())
    print("observed - forecast gap, mean wind (km/h) per year:")
    by_year = df.join(year, on=["event_id", "player_id", "round"]).groupby("year").diff_wind_mean
    print(by_year.describe()[["mean", "std"]].round(2).to_string())
    if failed:
        sys.exit(f"{len(failed)} failed call(s): rerun to retry.")


if __name__ == "__main__":
    main()
