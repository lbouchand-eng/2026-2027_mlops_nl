"""Étape 5 : météo de la fenêtre de jeu de chaque tour (prévue et observée).

Par tournoi, deux appels Open-Meteo sur le parcours hôte (heures UTC) :
- historical-forecast-api : la prévision archivée (ce qu'on aurait su avant le tour)
  -> sert de variable d'entrée au modèle (pas de fuite d'information) ;
- archive-api (ERA5) : la météo observée -> sert à mesurer l'écart observé - prévu.
La prévision archivée n'existe pas pour les plus vieilles années (≈ avant 2018) :
ces lignes ont fc_* = NaN, ce qui ne gêne pas car 2016-2020 ne sert qu'à l'historique.

Agrégation sur [teeTime, teeTime + window_hours[ pour chaque joueur-tour.
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
# variable horaire -> (nom agrégé, fonction d'agrégation)
AGG = {
    "temperature_2m": ("temp_mean", np.nanmean),
    "relative_humidity_2m": ("humidity_mean", np.nanmean),
    "precipitation": ("precip_sum", np.nansum),
    "wind_speed_10m": ("wind_mean", np.nanmean),
    "wind_gusts_10m": ("gust_max", np.nanmax),
}


def check_wind_units(units: dict) -> None:
    """Refuse une réponse dont le vent n'est pas en km/h (cause de l'incident : docs/incident.md)."""
    for col in ("wind_speed_10m", "wind_gusts_10m"):
        if units.get(col) != "km/h":
            raise ValueError(f"unité inattendue pour {col} : {units.get(col)!r} (attendu : km/h)")


def wind_median_problem(wind: pd.Series, bounds: list[float]) -> str | None:
    """Motif de refus si la médiane du vent prévu sort de la plage historique, sinon None."""
    median = float(wind.median())
    if bounds[0] <= median <= bounds[1]:
        return None
    return f"médiane du vent {median:.1f} km/h hors de la plage [{bounds[0]:g}, {bounds[1]:g}]"


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
    """Une ligne par tee time : agrégats horaires sur la fenêtre de jeu.

    Les joueurs d'un même groupe partagent leur tee time : on calcule une fois par
    tee time unique puis on redistribue.
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
        sys.exit(f"backfill météo refusé : {problem}")
    WEATHER_ROUNDS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WEATHER_ROUNDS, index=False)
    year = rounds.set_index(["event_id", "player_id", "round"]).year
    cov = df.set_index(["event_id", "player_id", "round"]).join(year).groupby("year")[
        ["fc_wind_mean", "obs_wind_mean"]
    ].agg(lambda s: round(100 * s.notna().mean(), 1))
    print(f"{len(df):,} lignes -> {WEATHER_ROUNDS}")
    print("% de tours avec météo (prévue / observée) :")
    print(cov.to_string())
    print("écart observé - prévu, vent moyen (km/h) par année :")
    by_year = df.join(year, on=["event_id", "player_id", "round"]).groupby("year").diff_wind_mean
    print(by_year.describe()[["mean", "std"]].round(2).to_string())
    if failed:
        sys.exit(f"{len(failed)} appel(s) en échec : relancer pour réessayer.")


if __name__ == "__main__":
    main()
