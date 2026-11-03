"""Étape 4 : géocodage des parcours (latitude/longitude) via Nominatim (OpenStreetMap).

Source : `events[0].courses[].address` des leaderboards (ville, état, pays, code postal).
Une précision « ville » (~5-10 km) suffit : la grille ERA5 fait ~25 km.
Les erreurs se corrigent à la main dans data/courses_override.csv (prioritaire).
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from pga.config import COURSES, COURSES_OVERRIDE, RAW_GEOCODE, RAW_LEADERBOARD, ROUNDS, params
from pga.http import get_json

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def collect_courses(event_ids: set[str]) -> pd.DataFrame:
    """Tous les parcours (hôte et secondaires) des tournois retenus."""
    rows = {}
    for eid in sorted(event_ids):
        ev = json.loads((RAW_LEADERBOARD / f"{eid}.json").read_text())["events"][0]
        for c in ev.get("courses") or []:
            addr = c.get("address") or {}
            rows[str(c["id"])] = {
                "course_id": str(c["id"]),
                "course_name": c.get("name"),
                "city": addr.get("city"),
                "state": addr.get("state"),
                "country": addr.get("country"),
                "zip_code": addr.get("zipCode"),
            }
    return pd.DataFrame(rows.values())


def _queries(row) -> list[dict]:
    """Requêtes de la plus précise à la plus large."""
    country = row.country if isinstance(row.country, str) else None
    state = row.state if isinstance(row.state, str) else None
    city = row.city if isinstance(row.city, str) else None
    qs = []
    if city:
        qs.append({"city": city, "state": state, "country": country})
        qs.append({"q": ", ".join(x for x in (city, state, country) if x)})
    if isinstance(row.zip_code, str) and row.zip_code:
        qs.append({"postalcode": row.zip_code, "country": country})
    qs.append({"q": row.course_name})
    return [{k: v for k, v in q.items() if v} for q in qs]


def geocode(row, interval: float, timeout: float) -> dict | None:
    for i, q in enumerate(_queries(row)):
        res = get_json(
            NOMINATIM_URL,
            RAW_GEOCODE / f"{row.course_id}_{i}.json",
            {**q, "format": "jsonv2", "limit": 1},
            timeout=timeout,
            interval=max(interval, 1.0),  # politique d'usage Nominatim : 1 req/s max
        )
        # Nominatim renvoie une liste ; get_json la stocke telle quelle
        if res:
            hit = res[0]
            return {"lat": float(hit["lat"]), "lon": float(hit["lon"]), "geo_source": f"nominatim:{list(q)}"}
    return None


def main() -> None:
    p = params()["collect"]
    event_ids = set(pd.read_parquet(ROUNDS, columns=["event_id"]).event_id)
    courses = collect_courses(event_ids)

    overrides = pd.read_csv(COURSES_OVERRIDE, dtype={"course_id": str}).set_index("course_id")
    out = []
    for row in courses.itertuples():
        if row.course_id in overrides.index:
            o = overrides.loc[row.course_id]
            geo = {"lat": float(o.lat), "lon": float(o.lon), "geo_source": "override"}
        else:
            geo = geocode(row, p["request_interval_s"], p["timeout_s"])
        out.append({**row._asdict(), **(geo or {"lat": None, "lon": None, "geo_source": None})})
    df = pd.DataFrame(out).drop(columns="Index")

    COURSES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(COURSES, index=False)
    missing = df[df.lat.isna()]
    print(f"{len(df)} parcours géocodés -> {COURSES} ({len(missing)} manquants)")
    print(df.geo_source.str.split(":").str[0].value_counts(dropna=False).to_string())
    if len(missing):
        print(missing[["course_id", "course_name", "city", "state", "country"]].to_string())
        sys.exit("Parcours sans coordonnées : les ajouter dans data/courses_override.csv")


if __name__ == "__main__":
    main()
