"""Étape 1 : calendrier des tournois PGA Tour (ESPN scoreboard).

Un seul appel par saison ESPN suffit : `leagues[0].calendar` liste tous les
événements de la saison. Les saisons chevauchent deux années civiles, on interroge
donc first_year..last_year+1 et on filtre sur la date de début.
"""
from __future__ import annotations

import pandas as pd

from pga.config import EVENTS, RAW_SCOREBOARD, params
from pga.http import get_json

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"


def parse_calendar(scoreboard: dict, season: int) -> list[dict]:
    rows = []
    for e in (scoreboard.get("leagues") or [{}])[0].get("calendar") or []:
        rows.append(
            {
                "event_id": str(e["id"]),
                "event_name": e.get("label"),
                "start": e.get("startDate"),
                "end": e.get("endDate"),
                "espn_season": season,
            }
        )
    return rows


def main() -> None:
    p = params()["collect"]
    rows = []
    for season in range(p["first_year"], p["last_year"] + 2):
        sb = get_json(
            SCOREBOARD_URL,
            RAW_SCOREBOARD / f"season_{season}.json",
            {"dates": f"{season}0601"},  # juin = toujours au milieu de la saison
            timeout=p["timeout_s"],
            interval=p["request_interval_s"],
        )
        rows += parse_calendar(sb, season)

    df = pd.DataFrame(rows).drop_duplicates("event_id")
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    df["year"] = df["start"].dt.year
    df = df[df["year"].between(p["first_year"], p["last_year"])]
    df = df.sort_values("start").reset_index(drop=True)

    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(EVENTS, index=False)
    print(f"{len(df)} événements -> {EVENTS}")
    print(df.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
