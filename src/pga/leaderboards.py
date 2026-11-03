"""Étape 2 : téléchargement des leaderboards ESPN (JSON bruts, un fichier par tournoi).

Idempotent : un tournoi dont le JSON est déjà sur disque n'est pas re-téléchargé.
"""
from __future__ import annotations

import sys

import pandas as pd

from pga.config import EVENTS, RAW_LEADERBOARD, params
from pga.http import get_json

LEADERBOARD_URL = "https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard"


def main() -> None:
    p = params()["collect"]
    events = pd.read_parquet(EVENTS)
    failed = []
    for i, ev in enumerate(events.itertuples(), 1):
        path = RAW_LEADERBOARD / f"{ev.event_id}.json"
        cached = path.exists()
        try:
            get_json(
                LEADERBOARD_URL,
                path,
                {"event": ev.event_id},
                timeout=p["timeout_s"],
                interval=p["request_interval_s"],
            )
        except Exception as exc:  # noqa: BLE001 - on note l'échec et on continue
            failed.append((ev.event_id, ev.event_name, str(exc)))
            print(f"  ! {ev.event_id} {ev.event_name}: {exc}", file=sys.stderr)
            continue
        if not cached:
            print(f"[{i}/{len(events)}] {ev.start:%Y-%m-%d} {ev.event_name}")
    print(f"{len(events) - len(failed)}/{len(events)} leaderboards dans {RAW_LEADERBOARD}")
    if failed:
        sys.exit(f"{len(failed)} échec(s) : relancer le script pour réessayer.")


if __name__ == "__main__":
    main()
