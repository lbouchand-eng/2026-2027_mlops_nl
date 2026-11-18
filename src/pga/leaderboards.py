"""Stage 2: download ESPN leaderboards (raw JSON, one file per tournament).

Idempotent: a tournament whose JSON is already on disk is not downloaded again.
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
        except Exception as exc:  # noqa: BLE001 - record the failure and move on
            failed.append((ev.event_id, ev.event_name, str(exc)))
            print(f"  ! {ev.event_id} {ev.event_name}: {exc}", file=sys.stderr)
            continue
        if not cached:
            print(f"[{i}/{len(events)}] {ev.start:%Y-%m-%d} {ev.event_name}")
    print(f"{len(events) - len(failed)}/{len(events)} leaderboards in {RAW_LEADERBOARD}")
    if failed:
        sys.exit(f"{len(failed)} failure(s): rerun the script to retry.")


if __name__ == "__main__":
    main()
