"""Replays production: sends the player-rounds of a period to the API (in batches).

  uv run python scripts/send_requests.py --from 2024-03-01 --to 2024-04-21 [--api-url http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import math
import sys

import pandas as pd
import requests

from pga.config import FEATURES_PATH
from pga.features import FEATURES


def to_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for r in df.itertuples(index=False):
        row = {"event_id": r.event_id, "player_id": r.player_id, "round": int(r.round)}
        for f in FEATURES:
            if f != "round":
                v = float(getattr(r, f))
                row[f] = None if math.isnan(v) else v
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    ap.add_argument("--api-url", default="http://127.0.0.1:8000")
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_parquet(FEATURES_PATH)
    lo, hi = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["event_start"] >= lo) & (df["event_start"] < hi)]
    df[FEATURES] = df[FEATURES].astype(float)
    if df.empty:
        print("no rounds in this period", file=sys.stderr)
        return 1
    rows, versions = to_rows(df), set()
    for i in range(0, len(rows), args.batch):
        r = requests.post(f"{args.api_url.rstrip('/')}/predict", json={"rows": rows[i : i + args.batch]}, timeout=60)
        r.raise_for_status()
        versions.add(r.json()["model_version"])
    print(f"{len(rows)} predictions ({df['event_id'].nunique()} tournaments, {args.start} -> {args.end}) "
          f"served by v{', v'.join(sorted(versions))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
