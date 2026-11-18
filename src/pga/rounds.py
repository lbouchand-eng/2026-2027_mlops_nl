"""Stage 3: raw leaderboards -> player-round table (data/processed/rounds.parquet).

One row = (tournament, player, round). Target: `to_par` (round score vs par).

Cleaning rules:
- only individual stroke-play tournaments are kept (scoringSystem == "Medal");
  match play, team events (Zurich, QBE…) and plain-text JSON are excluded;
- playoff rounds are excluded;
- for a WD/DQ player, the last recorded round is assumed incomplete and dropped;
- `to_par` comes from `displayValue` (computed by ESPN on the course actually played,
  important for multi-course tournaments), falling back to strokes - host par.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from pga.config import EVENTS, RAW_LEADERBOARD, ROUNDS, params

STROKES_RANGE = (55, 100)


def parse_to_par(display: str | None) -> int | None:
    if display is None:
        return None
    s = display.strip().upper()
    if s in {"E", "EVEN"}:
        return 0
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    return None


def _scoring_system(ev: dict, comp: dict) -> str | None:
    for src in (comp, ev.get("tournament") or {}):
        name = (src.get("scoringSystem") or {}).get("name")
        if name:
            return name
    return None


def parse_leaderboard(lb: dict, exclude_patterns: tuple[str, ...] = ()) -> tuple[pd.DataFrame, str | None]:
    """Returns (player-round rows, exclusion reason or None)."""
    events = lb.get("events") or []
    if not events:
        return pd.DataFrame(), "no event"
    ev = events[0]
    name = ev.get("name") or ""
    if any(p.lower() in name.lower() for p in exclude_patterns):
        return pd.DataFrame(), "excluded by name"
    comps = ev.get("competitions") or []
    comp = comps[0] if comps else None
    if not isinstance(comp, dict):
        return pd.DataFrame(), "non-standard format"
    scoring = _scoring_system(ev, comp)
    if scoring != "Medal":
        return pd.DataFrame(), f"scoring={scoring}"
    if not comp.get("competitors"):
        return pd.DataFrame(), "no players"

    courses = ev.get("courses") or []
    host = next((c for c in courses if c.get("host")), courses[0] if courses else {})
    host_par = host.get("shotsToPar")
    tournament = ev.get("tournament") or {}

    rows = []
    for c in comp["competitors"]:
        ath = c.get("athlete") or {}
        status = c.get("status") or {}
        status_name = (status.get("type") or {}).get("name")
        status_display = status.get("displayValue")
        player_rows = []
        for ls in c.get("linescores") or []:
            strokes = ls.get("value")
            if ls.get("isPlayoff") or not strokes:
                continue
            strokes = int(strokes)
            if not STROKES_RANGE[0] <= strokes <= STROKES_RANGE[1]:
                continue
            to_par = parse_to_par(ls.get("displayValue"))
            to_par_source = "espn"
            if to_par is None and host_par:
                to_par, to_par_source = strokes - int(host_par), "host_par"
            if to_par is None:
                continue
            player_rows.append(
                {
                    "round": int(ls.get("period")),
                    "strokes": strokes,
                    "to_par": to_par,
                    "par_played": strokes - to_par,
                    "to_par_source": to_par_source,
                    "tee_time_utc": ls.get("teeTime"),
                }
            )
        if status_display in {"WD", "DQ"} and player_rows:
            player_rows = player_rows[:-1]
        for r in player_rows:
            rows.append(
                {
                    "event_id": str(ev.get("id")),
                    "event_name": name,
                    "event_start": ev.get("date"),
                    "major": bool(tournament.get("major")),
                    "n_rounds": tournament.get("numberOfRounds"),
                    "course_id": str(host.get("id")) if host.get("id") else None,
                    "course_name": host.get("name"),
                    "host_par": host_par,
                    "yards": host.get("totalYards"),
                    "multi_course": len(courses) > 1,
                    "player_id": str(ath.get("id") or c.get("id")),
                    "player_name": ath.get("displayName"),
                    "amateur": bool(c.get("amateur")),
                    "status": status_display,
                    "made_cut": status_name != "STATUS_CUT",
                    **r,
                }
            )
    return pd.DataFrame(rows), None


def build(lb_dir: Path, events: pd.DataFrame, exclude_patterns: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, skipped = [], []
    for ev in events.itertuples():
        path = lb_dir / f"{ev.event_id}.json"
        if not path.exists():
            skipped.append((ev.event_id, ev.event_name, "missing JSON"))
            continue
        df, reason = parse_leaderboard(json.loads(path.read_text()), exclude_patterns)
        if reason or df.empty:
            skipped.append((ev.event_id, ev.event_name, reason or "no rounds"))
            continue
        frames.append(df)
    rounds = pd.concat(frames, ignore_index=True)
    rounds["event_start"] = pd.to_datetime(rounds["event_start"], utc=True)
    rounds["tee_time_utc"] = pd.to_datetime(rounds["tee_time_utc"], utc=True)
    rounds["year"] = rounds["event_start"].dt.year
    rounds = rounds.drop_duplicates(["event_id", "player_id", "round"])
    rounds = rounds.sort_values(["event_start", "event_id", "round", "player_id"]).reset_index(drop=True)
    return rounds, pd.DataFrame(skipped, columns=["event_id", "event_name", "reason"])


def main() -> None:
    p = params()["rounds"]
    events = pd.read_parquet(EVENTS)
    rounds, skipped = build(RAW_LEADERBOARD, events, tuple(p["exclude_name_patterns"]))
    ROUNDS.parent.mkdir(parents=True, exist_ok=True)
    rounds.to_parquet(ROUNDS, index=False)

    print(f"{len(rounds):,} player-rounds -> {ROUNDS}")
    print(f"{rounds.event_id.nunique()} tournaments kept, {len(skipped)} excluded:")
    print(skipped.groupby("reason").size().to_string())
    summary = rounds.groupby("year").agg(
        tournaments=("event_id", "nunique"),
        rounds=("to_par", "size"),
        players=("player_id", "nunique"),
        pct_tee_time=("tee_time_utc", lambda s: round(100 * s.notna().mean(), 1)),
        mean_to_par=("to_par", lambda s: round(s.mean(), 2)),
    )
    print(summary.to_string())
    bad = (rounds.to_par_source == "espn") & ~rounds.par_played.between(68, 73)
    print(f"inferred par outside [68,73]: {bad.sum()} rows")


if __name__ == "__main__":
    main()
