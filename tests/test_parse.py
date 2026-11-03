import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pga.calendar import parse_calendar
from pga.rounds import parse_leaderboard, parse_to_par
from pga.weather import aggregate_window

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text())


@pytest.mark.parametrize("s,expected", [("E", 0), ("+3", 3), ("-7", -7), ("12", 12), ("--", None), (None, None)])
def test_parse_to_par(s, expected):
    assert parse_to_par(s) == expected


def test_calendar():
    rows = parse_calendar(load("espn_scoreboard_20230309.json"), 2023)
    assert len(rows) > 40
    assert {"event_id", "event_name", "start", "end", "espn_season"} <= rows[0].keys()
    assert any(r["event_name"] == "THE PLAYERS Championship" for r in rows)


def test_masters_2024():
    df, reason = parse_leaderboard(load("espn_leaderboard_401580344.json"))
    assert reason is None
    assert df.event_id.nunique() == 1 and df.course_id.iloc[0] == "21"
    dech = df[(df.player_name == "Bryson DeChambeau")].set_index("round")
    assert dech.loc[1, "strokes"] == 65 and dech.loc[1, "to_par"] == -7
    assert dech.loc[1, "tee_time_utc"] == "2024-04-11T15:18Z"
    assert (df.par_played == 72).all()
    # 89 joueurs au départ : 4 tours pour ceux qui passent le cut, 2 pour les autres
    assert df.groupby("player_id")["round"].max().isin([2, 4]).mean() > 0.95
    assert not df.duplicated(["player_id", "round"]).any()


def test_masters_2016_has_tee_times():
    df, reason = parse_leaderboard(load("espn_leaderboard_2493.json"))
    assert reason is None
    assert df.tee_time_utc.notna().mean() > 0.95
    assert df.made_cut.any() and (~df.made_cut).any()


def test_players_2023():
    df, reason = parse_leaderboard(load("espn_leaderboard_401465526.json"))
    assert reason is None and len(df) > 400


def test_excluded_by_name():
    _, reason = parse_leaderboard(load("espn_leaderboard_401580344.json"), ("Masters",))
    assert reason == "exclu par nom"


def test_non_medal_excluded():
    lb = load("espn_leaderboard_401580344.json")
    lb["events"][0]["competitions"][0]["scoringSystem"] = {"name": "Match"}
    _, reason = parse_leaderboard(lb)
    assert reason == "scoring=Match"


def test_withdrawn_last_round_dropped():
    lb = load("espn_leaderboard_401580344.json")
    c = lb["events"][0]["competitions"][0]["competitors"][0]
    c["status"]["displayValue"] = "WD"
    df, _ = parse_leaderboard(lb)
    assert sorted(df[df.player_id == c["athlete"]["id"]]["round"]) == [1, 2, 3]


def test_aggregate_window():
    times = pd.date_range("2024-04-11", periods=48, freq="h", tz="UTC")
    hourly = pd.DataFrame(
        {
            "temperature_2m": np.arange(48.0),
            "relative_humidity_2m": 50.0,
            "precipitation": 1.0,
            "wind_speed_10m": np.arange(48.0),
            "wind_gusts_10m": np.arange(48.0) * 2,
        },
        index=times,
    )
    tees = pd.Series(pd.to_datetime(["2024-04-11T10:20Z", "2024-04-11T10:20Z", None], utc=True), index=[7, 8, 9])
    out = aggregate_window(hourly, tees, 5, "fc")
    assert list(out.index) == [7, 8, 9]
    assert out.loc[7, "fc_wind_mean"] == pytest.approx(12.0)  # heures 10..14
    assert out.loc[7, "fc_gust_max"] == 28.0
    assert out.loc[7, "fc_precip_sum"] == 5.0
    assert out.loc[9].isna().all()
