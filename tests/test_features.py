"""Tests de l'étape features : calculs et absence de fuite d'information."""
import numpy as np
import pandas as pd
import pytest

from pga.features import FEATURES, FORBIDDEN, WEATHER, add_field_relative, build_features

P = {
    "form_window_short": 2,
    "form_window_long": 4,
    "form_window_xlong": 6,
    "form_ewm_halflife": 2,
    "form_ewm_halflife_slow": 4,
    "form_min_periods": 1,
    "form_min_periods_xlong": 1,
    "course_shrink_k": 0,
}
COLS = [c for c in FEATURES if c != "round"]
SPLIT = {"history": [2020, 2020], "reference": [2021, 2021]}


def make_rounds() -> pd.DataFrame:
    """2 tournois (même parcours, 2020 puis 2021), 2 tours chacun, 3 joueurs."""
    rows = []
    scores = {  # (année, tour) -> to_par des joueurs A, B, C
        (2020, 1): [-2, 0, 2],
        (2020, 2): [-1, 1, 3],
        (2021, 1): [-3, 0, 0],
        (2021, 2): [0, 0, 3],
    }
    for (year, rnd), vals in scores.items():
        start = pd.Timestamp(f"{year}-03-05", tz="UTC")
        for pid, v in zip("ABC", vals, strict=True):
            rows.append(
                {
                    "event_id": f"E{year}",
                    "event_name": f"Open {year}",
                    "event_start": start,
                    "tee_time_utc": start + pd.Timedelta(days=rnd - 1, hours=12),
                    "player_id": pid,
                    "player_name": pid,
                    "round": rnd,
                    "to_par": v,
                    "course_id": "C1",
                    "year": year,
                    "major": False,
                    "multi_course": False,
                    "yards": 7200,
                    "host_par": 72,
                }
            )
    return pd.DataFrame(rows)


def make_weather(rounds: pd.DataFrame) -> pd.DataFrame:
    w = rounds[["event_id", "player_id", "round"]].copy()
    for c in WEATHER:
        w[c] = 1.0
    return w


def build(rounds: pd.DataFrame) -> pd.DataFrame:
    out = build_features(rounds, make_weather(rounds), P, SPLIT)
    return out.set_index(["event_id", "player_id", "round"]).sort_index()


def test_field_relative():
    r = add_field_relative(make_rounds())
    first = r[(r.event_id == "E2020") & (r["round"] == 1)].set_index("player_id").rel
    assert first.to_dict() == {"A": -2.0, "B": 0.0, "C": 2.0}
    assert r.groupby(["event_id", "round"]).rel.sum().abs().max() < 1e-9


def test_first_round_has_no_history():
    f = build(make_rounds())
    first = f.xs(("E2020", 1), level=["event_id", "round"])
    assert first.form_rel_short.isna().all()
    assert (first.n_rounds_hist == 0).all()
    assert first.is_new_player.all()
    assert first.is_new_course.all()
    assert first.course_diff_mean.isna().all()


def test_form_uses_previous_rounds_only():
    f = build(make_rounds())
    # A au 1er tour 2021 : rel antérieurs = -2 (2020 R1), -2 (2020 R2) ; fenêtre courte = 2
    assert f.loc[("E2021", "A", 1), "form_rel_short"] == pytest.approx(-2.0)
    assert f.loc[("E2021", "A", 1), "n_rounds_hist"] == 2
    assert f.loc[("E2021", "A", 1), "form_rel_xlong"] == pytest.approx(-2.0)
    assert np.isnan(f.loc[("E2020", "A", 1), "form_ewm_slow"])
    # A au 2e tour 2021 : rel antérieurs -2 (2020 R2) et -2 (2021 R1)
    assert f.loc[("E2021", "A", 2), "form_rel_short"] == pytest.approx(-2.0)
    assert f.loc[("E2021", "A", 2), "days_since_last"] == pytest.approx(1.0)


def test_changing_future_round_does_not_change_past_features():
    base = build(make_rounds())
    r = make_rounds()
    r.loc[(r.event_id == "E2021") & (r["round"] == 2), "to_par"] = [10, -10, 7]
    changed = build(r)
    past = base.index.get_level_values("event_id") == "E2020"
    past |= base.index.get_level_values("round") == 1
    pd.testing.assert_frame_equal(base.loc[past, COLS], changed.loc[past, COLS])


def test_course_history_uses_previous_editions_only():
    f = build(make_rounds())
    # Parcours C1 en 2021 : uniquement 2020 -> to_par = [-2, 0, 2, -1, 1, 3]
    row = f.loc[("E2021", "A", 2)]
    assert row.course_diff_mean == pytest.approx(0.5)
    assert row.course_diff_std == pytest.approx(np.std([-2, 0, 2, -1, 1, 3], ddof=1))
    # A sur C1 en 2020 : rel = -2, -2 ; k = 0 -> moyenne -2, même au 2e tour 2021
    assert row.course_rel_mean == pytest.approx(-2.0)
    assert row.course_n_rounds == 2
    assert not row.is_new_course


def test_split_and_schema():
    f = build(make_rounds())
    assert set(f.split) == {"history", "reference"}
    assert not FORBIDDEN & set(FEATURES)
    assert not any(c.startswith(("obs_", "diff_")) for c in FEATURES)
    assert len(FEATURES) == len(set(FEATURES))
