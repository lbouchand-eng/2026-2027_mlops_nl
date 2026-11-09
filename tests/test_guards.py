"""Tests des garde-fous : unité du vent, profil de référence du modèle servi."""
import numpy as np
import pandas as pd
import pytest

from pga import monitor as M
from pga.profile import build_profile
from pga.weather import check_wind_units, wind_median_problem


def test_wind_in_kmh_is_accepted():
    check_wind_units({"wind_speed_10m": "km/h", "wind_gusts_10m": "km/h"})


@pytest.mark.parametrize("units", [{"wind_speed_10m": "m/s", "wind_gusts_10m": "m/s"}, {}])
def test_other_wind_units_are_refused(units):
    with pytest.raises(ValueError, match="km/h"):
        check_wind_units(units)


def test_wind_median_bounds():
    assert wind_median_problem(pd.Series([10.0, 11.0, 12.0]), [6, 20]) is None
    assert "3.0" in wind_median_problem(pd.Series([3.0, 3.0]), [6, 20])


def frame(loc: float, n: int = 5000) -> pd.DataFrame:
    return pd.DataFrame({"x": np.random.default_rng(0).normal(loc, 1.0, n), "empty": np.nan})


def test_profile_gives_the_same_psi_as_the_raw_data():
    ref, cur = frame(0.0), frame(0.4)
    from_data = M.Reference(ref, ["x", "empty"], 10)
    from_profile = M.Reference(None, ["x", "empty"], 10, profile=build_profile(ref, ["x", "empty"], 10))
    assert from_profile.drift(cur)["x"] == pytest.approx(from_data.drift(cur)["x"])
    assert np.isnan(from_profile.drift(cur)["empty"])


def test_champion_without_profile_uses_the_fixed_period():
    df = frame(0.0).assign(year=2021)
    champ = M.Champion("m", "1", 2.0, 0.85, "tag")
    p = {"reference_years": [2021, 2021], "psi_features": ["x"], "psi_bins": 10}
    assert M.champion_reference(champ, df, p).data is not None
    with_profile = M.Champion("m", "1", 2.0, 0.85, "tag", profile=build_profile(df, ["x"], 10))
    assert M.champion_reference(with_profile, df, p).data is None
