"""Stage 6: player-round feature table (data/processed/features.parquet).

Anti-leakage rule: a round's features only use rounds played **before** it
(earlier rounds of the same tournament included: we predict the day before the round).
Course history and course difficulty only use **previous editions**.
Weather is the **forecast** weather (fc_*); observed weather (obs_*, diff_*) is reserved
for drift monitoring.

`FEATURES` and `TARGET` are the single schema shared by training, the API and
monitoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pga.config import FEATURES_PATH, ROUNDS, WEATHER_ROUNDS, params

KEYS = ["event_id", "player_id", "round"]
TARGET = "to_par"
META = ["event_name", "event_start", "tee_time_utc", "player_name", "course_id", "year", "split"]

FORM = ["form_rel_short", "form_rel_long", "form_rel_xlong", "form_ewm", "form_ewm_slow",
        "n_rounds_hist", "days_since_last", "is_new_player"]
COURSE = ["course_rel_mean", "course_n_rounds", "course_diff_mean", "course_diff_std", "is_new_course"]
WEATHER = ["fc_temp_mean", "fc_humidity_mean", "fc_precip_sum", "fc_wind_mean", "fc_gust_max"]
CONTEXT = ["round", "major", "multi_course", "yards", "host_par"]
FEATURES = FORM + COURSE + WEATHER + CONTEXT

# Columns that must never enter the model (known after the round).
FORBIDDEN = {"to_par", "strokes", "made_cut", "status", "par_played"}


def add_field_relative(rounds: pd.DataFrame) -> pd.DataFrame:
    """rel = to_par - field mean over (tournament, round): form independent of the course."""
    out = rounds.copy()
    out["rel"] = out[TARGET] - out.groupby(["event_id", "round"])[TARGET].transform("mean")
    return out


def player_form(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Player form, computed on their strictly earlier rounds.

    Short / long / EWM = recent form; xlong / slow EWM = underlying level.
    """
    df = df.sort_values(["tee_time_utc", "event_id", "round", "player_id"], kind="stable")
    g = df.groupby("player_id", sort=False)
    prev_rel = g["rel"].shift(1)
    gp = prev_rel.groupby(df["player_id"], sort=False)
    out = pd.DataFrame(index=df.index)
    mp, mpx = p["form_min_periods"], p["form_min_periods_xlong"]
    out["form_rel_short"] = gp.transform(lambda s: s.rolling(p["form_window_short"], min_periods=mp).mean())
    out["form_rel_long"] = gp.transform(lambda s: s.rolling(p["form_window_long"], min_periods=mp).mean())
    out["form_rel_xlong"] = gp.transform(lambda s: s.rolling(p["form_window_xlong"], min_periods=mpx).mean())
    out["form_ewm"] = gp.transform(lambda s: s.ewm(halflife=p["form_ewm_halflife"], ignore_na=True).mean())
    out["form_ewm_slow"] = gp.transform(lambda s: s.ewm(halflife=p["form_ewm_halflife_slow"], ignore_na=True).mean())
    out["n_rounds_hist"] = g.cumcount()
    prev_tee = g["tee_time_utc"].shift(1)
    out["days_since_last"] = (df["tee_time_utc"] - prev_tee).dt.total_seconds() / 86400
    out["is_new_player"] = out["n_rounds_hist"] == 0
    return out.sort_index()


def _prior_cumsum(agg: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    """Cumulative sum over strictly earlier years (agg sorted by year)."""
    g = agg.groupby(by, sort=False)
    return pd.DataFrame({c: g[c].cumsum() - agg[c] for c in cols}, index=agg.index)


def course_history(df: pd.DataFrame, shrink_k: float) -> pd.DataFrame:
    """Player history on the course and course difficulty (previous editions)."""
    # Player x course: mean of rel shrunk towards 0 with k pseudo-rounds.
    pc = (
        df.groupby(["player_id", "course_id", "year"])["rel"]
        .agg(rel_sum="sum", rel_n="count")
        .reset_index()
        .sort_values("year")
    )
    pc[["rel_sum", "rel_n"]] = _prior_cumsum(pc, ["player_id", "course_id"], ["rel_sum", "rel_n"])
    pc["course_rel_mean"] = pc["rel_sum"] / (pc["rel_n"] + shrink_k)
    pc["course_n_rounds"] = pc["rel_n"]

    # Course: mean and standard deviation of to_par over previous editions.
    tmp = df.assign(sq=df[TARGET] ** 2)
    cy = (
        tmp.groupby(["course_id", "year"])
        .agg(s=(TARGET, "sum"), sq=("sq", "sum"), n=(TARGET, "count"))
        .reset_index()
        .sort_values("year")
    )
    cy[["s", "sq", "n"]] = _prior_cumsum(cy, ["course_id"], ["s", "sq", "n"])
    n = cy["n"].where(cy["n"] > 0)
    cy["course_diff_mean"] = cy["s"] / n
    var = (cy["sq"] / n - cy["course_diff_mean"] ** 2) * n / (n - 1)
    cy["course_diff_std"] = np.sqrt(var.clip(lower=0))
    cy["is_new_course"] = cy["n"] == 0

    out = df[["player_id", "course_id", "year"]].merge(
        pc[["player_id", "course_id", "year", "course_rel_mean", "course_n_rounds"]],
        on=["player_id", "course_id", "year"],
        how="left",
    ).merge(
        cy[["course_id", "year", "course_diff_mean", "course_diff_std", "is_new_course"]],
        on=["course_id", "year"],
        how="left",
    )
    out.index = df.index
    return out[COURSE]


def assign_split(year: pd.Series, split: dict) -> pd.Series:
    out = pd.Series("other", index=year.index)
    for name, (lo, hi) in split.items():
        out[year.between(lo, hi)] = name
    return out


def build_features(rounds: pd.DataFrame, weather: pd.DataFrame, p: dict, split: dict) -> pd.DataFrame:
    df = add_field_relative(rounds.reset_index(drop=True))
    form = player_form(df, p)
    course = course_history(df, p["course_shrink_k"])
    df = pd.concat([df, form, course], axis=1)
    df = df.merge(weather[KEYS + WEATHER], on=KEYS, how="left")
    df["split"] = assign_split(df["year"], split)
    for c in ["major", "multi_course", "is_new_player", "is_new_course"]:
        df[c] = df[c].astype(bool)
    cols = KEYS + META + [c for c in FEATURES if c not in KEYS] + [TARGET]
    return df[cols]


def main() -> None:
    p_all = params()
    rounds = pd.read_parquet(ROUNDS)
    weather = pd.read_parquet(WEATHER_ROUNDS)
    feats = build_features(rounds, weather, p_all["features"], p_all["split"])
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATURES_PATH, index=False)

    print(f"{len(feats):,} rows x {len(FEATURES)} features -> {FEATURES_PATH}")
    print("% missing values per split:")
    print((feats.groupby("split")[FEATURES].agg(lambda s: s.isna().mean()) * 100).round(1).T.to_string())
    ref = feats[feats.split == "reference"]
    num = ref[FEATURES].astype(float)
    print("correlation with to_par (reference):")
    print(num.corrwith(ref[TARGET]).round(3).sort_values().to_string())


if __name__ == "__main__":
    main()
