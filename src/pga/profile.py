"""A model's reference profile: the distribution of the data it learned from.

Saved with each run (`reference_profile.json`) and read back by monitoring: the PSI
thus compares production with what the served model actually saw. After a rollback,
the reference follows the champion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_ARTIFACT = "reference_profile.json"


def make_edges(ref, bins: int = 10) -> np.ndarray:
    """Inner cut points = reference quantiles (outer bounds open)."""
    ref = np.asarray(ref, dtype=float)
    ref = ref[~np.isnan(ref)]
    if len(ref) == 0:
        return np.array([])
    return np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])


def shares(x, edges) -> np.ndarray:
    """Share of values in each bin (NaN ignored, floored for the PSI log)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
    return np.clip(counts / max(len(x), 1), 1e-4, None)


def build_profile(frame: pd.DataFrame, features: list[str], bins: int) -> dict:
    """Edges and shares per feature; `shares` is None for an entirely missing feature."""
    out = {}
    for f in features:
        values = frame[f].astype(float).to_numpy()
        edges = make_edges(values, bins)
        known = not np.isnan(values).all()
        out[f] = {"edges": edges.tolist(), "shares": shares(values, edges).tolist() if known else None}
    return {"bins": bins, "n_rows": len(frame), "features": out}
