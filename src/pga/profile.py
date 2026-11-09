"""Profil de référence d'un modèle : la distribution des données sur lesquelles il a appris.

Enregistré avec chaque run (`reference_profile.json`) et relu par le monitoring : le PSI
compare ainsi la production à ce que le modèle servi a réellement vu. Après un rollback,
la référence suit le champion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_ARTIFACT = "reference_profile.json"


def make_edges(ref, bins: int = 10) -> np.ndarray:
    """Points de coupure internes = quantiles de la référence (bornes extrêmes ouvertes)."""
    ref = np.asarray(ref, dtype=float)
    ref = ref[~np.isnan(ref)]
    if len(ref) == 0:
        return np.array([])
    return np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])


def shares(x, edges) -> np.ndarray:
    """Part des valeurs dans chaque intervalle (NaN ignorés, planchers pour le log du PSI)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
    return np.clip(counts / max(len(x), 1), 1e-4, None)


def build_profile(frame: pd.DataFrame, features: list[str], bins: int) -> dict:
    """Bornes et parts par feature ; `shares` vaut None pour une feature entièrement manquante."""
    out = {}
    for f in features:
        values = frame[f].astype(float).to_numpy()
        edges = make_edges(values, bins)
        known = not np.isnan(values).all()
        out[f] = {"edges": edges.tolist(), "shares": shares(values, edges).tolist() if known else None}
    return {"bins": bins, "n_rows": len(frame), "features": out}
