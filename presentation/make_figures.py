"""Graphiques de la présentation qui n'existent pas déjà dans reports/.

Sources : reports/metrics.json et reports/monitoring/timeline_incident.csv.
Usage : uv run python presentation/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "figures"

# Même code couleur que les timelines de pga.monitor
NAVY = "#1f3b73"
ALERT = "#9b1c1c"
GREY = "#9aa3ad"
INK = "#1d2330"
MUTED = "#5b6472"

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 15,
    "axes.edgecolor": "#c9ced6",
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e6e9ee",
    "grid.linewidth": 1,
    "legend.frameon": False,
    "axes.axisbelow": True,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def fr(x: float, nd: int = 2, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def metrics_par_annee() -> None:
    """MAE (modèle vs baseline) et biais par année : la MAE tient, le biais dérive."""
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    years = [("2022\n(validation)", "valid"), ("2023", "eval_2023"), ("2024", "eval_2024")]
    labels = [y for y, _ in years]
    mae = [m[f"{k}_mae"] for _, k in years]
    base = [m[f"{k}_mae_baseline"] for _, k in years]
    bias = [m[f"{k}_bias"] for _, k in years]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.3})
    x = list(range(len(years)))
    for vals, color, name, dy in [(base, GREY, "baseline (moyenne du parcours)", 10),
                                  (mae, NAVY, "modèle", -22)]:
        ax1.plot(x, vals, color=color, lw=2, marker="o", ms=10, label=name)
        for i, v in zip(x, vals):
            ax1.annotate(fr(v), (i, v), xytext=(0, dy), textcoords="offset points",
                         ha="center", fontsize=13, color=INK)
    ax1.set_xticks(x, labels)
    ax1.set_xlim(-0.4, 2.4)
    ax1.set_ylim(2.25, 2.65)
    ax1.set_ylabel("MAE (coups par tour)")
    ax1.set_title("La MAE tient")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=13)

    bars = ax2.bar(list(x), bias, 0.5, color=NAVY)
    for r, v in zip(bars, bias):
        ax2.text(r.get_x() + r.get_width() / 2, v + 0.006, fr(v, sign=True),
                 ha="center", va="bottom", fontsize=13, color=INK)
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 0.3)
    ax2.set_ylabel("biais moyen (coups, prédit − réel)")
    ax2.set_title("…mais le biais dérive")

    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(COMMA)
    fig.savefig(OUT / "metrics_par_annee.png")
    plt.close(fig)


def short(name: str) -> str:
    return name.split(" pres")[0].replace("Texas Children's ", "")


def zscore_vs_cusum() -> None:
    """Injection « vent prévu +15 km/h » au 1er mars 2024 : z-score de la fenêtre vs CUSUM."""
    t = pd.read_csv(ROOT / "reports" / "monitoring" / "timeline_incident.csv",
                    parse_dates=["event_start"])
    t = t[(t["event_start"] >= "2023-12-20") & (t["event_start"] <= "2024-04-12")]
    since = pd.Timestamp("2024-03-01", tz="UTC")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True,
                                   gridspec_kw={"hspace": 0.35})
    for ax, col, warn, alert, title in [
        (ax1, "bias_z", 2.0, 3.0, "z-score du biais (moyenne sur 8 tournois)"),
        (ax2, "cusum_up", 3.0, 4.0, "CUSUM du biais (cumulé, sans fenêtre)"),
    ]:
        ax.plot(t["event_start"], t[col], color=NAVY, lw=2, marker="o", ms=6)
        ax.axhline(warn, color=ALERT, lw=1.2, ls="--")
        ax.axhline(alert, color=ALERT, lw=1.6)
        ax.text(t["event_start"].iloc[0], alert + 0.15, "seuil d'alerte", color=ALERT, fontsize=12)
        ax.axvline(since, color=INK, lw=1.2, ls=":")
        ax.set_title(title, fontsize=15)
        ax.yaxis.set_major_formatter(COMMA)
        first = t[t[col] > alert].iloc[0]
        ax.annotate(f"1re alerte : {first['event_start']:%d/%m}\n({short(first['event_name'])})",
                    (first["event_start"], first[col]), xytext=(18, -6), textcoords="offset points",
                    fontsize=12, color=INK, va="top",
                    arrowprops={"arrowstyle": "-", "color": MUTED})
    ax1.text(since, ax1.get_ylim()[1], " injection : 1er mars", va="top", fontsize=12, color=INK)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
    ax2.set_xlabel("date du tournoi contrôlé")
    fig.savefig(OUT / "zscore_vs_cusum.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_par_annee()
    zscore_vs_cusum()
    print(f"figures écrites dans {OUT.relative_to(ROOT)}")
