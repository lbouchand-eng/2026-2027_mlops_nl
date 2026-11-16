"""Figures of the project report, with English labels.

Every figure is drawn from files already in the repository (reports/, params.yaml);
nothing is recomputed. Usage: uv run python report/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
P = yaml.safe_load((ROOT / "params.yaml").read_text())["monitor"]

# Same colour code as the timelines drawn by pga.monitor
NAVY = "#1f3b73"
ORANGE = "#d9762b"
ALERT = "#9b1c1c"
GREY = "#9aa3ad"
INK = "#1d2330"
MUTED = "#5b6472"
STATUS = {"ok": "#ffffff", "warning": "#f6d58b", "alert": "#ee8f8f"}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 12,
    "axes.edgecolor": "#c9ced6",
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e6e9ee",
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def short(name: str) -> str:
    return name.split(" pres")[0].replace("Texas Children's ", "")


def shade(ax, t: pd.Series, status: pd.Series) -> None:
    """Background colour = status of each check, as in pga.monitor."""
    for k in range(len(t)):
        right = t.iloc[k + 1] if k + 1 < len(t) else t.iloc[k] + pd.Timedelta(days=7)
        ax.axvspan(t.iloc[k], right, color=STATUS[status.iloc[k]], alpha=0.6, lw=0)


def thresholds(ax, warn: float | None, alert: float, both_sides: bool = False) -> None:
    signs = (1, -1) if both_sides else (1,)
    for s in signs:
        if warn is not None:
            ax.axhline(s * warn, color=ALERT, lw=1, ls="--")
        ax.axhline(s * alert, color=ALERT, lw=1.3)


def metrics_by_year() -> None:
    """MAE (model vs baseline) and mean bias per year: MAE holds, bias drifts."""
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    years = [("2022\n(validation)", "valid"), ("2023", "eval_2023"), ("2024", "eval_2024")]
    labels = [y for y, _ in years]
    x = list(range(len(years)))
    mae = [m[f"{k}_mae"] for _, k in years]
    base = [m[f"{k}_mae_baseline"] for _, k in years]
    bias = [m[f"{k}_bias"] for _, k in years]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.3})
    for vals, color, name, dy in [(base, GREY, "baseline (course average)", 9), (mae, NAVY, "model", -19)]:
        ax1.plot(x, vals, color=color, lw=2, marker="o", ms=8, label=name)
        for i, v in zip(x, vals, strict=True):
            ax1.annotate(f"{v:.2f}", (i, v), xytext=(0, dy), textcoords="offset points",
                         ha="center", fontsize=11, color=INK)
    ax1.set_xticks(x, labels)
    ax1.set_xlim(-0.4, 2.4)
    ax1.set_ylim(2.25, 2.65)
    ax1.set_ylabel("MAE (strokes per round)")
    ax1.set_title("MAE holds…")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    bars = ax2.bar(x, bias, 0.5, color=NAVY)
    for r, v in zip(bars, bias, strict=True):
        ax2.text(r.get_x() + r.get_width() / 2, v + 0.006, f"{v:+.2f}", ha="center", va="bottom",
                 fontsize=11, color=INK)
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 0.3)
    ax2.set_ylabel("mean bias (strokes, predicted − actual)")
    ax2.set_title("…but the bias drifts")
    fig.savefig(OUT / "metrics_by_year.png")
    plt.close(fig)


def replay_2023_2024() -> None:
    """One monitoring check per tournament over the drift period (natural drift, no injection)."""
    t = pd.read_csv(ROOT / "reports" / "monitoring" / "timeline_replay.csv", parse_dates=["event_start"])
    d = t["event_start"]
    cusum = t[["cusum_up", "cusum_down"]].max(axis=1)
    panels = [
        (t["bias_z"], "Bias z-score (last 8 tournaments)", P["bias_z_warning"], P["bias_z_alert"], True),
        (cusum, "Bias CUSUM", P["cusum_h_warning"], P["cusum_h_alert"], False),
        (t["mae_ratio"], "MAE / validation MAE", None, P["mae_ratio_alert"], False),
        (t["psi_ratio_max"], "Max PSI / alert threshold (worst feature)", None, 1.0, False),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 9), sharex=True, gridspec_kw={"hspace": 0.35})
    for ax, (y, label, warn, alert, both) in zip(axes, panels, strict=True):
        shade(ax, d, t["status"])
        ax.plot(d, y, color=NAVY, lw=1.5, marker="o", ms=3)
        thresholds(ax, warn, alert, both)
        ax.set_title(label, fontsize=11)
    q = t[t["status"] == "alert"].iloc[0]
    axes[-1].annotate("only alert: Q-School", (q["event_start"], q["psi_ratio_max"]),
                      xytext=(pd.Timestamp("2023-11-25", tz="UTC"), 0.68), textcoords="data", ha="right", fontsize=10, color=INK,
                      arrowprops={"arrowstyle": "-", "color": MUTED})
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.text(0.01, 0.0, "background: white = ok, yellow = warning, red = alert · solid line = alert threshold, "
             "dashed = warning", fontsize=9, color=MUTED)
    fig.savefig(OUT / "replay_2023_2024.png")
    plt.close(fig)


def zscore_vs_cusum() -> None:
    """Injected scenario (forecast wind +15 km/h from 1 March 2024): windowed z-score vs CUSUM."""
    t = pd.read_csv(ROOT / "reports" / "monitoring" / "timeline_incident.csv", parse_dates=["event_start"])
    t = t[(t["event_start"] >= "2023-12-20") & (t["event_start"] <= "2024-04-12")]
    since = pd.Timestamp("2024-03-01", tz="UTC")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"hspace": 0.35})
    for ax, col, warn, alert, title in [
        (ax1, "bias_z", P["bias_z_warning"], P["bias_z_alert"], "Bias z-score (mean over the last 8 tournaments)"),
        (ax2, "cusum_up", P["cusum_h_warning"], P["cusum_h_alert"], "Bias CUSUM (cumulative, no window)"),
    ]:
        ax.plot(t["event_start"], t[col], color=NAVY, lw=2, marker="o", ms=5)
        thresholds(ax, warn, alert)
        ax.text(t["event_start"].iloc[0], alert + 0.15, "alert threshold", color=ALERT, fontsize=10)
        ax.axvline(since, color=INK, lw=1.1, ls=":")
        ax.set_title(title, fontsize=12)
        first = t[t[col] > alert].iloc[0]
        ax.annotate(f"first alert: {first['event_start']:%d %b}\n({short(first['event_name'])})",
                    (first["event_start"], first[col]), xytext=(16, -4), textcoords="offset points",
                    fontsize=10, color=INK, va="top", arrowprops={"arrowstyle": "-", "color": MUTED})
    ax1.text(since, ax1.get_ylim()[1], " injection starts: 1 Mar", va="top", fontsize=10, color=INK)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax2.set_xlabel("start date of the checked tournament")
    fig.savefig(OUT / "zscore_vs_cusum.png")
    plt.close(fig)


def incident() -> None:
    """The m/s backfill incident: bias z-score of v2 vs v1, and PSI before vs after the fixes."""
    inc = ROOT / "reports" / "incident"
    v2 = pd.read_csv(inc / "timeline_detection.csv", parse_dates=["event_start"])
    v1 = pd.read_csv(inc / "timeline_apres_rollback.csv", parse_dates=["event_start"])
    fixed = pd.read_csv(ROOT / "reports" / "incident_guarded" / "timeline_detection.csv",
                        parse_dates=["event_start"])
    promoted = pd.Timestamp("2024-03-01", tz="UTC")
    rollback = pd.Timestamp("2024-03-25", tz="UTC")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"hspace": 0.45})
    ax1.plot(v2["event_start"], v2["bias_z"], color=ORANGE, lw=2, marker="o", ms=5,
             label="v2, corrupted (if never rolled back)")
    ax1.plot(v1["event_start"], v1["bias_z"], color=NAVY, lw=2, marker="o", ms=5,
             label="v1, healthy (after rollback)")
    thresholds(ax1, P["bias_z_warning"], P["bias_z_alert"])
    ax1.set_title("Bias z-score: performance saw it", fontsize=12)

    ax2.plot(fixed["event_start"], fixed["psi_ratio_max"], color=GREY, lw=2, marker="s", ms=5,
             label="reference = training data of the served model (after fix #2)")
    ax2.plot(v2["event_start"], v2["psi_ratio_max"], color=ORANGE, lw=2, marker="o", ms=5,
             label="reference = 2018-2022 data (original setup)")
    ax2.axhline(1.0, color=ALERT, lw=1.3)
    ax2.set_title("Max PSI / alert threshold", fontsize=12)

    for ax in (ax1, ax2):
        for x in (promoted, rollback):
            ax.axvline(x, color=INK, lw=1.1, ls=":")
        ax.set_ylim(bottom=0)
    ax1.set_ylim(top=7.4)
    ax2.set_ylim(top=4.6)
    for ax in (ax1, ax2):
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=10)
    for x, lab in [(promoted, " v2 promoted"), (rollback, " rollback")]:
        ax1.text(x, 7.3, lab, va="top", fontsize=10, color=INK)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.set_xlabel("start date of the checked tournament (2024)")
    fig.savefig(OUT / "incident.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_by_year()
    replay_2023_2024()
    zscore_vs_cusum()
    incident()
    print(f"figures written to {OUT.relative_to(ROOT)}")
