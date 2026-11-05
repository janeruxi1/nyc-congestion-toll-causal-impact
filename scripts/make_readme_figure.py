"""Draw the README's summary figure from the recorded results (no data needed).

    python scripts/make_readme_figure.py   ->  reports/figures/00_readme_summary.png
"""
import csv
from pathlib import Path

# --- shared README figure style (same block in every repo) ---------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def save(fig, path):
    fig.tight_layout(w_pad=3)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("wrote", path.name)
# ---------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"


def rows(name):
    with open(M / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(x):
    return float(x) if x not in ("", None) else None


def pct(x):
    return f"{100 * x:+.1f}%".replace("-", "−")


fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.25, 1.15, 0.8]})

ax = axes[0]
meth = rows("methods_comparison.csv")
short = {"DiD: 2025 vs 2024": "Diff-in-diff, 2025 vs 2024", "DiD: 2024 vs 2023 (no toll)": "Same, 2024 vs 2023 (no toll)",
         "Synthetic control (standard)": "Synthetic control", "DiD with zone trends": "Diff-in-diff + zone trends (main)",
         "DiD, two-step (pre-toll zone trends)": "Diff-in-diff, pre-toll trends",
         "Forecast: CBD vs its own forecast": "Forecast counterfactual",
         "Forecast: CBD gap minus Manhattan gap": "Forecast, vs rest of Manhattan",
         "Synthetic control (trend-adjusted)": "Synthetic control, trend-adjusted"}
for i, r in enumerate(meth[::-1]):
    trend = r["accounts_for_trend"] == "True"
    c = BLUE if trend else GREY
    e, lo, hi = num(r["effect"]), num(r["low"]), num(r["high"])
    if lo is not None:
        ax.plot([100 * lo, 100 * hi], [i, i], color=c, lw=2.5, solid_capstyle="round")
    ax.plot(100 * e, i, "o", ms=9, color=c, mec="white", mew=2)
    ax.text(100 * (hi if hi is not None else e) + 0.35, i, pct(e), va="center", fontsize=10)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(range(len(meth)), [short.get(r["method"], r["method"]) for r in meth[::-1]])
ax.set_xlim(-8.5, 4.5)
ax.set_xlabel("effect on trips starting in the toll zone, year one (%, 95% interval)")
ax.grid(axis="y", visible=False)
ax.set_title("1 · Ignore the zone's pre-toll decline and trips\n\"fall\"; account for it (blue) and they don't")

ax = axes[1]
keep = [("Uber/Lyft", "Shared-ride requests", "Uber/Lyft shared-ride requests"),
        ("Uber/Lyft", "Average trip speed", "Uber/Lyft trip speed"),
        ("Yellow taxi", "Average trip speed", "Yellow taxi trip speed"),
        ("Uber/Lyft", "Driver pay per hour on trip", "Uber/Lyft driver pay per hour"),
        ("Taxi + Uber/Lyft", "Trips (main estimate)", "Trips (main estimate)")]
out = {(r["service"], r["outcome"]): r for r in rows("did_other_outcomes.csv")}
for i, (svc, oc, lab) in enumerate(keep[::-1]):
    r = out[(svc, oc)]
    e, lo, hi, band = (num(r[k]) for k in ("effect", "ci_low", "ci_high", "fake_rms"))
    ax.fill_betweenx([i - 0.3, i + 0.3], -100 * band, 100 * band, color="#f1f0ed", lw=0, zorder=0)
    c = BLUE if r["clear"] == "True" else GREY
    ax.plot([100 * lo, 100 * hi], [i, i], color=c, lw=2.5, solid_capstyle="round")
    ax.plot(100 * e, i, "o", ms=9, color=c, mec="white", mew=2)
    ax.text(100 * max(hi, band) + 0.6, i, pct(e), va="center", fontsize=10)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(range(len(keep)), [k[2] for k in keep[::-1]])
ax.set_xlim(-19, 11)
ax.set_xlabel("trend-adjusted change in 2025 (%, 95% interval)")
ax.grid(axis="y", visible=False)
ax.set_title("2 · What did change: faster trips,\nhigher pay per hour, fewer shared rides")

ax = axes[2]
fake = rows("did_fake_dates.csv")
for j, (col, lab, c) in enumerate((("quick", "Quick 2-year\ncomparison", GREY),
                                    ("trend_adjusted", "With zone\ntrends", BLUE))):
    ys = [100 * float(r[col]) for r in fake]
    ax.scatter([j] * len(ys), ys, s=70, color=c, edgecolor="white", linewidth=1.5, zorder=3)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks([0, 1], ["Quick 2-year\ncomparison", "With zone\ntrends"])
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("\"effect\" found on a fake 2023 toll date (%)")
ax.grid(axis="x", visible=False)
ax.set_title(f"3 · On {len(fake)} fake toll dates the quick\nmethod finds a \"drop\" every time")

save(fig, ROOT / "reports" / "figures" / "00_readme_summary.png")


# ---- hero: event study against the pre-toll trend ------------------------------------------------
import datetime as _dt  # noqa: E402

es = rows("event_study.csv")
d = [_dt.date.fromisoformat(r["start_week"]) for r in es]
y = [100 * float(r["pct"]) for r in es]
lo = [100 * float(r["pct_ci_low"]) for r in es]
hi = [100 * float(r["pct_ci_high"]) for r in es]
tr = [100 * float(r["pre_trend"]) for r in es]
post = [int(r["bin"]) >= 0 for r in es]
fig, ax = plt.subplots(figsize=(11, 4.6))
toll = _dt.date(2025, 1, 5)
ax.axvspan(toll, d[-1] + _dt.timedelta(days=60), color="#f1f0ed", lw=0)
ax.text(toll + _dt.timedelta(days=12), 15.5, "toll in force", color=MUTED, fontsize=10)
ax.plot(d, tr, color=ORANGE, lw=2, ls=(0, (5, 3)), label="the zone's pre-toll decline, extended")
for di, yi, low, high, p in zip(d, y, lo, hi, post):
    c = BLUE if p else GREY
    ax.plot([di, di], [low, high], color=c, lw=2, solid_capstyle="round")
    ax.plot(di, yi, "o", ms=8, color=c, mec="white", mew=1.5)
ax.plot([], [], "o", color=BLUE, label="after the toll (95% interval)")
ax.plot([], [], "o", color=GREY, label="before the toll")
ax.axhline(0, color=MUTED, lw=1)
ax.set_ylim(-14, 18)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%".replace("-", "−").replace("+0", "0")))
ax.set_ylabel("trips starting in the zone vs the\nrest of the city, against late 2024")
ax.legend(frameon=False, fontsize=10, loc="lower left")
ax.set_title("Trips in the toll zone kept to the decline they were already on: no break at the toll in 2025\n"
             "(the early-2026 dip looks Manhattan-wide and is flagged to watch, not attributed to the toll)")
save(fig, ROOT / "reports" / "figures" / "00_readme_hero.png")
