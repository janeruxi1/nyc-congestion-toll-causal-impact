"""Shared figure style and small helpers for the notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics"

COLORS = {
    "cbd": "#C2410C",             # toll zone
    "manhattan_north": "#1D4ED8",  # rest of Manhattan
    "outer": "#64748B",           # outer boroughs
    "border": "#A16207",
    "airport": "#7C3AED",
    "policy": "#111827",
    "band": "#FDBA74",
}
GROUP_LABEL = {
    "cbd": "Toll zone (CBD)",
    "manhattan_north": "Rest of Manhattan",
    "outer": "Outer boroughs",
    "border": "60th St border",
    "airport": "Airports",
}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "legend.frameon": False,
        "font.size": 10,
    })


def mark_policy(ax, label: bool = True) -> None:
    import pandas as pd
    ax.axvline(pd.Timestamp("2025-01-05"), color=COLORS["policy"], lw=1.2, ls="--")
    if label:
        ax.annotate("Toll starts\n5 Jan 2025", xy=(pd.Timestamp("2025-01-05"), 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", va="top", fontsize=8)


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    return path


def write_metrics(name: str, data: dict) -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    path = METRICS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=float) + "\n")
    print(f"Saved reports/metrics/{name}.json")


def read_metrics(name: str) -> dict:
    return json.loads((METRICS / f"{name}.json").read_text())
