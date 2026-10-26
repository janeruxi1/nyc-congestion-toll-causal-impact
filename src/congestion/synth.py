"""Synthetic control: build a 'CBD without the toll' from other zones.

The treated series (weekly log trips starting in the CBD) is matched over
the pre-policy weeks by a weighted average of comparison zones. Weights are
non-negative and sum to one. Both sides are demeaned over the pre-period
(Ferman & Pinto's demeaned estimator), so the donors only need to match the
CBD's ups and downs, not its level, which no single neighbourhood has.

Inference is by placebo-in-space: each donor zone in turn plays the treated
unit, and the CBD's post/pre error ratio is ranked against theirs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def fit_weights(y_pre: np.ndarray, X_pre: np.ndarray, sum_penalty: float = 1e4) -> np.ndarray:
    """Non-negative weights summing to 1 (soft constraint via an extra row)."""
    A = np.vstack([X_pre, sum_penalty * np.ones(X_pre.shape[1])])
    b = np.concatenate([y_pre, [sum_penalty]])
    w, _ = nnls(A, b, maxiter=50 * A.shape[1])
    return w / w.sum()


def _remove_pre_trend(df: pd.DataFrame, pre: np.ndarray) -> pd.DataFrame:
    """Subtract each column's linear trend fitted on the pre-period (extended
    over the whole sample)."""
    t = (df.index - df.index[0]).days.to_numpy() / 365.25
    out = df.copy()
    for c in df.columns:
        slope, icpt = np.polyfit(t[pre], df[c].to_numpy()[pre], 1)
        out[c] = df[c] - (icpt + slope * t)
    return out


def synthetic_control(wide: pd.DataFrame, treated: str, post_start: pd.Timestamp,
                      donors: list[str] | None = None, detrend: bool = False) -> dict:
    """`wide`: weeks x units of log trips. Returns weights, path and fit stats.

    `detrend=True` first removes every unit's own pre-period linear trend, so
    the donors only have to match the treated unit's seasonal ups and downs
    and the estimate is the break from its trend (the synthetic-control
    analogue of the trend-adjusted DiD).
    """
    donors = donors or [c for c in wide.columns if c != treated]
    pre = wide.index < post_start
    Y = wide[[treated]]
    X = wide[donors]
    if detrend:
        Y, X = _remove_pre_trend(Y, pre), _remove_pre_trend(X, pre)
    Y = Y[treated]
    y_c = Y - Y[pre].mean()
    X_c = X - X[pre].mean()
    w = fit_weights(y_c[pre].to_numpy(), X_c[pre].to_numpy())
    synth = X_c.to_numpy() @ w
    gap = y_c.to_numpy() - synth
    rmspe_pre = float(np.sqrt(np.mean(gap[pre] ** 2)))
    rmspe_post = float(np.sqrt(np.mean(gap[~pre] ** 2)))
    path = pd.DataFrame({"week": wide.index, "actual": y_c.to_numpy(), "synthetic": synth,
                         "gap": gap}).set_index("week")
    return {
        "weights": pd.Series(w, index=donors).sort_values(ascending=False),
        "path": path,
        "effect_log": float(gap[~pre].mean()),
        "effect_pct": float(np.expm1(gap[~pre].mean())),
        "rmspe_pre": rmspe_pre,
        "rmspe_post": rmspe_post,
        "ratio": rmspe_post / rmspe_pre,
    }


def placebo_in_space(wide: pd.DataFrame, treated: str, post_start: pd.Timestamp,
                     max_pre_rmspe_multiple: float = 5.0, detrend: bool = False) -> pd.DataFrame:
    """Run the method on every donor as if it were treated.

    Donors whose pre-period fit is more than `max_pre_rmspe_multiple` times
    worse than the CBD's are dropped from the comparison (Abadie et al.): a
    unit that cannot be matched before the policy says nothing after it.
    """
    real = synthetic_control(wide, treated, post_start, detrend=detrend)
    rows = [{"unit": treated, "effect_pct": real["effect_pct"], "rmspe_pre": real["rmspe_pre"],
             "ratio": real["ratio"], "is_treated": True}]
    donors = [c for c in wide.columns if c != treated]
    for d in donors:
        others = [c for c in donors if c != d]
        r = synthetic_control(wide[[d] + others], d, post_start, others, detrend=detrend)
        rows.append({"unit": d, "effect_pct": r["effect_pct"], "rmspe_pre": r["rmspe_pre"],
                     "ratio": r["ratio"], "is_treated": False})
    out = pd.DataFrame(rows)
    out["kept"] = out["rmspe_pre"] <= max_pre_rmspe_multiple * real["rmspe_pre"]
    return out


def placebo_p_value(placebos: pd.DataFrame) -> float:
    """Share of kept units whose post/pre ratio is at least the CBD's."""
    kept = placebos[placebos["kept"]]
    real = kept.loc[kept["is_treated"], "ratio"].iloc[0]
    return float((kept["ratio"] >= real).mean())
