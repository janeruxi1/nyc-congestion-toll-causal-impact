"""Difference-in-differences and event studies with two-way fixed effects.

Model:   y_it = a_i + g_t + beta * (treated_i x post_t) + e_it
with unit fixed effects a_i, week fixed effects g_t and standard errors
clustered by origin zone (the level at which treatment is assigned).
beta on log(trips + 1) is read as a percentage change: exp(beta) - 1.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class DidResult:
    outcome: str
    beta: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    n_obs: int
    n_treated_units: int
    n_control_units: int
    weighted: bool

    @property
    def pct(self) -> float:
        return float(np.expm1(self.beta))

    @property
    def pct_ci(self) -> tuple[float, float]:
        return float(np.expm1(self.ci_low)), float(np.expm1(self.ci_high))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["pct"], (d["pct_ci_low"], d["pct_ci_high"]) = self.pct, self.pct_ci
        return d


def _design(df: pd.DataFrame, unit: str, time: str, regressors: pd.DataFrame,
            unit_trends: bool = False) -> pd.DataFrame:
    unit_fe = pd.get_dummies(df[unit].astype(str), prefix="u", drop_first=True, dtype=float)
    time_fe = pd.get_dummies(df[time].astype(str), prefix="t", drop_first=True, dtype=float)
    parts = [regressors.reset_index(drop=True), unit_fe.reset_index(drop=True),
             time_fe.reset_index(drop=True)]
    if unit_trends:
        # One linear time trend per unit (years since the panel start). The
        # common trend is absorbed by the week effects, so one unit is left out.
        t = ((df[time] - df[time].min()).dt.days / 365.25).to_numpy()
        all_units = pd.get_dummies(df[unit].astype(str), prefix="tr", dtype=float)
        parts.append(all_units.iloc[:, 1:].mul(t, axis=0).reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    return sm.add_constant(X, has_constant="add")


def _fit(df, y, X, weights, cluster):
    """WLS/OLS with zone-clustered errors.

    The clustered covariance has rank at most the number of clusters, so
    some fixed-effect standard errors come out undefined (numpy warns about
    a square root of a negative number). Only the policy coefficients are
    used, and their errors are well defined, so that warning is silenced.
    """
    w = None if weights is None else df[weights].to_numpy(dtype=float)
    model = sm.WLS(df[y].to_numpy(dtype=float), X, weights=w) if w is not None else \
        sm.OLS(df[y].to_numpy(dtype=float), X)
    groups = pd.factorize(df[cluster])[0]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        _ = res.bse, res.pvalues  # computed here, inside the filter
    return res


def fit_did(panel: pd.DataFrame, y: str = "log_trips", unit: str = "pu_zone",
            time: str = "week", treat: str = "treated", post: str = "post",
            weights: str | None = None, cluster: str = "pu_zone",
            unit_trends: bool = False) -> DidResult:
    """Two-way fixed-effects DiD.

    With `unit_trends=True` every unit also gets its own linear trend, fitted
    on the whole window. The treatment effect is then the break from each
    unit's own path, which is what is needed when treated and comparison
    units were already drifting apart before the policy.
    """
    df = panel.dropna(subset=[y]).reset_index(drop=True)
    d = (df[treat].astype(bool) & df[post].astype(bool)).astype(float).rename("did")
    X = _design(df, unit, time, d.to_frame(), unit_trends)
    res = _fit(df, y, X, weights, cluster)
    lo, hi = res.conf_int().loc["did"]
    units = df.drop_duplicates(unit)
    return DidResult(
        outcome=y, beta=float(res.params["did"]), se=float(res.bse["did"]),
        ci_low=float(lo), ci_high=float(hi), p_value=float(res.pvalues["did"]),
        n_obs=len(df), n_treated_units=int(units[treat].sum()),
        n_control_units=int((~units[treat].astype(bool)).sum()), weighted=weights is not None,
    )


def detrend_by_unit(panel: pd.DataFrame, y: str = "log_trips", unit: str = "pu_zone",
                    time: str = "week", post: str = "post", out: str = "y_detrended") -> pd.DataFrame:
    """Subtract each unit's linear trend, fitted on its pre-policy rows only.

    The trend-adjusted DiD in `fit_did(unit_trends=True)` fits each unit's
    trend on the whole window, post-policy weeks included. If the effect grows
    over time, that trend can soak part of it up. Fitting the trends on the
    pre-period alone and extending them forward avoids this (the synthetic
    control's trend adjustment does the same).
    """
    df = panel.copy()
    t = ((df[time] - df[time].min()).dt.days / 365.25).to_numpy()
    yv = df[y].to_numpy(dtype=float)
    pre = ~df[post].astype(bool).to_numpy()
    fitted = np.full(len(df), np.nan)
    for _, idx in df.groupby(unit).indices.items():
        m = idx[pre[idx] & np.isfinite(yv[idx])]
        if len(m) < 2:
            continue
        slope, icpt = np.polyfit(t[m], yv[m], 1)
        fitted[idx] = icpt + slope * t[idx]
    df[out] = yv - fitted
    return df


def fit_did_pre_trend(panel: pd.DataFrame, y: str = "log_trips", unit: str = "pu_zone",
                      time: str = "week", post: str = "post", **kw) -> DidResult:
    """Two-step DiD: remove pre-period unit trends, then a standard two-way FE DiD.

    The standard errors treat the fitted trends as known, so they are somewhat
    too narrow; the fake-date checks in notebook 03 are the better guide to noise.
    """
    d = detrend_by_unit(panel, y, unit, time, post)
    r = fit_did(d.dropna(subset=["y_detrended"]), y="y_detrended", unit=unit, time=time, post=post, **kw)
    r.outcome = y
    return r


def event_study(panel: pd.DataFrame, y: str = "log_trips", unit: str = "pu_zone",
                time: str = "week", treat: str = "treated", weights: str | None = None,
                cluster: str = "pu_zone", policy_week: pd.Timestamp = pd.Timestamp("2025-01-06"),
                bin_weeks: int = 4, ref_bin: int = -1, return_vcov: bool = False):
    """Treated-minus-control gap in 4-week bins relative to the policy.

    Bin -1 (the four weeks before the toll) is the reference, so each
    coefficient reads as "how much bigger or smaller the gap is than just
    before the toll". Flat pre-policy coefficients support parallel trends.

    With ``return_vcov=True`` it also returns the covariance of the bin
    coefficients (reference bin excluded) and their bins, in order: the
    input HonestDiD needs.
    """
    df = panel.dropna(subset=[y]).reset_index(drop=True)
    rel = ((df[time] - policy_week).dt.days // 7) // bin_weeks
    bins = sorted(rel.unique())
    regs = {}
    for b in bins:
        if b == ref_bin:
            continue
        regs[f"b{b}"] = ((rel == b) & df[treat].astype(bool)).astype(float)
    R = pd.DataFrame(regs)
    X = _design(df, unit, time, R)
    res = _fit(df, y, X, weights, cluster)
    ci = res.conf_int()
    rows = [{"bin": ref_bin, "beta": 0.0, "ci_low": 0.0, "ci_high": 0.0}]
    for b in bins:
        if b == ref_bin:
            continue
        k = f"b{b}"
        rows.append({"bin": b, "beta": res.params[k], "ci_low": ci.loc[k, 0], "ci_high": ci.loc[k, 1]})
    out = pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
    out["start_week"] = policy_week + pd.to_timedelta(out["bin"] * bin_weeks * 7, unit="D")
    for c in ("beta", "ci_low", "ci_high"):
        out[f"pct_{c}" if c != "beta" else "pct"] = np.expm1(out[c])
    if return_vcov:
        kept = [b for b in bins if b != ref_bin]
        keys = [f"b{b}" for b in kept]
        return out, res.cov_params().loc[keys, keys].to_numpy(), kept
    return out


def placebo_in_time(panel: pd.DataFrame, fake_starts: list[str], post_weeks: int = 52,
                    pre_weeks: int = 52, **kw) -> pd.DataFrame:
    """Re-run the DiD on pre-policy data with a pretend start date.

    Uses `pre_weeks` before and `post_weeks` after each fake date, all before
    the real toll. A real design should find roughly nothing.
    """
    rows = []
    real = panel[~panel["post"].astype(bool)]
    for s in fake_starts:
        start = pd.Timestamp(s)
        win = real[(real["week"] >= start - pd.Timedelta(weeks=pre_weeks)) &
                   (real["week"] < start + pd.Timedelta(weeks=post_weeks))].copy()
        win["post"] = win["week"] >= start
        r = fit_did(win, **kw)
        rows.append({"fake_start": start.date().isoformat(), **r.as_dict()})
    return pd.DataFrame(rows)
