"""Forecast what CBD trips would have been without the toll.

Models are trained on pre-policy days only (from July 2022, after the
post-COVID rebound had levelled off) and judged on rolling-origin backtests
that pretend the policy started on 1 Oct 2023, 1 Jan 2024, 1 Apr 2024 or
1 Jul 2024 (each with at least 15 months of training data). Nothing happened
on those dates, so the forecast error there is both the accuracy check and
the "no-effect" noise band for the real estimate.

Models (all on log daily trips):
* ``seasonal_naive`` - same weekday 52 weeks earlier
* ``regression``     - linear trend, day of week, annual Fourier terms,
                        holidays and weather, fitted by OLS
* ``boosted``        - the regression plus LightGBM on its residuals
                        (trees cannot extrapolate a trend, so they only model
                        what the regression misses)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

MODELS = ("seasonal_naive", "regression", "boosted")
TRAIN_START = pd.Timestamp("2022-07-01")
POLICY_START = pd.Timestamp("2025-01-05")


def calendar_features(dates: pd.Series, weather: pd.DataFrame | None = None,
                      origin: pd.Timestamp = pd.Timestamp("2022-01-01")) -> pd.DataFrame:
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    X = pd.DataFrame(index=d.index)
    X["trend"] = (d - origin).dt.days / 365.25
    for k in range(7):
        if k:
            X[f"dow_{k}"] = (d.dt.day_of_week == k).astype(float)
    doy = d.dt.day_of_year / 365.25
    for k in range(1, 5):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * doy)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * doy)
    hol = USFederalHolidayCalendar().holidays(d.min() - pd.Timedelta(days=7), d.max() + pd.Timedelta(days=7))
    X["holiday"] = d.isin(hol).astype(float)
    X["xmas_week"] = (((d.dt.month == 12) & (d.dt.day >= 24)) | ((d.dt.month == 1) & (d.dt.day <= 1))).astype(float)
    X["thanksgiving_fri"] = d.isin(hol[hol.month == 11] + pd.Timedelta(days=1)).astype(float)
    if weather is not None:
        w = weather.set_index("date").reindex(d.values)
        X["rain"] = np.log1p(w["prcp_in"].fillna(0).to_numpy() * 10)
        X["snow"] = np.log1p(w["snow_in"].fillna(0).to_numpy())
        t = w["tmax_f"].ffill().to_numpy()
        X["hot"] = np.clip(t - 85, 0, None)
        X["cold"] = np.clip(40 - t, 0, None)
    return X


def _ols(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def _ols_predict(coef: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X.to_numpy()]) @ coef


def fit_predict(series: pd.DataFrame, value: str, train_end: pd.Timestamp,
                predict_dates: pd.Series, model: str, weather: pd.DataFrame | None = None,
                train_start: pd.Timestamp = TRAIN_START) -> np.ndarray:
    """Forecast log(value) for `predict_dates` from data up to `train_end`."""
    s = series.set_index("date")[value]
    train = s[(s.index >= train_start) & (s.index <= train_end)]
    y = np.log(train.to_numpy() + 1)
    pdates = pd.to_datetime(pd.Series(predict_dates)).reset_index(drop=True)
    if model == "seasonal_naive":
        full = np.log(s + 1)
        out = []
        for d in pdates:
            back = d - pd.Timedelta(weeks=52)
            while back > train_end:          # stay inside the training data
                back -= pd.Timedelta(weeks=52)
            out.append(full.get(back, np.nan))
        return np.asarray(out, dtype=float)
    Xtr = calendar_features(train.index.to_series(), weather)
    Xte = calendar_features(pdates, weather)
    coef = _ols(Xtr, y)
    pred = _ols_predict(coef, Xte)
    if model == "regression":
        return pred
    if model == "boosted":
        import lightgbm as lgb
        resid = y - _ols_predict(coef, Xtr)
        feats = [c for c in Xtr.columns if c != "trend"]
        gbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                min_child_samples=20, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, random_state=42, verbose=-1)
        gbm.fit(Xtr[feats], resid)
        return pred + gbm.predict(Xte[feats])
    raise ValueError(model)


def backtest(series: pd.DataFrame, value: str, weather: pd.DataFrame | None,
             origins: tuple[str, ...] = ("2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"),
             horizon_days: int = 365, train_start: pd.Timestamp = TRAIN_START,
             stop: pd.Timestamp = POLICY_START) -> pd.DataFrame:
    """Rolling-origin backtest: train to each origin, forecast up to a year
    ahead, never past `stop` (the real policy start)."""
    rows = []
    s = series.set_index("date")[value]
    for o in origins:
        origin = pd.Timestamp(o)
        end = min(origin + pd.Timedelta(days=horizon_days), stop - pd.Timedelta(days=1))
        dates = s.index[(s.index > origin) & (s.index <= end)]
        actual = np.log(s.loc[dates].to_numpy() + 1)
        for m in MODELS:
            pred = fit_predict(series, value, origin, pd.Series(dates), m, weather, train_start)
            rows.append(pd.DataFrame({"origin": origin.date().isoformat(), "model": m,
                                      "date": dates, "log_actual": actual, "log_pred": pred}))
    return pd.concat(rows, ignore_index=True)


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per model and origin, plus the placebo 'effect' (mean log gap)."""
    def f(g):
        err = g["log_actual"] - g["log_pred"]
        ape = np.abs(np.expm1(-err))           # |forecast - actual| / actual
        wk = g.assign(w=g["date"].dt.to_period("W")).groupby("w")[["log_actual", "log_pred"]].apply(
            lambda x: np.log(np.expm1(x["log_actual"]).sum() / np.expm1(x["log_pred"]).sum()))
        return pd.Series({
            "daily_mape": float(np.mean(ape)),
            "weekly_mape": float(np.mean(np.abs(np.expm1(-wk)))),
            "placebo_effect": float(np.expm1(err.mean())),
        })
    out = bt.groupby(["model", "origin"]).apply(f, include_groups=False).reset_index()
    naive = out[out["model"] == "seasonal_naive"].set_index("origin")["daily_mape"]
    out["mape_vs_naive"] = out["daily_mape"] / out["origin"].map(naive)
    return out


def weekly_interval(bt: pd.DataFrame, model: str, levels=(0.8, 0.95)) -> dict:
    """Empirical interval half-widths for weekly log errors, from backtests."""
    g = bt[bt["model"] == model].copy()
    g["w"] = g["date"].dt.to_period("W")
    wk = g.groupby(["origin", "w"]).apply(
        lambda x: np.log(np.expm1(x["log_actual"]).sum() / np.expm1(x["log_pred"]).sum()),
        include_groups=False)
    return {lv: float(np.quantile(np.abs(wk), lv)) for lv in levels}
