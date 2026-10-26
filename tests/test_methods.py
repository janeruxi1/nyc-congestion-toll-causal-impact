import numpy as np
import pandas as pd
from conftest import make_panel

from congestion.did import event_study, fit_did, fit_did_pre_trend, placebo_in_time
from congestion.forecast import backtest, calendar_features, fit_predict, score
from congestion.synth import placebo_in_space, placebo_p_value, synthetic_control


def test_did_recovers_known_effect():
    p = make_panel(effect=-0.10)
    r = fit_did(p)
    assert abs(r.pct - (-0.10)) < 0.01
    assert r.ci_low < np.log(0.90) < r.ci_high
    assert r.n_treated_units == 20 and r.n_control_units == 60


def test_did_weighted_and_null():
    p = make_panel(effect=0.0, seed=3)
    r = fit_did(p, weights="pre_mean_trips")
    assert r.weighted and abs(r.pct) < 0.02 and r.p_value > 0.01


def test_event_study_flat_before_and_shifted_after():
    p = make_panel(effect=-0.20, weeks=120, post_from=80)
    p.loc[:, "week"] = pd.to_datetime(p["week"])
    policy_week = p["week"].sort_values().unique()[80]
    es = event_study(p, policy_week=pd.Timestamp(policy_week))
    pre = es[es["bin"] < -1]
    post = es[es["bin"] >= 0]
    assert (pre["pct"].abs() < 0.03).all()
    assert (abs(post["pct"] - (-0.20)) < 0.03).all()
    assert es.loc[es["bin"] == -1, "beta"].iloc[0] == 0.0


def test_placebo_in_time_finds_nothing_before_policy():
    p = make_panel(effect=-0.20, weeks=200, post_from=150)
    start = pd.Timestamp(p["week"].sort_values().unique()[100]).date().isoformat()
    r = placebo_in_time(p, [start], post_weeks=40, pre_weeks=40)
    assert abs(r["pct"].iloc[0]) < 0.02


def _toy_series(days=1200, drop=0.0, drop_from=None, seed=1):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2022-01-01", periods=days, freq="D")
    X = calendar_features(pd.Series(d))
    y = 9 + 0.05 * X["trend"] - 0.2 * X.get("dow_6", 0) + 0.1 * X["sin1"] + rng.normal(0, 0.02, days)
    if drop_from is not None:
        y = y + np.where(d >= pd.Timestamp(drop_from), np.log1p(drop), 0)
    return pd.DataFrame({"date": d, "cbd": np.expm1(y)})


def test_regression_forecast_is_accurate_on_clean_seasonal_data():
    s = _toy_series()
    bt = backtest(s, "cbd", None, origins=("2023-12-31",), horizon_days=200)
    sc = score(bt).set_index("model")
    assert sc.loc["regression", "daily_mape"] < 0.03
    assert sc.loc["regression", "daily_mape"] < sc.loc["seasonal_naive", "daily_mape"]
    assert abs(sc.loc["regression", "placebo_effect"]) < 0.01


def test_forecast_gap_measures_a_planted_drop():
    s = _toy_series(drop=-0.08, drop_from="2025-01-05")
    post = s.loc[s["date"] >= "2025-01-05", "date"]
    pred = fit_predict(s, "cbd", pd.Timestamp("2024-12-31"), post, "regression")
    gap = np.expm1(np.mean(np.log1p(s.loc[post.index, "cbd"]) - pred))
    assert abs(gap - (-0.08)) < 0.01


def test_synthetic_control_recovers_weights_and_effect():
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2022-01-03", periods=150, freq="7D")
    donors = pd.DataFrame(rng.normal(0, 0.1, (150, 6)).cumsum(axis=0) + 5,
                          index=weeks, columns=[f"z{i}" for i in range(6)])
    treated = 0.6 * donors["z0"] + 0.4 * donors["z3"] + 2.0
    post = weeks >= weeks[110]
    treated = treated + np.where(post, np.log(0.9), 0)
    wide = donors.assign(cbd=treated)
    r = synthetic_control(wide, "cbd", weeks[110])
    assert abs(r["weights"]["z0"] - 0.6) < 0.05 and abs(r["weights"]["z3"] - 0.4) < 0.05
    assert abs(r["effect_pct"] - (-0.10)) < 0.01
    pl = placebo_in_space(wide, "cbd", weeks[110])
    assert placebo_p_value(pl) <= 1 / pl["kept"].sum() + 1e-9


def test_trend_adjustments_remove_a_pre_existing_drift():
    """Treated units drift down 4% a year before and after; true effect -5%."""
    p = make_panel(effect=-0.05, weeks=150, post_from=100, treated_trend_per_year=-0.04, seed=5)
    p["week"] = pd.to_datetime(p["week"])
    naive = fit_did(p)
    assert naive.pct < -0.07                      # the drift makes a plain DiD overstate the drop
    for r in (fit_did(p, unit_trends=True), fit_did_pre_trend(p)):
        assert abs(r.pct - (-0.05)) < 0.01


def test_pre_trend_version_is_not_fooled_by_a_growing_effect():
    """An effect that builds up after the policy: whole-window trends absorb part of it."""
    p = make_panel(effect=0.0, weeks=150, post_from=100, seed=6)
    p["week"] = pd.to_datetime(p["week"])
    k = (p["week"].rank(method="dense") - 100).clip(lower=0)      # weeks since the policy
    p.loc[p["treated"], "log_trips"] -= 0.002 * k[p["treated"]]   # -0.2% more each week
    true_avg = float(np.expm1(-0.002 * np.arange(1, 51).mean()))
    pre_trend = fit_did_pre_trend(p)
    whole = fit_did(p, unit_trends=True)
    assert abs(pre_trend.pct - true_avg) < 0.01
    assert abs(whole.pct - true_avg) > abs(pre_trend.pct - true_avg)
