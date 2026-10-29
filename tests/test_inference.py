"""Wild cluster bootstrap and HonestDiD (src/congestion/inference.py)."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from congestion.did import fit_did
from congestion.inference import (WEBB, WildClusterBootstrap, flci_breakdown, flci_sd, folded_normal_quantile,
                                  wild_bootstrap_did, worst_case_bias_sd)


def _toy(G=12, T=15, seed=0):
    rng = np.random.default_rng(seed)
    n = G * T
    g = np.repeat(np.arange(G), T)
    t = np.tile(np.arange(T), G)
    d = ((g < 5) & (t >= 8)).astype(float)
    X = np.c_[np.ones(n), np.eye(G)[g][:, 1:], np.eye(T)[t][:, 1:]]
    w = rng.uniform(0.5, 3, n)
    y = 0.3 * d + X @ rng.normal(size=X.shape[1]) + rng.normal(size=G)[g] + rng.normal(size=n)
    return y, d, X, g, w


# ------------------------------------------------------------- bootstrap
def test_point_estimate_and_t_match_statsmodels():
    y, d, X, g, w = _toy()
    wb = WildClusterBootstrap(y, d, X, g, w)
    ref = sm.WLS(y, np.c_[d, X], weights=w).fit(cov_type="cluster", cov_kwds={"groups": g})
    assert wb.beta == pytest.approx(ref.params[0], rel=1e-10)
    assert wb.t_stat(0.0) == pytest.approx(ref.tvalues[0], rel=1e-8)


def test_bootstrap_t_matches_brute_force_refits():
    """The cluster-level shortcut gives exactly what refitting each draw would."""
    y, d, X, g, w = _toy()
    wb = WildClusterBootstrap(y, d, X, g, w)
    v = np.random.default_rng(1).choice(WEBB, size=(wb.G, 20))
    b0 = 0.1
    fast = wb.boot_t(b0, v)
    restricted = sm.WLS(y - b0 * d, X, weights=w).fit()
    slow = []
    for j in range(v.shape[1]):
        ystar = restricted.fittedvalues + b0 * d + restricted.resid * v[g, j]
        r = sm.WLS(ystar, np.c_[d, X], weights=w).fit(cov_type="cluster", cov_kwds={"groups": g})
        slow.append((r.params[0] - b0) / r.bse[0])
    assert np.allclose(fast, slow, atol=1e-9)


def test_bootstrap_rejects_a_false_null_and_keeps_the_true_one():
    y, d, X, g, w = _toy(G=20, seed=3)
    wb = WildClusterBootstrap(y, d, X, g, w)
    v = np.random.default_rng(2).choice(WEBB, size=(wb.G, 999))
    lo, hi = wb.conf_int(v)
    assert lo < wb.beta < hi
    assert wb.p_value(hi + 1.0, v) < 0.05
    assert wb.p_value(wb.beta, v) > 0.5


def test_wild_bootstrap_did_matches_fit_did():
    rng = np.random.default_rng(4)
    zones, weeks = np.arange(16), pd.date_range("2024-01-01", periods=20, freq="W-MON")
    p = pd.MultiIndex.from_product([zones, weeks], names=["pu_zone", "week"]).to_frame(index=False)
    p["treated"] = p["pu_zone"] < 6
    p["post"] = p["week"] >= weeks[12]
    p["pre_mean_trips"] = p["pu_zone"].map(dict(zip(zones, rng.uniform(50, 500, len(zones)))))
    p["log_trips"] = 5 + 0.05 * (p["treated"] & p["post"]) + rng.normal(scale=0.1, size=len(p))
    for trends in (False, True):
        r = fit_did(p, weights="pre_mean_trips", unit_trends=trends)
        b = wild_bootstrap_did(p, weights="pre_mean_trips", unit_trends=trends, n_boot=499)
        assert b["beta"] == pytest.approx(r.beta, rel=1e-8)
        assert b["se_cr1"] == pytest.approx(r.se, rel=1e-6)
        assert b["ci_boot"][0] < r.beta < b["ci_boot"][1]


# ------------------------------------------------------------- HonestDiD
def test_folded_normal_quantile():
    assert folded_normal_quantile(0.95, 0.0) == pytest.approx(1.959964, abs=1e-5)
    # with a large mean the folded normal is just a shifted normal
    assert folded_normal_quantile(0.95, 10.0) == pytest.approx(10 + 1.644854, abs=1e-5)


@pytest.mark.parametrize("n_pre,n_post", [(1, 1), (3, 2), (4, 1), (2, 3)])
def test_bias_matches_the_r_package_formula(n_pre, n_post):
    """HonestDiD's .createObjectiveObjectForBias, in the pre-period weights w (l_pre = diff(w)).

    The package's constant term is written for target weights with t' l >= 0 (a basis vector or
    an average of post periods, as used here); with negative weights it is only an upper bound,
    while the LP stays exact. So the comparison uses non-negative l_post."""
    rng = np.random.default_rng(n_pre * 10 + n_post)
    l_post = np.abs(rng.normal(size=n_post))
    t_pre, t_post = np.arange(-n_pre, 0.0), np.arange(1, n_post + 1.0)
    free = rng.normal(size=n_pre - 1)
    l_pre = np.r_[free, (-(t_post @ l_post) - t_pre[:-1] @ free) / t_pre[-1]]
    w = np.cumsum(l_pre)
    const = sum(abs(np.arange(1, s + 1) @ l_post[n_post - s:]) for s in range(1, n_post + 1)) - t_post @ l_post
    closed = const + np.abs(np.cumsum(w)).sum()
    assert worst_case_bias_sd(l_pre, l_post, 1.0) == pytest.approx(closed, rel=1e-9)
    assert worst_case_bias_sd(l_pre, l_post, 2.5) == pytest.approx(2.5 * closed, rel=1e-9)


def test_bias_is_unbounded_if_a_linear_trend_survives():
    assert worst_case_bias_sd(np.array([0.0, 0.0]), np.array([1.0]), 1.0) == float("inf")


def test_one_pre_period_is_linear_extrapolation():
    """With one pre period, theta_hat = beta_post + beta_pre and the bias bound is M."""
    b = np.array([0.04, -0.02])
    V = np.array([[1e-4, 2e-5], [2e-5, 2e-4]])
    r0 = flci_sd(b, V, 1, np.array([1.0]), 0.0)
    assert r0["estimate"] == pytest.approx(0.02)
    sd = np.sqrt(1e-4 + 2e-4 + 2 * 2e-5)
    assert r0["half_length"] == pytest.approx(1.959964 * sd, rel=1e-5)
    r = flci_sd(b, V, 1, np.array([1.0]), 0.01)
    assert r["worst_case_bias"] == pytest.approx(0.01)
    assert r["half_length"] == pytest.approx(folded_normal_quantile(0.95, 0.01 / sd) * sd, rel=1e-6)


def test_flci_is_no_longer_than_simple_extrapolation():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(5, 5))
    V = (A @ A.T + 5 * np.eye(5)) * 1e-4
    b = rng.normal(scale=0.02, size=5)
    l_post = np.array([0.5, 0.5])
    opt = flci_sd(b, V, 3, l_post, 0.01, n_starts=4)
    simple_pre = np.array([0.0, 0.0, 1.5])               # extrapolate from the last pre period only
    sd = np.sqrt(np.r_[simple_pre, l_post] @ V @ np.r_[simple_pre, l_post])
    simple = folded_normal_quantile(0.95, worst_case_bias_sd(simple_pre, l_post, 0.01) / sd) * sd
    assert opt["half_length"] <= simple + 1e-9


def test_flci_covers_under_a_worst_case_violation():
    """Coverage simulation: trends that bend by exactly +/- M every period."""
    rng = np.random.default_rng(6)
    n_pre, n_post, M = 3, 2, 0.02
    V = np.diag(np.full(n_pre + n_post, 1e-4))
    l_post = np.array([0.0, 1.0])
    tau = np.array([-0.05, -0.05])
    # delta with second differences alternating at the bound; reference period is 0
    t = np.arange(-n_pre, n_post + 1)
    delta = np.zeros(len(t))
    slope = 0.01
    for i in range(1, len(t)):
        slope += M * (1 if i % 2 else -1)
        delta[i] = delta[i - 1] + slope
    delta -= delta[n_pre]
    delta = np.delete(delta, n_pre)
    theta = l_post @ tau
    cover = 0
    sims = 200
    first = flci_sd(np.zeros(n_pre + n_post), V, n_pre, l_post, M, n_starts=3)
    lvec, hl = np.r_[first["l_pre"], l_post], first["half_length"]   # the FLCI's weights do not depend on the data
    for _ in range(sims):
        bhat = np.r_[np.zeros(n_pre), tau] + delta + rng.multivariate_normal(np.zeros(n_pre + n_post), V)
        est = lvec @ bhat
        cover += est - hl <= theta <= est + hl
    assert cover / sims >= 0.93


def test_breakdown():
    b = np.array([0.04, 0.0])
    V = np.diag([1e-4, 1e-4])
    m = flci_breakdown(b, V, 1, np.array([1.0]), np.log(0.95), side="below")
    r = flci_sd(b, V, 1, np.array([1.0]), m)
    assert r["lb"] == pytest.approx(np.log(0.95), abs=1e-4)
    assert flci_breakdown(b, V, 1, np.array([1.0]), 0.5, side="below") == 0.0
