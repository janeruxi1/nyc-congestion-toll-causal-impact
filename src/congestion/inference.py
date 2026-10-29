"""Two checks on how much the DiD intervals can be trusted.

1. **Wild cluster bootstrap** (restricted, "WCR"; Cameron, Gelbach and Miller
   2008; MacKinnon and Webb 2017; implementation follows Roodman et al. 2019).
   Cluster-robust standard errors lean on having many clusters. Here a cluster
   is a pickup zone and there are 60, of which 38 are treated, weighted very
   unequally by trips. The bootstrap-t p-value and the confidence interval
   obtained by inverting it do not rely on the normal approximation.

2. **HonestDiD** (Rambachan and Roth 2023, *Review of Economic Studies*).
   Instead of assuming parallel trends, it asks how far the CBD-versus-rest gap
   may drift from a straight line after the toll before the answer changes.
   Under Delta^SD(M), the gap's trend may bend by at most M per period, per
   period (M = 0: the pre-toll trend continues exactly). The confidence
   interval is the fixed-length one (FLCI, Armstrong and Kolesar 2018) that
   the authors' R package uses by default for Delta^SD.

Only numpy, scipy and statsmodels are used, so CI needs nothing new.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .did import _design

# ---------------------------------------------------------------------------
# 1. Wild cluster restricted bootstrap
# ---------------------------------------------------------------------------
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _bootstrap_weights(n_clusters: int, n_boot: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "webb":
        return rng.choice(WEBB, size=(n_clusters, n_boot))
    if kind == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=(n_clusters, n_boot))
    raise ValueError(f"unknown weight type {kind!r}")


class WildClusterBootstrap:
    """Bootstrap-t for the coefficient on one regressor, clustered.

    The regression is y = beta * d + X gamma + e (X: constant and fixed
    effects, optionally unit trends), weighted. Everything is reduced to
    cluster-level quantities once, so each bootstrap draw costs O(G^2)
    instead of a full regression:

    * partial X out of d and y (weighted), giving d~ and y~;
    * under H0: beta = b0 the restricted residual is u = y~ - b0 d~;
    * a draw multiplies each cluster's u by v_g. Its score for cluster g is
      sum_{i in g} w_i d~_i (M_X(u * v))_i
        = c_g v_g - sum_h K_{gh} v_h,
      with c_g = sum_{i in g} w_i d~_i u_i and K = F' (X'WX)^-1 S, where
      F[:, g] = sum_{i in g} w_i d~_i x_i and S[:, h] = sum_{i in h} w_i x_i u_i.

    Standard errors use the same CR1 small-sample factor as statsmodels.
    """

    def __init__(self, y: np.ndarray, d: np.ndarray, X: np.ndarray, clusters: np.ndarray,
                 weights: np.ndarray | None = None):
        n, k = X.shape
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        self.groups, g = np.unique(clusters, return_inverse=True)
        self.G, self.n, self.k = len(self.groups), n, np.linalg.matrix_rank(X)
        Xw = X * w[:, None]
        XtWX_inv = np.linalg.pinv(X.T @ Xw)
        self._resid = lambda v: v - X @ (XtWX_inv @ (Xw.T @ v))
        self.d_t, self.y_t = self._resid(d.astype(float)), self._resid(y.astype(float))
        self.w, self.g = w, g
        onehot = np.zeros((n, self.G))
        onehot[np.arange(n), g] = 1.0
        self._onehot = onehot
        wd = w * self.d_t
        self.A = onehot.T @ (wd * self.d_t)                 # sum_g w d~^2
        self.denom = self.A.sum()
        F = (Xw * self.d_t[:, None]).T @ onehot             # k x G
        self._FQ = F.T @ XtWX_inv                           # G x k
        self._Xw, self._wd = Xw, wd
        self.beta = float(wd @ self.y_t / self.denom)
        self.small = (self.G / (self.G - 1)) * ((n - 1) / (n - self.k - 1))

    def _se_from_scores(self, scores: np.ndarray) -> np.ndarray:
        return np.sqrt(self.small * (scores ** 2).sum(axis=0)) / self.denom

    def t_stat(self, b0: float) -> float:
        e = self.y_t - self.beta * self.d_t
        scores = self._onehot.T @ (self._wd * e)
        return float((self.beta - b0) / self._se_from_scores(scores[:, None])[0])

    def boot_t(self, b0: float, v: np.ndarray) -> np.ndarray:
        """Bootstrap t-statistics for H0: beta = b0, one per column of v (G x B)."""
        u = self.y_t - b0 * self.d_t
        c = self._onehot.T @ (self._wd * u)                 # G
        S = (self._Xw * u[:, None]).T @ self._onehot        # k x G
        K = self._FQ @ S                                    # G x G
        num = c[:, None] * v - K @ v                        # sum_{i in g} w d~ (M_X(u v))_i
        num = num + b0 * self.A[:, None]                    # plus b0 * sum w d~^2: y* also has b0 d
        beta_star = num.sum(axis=0) / self.denom
        scores = num - beta_star[None, :] * self.A[:, None]
        return (beta_star - b0) / self._se_from_scores(scores)

    def p_value(self, b0: float, v: np.ndarray) -> float:
        t = self.t_stat(b0)
        tb = self.boot_t(b0, v)
        return float(np.mean(np.abs(tb) >= abs(t)))

    def conf_int(self, v: np.ndarray, level: float = 0.95, se_guess: float | None = None) -> tuple[float, float]:
        """Values of b0 the bootstrap test does not reject, found by root-finding on p(b0) - alpha.
        The same draws v are used for every b0, so p(b0) is a step function; the bounds are
        located to within a small fraction of the standard error."""
        alpha = 1 - level
        s = se_guess if se_guess is not None else abs(self.beta / self.t_stat(0.0)) if self.t_stat(0.0) else 1.0
        f = lambda b0: self.p_value(b0, v) - alpha  # noqa: E731
        bounds = []
        for sign in (-1, 1):
            step, inner = s, self.beta
            outer = self.beta + sign * step
            while f(outer) > 0:
                inner, outer = outer, outer + sign * step
                step *= 2
            for _ in range(40):
                mid = (inner + outer) / 2
                if f(mid) > 0:
                    inner = mid
                else:
                    outer = mid
                if abs(outer - inner) < 1e-4 * s:
                    break
            bounds.append((inner + outer) / 2)
        return bounds[0], bounds[1]


def wild_bootstrap_did(panel: pd.DataFrame, y: str = "log_trips", unit: str = "pu_zone",
                       time: str = "week", treat: str = "treated", post: str = "post",
                       weights: str | None = None, cluster: str = "pu_zone", unit_trends: bool = False,
                       outcome_col: str | None = None, n_boot: int = 9999, kind: str = "webb",
                       seed: int = 0) -> dict:
    """The DiD of did.fit_did, with a wild cluster restricted bootstrap.

    ``outcome_col`` lets a pre-computed outcome (e.g. the two-step detrended
    one) be used under the name ``y``.
    """
    df = panel.dropna(subset=[outcome_col or y]).reset_index(drop=True)
    d = (df[treat].astype(bool) & df[post].astype(bool)).astype(float)
    X = _design(df, unit, time, pd.DataFrame(index=df.index), unit_trends).to_numpy(dtype=float)
    wb = WildClusterBootstrap(df[outcome_col or y].to_numpy(float), d.to_numpy(), X,
                              df[cluster].to_numpy(), None if weights is None else df[weights].to_numpy(float))
    rng = np.random.default_rng(seed)
    v = _bootstrap_weights(wb.G, n_boot, kind, rng)
    t0 = wb.t_stat(0.0)
    se = wb.beta / t0
    lo, hi = wb.conf_int(v, se_guess=abs(se))
    return {"beta": wb.beta, "se_cr1": float(abs(se)), "t": t0,
            "p_analytic_t": float(2 * stats.t.sf(abs(t0), wb.G - 1)),
            "p_boot": wb.p_value(0.0, v), "ci_boot": [lo, hi],
            "pct": float(np.expm1(wb.beta)), "pct_ci_boot": [float(np.expm1(lo)), float(np.expm1(hi))],
            "clusters": wb.G, "n_obs": wb.n, "n_boot": n_boot, "weights": kind}


# ---------------------------------------------------------------------------
# 2. HonestDiD: the fixed-length confidence interval under Delta^SD(M)
# ---------------------------------------------------------------------------
def folded_normal_quantile(p: float, mu: float) -> float:
    """p-quantile of |Z + mu|, Z ~ N(0, 1): the c with Phi(c - mu) - Phi(-c - mu) = p.
    (The R package simulates this; here it is solved exactly.)"""
    mu = abs(mu)
    f = lambda c: stats.norm.cdf(c - mu) - stats.norm.cdf(-c - mu) - p  # noqa: E731
    return float(optimize.brentq(f, 0.0, mu + 10.0))


def worst_case_bias_sd(l_pre: np.ndarray, l_post: np.ndarray, M: float = 1.0) -> float:
    """max over delta in Delta^SD(M) of |l_pre' delta_pre + l_post' delta_post - l_post' delta_post_true|.

    For the affine estimator theta_hat = l_pre' beta_pre + l_post' beta_post of
    theta = l_post' tau_post, the bias is l_pre' delta_pre + l_post' delta_post.
    Solved as a linear programme over delta (pre periods, the reference period
    fixed at 0, post periods). Unbounded (inf) unless the estimator removes any
    linear trend, i.e. sum_t t * coefficient_t = 0 with t the period index.
    """
    l_pre, l_post = np.asarray(l_pre, float), np.asarray(l_post, float)
    T_pre, T_post = len(l_pre), len(l_post)
    T = T_pre + 1 + T_post                       # with the reference period
    c = np.r_[l_pre, 0.0, l_post]
    # second differences D delta, for t = 1..T-2
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i:i + 3] = [1.0, -2.0, 1.0]
    A_ub = np.r_[D, -D]
    b_ub = np.full(2 * (T - 2), M)
    bounds = [(None, None)] * T
    bounds[T_pre] = (0.0, 0.0)                   # reference period normalised to 0
    res = optimize.linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status == 3:
        return float("inf")
    if res.status != 0:
        raise RuntimeError(res.message)
    return float(-res.fun)


def _sd_of(l_pre: np.ndarray, l_post: np.ndarray, sigma: np.ndarray) -> float:
    v = np.r_[l_pre, l_post]
    return float(np.sqrt(v @ sigma @ v))


def flci_sd(betahat: np.ndarray, sigma: np.ndarray, n_pre: int, l_post: np.ndarray, M: float,
            alpha: float = 0.05, n_starts: int = 8, seed: int = 0) -> dict:
    """Fixed-length CI for theta = l_post' tau_post under Delta^SD(M).

    ``betahat`` holds the event-study coefficients without the reference
    period, pre periods first (earliest first), then post periods; ``sigma``
    is their covariance. The pre-period weights are chosen to minimise the
    half-length cv_{1-alpha}(bias / sd) * sd, as in Rambachan and Roth's
    findOptimalFLCI. Weights are parameterised as in their code: the pre
    weights must cancel any linear trend, which fixes one degree of freedom.
    """
    betahat, sigma, l_post = np.asarray(betahat, float), np.asarray(sigma, float), np.asarray(l_post, float)
    T_post = len(l_post)
    t_pre = np.arange(-n_pre, 0, dtype=float)            # reference period is 0
    t_post = np.arange(1, T_post + 1, dtype=float)
    target = -(t_post @ l_post)                          # need t_pre' l_pre = target

    def complete(free: np.ndarray) -> np.ndarray:
        # last pre weight (period -1) is pinned by the trend-cancelling constraint
        return np.r_[free, (target - t_pre[:-1] @ free) / t_pre[-1]]

    def half_length(free: np.ndarray) -> float:
        l_pre = complete(free)
        sd = _sd_of(l_pre, l_post, sigma)
        bias = worst_case_bias_sd(l_pre, l_post, M) if M > 0 else 0.0
        return folded_normal_quantile(1 - alpha, bias / sd) * sd

    if n_pre == 1:
        best = np.array([])
    else:
        rng = np.random.default_rng(seed)
        # first start: all weight on the period just before the reference (extrapolation
        # from the last pre-period slope); the others perturb it at random
        base = np.zeros(n_pre - 1)
        starts = [base] + [base + rng.normal(scale=abs(target) + 1, size=n_pre - 1) for _ in range(n_starts - 1)]
        best, best_val = None, np.inf
        for s0 in starts:
            r = optimize.minimize(half_length, s0, method="Nelder-Mead",
                                  options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 4000 * n_pre})
            r = optimize.minimize(half_length, r.x, method="Powell", options={"xtol": 1e-8, "ftol": 1e-12})
            if r.fun < best_val:
                best, best_val = r.x, r.fun
    l_pre = complete(best)
    hl = half_length(best)
    est = float(np.r_[l_pre, l_post] @ betahat)
    return {"M": float(M), "estimate": est, "lb": est - hl, "ub": est + hl, "half_length": float(hl),
            "l_pre": l_pre.tolist(), "sd": _sd_of(l_pre, l_post, sigma),
            "worst_case_bias": worst_case_bias_sd(l_pre, l_post, M) if M > 0 else 0.0}


def flci_breakdown(betahat: np.ndarray, sigma: np.ndarray, n_pre: int, l_post: np.ndarray, value: float,
                   side: str, alpha: float = 0.05, M_max: float = 1.0) -> float:
    """Smallest M at which the FLCI stops excluding ``value``.

    side="below": the interval's lower bound has to stay above ``value``;
    side="above": its upper bound has to stay below it. Returns 0 if the
    M = 0 interval already includes it, and inf if even M_max does not.
    """
    def excludes(M):
        r = flci_sd(betahat, sigma, n_pre, l_post, M, alpha)
        return r["lb"] > value if side == "below" else r["ub"] < value

    if not excludes(0.0):
        return 0.0
    if excludes(M_max):
        return float("inf")
    lo, hi = 0.0, M_max
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if excludes(mid) else (lo, mid)
        if hi - lo < 1e-6:
            break
    return (lo + hi) / 2
