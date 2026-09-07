"""
Tail risk: Value-at-Risk and Conditional VaR (Expected Shortfall).

Three estimators, deliberately compared against each other:

  historical  : empirical quantile of realised portfolio returns. No
                distributional assumption, but only knows about losses
                that actually happened in the window.
  parametric  : Gaussian, fit mean/vol. Fast, and systematically wrong in
                the tails for equities (understates them).
  monte_carlo : simulate from a multivariate Student-t fit to the asset
                returns, so tails are fat and cross-asset dependence is
                preserved.

VaR at level q is the loss threshold exceeded with probability (1-q).
CVaR is the expected loss *given* you are past VaR. CVaR is a coherent
risk measure (sub-additive); VaR is not, which is why risk desks report both.

Sign convention: all outputs are POSITIVE numbers representing losses,
as a fraction of portfolio value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def portfolio_returns(returns: pd.DataFrame, w: pd.Series) -> pd.Series:
    w = w.reindex(returns.columns).fillna(0.0)
    return returns @ w


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def historical_var_cvar(pr: pd.Series, q: float = 0.99) -> tuple[float, float]:
    losses = -pr.values
    var = np.quantile(losses, q)
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar)


def parametric_var_cvar(pr: pd.Series, q: float = 0.99) -> tuple[float, float]:
    mu, sig = pr.mean(), pr.std(ddof=1)
    z = stats.norm.ppf(q)
    var = -(mu - z * sig)
    # Closed-form Gaussian expected shortfall
    cvar = -(mu - sig * stats.norm.pdf(z) / (1 - q))
    return float(var), float(cvar)


def fit_multivariate_t(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a multivariate Student-t by (a) estimating degrees of freedom from
    the pooled standardised returns and (b) using the sample covariance,
    scaled so the t-distribution's covariance matches it.
    Returns (mu, scale_matrix, nu).
    """
    R = returns.values
    mu = R.mean(axis=0)
    cov = np.cov(R, rowvar=False)
    z = ((R - mu) / R.std(axis=0, ddof=1)).ravel()
    nu, _, _ = stats.t.fit(z, floc=0, fscale=1)
    nu = float(np.clip(nu, 2.5, 30))
    scale = cov * (nu - 2) / nu   # so that Cov = scale * nu/(nu-2) = cov
    return mu, scale, nu


def monte_carlo_var_cvar(returns: pd.DataFrame, w: pd.Series, q: float = 0.99,
                         n_sims: int = 50_000, seed: int = 0,
                         dist: str = "t") -> tuple[float, float, np.ndarray]:
    """
    Simulate one-day portfolio returns. dist='t' uses a multivariate
    Student-t (fat tails); dist='normal' uses a Gaussian for comparison.
    Returns (var, cvar, simulated_portfolio_returns).
    """
    rng = np.random.default_rng(seed)
    w = w.reindex(returns.columns).fillna(0.0).values
    R = returns.values
    mu = R.mean(axis=0)
    cov = np.cov(R, rowvar=False)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(mu)))

    Z = rng.standard_normal((n_sims, len(mu)))
    if dist == "t":
        _, scale, nu = fit_multivariate_t(returns)
        L = np.linalg.cholesky(scale + 1e-12 * np.eye(len(mu)))
        g = rng.chisquare(nu, n_sims) / nu
        sims = mu + (Z @ L.T) / np.sqrt(g)[:, None]
    else:
        sims = mu + Z @ L.T

    pr = sims @ w
    losses = -pr
    var = np.quantile(losses, q)
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar), pr


@dataclass
class TailReport:
    q: float
    historical: tuple[float, float]
    parametric: tuple[float, float]
    monte_carlo_t: tuple[float, float]
    monte_carlo_normal: tuple[float, float]
    nu: float  # fitted t degrees of freedom (lower = fatter tails)

    def table(self) -> pd.DataFrame:
        rows = {
            "Historical": self.historical,
            "Parametric (Normal)": self.parametric,
            "Monte Carlo (Normal)": self.monte_carlo_normal,
            "Monte Carlo (Student-t)": self.monte_carlo_t,
        }
        return pd.DataFrame(rows, index=[f"VaR {self.q:.0%}", f"CVaR {self.q:.0%}"]).T


def tail_report(returns: pd.DataFrame, w: pd.Series, q: float = 0.99,
                lookback_days: int | None = 500) -> TailReport:
    R = returns.iloc[-lookback_days:] if lookback_days else returns
    pr = portfolio_returns(R, w)
    _, _, nu = fit_multivariate_t(R)
    mc_t = monte_carlo_var_cvar(R, w, q, dist="t")[:2]
    mc_n = monte_carlo_var_cvar(R, w, q, dist="normal")[:2]
    return TailReport(
        q=q,
        historical=historical_var_cvar(pr, q),
        parametric=parametric_var_cvar(pr, q),
        monte_carlo_t=mc_t,
        monte_carlo_normal=mc_n,
        nu=nu,
    )


# --------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------

@dataclass
class VaRBacktest:
    q: float
    method: str
    n_obs: int
    n_breaches: int
    expected_breaches: float
    breach_rate: float
    kupiec_lr: float
    kupiec_pvalue: float
    var_series: pd.Series
    realised: pd.Series
    breaches: pd.Series

    @property
    def verdict(self) -> str:
        if self.kupiec_pvalue < 0.05:
            direction = "too many" if self.breach_rate > 1 - self.q else "too few"
            return f"REJECT: {direction} breaches (p={self.kupiec_pvalue:.3f})"
        return f"PASS: breach rate consistent with {1-self.q:.0%} (p={self.kupiec_pvalue:.3f})"


def kupiec_pof(n_obs: int, n_breaches: int, q: float) -> tuple[float, float]:
    """
    Kupiec (1995) proportion-of-failures test.

    H0: true breach probability equals p = 1 - q.
    LR = -2 ln[ (1-p)^(T-x) p^x ] + 2 ln[ (1-x/T)^(T-x) (x/T)^x ]  ~ chi2(1)
    """
    p = 1 - q
    T, x = n_obs, n_breaches
    if x == 0:
        lr = -2 * (T * np.log(1 - p))
    elif x == T:
        lr = -2 * (T * np.log(p))
    else:
        phat = x / T
        lr = -2 * ((T - x) * np.log(1 - p) + x * np.log(p)) \
             + 2 * ((T - x) * np.log(1 - phat) + x * np.log(phat))
    pval = 1 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(pval)


def backtest_var(returns: pd.DataFrame, w: pd.Series, q: float = 0.99,
                 window: int = 250, method: str = "historical") -> VaRBacktest:
    """
    Rolling out-of-sample VaR: at each day t, estimate VaR from the previous
    `window` days and compare to the realised return on day t.
    """
    pr = portfolio_returns(returns, w)
    var_vals, dates = [], []
    for t in range(window, len(pr)):
        hist = pr.iloc[t - window:t]
        if method == "historical":
            v, _ = historical_var_cvar(hist, q)
        elif method == "parametric":
            v, _ = parametric_var_cvar(hist, q)
        else:
            raise ValueError("method must be 'historical' or 'parametric'")
        var_vals.append(v)
        dates.append(pr.index[t])

    var_series = pd.Series(var_vals, index=dates)
    realised = pr.loc[dates]
    breaches = (-realised) > var_series
    n, x = len(breaches), int(breaches.sum())
    lr, pv = kupiec_pof(n, x, q)
    return VaRBacktest(
        q=q, method=method, n_obs=n, n_breaches=x,
        expected_breaches=n * (1 - q), breach_rate=x / n,
        kupiec_lr=lr, kupiec_pvalue=pv,
        var_series=var_series, realised=realised, breaches=breaches,
    )
