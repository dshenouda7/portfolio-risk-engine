"""
Covariance estimation and portfolio construction.

Why not just use the sample covariance? With N assets and T days you are
estimating N(N+1)/2 parameters from N*T numbers. For N=30, T=250 the
sample matrix is noisy enough that a minimum-variance optimiser will
happily load up on whatever pair of stocks happened to look negatively
correlated last year. Ledoit-Wolf shrinkage pulls the sample matrix
toward a structured target and demonstrably reduces out-of-sample error.

Optimisers here are all long-only with a max-weight cap, matching a
student long-only fund's constraints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Covariance
# --------------------------------------------------------------------------

def ledoit_wolf(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Ledoit & Wolf (2004) shrinkage toward the scaled identity, implemented
    from the paper's closed-form optimal intensity.

    Σ_shrunk = (1-δ) S + δ μ I,   μ = tr(S)/N
    Returns (shrunk covariance, δ).
    """
    X = returns.values - returns.values.mean(axis=0)
    T, N = X.shape
    S = X.T @ X / T
    mu = np.trace(S) / N
    delta2 = np.linalg.norm(S - mu * np.eye(N), "fro") ** 2   # ||S - μI||²_F

    # β² = (1/T²) Σ_t ||x_t x_t' - S||²_F
    beta2 = 0.0
    for t in range(T):
        xt = X[t][:, None]
        beta2 += np.linalg.norm(xt @ xt.T - S, "fro") ** 2
    beta2 /= T ** 2
    beta2 = min(beta2, delta2)

    shrink = beta2 / delta2 if delta2 > 0 else 0.0
    sigma = (1 - shrink) * S + shrink * mu * np.eye(N)
    return pd.DataFrame(sigma, index=returns.columns, columns=returns.columns), float(shrink)


# --------------------------------------------------------------------------
# Portfolio analytics
# --------------------------------------------------------------------------

def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w @ cov @ w))


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """
    Fraction of portfolio variance each position is responsible for:
    RC_i = w_i (Σw)_i / (w'Σw). Sums to 1. A 'risk parity' book has all
    RC_i equal; a concentrated book has a few names dominating.
    """
    w_ = w.reindex(cov.index).fillna(0.0).values
    C = cov.values
    mrc = C @ w_
    rc = w_ * mrc / (w_ @ C @ w_)
    return pd.Series(rc, index=cov.index)


def effective_number_of_bets(w: pd.Series, cov: pd.DataFrame) -> float:
    """1 / Σ RC_i². Equals N for perfect risk parity, 1 if one name is all the risk."""
    rc = risk_contributions(w, cov).values
    return float(1.0 / (rc ** 2).sum())


# --------------------------------------------------------------------------
# Optimisers
# --------------------------------------------------------------------------

def _solve(objective, n: int, max_weight: float, sectors: pd.Series | None = None,
           max_sector: float | None = None, w0: np.ndarray | None = None) -> np.ndarray:
    bounds = [(0.0, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if sectors is not None and max_sector is not None:
        for sec in sectors.unique():
            mask = (sectors == sec).values.astype(float)
            cons.append({"type": "ineq", "fun": lambda w, m=mask: max_sector - m @ w})
    w0 = np.full(n, 1.0 / n) if w0 is None else w0
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def min_variance(cov: pd.DataFrame, max_weight: float = 0.10,
                 sectors: pd.Series | None = None, max_sector: float | None = None) -> pd.Series:
    C = cov.values
    w = _solve(lambda w: w @ C @ w, len(C), max_weight,
               sectors.reindex(cov.index) if sectors is not None else None, max_sector)
    return pd.Series(w, index=cov.index)


def risk_parity(cov: pd.DataFrame, max_weight: float = 0.10,
                sectors: pd.Series | None = None, max_sector: float | None = None) -> pd.Series:
    """Equal risk contribution: minimise Σ_i (RC_i - 1/N)²."""
    C = cov.values
    n = len(C)
    target = 1.0 / n

    def obj(w):
        pv = w @ C @ w
        rc = w * (C @ w) / pv
        return ((rc - target) ** 2).sum() * 1e4

    w = _solve(obj, n, max_weight,
               sectors.reindex(cov.index) if sectors is not None else None, max_sector)
    return pd.Series(w, index=cov.index)


def max_diversification(cov: pd.DataFrame, max_weight: float = 0.10) -> pd.Series:
    """Choueifaty & Coignard: maximise (w'σ) / sqrt(w'Σw)."""
    C = cov.values
    sig = np.sqrt(np.diag(C))

    def obj(w):
        return -(w @ sig) / np.sqrt(w @ C @ w)

    w = _solve(obj, len(C), max_weight)
    return pd.Series(w, index=cov.index)


def compare_allocations(returns: pd.DataFrame, current: pd.Series,
                        max_weight: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build equal-weight, min-variance, risk-parity and max-diversification
    portfolios on the same universe and compare to the current book.
    Returns (weights_table, stats_table).
    """
    cov, shrink = ledoit_wolf(returns)
    n = len(cov)
    allocs = {
        "Current": current.reindex(cov.index).fillna(0.0),
        "Equal weight": pd.Series(1.0 / n, index=cov.index),
        "Min variance": min_variance(cov, max_weight),
        "Risk parity": risk_parity(cov, max_weight),
        "Max diversification": max_diversification(cov, max_weight),
    }
    weights = pd.DataFrame(allocs)

    stats = {}
    for name, w in allocs.items():
        stats[name] = {
            "Ann. vol (shrunk cov)": portfolio_vol(w.values, cov.values) * np.sqrt(TRADING_DAYS),
            "Ann. vol (realised)": (returns @ w).std() * np.sqrt(TRADING_DAYS),
            "Eff. # of bets": effective_number_of_bets(w, cov),
            "Max weight": w.max(),
            "Top-5 weight": w.nlargest(5).sum(),
        }
    stats_df = pd.DataFrame(stats).T
    stats_df.attrs["shrinkage_intensity"] = shrink
    return weights, stats_df
