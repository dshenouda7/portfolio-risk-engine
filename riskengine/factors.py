"""
Factor risk model.

For each holding, regress daily excess returns on the Fama-French 5 factors
plus momentum:

    r_i - rf = alpha_i + B_i . F + eps_i

Then decompose portfolio variance:

    Var(r_p) = w' (B Σ_F B' + D) w
             = factor variance + idiosyncratic variance

where B is the (N x K) exposure matrix, Σ_F the factor covariance, and
D the diagonal matrix of residual variances (the model assumes residuals
are uncorrelated across stocks; we report how well that holds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import MarketData

TRADING_DAYS = 252


@dataclass
class FactorModel:
    betas: pd.DataFrame          # N x K exposures
    alphas: pd.Series            # N, daily alpha
    tstats: pd.DataFrame         # N x K t-statistics of betas
    r2: pd.Series                # N, regression R^2
    resid_var: pd.Series         # N, daily residual variance
    residuals: pd.DataFrame      # T x N
    factor_cov: pd.DataFrame     # K x K daily factor covariance
    factor_names: list[str]

    # ---- portfolio-level analytics -------------------------------------

    def portfolio_exposures(self, w: pd.Series) -> pd.Series:
        w = w.reindex(self.betas.index).fillna(0.0)
        return self.betas.T @ w

    def variance_decomposition(self, w: pd.Series) -> dict:
        """
        Split annualised portfolio variance into factor and idiosyncratic
        parts, and attribute the factor part to individual factors.

        Per-factor contribution uses the standard marginal-contribution
        identity: contribution_k = x_k * (Σ_F x)_k, where x = B'w. These sum
        exactly to total factor variance (cross-factor covariance is split
        proportionally between the two factors involved).
        """
        w = w.reindex(self.betas.index).fillna(0.0)
        x = self.betas.T @ w                          # K portfolio exposures
        Sf = self.factor_cov.values
        factor_var = float(x.values @ Sf @ x.values)
        idio_var = float((w.values ** 2 * self.resid_var.values).sum())
        total = factor_var + idio_var

        per_factor = pd.Series(x.values * (Sf @ x.values), index=self.factor_names)

        return {
            "total_vol_annual": np.sqrt(total * TRADING_DAYS),
            "factor_vol_annual": np.sqrt(factor_var * TRADING_DAYS),
            "idio_vol_annual": np.sqrt(idio_var * TRADING_DAYS),
            "factor_share": factor_var / total,
            "idio_share": idio_var / total,
            "per_factor_share": per_factor / total,          # sums to factor_share
            "exposures": x,
        }

    def idio_contribution_by_stock(self, w: pd.Series) -> pd.Series:
        """Share of total variance coming from each name's specific risk."""
        w = w.reindex(self.betas.index).fillna(0.0)
        dec = self.variance_decomposition(w)
        total_daily = (dec["total_vol_annual"] ** 2) / TRADING_DAYS
        contrib = (w ** 2 * self.resid_var) / total_daily
        return contrib.sort_values(ascending=False)

    def model_covariance(self) -> pd.DataFrame:
        """Full N x N covariance implied by the factor model (daily)."""
        B = self.betas.values
        cov = B @ self.factor_cov.values @ B.T + np.diag(self.resid_var.values)
        return pd.DataFrame(cov, index=self.betas.index, columns=self.betas.index)

    def residual_correlation_check(self) -> float:
        """
        Mean absolute off-diagonal residual correlation. If this is large
        (> ~0.15) the 'uncorrelated residuals' assumption is being violated
        and idiosyncratic risk is understated.
        """
        c = self.residuals.corr().values
        off = c[~np.eye(len(c), dtype=bool)]
        return float(np.abs(off).mean())


def fit_factor_model(md: MarketData, tickers: list[str] | None = None,
                     lookback_days: int | None = None) -> FactorModel:
    """OLS of each stock's excess returns on the factors."""
    ex = md.excess_returns
    if tickers is not None:
        ex = ex[[t for t in tickers if t in ex.columns]]
    F = md.factors
    if lookback_days:
        ex, F = ex.iloc[-lookback_days:], F.iloc[-lookback_days:]

    X = sm.add_constant(F.values)
    names = list(F.columns)
    betas, alphas, tstats, r2, rvar, resid = {}, {}, {}, {}, {}, {}

    for t in ex.columns:
        y = ex[t].values
        res = sm.OLS(y, X).fit()
        alphas[t] = res.params[0]
        betas[t] = res.params[1:]
        tstats[t] = res.tvalues[1:]
        r2[t] = res.rsquared
        rvar[t] = res.resid.var(ddof=X.shape[1])
        resid[t] = res.resid

    return FactorModel(
        betas=pd.DataFrame(betas, index=names).T,
        alphas=pd.Series(alphas),
        tstats=pd.DataFrame(tstats, index=names).T,
        r2=pd.Series(r2),
        resid_var=pd.Series(rvar),
        residuals=pd.DataFrame(resid, index=ex.index),
        factor_cov=F.cov(),
        factor_names=names,
    )
