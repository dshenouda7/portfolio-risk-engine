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


def sector_factor_returns(returns: pd.DataFrame, sectors: pd.Series,
                          market_factor: pd.Series, exclude: str | None = None
                          ) -> pd.DataFrame:
    """
    Build one return series per sector: the equal-weighted return of the
    stocks in that sector, orthogonalised against the market factor so the
    sector factors capture *sector-specific* co-movement rather than
    re-measuring market beta. Returns a T x S DataFrame.

    `exclude` drops one ticker from its own sector average. With only a
    handful of names per sector, a stock regressed on an average that
    contains itself gets a spuriously good fit and its residual becomes
    negatively correlated with its peers'. Leave-one-out avoids that.
    """
    out = {}
    for sec in sorted(sectors.dropna().unique()):
        names = [t for t in sectors[sectors == sec].index
                 if t in returns.columns and t != exclude]
        if len(names) < 1 or (exclude is None and len(names) < 2):
            continue
        ew = returns[names].mean(axis=1)
        X = sm.add_constant(market_factor.reindex(ew.index).values)
        res = sm.OLS(ew.values, X).fit()
        out[f"SEC:{sec}"] = pd.Series(res.resid, index=ew.index)
    return pd.DataFrame(out)


def fit_factor_model(md: MarketData, tickers: list[str] | None = None,
                     lookback_days: int | None = None,
                     sectors: pd.Series | None = None) -> FactorModel:
    """
    OLS of each stock's excess returns on the factors.

    If `sectors` (ticker -> sector label) is given, market-orthogonalised
    sector return factors are appended to the Fama-French set. Compare
    `residual_correlation_check()` with and without them to see whether
    the FF factors alone were missing a common driver.
    """
    ex = md.excess_returns
    if tickers is not None:
        ex = ex[[t for t in tickers if t in ex.columns]]
    F = md.factors
    if lookback_days:
        ex, F = ex.iloc[-lookback_days:], F.iloc[-lookback_days:]
    sec_full = None
    if sectors is not None:
        # Full (all-in) sector factors define the factor set and covariance;
        # each stock's own regression uses leave-one-out versions.
        sec_full = sector_factor_returns(ex, sectors, F["Mkt-RF"])
        F = F.join(sec_full, how="inner")
        ex = ex.loc[F.index]

    names = list(F.columns)
    X_base = sm.add_constant(F.values)
    betas, alphas, tstats, r2, rvar, resid = {}, {}, {}, {}, {}, {}

    for t in ex.columns:
        y = ex[t].values
        X = X_base
        if sec_full is not None:
            loo = sector_factor_returns(ex, sectors, F["Mkt-RF"], exclude=t)
            Ft = F.copy()
            for c in sec_full.columns:
                # A sector with no *other* members has no leave-one-out
                # factor; zero the column so the stock gets beta 0 to it.
                Ft[c] = loo[c].reindex(Ft.index).values if c in loo.columns else 0.0
            X = sm.add_constant(Ft.values)
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


def rolling_betas(md: MarketData, w: pd.Series, window: int = 250, step: int = 5,
                  factors: list[str] | None = None) -> pd.DataFrame:
    """
    Portfolio-level factor exposures estimated on a rolling window, so you
    can see whether 'the book is 1.1 beta' was true a year ago too.
    Uses current weights throughout (what-you-hold-now, through time).
    Returns T' x K DataFrame indexed by window end date.
    """
    tickers = [t for t in w.index if t in md.returns.columns]
    w = w.reindex(tickers)
    w = w / w.sum()
    pr = md.excess_returns[tickers] @ w
    F = md.factors if factors is None else md.factors[factors]
    rows, idx = [], []
    for end in range(window, len(pr) + 1, step):
        y = pr.iloc[end - window:end].values
        X = sm.add_constant(F.iloc[end - window:end].values)
        rows.append(sm.OLS(y, X).fit().params[1:])
        idx.append(pr.index[end - 1])
    return pd.DataFrame(rows, index=idx, columns=F.columns)
