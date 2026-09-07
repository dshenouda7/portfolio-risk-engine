"""
Run the whole engine on a portfolio and summarise the findings in plain
English. This is what the CLI prints and what the dashboard's summary
panel shows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MarketData
from .factors import FactorModel, fit_factor_model, rolling_betas
from .optimize import compare_allocations, effective_number_of_bets, ledoit_wolf, risk_contributions
from .stress import run_all_scenarios
from .tail import TailReport, VaRBacktest, backtest_var, liquidity_adjusted_var, tail_report


@dataclass
class RiskReport:
    weights: pd.Series
    factor_model: FactorModel
    decomposition: dict
    idio_by_stock: pd.Series
    tail: TailReport
    backtest_hist: VaRBacktest
    backtest_param: VaRBacktest
    stress: pd.DataFrame
    risk_contrib: pd.Series
    enb: float
    alloc_weights: pd.DataFrame
    alloc_stats: pd.DataFrame
    # extensions (None when inputs unavailable)
    factor_model_sector: FactorModel | None = None
    resid_corr_ff: float | None = None
    resid_corr_sector: float | None = None
    rolling: pd.DataFrame | None = None
    liquidity: pd.DataFrame | None = None

    def key_findings(self) -> list[str]:
        """Bullet points a PM would actually want to hear."""
        d = self.decomposition
        f: list[str] = []

        f.append(
            f"Annualised volatility is {d['total_vol_annual']:.1%}: "
            f"{d['factor_share']:.0%} systematic (factor) risk, "
            f"{d['idio_share']:.0%} stock-specific."
        )

        pf = d["per_factor_share"].sort_values(ascending=False)
        top = pf.index[0]
        f.append(
            f"The largest single risk driver is {top} "
            f"({pf.iloc[0]:.0%} of total variance). "
            f"Portfolio beta to the market is {d['exposures']['Mkt-RF']:.2f}, "
            f"SMB exposure {d['exposures']['SMB']:+.2f}."
        )

        top_idio = self.idio_by_stock.index[0]
        f.append(
            f"{top_idio} alone contributes {self.idio_by_stock.iloc[0]:.1%} of total "
            f"variance through stock-specific risk; the top 3 names account for "
            f"{self.idio_by_stock.iloc[:3].sum():.1%}."
        )

        t = self.tail
        gap = t.monte_carlo_t[1] / max(t.parametric[1], 1e-9) - 1
        f.append(
            f"1-day {t.q:.0%} CVaR is {t.monte_carlo_t[1]:.2%} under a fat-tailed "
            f"(Student-t, ν={t.nu:.1f}) model vs {t.parametric[1]:.2%} assuming "
            f"normality. Normal assumption understates tail loss by {gap:.0%}."
        )

        bt = self.backtest_hist
        f.append(
            f"Rolling historical VaR backtest: {bt.n_breaches} breaches in "
            f"{bt.n_obs} days (expected {bt.expected_breaches:.1f}). "
            f"Kupiec: {bt.verdict}. Christoffersen: {bt.independence_verdict}."
        )

        worst = self.stress["Portfolio P&L"].idxmin()
        f.append(
            f"Worst stress scenario is '{worst}' at "
            f"{self.stress.loc[worst, 'Portfolio P&L']:.1%} factor-implied P&L."
        )

        f.append(
            f"Effective number of independent bets: {self.enb:.1f} "
            f"(out of {len(self.weights)} holdings). A risk-parity allocation "
            f"on the same names would run at "
            f"{self.alloc_stats.loc['Risk parity', 'Ann. vol (shrunk cov)']:.1%} vol vs "
            f"{self.alloc_stats.loc['Current', 'Ann. vol (shrunk cov)']:.1%} currently."
        )

        if self.resid_corr_sector is not None:
            f.append(
                f"Adding sector factors cut mean |residual correlation| from "
                f"{self.resid_corr_ff:.3f} to {self.resid_corr_sector:.3f}"
                + (": the Fama-French factors alone were missing a common sector driver."
                   if self.resid_corr_sector < 0.8 * self.resid_corr_ff
                   else "; the FF factors already capture most common movement.")
            )

        if self.rolling is not None and len(self.rolling) > 2:
            mb = self.rolling["Mkt-RF"]
            f.append(
                f"Rolling 1y market beta has ranged {mb.min():.2f} to {mb.max():.2f} "
                f"(now {mb.iloc[-1]:.2f}); exposures are not static."
            )

        if self.liquidity is not None:
            p = self.liquidity.attrs["portfolio"]
            slowest = self.liquidity.index[0]
            f.append(
                f"Liquidity at ${p['portfolio_value']/1e6:.1f}M AUM and {p['participation']:.0%} of ADV: "
                f"weighted exit is {p['avg_liquidation_days']:.1f} days, slowest name {slowest} at "
                f"{self.liquidity.loc[slowest, 'Days to liquidate']:.1f} days; liquidity-adjusted "
                f"{self.tail.q:.0%} VaR {p['liquidity_adjusted_var']:.2%} vs {p['base_var']:.2%}. "
                f"Capacity: {p['capacity_binding_name']} becomes a 5-day exit at "
                f"${p['capacity_aum']/1e6:.1f}M AUM."
            )
        return f

    def to_text(self) -> str:
        lines = ["=" * 72, "PORTFOLIO RISK REPORT", "=" * 72, ""]
        lines.append("KEY FINDINGS")
        for i, k in enumerate(self.key_findings(), 1):
            lines.append(f"  {i}. {k}")
        lines += ["", "FACTOR EXPOSURES (portfolio betas)"]
        lines.append(self.decomposition["exposures"].round(3).to_string())
        lines += ["", "VARIANCE SHARE BY FACTOR"]
        lines.append((self.decomposition["per_factor_share"]).map("{:.1%}".format).to_string())
        lines += ["", f"TAIL RISK (1-day, {self.tail.q:.0%})"]
        lines.append(self.tail.table().map("{:.2%}".format).to_string())
        lines += ["", "STRESS TESTS (factor-implied P&L)"]
        lines.append(self.stress[["Type", "Days", "Portfolio P&L", "Idio ±1σ"]]
                     .assign(**{"Portfolio P&L": lambda d: d["Portfolio P&L"].map("{:+.1%}".format),
                                "Idio ±1σ": lambda d: d["Idio ±1σ"].map("{:.1%}".format)})
                     .to_string())
        lines += ["", "ALLOCATION COMPARISON"]
        lines.append(self.alloc_stats.round(3).to_string())
        if self.liquidity is not None:
            lines += ["", "LIQUIDITY"]
            lines.append(self.liquidity[["Weight", "Days to liquidate", "Impact cost (% of position)"]]
                         .head(10).assign(Weight=lambda d: d["Weight"].map("{:.1%}".format),
                                          **{"Impact cost (% of position)":
                                             lambda d: d["Impact cost (% of position)"].map("{:.2%}".format)})
                         .round(1).to_string())
        lines.append("")
        return "\n".join(lines)


def run_full_analysis(md: MarketData, weights: pd.Series, q: float = 0.99,
                      var_window: int = 250, max_weight: float = 0.10,
                      portfolio_value: float = 1_100_000.0,
                      participation: float = 0.20) -> RiskReport:
    tickers = [t for t in weights.index if t in md.returns.columns]
    missing = set(weights.index) - set(tickers)
    if missing:
        print(f"Warning: no data for {sorted(missing)}, dropping and renormalising.")
    w = weights.reindex(tickers)
    w = w / w.sum()
    R = md.returns[tickers]

    fm = fit_factor_model(md, tickers)
    dec = fm.variance_decomposition(w)
    idio = fm.idio_contribution_by_stock(w)

    tail = tail_report(R, w, q)
    bt_h = backtest_var(R, w, q, var_window, "historical")
    bt_p = backtest_var(R, w, q, var_window, "parametric")

    stress = run_all_scenarios(fm, w, md.factors)

    cov, _ = ledoit_wolf(R.iloc[-500:])
    rc = risk_contributions(w, cov).sort_values(ascending=False)
    enb = effective_number_of_bets(w, cov)
    aw, ast = compare_allocations(R.iloc[-500:], w, max_weight)

    # --- extensions ---------------------------------------------------
    fm_sec, rc_ff, rc_sec = None, None, None
    if md.sectors is not None and md.sectors.reindex(tickers).notna().sum() >= 4:
        rc_ff = fm.residual_correlation_check()
        fm_sec = fit_factor_model(md, tickers, sectors=md.sectors.reindex(tickers))
        rc_sec = fm_sec.residual_correlation_check()

    roll = rolling_betas(md, w, window=250, step=5) if len(R) > 300 else None

    liq = None
    if md.adv is not None:
        daily_vols = R.iloc[-250:].std()
        liq = liquidity_adjusted_var(w, md.adv, portfolio_value, tail.historical[0],
                                     participation=participation, daily_vols=daily_vols)

    return RiskReport(
        weights=w, factor_model=fm, decomposition=dec, idio_by_stock=idio,
        tail=tail, backtest_hist=bt_h, backtest_param=bt_p, stress=stress,
        risk_contrib=rc, enb=enb, alloc_weights=aw, alloc_stats=ast,
        factor_model_sector=fm_sec, resid_corr_ff=rc_ff, resid_corr_sector=rc_sec,
        rolling=roll, liquidity=liq,
    )
