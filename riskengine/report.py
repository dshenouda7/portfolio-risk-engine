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
from .factors import FactorModel, fit_factor_model
from .optimize import compare_allocations, effective_number_of_bets, ledoit_wolf, risk_contributions
from .stress import run_all_scenarios
from .tail import TailReport, VaRBacktest, backtest_var, tail_report


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
            f"Kupiec test: {bt.verdict}."
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
        lines.append("")
        return "\n".join(lines)


def run_full_analysis(md: MarketData, weights: pd.Series, q: float = 0.99,
                      var_window: int = 250, max_weight: float = 0.10) -> RiskReport:
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

    return RiskReport(
        weights=w, factor_model=fm, decomposition=dec, idio_by_stock=idio,
        tail=tail, backtest_hist=bt_h, backtest_param=bt_p, stress=stress,
        risk_contrib=rc, enb=enb, alloc_weights=aw, alloc_stats=ast,
    )
