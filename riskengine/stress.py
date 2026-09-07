"""
Stress testing.

Two flavours:

  historical replay : take a named window (e.g. COVID crash), pull the
                      realised factor returns over that window, and push
                      them through the portfolio's *current* factor
                      exposures. Answers "if that happened again to the
                      book I hold today, what would it do?"

  hypothetical shock: specify factor moves directly ("small caps -15%,
                      market -10%") and propagate through exposures.

Both use the factor model, so the stress P&L is the factor-implied part
only. We also report the idiosyncratic 'noise band' so the user knows how
much stock-specific randomness could sit on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factors import FactorModel

# Windows chosen to be identifiable in FF daily data.
HISTORICAL_SCENARIOS = {
    "GFC (Sep-Nov 2008)": ("2008-09-01", "2008-11-30"),
    "Flash Crash week (May 2010)": ("2010-05-03", "2010-05-10"),
    "Aug 2011 US downgrade": ("2011-08-01", "2011-08-31"),
    "Dec 2018 selloff": ("2018-12-03", "2018-12-24"),
    "COVID crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 rate shock (Jan-Jun)": ("2022-01-03", "2022-06-30"),
    "Aug 2024 carry unwind": ("2024-08-01", "2024-08-07"),
}

# Hypothetical shocks: cumulative factor returns over the scenario.
HYPOTHETICAL_SCENARIOS = {
    "Broad selloff -10%": {"Mkt-RF": -0.10},
    "Small-cap rout": {"Mkt-RF": -0.08, "SMB": -0.10},
    "Value rotation": {"Mkt-RF": 0.0, "HML": 0.10, "Mom": -0.08},
    "Quality flight": {"Mkt-RF": -0.05, "RMW": 0.06, "SMB": -0.05},
    "Momentum crash": {"Mom": -0.20, "Mkt-RF": 0.03},
    "Rally +10%": {"Mkt-RF": 0.10, "SMB": 0.04},
}


@dataclass
class StressResult:
    name: str
    kind: str                     # 'historical' or 'hypothetical'
    factor_moves: pd.Series       # cumulative factor returns applied
    pnl_by_factor: pd.Series      # contribution of each factor
    total_pnl: float              # factor-implied portfolio return
    idio_band: float              # 1-sigma idiosyncratic noise over the window
    n_days: int


def _apply_shock(fm: FactorModel, w: pd.Series, moves: pd.Series,
                 n_days: int, name: str, kind: str) -> StressResult:
    x = fm.portfolio_exposures(w)
    moves = moves.reindex(fm.factor_names).fillna(0.0)
    pnl = x * moves
    w_al = w.reindex(fm.betas.index).fillna(0.0)
    idio_daily_var = float((w_al ** 2 * fm.resid_var).sum())
    return StressResult(
        name=name, kind=kind, factor_moves=moves, pnl_by_factor=pnl,
        total_pnl=float(pnl.sum()),
        idio_band=float(np.sqrt(idio_daily_var * n_days)),
        n_days=n_days,
    )


def historical_replay(fm: FactorModel, w: pd.Series, factors: pd.DataFrame,
                      name: str, start: str, end: str) -> StressResult | None:
    win = factors.loc[start:end]
    if len(win) < 2:
        return None
    cum = (1 + win).prod() - 1
    return _apply_shock(fm, w, cum, len(win), name, "historical")


def hypothetical_shock(fm: FactorModel, w: pd.Series, name: str,
                       shocks: dict[str, float], horizon_days: int = 20) -> StressResult:
    return _apply_shock(fm, w, pd.Series(shocks), horizon_days, name, "hypothetical")


def run_all_scenarios(fm: FactorModel, w: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Run every built-in scenario the data can support; return a summary table."""
    results: list[StressResult] = []
    for name, (s, e) in HISTORICAL_SCENARIOS.items():
        r = historical_replay(fm, w, factors, name, s, e)
        if r is not None:
            results.append(r)
    for name, shocks in HYPOTHETICAL_SCENARIOS.items():
        results.append(hypothetical_shock(fm, w, name, shocks))

    rows = []
    for r in results:
        row = {"Scenario": r.name, "Type": r.kind, "Days": r.n_days,
               "Portfolio P&L": r.total_pnl, "Idio ±1σ": r.idio_band}
        for f in fm.factor_names:
            row[f"via {f}"] = r.pnl_by_factor[f]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Scenario")
