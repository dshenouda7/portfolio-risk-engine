"""Portfolio risk engine for long-only small/mid-cap equity books."""

from .data import MarketData, load_market, load_portfolio, synthetic_market
from .factors import FactorModel, fit_factor_model
from .optimize import (compare_allocations, effective_number_of_bets, ledoit_wolf,
                       max_diversification, min_variance, risk_contributions, risk_parity)
from .report import RiskReport, run_full_analysis
from .stress import HISTORICAL_SCENARIOS, HYPOTHETICAL_SCENARIOS, historical_replay, hypothetical_shock, run_all_scenarios
from .tail import backtest_var, kupiec_pof, tail_report

__version__ = "0.1.0"
