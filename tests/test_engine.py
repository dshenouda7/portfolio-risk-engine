import numpy as np
import pandas as pd
import pytest

from riskengine import (fit_factor_model, kupiec_pof, ledoit_wolf, min_variance, risk_parity,
                        run_full_analysis, synthetic_market, tail_report, backtest_var)
from riskengine.optimize import risk_contributions, effective_number_of_bets
from riskengine.tail import historical_var_cvar, parametric_var_cvar


@pytest.fixture(scope="module")
def market():
    return synthetic_market(n_assets=10, n_days=1500, seed=3)


@pytest.fixture(scope="module")
def weights(market):
    return pd.Series(1.0 / 10, index=market.tickers)


def test_factor_model_recovers_true_betas(market):
    fm = fit_factor_model(market)
    corr = np.corrcoef(fm.betas.values.ravel(), market.true_betas.values.ravel())[0, 1]
    assert corr > 0.95
    # Market betas should be close in absolute terms too
    err = (fm.betas["Mkt-RF"] - market.true_betas["Mkt-RF"]).abs().mean()
    assert err < 0.1


def test_variance_decomposition_sums(market, weights):
    fm = fit_factor_model(market)
    d = fm.variance_decomposition(weights)
    assert abs(d["factor_share"] + d["idio_share"] - 1) < 1e-9
    assert abs(d["per_factor_share"].sum() - d["factor_share"]) < 1e-9
    # Model total vol should be close to realised vol
    realised = (market.returns @ weights).std() * np.sqrt(252)
    assert abs(d["total_vol_annual"] / realised - 1) < 0.15


def test_cvar_at_least_var(market, weights):
    pr = market.returns @ weights
    for fn in (historical_var_cvar, parametric_var_cvar):
        v, c = fn(pr, 0.99)
        assert c >= v > 0


def test_tail_report_fat_tails_exceed_normal(market, weights):
    t = tail_report(market.returns, weights, 0.99)
    # Data is Student-t generated; t-MC CVaR should exceed normal CVaR
    assert t.monte_carlo_t[1] > t.monte_carlo_normal[1]
    assert t.nu < 30


def test_kupiec_exact_rate_is_not_rejected():
    lr, p = kupiec_pof(1000, 10, 0.99)
    assert lr < 1e-9 and p > 0.99


def test_kupiec_rejects_bad_model():
    _, p = kupiec_pof(1000, 40, 0.99)
    assert p < 0.001


def test_backtest_breach_rate_reasonable(market, weights):
    bt = backtest_var(market.returns, weights, 0.99, 250, "historical")
    assert 0.002 < bt.breach_rate < 0.03


def test_ledoit_wolf_shrinkage_in_unit_interval(market):
    cov, delta = ledoit_wolf(market.returns)
    assert 0 <= delta <= 1
    eig = np.linalg.eigvalsh(cov.values)
    assert eig.min() > 0  # positive definite


def test_optimisers_respect_constraints(market):
    cov, _ = ledoit_wolf(market.returns)
    for fn in (min_variance, risk_parity):
        w = fn(cov, max_weight=0.15)
        assert abs(w.sum() - 1) < 1e-6
        assert (w >= -1e-9).all()
        assert w.max() <= 0.15 + 1e-6


def test_risk_parity_equalises_contributions(market):
    cov, _ = ledoit_wolf(market.returns)
    w = risk_parity(cov, max_weight=0.5)
    rc = risk_contributions(w, cov)
    assert rc.std() < 0.01
    assert effective_number_of_bets(w, cov) > 9.5


def test_min_variance_beats_equal_weight(market):
    cov, _ = ledoit_wolf(market.returns)
    w_mv = min_variance(cov, max_weight=0.5)
    w_eq = pd.Series(0.1, index=cov.index)
    v = lambda w: w.values @ cov.values @ w.values
    assert v(w_mv) <= v(w_eq) + 1e-12


def test_full_pipeline_runs(market, weights):
    rep = run_full_analysis(market, weights)
    assert len(rep.key_findings()) == 7
    assert "PORTFOLIO RISK REPORT" in rep.to_text()
