# Portfolio Risk Engine

Factor decomposition, tail risk, stress testing and portfolio construction for a long-only US small/mid-cap equity book. Built in Python, runs from the command line or as a Streamlit dashboard, and works offline on synthetic data.

The question it answers: *what does this portfolio actually own, and how much can it lose?*

## Quick start

```bash
pip install -r requirements.txt

# Offline demo on synthetic data (no network needed)
python cli.py --synthetic

# Real portfolio: CSV with ticker,weight columns
python cli.py examples/portfolio.csv --start 2018-01-01

# Dashboard
streamlit run app.py

# Tests
pytest
```

The first live run downloads prices (yfinance) and Fama-French factors (Ken French data library) and caches them in `data_cache/`; subsequent runs are instant.

## What it computes

**Factor risk model** (`riskengine/factors.py`)
Each holding's daily excess return is regressed on the Fama-French 5 factors plus momentum. Portfolio variance is then decomposed as

    Var(r_p) = w'(B Σ_F B' + D)w = factor variance + idiosyncratic variance

with per-factor attribution via marginal contributions (x_k · (Σ_F x)_k, where x = B'w), which sum exactly to total factor variance. Output: "the book is 1.14 beta, 81% of variance is market, SMB is the second driver, and one name contributes 3% of variance on its own." The module also reports mean absolute residual correlation as a diagnostic for missing factors.

**Tail risk** (`riskengine/tail.py`)
One-day VaR and CVaR at a chosen confidence level, estimated four ways and compared:

| Method | Assumption | Why include it |
|---|---|---|
| Historical | none | baseline; only knows about losses that already happened |
| Parametric | Gaussian | what most people compute; understates tails |
| Monte Carlo (Normal) | Gaussian, full covariance | isolates the effect of the distribution choice |
| Monte Carlo (Student-t) | fat tails, full covariance | closer to how equities actually behave |

Degrees of freedom for the t-distribution are fitted from pooled standardised returns. The gap between the t and Gaussian CVaR is reported as "how much a normal assumption understates tail loss."

**VaR backtesting**
Rolling out-of-sample VaR: each day's estimate uses only the preceding window. Breaches are counted and tested with the Kupiec proportion-of-failures test (likelihood ratio, χ²(1)). A model at 99% should be breached about 1% of days; the test says whether the observed count is statistically consistent with that.

**Stress testing** (`riskengine/stress.py`)
Historical replay pushes realised factor returns from named windows (GFC, COVID crash, 2022 rate shock, Aug 2024 unwind, …) through the portfolio's *current* exposures. Hypothetical shocks ("small-cap rout", "momentum crash") are specified directly as factor moves. Both report factor-implied P&L with an idiosyncratic ±1σ band so the user knows how much stock-specific noise could sit on top.

**Portfolio construction** (`riskengine/optimize.py`)
Ledoit-Wolf (2004) shrinkage of the covariance matrix toward scaled identity, implemented from the closed-form optimal intensity. On top of that: minimum variance, equal risk contribution (risk parity), and maximum diversification optimisers, all long-only with position (and optional sector) caps. Risk contributions and the effective number of bets (1/ΣRC²) are computed for the current book and each alternative.

## Repository layout

```
riskengine/
  data.py       prices, factor downloads, caching, synthetic market generator
  factors.py    factor regressions, variance decomposition
  tail.py       VaR/CVaR estimators, Monte Carlo, Kupiec backtest
  stress.py     historical replay and hypothetical shocks
  optimize.py   Ledoit-Wolf, risk contributions, optimisers
  report.py     orchestration + plain-English findings
app.py          Streamlit dashboard
cli.py          command-line runner
tests/          pytest suite (beta recovery, decomposition identities, Kupiec, optimiser constraints)
examples/       sample small/mid-cap portfolio
```

## Design choices worth knowing

- **Synthetic data is first-class.** `synthetic_market()` generates returns from a known 6-factor model with Student-t noise and an embedded crash. Tests check the engine recovers the true betas (corr > 0.95), which is the only way to verify the estimation code is right rather than merely running.
- **Sign convention:** all VaR/CVaR numbers are positive losses as a fraction of portfolio value.
- **Residual independence is checked, not assumed.** If residual correlation is high, the idiosyncratic component is understated and a sector factor should be added.
- **Shrinkage over sample covariance.** With N names and T days the sample matrix has N(N+1)/2 free parameters; optimisers on it chase noise. Shrinkage intensity δ is reported so you can see how much the data is being trusted.

## Limitations

- Factor model is linear and static over the estimation window; exposures drift.
- Historical replay applies *cumulative* factor moves to current betas and ignores path dependence.
- Liquidity risk is not modelled; for small caps this matters and is on the roadmap.
- yfinance data quality varies; for anything beyond a student fund, use a paid vendor.

## Roadmap

- Sector factor (GICS dummies) alongside the FF factors
- Rolling-beta view to show exposure drift
- Liquidity-adjusted VaR using ADV and position size
- Conditional (Christoffersen) backtest for breach clustering
- Trade-list generator: minimum-turnover path from current weights to a target allocation

## References

- Fama & French (2015), *A five-factor asset pricing model*
- Kupiec (1995), *Techniques for verifying the accuracy of risk measurement models*
- Ledoit & Wolf (2004), *A well-conditioned estimator for large-dimensional covariance matrices*
- Choueifaty & Coignard (2008), *Toward maximum diversification*
- Artzner et al. (1999), *Coherent measures of risk*
