"""
Command-line runner.

    python cli.py examples/portfolio.csv                 # live data
    python cli.py examples/portfolio.csv --start 2016-01-01 --q 0.95
    python cli.py --synthetic                            # offline demo
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from riskengine import load_market, load_portfolio, run_full_analysis, synthetic_market


def main() -> None:
    p = argparse.ArgumentParser(description="Portfolio risk engine")
    p.add_argument("portfolio", nargs="?", help="CSV with ticker,weight columns")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--q", type=float, default=0.99, help="VaR confidence level")
    p.add_argument("--window", type=int, default=250, help="Rolling VaR window (days)")
    p.add_argument("--max-weight", type=float, default=0.10)
    p.add_argument("--synthetic", action="store_true", help="Use generated data (no network)")
    p.add_argument("--out", default=None, help="Write report text to this file")
    args = p.parse_args()

    if args.synthetic:
        md = synthetic_market(n_assets=15, n_days=2000)
        rng = np.random.default_rng(1)
        w = pd.Series(rng.dirichlet(np.ones(15) * 3), index=md.tickers)
    else:
        if not args.portfolio:
            p.error("portfolio CSV required unless --synthetic")
        w = load_portfolio(args.portfolio)
        md = load_market(list(w.index), args.start, args.end)

    rep = run_full_analysis(md, w, q=args.q, var_window=args.window, max_weight=args.max_weight)
    text = rep.to_text()
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
