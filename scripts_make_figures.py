"""
Generate README figures. Run after a live `python cli.py examples/portfolio.csv`
so the cache is populated, or pass --synthetic.

    python scripts_make_figures.py [--synthetic]
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from riskengine import *
from riskengine.data import load_portfolio

INK, LOSS, MUTED, GAIN = "#1B2A41", "#C0392B", "#8A94A6", "#2E7D5B"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
out = Path("docs"); out.mkdir(exist_ok=True)

if "--synthetic" in sys.argv:
    md = synthetic_market(n_assets=20, n_days=2000, seed=11)
    w = pd.Series(np.random.default_rng(1).dirichlet(np.ones(20) * 3), index=md.tickers)
else:
    w, sectors = load_portfolio("examples/portfolio.csv", with_sectors=True)
    md = load_market(list(w.index), "2018-01-01", sectors=sectors)

rep = run_full_analysis(md, w)
d = rep.decomposition

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
d["exposures"].plot.bar(ax=ax[0], color=INK, title="Portfolio factor exposures"); ax[0].axhline(0, color="k", lw=0.5)
sh = d["per_factor_share"].copy(); sh["Stock-specific"] = d["idio_share"]
sh.plot.barh(ax=ax[1], color=[INK]*6 + [MUTED], title="Share of portfolio variance")
ax[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
plt.tight_layout(); plt.savefig(out / "factors.png", dpi=130); plt.close()

bt = rep.backtest_hist
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(bt.realised, color=INK, lw=0.6, label="Realised")
ax.plot(-bt.var_series, color=LOSS, lw=1.2, label="-VaR 99% (rolling hist.)")
br = bt.realised[bt.breaches]; ax.scatter(br.index, br, color=LOSS, marker="x", s=40, zorder=3, label="Breach")
ax.legend(loc="lower left"); ax.set_title(f"VaR backtest: {bt.n_breaches} breaches / {bt.n_obs} days. Kupiec {bt.verdict}")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
plt.tight_layout(); plt.savefig(out / "backtest.png", dpi=130); plt.close()

s = rep.stress
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.barh(s.index, s["Portfolio P&L"], xerr=s["Idio ±1σ"], color=[LOSS if v < 0 else GAIN for v in s["Portfolio P&L"]], ecolor=MUTED, capsize=3)
ax.axvline(0, color="k", lw=0.5); ax.set_title("Stress scenarios: factor-implied P&L ± idiosyncratic 1σ")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
plt.tight_layout(); plt.savefig(out / "stress.png", dpi=130); plt.close()

(out / "findings.txt").write_text("\n".join(f"{i}. {k}" for i, k in enumerate(rep.key_findings(), 1)))
print("wrote docs/factors.png, backtest.png, stress.png, findings.txt")
