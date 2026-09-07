"""
Streamlit dashboard.

    streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from riskengine import load_market, run_full_analysis, synthetic_market
from riskengine.data import load_portfolio
from riskengine.stress import hypothetical_shock
from riskengine.tail import monte_carlo_var_cvar, portfolio_returns

st.set_page_config(page_title="Portfolio Risk Engine", layout="wide")

# Palette: deep navy ink, one signal colour for losses, muted greys.
INK, LOSS, GAIN, MUTED = "#1B2A41", "#C0392B", "#2E7D5B", "#8A94A6"
FACTOR_COLORS = ["#1B2A41", "#3E5C76", "#748CAB", "#A8B5C8", "#C9A227", "#7A5C99"]

st.title("Portfolio risk engine")
st.caption("Factor decomposition, tail risk, stress tests and construction for a long-only equity book.")

# --------------------------------------------------------------------------
# Sidebar: inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Portfolio")
    mode = st.radio("Data source", ["Upload CSV (live prices)", "Synthetic demo"], index=1)
    q = st.select_slider("VaR confidence", [0.95, 0.975, 0.99], value=0.99)
    window = st.slider("Rolling VaR window (days)", 120, 500, 250, step=10)
    max_w = st.slider("Max position weight in optimisers", 0.05, 0.25, 0.10, step=0.01)
    start = st.text_input("Price history start", "2018-01-01")

    uploaded = None
    if mode.startswith("Upload"):
        uploaded = st.file_uploader("CSV with `ticker,weight` columns", type="csv")
        st.caption("Weights can be decimals or percents; they're renormalised.")


@st.cache_data(show_spinner="Pulling prices and factor data…")
def _load_live(csv_bytes: bytes, start: str):
    w = load_portfolio(io.BytesIO(csv_bytes))
    md = load_market(list(w.index), start)
    return md, w


@st.cache_data
def _load_synthetic():
    md = synthetic_market(n_assets=15, n_days=2000)
    rng = np.random.default_rng(1)
    w = pd.Series(rng.dirichlet(np.ones(15) * 3), index=md.tickers)
    return md, w


if mode.startswith("Upload"):
    if uploaded is None:
        st.info("Upload a portfolio CSV in the sidebar to begin, or switch to the synthetic demo.")
        st.stop()
    md, w = _load_live(uploaded.getvalue(), start)
else:
    md, w = _load_synthetic()

with st.spinner("Running risk engine…"):
    rep = run_full_analysis(md, w, q=q, var_window=window, max_weight=max_w)

d = rep.decomposition

# --------------------------------------------------------------------------
# Headline
# --------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Annualised vol", f"{d['total_vol_annual']:.1%}")
c2.metric("Market beta", f"{d['exposures']['Mkt-RF']:.2f}")
c3.metric(f"1-day CVaR {q:.0%} (t)", f"{rep.tail.monte_carlo_t[1]:.2%}")
c4.metric("Systematic share", f"{d['factor_share']:.0%}")
c5.metric("Effective bets", f"{rep.enb:.1f} / {len(w)}")

st.subheader("What the numbers say")
for k in rep.key_findings():
    st.markdown(f"- {k}")

tabs = st.tabs(["Factor risk", "Tail risk", "VaR backtest", "Stress tests", "Construction", "Holdings"])

# --------------------------------------------------------------------------
# Factor risk
# --------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Portfolio factor exposures**")
        ex = d["exposures"]
        fig = go.Figure(go.Bar(x=ex.index, y=ex.values, marker_color=FACTOR_COLORS))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="beta")
        st.plotly_chart(fig, width='stretch')
    with right:
        st.markdown("**Where the variance comes from**")
        shares = d["per_factor_share"].copy()
        shares["Stock-specific"] = d["idio_share"]
        fig = px.pie(values=shares.values, names=shares.index, hole=0.55,
                     color_discrete_sequence=FACTOR_COLORS + [MUTED])
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')

    st.markdown("**Stock-level exposures** (t-stats below 2 in absolute value are not statistically distinguishable from zero)")
    fm = rep.factor_model
    tbl = fm.betas.copy()
    tbl["R²"] = fm.r2
    tbl["Idio vol (ann.)"] = np.sqrt(fm.resid_var * 252)
    tbl["Weight"] = rep.weights
    st.dataframe(tbl.style.format("{:.2f}").format({"Weight": "{:.1%}", "Idio vol (ann.)": "{:.1%}"}),
                 width='stretch')
    with st.expander("t-statistics"):
        st.dataframe(fm.tstats.style.format("{:.1f}"), width='stretch')
    st.caption(f"Mean |residual correlation| across names: {fm.residual_correlation_check():.3f}. "
               "Above ~0.15 suggests a missing common factor (e.g. sector).")

# --------------------------------------------------------------------------
# Tail risk
# --------------------------------------------------------------------------
with tabs[1]:
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**1-day loss at {q:.0%}, four ways**")
        st.dataframe(rep.tail.table().style.format("{:.2%}"), width='stretch')
        st.caption(f"Fitted Student-t degrees of freedom: {rep.tail.nu:.1f}. "
                   "Lower means fatter tails; Gaussian is the limit as ν → ∞.")
    with right:
        R = md.returns[list(rep.weights.index)].iloc[-500:]
        pr = portfolio_returns(R, rep.weights)
        _, _, sims_t = monte_carlo_var_cvar(R, rep.weights, q, n_sims=20000, dist="t")
        _, _, sims_n = monte_carlo_var_cvar(R, rep.weights, q, n_sims=20000, dist="normal")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pr, nbinsx=80, histnorm="probability density",
                                   name="Realised", marker_color=INK, opacity=0.55))
        fig.add_trace(go.Histogram(x=sims_n, nbinsx=120, histnorm="probability density",
                                   name="Normal MC", marker_color=MUTED, opacity=0.45))
        fig.add_trace(go.Histogram(x=sims_t, nbinsx=120, histnorm="probability density",
                                   name="Student-t MC", marker_color=LOSS, opacity=0.35))
        fig.add_vline(x=-rep.tail.historical[0], line_dash="dash", line_color=LOSS,
                      annotation_text=f"Hist VaR {q:.0%}")
        fig.update_layout(barmode="overlay", height=380, xaxis_title="daily return",
                          margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
with tabs[2]:
    bt = rep.backtest_hist
    bp = rep.backtest_param
    c1, c2 = st.columns(2)
    c1.markdown(f"**Historical VaR** — {bt.n_breaches} breaches / {bt.n_obs} days "
                f"(expected {bt.expected_breaches:.1f})  \n{bt.verdict}")
    c2.markdown(f"**Parametric VaR** — {bp.n_breaches} breaches / {bp.n_obs} days "
                f"(expected {bp.expected_breaches:.1f})  \n{bp.verdict}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.realised.index, y=bt.realised.values, mode="lines",
                             name="Realised return", line=dict(color=INK, width=1)))
    fig.add_trace(go.Scatter(x=bt.var_series.index, y=-bt.var_series.values, mode="lines",
                             name=f"−VaR {q:.0%} (historical)", line=dict(color=LOSS, width=1.5)))
    fig.add_trace(go.Scatter(x=bp.var_series.index, y=-bp.var_series.values, mode="lines",
                             name=f"−VaR {q:.0%} (parametric)", line=dict(color=MUTED, width=1.5, dash="dot")))
    br = bt.realised[bt.breaches]
    fig.add_trace(go.Scatter(x=br.index, y=br.values, mode="markers", name="Breach",
                             marker=dict(color=LOSS, size=7, symbol="x")))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch')
    st.caption("Out-of-sample: each day's VaR uses only the preceding window. "
               "The Kupiec test asks whether the breach count is statistically consistent with the target rate.")

# --------------------------------------------------------------------------
# Stress
# --------------------------------------------------------------------------
with tabs[3]:
    s = rep.stress.copy()
    fig = go.Figure(go.Bar(
        x=s["Portfolio P&L"], y=s.index, orientation="h",
        marker_color=[LOSS if v < 0 else GAIN for v in s["Portfolio P&L"]],
        error_x=dict(type="data", array=s["Idio ±1σ"], visible=True, color=MUTED),
    ))
    fig.update_layout(height=420, xaxis_tickformat=".0%", xaxis_title="factor-implied P&L (± idiosyncratic 1σ)",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

    fcols = [c for c in s.columns if c.startswith("via ")]
    st.markdown("**Attribution by factor**")
    st.dataframe(s[["Type", "Days", "Portfolio P&L"] + fcols].style
                 .format({c: "{:+.1%}" for c in ["Portfolio P&L"] + fcols}),
                 width='stretch')

    st.markdown("**Build your own shock**")
    cols = st.columns(6)
    custom = {}
    for i, f in enumerate(rep.factor_model.factor_names):
        custom[f] = cols[i].number_input(f, value=0.0, step=0.01, format="%.2f", key=f"shock_{f}")
    res = hypothetical_shock(rep.factor_model, rep.weights, "Custom", custom)
    st.markdown(f"Factor-implied P&L: **{res.total_pnl:+.2%}** "
                f"(idiosyncratic ±1σ over 20 days: {res.idio_band:.2%})")

# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
with tabs[4]:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Risk contribution by position (current book)**")
        rc = rep.risk_contrib
        fig = go.Figure(go.Bar(x=rc.index, y=rc.values, marker_color=INK, name="Risk share"))
        fig.add_trace(go.Bar(x=rc.index, y=rep.weights.reindex(rc.index).values,
                             marker_color=MUTED, name="Weight"))
        fig.update_layout(barmode="group", height=340, yaxis_tickformat=".0%",
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width='stretch')
        st.caption("When risk share exceeds weight, the position is punching above its size in the risk budget.")
    with right:
        st.markdown("**Alternative allocations on the same names**")
        st.dataframe(rep.alloc_stats.style.format({
            "Ann. vol (shrunk cov)": "{:.1%}", "Ann. vol (realised)": "{:.1%}",
            "Eff. # of bets": "{:.1f}", "Max weight": "{:.1%}", "Top-5 weight": "{:.1%}"}),
            width='stretch')
        st.caption(f"Covariance shrunk with Ledoit-Wolf, intensity δ = "
                   f"{rep.alloc_stats.attrs.get('shrinkage_intensity', float('nan')):.2f}.")

    st.markdown("**Weights**")
    st.dataframe(rep.alloc_weights.style.format("{:.1%}"), width='stretch')

# --------------------------------------------------------------------------
# Holdings
# --------------------------------------------------------------------------
with tabs[5]:
    px_ = md.prices[list(rep.weights.index)]
    norm = px_ / px_.iloc[0]
    fig = px.line(norm, color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(height=400, yaxis_title="growth of $1", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')
    st.dataframe(pd.DataFrame({"Weight": rep.weights,
                               "Ann. vol": md.returns[list(rep.weights.index)].std() * np.sqrt(252),
                               "Idio share of total var": rep.idio_by_stock})
                 .style.format("{:.1%}"), width='stretch')

st.download_button("Download text report", rep.to_text(), "risk_report.txt")
