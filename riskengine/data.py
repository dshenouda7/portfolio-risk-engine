"""
Data layer.

Pulls daily adjusted prices (yfinance) and Fama-French factor returns
(Ken French data library), aligns them, and caches everything to parquet
so repeated runs are instant and offline.

Also exposes `synthetic_market()` which generates a realistic fake dataset
from a factor model. Useful for tests, demos, and building without a
network connection.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


@dataclass
class MarketData:
    """Aligned daily data. All returns are simple daily returns (decimals)."""

    prices: pd.DataFrame        # adjusted close, columns = tickers
    returns: pd.DataFrame       # simple returns, columns = tickers
    factors: pd.DataFrame       # factor returns, columns = FACTOR_NAMES
    rf: pd.Series               # daily risk-free rate
    sectors: pd.Series | None = None   # ticker -> sector label (optional)
    adv: pd.Series | None = None       # ticker -> avg daily dollar volume (optional)

    @property
    def excess_returns(self) -> pd.DataFrame:
        return self.returns.sub(self.rf, axis=0)

    @property
    def tickers(self) -> list[str]:
        return list(self.returns.columns)


# --------------------------------------------------------------------------
# Live data
# --------------------------------------------------------------------------

def _cache_key(tickers: list[str], start: str, end: str | None, kind: str) -> Path:
    import hashlib
    h = hashlib.md5("_".join(sorted(tickers)).encode()).hexdigest()[:10]
    return CACHE_DIR / f"{kind}_{h}_{start}_{end}.parquet"


def fetch_prices(tickers: list[str], start: str, end: str | None = None,
                 use_cache: bool = True) -> pd.DataFrame:
    """Adjusted close prices from yfinance, cached to parquet."""
    return _fetch_yf(tickers, start, end, use_cache)[0]


def _fetch_yf(tickers: list[str], start: str, end: str | None = None,
              use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(adjusted close, dollar volume) from yfinance, both cached."""
    c_px = _cache_key(tickers, start, end, "prices")
    c_dv = _cache_key(tickers, start, end, "dollarvol")
    if use_cache and c_px.exists() and c_dv.exists():
        return pd.read_parquet(c_px), pd.read_parquet(c_dv)

    import yfinance as yf  # imported lazily so tests don't need it

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column")
    if isinstance(raw.columns, pd.MultiIndex):
        px, vol = raw["Close"], raw["Volume"]
    else:  # single ticker
        px = raw[["Close"]].rename(columns={"Close": tickers[0]})
        vol = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    px = px.dropna(how="all").ffill()
    dv = (px * vol.reindex(px.index)).ffill()
    for df in (px, dv):
        df.index = pd.to_datetime(df.index).tz_localize(None)
    px.to_parquet(c_px)
    dv.to_parquet(c_dv)
    return px, dv


def fetch_sectors(tickers: list[str], use_cache: bool = True) -> pd.Series:
    """Sector labels from yfinance's `info`; slow-ish (one call per ticker), so cached."""
    cache = CACHE_DIR / "sectors.parquet"
    known = pd.read_parquet(cache)["sector"] if (use_cache and cache.exists()) else pd.Series(dtype=object)
    missing = [t for t in tickers if t not in known.index]
    if missing:
        import yfinance as yf
        new = {}
        for t in missing:
            try:
                new[t] = yf.Ticker(t).info.get("sector", "Unknown") or "Unknown"
            except Exception:
                new[t] = "Unknown"
        known = pd.concat([known, pd.Series(new)])
        known.to_frame("sector").to_parquet(cache)
    return known.reindex(tickers)


def _parse_french_csv(text: str, ncols: int) -> pd.DataFrame:
    """Ken French CSVs have header junk and a trailing annual block; keep daily rows."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip()[:8].isdigit())
    rows = []
    for l in lines[start:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) < ncols + 1 or not parts[0].isdigit() or len(parts[0]) != 8:
            break
        rows.append(parts[: ncols + 1])
    df = pd.DataFrame(rows)
    df[0] = pd.to_datetime(df[0], format="%Y%m%d")
    df = df.set_index(0).astype(float) / 100.0  # percent -> decimal
    return df


def fetch_factors(use_cache: bool = True) -> pd.DataFrame:
    """Fama-French 5 factors + momentum + RF, daily, decimals."""
    cache = CACHE_DIR / "ff5_mom_daily.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    import requests

    def _get(url, ncols, names):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        text = z.read(z.namelist()[0]).decode("latin1")
        df = _parse_french_csv(text, ncols)
        df.columns = names
        return df

    ff5 = _get(FF5_URL, 6, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    mom = _get(MOM_URL, 1, ["Mom"])
    out = ff5.join(mom, how="inner")
    out.to_parquet(cache)
    return out


def load_market(tickers: list[str], start: str = "2018-01-01",
                end: str | None = None, sectors: pd.Series | None = None,
                fetch_sector_info: bool = True) -> MarketData:
    """
    Fetch, align, and package prices + factors + (optionally) sectors and
    average daily dollar volume. If `sectors` is None and fetch_sector_info
    is True, sectors are pulled from yfinance.
    """
    px, dv = _fetch_yf(tickers, start, end)
    rets = px.pct_change().dropna(how="all")
    ff = fetch_factors()
    idx = rets.index.intersection(ff.index)
    rets = rets.loc[idx].dropna(axis=1, how="all")
    ff = ff.loc[idx]
    adv = dv.loc[idx].iloc[-60:].mean()  # 3-month average daily dollar volume
    if sectors is None and fetch_sector_info:
        try:
            sectors = fetch_sectors(list(rets.columns))
        except Exception:
            sectors = None
    return MarketData(
        prices=px.loc[px.index >= idx[0]],
        returns=rets.fillna(0.0),
        factors=ff[FACTOR_NAMES],
        rf=ff["RF"],
        sectors=sectors,
        adv=adv,
    )


# --------------------------------------------------------------------------
# Synthetic data (offline / testing)
# --------------------------------------------------------------------------

def synthetic_market(n_assets: int = 12, n_days: int = 1500, seed: int = 7,
                     tickers: list[str] | None = None) -> MarketData:
    """
    Generate a small/mid-cap-flavoured market from a 6-factor model with
    Student-t idiosyncratic noise (fat tails) and one embedded crash window.
    Everything is calibrated to plausible daily magnitudes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    # Factor returns: daily vols roughly matching FF history
    f_mean = np.array([0.0004, 0.0001, 0.0000, 0.0001, 0.0000, 0.0002])
    f_vol = np.array([0.011, 0.006, 0.006, 0.005, 0.004, 0.008])
    F = rng.standard_t(df=5, size=(n_days, 6)) / np.sqrt(5 / 3) * f_vol + f_mean

    # Embed a crash: 25 days of strongly negative market, small caps hit harder
    crash = slice(n_days // 3, n_days // 3 + 25)
    F[crash, 0] -= 0.012
    F[crash, 1] -= 0.004

    factors = pd.DataFrame(F, index=dates, columns=FACTOR_NAMES)
    rf = pd.Series(0.00015, index=dates, name="RF")

    if tickers is None:
        tickers = [f"SMID{i:02d}" for i in range(n_assets)]
    n = len(tickers)

    # Betas: high market beta, positive SMB (small caps), mixed value/quality
    betas = np.column_stack([
        rng.uniform(0.9, 1.4, n),      # Mkt
        rng.uniform(0.3, 1.0, n),      # SMB
        rng.normal(0.0, 0.4, n),       # HML
        rng.normal(0.1, 0.3, n),       # RMW
        rng.normal(0.0, 0.3, n),       # CMA
        rng.normal(0.0, 0.3, n),       # Mom
    ])
    idio_vol = rng.uniform(0.012, 0.025, n)
    eps = rng.standard_t(df=4, size=(n_days, n)) / np.sqrt(4 / 2) * idio_vol
    alpha = rng.normal(0.0, 0.0001, n)

    # Hidden sector factor: 4 sectors with their own common shock that the
    # FF factors don't capture. This is what a sector factor should find.
    sector_labels = ["Industrials", "Healthcare", "Technology", "Consumer"]
    sec_idx = rng.integers(0, 4, n)
    sec_shocks = rng.standard_t(df=6, size=(n_days, 4)) / np.sqrt(6 / 4) * 0.010
    sec_load = rng.uniform(0.6, 1.2, n)
    eps = eps + sec_shocks[:, sec_idx] * sec_load

    excess = F @ betas.T + eps + alpha
    rets = pd.DataFrame(excess + rf.values[:, None], index=dates, columns=tickers)
    prices = 50 * (1 + rets).cumprod()

    sectors = pd.Series([sector_labels[i] for i in sec_idx], index=tickers, name="sector")
    # Dollar ADV spanning micro to mid cap: $1m to $80m/day
    adv = pd.Series(10 ** rng.uniform(5.3, 7.5, n), index=tickers, name="adv")

    md = MarketData(prices=prices, returns=rets, factors=factors, rf=rf,
                    sectors=sectors, adv=adv)
    md.true_betas = pd.DataFrame(betas, index=tickers, columns=FACTOR_NAMES)  # type: ignore[attr-defined]
    return md


# --------------------------------------------------------------------------
# Portfolio file
# --------------------------------------------------------------------------

def load_portfolio(path, with_sectors: bool = False):
    """
    CSV with columns `ticker, weight` and optionally `sector`.
    Weights may be decimals or percents; they are normalised to sum to 1.
    Returns weights (and sectors if with_sectors=True and the column exists).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tick = df["ticker"].str.upper().str.strip()
    w = pd.Series(df["weight"].astype(float).values, index=tick)
    if w.sum() > 1.5:  # percent
        w = w / 100.0
    w = w / w.sum()
    if with_sectors:
        sec = pd.Series(df["sector"].values, index=tick) if "sector" in df.columns else None
        return w, sec
    return w
