from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

SECTOR_ETFS = {
    "Basic Materials": "XLB",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Healthcare": "XLV",
    "Utilities": "XLU",
    "Communication Services": "XLC",
    "Real Estate": "XLRE",
    "Consumer Cyclical": "XLY",
    "Financial Services": "XLF",
    "Technology": "XLK",
}

WIKI_SECTOR_MAP = {
    "Materials": "Basic Materials",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Consumer Staples": "Consumer Defensive",
    "Health Care": "Healthcare",
    "Utilities": "Utilities",
    "Communication Services": "Communication Services",
    "Real Estate": "Real Estate",
    "Consumer Discretionary": "Consumer Cyclical",
    "Financials": "Financial Services",
    "Information Technology": "Technology",
}

SECTOR_COLUMN_CANDIDATES = ("Sector", "GICS Sector", "GICS_Sector")


@dataclass
class SectorSignal:
    sector: str
    etf: str
    ytd_return_pct: float


def _ytd_start(today: date) -> date:
    return date(today.year, 1, 1)


def _safe_history(symbol: str, **kwargs) -> pd.DataFrame | None:
    """Fetch history for a single symbol safely; returns None on any failure."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(**kwargs)
        if hist is None or hist.empty:
            return None
        # Flatten MultiIndex if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [str(c[0]).lower() for c in hist.columns]
        else:
            hist.columns = [str(c).lower() for c in hist.columns]
        return hist
    except Exception:
        return None


def sector_performance(today: date | None = None) -> pd.DataFrame:
    as_of = today or date.today()
    start = _ytd_start(as_of).isoformat()

    rows = []
    for sector, etf in SECTOR_ETFS.items():
        hist = _safe_history(etf, start=start, end=as_of.isoformat(), auto_adjust=True)
        if hist is None:
            continue
        # Accept both 'close' and 'Close'
        close_col = next((c for c in hist.columns if c.lower() == "close"), None)
        if close_col is None:
            continue
        series = pd.to_numeric(hist[close_col], errors="coerce").dropna()
        if series.empty:
            continue
        ytd = (series.iloc[-1] / series.iloc[0] - 1.0) * 100
        rows.append({"sector": sector, "etf": etf, "ytd_return_pct": float(ytd)})

    if not rows:
        return pd.DataFrame(columns=["sector", "etf", "ytd_return_pct"])
    return pd.DataFrame(rows).sort_values("ytd_return_pct", ascending=False, ignore_index=True)


def _sp500_universe() -> pd.DataFrame:
    csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    response = requests.get(csv_url, timeout=20)
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    sector_col = next((col for col in SECTOR_COLUMN_CANDIDATES if col in raw.columns), None)
    if sector_col is None or "Symbol" not in raw.columns:
        raise ValueError(f"Unexpected S&P 500 CSV schema. Found columns: {list(raw.columns)}")

    df = raw[["Symbol", sector_col]].rename(columns={"Symbol": "symbol", sector_col: "sector"})
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    df["sector"] = df["sector"].map(WIKI_SECTOR_MAP)
    return df.dropna().reset_index(drop=True)


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def suggest_stocks(strong_sectors: Iterable[str], top_n: int = 8) -> pd.DataFrame:
    universe = _sp500_universe()
    candidates = universe[universe["sector"].isin(set(strong_sectors))].copy()

    if candidates.empty:
        return pd.DataFrame(columns=["symbol", "sector", "return_3m_pct", "avg_dollar_vol_20d_m", "score"])

    symbols = candidates["symbol"].tolist()

    records = []
    for symbol in symbols:
        try:
            hist = _safe_history(symbol, period="4mo", interval="1d", auto_adjust=True)
            if hist is None or len(hist) < 45:
                continue
            close_col = next((c for c in hist.columns if c.lower() == "close"), None)
            vol_col = next((c for c in hist.columns if c.lower() == "volume"), None)
            if close_col is None or vol_col is None:
                continue
            price = pd.to_numeric(hist[close_col], errors="coerce").dropna()
            vol = pd.to_numeric(hist[vol_col], errors="coerce").dropna()
            if len(price) < 20:
                continue
            ret_3m = (price.iloc[-1] / price.iloc[0] - 1.0) * 100
            avg_dollar_vol = (price.tail(20) * vol.tail(20)).mean() / 1_000_000
            records.append({
                "symbol": symbol,
                "return_3m_pct": float(ret_3m),
                "avg_dollar_vol_20d_m": float(avg_dollar_vol),
            })
        except Exception:
            continue

    scored = pd.DataFrame(records)
    if scored.empty:
        return pd.DataFrame(columns=["symbol", "sector", "return_3m_pct", "avg_dollar_vol_20d_m", "score"])

    scored["score"] = _zscore(scored["return_3m_pct"]) + _zscore(np.log1p(scored["avg_dollar_vol_20d_m"]))
    out = scored.merge(candidates, on="symbol", how="left")
    out = out[["symbol", "sector", "return_3m_pct", "avg_dollar_vol_20d_m", "score"]]
    out = out.sort_values("score", ascending=False, ignore_index=True)
    return out.head(top_n)


def run_sector_report(top_strong: int = 3, top_weak: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf = sector_performance()
    if perf.empty:
        empty = pd.DataFrame(columns=["sector", "etf", "ytd_return_pct"])
        return empty, empty.copy(), pd.DataFrame(columns=["symbol", "sector", "return_3m_pct", "avg_dollar_vol_20d_m", "score"])
    strong = perf.head(top_strong).copy()
    weak = perf.tail(top_weak).sort_values("ytd_return_pct", ascending=True, ignore_index=True)
    picks = suggest_stocks(strong["sector"].tolist())
    return strong, weak, picks
