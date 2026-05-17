"""
engine.py — Pure computation layer for Market Sentinel.

All functions are side-effect-free: they accept data and return typed dicts
or DataFrames. No print(), no tabulate(), no Alpaca API calls.
This layer is consumed by state.py (background cache) and dashboard.py (UI).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .analysis import run_sector_report, suggest_stocks

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


# ---------------------------------------------------------------------------
# Indicator computations
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return val if val == val else 50.0  # NaN -> neutral


def _score_signal_snapshot(snap: dict[str, Any]) -> int:
    score = 0
    if snap["trend_ok"]:
        score += 2
    if snap["breakout55"]:
        score += 2
    if isinstance(snap["relvol"], (int, float)) and snap["relvol"] >= 1.3:
        score += 1
    if snap["macd"] > snap["macd_signal"]:
        score += 1
    if snap["last"] > snap["sma50"]:
        score += 1
    if 40 <= snap["rsi"] <= 70:
        score += 1
    if snap["dist10"] < 3:
        score += 1
    if not snap["breakdown20"]:
        score += 1
    if snap["last"] > snap["sma200"]:
        score += 1
    return score


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, float]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": float(macd_line.iloc[-1]),
        "signal": float(signal_line.iloc[-1]),
        "histogram": float(histogram.iloc[-1]),
        "macd_series": macd_line.tolist(),
        "signal_series": signal_line.tolist(),
        "histogram_series": histogram.tolist(),
    }


def compute_bollinger(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> dict[str, Any]:
    sma = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper = sma + (rolling_std * std_dev)
    lower = sma - (rolling_std * std_dev)
    last = float(close.iloc[-1])
    mid_val = float(sma.iloc[-1])
    upper_val = float(upper.iloc[-1])
    lower_val = float(lower.iloc[-1])
    band_width = upper_val - lower_val
    pct_b = ((last - lower_val) / band_width) if band_width > 0 else 0.5
    return {
        "upper": upper_val,
        "mid": mid_val,
        "lower": lower_val,
        "pct_b": round(pct_b, 3),
        "upper_series": upper.tolist(),
        "mid_series": sma.tolist(),
        "lower_series": lower.tolist(),
    }


def compute_atr(bars: pd.DataFrame, period: int = 14) -> float:
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.rolling(window=period).mean().iloc[-1])


def compute_breakout55(bars: pd.DataFrame) -> dict[str, Any]:
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    volume = bars["volume"].astype(float)

    last_close = float(close.iloc[-1])
    prior_55_high = float(high.shift(1).rolling(55).max().iloc[-1])
    breakout55 = bool(last_close > prior_55_high) if prior_55_high == prior_55_high else False

    avg_vol20 = float(volume.shift(1).rolling(20).mean().iloc[-1])
    relvol = round(float(volume.iloc[-1] / avg_vol20), 2) if avg_vol20 > 0 else None

    avg_vol20_series = volume.shift(1).rolling(20).mean()
    prev_close = close.shift(1)
    dist_flag = (close < prev_close) & (volume > avg_vol20_series)
    dist10 = int(dist_flag.tail(10).sum())

    prior_20_low = float(low.shift(1).rolling(20).min().iloc[-1])
    breakdown20 = bool(last_close < prior_20_low) if prior_20_low == prior_20_low else False

    return {
        "breakout55": breakout55,
        "relvol": relvol,
        "dist10": dist10,
        "breakdown20": breakdown20,
    }


# ---------------------------------------------------------------------------
# Bar fetching helpers
# ---------------------------------------------------------------------------

def _normalize_columns(hist: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flatten any MultiIndex columns and return a DataFrame with lowercase OHLCV columns."""
    if isinstance(hist.columns, pd.MultiIndex):
        # MultiIndex: (field, ticker) — take the field name (level 0) and lowercase it
        flat = []
        for col in hist.columns:
            parts = [p for p in col if p is not None and str(p).strip()]
            flat.append(str(parts[0]).lower() if parts else "")
        hist.columns = flat
        # Drop any columns with empty names
        hist = hist.loc[:, hist.columns != ""]
    else:
        hist.columns = [str(c).lower() for c in hist.columns]

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in hist.columns]
    if missing:
        raise RuntimeError(f"Missing OHLCV columns for {symbol}: {missing}")

    df = pd.DataFrame({
        "open":   pd.to_numeric(hist["open"],   errors="coerce"),
        "high":   pd.to_numeric(hist["high"],   errors="coerce"),
        "low":    pd.to_numeric(hist["low"],    errors="coerce"),
        "close":  pd.to_numeric(hist["close"],  errors="coerce"),
        "volume": pd.to_numeric(hist["volume"], errors="coerce"),
    }).dropna()
    df.index = pd.to_datetime(df.index)
    return df


def fetch_bars_yfinance(symbol: str, lookback_days: int = 420) -> pd.DataFrame:
    """Fetch daily OHLCV bars using yf.Ticker().history() — avoids MultiIndex NoneType bugs."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=f"{lookback_days}d", interval="1d", auto_adjust=False)

    if hist is None or hist.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    df = _normalize_columns(hist, symbol)

    if df.empty:
        raise RuntimeError(f"All rows were NaN after normalising {symbol}")
    if len(df) < 30:
        raise RuntimeError(f"Insufficient history for {symbol}: only {len(df)} bars")

    return df


# ---------------------------------------------------------------------------
# Full signal snapshot for one symbol
# ---------------------------------------------------------------------------

def compute_signal_snapshot(symbol: str, bars: pd.DataFrame, atr_mult: float = 2.5) -> dict[str, Any]:
    """Return a complete signal dict for a single symbol. Pure — no I/O."""
    close = bars["close"].astype(float)
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    last = float(close.iloc[-1])

    rsi = compute_rsi(close)
    macd = compute_macd(close)
    bb = compute_bollinger(close)
    atr_val = compute_atr(bars)
    b55 = compute_breakout55(bars)
    trend_ok = bool(last > sma200 and sma50 >= sma200)

    signal_score = _score_signal_snapshot({
        "last": last,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": round(rsi, 2),
        "macd": round(macd["macd"], 4),
        "macd_signal": round(macd["signal"], 4),
        "bb_pct_b": round(bb["pct_b"], 3),
        "atr": round(atr_val, 2),
        "breakout55": b55["breakout55"],
        "relvol": b55["relvol"],
        "dist10": b55["dist10"],
        "breakdown20": b55["breakdown20"],
        "trend_ok": trend_ok,
    })

    return {
        "symbol": symbol.upper(),
        "last": round(last, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "rsi": round(rsi, 2),
        "macd": round(macd["macd"], 4),
        "macd_signal": round(macd["signal"], 4),
        "macd_hist": round(macd["histogram"], 4),
        "bb_upper": round(bb["upper"], 2),
        "bb_mid": round(bb["mid"], 2),
        "bb_lower": round(bb["lower"], 2),
        "bb_pct_b": round(bb["pct_b"], 3),
        "atr": round(atr_val, 2),
        "breakout55": b55["breakout55"],
        "relvol": b55["relvol"],
        "dist10": b55["dist10"],
        "breakdown20": b55["breakdown20"],
        "trend_ok": trend_ok,
        "signal_score": signal_score,
        # chart series (full history for plotting)
        "dates": [str(d.date()) for d in bars.index],
        "open_series": bars["open"].astype(float).tolist(),
        "close_series": close.tolist(),
        "high_series": bars["high"].astype(float).tolist(),
        "low_series": bars["low"].astype(float).tolist(),
        "volume_series": bars["volume"].astype(float).tolist(),
        "bb_upper_series": bb["upper_series"],
        "bb_mid_series": bb["mid_series"],
        "bb_lower_series": bb["lower_series"],
        "macd_series": macd["macd_series"],
        "macd_signal_series": macd["signal_series"],
        "macd_hist_series": macd["histogram_series"],
    }


# ---------------------------------------------------------------------------
# Trade plan row logic  (pure — accepts pre-computed snapshot)
# ---------------------------------------------------------------------------

def decide_action(
    snap: dict[str, Any],
    entry_style: str,
    qty: float,
    days_held: int | None,
    atr_mult: float,
) -> tuple[str, str]:
    """Return (action, reason) for a symbol given its signal snapshot and position."""
    entry_style = (entry_style or "breakout55").strip().lower()
    last = snap["last"]
    atr_val = snap["atr"]
    market_gate = snap.get("market_gate", True)
    macd_bull = snap["macd"] > snap["macd_signal"]

    def _hybrid_buy_setup() -> tuple[bool, str]:
        relvol = snap["relvol"]
        relvol_ok = isinstance(relvol, (int, float)) and relvol >= 1.1
        rsi_ok = 40 <= snap["rsi"] <= 72
        dist_ok = snap["dist10"] < 4
        near_sma50 = abs(last - snap["sma50"]) <= max(atr_val * 0.75, 0.01)
        gentle_pullback = (
            snap["bb_pct_b"] <= 0.55
            and last <= snap["bb_mid"]
            and last >= (snap["sma50"] - atr_val)
        )

        breakout_ready = (
            snap["breakout55"]
            and relvol_ok
            and snap["dist10"] < 3
            and macd_bull
            and snap["rsi"] <= 78
        )
        if breakout_ready:
            return True, (
                f"Hybrid breakout confirmed (RelVol {relvol:.2f}, "
                f"Dist10={snap['dist10']})"
            )

        pullback_ready = (
            near_sma50
            and gentle_pullback
            and macd_bull
            and rsi_ok
            and dist_ok
        )
        if pullback_ready:
            return True, (
                f"Hybrid pullback entry near SMA50 ({snap['sma50']:.2f}) "
                f"with bullish MACD"
            )

        if not snap["trend_ok"]:
            return False, "Hybrid waiting for trend alignment"
        if not macd_bull:
            return False, "Hybrid waiting for MACD confirmation"
        if snap["breakout55"] and not relvol_ok:
            return False, "Hybrid breakout lacks volume confirmation"
        if not dist_ok:
            return False, f"Hybrid waiting for cleaner tape (Dist10={snap['dist10']})"
        return False, "Hybrid waiting for breakout or pullback entry"

    # ATR trailing stop (universal across all styles)
    atr_trailing_stop = 0.0
    if qty > 0 and days_held is not None and days_held > 0:
        high_series = pd.Series(snap["high_series"])
        recent_high = float(high_series.iloc[-days_held:].max())
        atr_trailing_stop = recent_high - (atr_mult * atr_val)

    if qty > 0:
        if last < atr_trailing_stop and atr_trailing_stop > 0:
            return "SELL", f"ATR trailing stop hit ({last:.2f} < {atr_trailing_stop:.2f})"
        if entry_style == "hybrid":
            if snap["breakdown20"] and last < snap["sma50"]:
                return "SELL", "Hybrid trend failed (20-day breakdown below SMA50)"
            if (not macd_bull) and last < snap["sma50"] and snap["dist10"] >= 4:
                return "SELL", f"Hybrid momentum failed (Dist10={snap['dist10']})"
        if entry_style == "macd" and snap["macd"] < snap["macd_signal"]:
            return "SELL", "MACD crossed below signal"
        if entry_style == "bollinger" and last >= snap["bb_mid"]:
            return "SELL", f"Bollinger mid reversion ({snap['bb_mid']:.2f})"
        if entry_style == "breakout55" and snap["breakdown20"]:
            rv = snap["relvol"]
            if isinstance(rv, (int, float)) and rv >= 1.3:
                return "SELL", f"Breakdown20 high vol (RelVol {rv:.2f})"
        return "HOLD", f"In position. Stop: {atr_trailing_stop:.2f}"

    # Entry logic
    if market_gate is None:
        return "HOLD", "Market gate unknown (SPY data unavailable)"
    if not market_gate:
        return "HOLD", "Market gate off (SPY < SMA200)"
    if entry_style in ("breakout55", "macd", "hybrid") and not snap["trend_ok"]:
        return "HOLD", "Trend not OK (Close < SMA200 or SMA50 < SMA200)"

    if entry_style == "hybrid":
        is_buy, reason = _hybrid_buy_setup()
        return ("BUY", reason) if is_buy else ("HOLD", reason)

    if entry_style == "breakout55":
        rv = snap["relvol"]
        relvol_ok = isinstance(rv, (int, float)) and rv >= 1.3
        if snap["breakout55"] and relvol_ok and snap["dist10"] < 3:
            return "BUY", f"Breakout55 confirmed (RelVol {rv:.2f}, Dist10={snap['dist10']})"
        return "HOLD", f"Waiting breakout55 (Dist10={snap['dist10']})"

    if entry_style == "macd":
        if snap["macd"] > snap["macd_signal"] and last > snap["sma50"]:
            return "BUY", "MACD > Signal and Close > 50-SMA"
        return "HOLD", "Waiting MACD crossover"

    if entry_style == "bollinger":
        # require close below lower BB AND bullish candle
        if snap.get("close_series") and snap.get("open_series"):
            last_close = snap["close_series"][-1]
            last_open = snap["open_series"][-1]
        else:
            last_close, last_open = last, last
        if last < snap["bb_lower"] and last_close > last_open:
            return "BUY", "Close below lower BB + bullish candle"
        return "HOLD", f"Waiting to cross below lower BB ({snap['bb_lower']:.2f})"

    # rsi30 legacy
    if snap["rsi"] < 30:
        return "BUY", f"RSI {snap['rsi']:.1f} < 30"
    return "HOLD", f"RSI {snap['rsi']:.1f} ≥ 30"


# ---------------------------------------------------------------------------
# Sector report
# ---------------------------------------------------------------------------

def get_sector_report() -> pd.DataFrame:
    """Return sector performance DataFrame sorted by YTD return."""
    strong, weak, _ = run_sector_report(top_strong=11, top_weak=0)
    return strong


def get_spy_above_sma200() -> bool | None:
    try:
        bars = fetch_bars_yfinance("SPY", lookback_days=420)
        close = bars["close"].astype(float)
        sma200 = float(close.rolling(200).mean().iloc[-1])
        return bool(float(close.iloc[-1]) > sma200)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Trade log reader
# ---------------------------------------------------------------------------

def load_trade_logs(logs_dir: str) -> pd.DataFrame:
    """Read all trade_plan_*.json files and return a consolidated DataFrame."""
    import json
    import glob
    rows = []
    for path in sorted(glob.glob(f"{logs_dir}/trade_plan_*.json")):
        try:
            data = json.loads(open(path).read())
            date_str = data.get("META", {}).get("GENERATED AT UTC", path)[:10]
            style = data.get("META", {}).get("ENTRY STYLE", "-")
            for row in data.get("ROWS", []):
                rows.append({
                    "date": date_str,
                    "style": style,
                    "symbol": row.get("SYMBOL", "-"),
                    "action": row.get("ACTION", "-"),
                    "rsi": row.get("RSI", None),
                    "reason": row.get("REASON", "-"),
                })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["date", "style", "symbol", "action", "rsi", "reason"])
    return pd.DataFrame(rows)
