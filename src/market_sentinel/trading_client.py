from __future__ import annotations

import argparse
import json
import os
import select
import smtplib
import sys
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from zoneinfo import ZoneInfo

import pandas as pd
from tabulate import tabulate
import yfinance as yf

from .analysis import run_sector_report, suggest_stocks
from .engine import compute_signal_snapshot, decide_action

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"

DEFAULT_MAX_SYMBOLS = 8
AUTO_DEFAULT_MAX_SYMBOLS = 20
AUTO_DEFAULT_REFRESH_MINUTES = 60
AUTO_TRIGGER_MINUTES_AFTER_OPEN = 30
MARKET_TZ = ZoneInfo("America/New_York")
DEFAULT_ENTRY_STYLE = os.getenv("ENTRY_STYLE", "breakout55").strip().lower()

HUMAN_LABELS = {
    "symbol": "SYMBOL",
    "rsi": "RSI",
    "position_qty": "POSITION QTY",
    "days_held": "DAYS HELD",
    "action": "ACTION",
    "reason": "REASON",
    "timestamp_utc": "TIMESTAMP UTC",
    "is_open": "IS OPEN",
    "next_open": "NEXT OPEN",
    "next_close": "NEXT CLOSE",
    "qty": "QTY",
    "avg_entry": "AVG ENTRY",
    "market_value": "MARKET VALUE",
    "u_pl": "UNREALIZED P/L",
    "u_pl_pct": "UNREALIZED P/L %",
    "market_day_et": "MARKET DAY (ET)",
    "trigger_time_et": "TRIGGER TIME (ET)",
    "triggered_today": "TRIGGERED TODAY",
    "next_action": "NEXT ACTION",
    "triggered_market_day": "TRIGGERED MARKET DAY",
    "triggered_at_utc": "TRIGGERED AT UTC",
    "plan_file": "PLAN FILE",
    "buy": "BUY",
    "sell": "SELL",
    "hold": "HOLD",
    "generated_at_utc": "GENERATED AT UTC",
    "max_symbols": "MAX SYMBOLS",
    "plan_size": "PLAN SIZE",
    "rating": "RATING",
    "buy_zone": "BUY ZONE",
    "timing": "TIMING",
    "signal_score": "SIGNAL SCORE",
    "data_source": "DATA SOURCE",
    "entry_style": "ENTRY STYLE",
    "relvol": "REL VOL",
    "breakout55": "BREAKOUT55",
    "dist10": "DIST10",
    "trend_ok": "TREND OK",
    "risk_ok": "RISK OK",
    "market_gate": "MARKET GATE",
    "strategy": "STRATEGY",
    "direction": "DIRECTION",
    "score": "SCORE",
    "source": "SOURCE",
    "options_plan_file": "OPTIONS PLAN FILE",
}


def _alpaca_imports() -> dict[str, Any]:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        try:
            from alpaca.data.enums import DataFeed
        except Exception:
            DataFeed = None
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
        from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'alpaca-py'. Install with: pip install alpaca-py"
        ) from exc
    return {
        "StockHistoricalDataClient": StockHistoricalDataClient,
        "DataFeed": DataFeed,
        "StockBarsRequest": StockBarsRequest,
        "TimeFrame": TimeFrame,
        "TradingClient": TradingClient,
        "OrderSide": OrderSide,
        "QueryOrderStatus": QueryOrderStatus,
        "TimeInForce": TimeInForce,
        "GetOrdersRequest": GetOrdersRequest,
        "MarketOrderRequest": MarketOrderRequest,
    }


def _build_clients() -> tuple[dict[str, Any], Any, Any]:
    if load_dotenv is None:
        raise RuntimeError("Missing dependency 'python-dotenv'. Install with: pip install python-dotenv")

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
    api_key = os.getenv("ALPACA_KEY")
    api_secret = os.getenv("ALPACA_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_KEY/ALPACA_SECRET must be set in src/market_sentinel/.env")

    is_paper = os.getenv("ALPACA_PAPER", "true").lower() in {"1", "true", "yes"}
    a = _alpaca_imports()
    trading_client = a["TradingClient"](api_key, api_secret, paper=is_paper)
    data_client = a["StockHistoricalDataClient"](api_key, api_secret)
    return a, trading_client, data_client


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _notify_successful_transaction(symbol: str, side: str, qty: Any, order_id: Any = None) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    body = (
        f"Successful transaction\n"
        f"Symbol: {symbol}\n"
        f"Side: {side}\n"
        f"Quantity: {qty}\n"
        f"Order ID: {order_id}\n"
        f"Timestamp UTC: {timestamp}\n"
    )
    subject = f"Trade Success: {side} {symbol} ({qty})"

    # Email notification (optional).
    if _env_bool("NOTIFY_EMAIL_ENABLED", default=False):
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")
        from_email = os.getenv("NOTIFY_FROM_EMAIL", smtp_username or "")
        to_email = os.getenv("NOTIFY_TO_EMAIL")
        use_tls = _env_bool("SMTP_USE_TLS", default=True)
        try:
            if not smtp_host or not smtp_username or not smtp_password or not from_email or not to_email:
                print("Notification warning: email is enabled but SMTP/recipient env vars are incomplete.")
            else:
                msg = EmailMessage()
                msg["Subject"] = subject
                msg["From"] = from_email
                msg["To"] = to_email
                msg.set_content(body)
                with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
                    if use_tls:
                        server.starttls()
                    server.login(smtp_username, smtp_password)
                    server.send_message(msg)
        except Exception as exc:
            print(f"Notification warning: email send failed: {exc}")

    # Webhook notification (optional).
    webhook_url = os.getenv("NOTIFY_WEBHOOK_URL", "").strip()
    if webhook_url:
        payload = {
            "text": subject,
            "symbol": symbol,
            "side": side,
            "qty": str(qty),
            "order_id": str(order_id),
            "timestamp_utc": timestamp,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=20):
                pass
        except (urllib_error.URLError, urllib_error.HTTPError, ValueError) as exc:
            print(f"Notification warning: webhook send failed: {exc}")


def get_rsi(symbol: str, data_client: Any, stock_bars_request: Any, timeframe_day: Any) -> float:
    """Calculates a simple 14-day RSI for a given symbol."""
    req = stock_bars_request(
        symbol_or_symbols=[symbol],
        timeframe=timeframe_day,
        start=datetime.now(timezone.utc) - timedelta(days=40),
    )
    bars = data_client.get_stock_bars(req).df
    if bars.empty:
        raise RuntimeError(f"No bars returned for symbol {symbol}")

    if "close" not in bars.columns:
        raise RuntimeError(f"Unexpected bars schema for {symbol}: columns={list(bars.columns)}")

    closes = bars["close"].astype(float)
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    value = float(rsi.iloc[-1])
    if value != value:  # NaN check
        raise RuntimeError(f"Could not compute RSI for {symbol}; insufficient data.")
    return value


def _latest_entry_age_days(
    trading_client: Any,
    get_orders_request: Any,
    query_order_status: Any,
    order_side: Any,
    symbol: str,
) -> int | None:
    # alpaca-py versions differ: some don't have QueryOrderStatus.FILLED.
    # Use CLOSED and filter to the most recent filled BUY as a proxy for "entry".
    status = getattr(query_order_status, "FILLED", None) or getattr(query_order_status, "CLOSED", None)
    order_params = get_orders_request(status=status, symbols=[symbol], side=order_side.BUY, limit=50)
    orders = trading_client.get_orders(order_params)
    if not orders:
        return None

    filled_orders = [o for o in orders if getattr(o, "filled_at", None) is not None]
    if not filled_orders:
        return None

    latest = max(filled_orders, key=lambda o: o.filled_at)
    entry_date = latest.filled_at
    return (datetime.now(entry_date.tzinfo) - entry_date).days


def run_swing_logic(symbol: str = "AAPL") -> None:
    a, trading_client, data_client = _build_clients()
    market_order_request = a["MarketOrderRequest"]
    get_orders_request = a["GetOrdersRequest"]
    order_side = a["OrderSide"]
    time_in_force = a["TimeInForce"]
    query_order_status = a["QueryOrderStatus"]

    positions = trading_client.get_all_positions()
    for pos in positions:
        if pos.symbol != symbol:
            continue

        days_held = _latest_entry_age_days(trading_client, get_orders_request, query_order_status, order_side, symbol)
        if days_held is not None and days_held >= 10:
            print(f"Holding limit reached (Day {days_held}). Selling {symbol}...")
            sell_order = market_order_request(
                symbol=symbol,
                qty=pos.qty,
                side=order_side.SELL,
                time_in_force=time_in_force.GTC,
            )
            order = trading_client.submit_order(sell_order)
            _notify_successful_transaction(symbol=symbol, side="SELL", qty=pos.qty, order_id=getattr(order, "id", None))
            return

    current_rsi = get_rsi(symbol, data_client, a["StockBarsRequest"], a["TimeFrame"].Day)
    if current_rsi < 30:
        print(f"RSI is {current_rsi:.2f} (Oversold). Buying {symbol}...")
        buy_order = market_order_request(symbol=symbol, qty=1, side=order_side.BUY, time_in_force=time_in_force.GTC)
        order = trading_client.submit_order(buy_order)
        _notify_successful_transaction(symbol=symbol, side="BUY", qty=1, order_id=getattr(order, "id", None))
    else:
        print(f"RSI is {current_rsi:.2f}. No trade signal for {symbol}.")


def _fmt_num(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _format_duration(minutes_total: int) -> str:
    hours = minutes_total // 60
    minutes = minutes_total % 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _market_trigger_time_utc(now_utc: datetime, minutes_after_open: int = AUTO_TRIGGER_MINUTES_AFTER_OPEN) -> datetime:
    market_now = now_utc.astimezone(MARKET_TZ)
    trigger_market = market_now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(minutes=minutes_after_open)
    return trigger_market.astimezone(timezone.utc)


def _supports_color() -> bool:
    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("TERM", "").lower() == "dumb":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _supports_screen_refresh() -> bool:
    if os.getenv("TERM", "").lower() == "dumb":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _clear_screen() -> None:
    # ANSI clear + cursor home for dashboard-like refresh.
    print("\033[2J\033[H", end="")


def _paint(text: Any, color: str, use_color: bool) -> str:
    value = str(text)
    if not use_color:
        return value
    return f"{color}{value}{ANSI_RESET}"


def _style_rows(rows: list[dict[str, Any]], use_color: bool) -> list[dict[str, Any]]:
    styled: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if "symbol" in item:
            item["symbol"] = _paint(item["symbol"], ANSI_CYAN, use_color)
        if "action" in item:
            action = str(item["action"]).upper()
            if action == "BUY":
                item["action"] = _paint(action, ANSI_GREEN, use_color)
            elif action == "SELL":
                item["action"] = _paint(action, ANSI_RED, use_color)
            else:
                item["action"] = _paint(action, ANSI_YELLOW, use_color)
        if "is_open" in item:
            item["is_open"] = _paint(item["is_open"], ANSI_GREEN if item["is_open"] else ANSI_RED, use_color)
        if "u_pl" in item:
            pl = _to_float(item["u_pl"], 0.0)
            color = ANSI_GREEN if pl > 0 else ANSI_RED if pl < 0 else ANSI_YELLOW
            item["u_pl"] = _paint(f"{pl:.2f}", color, use_color)
        styled.append(item)
    return styled


def _human_label(key: str) -> str:
    if key in HUMAN_LABELS:
        return HUMAN_LABELS[key]
    return key.replace("_", " ").upper()


def _humanize_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{_human_label(str(key)): value for key, value in row.items()} for row in rows]


def _print_title(title: str, use_color: bool) -> None:
    if use_color:
        print(f"{ANSI_BOLD}{title}{ANSI_RESET}")
    else:
        print(title)


def _print_notice(message: str, color: str, use_color: bool) -> None:
    if use_color:
        print(f"{color}{message}{ANSI_RESET}")
    else:
        print(message)


def _maybe_notify_market_status(
    clock: Any,
    now: datetime,
    use_color: bool,
    state: dict[str, Any],
    preopen_notify_minutes: int = 60,
    preopen_interval_minutes: int = 10,
    closed_interval_minutes: int = 60,
) -> None:
    if clock.is_open:
        if state.get("status") != "open":
            _print_notice(
                f"[MARKET OPEN] Market is live. Next close: {clock.next_close}",
                ANSI_GREEN,
                use_color,
            )
        state["status"] = "open"
        state.pop("last_preopen_bucket", None)
        state.pop("last_closed_bucket", None)
        return

    next_open = clock.next_open
    minutes_to_open = max(0, int((next_open - now).total_seconds() // 60))
    status_key = f"{next_open.isoformat()}:{minutes_to_open}"

    just_transitioned_closed = state.get("status") != "closed"
    if just_transitioned_closed:
        _print_notice(
            f"[MARKET OFF] Next open: {next_open} ({_format_duration(minutes_to_open)} away)",
            ANSI_RED,
            use_color,
        )
    state["status"] = "closed"

    if minutes_to_open <= preopen_notify_minutes:
        bucket = minutes_to_open // max(1, preopen_interval_minutes)
        bucket_key = (next_open.isoformat(), bucket)
        if state.get("last_preopen_bucket") != bucket_key:
            _print_notice(
                f"[PRE-OPEN] Market opens in {_format_duration(minutes_to_open)} (at {next_open})",
                ANSI_YELLOW,
                use_color,
            )
            state["last_preopen_bucket"] = bucket_key
        state["last_closed_bucket"] = status_key
        return

    bucket = minutes_to_open // max(1, closed_interval_minutes)
    bucket_key = (next_open.isoformat(), bucket)
    if state.get("last_closed_bucket") != bucket_key and not just_transitioned_closed:
        _print_notice(
            f"[MARKET OFF] Market closed. Next open in {_format_duration(minutes_to_open)} (at {next_open})",
            ANSI_RED,
            use_color,
        )
        state["last_closed_bucket"] = bucket_key


def _print_table(
    rows: list[dict[str, Any]],
    tablefmt: str = "fancy_grid",
    use_color: bool = True,
    title: str | None = None,
) -> None:
    if not rows:
        print("(no rows)")
        return
    if title:
        _print_title(title, use_color)
    display_rows = _style_rows(rows, use_color)
    display_rows = _humanize_records(display_rows)
    print(tabulate(display_rows, headers="keys", tablefmt=tablefmt, showindex=False, floatfmt=".2f"))


def _current_holdings_rows(trading_client: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos in trading_client.get_all_positions():
        rows.append({
            "symbol": pos.symbol,
            "qty": round(_to_float(getattr(pos, "qty", 0.0)), 4),
            "avg_entry": round(_to_float(getattr(pos, "avg_entry_price", 0.0)), 2),
            "market_value": round(_to_float(getattr(pos, "market_value", 0.0)), 2),
            "u_pl": round(_to_float(getattr(pos, "unrealized_pl", 0.0)), 2),
            "u_pl_pct": f"{_to_float(getattr(pos, 'unrealized_plpc', 0.0)) * 100:.2f}%",
        })
    rows.sort(key=lambda r: str(r["symbol"]))
    return rows


def monitor_market_hours(
    poll_seconds: int = 60,
    iterations: int = 1,
    tablefmt: str = "fancy_grid",
    use_color: bool = True,
    refresh_screen: bool = True,
) -> None:
    _, trading_client, _ = _build_clients()
    count = 0
    while True:
        if refresh_screen:
            _clear_screen()
        now = datetime.now(timezone.utc)
        clock = trading_client.get_clock()
        rows = [{
            "timestamp_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
            "is_open": bool(clock.is_open),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }]
        _print_table(rows, tablefmt=tablefmt, use_color=use_color, title="Market Hours")
        sys.stdout.flush()

        count += 1
        if iterations > 0 and count >= iterations:
            break
        time.sleep(max(1, poll_seconds))


def _build_symbol_watchlist(max_symbols: int) -> list[str]:
    strong, _, _ = run_sector_report(top_strong=3, top_weak=2)
    symbols: list[str] = []

    if not strong.empty:
        picks = suggest_stocks(strong["sector"].tolist(), top_n=max_symbols)
        if not picks.empty:
            symbols.extend(picks["symbol"].head(max_symbols).tolist())

    if not symbols and not strong.empty:
        fallback = {
            "Technology": "AAPL",
            "Financial Services": "JPM",
            "Healthcare": "JNJ",
            "Energy": "XOM",
            "Industrials": "CAT",
        }
        for sector in strong["sector"].tolist():
            if sector in fallback and fallback[sector] not in symbols:
                symbols.append(fallback[sector])
            if len(symbols) >= max_symbols:
                break

    if not symbols:
        symbols = ["AAPL", "MSFT", "NVDA"][:max_symbols]

    return symbols


def _extract_symbol_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        if "symbol" in names:
            try:
                df = df.xs(symbol, level="symbol")
            except KeyError:
                return pd.DataFrame()
        else:
            first_level = df.index.get_level_values(0)
            if symbol in set(first_level):
                df = df.xs(symbol, level=0)
    return df.sort_index()


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(f"Missing required OHLCV columns: {missing}")
    out = pd.DataFrame({
        "open": pd.to_numeric(df[cols["open"]], errors="coerce"),
        "high": pd.to_numeric(df[cols["high"]], errors="coerce"),
        "low": pd.to_numeric(df[cols["low"]], errors="coerce"),
        "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
        "volume": pd.to_numeric(df[cols["volume"]], errors="coerce"),
    })
    return out.dropna()


def _fetch_research_bars_alpaca(
    *,
    data_client: Any,
    stock_bars_request: Any,
    timeframe: Any,
    symbol: str,
    start: datetime,
    end: datetime,
    feed: Any | None = None,
) -> pd.DataFrame:
    kwargs: dict[str, Any] = {
        "symbol_or_symbols": [symbol],
        "timeframe": timeframe.Day,
        "start": start,
        "end": end,
    }
    if feed is not None:
        kwargs["feed"] = feed
    req = stock_bars_request(**kwargs)
    bars = _extract_symbol_bars(data_client.get_stock_bars(req).df, symbol)
    if bars.empty:
        raise RuntimeError("No bars from Alpaca.")
    return _normalize_ohlcv_columns(bars)


def _fetch_research_bars_yfinance(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    hist = yf.download(
        tickers=symbol,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if hist.empty:
        raise RuntimeError("No bars from yfinance.")
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [str(c[0]) for c in hist.columns]
    return _normalize_ohlcv_columns(hist)


def _load_research_bars_alpaca(symbol: str, a: dict[str, Any], data_client: Any) -> tuple[pd.DataFrame, str]:
    stock_bars_request = a["StockBarsRequest"]
    timeframe = a["TimeFrame"]
    data_feed = a.get("DataFeed")
    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(days=420)
    end = now_utc

    try:
        bars = _fetch_research_bars_alpaca(
            data_client=data_client,
            stock_bars_request=stock_bars_request,
            timeframe=timeframe,
            symbol=symbol,
            start=start,
            end=end,
            feed=None,
        )
        return bars, "alpaca-default"
    except Exception:
        pass

    if data_feed is not None and hasattr(data_feed, "IEX"):
        try:
            bars = _fetch_research_bars_alpaca(
                data_client=data_client,
                stock_bars_request=stock_bars_request,
                timeframe=timeframe,
                symbol=symbol,
                start=start,
                end=end,
                feed=data_feed.IEX,
            )
            return bars, "alpaca-iex"
        except Exception:
            pass

    raise RuntimeError("Alpaca source failed (default + IEX).")


def _load_research_bars_yfinance(symbol: str) -> tuple[pd.DataFrame, str]:
    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(days=420)
    end = now_utc
    bars = _fetch_research_bars_yfinance(symbol=symbol, start=start, end=end)
    return bars, "yfinance"

def _compute_breakout55_metrics(bars: pd.DataFrame) -> dict[str, Any]:
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    volume = bars["volume"].astype(float)

    last_close = float(close.iloc[-1])
    prev_close = close.shift(1)

    prior_55_high = float(high.shift(1).rolling(55).max().iloc[-1])
    breakout55 = bool(last_close > prior_55_high) if prior_55_high == prior_55_high else False

    avg_vol20 = float(volume.shift(1).rolling(20).mean().iloc[-1])
    relvol = float(volume.iloc[-1] / avg_vol20) if avg_vol20 and avg_vol20 > 0 else float("nan")

    avg_vol20_series = volume.shift(1).rolling(20).mean()
    dist_flag = (close < prev_close) & (volume > avg_vol20_series)
    dist10 = int(dist_flag.tail(10).sum())

    prior_20_low = float(low.shift(1).rolling(20).min().iloc[-1])
    breakdown20 = bool(last_close < prior_20_low) if prior_20_low == prior_20_low else False

    return {
        "relvol": round(relvol, 2) if relvol == relvol else "-",
        "breakout55": breakout55,
        "dist10": dist10,
        "breakdown20": breakdown20,
    }


def _market_gate_spy_yfinance() -> bool | None:
    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(days=420)
    end = now_utc
    try:
        bars = _fetch_research_bars_yfinance(symbol="SPY", start=start, end=end)
        close = bars["close"].astype(float)
        sma200 = float(close.rolling(200).mean().iloc[-1])
        last = float(close.iloc[-1])
        if sma200 != sma200 or last != last:
            return None
        return bool(last > sma200)
    except Exception:
        return None


def _load_trade_bars_with_fallback(symbol: str, a: dict[str, Any], data_client: Any) -> tuple[pd.DataFrame, str]:
    try:
        return _load_research_bars_alpaca(symbol=symbol, a=a, data_client=data_client)
    except Exception:
        return _load_research_bars_yfinance(symbol=symbol)


def _rsi_from_close(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _macd_from_close(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])


def _bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[float, float, float]:
    sma = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    return float(upper_band.iloc[-1]), float(sma.iloc[-1]), float(lower_band.iloc[-1])


def _atr(bars: pd.DataFrame, period: int = 14) -> float:
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period).mean()
    return float(atr_series.iloc[-1])


def _score_research_snapshot(symbol: str, bars: pd.DataFrame, data_source: str, timing: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    volume = bars["volume"].astype(float)

    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    rsi14 = _rsi_from_close(close, period=14)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])
    last = float(close.iloc[-1])
    atr_pct = (atr14 / last * 100.0) if last > 0 else 0.0
    adv20 = float((close * volume).tail(20).mean())
    breakout_20d = float(high.tail(20).max())

    score = 0
    reasons: list[str] = []
    if last > sma200:
        score += 2
        reasons.append("Price above 200D trend.")
    else:
        score -= 2
        reasons.append("Price below 200D trend.")

    if sma50 > sma200:
        score += 1
        reasons.append("50D trend above 200D trend.")
    else:
        score -= 1
        reasons.append("50D trend below 200D trend.")

    if last > sma50:
        score += 1
        reasons.append("Price above 50D support.")
    else:
        score -= 1
        reasons.append("Price below 50D support.")

    if 40 <= rsi14 <= 65:
        score += 1
        reasons.append("RSI in healthy buyable zone (40-65).")
    elif rsi14 > 75:
        score -= 1
        reasons.append("RSI overheated (>75).")
    elif rsi14 < 30:
        score -= 1
        reasons.append("RSI weak/oversold (<30).")

    if adv20 >= 20_000_000:
        score += 1
        reasons.append("High liquidity (ADV20 >= $20M).")
    else:
        score -= 1
        reasons.append("Lower liquidity (ADV20 < $20M).")

    if atr_pct <= 5:
        score += 1
        reasons.append("Volatility acceptable (ATR% <= 5%).")
    else:
        score -= 1
        reasons.append("Volatility elevated (ATR% > 5%).")

    if score >= 4:
        rating = "GOOD BUY SETUP"
        zone_low = max(sma50, last - (0.5 * atr14))
        zone_high = min(last, breakout_20d)
    elif score >= 2:
        rating = "WATCH / WAIT"
        zone_low = max(sma50, last - atr14)
        zone_high = breakout_20d
    else:
        rating = "AVOID FOR NOW"
        zone_low = sma50
        zone_high = breakout_20d

    summary = {
        "symbol": symbol.upper(),
        "data_source": data_source,
        "rating": rating,
        "signal_score": score,
        "last": round(last, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "rsi14": round(rsi14, 2),
        "atr14": round(atr14, 2),
        "adv20_usd_m": round(adv20 / 1_000_000, 2),
        "buy_zone": f"{zone_low:.2f} - {zone_high:.2f}",
        "timing": timing,
    }
    if data_source == "yfinance":
        reasons.append("Using yfinance fallback data source (non-Alpaca feed).")
    reason_rows = [{"symbol": symbol.upper(), "reason": reason} for reason in reasons]
    return summary, reason_rows


def run_research_mode(symbol: str, tablefmt: str, use_color: bool) -> None:
    a, trading_client, data_client = _build_clients()
    timing = "OPEN" if bool(trading_client.get_clock().is_open) else "CLOSED"

    research_rows: list[dict[str, Any]] = []
    all_reasons: list[dict[str, Any]] = []

    for source in ("alpaca", "yfinance"):
        try:
            if source == "alpaca":
                bars, data_source = _load_research_bars_alpaca(symbol=symbol, a=a, data_client=data_client)
            else:
                bars, data_source = _load_research_bars_yfinance(symbol=symbol)

            summary, reasons = _score_research_snapshot(symbol=symbol, bars=bars, data_source=data_source, timing=timing)
            research_rows.append(summary)
            for rr in reasons:
                rr["data_source"] = data_source
            all_reasons.extend(reasons)
        except Exception as exc:
            failed_source = "alpaca" if source == "alpaca" else "yfinance"
            research_rows.append({
                "symbol": symbol.upper(),
                "data_source": failed_source,
                "rating": "ERROR",
                "signal_score": "-",
                "last": "-",
                "sma50": "-",
                "sma200": "-",
                "rsi14": "-",
                "atr14": "-",
                "adv20_usd_m": "-",
                "buy_zone": "-",
                "timing": timing,
            })
            all_reasons.append({
                "symbol": symbol.upper(),
                "data_source": failed_source,
                "reason": f"Source failed: {exc}",
            })

    _print_table(research_rows, tablefmt=tablefmt, use_color=use_color, title=f"Research: {symbol.upper()}")
    _print_table(all_reasons, tablefmt=tablefmt, use_color=use_color, title="Why This Rating")

def build_trade_plan(max_symbols: int = 8, entry_style: str = DEFAULT_ENTRY_STYLE, atr_mult: float = 2.5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    a, trading_client, data_client = _build_clients()

    get_orders_request = a["GetOrdersRequest"]
    query_order_status = a["QueryOrderStatus"]
    order_side = a["OrderSide"]

    positions = {p.symbol: p for p in trading_client.get_all_positions()}
    symbols = _build_symbol_watchlist(max_symbols=max_symbols)
    entry_style = (entry_style or DEFAULT_ENTRY_STYLE).strip().lower()
    market_gate = _market_gate_spy_yfinance()

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        bars, source = _load_trade_bars_with_fallback(symbol=symbol, a=a, data_client=data_client)
        snap = compute_signal_snapshot(symbol, bars, atr_mult=atr_mult)

        pos = positions.get(symbol)
        qty = float(pos.qty) if pos else 0.0
        days_held = _latest_entry_age_days(trading_client, get_orders_request, query_order_status, order_side, symbol) if pos else None
        risk_ok = True  # lightweight: ATR% gating is shown in research; implement later if needed
        snap_with_gate = dict(snap)
        snap_with_gate["market_gate"] = market_gate
        action, reason = decide_action(
            snap_with_gate,
            entry_style,
            qty=qty,
            days_held=days_held,
            atr_mult=atr_mult,
        )

        rows.append({
            "symbol": symbol,
            "signal_score": snap.get("signal_score", 0),
            "rsi": snap["rsi"],
            "relvol": snap["relvol"],
            "breakout55": snap["breakout55"],
            "dist10": snap["dist10"],
            "trend_ok": snap["trend_ok"],
            "risk_ok": risk_ok,
            "market_gate": market_gate if market_gate is not None else "-",
            "position_qty": round(qty, 4),
            "days_held": days_held if days_held is not None else "-",
            "action": action,
            "reason": f"{reason} (source={source})",
        })

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_symbols": max_symbols,
        "plan_size": len(rows),
        "entry_style": entry_style,
    }
    return rows, meta


def _save_plan(rows: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
    root = Path(__file__).resolve().parents[2]
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    filename = f"trade_plan_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    path = logs_dir / filename
    payload = {
        "META": _humanize_records([meta])[0],
        "ROWS": _humanize_records(rows),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_options_plan(max_symbols: int = 10, atr_mult: float = 2.5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Late import to prevent circular or missing dependencies on boot
    from .engine import compute_signal_snapshot
    from .options_engine import recommend_strategy

    a, trading_client, data_client = _build_clients()
    symbols = _build_symbol_watchlist(max_symbols=max_symbols)
    
    rows: list[dict[str, Any]] = []
    
    for symbol in symbols:
        try:
            bars, source = _load_trade_bars_with_fallback(symbol=symbol, a=a, data_client=data_client)
            snap = compute_signal_snapshot(symbol, bars, atr_mult=atr_mult)
            # Ensure symbol is set explicitly 
            snap["symbol"] = symbol
            rec = recommend_strategy(snap)
            top = rec.top
            rows.append({
                "symbol": symbol,
                "strategy": top.name,
                "direction": top.direction,
                "score": float(top.score),
                "reason": top.brief,
                "source": source,
            })
        except Exception:
            continue
            
    # Sort by score descending
    rows.sort(key=lambda x: x["score"], reverse=True)
    
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_symbols": max_symbols,
        "plan_size": len(rows),
        "type": "options_plan",
    }
    return rows, meta


def _save_options_plan(rows: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
    root = Path(__file__).resolve().parents[2]
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    filename = f"options_plan_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    path = logs_dir / filename
    payload = {
        "META": _humanize_records([meta])[0],
        "ROWS": _humanize_records(rows),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path



def execute_plan(rows: list[dict[str, Any]]) -> None:
    a, trading_client, _ = _build_clients()
    market_order_request = a["MarketOrderRequest"]
    order_side = a["OrderSide"]
    time_in_force = a["TimeInForce"]

    positions = {p.symbol: p for p in trading_client.get_all_positions()}

    for row in rows:
        symbol = row["symbol"]
        action = row["action"]

        if action == "BUY":
            req = market_order_request(
                symbol=symbol,
                qty=1,
                side=order_side.BUY,
                time_in_force=time_in_force.DAY,
            )
            order = trading_client.submit_order(req)
            print(f"Submitted BUY for {symbol}")
            _notify_successful_transaction(symbol=symbol, side="BUY", qty=1, order_id=getattr(order, "id", None))
        elif action == "SELL":
            pos = positions.get(symbol)
            if pos is None:
                print(f"Skipped SELL for {symbol}: no position")
                continue
            req = market_order_request(
                symbol=symbol,
                qty=pos.qty,
                side=order_side.SELL,
                time_in_force=time_in_force.DAY,
            )
            order = trading_client.submit_order(req)
            print(f"Submitted SELL for {symbol} qty={pos.qty}")
            _notify_successful_transaction(symbol=symbol, side="SELL", qty=pos.qty, order_id=getattr(order, "id", None))


def execute_options_trade(legs: list[dict[str, Any]], multiplier: int) -> None:
    a, trading_client, _ = _build_clients()
    market_order_request = a["MarketOrderRequest"]
    order_side = a["OrderSide"]
    time_in_force = a["TimeInForce"]
    
    for leg in legs:
        action_str = str(leg.get("action", "")).lower()
        if "buy" in action_str:
            side = order_side.BUY
        elif "sell" in action_str:
            side = order_side.SELL
        else:
            continue
            
        symbol = str(leg.get("contract", ""))
        if not symbol:
            continue
            
        req = market_order_request(
            symbol=symbol,
            qty=max(1, int(multiplier)),
            side=side,
            time_in_force=time_in_force.DAY,
        )
        try:
            order = trading_client.submit_order(req)
            print(f"Submitted options {side} for {symbol} qty={multiplier}")
            _notify_successful_transaction(symbol=symbol, side=f"OPTIONS {side}", qty=max(1, int(multiplier)), order_id=getattr(order, "id", None))
        except Exception as e:
            print(f"Error submitting options order for {symbol}: {e}")
            raise e



def _read_live_command_nonblocking(timeout_seconds: float = 0.0) -> str | None:
    if not getattr(sys.stdin, "isatty", lambda: False)():
        return None
    try:
        ready, _, _ = select.select([sys.stdin], [], [], max(0.0, timeout_seconds))
    except Exception:
        return None
    if not ready:
        return None
    line = sys.stdin.readline()
    if not line:
        return None
    return line.strip()


def _print_live_command_help(use_color: bool) -> None:
    _print_title("Live Commands", use_color)
    print("  buy <SYMBOL> [QTY]      Submit immediate market BUY (default qty=1)")
    print("  sell <SYMBOL> [QTY|all] Submit immediate market SELL (default all held)")
    print("  research <SYMBOL>       Analyze ticker before buying")
    print("  holdings                Print current holdings")
    print("  plan [MAX_SYMBOLS]      Generate and print current plan now")
    print("  options                 Generate options plan for top 10 tickers")
    print("  help                    Show commands")
    print("  quit                    Stop auto mode")


def _handle_live_command(
    raw_command: str,
    *,
    a: dict[str, Any],
    trading_client: Any,
    max_symbols: int,
    entry_style: str,
    atr_mult: float,
    tablefmt: str,
    use_color: bool,
) -> bool:
    command = raw_command.strip()
    if not command:
        return True
    parts = command.split()
    cmd = parts[0].lower()

    market_order_request = a["MarketOrderRequest"]
    order_side = a["OrderSide"]
    time_in_force = a["TimeInForce"]

    try:
        if cmd in {"help", "?"}:
            _print_live_command_help(use_color)
            return True

        if cmd in {"quit", "exit", "q"}:
            print("Stopping auto mode by user request.")
            return False

        if cmd == "holdings":
            _print_table(_current_holdings_rows(trading_client), tablefmt=tablefmt, use_color=use_color, title="Current Holdings")
            return True

        if cmd == "research":
            if len(parts) < 2:
                print("Usage: research <SYMBOL>")
                return True
            run_research_mode(parts[1].upper(), tablefmt=tablefmt, use_color=use_color)
            return True

        if cmd == "plan":
            live_max_symbols = max_symbols
            if len(parts) >= 2:
                live_max_symbols = max(1, int(parts[1]))
            rows, meta = build_trade_plan(max_symbols=live_max_symbols, entry_style=entry_style, atr_mult=atr_mult)
            _print_table(rows, tablefmt=tablefmt, use_color=use_color, title="Live Plan")
            print(f"Live plan meta: {meta}")
            return True

        if cmd == "options":
            opt_rows, opt_meta = build_options_plan(max_symbols=10, atr_mult=atr_mult)
            _print_table(opt_rows, tablefmt=tablefmt, use_color=use_color, title="Live Options Plan")
            print(f"Live options plan meta: {opt_meta}")
            return True

        if cmd == "buy":
            if len(parts) < 2:
                print("Usage: buy <SYMBOL> [QTY]")
                return True
            symbol = parts[1].upper()
            qty = 1.0
            if len(parts) >= 3:
                qty = float(parts[2])
            if qty <= 0:
                print("BUY skipped: qty must be > 0")
                return True
            req = market_order_request(
                symbol=symbol,
                qty=qty,
                side=order_side.BUY,
                time_in_force=time_in_force.DAY,
            )
            order = trading_client.submit_order(req)
            print(f"LIVE ORDER SUBMITTED: BUY {symbol} qty={qty}")
            _notify_successful_transaction(symbol=symbol, side="BUY", qty=qty, order_id=getattr(order, "id", None))
            return True

        if cmd == "sell":
            if len(parts) < 2:
                print("Usage: sell <SYMBOL> [QTY|all]")
                return True
            symbol = parts[1].upper()
            positions = {p.symbol: p for p in trading_client.get_all_positions()}
            pos = positions.get(symbol)
            if pos is None:
                print(f"SELL skipped: no open position for {symbol}")
                return True

            qty: float
            if len(parts) >= 3 and parts[2].lower() != "all":
                qty = float(parts[2])
            else:
                qty = _to_float(getattr(pos, "qty", 0.0), 0.0)

            held_qty = _to_float(getattr(pos, "qty", 0.0), 0.0)
            if qty <= 0:
                print("SELL skipped: qty must be > 0")
                return True
            if qty > held_qty:
                print(f"SELL skipped: qty {qty} exceeds held quantity {held_qty}")
                return True

            req = market_order_request(
                symbol=symbol,
                qty=qty,
                side=order_side.SELL,
                time_in_force=time_in_force.DAY,
            )
            order = trading_client.submit_order(req)
            print(f"LIVE ORDER SUBMITTED: SELL {symbol} qty={qty}")
            _notify_successful_transaction(symbol=symbol, side="SELL", qty=qty, order_id=getattr(order, "id", None))
            return True

        print(f"Unknown command: {command}. Type 'help' for available commands.")
        return True
    except ValueError as exc:
        print(f"Invalid command values: {exc}")
        return True
    except Exception as exc:
        print(f"Live command failed: {exc}")
        return True


def _process_live_commands_for_duration(
    duration_seconds: int,
    *,
    a: dict[str, Any],
    trading_client: Any,
    max_symbols: int,
    entry_style: str,
    atr_mult: float,
    tablefmt: str,
    use_color: bool,
) -> bool:
    if duration_seconds <= 0:
        duration_seconds = 1
    deadline = time.time() + duration_seconds
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return True
        poll_timeout = min(0.25, remaining)
        raw_command = _read_live_command_nonblocking(timeout_seconds=poll_timeout)
        if raw_command is None:
            continue
        keep_running = _handle_live_command(
            raw_command,
            a=a,
            trading_client=trading_client,
            max_symbols=max_symbols,
            entry_style=entry_style,
            atr_mult=atr_mult,
            tablefmt=tablefmt,
            use_color=use_color,
        )
        sys.stdout.flush()
        if not keep_running:
            return False


def run_plan_mode(max_symbols: int, execute: bool, tablefmt: str, use_color: bool, entry_style: str, atr_mult: float, options_plan: bool) -> None:
    rows, meta = build_trade_plan(max_symbols=max_symbols, entry_style=entry_style, atr_mult=atr_mult)
    _print_table(rows, tablefmt=tablefmt, use_color=use_color, title="Trade Plan")
    saved = _save_plan(rows, meta)
    print(f"Plan saved: {saved}")
    if execute:
        execute_plan(rows)
        
    if options_plan:
        opt_rows, opt_meta = build_options_plan(max_symbols=min(10, max_symbols), atr_mult=atr_mult)
        _print_table(opt_rows, tablefmt=tablefmt, use_color=use_color, title="Options Trade Plan")
        opt_saved = _save_options_plan(opt_rows, opt_meta)
        print(f"Options plan saved: {opt_saved}")


def run_auto_mode(
    max_symbols: int,
    refresh_minutes: int,
    execute: bool,
    tablefmt: str,
    loop_seconds: int,
    use_color: bool,
    refresh_screen: bool,
    entry_style: str,
    atr_mult: float,
    options_plan: bool,
) -> None:
    _ = refresh_minutes  # legacy argument retained for backward-compatible CLI
    market_notice_state: dict[str, Any] = {}
    last_triggered_market_day: str | None = None
    last_run_summary: dict[str, Any] | None = None
    closed_snapshot_printed_for_day: str | None = None
    a, trading_client, _ = _build_clients()
    command_hint_printed = False

    while True:
        now = datetime.now(timezone.utc)
        clock = trading_client.get_clock()
        trigger_time_utc = _market_trigger_time_utc(now)
        market_day = now.astimezone(MARKET_TZ).date().isoformat()
        triggered_today = last_triggered_market_day == market_day
        did_trigger_now = False

        if clock.is_open:
            closed_snapshot_printed_for_day = None
            if refresh_screen:
                _clear_screen()

        _maybe_notify_market_status(clock=clock, now=now, use_color=use_color, state=market_notice_state)

        should_render_dashboard = bool(clock.is_open)
        if (not clock.is_open) and closed_snapshot_printed_for_day != market_day:
            should_render_dashboard = True
            closed_snapshot_printed_for_day = market_day

        if clock.is_open and now >= trigger_time_utc and not triggered_today:
            plan_rows, meta = build_trade_plan(max_symbols=max_symbols, entry_style=entry_style, atr_mult=atr_mult)
            saved = _save_plan(plan_rows, meta)
            if should_render_dashboard:
                _print_table(plan_rows, tablefmt=tablefmt, use_color=use_color, title="Triggered Trade Plan")
                _print_table(_current_holdings_rows(trading_client), tablefmt=tablefmt, use_color=use_color, title="Current Holdings")
            if execute:
                execute_plan(plan_rows)

            opt_saved = "-"
            if options_plan:
                opt_rows, opt_meta = build_options_plan(max_symbols=min(10, max_symbols), atr_mult=atr_mult)
                opt_saved_path = _save_options_plan(opt_rows, opt_meta)
                opt_saved = str(opt_saved_path)
                if should_render_dashboard:
                    _print_table(opt_rows, tablefmt=tablefmt, use_color=use_color, title="Triggered Options Plan (Top 10)")

            last_triggered_market_day = market_day
            buy_count = sum(1 for row in plan_rows if row.get("action") == "BUY")
            sell_count = sum(1 for row in plan_rows if row.get("action") == "SELL")
            hold_count = sum(1 for row in plan_rows if row.get("action") == "HOLD")
            last_run_summary = {
                "triggered_market_day": market_day,
                "triggered_at_utc": now.isoformat(),
                "plan_file": str(saved),
                "options_plan_file": opt_saved,
                "buy": buy_count,
                "sell": sell_count,
                "hold": hold_count,
            }
            triggered_today = True
            did_trigger_now = True
        elif should_render_dashboard:
            _print_table(_current_holdings_rows(trading_client), tablefmt=tablefmt, use_color=use_color, title="Current Holdings")

        if did_trigger_now:
            next_action = "RUNNING_NOW"
        elif triggered_today:
            next_action = "TRIGGERED_TODAY"
        elif not clock.is_open:
            next_action = "WAIT_MARKET_OPEN"
        elif now < trigger_time_utc:
            next_action = "WAIT_TRIGGER_TIME"
        else:
            next_action = "WAIT_NEXT_MARKET_DAY"

        if should_render_dashboard:
            clock_rows = [{
                "timestamp_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                "is_open": bool(clock.is_open),
                "next_open": str(clock.next_open),
                "next_close": str(clock.next_close),
            }]
            _print_table(clock_rows, tablefmt=tablefmt, use_color=use_color, title="Market Hours")

            status_rows = [{
                "market_day_et": market_day,
                "trigger_time_et": trigger_time_utc.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "triggered_today": triggered_today,
                "next_action": next_action,
            }]
            _print_table(status_rows, tablefmt=tablefmt, use_color=use_color, title="Auto Status")
            if last_run_summary:
                _print_table([last_run_summary], tablefmt=tablefmt, use_color=use_color, title="Last Trigger")

        if not command_hint_printed and getattr(sys.stdin, "isatty", lambda: False)():
            print("Live commands enabled. Type 'help' and press Enter.")
            command_hint_printed = True

        sys.stdout.flush()
        keep_running = _process_live_commands_for_duration(
            max(1, loop_seconds),
            a=a,
            trading_client=trading_client,
            max_symbols=max_symbols,
            entry_style=entry_style,
            atr_mult=atr_mult,
            tablefmt=tablefmt,
            use_color=use_color,
        )
        if not keep_running:
            return


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alpaca trading helper")
    parser.add_argument("--mode", choices=["swing", "hours", "plan", "auto", "research"], default="swing")
    parser.add_argument("--symbol", default="AAPL", help="Symbol for swing mode")
    parser.add_argument("--max-symbols", type=int, default=DEFAULT_MAX_SYMBOLS, help="Watchlist size for plan/auto mode")
    parser.add_argument("--entry-style", choices=["hybrid", "breakout55", "rsi30", "macd", "bollinger"], default=DEFAULT_ENTRY_STYLE, help="Entry logic for plan/auto")
    parser.add_argument("--atr-mult", type=float, default=2.5, help="Multiplier for ATR trailing stop")
    parser.add_argument(
        "--refresh-minutes",
        type=int,
        default=AUTO_DEFAULT_REFRESH_MINUTES,
        help="Legacy option (currently unused in trigger-once auto mode)",
    )
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval for hours mode")
    parser.add_argument("--iterations", type=int, default=1, help="Rows to print in hours mode (0 for infinite)")
    parser.add_argument("--loop-seconds", type=int, default=60, help="Auto mode loop interval")
    parser.add_argument("--execute", action="store_true", help="Submit orders for BUY/SELL actions")
    parser.add_argument("--options-plan", action="store_true", help="Generate an options play plan for top tickers")
    parser.add_argument("--tablefmt", default="fancy_grid", help="tabulate format (fancy_grid, github, simple, plain, grid, ...)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color in tables")
    parser.add_argument("--no-refresh", action="store_true", help="Disable screen refresh and print output as logs")
    return parser


def main() -> None:
    args = _parser().parse_args()
    use_color = _supports_color() and not args.no_color
    refresh_screen = _supports_screen_refresh() and not args.no_refresh

    if args.mode == "swing":
        run_swing_logic(symbol=args.symbol)
    elif args.mode == "hours":
        monitor_market_hours(
            poll_seconds=args.poll_seconds,
            iterations=args.iterations,
            tablefmt=args.tablefmt,
            use_color=use_color,
            refresh_screen=refresh_screen,
        )
    elif args.mode == "plan":
        run_plan_mode(
            max_symbols=args.max_symbols,
            execute=args.execute,
            tablefmt=args.tablefmt,
            use_color=use_color,
            entry_style=args.entry_style,
            atr_mult=args.atr_mult,
            options_plan=args.options_plan,
        )
    elif args.mode == "auto":
        auto_max_symbols = AUTO_DEFAULT_MAX_SYMBOLS if args.max_symbols == DEFAULT_MAX_SYMBOLS else args.max_symbols
        run_auto_mode(
            max_symbols=auto_max_symbols,
            refresh_minutes=args.refresh_minutes,
            execute=True,
            tablefmt=args.tablefmt,
            loop_seconds=args.loop_seconds,
            use_color=use_color,
            refresh_screen=refresh_screen,
            entry_style=args.entry_style,
            atr_mult=args.atr_mult,
            options_plan=args.options_plan,
        )
    elif args.mode == "research":
        run_research_mode(symbol=args.symbol, tablefmt=args.tablefmt, use_color=use_color)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Trading client error: {exc}")
