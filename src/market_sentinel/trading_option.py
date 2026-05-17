from __future__ import annotations

import argparse
import os
import select
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from tabulate import tabulate

from .analysis import run_sector_report, suggest_stocks

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_MAX_SYMBOLS = 15
AUTO_TRIGGER_MINUTES_AFTER_OPEN = 30
MARKET_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class UnderlyingSignal:
    symbol: str
    last: float
    sma50: float
    sma200: float
    breakout55: bool
    relvol: float | None
    dist10: int
    trend_ok: bool


@dataclass(frozen=True)
class DebitSpreadSuggestion:
    symbol: str
    expiry: str
    long_contract: str
    short_contract: str
    long_strike: float
    short_strike: float
    debit: float
    max_profit: float
    max_loss: float
    breakeven: float
    rr: float
    notes: str


def _alpaca_imports() -> dict[str, Any]:
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, QueryOrderStatus, TimeInForce
        from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, OptionLegRequest
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'alpaca-py'. Install with: pip install alpaca-py") from exc
    return {
        "TradingClient": TradingClient,
        "OrderClass": OrderClass,
        "OrderSide": OrderSide,
        "PositionIntent": PositionIntent,
        "QueryOrderStatus": QueryOrderStatus,
        "TimeInForce": TimeInForce,
        "GetOrdersRequest": GetOrdersRequest,
        "MarketOrderRequest": MarketOrderRequest,
        "OptionLegRequest": OptionLegRequest,
    }


def _build_clients() -> tuple[dict[str, Any], Any]:
    if load_dotenv is None:
        raise RuntimeError("Missing dependency 'python-dotenv'. Install with: pip install python-dotenv")
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

    api_key = os.getenv("ALPACA_KEY")
    api_secret = os.getenv("ALPACA_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_KEY/ALPACA_SECRET must be set in src/market_sentinel/.env")

    is_paper = os.getenv("ALPACA_PAPER", "true").strip().lower() in {"1", "true", "yes"}
    a = _alpaca_imports()
    trading_client = a["TradingClient"](api_key, api_secret, paper=is_paper)
    return a, trading_client


def _supports_screen_refresh() -> bool:
    if os.getenv("TERM", "").lower() == "dumb":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _clear_screen() -> None:
    print("\033[2J\033[H", end="")


def _print_table(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n{title}")
    if not rows:
        print("(no rows)")
        return
    print(tabulate(rows, headers="keys", tablefmt="fancy_grid", showindex=False))


def _occ_underlying(symbol: str) -> str:
    s = str(symbol).upper()
    out = []
    for ch in s:
        if ch.isalpha():
            out.append(ch)
        else:
            break
    return "".join(out) or s


def _current_holdings_rows(trading_client: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos in trading_client.get_all_positions():
        rows.append(
            {
                "symbol": pos.symbol,
                "underlying": _occ_underlying(pos.symbol),
                "qty": getattr(pos, "qty", "-"),
                "avg_entry": getattr(pos, "avg_entry_price", "-"),
                "market_value": getattr(pos, "market_value", "-"),
                "u_pl": getattr(pos, "unrealized_pl", "-"),
                "u_pl_pct": getattr(pos, "unrealized_plpc", "-"),
            }
        )
    rows.sort(key=lambda r: (str(r["underlying"]), str(r["symbol"])))
    return rows


def _market_trigger_time_utc(now_utc: datetime, minutes_after_open: int = AUTO_TRIGGER_MINUTES_AFTER_OPEN) -> datetime:
    market_now = now_utc.astimezone(MARKET_TZ)
    trigger_market = market_now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(minutes=minutes_after_open)
    return trigger_market.astimezone(timezone.utc)


def _build_underlying_watchlist(max_symbols: int) -> tuple[list[str], dict[str, str]]:
    fallback = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "XOM", "LLY", "AVGO", "COST", "AMD", "CAT", "GS"]
    try:
        strong, _weak, _ = run_sector_report(top_strong=3, top_weak=2)
        if strong.empty:
            symbols = fallback[:max_symbols]
            reasons = {s: "Fallback: liquid large-cap/benchmark name." for s in symbols}
            return symbols, reasons

        picks = suggest_stocks(strong["sector"].tolist(), top_n=max_symbols)
        if picks.empty:
            symbols = fallback[:max_symbols]
            reasons = {s: "Fallback: liquid large-cap/benchmark name." for s in symbols}
            return symbols, reasons

        symbols = picks["symbol"].dropna().astype(str).head(max_symbols).tolist()
        reasons: dict[str, str] = {}
        for _, row in picks.head(max_symbols).iterrows():
            sym = str(row.get("symbol", "")).strip()
            if not sym:
                continue
            reasons[sym] = f"Sector={row.get('sector')} score={row.get('score'):.2f} 3M={row.get('return_3m_pct'):.2f}% ADV20=${row.get('avg_dollar_vol_20d_m'):.1f}M"
        return symbols, reasons
    except Exception:
        symbols = fallback[:max_symbols]
        reasons = {s: "Fallback: analysis unavailable." for s in symbols}
        return symbols, reasons


def _download_daily_ohlcv(symbol: str, lookback_days: int = 420) -> pd.DataFrame:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)
    hist = yf.download(
        tickers=symbol,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if hist.empty:
        raise RuntimeError(f"No daily data returned for {symbol}.")
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [str(c[0]) for c in hist.columns]
    cols = {c.lower(): c for c in hist.columns}
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(f"Missing required columns for {symbol}: {missing}")
    df = pd.DataFrame(
        {
            "open": pd.to_numeric(hist[cols["open"]], errors="coerce"),
            "high": pd.to_numeric(hist[cols["high"]], errors="coerce"),
            "low": pd.to_numeric(hist[cols["low"]], errors="coerce"),
            "close": pd.to_numeric(hist[cols["close"]], errors="coerce"),
            "volume": pd.to_numeric(hist[cols["volume"]], errors="coerce"),
        }
    ).dropna()
    return df


def compute_breakout55_signal(symbol: str) -> UnderlyingSignal:
    bars = _download_daily_ohlcv(symbol)
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    volume = bars["volume"].astype(float)

    if len(bars) < 220:
        raise RuntimeError(f"Insufficient history for {symbol} (need ~220+ daily bars).")

    last = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    trend_ok = bool(last > sma200 and sma50 >= sma200)

    prior_55_high = float(high.shift(1).rolling(55).max().iloc[-1])
    breakout55 = bool(last > prior_55_high) if prior_55_high == prior_55_high else False

    avg_vol20 = float(volume.shift(1).rolling(20).mean().iloc[-1])
    relvol = float(volume.iloc[-1] / avg_vol20) if avg_vol20 and avg_vol20 > 0 else None

    prev_close = close.shift(1)
    avg_vol20_series = volume.shift(1).rolling(20).mean()
    dist_flag = (close < prev_close) & (volume > avg_vol20_series)
    dist10 = int(dist_flag.tail(10).sum())

    return UnderlyingSignal(
        symbol=symbol.upper(),
        last=round(last, 2),
        sma50=round(sma50, 2),
        sma200=round(sma200, 2),
        breakout55=breakout55,
        relvol=round(relvol, 2) if relvol is not None else None,
        dist10=dist10,
        trend_ok=trend_ok,
    )


def _mid(bid: float | None, ask: float | None, last: float | None) -> float | None:
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return float((bid + ask) / 2.0)
    if last is not None and last > 0:
        return float(last)
    return None


def _pick_expiry(options: list[str], dte_min: int, dte_max: int, target_dte: int) -> str:
    today = date.today()
    best: tuple[int, str] | None = None
    for exp in options:
        try:
            d = date.fromisoformat(exp)
        except ValueError:
            continue
        dte = (d - today).days
        if dte < dte_min or dte > dte_max:
            continue
        distance = abs(dte - target_dte)
        cand = (distance, exp)
        if best is None or cand < best:
            best = cand
    if best is None:
        raise RuntimeError(f"No expiry found in DTE range {dte_min}-{dte_max}.")
    return best[1]


def suggest_call_debit_spread(
    symbol: str,
    *,
    dte_min: int = 30,
    dte_max: int = 75,
    target_dte: int = 45,
    width: float = 5.0,
    min_oi: int = 100,
) -> tuple[UnderlyingSignal, DebitSpreadSuggestion | None]:
    sig = compute_breakout55_signal(symbol)

    ticker = yf.Ticker(symbol)
    options = list(getattr(ticker, "options", []) or [])
    if not options:
        raise RuntimeError(f"No options chain available via yfinance for {symbol}.")

    expiry = _pick_expiry(options, dte_min=dte_min, dte_max=dte_max, target_dte=target_dte)
    chain = ticker.option_chain(expiry)
    calls = chain.calls.copy()
    if calls.empty:
        raise RuntimeError(f"No calls returned for {symbol} {expiry}.")

    if "openInterest" in calls.columns:
        calls = calls[calls["openInterest"].fillna(0) >= min_oi]
    if calls.empty:
        raise RuntimeError(f"No sufficiently liquid calls (OI>={min_oi}) for {symbol} {expiry}.")

    spot = float(sig.last)
    calls["strike"] = pd.to_numeric(calls["strike"], errors="coerce")
    calls = calls.dropna(subset=["strike"]).sort_values("strike")

    below = calls[calls["strike"] <= spot]
    if not below.empty:
        long_strike = float(below["strike"].iloc[-1])
    else:
        long_strike = float(calls["strike"].iloc[0])
    short_strike = float(long_strike + width)

    long_row = calls.loc[calls["strike"] == long_strike].iloc[0]
    short_candidates = calls.loc[calls["strike"] == short_strike]
    if short_candidates.empty:
        above = calls[calls["strike"] > long_strike]
        if above.empty:
            return sig, None
        short_row = above.iloc[min(len(above) - 1, 1)]
        short_strike = float(short_row["strike"])
    else:
        short_row = short_candidates.iloc[0]

    long_mid = _mid(
        bid=float(long_row.get("bid")) if pd.notna(long_row.get("bid")) else None,
        ask=float(long_row.get("ask")) if pd.notna(long_row.get("ask")) else None,
        last=float(long_row.get("lastPrice")) if pd.notna(long_row.get("lastPrice")) else None,
    )
    short_mid = _mid(
        bid=float(short_row.get("bid")) if pd.notna(short_row.get("bid")) else None,
        ask=float(short_row.get("ask")) if pd.notna(short_row.get("ask")) else None,
        last=float(short_row.get("lastPrice")) if pd.notna(short_row.get("lastPrice")) else None,
    )
    if long_mid is None or short_mid is None:
        return sig, None

    long_contract = str(long_row.get("contractSymbol", "")).strip()
    short_contract = str(short_row.get("contractSymbol", "")).strip()
    if not long_contract or not short_contract:
        return sig, None

    debit = max(0.01, float(long_mid - short_mid))
    spread_width = float(short_strike - long_strike)
    max_profit = max(0.0, spread_width - debit)
    max_loss = debit
    breakeven = long_strike + debit
    rr = (max_profit / max_loss) if max_loss > 0 else 0.0

    notes = []
    if not sig.trend_ok:
        notes.append("Trend not OK (Close>SMA200 and SMA50>=SMA200).")
    if not sig.breakout55:
        notes.append("No Breakout55 today; treat as WATCH.")
    if sig.relvol is not None and sig.relvol < 1.3:
        notes.append(f"Volume weak (RelVol {sig.relvol}).")
    if sig.dist10 >= 3:
        notes.append(f"Distribution elevated (Dist10={sig.dist10}).")
    if rr < 1.0:
        notes.append(f"Reward/Risk low ({rr:.2f}).")
    if not notes:
        notes.append("Manage at +50% max profit or at 21DTE; cut at -50% debit.")

    sugg = DebitSpreadSuggestion(
        symbol=sig.symbol,
        expiry=expiry,
        long_contract=long_contract,
        short_contract=short_contract,
        long_strike=round(long_strike, 2),
        short_strike=round(short_strike, 2),
        debit=round(debit, 2),
        max_profit=round(max_profit, 2),
        max_loss=round(max_loss, 2),
        breakeven=round(breakeven, 2),
        rr=round(rr, 2),
        notes=" ".join(notes),
    )
    return sig, sugg


def build_option_plan(max_symbols: int, dte_min: int, dte_max: int, target_dte: int, width: float, min_oi: int, trading_client: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    symbols, reasons = _build_underlying_watchlist(max_symbols=max_symbols)
    held = _current_holdings_rows(trading_client)
    held_underlyings = {r["underlying"] for r in held if r.get("underlying")}

    rows: list[dict[str, Any]] = []
    for sym in symbols:
        try:
            sig, sugg = suggest_call_debit_spread(
                sym,
                dte_min=dte_min,
                dte_max=dte_max,
                target_dte=target_dte,
                width=width,
                min_oi=min_oi,
            )
            have_pos = sig.symbol in held_underlyings
            action = "HOLD" if have_pos else "WATCH"
            reason = "Existing position" if have_pos else "Waiting setup"

            relvol_ok = sig.relvol is not None and sig.relvol >= 1.3
            if (not have_pos) and sig.trend_ok and sig.breakout55 and relvol_ok and sig.dist10 < 3 and sugg is not None:
                action = "BUY_SPREAD"
                reason = f"Breakout55 + RelVol {sig.relvol} + Dist10 {sig.dist10}"

            rows.append(
                {
                    "symbol": sig.symbol,
                    "why_watchlist": reasons.get(sig.symbol, "-"),
                    "last": sig.last,
                    "trend_ok": sig.trend_ok,
                    "breakout55": sig.breakout55,
                    "relvol": sig.relvol if sig.relvol is not None else "-",
                    "dist10": sig.dist10,
                    "action": action,
                    "spread": "-" if sugg is None else f"{sugg.expiry} {sugg.long_strike}/{sugg.short_strike} debit~{sugg.debit} rr~{sugg.rr}",
                    "notes": "-" if sugg is None else sugg.notes,
                    "reason": reason,
                }
            )
        except Exception as exc:
            rows.append({"symbol": sym, "action": "ERROR", "reason": str(exc)})

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_symbols": max_symbols,
        "plan_size": len(rows),
    }
    return rows, meta


def execute_debit_spread(trading_client: Any, a: dict[str, Any], sugg: DebitSpreadSuggestion, qty: float = 1.0) -> Any:
    order_side = a["OrderSide"]
    time_in_force = a["TimeInForce"]
    order_class = a["OrderClass"]
    market_order_request = a["MarketOrderRequest"]
    option_leg_request = a["OptionLegRequest"]
    pos_intent = a["PositionIntent"]

    legs = [
        option_leg_request(symbol=sugg.long_contract, ratio_qty=1, side=order_side.BUY, position_intent=pos_intent.BUY_TO_OPEN),
        option_leg_request(symbol=sugg.short_contract, ratio_qty=1, side=order_side.SELL, position_intent=pos_intent.SELL_TO_OPEN),
    ]

    req = market_order_request(
        symbol=sugg.symbol,
        qty=qty,
        side=order_side.BUY,
        time_in_force=time_in_force.DAY,
        order_class=order_class.MLEG,
        legs=legs,
    )
    return trading_client.submit_order(req)


def _read_command(timeout_seconds: float = 0.0) -> str | None:
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


def _print_help() -> None:
    print("\nCommands: buy <TICKER> [QTY] | sell <TICKER|OPTION_SYMBOL> [QTY|all] | research <TICKER> | holdings | plan [N] | help | quit")


def _close_option_positions_for_underlying(trading_client: Any, a: dict[str, Any], underlying: str) -> None:
    order_side = a["OrderSide"]
    time_in_force = a["TimeInForce"]
    market_order_request = a["MarketOrderRequest"]
    pos_intent = a["PositionIntent"]

    positions = trading_client.get_all_positions()
    to_close = [p for p in positions if _occ_underlying(p.symbol) == underlying.upper() and any(ch.isdigit() for ch in p.symbol)]
    if not to_close:
        print(f"No option positions found for underlying {underlying}.")
        return

    for pos in to_close:
        qty = float(getattr(pos, "qty", 0.0))
        if qty == 0:
            continue
        if qty > 0:
            side = order_side.SELL
            intent = pos_intent.SELL_TO_CLOSE
            close_qty = qty
        else:
            side = order_side.BUY
            intent = pos_intent.BUY_TO_CLOSE
            close_qty = abs(qty)

        req = market_order_request(
            symbol=pos.symbol,
            qty=close_qty,
            side=side,
            time_in_force=time_in_force.DAY,
            position_intent=intent,
        )
        trading_client.submit_order(req)
        print(f"Submitted close for {pos.symbol} qty={close_qty}")


def run_research(symbol: str, dte_min: int, dte_max: int, target_dte: int, width: float, min_oi: int) -> None:
    sig, sugg = suggest_call_debit_spread(
        symbol,
        dte_min=dte_min,
        dte_max=dte_max,
        target_dte=target_dte,
        width=width,
        min_oi=min_oi,
    )
    _print_table(
        [
            {
                "symbol": sig.symbol,
                "last": sig.last,
                "sma50": sig.sma50,
                "sma200": sig.sma200,
                "trend_ok": sig.trend_ok,
                "breakout55": sig.breakout55,
                "relvol": sig.relvol if sig.relvol is not None else "-",
                "dist10": sig.dist10,
            }
        ],
        title="Underlying Signal (Breakout55)",
    )
    if sugg is None:
        _print_table([{"symbol": sig.symbol, "result": "No suggestion (chain/pricing missing)."}], title="Option Suggestion")
        return

    _print_table(
        [
            {
                "symbol": sugg.symbol,
                "expiry": sugg.expiry,
                "long": f"{sugg.long_contract} ({sugg.long_strike})",
                "short": f"{sugg.short_contract} ({sugg.short_strike})",
                "debit": sugg.debit,
                "max_profit": sugg.max_profit,
                "max_loss": sugg.max_loss,
                "breakeven": sugg.breakeven,
                "rr": sugg.rr,
                "notes": sugg.notes,
            }
        ],
        title="Call Debit Spread Suggestion",
    )


def run_plan_mode(max_symbols: int, dte_min: int, dte_max: int, target_dte: int, width: float, min_oi: int) -> None:
    _a, trading_client = _build_clients()
    watch, reasons = _build_underlying_watchlist(max_symbols=max_symbols)
    _print_table([{"symbol": s, "why": reasons.get(s, "-")} for s in watch], title=f"Watchlist (Options) top={len(watch)}")
    _print_table(_current_holdings_rows(trading_client), title="Current Holdings")
    rows, meta = build_option_plan(max_symbols, dte_min, dte_max, target_dte, width, min_oi, trading_client)
    _print_table(rows, title="Option Plan")
    _print_table([meta], title="Plan Meta")


def run_auto_mode(
    max_symbols: int,
    dte_min: int,
    dte_max: int,
    target_dte: int,
    width: float,
    min_oi: int,
    loop_seconds: int,
    execute: bool,
    refresh_screen: bool,
) -> None:
    a, trading_client = _build_clients()
    last_triggered_market_day: str | None = None
    last_run_summary: dict[str, Any] | None = None

    print("Live commands enabled. Type 'help' and press Enter.")

    while True:
        now = datetime.now(timezone.utc)
        clock = trading_client.get_clock()
        trigger_time_utc = _market_trigger_time_utc(now)
        market_day = now.astimezone(MARKET_TZ).date().isoformat()
        triggered_today = last_triggered_market_day == market_day
        did_trigger_now = False

        if refresh_screen and clock.is_open:
            _clear_screen()

        _print_table(
            [
                {
                    "timestamp_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "is_open": bool(clock.is_open),
                    "next_open": str(clock.next_open),
                    "next_close": str(clock.next_close),
                }
            ],
            title="Market Status",
        )

        watch, reasons = _build_underlying_watchlist(max_symbols=max_symbols)
        _print_table([{"symbol": s, "why": reasons.get(s, "-")} for s in watch], title=f"Watchlist (Options) top={len(watch)}")
        _print_table(_current_holdings_rows(trading_client), title="Current Holdings")

        if clock.is_open and now >= trigger_time_utc and not triggered_today:
            plan_rows, meta = build_option_plan(max_symbols, dte_min, dte_max, target_dte, width, min_oi, trading_client)
            _print_table(plan_rows, title="Triggered Option Plan")
            _print_table([meta], title="Plan Meta")

            if execute:
                for row in plan_rows:
                    if row.get("action") != "BUY_SPREAD":
                        continue
                    sym = str(row.get("symbol"))
                    try:
                        _sig, sugg = suggest_call_debit_spread(sym, dte_min=dte_min, dte_max=dte_max, target_dte=target_dte, width=width, min_oi=min_oi)
                        if sugg is None:
                            continue
                        order = execute_debit_spread(trading_client, a, sugg, qty=1.0)
                        print(f"ORDER SUBMITTED: BUY_SPREAD {sym} id={getattr(order, 'id', None)}")
                    except Exception as exc:
                        print(f"Execute failed for {sym}: {exc}")

            last_triggered_market_day = market_day
            buy_count = sum(1 for r in plan_rows if r.get("action") == "BUY_SPREAD")
            hold_count = sum(1 for r in plan_rows if r.get("action") == "HOLD")
            watch_count = sum(1 for r in plan_rows if r.get("action") == "WATCH")
            last_run_summary = {
                "triggered_market_day": market_day,
                "triggered_at_utc": now.isoformat(),
                "buy_spread": buy_count,
                "hold": hold_count,
                "watch": watch_count,
            }
            triggered_today = True
            did_trigger_now = True

        next_action = "WAIT"
        if did_trigger_now:
            next_action = "TRIGGERED_NOW"
        elif triggered_today:
            next_action = "TRIGGERED_TODAY"
        elif not clock.is_open:
            next_action = "WAIT_MARKET_OPEN"
        elif now < trigger_time_utc:
            next_action = "WAIT_TRIGGER_TIME"

        _print_table(
            [
                {
                    "market_day_et": market_day,
                    "trigger_time_et": trigger_time_utc.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "triggered_today": triggered_today,
                    "next_action": next_action,
                }
            ],
            title="Auto Status",
        )
        if last_run_summary:
            _print_table([last_run_summary], title="Last Trigger")

        sys.stdout.flush()

        # Live command loop for up to loop_seconds (responds quickly).
        deadline = time.time() + max(1, loop_seconds)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            cmdline = _read_command(timeout_seconds=min(0.25, remaining))
            if cmdline is None:
                continue
            cmdline = cmdline.strip()
            if not cmdline:
                continue
            parts = cmdline.split()
            cmd = parts[0].lower()

            try:
                if cmd in {"help", "?"}:
                    _print_help()
                elif cmd in {"quit", "exit", "q"}:
                    print("Stopping auto mode by user request.")
                    return
                elif cmd == "holdings":
                    _print_table(_current_holdings_rows(trading_client), title="Current Holdings")
                elif cmd == "plan":
                    n = max_symbols
                    if len(parts) >= 2:
                        n = max(1, int(parts[1]))
                    plan_rows, meta = build_option_plan(n, dte_min, dte_max, target_dte, width, min_oi, trading_client)
                    _print_table(plan_rows, title="Live Option Plan")
                    _print_table([meta], title="Plan Meta")
                elif cmd == "research":
                    if len(parts) < 2:
                        print("Usage: research <TICKER>")
                    else:
                        run_research(parts[1].upper(), dte_min, dte_max, target_dte, width, min_oi)
                elif cmd == "buy":
                    if len(parts) < 2:
                        print("Usage: buy <TICKER> [QTY]")
                    else:
                        sym = parts[1].upper()
                        qty = float(parts[2]) if len(parts) >= 3 else 1.0
                        _sig, sugg = suggest_call_debit_spread(sym, dte_min=dte_min, dte_max=dte_max, target_dte=target_dte, width=width, min_oi=min_oi)
                        if sugg is None:
                            print(f"No suggestion available for {sym}.")
                        elif not execute:
                            print("Execute is disabled. Re-run with --execute to place orders.")
                        else:
                            order = execute_debit_spread(trading_client, a, sugg, qty=qty)
                            print(f"LIVE ORDER SUBMITTED: BUY_SPREAD {sym} qty={qty} id={getattr(order, 'id', None)}")
                elif cmd == "sell":
                    if len(parts) < 2:
                        print("Usage: sell <TICKER|OPTION_SYMBOL> [QTY|all]")
                    else:
                        target = parts[1].upper()
                        if any(ch.isdigit() for ch in target) and len(target) > 8:
                            # Close single contract position (if exists) by looking at open positions.
                            positions = {p.symbol: p for p in trading_client.get_all_positions()}
                            pos = positions.get(target)
                            if pos is None:
                                print(f"No open position for {target}.")
                            else:
                                qty = float(getattr(pos, "qty", 0.0))
                                if len(parts) >= 3 and parts[2].lower() != "all":
                                    close_qty = float(parts[2])
                                else:
                                    close_qty = abs(qty)
                                if qty > 0:
                                    side = a["OrderSide"].SELL
                                    intent = a["PositionIntent"].SELL_TO_CLOSE
                                else:
                                    side = a["OrderSide"].BUY
                                    intent = a["PositionIntent"].BUY_TO_CLOSE
                                req = a["MarketOrderRequest"](
                                    symbol=target,
                                    qty=close_qty,
                                    side=side,
                                    time_in_force=a["TimeInForce"].DAY,
                                    position_intent=intent,
                                )
                                trading_client.submit_order(req)
                                print(f"LIVE ORDER SUBMITTED: CLOSE {target} qty={close_qty}")
                        else:
                            if not execute:
                                print("Execute is disabled. Re-run with --execute to place orders.")
                            else:
                                _close_option_positions_for_underlying(trading_client, a, target)
                else:
                    print(f"Unknown command: {cmdline}")
            except Exception as exc:
                print(f"Command failed: {exc}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Options swing dashboard (Breakout55, 15-30 day holds)")
    p.add_argument("--mode", choices=["plan", "auto", "research"], default="auto")
    p.add_argument("--symbol", default="AAPL", help="Symbol for research mode")
    p.add_argument("--max-symbols", type=int, default=DEFAULT_MAX_SYMBOLS)
    p.add_argument("--loop-seconds", type=int, default=60, help="Auto mode loop interval")
    p.add_argument("--execute", action="store_true", help="Enable order submission (buy/sell/auto)")
    p.add_argument("--no-refresh", action="store_true", help="Disable screen refresh")
    p.add_argument("--dte-min", type=int, default=30)
    p.add_argument("--dte-max", type=int, default=75)
    p.add_argument("--target-dte", type=int, default=45)
    p.add_argument("--width", type=float, default=5.0)
    p.add_argument("--min-oi", type=int, default=100)
    return p


def main() -> None:
    args = _parser().parse_args()
    refresh_screen = _supports_screen_refresh() and not args.no_refresh

    if args.mode == "research":
        run_research(args.symbol.upper(), args.dte_min, args.dte_max, args.target_dte, args.width, args.min_oi)
        return
    if args.mode == "plan":
        run_plan_mode(args.max_symbols, args.dte_min, args.dte_max, args.target_dte, args.width, args.min_oi)
        return

    run_auto_mode(
        max_symbols=args.max_symbols,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        target_dte=args.target_dte,
        width=args.width,
        min_oi=args.min_oi,
        loop_seconds=args.loop_seconds,
        execute=args.execute,
        refresh_screen=refresh_screen,
    )


if __name__ == "__main__":
    main()

