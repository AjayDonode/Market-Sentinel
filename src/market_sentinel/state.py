"""
state.py — Background refresh cache for Market Sentinel dashboard.

A singleton MarketState object holds the latest fetched data.
A background thread calls refresh() on a configurable interval so the
Streamlit UI never blocks on network I/O and always reads from memory.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .engine import (
    compute_signal_snapshot,
    decide_action,
    fetch_bars_yfinance,
    get_sector_report,
    get_spy_above_sma200,
    load_trade_logs,
)
from .analysis import run_sector_report, suggest_stocks

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

LOGS_DIR = str(Path(__file__).resolve().parents[2] / "logs")


def _alpaca_clients() -> tuple[Any, Any] | tuple[None, None]:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        key = os.getenv("ALPACA_KEY", "")
        secret = os.getenv("ALPACA_SECRET", "")
        if not key or not secret:
            return None, None
        is_paper = os.getenv("ALPACA_PAPER", "true").lower() in {"1", "true", "yes"}
        trading = TradingClient(key, secret, paper=is_paper)
        data = StockHistoricalDataClient(key, secret)
        return trading, data
    except Exception:
        return None, None


def _get_holdings(trading_client: Any) -> list[dict[str, Any]]:
    if trading_client is None:
        return []
    try:
        rows = []
        for pos in trading_client.get_all_positions():
            rows.append({
                "symbol": pos.symbol,
                "qty": round(float(getattr(pos, "qty", 0)), 4),
                "avg_entry": round(float(getattr(pos, "avg_entry_price", 0)), 2),
                "market_value": round(float(getattr(pos, "market_value", 0)), 2),
                "unrealized_pl": round(float(getattr(pos, "unrealized_pl", 0)), 2),
                "unrealized_plpc": round(float(getattr(pos, "unrealized_plpc", 0)) * 100, 2),
            })
        return sorted(rows, key=lambda r: r["symbol"])
    except Exception:
        return []


def _get_account(trading_client: Any) -> dict[str, Any]:
    if trading_client is None:
        return {}
    try:
        acct = trading_client.get_account()
        return {
            "equity": round(float(getattr(acct, "equity", 0)), 2),
            "cash": round(float(getattr(acct, "cash", 0)), 2),
            "buying_power": round(float(getattr(acct, "buying_power", 0)), 2),
        }
    except Exception:
        return {}


def _get_clock(trading_client: Any) -> dict[str, Any]:
    if trading_client is None:
        return {"is_open": None, "next_open": "unknown", "next_close": "unknown"}
    try:
        clock = trading_client.get_clock()
        return {
            "is_open": bool(clock.is_open),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }
    except Exception:
        return {"is_open": None, "next_open": "unknown", "next_close": "unknown"}


def _get_position_map(holdings: list[dict]) -> dict[str, dict]:
    return {h["symbol"]: h for h in holdings}


class MarketState:
    """Singleton holding all live data for the dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Public fields — read from dashboard thread
        self.sector_df: pd.DataFrame = pd.DataFrame()
        self.trade_plan: list[dict[str, Any]] = []
        self.holdings: list[dict[str, Any]] = []
        self.account: dict[str, Any] = {}
        self.clock: dict[str, Any] = {"is_open": None, "next_open": "", "next_close": ""}
        self.symbol_snapshots: dict[str, dict[str, Any]] = {}
        self.trade_logs: pd.DataFrame = pd.DataFrame()
        self.last_updated: datetime | None = None
        self.is_refreshing: bool = False
        self.last_error: str = ""
        # Config (set from UI)
        self.entry_style: str = "macd"
        self.atr_mult: float = 2.5
        self.max_symbols: int = 15

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_watchlist(self) -> list[str]:
        try:
            strong, _, _ = run_sector_report(top_strong=3, top_weak=2)
            if strong.empty:
                return ["AAPL", "MSFT", "NVDA"][:self.max_symbols]
            picks = suggest_stocks(strong["sector"].tolist(), top_n=self.max_symbols)
            if picks.empty:
                return ["AAPL", "MSFT", "NVDA"][:self.max_symbols]
            return picks["symbol"].head(self.max_symbols).tolist()
        except Exception:
            return ["AAPL", "MSFT", "NVDA", "XOM", "JPM"][:self.max_symbols]

    def _build_plan_row(
        self,
        snap: dict[str, Any],
        positions: dict[str, dict],
        entry_style: str,
        atr_mult: float,
        market_gate: bool | None,
    ) -> dict[str, Any]:
        symbol = snap["symbol"]
        pos = positions.get(symbol)
        qty = float(pos["qty"]) if pos else 0.0
        days_held = None
        if pos:
            try:
                trading_client, _ = _alpaca_clients()
                if trading_client:
                    from alpaca.trading.enums import OrderSide, QueryOrderStatus
                    from alpaca.trading.requests import GetOrdersRequest
                    status = getattr(QueryOrderStatus, "FILLED", None) or getattr(QueryOrderStatus, "CLOSED", None)
                    order_params = GetOrdersRequest(status=status, symbols=[symbol], side=OrderSide.BUY, limit=10)
                    orders = trading_client.get_orders(order_params)
                    filled = [o for o in orders if getattr(o, "filled_at", None)]
                    if filled:
                        latest = max(filled, key=lambda o: o.filled_at)
                        days_held = (datetime.now(latest.filled_at.tzinfo) - latest.filled_at).days
            except Exception:
                pass

        snap_with_gate = dict(snap)
        snap_with_gate["market_gate"] = market_gate
        action, reason = decide_action(snap_with_gate, entry_style, qty, days_held, atr_mult)

        return {
            "symbol": symbol,
            "last": snap["last"],
            "rsi": snap["rsi"],
            "macd": snap["macd"],
            "macd_signal": snap["macd_signal"],
            "bb_pct_b": snap["bb_pct_b"],
            "bb_lower": snap["bb_lower"],
            "bb_upper": snap["bb_upper"],
            "atr": snap["atr"],
            "trend_ok": snap["trend_ok"],
            "breakout55": snap["breakout55"],
            "relvol": snap["relvol"],
            "dist10": snap["dist10"],
            "position_qty": round(qty, 4),
            "days_held": days_held if days_held is not None else "-",
            "action": action,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Public: refresh all state
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        with self._lock:
            if self.is_refreshing:
                return
            self.is_refreshing = True

        try:
            trading_client, _ = _alpaca_clients()

            # Market clock & account
            clock = _get_clock(trading_client)
            account = _get_account(trading_client)
            holdings = _get_holdings(trading_client)
            positions = _get_position_map(holdings)

            # Sector data
            try:
                sector_df = get_sector_report()
            except Exception:
                sector_df = pd.DataFrame()

            # Market gate (SPY vs SMA200)
            try:
                market_gate = get_spy_above_sma200()
            except Exception:
                market_gate = None

            # Build watchlist + fetch bars
            symbols = self._build_watchlist()
            snapshots: dict[str, dict] = {}
            plan_rows: list[dict] = []

            for sym in symbols:
                try:
                    bars = fetch_bars_yfinance(sym)
                    snap = compute_signal_snapshot(sym, bars, self.atr_mult)
                    snapshots[sym] = snap
                    plan_row = self._build_plan_row(
                        snap, positions, self.entry_style, self.atr_mult, market_gate
                    )
                    plan_rows.append(plan_row)
                except Exception as exc:
                    plan_rows.append({
                        "symbol": sym,
                        "last": "-", "rsi": "-", "macd": "-",
                        "macd_signal": "-", "bb_pct_b": "-",
                        "bb_lower": "-", "bb_upper": "-",
                        "atr": "-", "trend_ok": "-",
                        "breakout55": "-", "relvol": "-", "dist10": "-",
                        "position_qty": 0, "days_held": "-",
                        "action": "ERROR", "reason": str(exc),
                    })

            # Trade log history
            try:
                trade_logs = load_trade_logs(LOGS_DIR)
            except Exception:
                trade_logs = pd.DataFrame()

            with self._lock:
                self.clock = clock
                self.account = account
                self.holdings = holdings
                self.sector_df = sector_df
                self.trade_plan = plan_rows
                self.symbol_snapshots = snapshots
                self.trade_logs = trade_logs
                self.last_updated = datetime.now(timezone.utc)
                self.last_error = ""

        except Exception as exc:
            with self._lock:
                self.last_error = str(exc)
        finally:
            with self._lock:
                self.is_refreshing = False

    # ------------------------------------------------------------------
    # Public: read snapshots safely
    # ------------------------------------------------------------------

    def get_snapshot(self, symbol: str) -> dict[str, Any] | None:
        with self._lock:
            return self.symbol_snapshots.get(symbol.upper())

    def get_plan_df(self) -> pd.DataFrame:
        with self._lock:
            if not self.trade_plan:
                return pd.DataFrame()
            return pd.DataFrame(self.trade_plan)

    def get_holdings_df(self) -> pd.DataFrame:
        with self._lock:
            if not self.holdings:
                return pd.DataFrame()
            return pd.DataFrame(self.holdings)

    # ------------------------------------------------------------------
    # Auto-refresh loop (runs in background thread)
    # ------------------------------------------------------------------

    def start_background_refresh(self, interval_seconds: int = 300) -> None:
        def _loop() -> None:
            import time
            while True:
                try:
                    self.refresh()
                except Exception:
                    pass
                time.sleep(interval_seconds)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()


# Module-level singleton — Streamlit imports this
_state: MarketState | None = None


def get_state() -> MarketState:
    global _state
    if _state is None:
        _state = MarketState()
    return _state
