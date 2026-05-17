"""
dashboard.py — Market Sentinel Streamlit Web UI

Run with:
    streamlit run src/market_sentinel/dashboard.py
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, date as date_type, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── path setup ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from market_sentinel.state import get_state, MarketState

DASHBOARD_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "logs" / "dashboard_ui_settings.json"


def _load_dashboard_settings() -> dict[str, Any]:
    try:
        if DASHBOARD_SETTINGS_PATH.exists():
            return json.loads(DASHBOARD_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "auto_execute": False,
        "entry_style": "hybrid",
        "atr_mult": 2.5,
        "max_symbols": 15,
        "refresh_interval": "300s",
    }


def _save_dashboard_settings(settings: dict[str, Any]) -> None:
    try:
        DASHBOARD_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DASHBOARD_SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception:
        pass

# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentinel",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark glassmorphism background */
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%); }

/* KPI cards */
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.kpi-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.kpi-value { color: #e6edf3; font-size: 24px; font-weight: 700; }
.kpi-value.green { color: #3fb950; }
.kpi-value.red   { color: #f85149; }
.kpi-value.blue  { color: #58a6ff; }
.kpi-value.yellow{ color: #d29922; }

/* Sector heat boxes */
.sector-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
.sector-box {
    border-radius: 8px; padding: 8px 12px;
    font-size: 12px; font-weight: 600;
    cursor: pointer; border: 1px solid transparent;
    transition: transform 0.15s;
}
.sector-box:hover { transform: scale(1.04); }

/* Section headers */
.section-header {
    font-size: 13px; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: 1.5px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 6px; margin: 20px 0 12px 0;
}

/* Action badges */
.badge { padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 700; }
.badge-buy  { background: rgba(63,185,80,0.2);  color: #3fb950; border: 1px solid #3fb950; }
.badge-sell { background: rgba(248,81,73,0.2);  color: #f85149; border: 1px solid #f85149; }
.badge-hold { background: rgba(139,148,158,0.2);color: #8b949e; border: 1px solid #8b949e; }
.badge-error{ background: rgba(210,153,34,0.2); color: #d29922; border: 1px solid #d29922; }

/* Refresh pulse */
@keyframes pulse { 0%{opacity:1} 50%{opacity:0.4} 100%{opacity:1} }
.refreshing { animation: pulse 1.2s infinite; color: #58a6ff; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

/* ── Scanner table ── */
.scan-th {
    font-size: 10px; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.8px;
    padding: 0 2px 6px; border-bottom: 1px solid rgba(255,255,255,0.08);
}
.scan-cell { font-size: 12px; color: #c9d1d9; padding: 3px 2px; }
.scan-cell.muted { color: #8b949e; font-size: 11px; }
.scan-sep { border-bottom: 1px solid rgba(255,255,255,0.04); margin: 2px 0; }

/* Action strip (selected row) */
.action-strip {
    display: flex; align-items: center; gap: 12px;
    background: rgba(88,166,255,0.06);
    border: 1px solid rgba(88,166,255,0.18);
    border-radius: 10px; padding: 10px 16px; margin: 8px 0 4px;
}

/* Score bar */
.score-bar-bg {
    background: rgba(255,255,255,0.07); border-radius: 4px;
    height: 6px; flex: 1; overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 4px; }

/* Freshness bar */
.fresh-bar {
    height: 3px; border-radius: 2px; margin: 4px 0 10px; width: 100%;
}

/* Sector pills override — tighter padding */
div[data-testid="stPills"] label { font-size: 11px !important; }

/* ── Compact scanner row layout ── */
/* Tighten column horizontal padding */
.scanner-grid [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    padding-left: 2px !important; padding-right: 2px !important;
}
/* Align column children to center vertically */
.scanner-grid [data-testid="stHorizontalBlock"] {
    align-items: center !important; gap: 0 !important;
}
/* Shrink markdown paragraph margins to zero */
.scanner-grid [data-testid="stMarkdownContainer"] p {
    margin: 0 !important; line-height: 1.3 !important;
}
/* Compact buttons — reduce min height & padding */
.scanner-grid button {
    min-height: 28px !important;
    height: 28px !important;
    padding: 0 6px !important;
    font-size: 11px !important;
    line-height: 1 !important;
}
.scanner-grid button p {
    font-size: 11px !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ────────────────────────────────────────────────────────────────────────────
if "state_initialized" not in st.session_state:
    st.session_state.state_initialized = False
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None
if "confirm_trade" not in st.session_state:
    st.session_state.confirm_trade = None  # {action, symbol, qty}
if "selected_sector" not in st.session_state:
    st.session_state.selected_sector = None
if "show_raw_indicators" not in st.session_state:
    st.session_state.show_raw_indicators = False
if "opt_analysis_cache" not in st.session_state:
    st.session_state.opt_analysis_cache = None   # list of strategy dicts
if "opt_analysis_ts" not in st.session_state:
    st.session_state.opt_analysis_ts = None      # datetime of last analysis
if "dashboard_settings" not in st.session_state:
    st.session_state.dashboard_settings = _load_dashboard_settings()
if "auto_execute" not in st.session_state:
    st.session_state.auto_execute = st.session_state.dashboard_settings.get("auto_execute", False)

ms: MarketState = get_state()

# First load — start background refresh thread
if not st.session_state.state_initialized:
    ms.start_background_refresh(interval_seconds=300)
    ms.refresh()  # synchronous first load
    st.session_state.state_initialized = True


# ────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.divider()

    st.markdown(
        "<span style='color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px'>Scanner Settings</span>",
        unsafe_allow_html=True,
    )
    entry_style = st.selectbox(
        "Strategy",
        ["hybrid", "macd", "bollinger", "breakout55", "rsi30"],
        index=0,
        help="Entry algorithm for BUY/SELL signals in the Live Scanner and Ticker Analysis tabs",
    )

    atr_mult = st.slider(
        "ATR Trailing Stop Multiplier",
        min_value=1.0,
        max_value=5.0,
        value=2.5,
        step=0.25,
        help="Higher = wider stop, lets winners run longer",
    )

    max_symbols = st.slider(
        "Scanner Watchlist Size",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
        help="Number of symbols to monitor in the Live Scanner tab",
    )

    st.divider()
    st.markdown("### 🚦 Execution Mode")
    auto_execute = st.toggle(
        "Auto-Execute Trades",
        value=st.session_state.auto_execute,
        help="When ON, BUY/SELL signals are submitted directly.\nWhen OFF, you must confirm each trade manually.",
        key="auto_execute",
    )
    if auto_execute:
        st.warning("⚠️ Auto-execute is **ON**. Trades will be submitted without confirmation.")
    else:
        st.success("✅ Manual mode — you'll confirm every trade before execution.")

    if auto_execute != st.session_state.dashboard_settings.get("auto_execute", False):
        st.session_state.dashboard_settings["auto_execute"] = auto_execute
        _save_dashboard_settings(st.session_state.dashboard_settings)

    st.divider()
    refresh_interval = st.selectbox(
        "Auto Refresh",
        ["60s", "120s", "300s", "600s", "Off"],
        index=2,
    )
    refresh_now = st.button("🔄 Refresh Now", use_container_width=True)

    # Apply config changes to state
    ms.entry_style = entry_style
    ms.atr_mult = atr_mult
    ms.max_symbols = max_symbols

    if refresh_now:
        with st.spinner("Fetching live data…"):
            ms.refresh()

    if ms.last_updated:
        age = (datetime.now(timezone.utc) - ms.last_updated).seconds
        if ms.is_refreshing:
            fresh_color, fresh_label = "#58a6ff", "🔄 Refreshing…"
        elif age < 60:
            fresh_color, fresh_label = "#3fb950", f"✅ Live · {age}s ago"
        elif age < 180:
            fresh_color, fresh_label = "#d29922", f"⚠️ Stale · {age}s ago"
        else:
            fresh_color, fresh_label = "#f85149", f"🔴 Old · {age}s ago"
        bar_w = max(5, min(100, int(100 - age / 3)))
        st.markdown(
            f"<div class='fresh-bar' style='background:rgba(255,255,255,0.07)'>"
            f"<div style='width:{bar_w}%;height:100%;background:{fresh_color};border-radius:2px;transition:width 1s'></div></div>"
            f"<span style='font-size:11px;color:{fresh_color}'>{fresh_label}</span>",
            unsafe_allow_html=True,
        )
    if ms.last_error:
        st.error(f"Error: {ms.last_error}")

    st.divider()
    st.markdown(
        "<center><span style='color:#484f57;font-size:10px'>Market Sentinel · Paper Trading Mode"
        "<br>For informational purposes only · Not financial advice</span></center>",
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Auto-refresh timer (non-blocking)
# ────────────────────────────────────────────────────────────────────────────
if refresh_interval != "Off":
    interval_s = int(refresh_interval.replace("s", ""))
    st_autorefresh_key = f"autorefresh_{interval_s}"
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore
        st_autorefresh(interval=interval_s * 1000, key=st_autorefresh_key)
    except ImportError:
        pass  # Optional — works fine without it


# ────────────────────────────────────────────────────────────────────────────
# Helper: execute a trade via Alpaca
# ────────────────────────────────────────────────────────────────────────────
def _submit_order(symbol: str, action: str, qty: float) -> str:
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        key = os.getenv("ALPACA_KEY", "")
        secret = os.getenv("ALPACA_SECRET", "")
        if not key or not secret:
            return "❌ Alpaca credentials not set in .env"
        is_paper = os.getenv("ALPACA_PAPER", "true").lower() in {"1", "true", "yes"}
        client = TradingClient(key, secret, paper=is_paper)

        side = OrderSide.BUY if action.upper() == "BUY" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return f"✅ Order submitted: {action} {qty} {symbol} (id={getattr(order, 'id', '?')})"
    except Exception as exc:
        return f"❌ Order failed: {exc}"


# ────────────────────────────────────────────────────────────────────────────
# Page Header
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#e6edf3;font-size:28px;font-weight:700;margin-bottom:4px'>📡 Market Sentinel</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<span style='color:#8b949e;font-size:13px'>Strategy: <b style='color:#58a6ff'>{entry_style.upper()}</b>"
    f" &nbsp;|&nbsp; ATR Mult: <b style='color:#58a6ff'>{atr_mult}x</b>"
    f" &nbsp;|&nbsp; Watchlist: <b style='color:#58a6ff'>{max_symbols}</b> symbols</span>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

tab_scanner, tab_analyze, tab_options = st.tabs(
    ["📡 Live Scanner", "🔬 Ticker Analysis", "📜 Options Strategy"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: Live Scanner
# ════════════════════════════════════════════════════════════════════════════
with tab_scanner:

    # ── Confirm Trade Dialog (scoped inside scanner tab) ───────────────────
    if st.session_state.confirm_trade and not auto_execute:
        ct = st.session_state.confirm_trade
        action_color = "🟢" if ct["action"] == "BUY" else "🔴"
        with st.container():
            st.markdown(f"### {action_color} Confirm Trade")
            st.write(f"**{ct['action']} {ct['qty']} share(s) of {ct['symbol']}** at market price.")
            c1, c2, _ = st.columns([1, 1, 5])
            if c1.button("✅ Confirm", key="confirm_yes", type="primary"):
                result = _submit_order(ct["symbol"], ct["action"], ct["qty"])
                st.success(result)
                st.session_state.confirm_trade = None
                ms.refresh()
            if c2.button("❌ Cancel", key="confirm_no"):
                st.session_state.confirm_trade = None
            st.divider()

    # ── Section 1: KPI Header Bar (7 cards) ───────────────────────────────
    clock = ms.clock
    account = ms.account

    market_open = clock.get("is_open")
    if market_open is True:
        market_label = "🟢 OPEN"
        market_cls = "green"
    elif market_open is False:
        market_label = "🔴 CLOSED"
        market_cls = "red"
    else:
        market_label = "⚪ UNKNOWN"
        market_cls = "yellow"

    equity        = account.get("equity", 0.0)
    cash          = account.get("cash", 0.0)
    buying_power  = account.get("buying_power", 0.0)

    holdings_df = ms.get_holdings_df()
    total_upl = float(holdings_df["unrealized_pl"].sum()) if not holdings_df.empty and "unrealized_pl" in holdings_df.columns else 0.0
    upl_cls  = "green" if total_upl >= 0 else "red"
    upl_sign = "+" if total_upl >= 0 else ""

    plan_df = ms.get_plan_df()
    buy_signals  = int((plan_df["action"] == "BUY").sum())  if not plan_df.empty and "action" in plan_df.columns else 0
    sell_signals = int((plan_df["action"] == "SELL").sum()) if not plan_df.empty and "action" in plan_df.columns else 0

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Market</div><div class="kpi-value {market_cls}">{market_label}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-label">Portfolio Equity</div><div class="kpi-value blue">${equity:,.2f}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-label">Cash</div><div class="kpi-value blue">${cash:,.2f}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-label">Buying Power</div><div class="kpi-value blue">${buying_power:,.2f}</div></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="kpi-card"><div class="kpi-label">Unrealized P/L</div><div class="kpi-value {upl_cls}">{upl_sign}${total_upl:,.2f}</div></div>', unsafe_allow_html=True)
    k6.markdown(f'<div class="kpi-card"><div class="kpi-label">BUY Signals</div><div class="kpi-value green">{buy_signals}</div></div>', unsafe_allow_html=True)
    k7.markdown(f'<div class="kpi-card"><div class="kpi-label">SELL Signals</div><div class="kpi-value red">{sell_signals}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Sector Heatmap ──────────────────────────────────────────
    st.markdown('<div class="section-header">🗺️ Sector Heatmap (YTD)</div>', unsafe_allow_html=True)
    sector_df = ms.sector_df
    if not sector_df.empty:
        max_abs = max(abs(sector_df["ytd_return_pct"].max()), abs(sector_df["ytd_return_pct"].min()), 0.01)
        html_boxes = '<div class="sector-wrap">'
        for _, row in sector_df.iterrows():
            pct = float(row["ytd_return_pct"])
            intensity = min(abs(pct) / max_abs, 1.0)
            if pct >= 0:
                r, g, b = int(15 + 20*intensity), int(90 + 95*intensity), int(30 + 50*intensity)
            else:
                r, g, b = int(90 + 100*intensity), int(20 + 25*intensity), int(20 + 25*intensity)
            sign = "+" if pct >= 0 else ""
            selected = st.session_state.selected_sector == row["sector"]
            border = "2px solid #58a6ff" if selected else "1px solid transparent"
            html_boxes += (
                f'<div class="sector-box" style="background:rgba({r},{g},{b},0.6);border:{border}">'
                f'<b>{row["etf"]}</b><br>'
                f'<span style="font-size:9px;color:#ddd">{row["sector"][:14]}</span><br>'
                f'<span style="color:#fff">{sign}{pct:.1f}%</span>'
                f'</div>'
            )
        html_boxes += '</div>'
        st.markdown(html_boxes, unsafe_allow_html=True)

        # Clickable pills filter (replaces the hidden selectbox)
        pill_options = ["All"] + sector_df["etf"].tolist()
        pill_labels  = {
            "All": "🌐 All",
            **{row["etf"]: f'{row["etf"]}  {"+" if row["ytd_return_pct"]>=0 else ""}{row["ytd_return_pct"]:.1f}%'
               for _, row in sector_df.iterrows()}
        }
        selected_pill = st.pills(
            "Filter by sector",
            options=pill_options,
            format_func=lambda x: pill_labels.get(x, x),
            default="All",
            key="sector_pills",
            label_visibility="collapsed",
        )
        if selected_pill and selected_pill != "All":
            matched = sector_df[sector_df["etf"] == selected_pill]["sector"].values
            st.session_state.selected_sector = matched[0] if len(matched) else None
        else:
            st.session_state.selected_sector = None
    else:
        st.info("Sector data loading…")
    st.markdown("<br>", unsafe_allow_html=True)


    # ── Section 3: Signal Scanner ──────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Signal Scanner</div>', unsafe_allow_html=True)

    def _action_badge(action: str) -> str:
        a = str(action).upper()
        if a == "BUY":   return "🟢 BUY"
        if a == "SELL":  return "🔴 SELL"
        if a == "ERROR": return "🟡 ERROR"
        return "⚪ HOLD"

    def _queue_trade(action: str, sym: str, qty: float = 1.0) -> None:
        if auto_execute:
            result = _submit_order(sym, action, qty)
            st.toast(result)
            ms.refresh()
        else:
            st.session_state.confirm_trade = {"action": action, "symbol": sym, "qty": qty}
            st.rerun()

    def highlight_rows(row: pd.Series) -> list[str]:
        sig = str(row.get("Signal", ""))
        if "BUY" in sig:  return ["background-color: rgba(63,185,80,0.18)"] * len(row)
        if "SELL" in sig: return ["background-color: rgba(248,81,73,0.18)"] * len(row)
        return [""] * len(row)

    if not plan_df.empty:
        display_df = plan_df.copy()
        display_df["days_held"] = display_df["days_held"].astype(str)
        display_df["signal_score"] = display_df.get("signal_score", pd.Series([0] * len(display_df), index=display_df.index)).fillna(0).astype(int)
        display_df = display_df.sort_values(by=["signal_score", "action"], ascending=[False, True])
        display_df["Signal"] = display_df["action"].apply(_action_badge)

        show_raw = st.toggle(
            "🔬 Show All Indicators",
            value=st.session_state.show_raw_indicators,
            key="raw_toggle",
        )
        st.session_state.show_raw_indicators = show_raw

        # Column widths & headers — last 2 are always Buy / Sell
        if show_raw:
            _cw = [0.9, 0.65, 0.35, 0.45, 0.45, 0.45, 0.45, 0.5, 0.5, 0.8, 2.2, 0.45, 0.45]
            _hds = ["Symbol","Price","Score","RSI","MACD","%B","ATR","Trend","RelVol","Signal","Reason","Buy","Sell"]
        else:
            _cw = [0.9, 0.65, 0.35, 0.45, 0.45, 0.5, 0.8, 2.2, 0.45, 0.45, 0.45]
            _hds = ["Symbol","Price","Score","RSI","%B","ATR","Trend","Signal","Reason","Buy","Sell"]

        # Helper
        def _fmt(v, fmt=".2f"):
            try: return f"{float(v):{fmt}}"
            except: return str(v)

        # Open scanner-grid scope div so CSS targets only this section
        st.markdown('<div class="scanner-grid">', unsafe_allow_html=True)

        # Header row
        hc = st.columns(_cw)
        for col, hd in zip(hc, _hds):
            col.markdown(f"<span class='scan-th'>{hd}</span>", unsafe_allow_html=True)

        # Divider
        st.markdown("<hr style='margin:2px 0 4px;border-color:rgba(255,255,255,0.08)'>", unsafe_allow_html=True)

        # Data rows
        for _ri, (_, dr) in enumerate(display_df.iterrows()):
            sym = str(dr.get("symbol",""))
            act = str(dr.get("action","")).upper()
            is_sel = st.session_state.selected_symbol == sym

            # Row background tint via a 0-height div trick
            if act == "BUY":   _bg = "rgba(63,185,80,0.10)"; _bl = "3px solid rgba(63,185,80,0.55)"
            elif act == "SELL": _bg = "rgba(248,81,73,0.10)"; _bl = "3px solid rgba(248,81,73,0.55)"
            else:               _bg = "transparent";           _bl = "3px solid transparent"
            _sel_outline = "outline:1px solid rgba(88,166,255,0.5);outline-offset:-1px;" if is_sel else ""
            st.markdown(
                f"<div style='background:{_bg};border-left:{_bl};border-radius:5px;"
                f"margin-bottom:1px;height:2px;{_sel_outline}'></div>",
                unsafe_allow_html=True,
            )

            rc = st.columns(_cw)

            # ── Symbol (click = select for chart) ─────────────────────────
            sym_lbl = f"**{sym}**" if is_sel else sym
            if rc[0].button(sym_lbl, key=f"rs_{sym}_{_ri}", use_container_width=True, help="Click to view chart"):
                st.session_state.selected_symbol = sym
                st.rerun()

            # ── Price ──────────────────────────────────────────────────────
            _last = dr.get("last", "-")
            rc[1].markdown(
                f"<span class='scan-cell'>${_fmt(_last,',.2f')}</span>" if _last != "-" else "<span class='scan-cell muted'>—</span>",
                unsafe_allow_html=True,
            )

            # ── Score ──────────────────────────────────────────────────────
            _score = dr.get("signal_score", "-")
            try:
                _score_i = int(_score)
                _score_color = "#3fb950" if _score_i >= 7 else ("#d29922" if _score_i >= 4 else "#8b949e")
                rc[2].markdown(f"<span class='scan-cell' style='color:{_score_color};font-weight:700'>{_score_i}</span>", unsafe_allow_html=True)
            except:
                rc[2].markdown("<span class='scan-cell muted'>—</span>", unsafe_allow_html=True)

            # ── RSI (colour-coded) ─────────────────────────────────────────
            _rv = dr.get("rsi", "-")
            try:
                _rvf = float(_rv)
                _rc_col = "#3fb950" if _rvf < 40 else ("#f85149" if _rvf > 70 else "#c9d1d9")
                rc[3].markdown(f"<span class='scan-cell' style='color:{_rc_col}'>{_rvf:.1f}</span>", unsafe_allow_html=True)
            except:
                rc[3].markdown("<span class='scan-cell muted'>—</span>", unsafe_allow_html=True)

            if show_raw:
                rc[4].markdown(f"<span class='scan-cell muted'>{_fmt(dr.get('macd','-'))}</span>", unsafe_allow_html=True)
                rc[5].markdown(f"<span class='scan-cell muted'>{_fmt(dr.get('bb_pct_b','-'))}</span>", unsafe_allow_html=True)
                rc[6].markdown(f"<span class='scan-cell muted'>{_fmt(dr.get('atr','-'))}</span>", unsafe_allow_html=True)
                rc[7].markdown(f"<span class='scan-cell'>{'✅' if dr.get('trend_ok') else '❌'}</span>", unsafe_allow_html=True)
                rc[8].markdown(f"<span class='scan-cell muted'>{_fmt(dr.get('relvol','-'))}</span>", unsafe_allow_html=True)
                _sc, _rsc, _buyc, _selc = rc[9], rc[10], rc[11], rc[12]
            else:
                rc[4].markdown(f"<span class='scan-cell muted'>{_fmt(dr.get('bb_pct_b','-'))}</span>", unsafe_allow_html=True)
                rc[5].markdown(f"<span class='scan-cell muted'>{_fmt(dr.get('atr','-'))}</span>", unsafe_allow_html=True)
                rc[6].markdown(f"<span class='scan-cell'>{'✅' if dr.get('trend_ok') else '❌'}</span>", unsafe_allow_html=True)
                _sc, _rsc, _buyc, _selc = rc[7], rc[8], rc[9], rc[10]

            # ── Signal badge ───────────────────────────────────────────────
            _sc_clr = "#3fb950" if act=="BUY" else ("#f85149" if act=="SELL" else "#8b949e")
            _sc_ico = "🟢" if act=="BUY" else ("🔴" if act=="SELL" else "⚪")
            _sc.markdown(f"<span class='scan-cell' style='font-weight:700;color:{_sc_clr}'>{_sc_ico} {act}</span>", unsafe_allow_html=True)

            # ── Reason ─────────────────────────────────────────────────────
            _rsc.markdown(f"<span class='scan-cell muted'>{str(dr.get('reason',''))[:46]}</span>", unsafe_allow_html=True)

            # ── BUY button ─────────────────────────────────────────────────
            if _buyc.button("Buy", key=f"buy_{sym}_{_ri}", use_container_width=True, help=f"BUY {sym}"):
                _queue_trade("BUY", sym)

            # ── SELL button ────────────────────────────────────────────────
            if _selc.button("Sell", key=f"sell_{sym}_{_ri}", use_container_width=True, help=f"SELL {sym}"):
                _queue_trade("SELL", sym)

        # Close scanner-grid scope
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Loading scanner data… please wait for first refresh.")


    # ── Section 4: Deep-Dive Chart (only shown when a symbol is selected) ──
    sym_to_chart = st.session_state.selected_symbol
    if sym_to_chart:
        st.markdown('<div class="section-header">📈 Deep Dive</div>', unsafe_allow_html=True)
        snap = ms.get_snapshot(sym_to_chart)
        if snap:
            _d = snap.get("dates", [])[-120:]
            _c = snap.get("close_series", [])[-120:]
            _bbu = snap.get("bb_upper_series", [])[-120:]
            _bbm = snap.get("bb_mid_series", [])[-120:]
            _bbl = snap.get("bb_lower_series", [])[-120:]
            _ms  = snap.get("macd_series", [])[-120:]
            _ss  = snap.get("macd_signal_series", [])[-120:]
            _hs  = snap.get("macd_hist_series", [])[-120:]

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                vertical_spacing=0.04,
                subplot_titles=(f"{sym_to_chart} — Price + BB", "MACD"),
            )
            fig.add_trace(go.Scatter(x=_d, y=_c, name="Close", line=dict(color="#58a6ff", width=2)), row=1, col=1)
            if _bbu:
                fig.add_trace(go.Scatter(x=_d, y=_bbu, name="BB Upper", line=dict(color="rgba(248,81,73,0.5)", width=1, dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=_d, y=_bbm, name="BB Mid",   line=dict(color="rgba(255,255,255,0.3)", width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=_d, y=_bbl, name="BB Lower", line=dict(color="rgba(63,185,80,0.5)",  width=1, dash="dot"), fill="tonexty", fillcolor="rgba(63,185,80,0.04)"), row=1, col=1)
            if _c:
                _atr_v = snap.get("atr", 0)
                _stop = max(_c[-30:]) - ms.atr_mult * _atr_v
                fig.add_hline(
                    y=_stop, line_dash="dash", line_color="rgba(248,81,73,0.7)",
                    annotation_text=f"ATR Stop {_stop:.2f}", annotation_position="bottom right",
                    row=1, col=1,
                )
            if _ms:
                _ch = ["rgba(63,185,80,0.7)" if v >= 0 else "rgba(248,81,73,0.7)" for v in _hs]
                fig.add_trace(go.Bar(x=_d, y=_hs, name="Hist", marker_color=_ch, opacity=0.7), row=2, col=1)
                fig.add_trace(go.Scatter(x=_d, y=_ms, name="MACD",   line=dict(color="#58a6ff", width=1.5)), row=2, col=1)
                fig.add_trace(go.Scatter(x=_d, y=_ss, name="Signal", line=dict(color="#f0883e", width=1.5)), row=2, col=1)
            fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", row=2, col=1)
            fig.update_layout(
                height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#8b949e", size=11), showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(showgrid=False, zeroline=False, color="#8b949e"),
                xaxis2=dict(showgrid=False, zeroline=False, color="#8b949e"),
                yaxis=dict(showgrid=True,  gridcolor="rgba(255,255,255,0.05)", zeroline=False, color="#8b949e"),
                yaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, color="#8b949e"),
            )
            st.plotly_chart(fig, use_container_width=True)

            s1, s2, s3, s4, s5 = st.columns(5)
            rsi_col = "green" if snap["rsi"] < 40 else ("red" if snap["rsi"] > 70 else "blue")
            mc = "🟢 Bullish" if snap["macd"] > snap["macd_signal"] else "🔴 Bearish"
            bbp = f"{snap['bb_pct_b']:.2f}" + (" (Oversold)" if snap["bb_pct_b"] < 0.2 else (" (Overbought)" if snap["bb_pct_b"] > 0.8 else ""))
            tc = "green" if snap["trend_ok"] else "red"
            tl = "✅ Yes" if snap["trend_ok"] else "❌ No"
            s1.markdown(f'<div class="kpi-card"><div class="kpi-label">RSI (14)</div><div class="kpi-value {rsi_col}">{snap["rsi"]:.1f}</div></div>', unsafe_allow_html=True)
            s2.markdown(f'<div class="kpi-card"><div class="kpi-label">MACD Cross</div><div class="kpi-value">{mc}</div></div>', unsafe_allow_html=True)
            s3.markdown(f'<div class="kpi-card"><div class="kpi-label">BB %B</div><div class="kpi-value blue">{bbp}</div></div>', unsafe_allow_html=True)
            s4.markdown(f'<div class="kpi-card"><div class="kpi-label">ATR (14)</div><div class="kpi-value yellow">{snap["atr"]:.2f}</div></div>', unsafe_allow_html=True)
            s5.markdown(f'<div class="kpi-card"><div class="kpi-label">Trend OK</div><div class="kpi-value {tc}">{tl}</div></div>', unsafe_allow_html=True)
        else:
            st.info(f"No chart data for {sym_to_chart} yet. Click Refresh Now.")
    else:
        st.markdown(
            "<div style='text-align:center;padding:40px 0;color:#484f57'>"
            "<div style='font-size:36px'>📈</div>"
            "<div style='font-size:14px;margin-top:8px'>"
            "Select a symbol from the scanner table above to view the deep-dive chart"
            "</div></div>",
            unsafe_allow_html=True,
        )

    # ── Section 5: Holdings ────────────────────────────────────────────────
    st.markdown('<div class="section-header">💼 Current Holdings</div>', unsafe_allow_html=True)
    if not holdings_df.empty:
        h_display = holdings_df.copy()
        h_display["P/L"]   = h_display["unrealized_pl"].apply(
            lambda v: f'{"🟢" if float(v) > 0 else ("🔴" if float(v) < 0 else "⚪")} ${float(v):+.2f}'
        )
        h_display["P/L %"] = h_display["unrealized_plpc"].apply(lambda v: f'{float(v):+.2f}%')
        display_cols = ["symbol", "qty", "avg_entry", "market_value", "P/L", "P/L %"]
        col_rename = {
            "symbol": "Symbol", "qty": "Qty", "avg_entry": "Avg Entry",
            "market_value": "Market Value", "P/L": "P/L", "P/L %": "P/L %",
        }
        h_show = h_display[[c for c in display_cols if c in h_display.columns]].rename(columns=col_rename)
        st.dataframe(h_show, use_container_width=True, hide_index=True)

        # Tight Sell button row — one button per holding, aligned left
        _h_syms = holdings_df["symbol"].tolist()
        _h_chunk = 6
        for _hcs in range(0, len(_h_syms), _h_chunk):
            _syms_c = _h_syms[_hcs: _hcs + _h_chunk]
            _hbcols = st.columns([1.0] * len(_syms_c) + [max(0.01, 6 - len(_syms_c))])
            for _hbi, _hs in enumerate(_syms_c):
                _hqty = float(holdings_df.loc[holdings_df["symbol"] == _hs, "qty"].values[0])
                _hupl = float(holdings_df.loc[holdings_df["symbol"] == _hs, "unrealized_pl"].values[0])
                _huplpc = float(holdings_df.loc[holdings_df["symbol"] == _hs, "unrealized_plpc"].values[0])
                _hpl = f"{'+' if _hupl >= 0 else ''}${_hupl:,.0f}"
                if _hbcols[_hbi].button(
                    f"Sell {_hs}  {_hpl}", key=f"sell_hold_{_hs}_{_hcs}",
                    use_container_width=True, help=f"SELL {_hs} · {_huplpc:+.1f}%",
                ):
                    _queue_trade("SELL", _hs, _hqty)
    else:
        st.info("No open positions, or Alpaca credentials not set.")

    # ── Section 6: Trade Signal History ───────────────────────────────────
    st.markdown('<div class="section-header">📋 Trade Signal History</div>', unsafe_allow_html=True)
    trade_logs = ms.trade_logs
    if not trade_logs.empty:
        f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
        sym_filter    = f1.text_input("Filter symbol", value="", placeholder="e.g. FCX", label_visibility="collapsed")
        action_filter = f2.selectbox("Filter action", ["All", "BUY", "SELL", "HOLD"], label_visibility="collapsed")

        # Date range filter (Step 6)
        date_start: date_type | None = None
        date_end:   date_type | None = None
        if "date" in trade_logs.columns:
            try:
                all_dates = pd.to_datetime(trade_logs["date"]).dt.date
                _min_d = all_dates.min()
                _max_d = all_dates.max()
                _dr_from = f3.date_input("From", value=_min_d, min_value=_min_d, max_value=_max_d, label_visibility="collapsed")
                _dr_to   = f4.date_input("To",   value=_max_d, min_value=_min_d, max_value=_max_d, label_visibility="collapsed")
                date_start = _dr_from if isinstance(_dr_from, date_type) else _min_d
                date_end   = _dr_to   if isinstance(_dr_to,   date_type) else _max_d
            except Exception:
                pass

        filtered_logs = trade_logs.copy()
        if sym_filter:
            filtered_logs = filtered_logs[filtered_logs["symbol"].str.upper() == sym_filter.upper()]
        if action_filter != "All":
            filtered_logs = filtered_logs[filtered_logs["action"] == action_filter]
        if date_start and date_end and "date" in filtered_logs.columns:
            try:
                _mask = (
                    (pd.to_datetime(filtered_logs["date"]).dt.date >= date_start) &
                    (pd.to_datetime(filtered_logs["date"]).dt.date <= date_end)
                )
                filtered_logs = filtered_logs[_mask]
            except Exception:
                pass

        if not filtered_logs.empty:
            buy_rows = filtered_logs[filtered_logs["action"] == "BUY"]
            if not buy_rows.empty:
                ss = buy_rows.groupby("symbol").size()
                st.caption(f"📊 Most persistent BUY: **{ss.idxmax()}** ({ss.max()} days)")

        def _log_badge(row: pd.Series) -> list[str]:
            a = str(row.get("action", ""))
            if a == "BUY":  return ["background-color: rgba(63,185,80,0.10)"] * len(row)
            if a == "SELL": return ["background-color: rgba(248,81,73,0.10)"] * len(row)
            return [""] * len(row)

        st.dataframe(filtered_logs.style.apply(_log_badge, axis=1), width="stretch", hide_index=True, height=300)
    else:
        st.info("No trade logs found in logs/ directory.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: Custom Ticker Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    st.markdown('<div class="section-header">🔬 Custom Ticker Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#8b949e;font-size:13px'>Enter any US stock ticker to fetch live data "
        "and get a full buy/sell recommendation based on your active strategy settings.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Single-row input — strategy comes from the global sidebar (Step 7)
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        custom_ticker = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get("custom_ticker", ""),
            placeholder="e.g. NVDA, META, AMD…",
            label_visibility="collapsed",
            key="custom_ticker_input",
        ).strip().upper()
    with col_btn:
        run_analysis = st.button("🔍 Analyze", type="primary", use_container_width=True)

    # Show which strategy is active (replaces the removed duplicate selector)
    st.caption(f"Using strategy: **{entry_style.upper()}** · ATR Mult: **{atr_mult}x** — change in sidebar ↖")

    if run_analysis and custom_ticker:
        st.session_state["custom_ticker"] = custom_ticker
        with st.spinner(f"Fetching {custom_ticker} data…"):
            try:
                from market_sentinel.engine import fetch_bars_yfinance, compute_signal_snapshot, decide_action
                _bars = fetch_bars_yfinance(custom_ticker)
                _snap = compute_signal_snapshot(custom_ticker, _bars, atr_mult)
                _plan_df = ms.get_plan_df()
                _market_gate = None
                if not _plan_df.empty and "market_gate" in _plan_df.columns:
                    _gate_val = _plan_df["market_gate"].iloc[0]
                    if isinstance(_gate_val, bool):
                        _market_gate = _gate_val
                _snap_with_gate = dict(_snap)
                _snap_with_gate["market_gate"] = _market_gate
                _action, _reason = decide_action(_snap_with_gate, entry_style, qty=0.0, days_held=None, atr_mult=atr_mult)

                # ── Action Banner ─────────────────────────────────────────
                _banner_cfg = {
                    "BUY":  ("rgba(63,185,80,0.15)",  "#3fb950", "🟢 BUY"),
                    "SELL": ("rgba(248,81,73,0.15)",  "#f85149", "🔴 SELL"),
                    "HOLD": ("rgba(139,148,158,0.12)", "#8b949e", "⚪ HOLD"),
                }.get(_action, ("rgba(139,148,158,0.12)", "#8b949e", f"⚪ {_action}"))
                _bg, _col, _lbl = _banner_cfg
                st.markdown(
                    f"<div style='background:{_bg};border:1px solid {_col};border-radius:12px;"
                    f"padding:16px 24px;margin:12px 0'>"
                    f"<span style='font-size:22px;font-weight:800;color:{_col}'>{_lbl}</span>"
                    f"<span style='color:#8b949e;font-size:14px;margin-left:16px'>{_reason}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ── Signal KPI Cards ──────────────────────────────────────
                _rsi_cls = "green" if _snap["rsi"] < 40 else ("red" if _snap["rsi"] > 70 else "blue")
                _macd_cross = "🟢 Bullish" if _snap["macd"] > _snap["macd_signal"] else "🔴 Bearish"
                _bb_text = f"{_snap['bb_pct_b']:.2f}"
                if _snap["bb_pct_b"] < 0.2:  _bb_text += " (Oversold)"
                elif _snap["bb_pct_b"] > 0.8: _bb_text += " (Overbought)"
                _trend_cls = "green" if _snap["trend_ok"] else "red"
                _trend_lbl = "✅ Yes" if _snap["trend_ok"] else "❌ No"

                _c1, _c2, _c3, _c4, _c5, _c6, _c7 = st.columns(7)
                _c1.markdown(f'<div class="kpi-card"><div class="kpi-label">Price</div><div class="kpi-value blue">${_snap["last"]:,.2f}</div></div>', unsafe_allow_html=True)
                _c2.markdown(f'<div class="kpi-card"><div class="kpi-label">RSI (14)</div><div class="kpi-value {_rsi_cls}">{_snap["rsi"]:.1f}</div></div>', unsafe_allow_html=True)
                _c3.markdown(f'<div class="kpi-card"><div class="kpi-label">MACD</div><div class="kpi-value">{_macd_cross}</div></div>', unsafe_allow_html=True)
                _c4.markdown(f'<div class="kpi-card"><div class="kpi-label">BB %B</div><div class="kpi-value blue">{_bb_text}</div></div>', unsafe_allow_html=True)
                _c5.markdown(f'<div class="kpi-card"><div class="kpi-label">ATR (14)</div><div class="kpi-value yellow">{_snap["atr"]:.2f}</div></div>', unsafe_allow_html=True)
                _c6.markdown(f'<div class="kpi-card"><div class="kpi-label">Trend OK</div><div class="kpi-value {_trend_cls}">{_trend_lbl}</div></div>', unsafe_allow_html=True)
                _c7.markdown(f'<div class="kpi-card"><div class="kpi-label">RelVol</div><div class="kpi-value">{_snap["relvol"] if _snap["relvol"] else "—"}</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Chart (moved before the breakdown table — Step 8) ─────
                _dates   = _snap.get("dates", [])[-120:]
                _close_s = _snap.get("close_series", [])[-120:]
                _bb_u    = _snap.get("bb_upper_series", [])[-120:]
                _bb_m    = _snap.get("bb_mid_series", [])[-120:]
                _bb_l    = _snap.get("bb_lower_series", [])[-120:]
                _macd_s  = _snap.get("macd_series", [])[-120:]
                _sig_s   = _snap.get("macd_signal_series", [])[-120:]
                _hist_s  = _snap.get("macd_hist_series", [])[-120:]

                _fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.04,
                    subplot_titles=(f"{custom_ticker} — Price + Bollinger Bands", "MACD"),
                )
                _fig.add_trace(go.Scatter(x=_dates, y=_close_s, name="Close", line=dict(color="#58a6ff", width=2.5)), row=1, col=1)
                if _bb_u:
                    _fig.add_trace(go.Scatter(x=_dates, y=_bb_u, name="BB Upper", line=dict(color="rgba(248,81,73,0.5)", width=1, dash="dot")), row=1, col=1)
                    _fig.add_trace(go.Scatter(x=_dates, y=_bb_m, name="BB Mid",   line=dict(color="rgba(255,255,255,0.3)", width=1)), row=1, col=1)
                    _fig.add_trace(go.Scatter(x=_dates, y=_bb_l, name="BB Lower", line=dict(color="rgba(63,185,80,0.5)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(63,185,80,0.04)"), row=1, col=1)
                if _close_s:
                    _atr_stop = max(_close_s[-30:]) - atr_mult * _snap["atr"]
                    _fig.add_hline(
                        y=_atr_stop, line_dash="dash", line_color="rgba(248,81,73,0.7)",
                        annotation_text=f"ATR Stop {_atr_stop:.2f}", annotation_position="bottom right",
                        row=1, col=1,
                    )
                _close_series_full = pd.Series(_snap["close_series"])
                _sma50_s  = _close_series_full.rolling(50).mean().tolist()[-120:]
                _sma200_s = _close_series_full.rolling(200).mean().tolist()[-120:]
                _fig.add_trace(go.Scatter(x=_dates, y=_sma50_s,  name="SMA50",  line=dict(color="rgba(210,153,34,0.6)",  width=1)), row=1, col=1)
                _fig.add_trace(go.Scatter(x=_dates, y=_sma200_s, name="SMA200", line=dict(color="rgba(88,166,255,0.4)",  width=1)), row=1, col=1)
                if _macd_s:
                    _colors_h = ["rgba(63,185,80,0.7)" if v >= 0 else "rgba(248,81,73,0.7)" for v in _hist_s]
                    _fig.add_trace(go.Bar(x=_dates, y=_hist_s, name="Histogram", marker_color=_colors_h, opacity=0.7), row=2, col=1)
                    _fig.add_trace(go.Scatter(x=_dates, y=_macd_s, name="MACD",   line=dict(color="#58a6ff", width=1.5)), row=2, col=1)
                    _fig.add_trace(go.Scatter(x=_dates, y=_sig_s,  name="Signal", line=dict(color="#f0883e", width=1.5)), row=2, col=1)
                _fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", row=2, col=1)
                _fig.update_layout(
                    height=540,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#8b949e", size=11),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis=dict(showgrid=False, zeroline=False, color="#8b949e"),
                    xaxis2=dict(showgrid=False, zeroline=False, color="#8b949e"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, color="#8b949e"),
                    yaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, color="#8b949e"),
                )
                st.plotly_chart(_fig, use_container_width=True)

                # ── Detailed Breakdown Table (collapsed by default — Step 8) ─
                with st.expander("📊 Full Indicator Breakdown", expanded=False):
                    breakdown_rows = [
                        {"Indicator": "Price",         "Value": f"${_snap['last']:,.2f}",                                               "Signal": "—"},
                        {"Indicator": "SMA 50",         "Value": f"${_snap['sma50']:,.2f}",          "Signal": "🟢 Above" if _snap["last"] > _snap["sma50"]  else "🔴 Below"},
                        {"Indicator": "SMA 200",        "Value": f"${_snap['sma200']:,.2f}",         "Signal": "🟢 Above" if _snap["last"] > _snap["sma200"] else "🔴 Below"},
                        {"Indicator": "RSI (14)",       "Value": f"{_snap['rsi']:.2f}",              "Signal": "🔴 Overbought" if _snap["rsi"] > 70 else ("🟢 Oversold" if _snap["rsi"] < 30 else "⚪ Neutral")},
                        {"Indicator": "MACD Line",      "Value": f"{_snap['macd']:.4f}",             "Signal": "🟢 Bullish" if _snap["macd"] > _snap["macd_signal"] else "🔴 Bearish"},
                        {"Indicator": "MACD Signal",    "Value": f"{_snap['macd_signal']:.4f}",      "Signal": "—"},
                        {"Indicator": "MACD Histogram", "Value": f"{_snap['macd_hist']:.4f}",        "Signal": "🟢 Rising" if _snap["macd_hist"] > 0 else "🔴 Falling"},
                        {"Indicator": "BB Upper",       "Value": f"${_snap['bb_upper']:,.2f}",       "Signal": "—"},
                        {"Indicator": "BB Mid (SMA20)", "Value": f"${_snap['bb_mid']:,.2f}",         "Signal": "—"},
                        {"Indicator": "BB Lower",       "Value": f"${_snap['bb_lower']:,.2f}",       "Signal": "🟢 Near support" if _snap["last"] < _snap["bb_lower"] * 1.02 else "—"},
                        {"Indicator": "BB %B",          "Value": f"{_snap['bb_pct_b']:.3f}",         "Signal": "🔴 Overbought" if _snap["bb_pct_b"] > 0.8 else ("🟢 Oversold" if _snap["bb_pct_b"] < 0.2 else "⚪ Neutral")},
                        {"Indicator": "ATR (14)",       "Value": f"{_snap['atr']:.2f}",              "Signal": "—"},
                        {"Indicator": f"ATR Stop ({atr_mult}x)", "Value": f"${max(_snap['close_series'][-30:]) - atr_mult * _snap['atr']:,.2f}", "Signal": "🛡️ Trailing Stop"},
                        {"Indicator": "Breakout55",     "Value": str(_snap["breakout55"]),            "Signal": "🟢 Yes" if _snap["breakout55"] else "⚪ No"},
                        {"Indicator": "Dist10",         "Value": str(_snap["dist10"]),                "Signal": "🔴 Heavy" if _snap["dist10"] >= 4 else ("🟡 Moderate" if _snap["dist10"] >= 2 else "🟢 Light")},
                        {"Indicator": "Trend OK",       "Value": str(_snap["trend_ok"]),              "Signal": "🟢 Confirmed" if _snap["trend_ok"] else "🔴 Failed"},
                    ]
                    st.dataframe(pd.DataFrame(breakdown_rows), width="stretch", hide_index=True)

            except Exception as _exc:
                st.error(f"❌ Could not analyze **{custom_ticker}**: {_exc}")
                st.caption("Check the ticker is a valid US equity symbol available via yfinance.")

    elif run_analysis and not custom_ticker:
        st.warning("Please enter a ticker symbol first.")
    else:
        st.markdown(
            "<div style='text-align:center;padding:60px 0;color:#484f57'>"
            "<div style='font-size:48px'>🔬</div>"
            "<div style='font-size:16px;margin-top:12px'>Enter a ticker symbol above and click <b>Analyze</b></div>"
            "<div style='font-size:13px;margin-top:8px;color:#30363d'>Supports any US equity: NVDA, TSLA, AMZN, etc.</div>"
            "</div>",
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: Options Strategy Hub
# ════════════════════════════════════════════════════════════════════════════
with tab_options:
    from market_sentinel.options_engine import (
        recommend_strategy as _rec_strategy,
        analyze_strategy   as _analyze_strategy,
        STRATEGY_META      as _SMETA,
        STRATEGY_IDS       as _SIDS,
    )

    st.markdown('<div class="section-header">📜 Options Strategy Hub</div>', unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#8b949e;font-size:13px'>Enter any US stock ticker to get an AI-powered options "
        "strategy recommendation based on live technical signals, then fetch the options chain to compute specific contracts.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Score colour helper
    _o_score_cfg = {
        10.0: ("#3fb950", "🟢"), 7.0: ("#d29922", "🟡"),
        4.0:  ("#f0883e", "🟠"), 0.1: ("#f85149", "🔴"), 0.0: ("#8b949e", "⚪"),
    }
    def _osc(sc):
        for thr, (c, i) in sorted(_o_score_cfg.items(), reverse=True):
            if sc >= thr: return c, i
        return "#484f57", "⚪"

    # ── Ticker Input Row (Entry Price removed here — shown conditionally later) ─
    _oi1, _oi2, _oi3 = st.columns([3, 1, 1.5])
    _opt_ticker = _oi1.text_input(
        "Ticker", value=st.session_state.get("opt_ticker", ""),
        placeholder="e.g. AAPL, NVDA, SPY…",
        label_visibility="collapsed", key="opt_ticker_input",
    ).strip().upper()
    _run_opt_analysis = _oi2.button("📊 Analyze", type="primary", use_container_width=True, key="run_opt_analysis")
    _run_auto_top_10  = _oi3.button("🪄 Auto-Scan Top 10", type="secondary", use_container_width=True, key="run_auto_opt")

    if st.session_state.get("opt_trigger_fetch"):
        _do_fetch = True
        st.session_state["opt_trigger_fetch"] = False
        _opt_ticker = st.session_state.get("opt_ticker", _opt_ticker)
    else:
        _do_fetch = bool(_run_opt_analysis and _opt_ticker)

    if _do_fetch:
        st.session_state["opt_auto_scan"] = False
        st.session_state["opt_ticker"] = _opt_ticker
        with st.spinner(f"Fetching signals for {_opt_ticker}…"):
            try:
                from market_sentinel.engine import fetch_bars_yfinance, compute_signal_snapshot
                _ob = fetch_bars_yfinance(_opt_ticker)
                _os = compute_signal_snapshot(_opt_ticker, _ob, atr_mult)
                _os["symbol"] = _opt_ticker
                st.session_state["opt_snap"] = _os
            except Exception as _se:
                st.error(f"❌ Could not fetch {_opt_ticker}: {_se}")
                st.session_state["opt_snap"] = None
    elif _run_auto_top_10:
        st.session_state["opt_auto_scan"] = True
        st.session_state["opt_snap"] = None
    elif not _run_opt_analysis and not st.session_state.get("opt_snap") and not st.session_state.get("opt_auto_scan"):
        st.markdown(
            "<div style='text-align:center;padding:60px 0;color:#484f57'>"
            "<div style='font-size:48px'>📜</div>"
            "<div style='font-size:16px;margin-top:12px'>Enter a ticker and click <b>Analyze</b> "
            "to get your AI options recommendation</div>"
            "<div style='font-size:13px;margin-top:8px;color:#30363d'>"
            "Supports all 8 options strategies · AI-scored against live signals</div>"
            "</div>", unsafe_allow_html=True,
        )

    # ── Auto-Scan Top 10 ──────────────────────────────────────────────────
    if st.session_state.get("opt_auto_scan"):
        st.markdown('<div class="section-header">🏆 Top 10 Recommended Options Plays</div>', unsafe_allow_html=True)
        with st.spinner("Scanning market and calculating signals for top 10 tickers..."):
            try:
                from market_sentinel.trading_client import build_options_plan
                opt_rows, opt_meta = build_options_plan(max_symbols=10, atr_mult=atr_mult)
                if not opt_rows:
                    st.info("No options recommendations generated at this time.")
                else:
                    opt_df = pd.DataFrame(opt_rows)
                    opt_df.rename(columns={
                        "symbol": "Ticker", "strategy": "Strategy", "direction": "Direction",
                        "score": "Score", "reason": "Signal Driver", "source": "Data Source",
                    }, inplace=True)
                    st.dataframe(opt_df, width="stretch", hide_index=True)
                    st.caption(f"Scanned top {opt_meta.get('plan_size')} tickers. Generated at {opt_meta.get('generated_at_utc')}.")

                    # ── Click-through: select a result and open it in the analyzer (Step 11)
                    if "Ticker" in opt_df.columns:
                        _as_c1, _as_c2 = st.columns([2, 1])
                        _autoscan_sel = _as_c1.selectbox(
                            "Choose a ticker to analyze in detail",
                            opt_df["Ticker"].tolist(),
                            label_visibility="collapsed",
                            key="autoscan_ticker_sel",
                        )
                        if _as_c2.button("🔍 Analyze This Ticker", use_container_width=True, key="autoscan_analyze_btn"):
                            st.session_state["opt_ticker"]        = _autoscan_sel
                            st.session_state["opt_trigger_fetch"] = True
                            st.session_state["opt_auto_scan"]     = False
                            st.rerun()
            except Exception as e:
                import traceback
                traceback.print_exc()
                st.error(f"❌ Could not run auto-scan: {e}")

    # ── Single Ticker Analysis Results ────────────────────────────────────
    _os_snap   = st.session_state.get("opt_snap")
    _os_ticker = st.session_state.get("opt_ticker", _opt_ticker)

    if _os_snap and not st.session_state.get("opt_auto_scan"):

        # ── Signal Summary Bar ────────────────────────────────────────────
        _soc1, _soc2, _soc3, _soc4, _soc5 = st.columns(5)
        _rsi_c = "green" if _os_snap["rsi"] < 40 else ("red" if _os_snap["rsi"] > 70 else "blue")
        _soc1.markdown(f'<div class="kpi-card"><div class="kpi-label">Price</div><div class="kpi-value blue">${_os_snap["last"]:,.2f}</div></div>', unsafe_allow_html=True)
        _soc2.markdown(f'<div class="kpi-card"><div class="kpi-label">RSI</div><div class="kpi-value {_rsi_c}">{_os_snap["rsi"]:.1f}</div></div>', unsafe_allow_html=True)
        _soc3.markdown(f'<div class="kpi-card"><div class="kpi-label">MACD</div><div class="kpi-value">{"🟢 Bull" if _os_snap["macd"] > _os_snap["macd_signal"] else "🔴 Bear"}</div></div>', unsafe_allow_html=True)
        _trend_kpi_cls = "green" if _os_snap["trend_ok"] else "red"
        _soc4.markdown(f'<div class="kpi-card"><div class="kpi-label">Trend OK</div><div class="kpi-value {_trend_kpi_cls}">{"✅ Yes" if _os_snap["trend_ok"] else "❌ No"}</div></div>', unsafe_allow_html=True)
        _bo55_kpi_cls = "green" if _os_snap["breakout55"] else "yellow"
        _soc5.markdown(f'<div class="kpi-card"><div class="kpi-label">Breakout55</div><div class="kpi-value {_bo55_kpi_cls}">{"✅ Yes" if _os_snap["breakout55"] else "⚪ No"}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── AI Recommendation Banner ──────────────────────────────────────
        _orec   = _rec_strategy(_os_snap)
        _otop   = _orec.top
        _oranged = _orec.ranked
        _orc, _ori = _osc(_otop.score)
        st.markdown(
            f"<div style='background:rgba(63,185,80,0.07);border:1.5px solid {_orc};border-radius:14px;padding:18px 24px;margin:10px 0'>"
            f"<div style='font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px'>🤖 AI Recommendation for {_os_ticker}</div>"
            f"<div style='font-size:22px;font-weight:800;color:{_orc};margin:6px 0'>{_ori} {_otop.name} "
            f"<span style='font-size:14px;font-weight:400;color:#8b949e'>Score: {_otop.score}/10</span></div>"
            f"<div style='font-size:13px;color:#c9d1d9;line-height:1.7'>{_orec.reasoning}</div>"
            f"</div>", unsafe_allow_html=True,
        )

        # ── All 8 Strategy Scores — Visual Score Bars ─────────────────────
        st.markdown("<span style='color:#8b949e;font-size:12px'>All 8 strategies scored against current signals:</span>", unsafe_allow_html=True)
        _score_html = ""
        for _si, _oss in enumerate(_oranged):
            _oc, _oic = _osc(_oss.score)
            _bar_w = int(_oss.score / 10 * 100)
            _is_top = _si == 0
            _bg = "rgba(63,185,80,0.08)" if _is_top else "rgba(255,255,255,0.02)"
            _border = f"1px solid {_oc}" if _is_top else "1px solid rgba(255,255,255,0.05)"
            _score_html += (
                f"<div style='background:{_bg};border:{_border};border-radius:8px;"
                f"padding:8px 12px;margin:4px 0;display:flex;align-items:center;gap:10px'>"
                f"<span style='font-size:11px;color:#484f57;width:20px'>#{_si+1}</span>"
                f"<span style='font-size:12px;font-weight:600;color:#c9d1d9;width:170px'>{_oic} {_oss.name}</span>"
                f"<div style='flex:1;background:rgba(255,255,255,0.07);border-radius:4px;height:6px'>"
                f"<div style='width:{_bar_w}%;background:{_oc};height:100%;border-radius:4px'></div></div>"
                f"<span style='font-size:12px;font-weight:700;color:{_oc};width:35px;text-align:right'>{_oss.score}/10</span>"
                f"<span style='font-size:11px;color:#8b949e;width:60px'>{_oss.direction}</span>"
                f"</div>"
            )
        st.markdown(_score_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


        # ── Live Chain Analyzer ───────────────────────────────────────────
        st.markdown('<div class="section-header">🔍 Live Options Chain Analysis</div>', unsafe_allow_html=True)
        _la1, _la2, _la3, _la4 = st.columns([2, 1, 1, 1])
        _o_strat_labels = [f"{'⭐ ' if s.strategy_id == _otop.strategy_id else ''}{s.name}" for s in _oranged]
        _o_strat_ids    = [s.strategy_id for s in _oranged]
        _o_sel_lbl = _la1.selectbox("Strategy", _o_strat_labels, index=0, key="opt_sel_strat", label_visibility="collapsed")
        _o_sel_sid = _o_strat_ids[_o_strat_labels.index(_o_sel_lbl)]
        _o_dte_min = _la2.number_input("Min DTE", min_value=7,  max_value=120, value=30, step=5,   key="opt_dte_min3")
        _o_dte_max = _la3.number_input("Max DTE", min_value=14, max_value=180, value=75, step=5,   key="opt_dte_max3")
        _o_width   = _la4.number_input("Width / Wing ($)", min_value=1.0, max_value=100.0, value=5.0, step=1.0, key="opt_width3")

        # Conditional Entry Price — only visible for Covered Call (Step 9)
        _opt_entry_px = 0.0
        if _o_sel_sid == "covered_call":
            _opt_entry_px = st.number_input(
                "📌 Your avg entry price (required for Covered Call P/L calculation)",
                min_value=0.0, value=float(_os_snap.get("last", 0)), step=1.0, key="opt_entry_px",
            )

        # Strategy explainer (collapsed by default)
        _o_smeta = _SMETA[_o_sel_sid]
        _o_ssel  = next((s for s in _oranged if s.strategy_id == _o_sel_sid), _oranged[0])
        _o_mc, _o_mi = _osc(_o_ssel.score)
        with st.expander(f"📖 {_o_smeta['name']} — Strategy Details & Signal Fit", expanded=False):
            _ed1, _ed2 = st.columns(2)
            _ed1.markdown(f"""
| | |
|---|---|
| **Direction** | {_o_smeta['direction']} |
| **Risk** | {_o_smeta['risk']} |
| **Complexity** | {_o_smeta['complexity']} |
| **Best When** | {_o_smeta['when']} |
| **Legs** | {_o_smeta['legs']} |
""")
            _ed2.markdown(f"""
| | |
|---|---|
| **Max Profit** | {_o_smeta['profit']} |
| **Max Loss** | {_o_smeta['loss']} |
| **Signal Score** | {_o_mi} {_o_ssel.score}/10 |
""")
            _ef1, _ef2 = st.columns(2)
            _ef1.markdown("**✅ Why it fits:**")
            for _ecm in _o_ssel.conditions_met:   _ef1.markdown(f"- {_ecm}")
            _ef2.markdown("**❌ What's missing:**")
            for _ecm in _o_ssel.conditions_missed: _ef2.markdown(f"- {_ecm}")

        _run_chain = st.button(f"🔍 Fetch Live Chain & Compute {_o_smeta['name']}", key="run_chain_btn", type="primary")

        if _run_chain:
            with st.spinner(f"Fetching {_os_ticker} options chain…"):
                try:
                    _ep = _opt_entry_px if _opt_entry_px > 0 else _os_snap.get("last", 0)
                    _ores_val = _analyze_strategy(
                        _os_ticker, _o_sel_sid, _os_snap,
                        dte_min=int(_o_dte_min), dte_max=int(_o_dte_max),
                        width=float(_o_width), min_oi=50,
                        entry_price=float(_ep),
                    )
                    st.session_state["opt_ores"] = _ores_val
                except Exception as _oexc:
                    st.session_state["opt_ores"] = None
                    st.error(f"❌ Options chain failed: {_oexc}")
                    st.caption("Options chains require liquid symbols. Best results: AAPL, NVDA, TSLA, SPY, QQQ.")

        _ores = st.session_state.get("opt_ores")
        if _ores:
            _is_cr = _ores.debit_or_credit < 0
            _cost  = f"Credit: ${abs(_ores.debit_or_credit):.2f}" if _is_cr else f"Debit: ${_ores.debit_or_credit:.2f}"
            _rrcol = "green" if _ores.rr >= 1.0 else ("yellow" if _ores.rr >= 0.5 else "red")

            # Result banner
            st.markdown(
                f"<div style='background:rgba(13,17,23,0.9);border:1px solid {_o_mc};border-radius:12px;padding:16px 20px;margin:10px 0'>"
                f"<span style='font-size:18px;font-weight:700;color:{_o_mc}'>{_o_mi} {_ores.name} — {_os_ticker} {_ores.expiry}</span>"
                f"<span style='color:#8b949e;font-size:13px;margin-left:12px'>{_cost}/share</span>"
                f"</div>", unsafe_allow_html=True,
            )

            # KPIs
            _okc = st.columns(5)
            _cr_cls = "green" if _is_cr else "red"
            _okc[0].markdown(f'<div class="kpi-card"><div class="kpi-label">Expiry</div><div class="kpi-value blue" style="font-size:14px">{_ores.expiry}</div></div>', unsafe_allow_html=True)
            _okc[1].markdown(f'<div class="kpi-card"><div class="kpi-label">{"Credit" if _is_cr else "Debit"}/share</div><div class="kpi-value {_cr_cls}">${abs(_ores.debit_or_credit):.2f}</div></div>', unsafe_allow_html=True)
            _okc[2].markdown(f'<div class="kpi-card"><div class="kpi-label">Max Profit</div><div class="kpi-value green">{"∞" if _ores.max_profit > 999 else f"${_ores.max_profit:.2f}"}</div></div>', unsafe_allow_html=True)
            _okc[3].markdown(f'<div class="kpi-card"><div class="kpi-label">Max Loss</div><div class="kpi-value red">${_ores.max_loss:.2f}</div></div>', unsafe_allow_html=True)
            _okc[4].markdown(f'<div class="kpi-card"><div class="kpi-label">R/R</div><div class="kpi-value {_rrcol}">{"N/A" if _ores.rr == 0 else f"{_ores.rr:.2f}x"}</div></div>', unsafe_allow_html=True)

            _obes = "  |  ".join([f"${b:,.2f}" for b in _ores.breakeven])
            _per_c_profit = 999999 if _ores.max_profit > 999 else _ores.max_profit * 100
            st.markdown(
                f"<span style='color:#8b949e;font-size:12px'>📍 Breakeven: <b style='color:#c9d1d9'>{_obes}</b>"
                f"&nbsp;·&nbsp;Per contract: cost <b>${abs(_ores.debit_or_credit)*100:,.0f}</b>"
                f" · max profit <b>${_per_c_profit:,.0f}</b></span>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Legs
            with st.expander("📋 Legs & Contracts", expanded=True):
                _oldf = pd.DataFrame(_ores.legs)[["leg", "action", "type", "strike", "contract", "premium"]]
                _oldf.columns = ["Leg", "Action", "Type", "Strike ($)", "Contract Symbol", "Premium ($)"]
                st.dataframe(_oldf, width="stretch", hide_index=True)
                st.caption(_ores.notes)

            # Execution UI
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🛒 Execute Trade via Alpaca")
            _b1, _b2, _b3 = st.columns([1, 2, 1])
            _buy_qty = _b1.number_input("Multiplier (Contracts)", min_value=1, value=1, step=1, key="opt_buy_qty")
            if _b2.button("🚨 BUY OPTIONS", type="primary", use_container_width=True, key="opt_buy_btn"):
                with st.spinner("Submitting order..."):
                    try:
                        from market_sentinel.trading_client import execute_options_trade
                        execute_options_trade(legs=_ores.legs, multiplier=int(_buy_qty))
                        st.success(f"✅ Executed {int(_buy_qty)}x {_ores.name} contracts to Alpaca.")
                    except Exception as _be:
                        st.error(f"❌ Order Failed: {_be}")

            # P&L at Expiry
            with st.expander("📊 P&L at Expiry", expanded=True):
                _orrt = pd.DataFrame(_ores.risk_reward_table)
                def _ort_color(row):
                    if "Profit" in str(row.get("Result", "")): return ["background-color:rgba(63,185,80,0.1)"] * len(row)
                    if "Loss"   in str(row.get("Result", "")): return ["background-color:rgba(248,81,73,0.1)"] * len(row)
                    return [""] * len(row)
                st.dataframe(_orrt.style.apply(_ort_color, axis=1), width="stretch", hide_index=True)

            # Management guide
            _omgmt = {
                "call_debit_spread": "📌 **Close at +50% max profit or 21 DTE.** Cut if stock drops below long strike.",
                "put_debit_spread":  "📌 **Close at +50% max profit or 21 DTE.** Cut if stock bounces above short put.",
                "long_call":         "📌 **Target 2× premium.** Stop: −50% of premium. Close 21 DTE.",
                "long_put":          "📌 **Target 2× premium.** Stop: −50%. Roll down/out if near breakeven.",
                "cash_secured_put":  "📌 **Roll down/out for credit** if breached. Accept assignment only if bullish on stock.",
                "covered_call":      "📌 **Let expire worthless.** Roll up/out if stock breaks above short strike.",
                "bull_put_spread":   "📌 **Close at 50% credit captured.** Exit if stock closes below short put for 2 days.",
                "iron_condor":       "📌 **Close at 50% credit or 21 DTE.** Close threatened side early if within $2 of short strike.",
            }.get(_o_sel_sid, "📌 Follow standard position management rules.")
            st.info(_omgmt)

    elif not _opt_ticker:
        pass  # placeholder already shown above

    # ── Active Options Portfolio — moved to BOTTOM of tab (Step 9) ────────
    _ohdf = ms.get_holdings_df()
    if not _ohdf.empty:
        import re
        _opt_rows_port = []
        for _, row in _ohdf.iterrows():
            sym = str(row.get("symbol", ""))
            m = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", sym)
            if m:
                u, d_str, cp, s_str = m.groups()
                yr = 2000 + int(d_str[0:2])
                mo = int(d_str[2:4])
                da = int(d_str[4:6])
                exp = f"{yr}-{mo:02d}-{da:02d}"
                try:
                    dte = (datetime(yr, mo, da).date() - datetime.now().date()).days
                except Exception:
                    dte = 0
                typ = "Call" if cp == "C" else "Put"
                strk = float(s_str) / 1000.0
                qty_raw = float(row.get("qty", 0))
                pos_type = "Long" if qty_raw > 0 else "Short"
                _opt_rows_port.append({
                    "Contract":    sym,
                    "Asset":       f"{u} {pos_type} {typ}",
                    "Strike":      f"${strk:,.2f}",
                    "Expiry":      exp,
                    "DTE":         dte,
                    "Qty":         int(abs(qty_raw)),
                    "Avg Entry":   f"${row.get('avg_entry', 0):.2f}",
                    "Current Val": f"${float(row.get('market_value', 0)):.2f}",
                    "P/L":         float(row.get("unrealized_pl", 0)),
                })

        if _opt_rows_port:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">💼 Active Options Portfolio</div>', unsafe_allow_html=True)
            with st.expander("View Active Options Positions", expanded=True):
                _odf = pd.DataFrame(_opt_rows_port)

                def _oc_pl(r):
                    if not isinstance(r, pd.Series): return [""] * len(r)
                    v_raw = r.get("P/L", 0.0)
                    if isinstance(v_raw, str):
                        v_raw = v_raw.replace("$", "").replace("+", "").replace(",", "")
                    try:
                        v = float(v_raw)
                    except ValueError:
                        v = 0.0
                    if v < 0: return ["background-color:rgba(248,81,73,0.15)"] * len(r)
                    if v > 0: return ["background-color:rgba(63,185,80,0.15)"] * len(r)
                    return [""] * len(r)

                _odf_display = _odf.copy()
                _odf_display["P/L"] = _odf["P/L"].apply(lambda x: f"${x:+.2f}")
                st.dataframe(_odf_display.style.apply(_oc_pl, axis=1), width="stretch", hide_index=True)

                _o_cols = st.columns(min(len(_odf), 6) or 1)
                for enum_opt_i, (_, opt_row) in enumerate(_odf.iterrows()):
                    c_idx = enum_opt_i % min(len(_odf), 6)
                    if _o_cols[c_idx].button(
                        f"Close {opt_row['Asset']}",
                        key=f"close_opt_{opt_row['Contract']}",
                        use_container_width=True,
                    ):
                        try:
                            from alpaca.trading.client import TradingClient
                            k = os.getenv("ALPACA_KEY", "")
                            s = os.getenv("ALPACA_SECRET", "")
                            p = os.getenv("ALPACA_PAPER", "true").lower() in {"1", "true", "yes"}
                            TradingClient(k, s, paper=p).close_position(opt_row["Contract"])
                            st.success(f"Sent close order for {opt_row['Contract']}")
                            ms.refresh()
                            st.rerun()
                        except Exception as close_e:
                            st.error(f"Failed to close: {close_e}")

            # ── Portfolio Intelligence — hourly analysis ───────────────────
            _OPT_ANALYSIS_TTL = 3600  # seconds

            def _build_opt_analysis(rows: list[dict]) -> list[dict]:
                """Group legs into strategies and generate action recommendations."""
                from collections import defaultdict
                groups: dict[str, list[dict]] = defaultdict(list)
                for r in rows:
                    # group key = underlying + expiry + option_type
                    asset_parts = str(r.get("Asset", "")).split()
                    underlying = asset_parts[0] if asset_parts else "?"
                    key = f"{underlying}_{r.get('Expiry','')}"
                    groups[key].append(r)

                results = []
                for key, legs in groups.items():
                    underlying = legs[0]["Asset"].split()[0]
                    expiry = legs[0]["Expiry"]
                    dte = legs[0]["DTE"]
                    net_pl = sum(float(l["P/L"]) for l in legs)
                    total_contracts = sum(int(l["Qty"]) for l in legs)

                    # Classify strategy
                    long_legs  = [l for l in legs if "Long"  in l["Asset"]]
                    short_legs = [l for l in legs if "Short" in l["Asset"]]
                    calls = [l for l in legs if "Call" in l["Asset"]]
                    puts  = [l for l in legs if "Put"  in l["Asset"]]

                    if len(legs) == 1 and long_legs:
                        strat = f"Long {'Call' if calls else 'Put'}"
                        max_loss_est = sum(
                            float(str(l["Avg Entry"]).replace("$","")) * int(l["Qty"]) * 100
                            for l in legs
                        )
                    elif len(legs) == 2 and long_legs and short_legs:
                        if calls:
                            strat = "Bull Call Spread"
                        elif puts:
                            # net credit = short put premium - long put premium
                            long_entry  = float(str(long_legs[0]["Avg Entry"]).replace("$",""))
                            short_entry = float(str(short_legs[0]["Avg Entry"]).replace("$",""))
                            if short_entry > long_entry:
                                strat = "Bull Put Spread"
                            else:
                                strat = "Bear Put Spread"
                        else:
                            strat = "Vertical Spread"
                        # Max loss: width of strikes × qty × 100 (approximate)
                        try:
                            strikes = sorted([
                                float(str(l["Strike"]).replace("$","")) for l in legs
                            ])
                            width = abs(strikes[-1] - strikes[0])
                            qty = int(long_legs[0]["Qty"])
                            net_credit_or_debit = sum(
                                (1 if "Long" in l["Asset"] else -1) *
                                float(str(l["Avg Entry"]).replace("$","")) * int(l["Qty"]) * 100
                                for l in legs
                            )
                            if net_credit_or_debit < 0:  # net credit received
                                max_loss_est = width * qty * 100 + net_credit_or_debit
                            else:                         # net debit paid
                                max_loss_est = abs(net_credit_or_debit)
                        except Exception:
                            max_loss_est = abs(net_pl) * 3
                    else:
                        strat = f"{len(legs)}-Leg Strategy"
                        max_loss_est = abs(net_pl) * 2

                    # Loss % of estimated max loss
                    loss_pct = abs(net_pl) / max(max_loss_est, 1) * 100 if net_pl < 0 else 0.0

                    # Urgency scoring
                    if net_pl > 0:
                        if dte <= 21:
                            urgency = "🔴 URGENT"; urgency_color = "#f85149"
                            action  = f"Take profit now — only {dte} DTE left. Lock in gains before theta destroys them."
                        else:
                            urgency = "🟢 HOLD"; urgency_color = "#3fb950"
                            action  = f"Position profitable. Target 50% of max profit, then close. Monitor at 21 DTE."
                    elif net_pl < 0:
                        if dte <= 14:
                            urgency = "🔴 URGENT"; urgency_color = "#f85149"
                            action  = f"Only {dte} DTE — close immediately to stop theta burn. Don't wait for recovery."
                        elif loss_pct > 75:
                            urgency = "🔴 CLOSE"; urgency_color = "#f85149"
                            action  = f"At {loss_pct:.0f}% of max loss. Cut position — risk/reward no longer favors holding."
                        elif loss_pct > 40:
                            urgency = "🟡 ROLL"; urgency_color = "#d29922"
                            action  = f"At {loss_pct:.0f}% of max loss with {dte} DTE. Consider rolling out or down for credit."
                        elif dte <= 21:
                            urgency = "🟡 MONITOR"; urgency_color = "#d29922"
                            action  = f"{dte} DTE approaching. Set stop: close if loss exceeds 50% of max before expiry."
                        else:
                            urgency = "🟡 WATCH"; urgency_color = "#d29922"
                            action  = f"Down {loss_pct:.0f}% of max loss. Hold if thesis intact; close if stock breaks key level."
                    else:
                        urgency = "⚪ FLAT"; urgency_color = "#8b949e"
                        action  = "At breakeven. Let position develop — reassess at 21 DTE."

                    results.append({
                        "key":        key,
                        "underlying": underlying,
                        "strategy":   strat,
                        "expiry":     expiry,
                        "dte":        dte,
                        "net_pl":     net_pl,
                        "max_loss":   max_loss_est,
                        "loss_pct":   loss_pct,
                        "legs":       legs,
                        "urgency":    urgency,
                        "color":      urgency_color,
                        "action":     action,
                    })

                # Sort: most urgent first (by urgency color, then largest loss)
                priority = {"#f85149": 0, "#d29922": 1, "#3fb950": 2, "#8b949e": 3}
                results.sort(key=lambda x: (priority.get(x["color"], 9), x["net_pl"]))
                return results

            # Check cache freshness (1 hour TTL)
            _now_ts = datetime.now(timezone.utc)
            _cache_age = (
                (_now_ts - st.session_state.opt_analysis_ts).total_seconds()
                if st.session_state.opt_analysis_ts else 9999
            )
            _needs_refresh = _cache_age > _OPT_ANALYSIS_TTL

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">🧠 Portfolio Intelligence</div>', unsafe_allow_html=True)

            _ai_c1, _ai_c2 = st.columns([5, 1])
            if st.session_state.opt_analysis_ts:
                _age_min = int(_cache_age / 60)
                _ai_c1.caption(
                    f"Last analyzed: **{_age_min}m ago** · Auto-refreshes every hour"
                    f"{' — 🔄 Refresh due' if _needs_refresh else ''}"
                )
            if _ai_c2.button("🔄 Re-analyze", key="reanalyze_opts", use_container_width=True) or _needs_refresh:
                st.session_state.opt_analysis_cache = _build_opt_analysis(_opt_rows_port)
                st.session_state.opt_analysis_ts    = _now_ts

            _analysis = st.session_state.opt_analysis_cache
            if not _analysis:
                # First load — run immediately
                st.session_state.opt_analysis_cache = _build_opt_analysis(_opt_rows_port)
                st.session_state.opt_analysis_ts    = _now_ts
                _analysis = st.session_state.opt_analysis_cache

            # Render each strategy card
            _total_net_pl = sum(a["net_pl"] for a in _analysis)
            _total_color  = "#3fb950" if _total_net_pl >= 0 else "#f85149"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:16px;margin-bottom:12px'>"
                f"<span style='font-size:13px;color:#8b949e'>Portfolio Net P/L:</span>"
                f"<span style='font-size:18px;font-weight:800;color:{_total_color}'>"
                f"${ _total_net_pl:+,.2f}</span></div>",
                unsafe_allow_html=True,
            )

            for _ana in _analysis:
                _pl_color = "#3fb950" if _ana["net_pl"] >= 0 else "#f85149"
                _card_bg  = "rgba(63,185,80,0.05)" if _ana["net_pl"] >= 0 else "rgba(248,81,73,0.05)"
                if "URGENT" in _ana["urgency"] or "CLOSE" in _ana["urgency"]:
                    _card_bg = "rgba(248,81,73,0.09)"
                _loss_bar = ""
                if _ana["loss_pct"] > 0:
                    _lb_w = min(int(_ana["loss_pct"]), 100)
                    _lb_c = "#f85149" if _lb_w > 60 else "#d29922"
                    _loss_bar = (
                        f"<div style='margin:6px 0 2px;'>"
                        f"<div style='font-size:10px;color:#8b949e;margin-bottom:2px'>"
                        f"Max loss consumed: {_ana['loss_pct']:.0f}%</div>"
                        f"<div style='background:rgba(255,255,255,0.07);border-radius:3px;height:5px'>"
                        f"<div style='width:{_lb_w}%;background:{_lb_c};height:100%;border-radius:3px'></div>"
                        f"</div></div>"
                    )

                st.markdown(
                    f"<div style='background:{_card_bg};border:1px solid {_ana['color']}40;"
                    f"border-left:4px solid {_ana['color']};border-radius:10px;"
                    f"padding:14px 18px;margin:8px 0'>"
                    f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap'>"
                    f"<span style='font-size:15px;font-weight:800;color:#e6edf3'>{_ana['underlying']}</span>"
                    f"<span style='font-size:11px;color:#8b949e;background:rgba(255,255,255,0.06);"
                    f"border-radius:4px;padding:2px 8px'>{_ana['strategy']}</span>"
                    f"<span style='font-size:11px;color:#8b949e'>Exp: {_ana['expiry']} · {_ana['dte']} DTE</span>"
                    f"<span style='font-size:13px;font-weight:700;color:{_pl_color};margin-left:auto'>"
                    f"${_ana['net_pl']:+,.2f}</span>"
                    f"<span style='font-size:12px;font-weight:700;color:{_ana['color']}'>{_ana['urgency']}</span>"
                    f"</div>"
                    f"{_loss_bar}"
                    f"<div style='font-size:12px;color:#c9d1d9;margin-top:8px;line-height:1.6'>"
                    f"📌 {_ana['action']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div style='font-size:10px;color:#484f57;margin-top:12px'>"
                "⚠️ Informational analysis only — not financial advice. Always verify with your broker."
                "</div>",
                unsafe_allow_html=True,
            )
