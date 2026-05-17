"""
options_engine.py — All options strategies for Market Sentinel.

Strategies:
  1. Call Debit Spread  — Bullish, defined risk/reward
  2. Put Debit Spread   — Bearish, defined risk/reward
  3. Long Call          — Bullish, high conviction
  4. Long Put           — Bearish / portfolio hedge
  5. Cash-Secured Put   — Neutral/Bullish, income + entry
  6. Covered Call       — Neutral income on existing shares
  7. Bull Put Spread    — Neutral/Bullish, credit collected
  8. Iron Condor        — Neutral, range-bound premium collection

Entry point: recommend_strategy(snap) → StrategyRecommendation
             analyze_strategy(symbol, strategy_id, snap, **kwargs) → StrategyResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# Metadata registry
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_META: dict[str, dict[str, str]] = {
    "call_debit_spread": {
        "name":        "Call Debit Spread",
        "direction":   "🟢 Bullish",
        "risk":        "Defined — max loss = debit paid",
        "complexity":  "⭐⭐ Intermediate",
        "when":        "Stock breaking out, trending up, want limited-risk bullish exposure",
        "legs":        "BUY ATM call + SELL OTM call (same expiry)",
        "profit":      "Width – debit (if stock closes above short strike)",
        "loss":        "Debit paid (if stock stays below long strike)",
    },
    "put_debit_spread": {
        "name":        "Put Debit Spread",
        "direction":   "🔴 Bearish",
        "risk":        "Defined — max loss = debit paid",
        "complexity":  "⭐⭐ Intermediate",
        "when":        "Stock in downtrend, overbought, distribution days rising",
        "legs":        "BUY ATM put + SELL OTM put (same expiry)",
        "profit":      "Width – debit (if stock closes below short strike)",
        "loss":        "Debit paid (if stock stays above long put strike)",
    },
    "long_call": {
        "name":        "Long Call",
        "direction":   "🟢 High-Conviction Bullish",
        "risk":        "Defined — max loss = premium",
        "complexity":  "⭐ Basic",
        "when":        "Strong breakout, elevated volume, conviction move expected",
        "legs":        "BUY ATM or slightly OTM call",
        "profit":      "Unlimited (stock price – strike – premium)",
        "loss":        "100% of premium if stock stays below strike at expiry",
    },
    "long_put": {
        "name":        "Long Put",
        "direction":   "🔴 Bearish / Hedge",
        "risk":        "Defined — max loss = premium",
        "complexity":  "⭐ Basic",
        "when":        "Stock overextended, overbought, or as portfolio hedge",
        "legs":        "BUY ATM or slightly OTM put",
        "profit":      "Unlimited downside (strike – stock price – premium)",
        "loss":        "100% of premium if stock stays above strike at expiry",
    },
    "cash_secured_put": {
        "name":        "Cash-Secured Put",
        "direction":   "⚪ Neutral / Bullish (income)",
        "risk":        "Assignment risk — must buy 100 shares if assigned",
        "complexity":  "⭐⭐ Intermediate",
        "when":        "Want to own the stock at a lower price, collect premium while waiting",
        "legs":        "SELL OTM put (hold cash equal to strike × 100)",
        "profit":      "Full premium if stock stays above strike",
        "loss":        "Assignment at strike minus premium received",
    },
    "covered_call": {
        "name":        "Covered Call",
        "direction":   "⚪ Neutral (income on existing shares)",
        "risk":        "Shares called away if stock rises above short strike",
        "complexity":  "⭐⭐ Intermediate",
        "when":        "Already long shares, stock stalling/consolidating, want monthly income",
        "legs":        "SELL OTM call against 100 existing long shares",
        "profit":      "Premium received + appreciation up to short strike",
        "loss":        "Capped upside; shares still fall if stock drops",
    },
    "bull_put_spread": {
        "name":        "Bull Put Spread",
        "direction":   "⚪ Neutral / Bullish (credit)",
        "risk":        "Defined — max loss = width – credit",
        "complexity":  "⭐⭐ Intermediate",
        "when":        "Stock above support, expect it to stay flat or rise, collect credit",
        "legs":        "SELL OTM put + BUY further OTM put (same expiry)",
        "profit":      "Full credit if stock stays above short put strike",
        "loss":        "Width – credit if stock falls below long put strike",
    },
    "iron_condor": {
        "name":        "Iron Condor",
        "direction":   "⚪ Neutral (range-bound)",
        "risk":        "Defined — max loss = wing width – credit",
        "complexity":  "⭐⭐⭐ Advanced",
        "when":        "Stock expected to trade in a range, low momentum, high IV relative to realized vol",
        "legs":        "SELL OTM put + BUY further OTM put + SELL OTM call + BUY further OTM call",
        "profit":      "Full credit if stock stays between short strikes at expiry",
        "loss":        "Wing width – credit if stock breaks beyond long strike",
    },
}

STRATEGY_IDS = list(STRATEGY_META.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyScore:
    strategy_id: str
    name: str
    score: float          # 0–10
    direction: str
    conditions_met: list[str] = field(default_factory=list)
    conditions_missed: list[str] = field(default_factory=list)
    brief: str = ""       # one-line reason


@dataclass
class StrategyRecommendation:
    top: StrategyScore
    ranked: list[StrategyScore]
    reasoning: str        # Multi-sentence AI-style analysis paragraph


@dataclass
class StrategyResult:
    strategy_id: str
    name: str
    symbol: str
    expiry: str
    legs: list[dict[str, Any]]      # [{leg, contract, strike, action, premium}]
    debit_or_credit: float          # positive = debit paid, negative = credit received
    max_profit: float
    max_loss: float
    breakeven: list[float]          # one or two breakeven prices
    rr: float
    notes: str
    risk_reward_table: list[dict]   # for display


# ─────────────────────────────────────────────────────────────────────────────
# Signal scoring — pure, no network
# ─────────────────────────────────────────────────────────────────────────────

def score_all_strategies(snap: dict[str, Any]) -> list[StrategyScore]:
    """Score all 8 strategies against the current signal snapshot. No I/O."""
    rsi           = float(snap.get("rsi", 50) or 50)
    macd          = float(snap.get("macd", 0) or 0)
    macd_sig      = float(snap.get("macd_signal", 0) or 0)
    bb_pct        = float(snap.get("bb_pct_b", 0.5) or 0.5)
    trend_ok      = bool(snap.get("trend_ok", False))
    breakout55    = bool(snap.get("breakout55", False))
    dist10        = int(snap.get("dist10", 0) or 0)
    relvol        = snap.get("relvol")
    relvol_f      = float(relvol) if isinstance(relvol, (int, float)) and relvol else 1.0

    macd_bull  = macd > macd_sig
    rsi_os     = rsi < 30
    rsi_ob     = rsi > 70
    rsi_neut   = 40 <= rsi <= 65
    low_dist   = dist10 < 3
    high_dist  = dist10 >= 4
    vol_up     = relvol_f >= 1.3
    last       = float(snap.get("last", 0) or 0)
    sma50      = float(snap.get("sma50", last) or last)
    sma200     = float(snap.get("sma200", last) or last)
    above_50   = last > sma50
    above_200  = last > sma200

    scores: list[StrategyScore] = []

    # ── 1. Call Debit Spread ─────────────────────────────────────────────────
    s, met, mis = 0.0, [], []
    if trend_ok:       s += 3.0; met.append("✅ Uptrend confirmed — price above SMA200, SMA50 ≥ SMA200")
    else:              mis.append("❌ Not in uptrend — price below key moving averages")
    if macd_bull:      s += 2.0; met.append("✅ MACD crossed above signal — bullish momentum building")
    else:              mis.append("❌ MACD below signal — momentum not confirmed")
    if breakout55:     s += 2.0; met.append("✅ 55-day breakout confirmed — strength signal")
    else:              mis.append("❌ No 55-day breakout — wait for confirmed move")
    if low_dist:       s += 1.5; met.append("✅ Low distribution days — no institutional selling")
    else:              mis.append(f"❌ {dist10} distribution days — selling pressure present")
    if not rsi_ob:     s += 1.0; met.append("✅ RSI not overbought — room to continue higher")
    else:              mis.append("❌ RSI overbought — crowded trade, entry risk elevated")
    if vol_up:         s += 0.5; met.append("✅ Above-average volume — conviction confirmed")
    brief = "Best when stock is breaking out on volume in an established uptrend."
    scores.append(StrategyScore("call_debit_spread", STRATEGY_META["call_debit_spread"]["name"], round(s, 1), "🟢 Bullish", met, mis, brief))

    # ── 2. Put Debit Spread ──────────────────────────────────────────────────
    s, met, mis = 0.0, [], []
    if not trend_ok:   s += 3.0; met.append("✅ Stock below SMA200 — downtrend favors puts")
    else:              mis.append("❌ Stock in uptrend — puts go against the trend")
    if not macd_bull:  s += 2.0; met.append("✅ MACD below signal — bearish momentum confirmed")
    else:              mis.append("❌ MACD bullish — going against momentum")
    if rsi_ob:         s += 2.0; met.append("✅ RSI overbought (>70) — reversal risk is high")
    elif rsi > 55:     s += 0.5
    if high_dist:      s += 2.0; met.append(f"✅ High distribution ({dist10} days) — institutions selling")
    else:              mis.append("❌ No heavy distribution — selling pressure not confirmed")
    if bb_pct > 0.8:   s += 1.0; met.append("✅ Price near upper BB — extended, mean-reversion likely")
    brief = "Best when stock is overbought, breaking down, with institutions selling."
    scores.append(StrategyScore("put_debit_spread", STRATEGY_META["put_debit_spread"]["name"], round(s, 1), "🔴 Bearish", met, mis, brief))

    # ── 3. Long Call ─────────────────────────────────────────────────────────
    s, met, mis = 0.0, [], []
    if trend_ok:       s += 2.0; met.append("✅ Uptrend confirmed — directional bet justified")
    else:              mis.append("❌ Not in uptrend — naked calls are very risky here")
    if breakout55:     s += 3.0; met.append("✅ Breakout55 — classic momentum entry for calls")
    else:              mis.append("❌ No breakout — calls need strong directional move")
    if macd_bull:      s += 2.0; met.append("✅ MACD bullish crossover — momentum aligned")
    else:              mis.append("❌ MACD not bullish — momentum not behind the move")
    if vol_up:         s += 2.0; met.append("✅ Elevated volume — institutional participation")
    else:              mis.append("❌ Normal volume — breakout not fully confirmed")
    if low_dist:       s += 1.0; met.append("✅ Clean chart — minimal selling pressure")
    if rsi_ob:         s -= 1.0; mis.append("⚠️ RSI overbought — chasing extended move, premium expensive")
    brief = "Best when stock is breaking out on high volume — high conviction, unlimited upside."
    scores.append(StrategyScore("long_call", STRATEGY_META["long_call"]["name"], round(max(s, 0), 1), "🟢 High-Conviction Bullish", met, mis, brief))

    # ── 4. Long Put ──────────────────────────────────────────────────────────
    s, met, mis = 0.0, [], []
    if not trend_ok:   s += 2.5; met.append("✅ Stock below SMA200 — downtrend supports put buying")
    else:              mis.append("❌ Stock in uptrend — puts fight the tape")
    if rsi_ob:         s += 3.0; met.append("✅ RSI overbought >70 — reversal candidate")
    elif rsi > 60:     s += 1.0; met.append("⚠️ RSI elevated — mild reversal risk")
    else:              mis.append("❌ RSI not overbought — puts speculative here")
    if high_dist:      s += 3.0; met.append(f"✅ {dist10} distribution days — clear selling signal")
    else:              mis.append("❌ No distribution — puts speculative without selling confirmation")
    if bb_pct > 0.85:  s += 1.5; met.append("✅ At upper BB extreme — mean reversion highly likely")
    else:              mis.append("❌ Not at BB extreme — better to wait")
    brief = "Best as a hedge or when stock is clearly overextended and distribution is high."
    scores.append(StrategyScore("long_put", STRATEGY_META["long_put"]["name"], round(max(s, 0), 1), "🔴 Bearish / Hedge", met, mis, brief))

    # ── 5. Cash-Secured Put ──────────────────────────────────────────────────
    s, met, mis = 0.0, [], []
    if trend_ok:       s += 2.0; met.append("✅ Uptrend intact — safe to be assigned at lower price")
    else:              mis.append("❌ Not in uptrend — assignment risk is high, stock may keep falling")
    if rsi_os or bb_pct < 0.2:
                       s += 3.0; met.append("✅ Oversold / at lower Bollinger Band — ideal assignment level")
    elif rsi < 45:     s += 1.0; met.append("⚠️ Mildly oversold — reasonable entry level")
    else:              mis.append("❌ Not oversold — premium will be thin for OTM puts")
    if macd_bull:      s += 1.0; met.append("✅ MACD turning bullish — timing aligns")
    if low_dist:       s += 2.0; met.append("✅ No distribution — healthy base formation")
    else:              mis.append(f"❌ {dist10} distribution days — downside risk if assigned")
    if not breakout55: s += 1.0; met.append("✅ Stock not extended — IV reasonable, put writing makes sense")
    else:              mis.append("⚠️ Stock extended post-breakout — IV elevated, wait for pullback")
    brief = "Best near support in an uptrend. You collect premium AND get willing to buy the dip."
    scores.append(StrategyScore("cash_secured_put", STRATEGY_META["cash_secured_put"]["name"], round(max(s, 0), 1), "⚪ Neutral / Bullish", met, mis, brief))

    # ── 6. Covered Call ──────────────────────────────────────────────────────
    s, met, mis = 0.0, [], []
    if rsi > 60:       s += 3.0; met.append("✅ RSI elevated — stock may consolidate, ideal for call selling")
    else:              mis.append("❌ RSI not elevated — call premium will be thin")
    if not macd_bull:  s += 2.0; met.append("✅ MACD fading — momentum slowing, calls less risky to sell")
    else:              mis.append("❌ MACD strongly bullish — selling calls here caps your upside")
    if bb_pct > 0.7:   s += 2.0; met.append("✅ Near upper Bollinger Band — excellent call selling zone")
    else:              mis.append("❌ Not near upper BB — wait for stock to run up before writing calls")
    if not breakout55: s += 1.0; met.append("✅ Range-bound behavior — covered call works best in consolidation")
    else:              mis.append("⚠️ Breakout in progress — selling calls will cap a potential multi-week run")
    if rsi_neut:       s += 2.0; met.append("✅ RSI neutral zone — steady stock, theta decays best here")
    brief = "Best in consolidation after a run-up. You already own 100+ shares and want monthly income."
    scores.append(StrategyScore("covered_call", STRATEGY_META["covered_call"]["name"], round(max(s, 0), 1), "⚪ Neutral Income", met, mis, brief))

    # ── 7. Bull Put Spread ───────────────────────────────────────────────────
    s, met, mis = 0.0, [], []
    if trend_ok:       s += 2.5; met.append("✅ Uptrend — puts you sell are unlikely to be hit")
    else:              mis.append("❌ Not in uptrend — short put strikes could be breached quickly")
    if rsi_neut:       s += 3.0; met.append("✅ RSI neutral (40-65) — stock staying in range, puts decay safely")
    elif rsi_os:       s += 1.5; met.append("✅ Oversold — bounce likely; short put well below current price")
    else:              mis.append("❌ RSI not in neutral zone — spread may be challenged")
    if macd_bull:      s += 2.0; met.append("✅ MACD bullish — upward bias keeps puts safe")
    else:              mis.append("❌ MACD not bullish — put position at risk")
    if low_dist:       s += 2.0; met.append("✅ Low distribution — no institutional selling near put strikes")
    else:              mis.append(f"❌ {dist10} distribution days — dangerous for short puts")
    if above_50:       s += 0.5; met.append("✅ Above 50-day SMA — support below the strike")
    brief = "Best in neutral, slightly rising market. Collect credit with defined risk below a support level."
    scores.append(StrategyScore("bull_put_spread", STRATEGY_META["bull_put_spread"]["name"], round(max(s, 0), 1), "⚪ Neutral / Bullish", met, mis, brief))

    # ── 8. Iron Condor ───────────────────────────────────────────────────────
    s, met, mis = 0.0, [], []
    if rsi_neut:       s += 3.0; met.append("✅ RSI neutral — range-bound behavior, condor stays safe")
    else:              mis.append("❌ RSI directional (overbought/oversold) — likely to move, condor at risk")
    if not breakout55: s += 2.0; met.append("✅ No breakout — stock respecting range, strikes have margin")
    else:              mis.append("❌ Breakout occurred — stock may trend strongly, will blow through strikes")
    if 0.3 < bb_pct < 0.7:
                       s += 3.0; met.append("✅ Price in middle of Bollinger Bands — classic iron condor zone")
    else:              mis.append("❌ Price near BB extremes — not range-bound, condor legs exposed")
    if low_dist:       s += 1.0; met.append("✅ Low distribution — neutral drift, ideal for condor")
    else:              mis.append(f"❌ {dist10} distribution days — directional risk increasing")
    if not macd_bull and abs(macd - macd_sig) < abs(macd_sig) * 0.1:
                       s += 1.0; met.append("✅ MACD near flat — momentum neutral, confirms range view")
    brief = "Best when stock has been trading in a tight range and volatility is elevated relative to expected move."
    scores.append(StrategyScore("iron_condor", STRATEGY_META["iron_condor"]["name"], round(max(s, 0), 1), "⚪ Neutral Range-Bound", met, mis, brief))

    # Sort descending by score
    scores.sort(key=lambda x: x.score, reverse=True)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(snap: dict[str, Any], top: StrategyScore, ranked: list[StrategyScore]) -> str:
    """Generate a plain-English paragraph explaining the recommendation."""
    rsi      = float(snap.get("rsi", 50) or 50)
    trend_ok = bool(snap.get("trend_ok", False))
    macd     = float(snap.get("macd", 0) or 0)
    macd_s   = float(snap.get("macd_signal", 0) or 0)
    bb_pct   = float(snap.get("bb_pct_b", 0.5) or 0.5)
    dist10   = int(snap.get("dist10", 0) or 0)
    breakout = bool(snap.get("breakout55", False))
    symbol   = str(snap.get("symbol", "this stock")).upper()

    parts = [f"**{symbol}** is currently showing "]
    signals = []
    if trend_ok:           signals.append("a confirmed uptrend (above SMA200)")
    else:                  signals.append("a broken trend (below SMA200)")
    if macd > macd_s:      signals.append("bullish MACD momentum")
    else:                  signals.append("bearish MACD momentum")
    if rsi < 30:           signals.append("an oversold RSI")
    elif rsi > 70:         signals.append("an overbought RSI (risk of reversal)")
    else:                  signals.append(f"a neutral RSI of {rsi:.0f}")
    if bb_pct < 0.2:       signals.append("price near the lower Bollinger Band (support zone)")
    elif bb_pct > 0.8:     signals.append("price near the upper Bollinger Band (extended)")
    if breakout:           signals.append("a 55-day high breakout")
    if dist10 >= 4:        signals.append(f"{dist10} distribution days (institutional selling)")

    parts.append(", ".join(signals) + ". ")

    # Core reasoning
    name = top.name
    score = top.score
    met_count = len(top.conditions_met)
    total = met_count + len(top.conditions_missed)
    parts.append(
        f"Based on these conditions, a **{name}** scores highest ({score}/10, "
        f"{met_count}/{total} conditions met). "
    )

    # Why this strategy
    if top.strategy_id == "call_debit_spread":
        parts.append("The defined-risk structure locks your max loss to the premium paid, while the short call reduces your cost basis — ideal for a trending breakout where you want exposure without naked options risk. ")
    elif top.strategy_id == "put_debit_spread":
        parts.append("The put spread lets you profit from a decline while defining your risk — far better than naked puts which can go to zero if wrong. The bearish signals here justify paying the debit. ")
    elif top.strategy_id == "long_call":
        parts.append("The breakout + volume combination is the strongest setup for an outright long call. The unlimited upside justifies the full premium cost, and the momentum signals reduce the probability of paying for time decay without movement. ")
    elif top.strategy_id == "long_put":
        parts.append("With distribution elevated and the stock extended, a long put protects against a sharp reversal. The limited risk (premium only) makes this a clean hedge position. ")
    elif top.strategy_id == "cash_secured_put":
        parts.append("Selling a put below current support lets you collect income while waiting to buy the stock at a lower entry price — effectively getting paid to be patient. ")
    elif top.strategy_id == "covered_call":
        parts.append("With momentum fading and the stock near resistance, selling an OTM call generates income while you hold shares. This is the most efficient use of an extended position. ")
    elif top.strategy_id == "bull_put_spread":
        parts.append("Neutral RSI with a bullish trend means the stock is comfortably above the put strikes you'll sell. You collect credit immediately, and the long put leg caps your downside if you're wrong. ")
    elif top.strategy_id == "iron_condor":
        parts.append("With the stock in the middle of its Bollinger range and RSI neutral, an iron condor collects premium from both sides and profits from time decay as long as the stock stays range-bound. ")

    # Alternatives
    if len(ranked) > 1:
        alt2 = ranked[1]
        parts.append(f"The next best alternative is a **{alt2.name}** (score {alt2.score}/10) — consider this if {alt2.brief.lower()}")

    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main recommendation function
# ─────────────────────────────────────────────────────────────────────────────

def recommend_strategy(snap: dict[str, Any]) -> StrategyRecommendation:
    """Pure — no I/O. Returns the top strategy with reasoning."""
    ranked = score_all_strategies(snap)
    top = ranked[0]
    reasoning = generate_reasoning(snap, top, ranked)
    return StrategyRecommendation(top=top, ranked=ranked, reasoning=reasoning)


# ─────────────────────────────────────────────────────────────────────────────
# Options chain helpers (require yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def _mid_price(row: pd.Series) -> float | None:
    bid  = float(row["bid"])  if pd.notna(row.get("bid"))       and row.get("bid", 0) > 0 else None
    ask  = float(row["ask"])  if pd.notna(row.get("ask"))       and row.get("ask", 0) > 0 else None
    last = float(row["lastPrice"]) if pd.notna(row.get("lastPrice")) and row.get("lastPrice", 0) > 0 else None
    if bid and ask:
        return (bid + ask) / 2
    return last


def _pick_expiry(options: list[str], dte_min: int, dte_max: int, target_dte: int) -> str:
    today = date.today()
    best: tuple[int, str] | None = None
    for exp in options:
        try:
            d = date.fromisoformat(exp)
        except ValueError:
            continue
        dte = (d - today).days
        if dte_min <= dte <= dte_max:
            dist = abs(dte - target_dte)
            if best is None or dist < best[0]:
                best = (dist, exp)
    if best is None:
        raise RuntimeError(f"No expiry in DTE range {dte_min}–{dte_max}")
    return best[1]


def _get_chain(symbol: str, dte_min: int, dte_max: int, target_dte: int, min_oi: int = 50):
    ticker = yf.Ticker(symbol)
    exps = list(getattr(ticker, "options", []) or [])
    if not exps:
        raise RuntimeError(f"No options chain for {symbol}")
    expiry = _pick_expiry(exps, dte_min, dte_max, target_dte)
    chain  = ticker.option_chain(expiry)
    calls  = chain.calls.copy()
    puts   = chain.puts.copy()
    # Apply OI filter only when it won't wipe out the chain
    if "openInterest" in calls.columns:
        filtered = calls[calls["openInterest"].fillna(0) >= min_oi]
        calls = filtered if not filtered.empty else calls  # fall back to unfiltered
    if "openInterest" in puts.columns:
        filtered = puts[puts["openInterest"].fillna(0) >= min_oi]
        puts = filtered if not filtered.empty else puts
    calls["strike"] = pd.to_numeric(calls["strike"], errors="coerce")
    puts["strike"]  = pd.to_numeric(puts["strike"],  errors="coerce")
    calls = calls.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    puts  = puts.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    if calls.empty:
        raise RuntimeError(f"No call strikes available for {symbol} expiry {expiry}")
    if puts.empty:
        raise RuntimeError(f"No put strikes available for {symbol} expiry {expiry}")
    return expiry, calls, puts


def _atm_strike(df: pd.DataFrame, spot: float, otm_pct: float = 0.0, direction: str = "call") -> pd.Series | None:
    target = spot * (1 + otm_pct) if direction == "call" else spot * (1 - otm_pct)
    below = df[df["strike"] <= target]
    above = df[df["strike"] >  target]
    if direction == "call":
        # ATM call: first strike at or just above spot
        row = above.iloc[0] if not above.empty else (below.iloc[-1] if not below.empty else None)
    else:
        # ATM put: first strike at or just below spot
        row = below.iloc[-1] if not below.empty else (above.iloc[0] if not above.empty else None)
    return row


def _risk_reward_table(strategy_id: str, spot: float, sr: StrategyResult) -> list[dict]:
    """Build a simple scenario table for display."""
    rows = []
    if strategy_id in ("call_debit_spread", "put_debit_spread", "bull_put_spread", "iron_condor"):
        scenarios = [spot * m for m in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]]
    else:
        scenarios = [spot * m for m in [0.80, 0.90, 1.0, 1.10, 1.20, 1.30]]
    for s in scenarios:
        pnl = _compute_pnl(strategy_id, sr, s)
        rows.append({"Stock @ Expiry": f"${s:,.2f}", "Est. P/L per share": f"${pnl:+.2f}", "Result": "✅ Profit" if pnl > 0 else ("❌ Loss" if pnl < 0 else "⚪ Breakeven")})
    return rows


def _compute_pnl(strategy_id: str, sr: StrategyResult, stock_at_expiry: float) -> float:
    if not sr.legs:
        return 0.0
    try:
        if strategy_id == "call_debit_spread":
            long_k = sr.legs[0]["strike"]; short_k = sr.legs[1]["strike"]
            intrinsic_l = max(0, stock_at_expiry - long_k)
            intrinsic_s = max(0, stock_at_expiry - short_k)
            return intrinsic_l - intrinsic_s - sr.debit_or_credit
        if strategy_id == "put_debit_spread":
            long_k = sr.legs[0]["strike"]; short_k = sr.legs[1]["strike"]
            intrinsic_l = max(0, long_k - stock_at_expiry)
            intrinsic_s = max(0, short_k - stock_at_expiry)
            return intrinsic_l - intrinsic_s - sr.debit_or_credit
        if strategy_id == "long_call":
            k = sr.legs[0]["strike"]
            return max(0, stock_at_expiry - k) - sr.debit_or_credit
        if strategy_id == "long_put":
            k = sr.legs[0]["strike"]
            return max(0, k - stock_at_expiry) - sr.debit_or_credit
        if strategy_id == "cash_secured_put":
            k = sr.legs[0]["strike"]
            return sr.debit_or_credit if stock_at_expiry >= k else (sr.debit_or_credit - (k - stock_at_expiry))
        if strategy_id == "covered_call":
            entry = sr.legs[0].get("strike", stock_at_expiry)
            k = sr.legs[1]["strike"]
            stock_pnl = stock_at_expiry - entry
            call_pnl  = sr.debit_or_credit - max(0, stock_at_expiry - k)
            return stock_pnl + call_pnl
        if strategy_id == "bull_put_spread":
            short_k = sr.legs[0]["strike"]; long_k = sr.legs[1]["strike"]
            if stock_at_expiry >= short_k:
                return -sr.debit_or_credit  # full credit kept (credit is negative)
            if stock_at_expiry <= long_k:
                return sr.max_loss
            ratio = (short_k - stock_at_expiry) / (short_k - long_k)
            return -(ratio * (short_k - long_k)) + (-sr.debit_or_credit)
        if strategy_id == "iron_condor":
            put_short_k  = sr.legs[0]["strike"]; put_long_k   = sr.legs[1]["strike"]
            call_short_k = sr.legs[2]["strike"]; call_long_k  = sr.legs[3]["strike"]
            credit = -sr.debit_or_credit
            if put_long_k < stock_at_expiry < call_short_k:
                return credit
            if stock_at_expiry <= put_long_k:
                return credit - (put_short_k - put_long_k)
            if stock_at_expiry >= call_long_k:
                return credit - (call_long_k - call_short_k)
            if stock_at_expiry < put_short_k:
                return credit - (put_short_k - stock_at_expiry)
            return credit - (stock_at_expiry - call_short_k)
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_call_debit_spread(symbol: str, spot: float, expiry: str, calls: pd.DataFrame, width: float) -> StrategyResult:
    if calls.empty:
        raise RuntimeError("No call strikes in chain")
    below = calls[calls["strike"] <= spot]
    long_row = below.iloc[-1] if not below.empty else calls.iloc[0]
    long_k = float(long_row["strike"])
    short_k = long_k + width
    short_cands = calls[calls["strike"] == short_k]
    if short_cands.empty:
        above = calls[calls["strike"] > long_k]
        if above.empty:
            raise RuntimeError("No short strike available")
        short_row = above.iloc[0]
        short_k = float(short_row["strike"])
    else:
        short_row = short_cands.iloc[0]

    lp = _mid_price(long_row); sp = _mid_price(short_row)
    if not lp or not sp:
        raise RuntimeError("Cannot determine mid prices")
    debit = max(0.01, lp - sp)
    spread = short_k - long_k
    max_p = max(0.0, spread - debit); max_l = debit; be = long_k + debit
    rr = max_p / max_l if max_l > 0 else 0
    result = StrategyResult(
        strategy_id="call_debit_spread", name="Call Debit Spread", symbol=symbol, expiry=expiry,
        legs=[
            {"leg":"LONG",  "action":"Buy to Open", "type":"Call", "strike":long_k,  "contract":str(long_row.get("contractSymbol","")),  "premium":round(lp,2)},
            {"leg":"SHORT", "action":"Sell to Open","type":"Call", "strike":short_k, "contract":str(short_row.get("contractSymbol","")), "premium":round(sp,2)},
        ],
        debit_or_credit=round(debit,2), max_profit=round(max_p,2), max_loss=round(max_l,2),
        breakeven=[round(be,2)], rr=round(rr,2),
        notes=f"Max profit if stock above ${short_k:.2f} at expiry. Close at +50% profit or 21 DTE.",
        risk_reward_table=[],
    )
    result.risk_reward_table = _risk_reward_table("call_debit_spread", spot, result)
    return result


def _analyze_put_debit_spread(symbol: str, spot: float, expiry: str, puts: pd.DataFrame, width: float) -> StrategyResult:
    if puts.empty:
        raise RuntimeError("No put strikes in chain")
    above = puts[puts["strike"] >= spot]
    long_row = above.iloc[0] if not above.empty else puts.iloc[-1]
    long_k = float(long_row["strike"])
    short_k = long_k - width
    short_cands = puts[puts["strike"] == short_k]
    if short_cands.empty:
        below = puts[puts["strike"] < long_k]
        if below.empty:
            raise RuntimeError("No short put strike available")
        short_row = below.iloc[-1]
        short_k = float(short_row["strike"])
    else:
        short_row = short_cands.iloc[0]

    lp = _mid_price(long_row); sp = _mid_price(short_row)
    if not lp or not sp:
        raise RuntimeError("Cannot determine mid prices")
    debit = max(0.01, lp - sp)
    spread = long_k - short_k
    max_p = max(0.0, spread - debit); max_l = debit; be = long_k - debit
    rr = max_p / max_l if max_l > 0 else 0
    result = StrategyResult(
        strategy_id="put_debit_spread", name="Put Debit Spread", symbol=symbol, expiry=expiry,
        legs=[
            {"leg":"LONG",  "action":"Buy to Open",  "type":"Put", "strike":long_k,  "contract":str(long_row.get("contractSymbol","")),  "premium":round(lp,2)},
            {"leg":"SHORT", "action":"Sell to Open", "type":"Put", "strike":short_k, "contract":str(short_row.get("contractSymbol","")), "premium":round(sp,2)},
        ],
        debit_or_credit=round(debit,2), max_profit=round(max_p,2), max_loss=round(max_l,2),
        breakeven=[round(be,2)], rr=round(rr,2),
        notes=f"Max profit if stock below ${short_k:.2f} at expiry. Close at +50% profit or 21 DTE.",
        risk_reward_table=[],
    )
    result.risk_reward_table = _risk_reward_table("put_debit_spread", spot, result)
    return result


def _analyze_long_call(symbol: str, spot: float, expiry: str, calls: pd.DataFrame) -> StrategyResult:
    row = _atm_strike(calls, spot, otm_pct=0.0, direction="call")
    if row is None:
        raise RuntimeError("No ATM call found")
    k = float(row["strike"]); p = _mid_price(row)
    if not p:
        raise RuntimeError("No mid price for call")
    be = k + p
    result = StrategyResult(
        strategy_id="long_call", name="Long Call", symbol=symbol, expiry=expiry,
        legs=[{"leg":"LONG","action":"Buy to Open","type":"Call","strike":k,"contract":str(row.get("contractSymbol","")),"premium":round(p,2)}],
        debit_or_credit=round(p,2), max_profit=float("inf"), max_loss=round(p,2),
        breakeven=[round(be,2)], rr=0.0,
        notes=f"Unlimited upside above ${be:.2f}. Stop: −50% of premium. Close 21 DTE.",
        risk_reward_table=[],
    )
    result.risk_reward_table = _risk_reward_table("long_call", spot, result)
    return result


def _analyze_long_put(symbol: str, spot: float, expiry: str, puts: pd.DataFrame) -> StrategyResult:
    row = _atm_strike(puts, spot, otm_pct=0.0, direction="put")
    if row is None:
        raise RuntimeError("No ATM put found")
    k = float(row["strike"]); p = _mid_price(row)
    if not p:
        raise RuntimeError("No mid price for put")
    be = k - p
    result = StrategyResult(
        strategy_id="long_put", name="Long Put", symbol=symbol, expiry=expiry,
        legs=[{"leg":"LONG","action":"Buy to Open","type":"Put","strike":k,"contract":str(row.get("contractSymbol","")),"premium":round(p,2)}],
        debit_or_credit=round(p,2), max_profit=round(k - p, 2), max_loss=round(p,2),
        breakeven=[round(be,2)], rr=round((k - p) / p, 2) if p > 0 else 0,
        notes=f"Max profit if stock goes to $0. Breakeven at ${be:.2f}. Close 21 DTE.",
        risk_reward_table=[],
    )
    result.risk_reward_table = _risk_reward_table("long_put", spot, result)
    return result


def _analyze_cash_secured_put(symbol: str, spot: float, expiry: str, puts: pd.DataFrame, otm_pct: float = 0.05) -> StrategyResult:
    row = _atm_strike(puts, spot, otm_pct=otm_pct, direction="put")
    if row is None:
        raise RuntimeError("No OTM put found")
    k = float(row["strike"]); p = _mid_price(row)
    if not p:
        raise RuntimeError("No mid price for put")
    be = k - p
    cash_req = k * 100
    return_pct = round((p / k) * 100, 2)
    result = StrategyResult(
        strategy_id="cash_secured_put", name="Cash-Secured Put", symbol=symbol, expiry=expiry,
        legs=[{"leg":"SHORT","action":"Sell to Open","type":"Put","strike":k,"contract":str(row.get("contractSymbol","")),"premium":round(p,2)}],
        debit_or_credit=-round(p,2), max_profit=round(p,2), max_loss=round(k - p,2),
        breakeven=[round(be,2)], rr=round(p / (k - p), 2) if (k - p) > 0 else 0,
        notes=f"Need ${cash_req:,.0f} cash reserved. Return if unassigned: {return_pct}%. Assigned at ${k:.2f}.",
        risk_reward_table=[],
    )
    result.risk_reward_table = _risk_reward_table("cash_secured_put", spot, result)
    return result


def _analyze_covered_call(symbol: str, spot: float, expiry: str, calls: pd.DataFrame, entry_price: float, otm_pct: float = 0.05) -> StrategyResult:
    row = _atm_strike(calls, spot, otm_pct=otm_pct, direction="call")
    if row is None:
        raise RuntimeError("No OTM call found")
    k = float(row["strike"]); p = _mid_price(row)
    if not p:
        raise RuntimeError("No mid price for call")
    max_profit_share = (k - entry_price) + p
    be = entry_price - p
    result = StrategyResult(
        strategy_id="covered_call", name="Covered Call", symbol=symbol, expiry=expiry,
        legs=[
            {"leg":"LONG STOCK","action":"Already Held","type":"Stock","strike":entry_price,"contract":symbol,"premium":round(entry_price,2)},
            {"leg":"SHORT","action":"Sell to Open","type":"Call","strike":k,"contract":str(row.get("contractSymbol","")),"premium":round(p,2)},
        ],
        debit_or_credit=-round(p,2), max_profit=round(max_profit_share,2), max_loss=round(entry_price - p, 2),
        breakeven=[round(be,2)], rr=0.0,
        notes=f"Shares called away at ${k:.2f}. Collect ${p:.2f}/share premium. Downside protected to ${be:.2f}.",
        risk_reward_table=[],
    )
    result.risk_reward_table = _risk_reward_table("covered_call", spot, result)
    return result


def _analyze_bull_put_spread(symbol: str, spot: float, expiry: str, puts: pd.DataFrame, width: float, otm_pct: float = 0.05) -> StrategyResult:
    short_row = _atm_strike(puts, spot, otm_pct=otm_pct, direction="put")
    if short_row is None:
        raise RuntimeError("No short put found")
    short_k = float(short_row["strike"])
    long_k  = short_k - width
    long_cands = puts[puts["strike"] == long_k]
    if long_cands.empty:
        below = puts[puts["strike"] < short_k]
        if below.empty:
            raise RuntimeError("No long put found")
        long_row = below.iloc[-1]
        long_k = float(long_row["strike"])
    else:
        long_row = long_cands.iloc[0]

    sp = _mid_price(short_row); lp = _mid_price(long_row)
    if not sp or not lp:
        raise RuntimeError("Cannot price bull put spread")
    credit = max(0.01, sp - lp)
    spread = short_k - long_k
    max_p = credit; max_l = max(0.0, spread - credit)
    be = short_k - credit
    rr = max_p / max_l if max_l > 0 else 0
    result = StrategyResult(
        strategy_id="bull_put_spread", name="Bull Put Spread", symbol=symbol, expiry=expiry,
        legs=[
            {"leg":"SHORT","action":"Sell to Open","type":"Put","strike":short_k,"contract":str(short_row.get("contractSymbol","")),"premium":round(sp,2)},
            {"leg":"LONG", "action":"Buy to Open", "type":"Put","strike":long_k, "contract":str(long_row.get("contractSymbol","")),"premium":round(lp,2)},
        ],
        debit_or_credit=-round(credit,2), max_profit=round(max_p,2), max_loss=round(max_l,2),
        breakeven=[round(be,2)], rr=round(rr,2),
        notes=f"Keep full credit ${credit:.2f} if stock stays above ${short_k:.2f}. Close at 50% credit captured.",
        risk_reward_table=[],
    )
    result.risk_reward_table = _risk_reward_table("bull_put_spread", spot, result)
    return result


def _analyze_iron_condor(symbol: str, spot: float, expiry: str, calls: pd.DataFrame, puts: pd.DataFrame, wing_width: float, body_pct: float = 0.05) -> StrategyResult:
    call_short_row = _atm_strike(calls, spot, otm_pct=body_pct, direction="call")
    put_short_row  = _atm_strike(puts,  spot, otm_pct=body_pct, direction="put")
    if call_short_row is None or put_short_row is None:
        raise RuntimeError("Cannot find short strikes for iron condor")

    call_short_k = float(call_short_row["strike"])
    put_short_k  = float(put_short_row["strike"])
    call_long_k  = call_short_k + wing_width
    put_long_k   = put_short_k  - wing_width

    call_long_cands = calls[calls["strike"] == call_long_k]
    put_long_cands  = puts[puts["strike"]   == put_long_k]

    # Safe fallback for long call wing: use the next available strike above short
    if not call_long_cands.empty:
        call_long_row = call_long_cands.iloc[0]
    else:
        _calls_above = calls[calls["strike"] > call_short_k]
        if _calls_above.empty:
            raise RuntimeError("No long call wing available for iron condor")
        _idx = min(1, len(_calls_above) - 1)
        call_long_row = _calls_above.iloc[_idx]

    # Safe fallback for long put wing: use the next available strike below short
    if not put_long_cands.empty:
        put_long_row = put_long_cands.iloc[0]
    else:
        _puts_below = puts[puts["strike"] < put_short_k]
        if _puts_below.empty:
            raise RuntimeError("No long put wing available for iron condor")
        put_long_row = _puts_below.iloc[-1]

    csp = _mid_price(call_short_row); clp = _mid_price(call_long_row)
    psp = _mid_price(put_short_row);  plp = _mid_price(put_long_row)
    if not all([csp, clp, psp, plp]):
        raise RuntimeError("Cannot price iron condor legs")

    call_long_k = float(call_long_row["strike"])
    put_long_k  = float(put_long_row["strike"])
    credit = max(0.01, (csp - clp) + (psp - plp))
    wing = max(call_short_k - float(call_long_row["strike"]), put_short_k - put_long_k)
    max_l = max(0.0, wing - credit)

    result = StrategyResult(
        strategy_id="iron_condor", name="Iron Condor", symbol=symbol, expiry=expiry,
        legs=[
            {"leg":"SHORT PUT", "action":"Sell to Open","type":"Put", "strike":put_short_k, "contract":str(put_short_row.get("contractSymbol","")),"premium":round(psp,2)},
            {"leg":"LONG PUT",  "action":"Buy to Open", "type":"Put", "strike":put_long_k,  "contract":str(put_long_row.get("contractSymbol","")),"premium":round(plp,2)},
            {"leg":"SHORT CALL","action":"Sell to Open","type":"Call","strike":call_short_k,"contract":str(call_short_row.get("contractSymbol","")),"premium":round(csp,2)},
            {"leg":"LONG CALL", "action":"Buy to Open", "type":"Call","strike":call_long_k, "contract":str(call_long_row.get("contractSymbol","")),"premium":round(clp,2)},
        ],
        debit_or_credit=-round(credit,2), max_profit=round(credit,2), max_loss=round(max_l,2),
        breakeven=[round(put_short_k - credit/2, 2), round(call_short_k + credit/2, 2)],
        rr=round(credit / max_l, 2) if max_l > 0 else 0,
        notes=f"Profit zone: ${put_short_k:.2f} – ${call_short_k:.2f}. Close at 50% credit or 21 DTE.",
        risk_reward_table=[],
    )
    result.risk_reward_table = _risk_reward_table("iron_condor", spot, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def analyze_strategy(
    symbol: str,
    strategy_id: str,
    snap: dict[str, Any],
    dte_min: int = 30,
    dte_max: int = 75,
    width: float = 5.0,
    min_oi: int = 50,
    entry_price: float | None = None,
) -> StrategyResult:
    """Fetch options chain and compute the specific strategy suggestion."""
    spot = float(snap.get("last", 0))
    if spot <= 0:
        raise RuntimeError("Invalid spot price in snapshot")
    target_dte = (dte_min + dte_max) // 2
    expiry, calls, puts = _get_chain(symbol, dte_min, dte_max, target_dte, min_oi)

    if strategy_id == "call_debit_spread":
        return _analyze_call_debit_spread(symbol, spot, expiry, calls, width)
    if strategy_id == "put_debit_spread":
        return _analyze_put_debit_spread(symbol, spot, expiry, puts, width)
    if strategy_id == "long_call":
        return _analyze_long_call(symbol, spot, expiry, calls)
    if strategy_id == "long_put":
        return _analyze_long_put(symbol, spot, expiry, puts)
    if strategy_id == "cash_secured_put":
        return _analyze_cash_secured_put(symbol, spot, expiry, puts)
    if strategy_id == "covered_call":
        ep = entry_price or spot
        return _analyze_covered_call(symbol, spot, expiry, calls, ep)
    if strategy_id == "bull_put_spread":
        return _analyze_bull_put_spread(symbol, spot, expiry, puts, width)
    if strategy_id == "iron_condor":
        return _analyze_iron_condor(symbol, spot, expiry, calls, puts, width)
    raise ValueError(f"Unknown strategy: {strategy_id}")
