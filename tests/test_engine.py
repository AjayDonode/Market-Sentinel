from __future__ import annotations

import unittest

from market_sentinel.engine import compute_signal_snapshot, decide_action

import pandas as pd


def _base_snapshot() -> dict[str, object]:
    return {
        "symbol": "TEST",
        "last": 100.0,
        "sma50": 99.0,
        "sma200": 92.0,
        "rsi": 52.0,
        "macd": 1.2,
        "macd_signal": 0.8,
        "macd_hist": 0.4,
        "bb_upper": 108.0,
        "bb_mid": 101.0,
        "bb_lower": 94.0,
        "bb_pct_b": 0.45,
        "atr": 2.0,
        "breakout55": False,
        "relvol": 1.0,
        "dist10": 1,
        "breakdown20": False,
        "trend_ok": True,
        "market_gate": True,
        "open_series": [99.0, 99.5, 99.8],
        "close_series": [99.4, 99.9, 100.0],
        "high_series": [100.5, 100.8, 101.0],
        "low_series": [98.8, 99.0, 99.4],
    }


class EngineDecisionTests(unittest.TestCase):
    def test_hybrid_buys_confirmed_breakout(self) -> None:
        snap = _base_snapshot()
        snap.update({
            "last": 110.0,
            "sma50": 102.0,
            "bb_mid": 105.0,
            "bb_pct_b": 0.9,
            "rsi": 67.0,
            "breakout55": True,
            "relvol": 1.4,
            "dist10": 1,
            "macd": 2.1,
            "macd_signal": 1.3,
        })

        action, reason = decide_action(snap, "hybrid", qty=0.0, days_held=None, atr_mult=2.5)

        self.assertEqual(action, "BUY")
        self.assertIn("Hybrid breakout", reason)

    def test_hybrid_buys_pullback_near_sma50(self) -> None:
        snap = _base_snapshot()
        snap.update({
            "last": 100.6,
            "sma50": 100.0,
            "bb_mid": 101.2,
            "bb_pct_b": 0.38,
            "rsi": 48.0,
            "breakout55": False,
            "relvol": 0.9,
            "dist10": 2,
            "macd": 0.7,
            "macd_signal": 0.4,
        })

        action, reason = decide_action(snap, "hybrid", qty=0.0, days_held=None, atr_mult=2.5)

        self.assertEqual(action, "BUY")
        self.assertIn("Hybrid pullback", reason)

    def test_hybrid_holds_when_market_gate_unknown(self) -> None:
        snap = _base_snapshot()
        snap["market_gate"] = None

        action, reason = decide_action(snap, "hybrid", qty=0.0, days_held=None, atr_mult=2.5)

        self.assertEqual(action, "HOLD")
        self.assertIn("Market gate unknown", reason)

    def test_compute_signal_snapshot_includes_signal_score(self) -> None:
        dates = pd.date_range(end=pd.Timestamp("2026-05-01"), periods=260, freq="B")
        base = pd.Series(range(len(dates)), index=dates).astype(float)
        bars = pd.DataFrame({
            "open": base + 0.2,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": pd.Series(1_000_000, index=dates),
        }, index=dates)

        snap = compute_signal_snapshot("TEST", bars, atr_mult=2.5)

        self.assertIn("signal_score", snap)
        self.assertIsInstance(snap["signal_score"], int)
        self.assertGreaterEqual(snap["signal_score"], 0)
        self.assertLessEqual(snap["signal_score"], 10)


if __name__ == "__main__":
    unittest.main()
