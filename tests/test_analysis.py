from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import pandas as pd

from market_sentinel import analysis


class AnalysisTests(unittest.TestCase):
    def test_sector_performance_returns_sorted_rows(self) -> None:
        idx = pd.date_range("2026-01-01", periods=2, freq="D")
        closes = {
            "XLB": [100.0, 110.0],
            "XLE": [100.0, 105.0],
            "XLI": [100.0, 103.0],
            "XLP": [100.0, 99.0],
            "XLV": [100.0, 101.0],
            "XLU": [100.0, 98.0],
            "XLC": [100.0, 102.0],
            "XLRE": [100.0, 97.0],
            "XLY": [100.0, 96.0],
            "XLF": [100.0, 95.0],
            "XLK": [100.0, 94.0],
        }

        def _fake_history(symbol: str, **_: object) -> pd.DataFrame | None:
            values = closes.get(symbol)
            if values is None:
                return None
            return pd.DataFrame({"close": values}, index=idx)

        with patch("market_sentinel.analysis._safe_history", side_effect=_fake_history):
            df = analysis.sector_performance()

        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]["etf"], "XLB")
        self.assertEqual(df.iloc[-1]["etf"], "XLK")

    def test_run_sector_report_handles_empty_market_data(self) -> None:
        with patch("market_sentinel.analysis._safe_history", return_value=None):
            strong, weak, picks = analysis.run_sector_report()

        self.assertTrue(strong.empty)
        self.assertTrue(weak.empty)
        self.assertTrue(picks.empty)

    def test_sp500_universe_accepts_gics_sector_column(self) -> None:
        csv = "Symbol,GICS Sector\nAAPL,Information Technology\nJPM,Financials\n"
        response = Mock()
        response.text = csv
        response.raise_for_status = Mock()

        with patch("market_sentinel.analysis.requests.get", return_value=response):
            df = analysis._sp500_universe()

        self.assertEqual(list(df.columns), ["symbol", "sector"])
        self.assertIn("Technology", set(df["sector"]))


if __name__ == "__main__":
    unittest.main()
