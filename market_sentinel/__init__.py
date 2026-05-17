from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_SRC_PACKAGE = _ROOT / "src" / "market_sentinel"

__path__ = [str(_SRC_PACKAGE)] if _SRC_PACKAGE.is_dir() else []

from .analysis import run_sector_report, suggest_stocks

__all__ = ["run_sector_report", "suggest_stocks"]
