# Market Sentinel

Python 3 project to:
- Track US sector strength using SPDR sector ETFs (YTD).
- Surface strongest and weakest sectors.
- Suggest stocks from strongest sectors using a momentum + liquidity score.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
market-sentinel
```

## CLI Options

```bash
market-sentinel --strong 3 --weak 2 --picks 8
```

Trading planner entry styles:

```bash
# Hybrid trend + pullback strategy
PYTHONPATH=src python -m market_sentinel.trading_client --mode plan --entry-style hybrid --max-symbols 8

# Other supported styles: breakout55, macd, bollinger, rsi30
```

## Notes

- Sector proxies: XLB, XLE, XLI, XLP, XLV, XLU, XLC, XLRE, XLY, XLF, XLK.
- Universe for stock picks: S&P 500 constituents.
- This tool is informational only, not investment advice.

## Trading Client

Set Alpaca credentials in `src/market_sentinel/.env`:

```bash
ALPACA_KEY=your_key
ALPACA_SECRET=your_secret
ALPACA_PAPER=true

# Optional: notifications on successful BUY/SELL order submissions
NOTIFY_EMAIL_ENABLED=false
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=you@example.com
SMTP_PASSWORD=app_password
SMTP_USE_TLS=true
NOTIFY_FROM_EMAIL=you@example.com
NOTIFY_TO_EMAIL=you@example.com

# Optional: webhook message notifications (Slack/Discord/custom endpoint)
NOTIFY_WEBHOOK_URL=
```

Install deps:

```bash
pip install -e .
```

Run modes:

```bash
# RSI swing mode (default)
PYTHONPATH=src python -m market_sentinel.trading_client --mode swing --symbol AAPL

# Market-hours monitor
PYTHONPATH=src python -m market_sentinel.trading_client --mode hours

# First-hour planner (dry run, prints buy/sell plans + reasons)
PYTHONPATH=src python -m market_sentinel.trading_client --mode plan --max-symbols 8

# First-hour planner with hybrid trend + pullback entries
PYTHONPATH=src python -m market_sentinel.trading_client --mode plan --entry-style hybrid --max-symbols 8

# First-hour planner and submit orders
PYTHONPATH=src python -m market_sentinel.trading_client --mode plan --max-symbols 8 --execute

# Auto mode: wait for market open + 30 minutes, trigger once per market day
PYTHONPATH=src python -m market_sentinel.trading_client --mode auto --max-symbols 20
```

Planner output is also saved to `logs/trade_plan_YYYYMMDD.json`.
Terminal output now defaults to structured `fancy_grid` tables with colored symbols/actions.
If your terminal does not support ANSI colors, add `--no-color`.
`--mode hours` and `--mode auto` now refresh the screen in place (dashboard style) instead of continuously appending logs.
To keep classic log-style output, add `--no-refresh`.
In `--mode auto`, you also get market-state notifications: closed-market status, then pre-open reminders every 10 minutes during the last hour before open.
`--mode auto` now waits for market open + 30 minutes (10:00 AM ET), then triggers once per market day:
Market Hours table -> Triggered Trade Plan -> Current Holdings -> execute BUY/SELL actions.
If `--max-symbols` is not specified, auto mode defaults to 20 symbols.

source .venv/bin/activate && streamlit run src/market_sentinel/dashboard.py --server.port 8501
# Market-Sentinel
