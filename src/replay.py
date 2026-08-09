# -*- coding: utf-8 -*-
"""
replay.py

CLI entry point for the historical replay comparison:

    python -m src.replay --raw       # original, no-memory behaviour
    python -m src.replay --memory    # memory-augmented path
    python -m src.replay --reset     # clears data/ledger.csv and data/learnings.md
    python -m src.replay --paper     # live Deriv forward-testing (paper only, no orders)

Supports BTCUSDT (Binance), GBPUSD/USDJPY/XAUUSD (Yahoo Finance), and
XAUUSD_DERIV (Deriv) via --symbol. All data is real, live historical
market data -- no generated or fixture candles are ever used.

--paper is READ-ONLY with respect to trading: it logs what the
strategy+memory system would do, to data/paper_trades.csv, and never
places a real order -- there is no broker authentication or
order-execution code anywhere in this project. It only supports symbols
with a live-ish data path: Deriv symbols (e.g. XAUUSD_DERIV) stream real
live ticks; Yahoo Finance symbols (XAUUSD, USDJPY, GBPUSD) are instead
polled periodically for new daily bars, since Yahoo's free interface has
no live feed (see src/live_monitor.py's docstring for the fidelity
difference). --symbol defaults to XAUUSD_DERIV specifically under
--paper (BTCUSDT, the default for --raw/--memory, has no live data path
and isn't in this project's active client-facing scope anyway).
Dukascopy-sourced data (used elsewhere for deep historical backtesting)
has no live capability at all and can't back --paper.

Examples:
    python -m src.replay --raw --symbol BTCUSDT
    python -m src.replay --raw --symbol GBPUSD --start 2018-01-01
    python -m src.replay --memory --symbol XAUUSD_DERIV --start 2018-01-01
    python -m src.replay --paper --max-seconds 60                  # defaults to XAUUSD_DERIV, live ticks
    python -m src.replay --paper --symbol XAUUSD --max-seconds 60  # Yahoo-polled instead
"""

from __future__ import annotations

import argparse
import asyncio

from src import market_data
from src import memory
from src.trading_robot import run_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-asset crossover replay: raw vs memory-augmented vs live paper.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--raw", action="store_true", help="Run the original, no-memory replay.")
    mode_group.add_argument("--memory", action="store_true", help="Run the memory-augmented replay.")
    mode_group.add_argument("--paper", action="store_true",
                             help="Live Deriv forward-testing (paper trading only -- no real orders placed).")
    parser.add_argument("--reset", action="store_true", help="Clear data/ledger.csv and data/learnings.md, then exit.")
    parser.add_argument("--symbol", default=None,
                         help=f"One of {sorted(market_data.SUPPORTED_SYMBOLS)}. Defaults to BTCUSDT for "
                              f"--raw/--memory, or XAUUSD_DERIV for --paper (the only symbols with a "
                              f"live/pollable data path under --paper -- see the module docstring).")
    parser.add_argument("--interval", default=None,
                         help="Candle interval (default: 1h for Binance, 1d for yfinance/Deriv).")
    parser.add_argument("--limit", type=int, default=500,
                         help="Most-recent-N-candles mode (Binance only, ignored if --start is given).")
    parser.add_argument("--start", default=None, help="ISO start date, e.g. 2018-01-01. Required for yfinance symbols.")
    parser.add_argument("--end", default=None, help="ISO end date. Defaults to present.")
    parser.add_argument("--notional", type=float, default=10_000.0,
                         help="Fixed dollar notional per trade (quantity = notional / entry price).")
    parser.add_argument("--max-seconds", type=float, default=None,
                         help="--paper only: stop after N seconds (omit to run until Ctrl+C). Useful for smoke-testing.")
    parser.add_argument("--poll-seconds", type=float, default=1800.0,
                         help="--paper with a Yahoo-sourced --symbol only: how often to check for a new daily bar.")
    args = parser.parse_args()

    if args.reset:
        memory.reset_memory_files()
        print("Memory reset: data/ledger.csv and data/learnings.md cleared.")
        return

    if args.symbol is None:
        args.symbol = "XAUUSD_DERIV" if args.paper else "BTCUSDT"

    if args.paper:
        from src.live_monitor import run_paper_mode
        asyncio.run(run_paper_mode(symbol=args.symbol, notional=args.notional,
                                    max_seconds=args.max_seconds, poll_seconds=args.poll_seconds))
        return

    if not args.raw and not args.memory:
        parser.error("one of --raw, --memory, or --paper is required (or use --reset)")

    mode = "raw" if args.raw else "memory"
    run_replay(symbol=args.symbol, interval=args.interval, limit=args.limit,
               start=args.start, end=args.end, mode=mode, notional=args.notional)


if __name__ == "__main__":
    main()
