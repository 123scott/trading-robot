# -*- coding: utf-8 -*-
"""
replay.py

CLI entry point for the historical replay comparison:

    python -m src.replay --raw       # original, no-memory behaviour
    python -m src.replay --memory    # memory-augmented path
    python -m src.replay --reset     # clears data/ledger.csv and data/learnings.md

Supports BTCUSDT (Binance) as well as GBPUSD, USDJPY, XAUUSD (Yahoo
Finance) via --symbol. All data is real, live historical market data --
no generated or fixture candles are ever used.

Examples:
    python -m src.replay --raw --symbol BTCUSDT
    python -m src.replay --raw --symbol GBPUSD --start 2018-01-01
    python -m src.replay --memory --symbol XAUUSD --start 2018-01-01
"""

from __future__ import annotations

import argparse

from src import market_data
from src import memory
from src.trading_robot import run_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-asset crossover replay: raw vs memory-augmented.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--raw", action="store_true", help="Run the original, no-memory replay.")
    mode_group.add_argument("--memory", action="store_true", help="Run the memory-augmented replay.")
    parser.add_argument("--reset", action="store_true", help="Clear data/ledger.csv and data/learnings.md, then exit.")
    parser.add_argument("--symbol", default="BTCUSDT",
                         help=f"One of {sorted(market_data.SUPPORTED_SYMBOLS)}.")
    parser.add_argument("--interval", default=None,
                         help="Candle interval (default: 1h for Binance, 1d for yfinance -- Yahoo doesn't retain "
                              "multi-year intraday history, so multi-year backtests should use 1d).")
    parser.add_argument("--limit", type=int, default=500,
                         help="Most-recent-N-candles mode (Binance only, ignored if --start is given).")
    parser.add_argument("--start", default=None, help="ISO start date, e.g. 2018-01-01. Required for yfinance symbols.")
    parser.add_argument("--end", default=None, help="ISO end date. Defaults to present.")
    parser.add_argument("--notional", type=float, default=10_000.0,
                         help="Fixed dollar notional per trade (quantity = notional / entry price).")
    args = parser.parse_args()

    if args.reset:
        memory.reset_memory_files()
        print("Memory reset: data/ledger.csv and data/learnings.md cleared.")
        return

    if not args.raw and not args.memory:
        parser.error("one of --raw or --memory is required (or use --reset)")

    mode = "raw" if args.raw else "memory"
    run_replay(symbol=args.symbol, interval=args.interval, limit=args.limit,
               start=args.start, end=args.end, mode=mode, notional=args.notional)


if __name__ == "__main__":
    main()
