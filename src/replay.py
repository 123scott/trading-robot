# -*- coding: utf-8 -*-
"""
replay.py

CLI entry point for the historical replay comparison:

    python -m src.replay --raw       # original, no-memory behaviour
    python -m src.replay --memory    # memory-augmented path
    python -m src.replay --reset     # clears data/ledger.csv and data/learnings.md

Both modes fetch the same real, live candles from Binance's public klines
endpoint (default BTCUSDT) -- no generated or fixture data is ever used.
"""

from __future__ import annotations

import argparse

from src import memory
from src.trading_robot import run_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT crossover replay: raw vs memory-augmented.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--raw", action="store_true", help="Run the original, no-memory replay.")
    mode_group.add_argument("--memory", action="store_true", help="Run the memory-augmented replay.")
    parser.add_argument("--reset", action="store_true", help="Clear data/ledger.csv and data/learnings.md, then exit.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--quantity", type=float, default=0.01)
    args = parser.parse_args()

    if args.reset:
        memory.reset_memory_files()
        print("Memory reset: data/ledger.csv and data/learnings.md cleared.")
        return

    if not args.raw and not args.memory:
        parser.error("one of --raw or --memory is required (or use --reset)")

    mode = "raw" if args.raw else "memory"
    run_replay(symbol=args.symbol, interval=args.interval, limit=args.limit, mode=mode, quantity=args.quantity)


if __name__ == "__main__":
    main()
