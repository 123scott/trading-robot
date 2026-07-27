# -*- coding: utf-8 -*-
"""
bot.py

Shared CLI entry point for the two named XAUUSD bot profiles -- one
codebase, two isolated configurations, per the client requirement to
keep them separable while sharing the core strategy/memory/reporting
engine:

  XAUUSD_LOWFREQ  -- the original daily SMA(7,50) crossover + two-file
                     memory system (trading_robot.py / memory.py).
                     ~2-3 trades/year. This is the bot validated in
                     data/performance_report.md's train/test section.

  XAUUSD_MEDFREQ  -- Top-Down Multi-Timeframe (MTF) model: H4 200 EMA
                     macro trend + H1 RSI(14) momentum, both forward-filled
                     (no lookahead) onto M5, which is where the EMA(8,21)
                     crossover actually triggers entries and ATR-based
                     stop-loss/take-profit are computed (medfreq_strategy.py).
                     Targets 50-75 trades/year. Real M5 data (H1/H4 are
                     pure resamples of it) from src/data_dukascopy.py
                     (Yahoo/Deriv can't supply 2018-present intraday
                     history). No memory-system integration yet (v1, raw
                     only) -- see module docstring in medfreq_strategy.py.

Both profiles log their trades into the SAME data/ledger.csv, each under
its own ledger_symbol tag, so src/report.py, src/monte_carlo.py, and
src/alpha_test.py work on either profile without modification:

    python -m src.report --symbols XAUUSD_LOWFREQ,XAUUSD_MEDFREQ
    python -m src.monte_carlo --symbol XAUUSD_MEDFREQ --mode raw

Usage:
    python -m src.bot --profile lowfreq --mode memory --start 2018-01-01
    python -m src.bot --profile medfreq --start 2018-01-01
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

from src import memory
from src.trading_robot import run_replay
from src.medfreq_strategy import MedFreqConfig, Direction, simulate
from src import data_dukascopy

REAL_SYMBOL = "XAUUSD"
LOWFREQ_LEDGER_SYMBOL = "XAUUSD_LOWFREQ"
MEDFREQ_LEDGER_SYMBOL = "XAUUSD_MEDFREQ"
LOWFREQ_FAST, LOWFREQ_SLOW = 7, 50  # selected via src/optimize.py's training-only grid search


def _clear_ledger_symbol(tag: str) -> None:
    with open(memory.LEDGER_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if r["symbol"] != tag]
    if len(kept) != len(rows):
        with open(memory.LEDGER_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=memory.LEDGER_HEADER)
            w.writeheader()
            w.writerows(kept)


def run_lowfreq(mode: str = "memory", notional: float = 10_000.0, start: str = "2018-01-01",
                 reset: bool = False, log=print) -> None:
    if reset:
        _clear_ledger_symbol(LOWFREQ_LEDGER_SYMBOL)
    log(f"=== XAUUSD_LOWFREQ (SMA {LOWFREQ_FAST}/{LOWFREQ_SLOW}, daily, {mode} mode) ===")
    run_replay(symbol=REAL_SYMBOL, start=start, mode=mode, notional=notional,
               fast_period=LOWFREQ_FAST, slow_period=LOWFREQ_SLOW,
               ledger_symbol=LOWFREQ_LEDGER_SYMBOL, log=log)


def run_medfreq(notional: float = 10_000.0, start: str = "2018-01-01",
                 reset: bool = False, log=print) -> None:
    if reset:
        _clear_ledger_symbol(MEDFREQ_LEDGER_SYMBOL)
    log(f"=== XAUUSD_MEDFREQ (H1 EMA/RSI/ATR, raw mode -- no memory yet) ===")
    candles = data_dukascopy.load_m5_candles(start)
    if not candles:
        raise RuntimeError(f"No cached M5 candles from {start}. Run: python -m src.data_dukascopy --start {start}")
    log(f"Loaded {len(candles)} real M5 candles.")

    trades = simulate(candles, MedFreqConfig(), REAL_SYMBOL, notional)
    log(f"Simulated {len(trades)} trades. Logging to ledger as {MEDFREQ_LEDGER_SYMBOL}...")

    for t in trades:
        entry_action = "BUY" if t.direction == Direction.LONG else "SELL"
        exit_action = "SELL" if t.direction == Direction.LONG else "BUY"
        reason = f"M5 EMA(8/21) {'golden' if t.direction == Direction.LONG else 'death'} cross, " \
                 f"H4 200EMA trend + H1 RSI confirmed"
        entry_ts = t.entry_time.isoformat()
        exit_ts = t.exit_time.isoformat()
        memory.record_trade(MEDFREQ_LEDGER_SYMBOL, entry_action, t.entry_price, round(t.qty, 6),
                             reason, "raw", "OPEN", 0.0, entry_ts)
        outcome = "WIN" if t.pnl > 0 else ("LOSS" if t.pnl < 0 else "BREAKEVEN")
        memory.record_trade(MEDFREQ_LEDGER_SYMBOL, exit_action, t.exit_price, round(t.qty, 6),
                             f"{reason} ({t.exit_reason})", "raw", outcome, t.pnl, exit_ts)
    log(f"Done. {len(trades)} trades logged.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the XAUUSD_LOWFREQ or XAUUSD_MEDFREQ bot profile.")
    parser.add_argument("--profile", choices=["lowfreq", "medfreq"], required=True)
    parser.add_argument("--mode", choices=["raw", "memory"], default="memory",
                         help="lowfreq only -- medfreq has no memory system yet, always raw.")
    parser.add_argument("--notional", type=float, default=10_000.0)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--reset", action="store_true", help="Clear this profile's prior ledger rows before running.")
    args = parser.parse_args()

    if args.profile == "lowfreq":
        run_lowfreq(mode=args.mode, notional=args.notional, start=args.start, reset=args.reset)
    else:
        run_medfreq(notional=args.notional, start=args.start, reset=args.reset)


if __name__ == "__main__":
    main()
