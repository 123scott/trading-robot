# -*- coding: utf-8 -*-
"""
entries_v3_paper.py

Live paper-trading harness for XAUUSD_LOWFREQ v3 (src/entries_v3.py),
same architecture as src/entries_v2_paper.py: re-runs the real
entries_v3.simulate() against live-polled Deriv (XAUUSD_DERIV) H1/daily
candles, logs any newly-closed trade, places no real orders, needs no
broker credentials. Confirms this satisfies the round's "must work with
--paper without triggering the DNS/network bug" requirement -- that bug
(silently-swallowed Deriv API errors causing an infinite blind reconnect
loop) was already fixed in src/data_deriv.py; this harness uses the same
polling-based fetch_deriv_candles_async() path entries_v2_paper.py
already validated as unaffected by it (only the live tick-SUBSCRIBE path
was broken, not historical/candle fetching).

Given the backtested frequency here is roughly 3-4 trades/YEAR (not per
week -- this design is far more selective than v1 or v2, by the
deliberate stacking of four simultaneous filters), polls hourly
(matching H1 bar close) rather than every 30 minutes -- there is no
value in polling faster than the execution timeframe itself resolves.

Logs to data/entries_v3_paper_trades.csv -- a third, separate file from
both data/paper_trades.csv (v1) and data/entries_v2_paper_trades.csv
(v2), each with its own schema and each already written by its own
already-running process; sharing any of them would corrupt all three.

Usage:
    python -m src.entries_v3_paper --max-iterations 1   # smoke test
    python -m src.entries_v3_paper                       # run until Ctrl+C
    python -m src.entries_v3_paper_stats                 # print current stats vs backtest
"""

from __future__ import annotations

import asyncio
import csv
import os
import time
from datetime import datetime, timezone
from typing import Optional

from src import memory
from src.data_deriv import fetch_deriv_candles_async
from src.entries_v3 import TradeRecordV3, simulate, DEFAULT_COSTS

PAPER_LOG_PATH = os.path.join(memory.DATA_DIR, "entries_v3_paper_trades.csv")
PAPER_LOG_HEADER = ["logged_at", "pair", "direction", "entry_time", "entry_price", "stop", "target",
                     "rrr", "exit_time", "exit_price", "exit_reason", "pnl",
                     "running_trades", "running_win_rate_pct", "running_pf", "running_net_pnl_pct"]

H1_WINDOW = 3000   # ~4 months of H1 bars -- comfortably covers this strategy's own swing/FVG lookback needs
DAILY_WINDOW = 730  # ~2 years, plenty for D1 BOS/CHoCH structure to be well-established, not cold-started


def _ensure_paper_log() -> None:
    os.makedirs(memory.DATA_DIR, exist_ok=True)
    if not os.path.exists(PAPER_LOG_PATH):
        with open(PAPER_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(PAPER_LOG_HEADER)


def _read_logged_entry_times() -> set:
    _ensure_paper_log()
    with open(PAPER_LOG_PATH, newline="", encoding="utf-8") as f:
        return {row["entry_time"] for row in csv.DictReader(f)}


def _running_stats(all_pnls: list, notional: float) -> dict:
    n = len(all_pnls)
    if n == 0:
        return {"n": 0, "win_rate_pct": 0.0, "pf": None, "net_pnl_pct": 0.0}
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    return {"n": n, "win_rate_pct": len(wins) / n * 100,
            "pf": (gross_profit / gross_loss) if gross_loss > 0 else None,
            "net_pnl_pct": sum(all_pnls) / notional * 100}


def _log_trade(t: TradeRecordV3, symbol: str, running: dict) -> None:
    _ensure_paper_log()
    with open(PAPER_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(), symbol, t.direction.value,
            t.entry_time.isoformat(), f"{t.entry_price:.5f}", f"{t.stop:.5f}", f"{t.target:.5f}",
            f"{t.rrr:.3f}", t.exit_time.isoformat(), f"{t.exit_price:.5f}", t.exit_reason, f"{t.pnl:.2f}",
            running["n"], f"{running['win_rate_pct']:.2f}",
            f"{running['pf']:.3f}" if running["pf"] is not None else "undef",
            f"{running['net_pnl_pct']:.2f}",
        ])


async def run_paper_mode(symbol: str = "XAUUSD_DERIV", notional: float = 10_000.0,
                          poll_seconds: float = 3600.0, max_iterations: Optional[int] = None,
                          log=print) -> None:
    log(f"[ENTRIES_V3 PAPER] HTF alignment + session filter + liquidity-sweep/FVG confluence + "
        f"structural RRR>=1.5 SL/TP. Backtested frequency: ~3-4 trades/YEAR -- long silences between "
        f"entries are expected, not a stall.")
    log(f"[ENTRIES_V3 PAPER] Data source: Deriv ({symbol}), H1 execution / D1 trend. "
        f"Logging to {PAPER_LOG_PATH}\n")

    logged_entry_times = _read_logged_entry_times()
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        try:
            h1 = await fetch_deriv_candles_async(symbol=symbol, interval="1h", limit=H1_WINDOW)
            daily = await fetch_deriv_candles_async(symbol=symbol, interval="1d", limit=DAILY_WINDOW)
            trades = simulate(h1, daily, notional, DEFAULT_COSTS)
        except Exception as e:
            log(f"[ENTRIES_V3 PAPER] Poll failed ({type(e).__name__}: {e}), will retry next poll.")
            trades = []

        new_trades = [t for t in trades if t.entry_time.isoformat() not in logged_entry_times]
        if new_trades:
            with open(PAPER_LOG_PATH, newline="", encoding="utf-8") as f:
                existing_pnls = [float(row["pnl"]) for row in csv.DictReader(f)]
            for t in sorted(new_trades, key=lambda t: t.exit_time):
                existing_pnls.append(t.pnl)
                running = _running_stats(existing_pnls, notional)
                _log_trade(t, symbol, running)
                logged_entry_times.add(t.entry_time.isoformat())
                pf_str = f"{running['pf']:.2f}" if running["pf"] is not None else "undef"
                log(f"[ENTRIES_V3 PAPER] NEW closed trade: {t.direction.value} {t.entry_time.date()} "
                    f"@ {t.entry_price:.2f} (rrr={t.rrr:.2f}) -> {t.exit_time.date()} @ {t.exit_price:.2f} "
                    f"({t.exit_reason}), pnl {t.pnl:+.2f}. Running: n={running['n']} "
                    f"win%={running['win_rate_pct']:.1f} pf={pf_str} net%={running['net_pnl_pct']:+.2f}\n")
        else:
            log(f"[ENTRIES_V3 PAPER] Poll {iteration}: no newly-closed trades.")

        if max_iterations is None or iteration < max_iterations:
            await asyncio.sleep(poll_seconds)

    log("[ENTRIES_V3 PAPER] Stopped.")


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live paper-trading harness for XAUUSD_LOWFREQ v3 (Deriv-sourced, no real orders).")
    parser.add_argument("--symbol", default="XAUUSD_DERIV")
    parser.add_argument("--notional", type=float, default=10_000.0)
    parser.add_argument("--poll-seconds", type=float, default=3600.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(run_paper_mode(symbol=args.symbol, notional=args.notional,
                                poll_seconds=args.poll_seconds, max_iterations=args.max_iterations))


if __name__ == "__main__":
    _cli()
