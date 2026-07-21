# -*- coding: utf-8 -*-
"""
live_monitor.py

Real-time FORWARD-TESTING (paper mode) against Deriv's live tick stream.
This module is intentionally read-only with respect to trading: it
subscribes to live prices, aggregates them into daily candles, and runs
the exact same crossover + memory decision logic used in the historical
replay -- but it NEVER places an order. There is no broker authentication,
no order-management code, and no execution path here at all. Decisions
are logged to the console and to data/paper_trades.csv (kept separate
from data/ledger.csv so live monitoring never contaminates the backtest
memory that --memory mode learns from).

Going from this to real live trading would require: authenticated Deriv
API tokens, an order-management/execution layer, position reconciliation,
and real risk controls -- none of which exist here, deliberately.

Usage (equivalent -- pick either):
    python -m src.replay --paper --symbol XAUUSD_DERIV --notional 100
    python -m src.live_monitor --paper --symbol XAUUSD_DERIV --notional 100
    python -m src.live_monitor --paper --max-seconds 60   # bounded run, for smoke-testing
"""

from __future__ import annotations

import asyncio
import csv
import os
import time
from datetime import datetime, timezone
from typing import Optional

from src import market_data
from src.data_deriv import stream_deriv_ticks, deriv_ticker, fetch_deriv_candles_async
from src.backtest_structures import detect_crossovers
from src.backtest_entries import Action, PositionTracker
from src import memory

PAPER_LOG_PATH = os.path.join(memory.DATA_DIR, "paper_trades.csv")


def _ensure_paper_log() -> None:
    os.makedirs(memory.DATA_DIR, exist_ok=True)
    if not os.path.exists(PAPER_LOG_PATH):
        with open(PAPER_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(memory.LEDGER_HEADER)


def _log_paper_trade(symbol: str, action: str, price: float, quantity: float,
                      reason: str, outcome: str, pnl: float) -> None:
    _ensure_paper_log()
    ts = datetime.now(timezone.utc).isoformat()
    with open(PAPER_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, symbol, action, f"{price:.5f}", quantity, reason, "paper", outcome, f"{pnl:.2f}"])


RECONNECT_BASE_DELAY = 2.0
RECONNECT_MAX_DELAY = 60.0


async def _run_stream_with_reconnect(ticker: str, on_tick, stop_event: asyncio.Event, log) -> None:
    """
    Wraps stream_deriv_ticks with reconnect-on-drop + exponential backoff
    (capped, with jitter), so a dropped WebSocket (keepalive timeout,
    network blip, etc. -- a real failure observed in testing) doesn't kill
    the whole monitoring process. All strategy/position state lives in the
    caller's closure via `on_tick`, so it survives a reconnect untouched.
    """
    import random

    attempt = 0
    while not stop_event.is_set():
        try:
            await stream_deriv_ticks(ticker, on_tick, stop_event)
            return  # stream_deriv_ticks only returns normally once stop_event is set
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if stop_event.is_set():
                return
            attempt += 1
            delay = min(RECONNECT_MAX_DELAY, RECONNECT_BASE_DELAY * (2 ** (attempt - 1)))
            delay *= 0.8 + 0.4 * random.random()  # +/-20% jitter to avoid thundering-herd reconnects
            log(f"[PAPER MODE] Connection dropped ({type(e).__name__}: {e}). Reconnecting in {delay:.1f}s (attempt {attempt})...")
            await asyncio.sleep(delay)


async def run_paper_mode(symbol: str = "XAUUSD_DERIV", notional: float = 10_000.0,
                          seed_candles: int = 60, max_seconds: Optional[float] = None,
                          log=print) -> None:
    """
    Seeds strategy state from recent real history, then streams live Deriv
    ticks, aggregating into daily candles and re-running crossover + memory
    checks whenever a day rolls over. Runs until `max_seconds` elapses (None
    = run until interrupted with Ctrl+C).
    """
    if market_data.source_for(symbol) != "deriv":
        raise ValueError("--paper currently only supports Deriv-sourced symbols (e.g. XAUUSD_DERIV).")

    ticker = deriv_ticker(symbol)
    spread = market_data.spread_for(symbol)

    log(f"[PAPER MODE] Seeding {seed_candles} recent {symbol} daily candles from Deriv...")
    history = await fetch_deriv_candles_async(symbol=symbol, interval="1d", limit=seed_candles)
    log(f"[PAPER MODE] Seeded through {datetime.fromtimestamp(history[-1].open_time / 1000, tz=timezone.utc).date()}. "
        f"Subscribing to live {ticker} ticks (no orders will be placed)...\n")

    tracker = PositionTracker()
    # Fast-forward the tracker through seeded history so live decisions start from the right state.
    for sig in detect_crossovers(history):
        intent = tracker.next_intent(sig)
        if intent:
            tracker.apply(intent.action)
    open_entry_price: Optional[float] = None

    current_day: Optional[str] = None
    day_open = day_high = day_low = day_close = None
    last_bid = last_ask = None
    start_time = time.monotonic()
    stop_event = asyncio.Event()

    def on_tick(tick: dict) -> None:
        nonlocal current_day, day_open, day_high, day_low, day_close, open_entry_price, last_bid, last_ask
        quote = float(tick["quote"])
        # Real bid/ask, when the feed provides them, beat the historical
        # spread approximation used in backtests -- prefer these for fills.
        last_bid = float(tick["bid"]) if "bid" in tick else None
        last_ask = float(tick["ask"]) if "ask" in tick else None
        tick_time = datetime.fromtimestamp(int(tick["epoch"]), tz=timezone.utc)
        day_key = tick_time.strftime("%Y-%m-%d")

        if current_day is None:
            current_day = day_key
            day_open = day_high = day_low = day_close = quote
            log(f"[PAPER MODE] Live tick received: {ticker} @ {quote} (bid {last_bid}, ask {last_ask}) ({tick_time.isoformat()})")
            return

        if day_key != current_day:
            # Day rolled over -- finalize yesterday's candle and evaluate the strategy on it.
            closed_epoch_ms = int(tick_time.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000) - 86_400_000
            from src.candle import Candle
            history.append(Candle(open_time=closed_epoch_ms, open=day_open, high=day_high, low=day_low, close=day_close, volume=0.0))
            log(f"[PAPER MODE] Daily candle closed for {current_day}: O{day_open} H{day_high} L{day_low} C{day_close}")

            signals = detect_crossovers(history)
            if signals and signals[-1].index == len(history) - 1:
                signal = signals[-1]
                intent = tracker.next_intent(signal)
                if intent:
                    verdict = memory.check_memory(symbol, intent.action.value, intent.price, intent.reason)
                    for line in verdict.reasoning:
                        log(line)
                    final_action = verdict.final_action

                    if final_action == "SKIP":
                        _log_paper_trade(symbol, "SKIP", intent.price, 0, intent.reason, "SKIPPED", 0.0)
                        log(f"  => [PAPER] SKIPPED. Position remains {tracker.state.value} (no order placed).\n")
                    elif final_action == "BUY":
                        fill = last_ask if last_ask is not None else intent.price + spread / 2
                        tracker.apply(Action.BUY)
                        open_entry_price = fill
                        qty = notional / fill
                        _log_paper_trade(symbol, "BUY", fill, round(qty, 6), intent.reason, "OPEN", 0.0)
                        log(f"  => [PAPER] Would BUY @ {fill:.5f} (real Deriv ask price, qty {qty:.6f}). No real order placed.\n")
                    elif final_action == "SELL":
                        fill = last_bid if last_bid is not None else intent.price - spread / 2
                        qty = notional / fill
                        pnl = (fill - open_entry_price) * qty if open_entry_price else 0.0
                        tracker.apply(Action.SELL)
                        outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")
                        _log_paper_trade(symbol, "SELL", fill, round(qty, 6), intent.reason, outcome, pnl)
                        log(f"  => [PAPER] Would SELL @ {fill:.5f}. Simulated PnL {pnl:.2f} ({outcome}). No real order placed.\n")
                        open_entry_price = None

            current_day = day_key
            day_open = day_high = day_low = day_close = quote
        else:
            day_high = max(day_high, quote)
            day_low = min(day_low, quote)
            day_close = quote

        if max_seconds is not None and (time.monotonic() - start_time) >= max_seconds:
            stop_event.set()

    stream_task = asyncio.create_task(_run_stream_with_reconnect(ticker, on_tick, stop_event, log))
    if max_seconds is not None:
        async def _timer():
            await asyncio.sleep(max_seconds)
            stop_event.set()
        timer_task = asyncio.create_task(_timer())
        await asyncio.wait([stream_task, timer_task], return_when=asyncio.FIRST_COMPLETED)
        stream_task.cancel()
    else:
        try:
            await stream_task
        except asyncio.CancelledError:
            pass

    log("[PAPER MODE] Stopped.")


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live Deriv forward-testing (paper trading only -- no real orders).")
    parser.add_argument("--paper", action="store_true", help="Required, for symmetry with `replay.py --paper`.")
    parser.add_argument("--symbol", default="XAUUSD_DERIV")
    parser.add_argument("--notional", type=float, default=10_000.0,
                         help="Simulated per-trade dollar notional for paper PnL logging (e.g. --notional 100 "
                              "to match a small demo account balance). This does NOT translate directly into a "
                              "real MT5 lot size -- see src/mt5_executor.py's calc_lot_size for that.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Stop after N seconds (omit to run until Ctrl+C).")
    args = parser.parse_args()

    asyncio.run(run_paper_mode(symbol=args.symbol, notional=args.notional, max_seconds=args.max_seconds))


if __name__ == "__main__":
    _cli()
