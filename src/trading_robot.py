# -*- coding: utf-8 -*-
"""
trading_robot.py

Orchestrates one full historical replay: pulls real candles, runs the
crossover strategy, and -- depending on mode -- either executes signals
raw (mode="raw", no memory involvement at all, preserving the original
behaviour) or consults the two-file memory system before every BUY/SELL
(mode="memory", may downgrade to SKIP).

Both modes log every decision to data/ledger.csv, so the ledger accumulates
real trade history across runs regardless of mode. Only "memory" mode reads
that history back and acts on it -- "raw" mode's trading behaviour is
completely unaffected by memory, by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.data_binance import fetch_klines
from src.backtest_structures import detect_crossovers
from src.backtest_entries import Action, PositionTracker
from src import memory


@dataclass
class ReplayResult:
    trades: int
    skips: int
    wins: int
    losses: int
    total_pnl: float


def _fmt_time(open_time_ms: int) -> str:
    return datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).isoformat()


def run_replay(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500,
                mode: str = "raw", quantity: float = 0.01, log=print) -> ReplayResult:
    assert mode in ("raw", "memory"), "mode must be 'raw' or 'memory'"

    log(f"Fetching {limit} real {symbol} {interval} candles from Binance's public klines endpoint...")
    candles = fetch_klines(symbol=symbol, interval=interval, limit=limit)
    log(f"Fetched {len(candles)} candles ({_fmt_time(candles[0].open_time)} -> {_fmt_time(candles[-1].open_time)})\n")

    signals = detect_crossovers(candles)
    log(f"Detected {len(signals)} raw crossover signals (fast/slow SMA)\n")

    tracker = PositionTracker()
    trades = skips = wins = losses = 0
    total_pnl = 0.0
    open_entry_price: Optional[float] = None

    for signal in signals:
        intent = tracker.next_intent(signal)
        if intent is None:
            continue  # no state-eligible action for this signal (e.g. golden cross while already long)

        ts = _fmt_time(intent.open_time)
        final_action = intent.action.value
        log(f"--- {ts} | signal: {signal.direction.value} @ {intent.price:.2f} ---")

        if mode == "memory":
            verdict = memory.check_memory(symbol, intent.action.value, intent.price, intent.reason)
            for line in verdict.reasoning:
                log(line)
            final_action = verdict.final_action
        else:
            log(f"[RAW MODE] No memory check performed -- executing {final_action} directly.")

        if final_action == "SKIP":
            skips += 1
            memory.record_trade(symbol, "SKIP", intent.price, 0, intent.reason, mode, "SKIPPED", 0.0, ts)
            log(f"  => SKIPPED. Position remains {tracker.state.value}.\n")
            continue

        if final_action == "BUY":
            tracker.apply(Action.BUY)
            open_entry_price = intent.price
            memory.record_trade(symbol, "BUY", intent.price, quantity, intent.reason, mode, "OPEN", 0.0, ts)
            trades += 1
            log(f"  => BUY executed @ {intent.price:.2f} (qty {quantity}). Position opened.\n")

        elif final_action == "SELL":
            pnl = (intent.price - open_entry_price) * quantity if open_entry_price is not None else 0.0
            tracker.apply(Action.SELL)
            outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")
            memory.record_trade(symbol, "SELL", intent.price, quantity, intent.reason, mode, outcome, pnl, ts)
            trades += 1

            if outcome == "WIN":
                wins += 1
            elif outcome == "LOSS":
                losses += 1
                direction = memory.direction_keyword(intent.reason)
                memory.record_learning(symbol, direction, intent.price, pnl, ts)
                log("  -> Recorded a new learnings.md warning for this loss zone.")

            total_pnl += pnl
            log(f"  => SELL executed @ {intent.price:.2f} (qty {quantity}). Closed trade PnL: {pnl:.2f} ({outcome}).\n")
            open_entry_price = None

    log("=" * 60)
    log(f"Replay complete [{mode.upper()} MODE]: {trades} trade(s) executed, {skips} skipped, "
        f"{wins} win(s), {losses} loss(es), total PnL {total_pnl:.2f}")

    return ReplayResult(trades=trades, skips=skips, wins=wins, losses=losses, total_pnl=total_pnl)
