# -*- coding: utf-8 -*-
"""
trading_robot.py

Orchestrates one full historical replay: pulls real candles for any
supported symbol (BTCUSDT via Binance; GBPUSD/USDJPY/XAUUSD via Yahoo
Finance), runs the crossover strategy, and -- depending on mode -- either
executes signals raw (mode="raw", no memory involvement at all, preserving
the original behaviour) or consults the two-file memory system before
every BUY/SELL (mode="memory", may downgrade to SKIP).

Position sizing is notional-based (a fixed dollar exposure per trade,
quantity = notional / entry_price) rather than a fixed unit quantity, so
PnL is directly comparable in dollar terms across assets with wildly
different price scales (BTC ~$65k, GBPUSD ~$1.3, USDJPY ~$150, gold ~$2k).

Both modes log every decision to data/ledger.csv, so the ledger accumulates
real trade history across runs regardless of mode. Only "memory" mode reads
that history back and acts on it -- "raw" mode's trading behaviour is
completely unaffected by memory, by design.

Transaction costs (market_data.cost_profile_for()) are applied at every
fill: spread and slippage worsen the fill price (BUY fills above the
signal price, SELL fills below it), and commission is charged on both
entry and exit as a % of notional, deducted from the closed trade's PnL.
Signal detection and memory zone-matching still operate on the underlying
quote price -- only the recorded fill/PnL reflects costs. These are
illustrative cost assumptions, not live broker fee schedules -- see
market_data.COST_PROFILES.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src import market_data
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
    total_commission: float = 0.0


def _fmt_time(open_time_ms: int) -> str:
    return datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).isoformat()


def run_replay(symbol: str = "BTCUSDT", interval: Optional[str] = None, limit: int = 500,
                start: Optional[str] = None, end: Optional[str] = None,
                mode: str = "raw", notional: float = 10_000.0, log=print) -> ReplayResult:
    assert mode in ("raw", "memory"), "mode must be 'raw' or 'memory'"

    source = market_data.source_for(symbol)
    interval = interval or market_data.default_interval_for(symbol)
    costs = market_data.cost_profile_for(symbol)
    range_desc = f"{start} -> {end or 'present'}" if start else f"most recent {limit} bars"
    log(f"Fetching real {symbol} {interval} candles from {source} ({range_desc})...")
    if costs.spread or costs.slippage_pct or costs.commission_pct:
        log(f"Cost model: spread {costs.spread:.5f} (round-trip), slippage {costs.slippage_pct*100:.3f}% "
            f"per fill, commission {costs.commission_pct*100:.3f}% of notional per side")
    candles = market_data.fetch_candles(symbol=symbol, interval=interval, limit=limit, start=start, end=end)
    log(f"Fetched {len(candles)} candles ({_fmt_time(candles[0].open_time)} -> {_fmt_time(candles[-1].open_time)})\n")

    signals = detect_crossovers(candles)
    log(f"Detected {len(signals)} raw crossover signals (fast/slow SMA)\n")

    tracker = PositionTracker()
    trades = skips = wins = losses = 0
    total_pnl = 0.0
    total_commission = 0.0
    open_entry_price: Optional[float] = None
    open_quantity: Optional[float] = None
    open_entry_commission: Optional[float] = None

    for signal in signals:
        intent = tracker.next_intent(signal)
        if intent is None:
            continue  # no state-eligible action for this signal (e.g. golden cross while already long)

        ts = _fmt_time(intent.open_time)
        final_action = intent.action.value
        log(f"--- {ts} | signal: {signal.direction.value} @ {intent.price:.5f} ---")

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
            fill_price = intent.price + costs.spread / 2 + intent.price * costs.slippage_pct
            entry_commission = notional * costs.commission_pct
            tracker.apply(Action.BUY)
            open_entry_price = fill_price
            open_quantity = notional / fill_price
            open_entry_commission = entry_commission
            total_commission += entry_commission
            memory.record_trade(symbol, "BUY", fill_price, round(open_quantity, 6), intent.reason, mode, "OPEN", 0.0, ts)
            trades += 1
            log(f"  => BUY executed @ {fill_price:.5f} (qty {open_quantity:.6f}, notional ${notional:,.2f}, "
                f"commission ${entry_commission:.2f}). Position opened.\n")

        elif final_action == "SELL":
            fill_price = intent.price - costs.spread / 2 - intent.price * costs.slippage_pct
            qty = open_quantity or (notional / fill_price)
            exit_notional = fill_price * qty
            exit_commission = exit_notional * costs.commission_pct
            total_commission += exit_commission
            gross_pnl = (fill_price - open_entry_price) * qty if open_entry_price is not None else 0.0
            pnl = gross_pnl - (open_entry_commission or 0.0) - exit_commission
            tracker.apply(Action.SELL)
            outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")
            memory.record_trade(symbol, "SELL", fill_price, round(qty, 6), intent.reason, mode, outcome, pnl, ts)
            trades += 1

            if outcome == "WIN":
                wins += 1
            elif outcome == "LOSS":
                losses += 1
                direction = memory.direction_keyword(intent.reason)
                memory.record_learning(symbol, direction, intent.price, pnl, ts)
                log("  -> Recorded a new learnings.md warning for this loss zone.")

            total_pnl += pnl
            log(f"  => SELL executed @ {fill_price:.5f} (qty {qty:.6f}, commission ${exit_commission:.2f}). "
                f"Gross PnL {gross_pnl:.2f}, net of all costs: {pnl:.2f} ({outcome}).\n")
            open_entry_price = None
            open_quantity = None
            open_entry_commission = None

    log("=" * 60)
    log(f"Replay complete [{mode.upper()} MODE]: {trades} trade(s) executed, {skips} skipped, "
        f"{wins} win(s), {losses} loss(es), total PnL {total_pnl:.2f} (net of ${total_commission:.2f} total commission)")

    return ReplayResult(trades=trades, skips=skips, wins=wins, losses=losses,
                         total_pnl=total_pnl, total_commission=total_commission)
