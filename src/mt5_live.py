# -*- coding: utf-8 -*-
"""
mt5_live.py

*** WINDOWS-ONLY, UNTESTED -- SAME CAVEAT AS mt5_executor.py ***
Written and reviewed against the documented MT5 API; never executed,
because the `MetaTrader5` package has no macOS/Linux build. Run
`python -m src.mt5_executor` (the connectivity smoke test) successfully
on the real Windows/MT5 machine FIRST, before running this.

Connects the existing strategy + memory decision logic (the same
detect_crossovers / PositionTracker / memory.check_memory used by
src.replay and src.live_monitor) to a live MT5 terminal and places real
orders against it -- but `mt5_executor.connect()` hard-refuses any
account that isn't flagged as a demo account, and this module never
passes `allow_live=True`. In practice that means: everything this script
does trades demo/paper money by construction, not because of a flag you
have to remember to set, but because there is no code path here that can
touch a real-money account. To ever change that would require deliberately
editing mt5_executor.connect()'s call site -- a real decision, not an
accident.

This is a NEW risk-management layer, not a replication of the backtest:
the backtested strategy (src/trading_robot.py) has no stop-loss -- it
exits purely on the opposite crossover signal (see the "no explicit
stop-loss" caveat in data/performance_report.md). Sending real orders
(even demo ones) with no protective stop is bad practice regardless of
what the backtest did, so every order placed here carries an ATR-based
SL/TP. This means live/demo results will NOT exactly reproduce the
historical backtest numbers -- that's intentional, not a bug.

Position sizing is risk-based (risk a fixed dollar amount per trade,
sized against the ATR stop distance via mt5_executor.calc_lot_size),
which is different from the backtest's notional-based sizing
(trading_robot.py buys/sells a fixed dollar notional regardless of
distance to stop). Same reasoning: notional-only sizing has no concept
of per-trade risk, which is fine for a backtest comparing strategies but
not for real order placement.

Trade decisions are logged to data/mt5_live_trades.csv -- kept separate
from data/ledger.csv (same isolation pattern as live_monitor.py's
paper_trades.csv) so live/demo activity never contaminates the memory
system's backtest-derived history. Memory lookups (memory.check_memory)
still read the shared data/ledger.csv, i.e. this benefits from
backtest-learned losses without writing back into that ledger.

Usage (on the Windows/MT5 machine, after `python -m src.mt5_executor`
has been verified to connect successfully):
    python -m src.mt5_live --symbol XAUUSD --mt5-symbol XAUUSD --notional 100 --risk-pct 1.0
    python -m src.mt5_live --symbol XAUUSD --mt5-symbol XAUUSD.a --max-iterations 1   # smoke test, one poll then exit

Broker symbol names vary (e.g. "XAUUSD" vs "XAUUSD.a" vs "GOLD") -- check
your broker's Market Watch panel for the exact tradeable name and pass it
via --mt5-symbol; --symbol stays as the internal name used for cost
lookups (market_data.py) and memory/ledger tagging so it lines up with
the rest of this project's XAUUSD/USDJPY conventions.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

from src.candle import Candle
from src.indicators import atr as compute_atr
from src.backtest_structures import detect_crossovers
from src.backtest_entries import Action, PositionTracker
from src import memory
from src import mt5_executor

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

MAGIC = 202607
LIVE_LOG_PATH = os.path.join(memory.DATA_DIR, "mt5_live_trades.csv")
LIVE_LOG_HEADER = ["timestamp", "symbol", "mt5_symbol", "action", "price", "lot",
                    "sl", "tp", "ticket", "reason", "outcome", "pnl", "comment"]

ATR_PERIOD = 14
DEFAULT_ATR_SL_MULT = 2.0
DEFAULT_ATR_TP_MULT = 3.0
DEFAULT_RISK_PCT = 1.0  # % of account balance risked per trade, via calc_lot_size


def _ensure_live_log() -> None:
    os.makedirs(memory.DATA_DIR, exist_ok=True)
    if not os.path.exists(LIVE_LOG_PATH):
        with open(LIVE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LIVE_LOG_HEADER)


def _log_live_trade(symbol: str, mt5_symbol: str, action: str, price: Optional[float], lot: Optional[float],
                     sl: Optional[float], tp: Optional[float], ticket: Optional[int], reason: str,
                     outcome: str, pnl: float, comment: str) -> None:
    _ensure_live_log()
    ts = datetime.now(timezone.utc).isoformat()
    with open(LIVE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, symbol, mt5_symbol, action,
                                 f"{price:.5f}" if price is not None else "",
                                 f"{lot:.2f}" if lot is not None else "",
                                 f"{sl:.5f}" if sl is not None else "",
                                 f"{tp:.5f}" if tp is not None else "",
                                 ticket if ticket is not None else "",
                                 reason, outcome, f"{pnl:.2f}", comment])


def _rates_to_candles(rates) -> List[Candle]:
    """Converts MetaTrader5's numpy structured array (from copy_rates_from_pos) to our Candle type."""
    return [Candle(open_time=int(r["time"]) * 1000, open=float(r["open"]), high=float(r["high"]),
                    low=float(r["low"]), close=float(r["close"]), volume=float(r["tick_volume"]))
            for r in rates]


def _fetch_daily_history(mt5_symbol: str, count: int) -> List[Candle]:
    """Fetches the most recent `count` COMPLETED daily bars (pos=1 skips the still-forming current day)."""
    rates = mt5.copy_rates_from_pos(mt5_symbol, mt5.TIMEFRAME_D1, 1, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"copy_rates_from_pos returned no data for {mt5_symbol}: {mt5.last_error()}")
    return _rates_to_candles(rates)


def run_live(symbol: str = "XAUUSD", mt5_symbol: Optional[str] = None, notional: float = 10_000.0,
             risk_pct: float = DEFAULT_RISK_PCT, atr_sl_mult: float = DEFAULT_ATR_SL_MULT,
             atr_tp_mult: float = DEFAULT_ATR_TP_MULT, seed_candles: int = 60, poll_seconds: float = 3600.0,
             max_iterations: Optional[int] = None, env_path: Optional[str] = None, log=print) -> None:
    """
    Connects to MT5 (demo-only -- see module docstring), fast-forwards
    strategy state through recent history, then polls for newly-completed
    daily bars and places real (demo) orders on confirmed, memory-approved
    signals. `notional` is used only for the memory system's price-match
    bookkeeping (mirrors how the rest of this project tags decisions);
    actual order size comes from `risk_pct` + the ATR stop distance.
    """
    mt5_symbol = mt5_symbol or symbol
    info = mt5_executor.connect(env_path)
    log(f"[MT5 LIVE] Connected. Login {info.get('login')} on {info.get('server')} "
        f"(demo, balance {info.get('balance')} {info.get('currency')}). Trading {mt5_symbol} as '{symbol}'.\n")

    try:
        history = _fetch_daily_history(mt5_symbol, seed_candles)
        log(f"[MT5 LIVE] Seeded {len(history)} daily candles through "
            f"{datetime.fromtimestamp(history[-1].open_time / 1000, tz=timezone.utc).date()}.")

        tracker = PositionTracker()
        for sig in detect_crossovers(history):
            intent = tracker.next_intent(sig)
            if intent:
                tracker.apply(intent.action)
        last_seen_open_time = history[-1].open_time
        iteration = 0

        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            latest = _fetch_daily_history(mt5_symbol, 1)
            if latest and latest[0].open_time > last_seen_open_time:
                history.append(latest[0])
                last_seen_open_time = latest[0].open_time
                log(f"[MT5 LIVE] New daily candle: "
                    f"{datetime.fromtimestamp(latest[0].open_time / 1000, tz=timezone.utc).date()} "
                    f"O{latest[0].open} H{latest[0].high} L{latest[0].low} C{latest[0].close}")

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
                            _log_live_trade(symbol, mt5_symbol, "SKIP", intent.price, None, None, None,
                                             None, intent.reason, "SKIPPED", 0.0, "memory-downgraded")
                            log(f"  => [MT5 LIVE] SKIPPED. No order placed.\n")

                        elif final_action == "BUY":
                            atr_vals = compute_atr(history, ATR_PERIOD)
                            a = atr_vals[-1]
                            if not a or a <= 0:
                                log("  => [MT5 LIVE] SKIPPED: ATR not yet available (insufficient history).\n")
                            else:
                                tick = mt5.symbol_info_tick(mt5_symbol)
                                entry_est = tick.ask if tick else intent.price
                                sl = entry_est - atr_sl_mult * a
                                tp = entry_est + atr_tp_mult * a
                                balance = mt5_executor.get_balance()
                                risk_amount = balance * (risk_pct / 100.0)
                                lot = mt5_executor.calc_lot_size(mt5_symbol, risk_amount, atr_sl_mult * a)
                                result = mt5_executor.place_market_order(mt5_symbol, "buy", lot, sl=sl, tp=tp, magic=MAGIC)
                                tracker.apply(Action.BUY)
                                if result.success:
                                    log(f"  => [MT5 LIVE] BUY {lot} lots @ {result.price} (SL {sl:.2f} / TP {tp:.2f}, "
                                        f"ticket {result.ticket}).\n")
                                else:
                                    log(f"  => [MT5 LIVE] BUY FAILED: {result.comment} (retcode {result.retcode}).\n")
                                _log_live_trade(symbol, mt5_symbol, "BUY", result.price, lot, sl, tp,
                                                 result.ticket, intent.reason,
                                                 "OPEN" if result.success else "FAILED", 0.0, result.comment)

                        elif final_action == "SELL":
                            pos = mt5_executor.get_open_position(mt5_symbol, magic=MAGIC)
                            tracker.apply(Action.SELL)
                            if pos is None:
                                log("  => [MT5 LIVE] SELL signal but no tracked open position found -- nothing to close.\n")
                                _log_live_trade(symbol, mt5_symbol, "SELL", intent.price, None, None, None,
                                                 None, intent.reason, "NO_POSITION", 0.0, "")
                            else:
                                entry_price, volume = pos["price_open"], pos["volume"]
                                result = mt5_executor.close_position(mt5_symbol, pos["ticket"], volume,
                                                                      "buy", magic=MAGIC, comment="amaro-bot-close")
                                sym_info = mt5.symbol_info(mt5_symbol)
                                pnl = 0.0
                                if result.success and sym_info and sym_info.trade_tick_size:
                                    tick_value_per_unit = sym_info.trade_tick_value / sym_info.trade_tick_size
                                    pnl = (result.price - entry_price) * tick_value_per_unit * volume
                                outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")
                                if result.success:
                                    log(f"  => [MT5 LIVE] SELL (closed ticket {pos['ticket']}) @ {result.price}, "
                                        f"entry {entry_price}, est. P&L {pnl:.2f} (cross-check against MT5 terminal history).\n")
                                else:
                                    log(f"  => [MT5 LIVE] CLOSE FAILED: {result.comment} (retcode {result.retcode}).\n")
                                _log_live_trade(symbol, mt5_symbol, "SELL", result.price, volume, None, None,
                                                 pos["ticket"], intent.reason,
                                                 outcome if result.success else "FAILED", pnl, result.comment)

            if max_iterations is None or iteration < max_iterations:
                time.sleep(poll_seconds)

    finally:
        mt5_executor.disconnect()
        log("[MT5 LIVE] Disconnected.")


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Live MT5 demo execution of the crossover+memory strategy. Refuses non-demo accounts.")
    parser.add_argument("--symbol", default="XAUUSD", help="Internal symbol name for cost/memory/ledger tagging.")
    parser.add_argument("--mt5-symbol", default=None,
                         help="Broker's actual tradeable symbol name if different (e.g. XAUUSD.a) -- check Market Watch.")
    parser.add_argument("--notional", type=float, default=10_000.0)
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT,
                         help="Percent of account balance risked per trade (position sizing).")
    parser.add_argument("--atr-sl-mult", type=float, default=DEFAULT_ATR_SL_MULT)
    parser.add_argument("--atr-tp-mult", type=float, default=DEFAULT_ATR_TP_MULT)
    parser.add_argument("--seed-candles", type=int, default=60)
    parser.add_argument("--poll-seconds", type=float, default=3600.0, help="How often to check for a new daily bar.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop after N polls (omit to run until Ctrl+C).")
    args = parser.parse_args()

    run_live(symbol=args.symbol, mt5_symbol=args.mt5_symbol, notional=args.notional, risk_pct=args.risk_pct,
              atr_sl_mult=args.atr_sl_mult, atr_tp_mult=args.atr_tp_mult, seed_candles=args.seed_candles,
              poll_seconds=args.poll_seconds, max_iterations=args.max_iterations)


if __name__ == "__main__":
    _cli()
