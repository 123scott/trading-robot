# -*- coding: utf-8 -*-
"""
medfreq_strategy.py

The XAUUSD_MEDFREQ bot: intraday (H1) EMA crossover with a 200 EMA trend
filter, RSI momentum filter, and ATR-based stop-loss/take-profit exits --
a fundamentally different exit model from the lowfreq bot's
opposite-signal exit (backtest_entries.py/trading_robot.py), so this
module owns its own bar-by-bar simulation loop rather than reusing
PositionTracker.

Entry rules:
  LONG:  EMA(fast) crosses above EMA(slow), price > EMA(trend), RSI in
         [rsi_long_lo, rsi_long_hi].
  SHORT: EMA(fast) crosses below EMA(slow), price < EMA(trend), RSI in
         [rsi_short_lo, rsi_short_hi].
Only one position at a time (no pyramiding), either direction.

Exit rules: ATR-based stop-loss and take-profit, computed once at entry
from that bar's ATR and never moved. If a bar's range touches both
levels, the stop is assumed to hit first -- the standard, conservative
convention for bar-resolution backtests (H1 candles don't reveal
intra-bar path, so this avoids silently overstating performance on
ambiguous bars).

Real transaction costs (market_data.cost_profile_for) applied at both
entry and exit fills, same convention as trading_robot.py. This module
does not yet integrate the two-file memory system (that was built around
the lowfreq bot's long-only, opposite-signal-exit model) -- raw
simulation only, clearly scoped as a v1.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from src.candle import Candle
from src.indicators import ema, rsi, atr
from src import market_data

TRADING_DAYS_PER_YEAR = 252


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class MedFreqConfig:
    ema_fast: int = 8
    ema_slow: int = 21
    ema_trend: int = 200
    rsi_period: int = 14
    rsi_long: tuple = (40.0, 65.0)
    rsi_short: tuple = (35.0, 60.0)
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.75  # midpoint of the requested 2.5x-3.0x range


@dataclass
class TradeRecord:
    direction: Direction
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "SL" | "TP"
    qty: float
    pnl: float


def simulate(candles: List[Candle], config: MedFreqConfig, symbol: str,
             notional: float = 10_000.0) -> List[TradeRecord]:
    """Pure in-memory bar-by-bar simulation. No ledger I/O -- callers decide whether/how to log."""
    closes = [c.close for c in candles]
    fast = ema(closes, config.ema_fast)
    slow = ema(closes, config.ema_slow)
    trend = ema(closes, config.ema_trend)
    rsi_vals = rsi(closes, config.rsi_period)
    atr_vals = atr(candles, config.atr_period)
    costs = market_data.cost_profile_for(symbol)

    trades: List[TradeRecord] = []
    position: Optional[dict] = None

    for i in range(1, len(candles)):
        if None in (fast[i], slow[i], fast[i - 1], slow[i - 1], trend[i], rsi_vals[i], atr_vals[i]):
            continue

        c = candles[i]
        t = datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc)

        if position is not None:
            exit_price = exit_reason = None
            if position["direction"] == Direction.LONG:
                if c.low <= position["sl"]:
                    exit_price, exit_reason = position["sl"], "SL"
                elif c.high >= position["tp"]:
                    exit_price, exit_reason = position["tp"], "TP"
            else:
                if c.high >= position["sl"]:
                    exit_price, exit_reason = position["sl"], "SL"
                elif c.low <= position["tp"]:
                    exit_price, exit_reason = position["tp"], "TP"

            if exit_price is not None:
                spread_adj = costs.spread / 2 + exit_price * costs.slippage_pct
                fill = (exit_price - spread_adj) if position["direction"] == Direction.LONG else (exit_price + spread_adj)
                qty = position["qty"]
                gross = ((fill - position["entry_price"]) if position["direction"] == Direction.LONG
                         else (position["entry_price"] - fill)) * qty
                exit_commission = fill * qty * costs.commission_pct
                pnl = gross - position["entry_commission"] - exit_commission
                trades.append(TradeRecord(
                    direction=position["direction"], entry_time=position["entry_time"],
                    entry_price=position["entry_price"], exit_time=t, exit_price=fill,
                    exit_reason=exit_reason, qty=qty, pnl=pnl,
                ))
                position = None

        if position is not None:
            continue

        prev_diff = fast[i - 1] - slow[i - 1]
        curr_diff = fast[i] - slow[i]
        golden = prev_diff <= 0 and curr_diff > 0
        death = prev_diff >= 0 and curr_diff < 0
        price = c.close
        a = atr_vals[i]
        r = rsi_vals[i]

        direction = None
        if golden and price > trend[i] and config.rsi_long[0] <= r <= config.rsi_long[1]:
            direction = Direction.LONG
        elif death and price < trend[i] and config.rsi_short[0] <= r <= config.rsi_short[1]:
            direction = Direction.SHORT

        if direction is None or not a or a <= 0:
            continue

        spread_adj = costs.spread / 2 + price * costs.slippage_pct
        fill = (price + spread_adj) if direction == Direction.LONG else (price - spread_adj)
        qty = notional / fill
        entry_commission = notional * costs.commission_pct
        sl = (fill - config.atr_sl_mult * a) if direction == Direction.LONG else (fill + config.atr_sl_mult * a)
        tp = (fill + config.atr_tp_mult * a) if direction == Direction.LONG else (fill - config.atr_tp_mult * a)

        position = {"direction": direction, "entry_price": fill, "sl": sl, "tp": tp,
                    "entry_time": t, "qty": qty, "entry_commission": entry_commission}

    return trades


def compute_metrics(trades: List[TradeRecord], notional: float) -> dict:
    pnls = [t.pnl for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    win_rate = wins / n * 100 if n else 0.0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    equity = notional
    peak = notional
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)

    sharpe = None
    trades_per_year = None
    if n >= 5:
        times = sorted(t.exit_time for t in trades)
        years = (times[-1] - times[0]).total_seconds() / (365.25 * 24 * 3600)
        if years > 0:
            trades_per_year = n / years
            returns = [p / notional for p in pnls]
            mean_r = statistics.mean(returns)
            stdev_r = statistics.pstdev(returns)
            if stdev_r > 0:
                sharpe = mean_r / stdev_r * math.sqrt(trades_per_year)

    return {
        "n_trades": n, "wins": wins, "losses": losses, "win_rate_pct": win_rate,
        "profit_factor": profit_factor, "max_drawdown_pct": max_dd, "sharpe": sharpe,
        "net_pnl": sum(pnls), "net_pnl_pct": sum(pnls) / notional * 100 if notional else 0.0,
        "trades_per_year": trades_per_year,
    }
