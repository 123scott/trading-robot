# -*- coding: utf-8 -*-
"""
breakout_v1.py

XAUUSD Donchian-channel range-expansion breakout with an ATR chandelier
trailing stop -- a genuinely different strategy archetype from
everything else tried in this project so far. entries_v2 (pullback-to-
EMA) and entries_v3 (SMC liquidity-sweep-into-FVG) are both, at core,
mean-reversion-flavored: they wait for price to pull back/sweep against
the immediate move before entering with the higher-timeframe trend.
This is the opposite: it enters ON new extremes (range expansion), the
classic trend-CONTINUATION approach (same family as the Turtle Trading
System 1 entry), and deliberately has no fixed profit target -- exits
are managed purely by a trailing stop that only ever moves in the
trade's favor, letting winners run rather than capping them at a
structural or ATR-multiple target.

Execution timeframe: H4 (per this round's "daily/H4" brief -- H4 chosen
for materially more signal frequency than daily-only, while still being
a genuinely slower, higher-timeframe breakout system than either H1
strategy already tried). Both the Donchian channel and the ATR trail are
computed on H4 candles directly -- no separate HTF filter layer, since
range-expansion systems are traditionally single-timeframe by design
(adding a second-timeframe filter here would just be re-inventing
entries_v2/v3's HTF-alignment idea on a different execution frequency,
not testing a genuinely different hypothesis).

Fixed, standard, non-tuned parameters (consistent with this project's
anti-overfitting convention even though this round didn't mandate it):
  DONCHIAN_PERIOD = 20   -- the classic Turtle System 1 breakout length
  ATR_PERIOD = 14        -- standard default used everywhere else here
  ATR_TRAIL_MULT = 2.0   -- standard chandelier-stop multiple
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from src.candle import Candle
from src.indicators import atr

DONCHIAN_PERIOD = 20
ATR_PERIOD = 14
ATR_TRAIL_MULT = 2.0


@dataclass
class CostModel:
    spread: float = 0.25
    slippage_per_side: float = 0.05
    commission: float = 0.0


DEFAULT_COSTS = CostModel()


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeRecord:
    direction: Direction
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "TRAIL" (the only exit mechanism -- no fixed target)
    qty: float
    pnl: float


def simulate(h4_candles: List[Candle], notional: float = 10_000.0,
             costs: CostModel = DEFAULT_COSTS) -> List[TradeRecord]:
    """Pure in-memory bar-by-bar simulation. Long or short, one position at a time, ATR-trailed exit only."""
    highs = [c.high for c in h4_candles]
    lows = [c.low for c in h4_candles]
    closes = [c.close for c in h4_candles]
    times = [datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc) for c in h4_candles]
    atr_vals = atr(h4_candles, ATR_PERIOD)

    trades: List[TradeRecord] = []
    position: Optional[dict] = None

    for i in range(DONCHIAN_PERIOD, len(h4_candles)):
        a = atr_vals[i]
        t = times[i]

        if position is not None:
            # Trail: the stop only ever moves in the trade's favor, never back.
            if position["direction"] == Direction.LONG and a:
                position["trail"] = max(position["trail"], closes[i] - ATR_TRAIL_MULT * a)
            elif position["direction"] == Direction.SHORT and a:
                position["trail"] = min(position["trail"], closes[i] + ATR_TRAIL_MULT * a)

            hit = (lows[i] <= position["trail"]) if position["direction"] == Direction.LONG \
                else (highs[i] >= position["trail"])
            if hit:
                exit_price = position["trail"]
                fill = (exit_price - costs.slippage_per_side) if position["direction"] == Direction.LONG \
                    else (exit_price + costs.slippage_per_side)
                qty = position["qty"]
                gross = ((fill - position["entry_price"]) if position["direction"] == Direction.LONG
                         else (position["entry_price"] - fill)) * qty
                pnl = gross - costs.commission * qty * 2
                trades.append(TradeRecord(
                    direction=position["direction"], entry_time=position["entry_time"],
                    entry_price=position["entry_price"], exit_time=t, exit_price=fill,
                    exit_reason="TRAIL", qty=qty, pnl=pnl,
                ))
                position = None

        if position is not None or not a or a <= 0:
            continue

        window_high = max(highs[i - DONCHIAN_PERIOD:i])
        window_low = min(lows[i - DONCHIAN_PERIOD:i])
        price = closes[i]

        direction = None
        if price > window_high:
            direction = Direction.LONG
        elif price < window_low:
            direction = Direction.SHORT
        if direction is None:
            continue

        spread_adj = costs.spread / 2 + costs.slippage_per_side
        fill = (price + spread_adj) if direction == Direction.LONG else (price - spread_adj)
        qty = notional / fill
        initial_trail = (fill - ATR_TRAIL_MULT * a) if direction == Direction.LONG else (fill + ATR_TRAIL_MULT * a)

        position = {"direction": direction, "entry_price": fill, "trail": initial_trail,
                    "entry_time": t, "qty": qty}

    return trades


def compute_metrics(trades: List[TradeRecord], notional: float, risk_free_annual: float = 0.05) -> dict:
    pnls = [t.pnl for t in trades]
    n = len(pnls)
    wins_list = [p for p in pnls if p > 0]
    losses_list = [p for p in pnls if p < 0]
    win_rate = len(wins_list) / n * 100 if n else 0.0
    gross_profit, gross_loss = sum(wins_list), abs(sum(losses_list))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    avg_payoff = statistics.mean(pnls) if pnls else 0.0

    equity, peak, max_dd = notional, notional, 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)

    sharpe = None
    if n >= 2:
        times = sorted(t.exit_time for t in trades)
        years = (times[-1] - times[0]).total_seconds() / (365.25 * 86400)
        if years > 0:
            trades_per_year = n / years
            returns = [p / notional for p in pnls]
            mean_r = statistics.mean(returns)
            std_r = statistics.pstdev(returns)
            rf_per_trade = risk_free_annual / trades_per_year
            if std_r > 0:
                sharpe = (mean_r - rf_per_trade) / std_r * math.sqrt(trades_per_year)

    return {
        "n_trades": n, "win_rate_pct": win_rate, "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd, "avg_trade_payoff": avg_payoff, "sharpe": sharpe,
        "net_pnl": sum(pnls), "net_pnl_pct": sum(pnls) / notional * 100 if notional else 0.0,
    }
