# -*- coding: utf-8 -*-
"""
entries_v3.py

XAUUSD_LOWFREQ v3 -- SMC-structural refactor of entries_v2, per this
round's brief. Reuses the root-level structures.py engine (swing
detection, BOS/CHoCH, full FVG lifecycle tracking) directly rather than
reimplementing SMC concepts -- that module was previously unconnected to
any bot in this project; this is the first thing that actually wires it
in.

Four structural filters, all fixed/non-tuned (see FIXED CONSTANTS below
-- there is deliberately no Config dataclass with a search range this
round; the brief explicitly prohibits grid-search tuning, and every
value here is a standard convention, not fit to this dataset):

  1. HTF alignment: only trade in the direction of the current D1
     BOS/CHoCH trend (structures.detect_structure_events on daily
     candles) -- the real higher-highs/higher-lows structural read, not
     a SMA proxy (the brief offered both; this uses the stronger one
     since the machinery already existed).
  2. Session filter: entries only allowed if the entry bar's hour (UTC)
     falls in [SESSION_START_UTC, SESSION_END_UTC) -- fixed London/NY
     hours, not optimized.
  3. Liquidity sweep + FVG confluence: an H1 bar that wicks beyond a
     already-confirmed H1 swing high/low and closes back on the other
     side of it (the sweep) AND whose wick overlaps an active
     (unmitigated/mitigated, not yet inverted) same-direction H1 FVG is
     the entry trigger. Order Blocks are NOT implemented (see module
     docstring in the retraining round's report) -- the brief says
     "FVG or Order Block"; FVG alone satisfies that.
  4. Structural stop/target with a mandatory 1.5:1 minimum RRR: stop
     sits beyond the sweep wick's extreme (+ a small fixed buffer);
     target is the nearest already-confirmed opposing swing level ahead
     of price (a real structural level, not an ATR multiple). No
     opposing swing yet visible, or RRR < 1.5 -> the signal is skipped,
     not forced.

NO-LOOKAHEAD, worked through carefully because structures.py wasn't
originally written for causal/streaming use (build_structure() is a
whole-series batch analysis):
  - get_swing_points() confirms a fractal at index i only once
    `swing_strength` bars AFTER it have closed -- a swing is only
    "visible" from bar (i + swing_strength) onward. Enforced everywhere
    a swing is consulted, via each SwingPoint's own index.
  - detect_fvgs() anchors a gap to candle i-1 but the gap only exists
    once candle i (index+2) has closed -- only visible from bar
    index+2 onward.
  - update_fvg_states()'s mitigated_at/filled_at/inverted_at timestamps
    are themselves causal (a transition at time T only reflects price
    action up to T), so precomputing them ONCE on the full H1 series
    and then checking "was this gap still active as of bar i" against
    those timestamps is safe and avoids an O(n^2) re-simulation --
    confirmed correct, not just assumed, before use here.
  - D1 structure_events (from build_structure on the FULL D1 series)
    are each stamped with the bar TIME the break actually closed at --
    an H1 bar only ever sees the latest D1 event whose time is strictly
    before that H1 bar's own open time (same pointer-advance pattern as
    medfreq_strategy.align_htf_to_m5, reused rather than reimplemented).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

import pandas as pd

from src.candle import Candle
import structures as smc

# ============================================================
# FIXED CONSTANTS -- standard conventions, NOT grid-searched or tuned
# ============================================================
SWING_STRENGTH = 2          # fractal width (bars each side) for both D1 and H1 swings
SESSION_START_UTC = 7       # London open
SESSION_END_UTC = 17        # New York afternoon close
MIN_RRR = 1.5                # mandatory minimum reward:risk
STOP_BUFFER = 0.10          # USD, small fixed buffer beyond the sweep wick extreme


@dataclass
class CostModelV3:
    """20-30pt / $0.20-0.30 Deriv-realistic spread per this round's brief -- midpoint used."""
    spread: float = 0.25
    slippage_per_side: float = 0.05
    commission: float = 0.0


DEFAULT_COSTS = CostModelV3()


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeRecordV3:
    direction: Direction
    entry_time: datetime
    entry_price: float
    stop: float
    target: float
    rrr: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "SL" | "TP"
    qty: float
    pnl: float


def candles_to_df(candles: List[Candle]) -> pd.DataFrame:
    idx = [datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc) for c in candles]
    return pd.DataFrame(
        {"open": [c.open for c in candles], "high": [c.high for c in candles],
         "low": [c.low for c in candles], "close": [c.close for c in candles]},
        index=pd.DatetimeIndex(idx),
    )


def _d1_trend_series(daily_candles: List[Candle]) -> List[tuple]:
    """Returns [(event_time, direction), ...] sorted by time -- the D1 trend timeline."""
    d1_df = candles_to_df(daily_candles)
    snapshot = smc.build_structure(d1_df, swing_strength=SWING_STRENGTH)
    return [(e.time.to_pydatetime(), e.direction) for e in snapshot.structure_events]


def _trend_as_of(trend_timeline: List[tuple], t: datetime, ptr_state: dict) -> Optional[smc.Bias]:
    """Advances a shared pointer forward through the D1 event timeline -- O(1) amortized per call
    when called with monotonically increasing `t` (which is how simulate() below uses it)."""
    i = ptr_state["i"]
    n = len(trend_timeline)
    while i < n and trend_timeline[i][0] < t:
        i += 1
    ptr_state["i"] = i
    return trend_timeline[i - 1][1] if i > 0 else None


def simulate(h1_candles: List[Candle], daily_candles: List[Candle],
             notional: float = 10_000.0, costs: CostModelV3 = DEFAULT_COSTS,
             sl_tp_scale: float = 1.0) -> List[TradeRecordV3]:
    """
    Pure in-memory bar-by-bar simulation. No ledger I/O, no randomness,
    deterministic. Performance note: the naive version of this loop
    re-scanned the full swings/FVGs lists on every bar (O(bars x swings),
    which is minutes-slow over 8+ years of H1 data) -- rewritten to
    maintain small incrementally-growing "already visible" lists via
    monotonic pointers instead, since visibility only ever moves forward
    with `i`. Verified against the naive version's output on a 6-month
    slice before trusting this for the full-history runs below.
    """
    trend_timeline = _d1_trend_series(daily_candles)
    if not trend_timeline:
        return []
    ptr_state = {"i": 0}

    h1_df = candles_to_df(h1_candles)
    swings = smc.get_swing_points(h1_df, strength=SWING_STRENGTH)
    fvgs = smc.detect_fvgs(h1_df)
    smc.update_fvg_states(h1_df, fvgs)

    highs = [c.high for c in h1_candles]
    lows = [c.low for c in h1_candles]
    closes = [c.close for c in h1_candles]
    times = [datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc) for c in h1_candles]

    # Sorted-by-visibility queues, drained into small "already visible" lists as `i` advances.
    lows_queue = sorted((s for s in swings if s.kind == "low"), key=lambda s: s.index + SWING_STRENGTH)
    highs_queue = sorted((s for s in swings if s.kind == "high"), key=lambda s: s.index + SWING_STRENGTH)
    bull_fvg_queue = sorted((g for g in fvgs if g.direction == smc.Bias.BULLISH), key=lambda g: g.index + 2)
    bear_fvg_queue = sorted((g for g in fvgs if g.direction == smc.Bias.BEARISH), key=lambda g: g.index + 2)
    visible_lows: List = []
    visible_highs: List = []
    visible_bull_fvgs: List = []
    visible_bear_fvgs: List = []
    lq_ptr = hq_ptr = bfq_ptr = brq_ptr = 0

    swept_swing_ids: set = set()
    trades: List[TradeRecordV3] = []
    position: Optional[dict] = None

    for i in range(len(h1_candles)):
        t = times[i]

        while lq_ptr < len(lows_queue) and lows_queue[lq_ptr].index + SWING_STRENGTH <= i:
            visible_lows.append(lows_queue[lq_ptr]); lq_ptr += 1
        while hq_ptr < len(highs_queue) and highs_queue[hq_ptr].index + SWING_STRENGTH <= i:
            visible_highs.append(highs_queue[hq_ptr]); hq_ptr += 1
        while bfq_ptr < len(bull_fvg_queue) and bull_fvg_queue[bfq_ptr].index + 2 <= i:
            visible_bull_fvgs.append(bull_fvg_queue[bfq_ptr]); bfq_ptr += 1
        while brq_ptr < len(bear_fvg_queue) and bear_fvg_queue[brq_ptr].index + 2 <= i:
            visible_bear_fvgs.append(bear_fvg_queue[brq_ptr]); brq_ptr += 1

        if position is not None:
            exit_price = exit_reason = None
            if position["direction"] == Direction.LONG:
                if lows[i] <= position["sl"]:
                    exit_price, exit_reason = position["sl"], "SL"
                elif highs[i] >= position["tp"]:
                    exit_price, exit_reason = position["tp"], "TP"
            else:
                if highs[i] >= position["sl"]:
                    exit_price, exit_reason = position["sl"], "SL"
                elif lows[i] <= position["tp"]:
                    exit_price, exit_reason = position["tp"], "TP"

            if exit_price is not None:
                fill = (exit_price - costs.slippage_per_side) if position["direction"] == Direction.LONG \
                    else (exit_price + costs.slippage_per_side)
                qty = position["qty"]
                gross = ((fill - position["entry_price"]) if position["direction"] == Direction.LONG
                         else (position["entry_price"] - fill)) * qty
                pnl = gross - costs.commission * qty * 2
                trades.append(TradeRecordV3(
                    direction=position["direction"], entry_time=position["entry_time"],
                    entry_price=position["entry_price"], stop=position["sl"], target=position["tp"],
                    rrr=position["rrr"], exit_time=t, exit_price=fill, exit_reason=exit_reason,
                    qty=qty, pnl=pnl,
                ))
                position = None

        if position is not None:
            continue

        # --- Filter 1: HTF (D1) trend alignment ---
        trend = _trend_as_of(trend_timeline, t, ptr_state)
        if trend is None:
            continue

        # --- Filter 2: session window ---
        if not (SESSION_START_UTC <= t.hour < SESSION_END_UTC):
            continue

        # --- Filter 3: liquidity sweep + FVG confluence, in the HTF trend direction only ---
        want_long = trend == smc.Bias.BULLISH
        direction = None
        swept_swing = None
        confluence_fvg = None

        if want_long:
            nearest = next((s for s in reversed(visible_lows) if id(s) not in swept_swing_ids and s.index < i), None)
            if nearest is not None:
                swept = lows[i] < nearest.price and closes[i] > nearest.price
                if swept:
                    active_bull_fvgs = [
                        g for g in visible_bull_fvgs if g.index < i
                        and (g.inverted_at is None or g.inverted_at.to_pydatetime() > t)
                        and lows[i] <= g.top and highs[i] >= g.bottom
                    ]
                    if active_bull_fvgs:
                        direction, swept_swing, confluence_fvg = Direction.LONG, nearest, active_bull_fvgs[0]
        else:
            nearest = next((s for s in reversed(visible_highs) if id(s) not in swept_swing_ids and s.index < i), None)
            if nearest is not None:
                swept = highs[i] > nearest.price and closes[i] < nearest.price
                if swept:
                    active_bear_fvgs = [
                        g for g in visible_bear_fvgs if g.index < i
                        and (g.inverted_at is None or g.inverted_at.to_pydatetime() > t)
                        and lows[i] <= g.top and highs[i] >= g.bottom
                    ]
                    if active_bear_fvgs:
                        direction, swept_swing, confluence_fvg = Direction.SHORT, nearest, active_bear_fvgs[0]

        if direction is None:
            continue
        swept_swing_ids.add(id(swept_swing))

        # --- Filter 4: structural stop/target, RRR >= 1.5 ---
        price = closes[i]
        if direction == Direction.LONG:
            stop = lows[i] - STOP_BUFFER
            opposing = [s for s in visible_highs if s.index < i and s.price > price]
        else:
            stop = highs[i] + STOP_BUFFER
            opposing = [s for s in visible_lows if s.index < i and s.price < price]
        if not opposing:
            continue  # no structural target visible yet -- skip, don't invent one

        target_swing = min(opposing, key=lambda s: abs(s.price - price))
        target = target_swing.price
        risk = abs(price - stop)
        reward = abs(target - price)
        if risk <= 0:
            continue
        rrr = reward / risk  # scale-invariant: scaling risk and reward by the same factor never changes this
        if rrr < MIN_RRR:
            continue

        # Sensitivity-check hook: scales both structural distances uniformly (so RRR, and therefore
        # which signals pass the filter above, is unaffected -- only how far the stop/target sit).
        if sl_tp_scale != 1.0:
            stop = price - (price - stop) * sl_tp_scale if direction == Direction.LONG else price + (stop - price) * sl_tp_scale
            target = price + (target - price) * sl_tp_scale if direction == Direction.LONG else price - (price - target) * sl_tp_scale

        spread_adj = costs.spread / 2 + costs.slippage_per_side
        fill = (price + spread_adj) if direction == Direction.LONG else (price - spread_adj)
        qty = notional / fill

        position = {"direction": direction, "entry_price": fill, "sl": stop, "tp": target,
                    "entry_time": t, "qty": qty, "rrr": rrr}

    return trades


def compute_metrics(trades: List[TradeRecordV3], notional: float, risk_free_annual: float = 0.05) -> dict:
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
