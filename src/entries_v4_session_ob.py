# -*- coding: utf-8 -*-
"""
entries_v4_session_ob.py

XAUUSD_SESSION_OB v4 -- Hypothesis 2 from the 2026-09-01 core-edge pivot
("higher-timeframe session order blocks"), isolated from every other idea
in this project: no daily SMA regime filter, no ADX filter, no persistence
bars, no confluence with entries_v2/entries_v3's logic. One mechanism,
tested alone, exactly the same "isolate the hypothesis" discipline used
for the raw-trigger diagnostic in the prior round (see
data/performance_report.md).

Concept (mechanical, ICT/SMC "order block" terminology, made fully
precise so it's actually backtestable rather than a vague pattern):

  H4 chart  -> IDENTIFY: a bullish order block is the last DOWN-closing H4
               candle before the very next H4 candle closes decisively
               above its high (a real displacement, not noise -- gated by
               displacement_atr_mult against that candle's own H4 ATR).
               Mirror image (last UP-closing candle before a decisive
               close below its low) for a bearish order block. Only
               candles whose OPEN falls inside [session_start_hour,
               session_end_hour) UTC are eligible -- order blocks are
               only considered where they formed during a specific
               session window (default: the London open, 07:00-10:00
               UTC), per the ICT premise that institutional positioning
               concentrates at session opens.
  H1 chart  -> ENTRY: once an order block is confirmed (i.e. its
               displacement candle has actually closed -- no lookahead,
               see _build_order_blocks), its price zone [low, high]
               becomes tradeable. The first H1 bar whose price re-enters
               that zone AND closes back in the displacement's original
               direction (a rejection, not a breakthrough) triggers one
               entry. Each zone can only ever produce one trade; zones
               unmitigated after ob_zone_expiry_bars H1 bars expire
               untraded.
  Exit      -> Same fixed ATR-based SL/TP convention as entries_v2 (H1
               ATR, set at entry) -- deliberately unchanged from the
               already-established, sound risk-management approach, so
               this experiment isolates the ENTRY signal as the only new
               variable, not the exit mechanics too.

Session-hour and displacement-threshold values are held as config fields
(tunable later, once real out-of-sample data justifies tuning anything) --
zero post-entry indicators or regime filters are layered on top of this by
design; this file's whole point is to find out whether the raw signal
carries edge on its own, same question the diagnostic asked of
entries_v2's trigger.

Modularity: reuses Direction/TradeRecordV2/CostModel/DEFAULT_COSTS/
compute_metrics from entries_v2.py rather than redefining them, so this
module's simulate() output plugs directly into the same
compute_metrics()/reporting path already used everywhere else in this
project. simulate()'s signature (h1_candles, h4_candles, config, notional,
costs) intentionally mirrors entries_v2.simulate()'s
(h1_candles, daily_candles, config, notional, costs) -- swap daily_candles
for h4_candles since order blocks are identified on H4 here, not daily.
run_walk_forward_folds() below is a drop-in analog of
lowfreq_v2_eval.py's generate_folds()+run_window() pairing, ready to run
against the 2012-2018 XAUUSD data once scripts/expand_dukascopy_history.py
finishes populating the cache -- see this module's __main__ block for a
smoke test against whatever's cached right now.

UNTESTED FOR REAL EDGE as of writing -- this file implements the
mechanism faithfully; it has not yet been run against the expanded
2012-2018 fold set (still fetching), and this module's docstring should
not be read as a claim that the hypothesis works.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from src.candle import Candle
from src.indicators import atr
from src.entries_v2 import Direction, TradeRecordV2, CostModel, DEFAULT_COSTS, compute_metrics

ATR_PERIOD = 14  # fixed, matching entries_v2's convention -- not tunable


@dataclass
class SessionOBConfig:
    htf_minutes: int = 240  # H4 -- the higher timeframe order blocks are identified on
    displacement_atr_mult: float = 1.5   # how decisive the confirming H4 candle's close must be
    session_start_hour: int = 7          # UTC, inclusive -- default: London open
    session_end_hour: int = 10           # UTC, exclusive
    # H1 bars; an unmitigated zone this old is dropped. 300 (~12.5 days) not a smaller
    # value -- checked directly against real 2018 data before picking this: the first
    # confirmed zone that year wasn't touched until 89 H1 bars (~3.7 days) later, and an
    # initial guess of 40 bars (~1.7 days) produced zero trades all year, not because the
    # mechanism was broken but because that window was simply too tight for how long real
    # order-block mitigation actually takes here.
    ob_zone_expiry_bars: int = 300
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5

    def as_dict(self) -> dict:
        return {
            "htf_minutes": self.htf_minutes,
            "displacement_atr_mult": self.displacement_atr_mult,
            "session_start_hour": self.session_start_hour,
            "session_end_hour": self.session_end_hour,
            "ob_zone_expiry_bars": self.ob_zone_expiry_bars,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
        }


@dataclass
class _OrderBlock:
    direction: Direction
    zone_low: float
    zone_high: float
    confirmed_time_ms: int  # the confirming H4 candle's CLOSE time -- the zone isn't tradeable before this


def _build_order_blocks(h4_candles: List[Candle], config: SessionOBConfig) -> List[_OrderBlock]:
    """One forward pass over H4 candles. An order block at index i is confirmed by
    candle i+1 alone (no further lookahead) -- confirmed_time_ms is i+1's close time,
    so nothing downstream can see this zone before that candle has actually finished."""
    h4_atr = atr(h4_candles, ATR_PERIOD)
    htf_duration_ms = config.htf_minutes * 60_000
    blocks: List[_OrderBlock] = []

    for i in range(len(h4_candles) - 1):
        if h4_atr[i] is None or h4_atr[i] <= 0:
            continue
        candle, confirm = h4_candles[i], h4_candles[i + 1]
        open_hour = datetime.fromtimestamp(candle.open_time / 1000, tz=timezone.utc).hour
        if not (config.session_start_hour <= open_hour < config.session_end_hour):
            continue

        is_down_candle = candle.close < candle.open
        is_up_candle = candle.close > candle.open
        confirm_close_time_ms = confirm.open_time + htf_duration_ms

        if is_down_candle and (confirm.close - candle.high) >= config.displacement_atr_mult * h4_atr[i]:
            blocks.append(_OrderBlock(Direction.LONG, candle.low, candle.high, confirm_close_time_ms))
        elif is_up_candle and (candle.low - confirm.close) >= config.displacement_atr_mult * h4_atr[i]:
            blocks.append(_OrderBlock(Direction.SHORT, candle.low, candle.high, confirm_close_time_ms))

    blocks.sort(key=lambda b: b.confirmed_time_ms)
    return blocks


def simulate(h1_candles: List[Candle], h4_candles: List[Candle], config: SessionOBConfig,
             notional: float = 10_000.0, costs: CostModel = DEFAULT_COSTS) -> List[TradeRecordV2]:
    """Pure in-memory bar-by-bar simulation, H1 execution, H4 order-block identification.
    No ledger I/O, no post-entry filters of any kind -- see module docstring."""
    order_blocks = _build_order_blocks(h4_candles, config)
    atr_vals = atr(h1_candles, ATR_PERIOD)

    trades: List[TradeRecordV2] = []
    position: Optional[dict] = None
    active_zones: List[dict] = []  # {"ob": _OrderBlock, "expires_at_index": int}
    ob_pointer = 0  # advances forward only -- order_blocks is sorted by confirmed_time_ms

    for i in range(1, len(h1_candles)):
        if atr_vals[i] is None or atr_vals[i] <= 0:
            continue
        c = h1_candles[i]
        t = datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc)
        a = atr_vals[i]

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
                fill = (exit_price - costs.slippage_per_side) if position["direction"] == Direction.LONG \
                    else (exit_price + costs.slippage_per_side)
                qty = position["qty"]
                gross = ((fill - position["entry_price"]) if position["direction"] == Direction.LONG
                         else (position["entry_price"] - fill)) * qty
                pnl = gross - costs.commission * qty * 2
                trades.append(TradeRecordV2(
                    direction=position["direction"], entry_time=position["entry_time"],
                    entry_price=position["entry_price"], stop=position["sl"], target=position["tp"],
                    exit_time=t, exit_price=fill, exit_reason=exit_reason, qty=qty, pnl=pnl,
                ))
                position = None

        # Admit newly-confirmed order blocks into the active set -- only ones confirmed
        # strictly before this H1 bar opens are visible (no lookahead).
        while ob_pointer < len(order_blocks) and order_blocks[ob_pointer].confirmed_time_ms <= c.open_time:
            active_zones.append({"ob": order_blocks[ob_pointer], "expires_at_index": i + config.ob_zone_expiry_bars})
            ob_pointer += 1
        active_zones = [z for z in active_zones if z["expires_at_index"] >= i]

        if position is not None:
            continue

        triggered = None
        for z in active_zones:
            ob = z["ob"]
            entered_zone = ob.zone_low <= c.low <= ob.zone_high or ob.zone_low <= c.high <= ob.zone_high or \
                (c.low <= ob.zone_low and c.high >= ob.zone_high)
            if not entered_zone:
                continue
            if ob.direction == Direction.LONG and c.close > ob.zone_high:
                triggered = z
                break
            if ob.direction == Direction.SHORT and c.close < ob.zone_low:
                triggered = z
                break

        if triggered is None:
            continue
        active_zones.remove(triggered)  # one trade per zone
        direction = triggered["ob"].direction

        price = c.close
        spread_adj = costs.spread / 2 + costs.slippage_per_side
        fill = (price + spread_adj) if direction == Direction.LONG else (price - spread_adj)
        qty = notional / fill
        sl = (fill - config.atr_sl_mult * a) if direction == Direction.LONG else (fill + config.atr_sl_mult * a)
        tp = (fill + config.atr_tp_mult * a) if direction == Direction.LONG else (fill - config.atr_tp_mult * a)

        position = {"direction": direction, "entry_price": fill, "sl": sl, "tp": tp, "entry_time": t, "qty": qty}

    return trades


def run_walk_forward_folds(h1_all: List[Candle], h4_all: List[Candle], config: SessionOBConfig,
                            folds: List[dict], notional: float = 10_000.0,
                            lookback_days: int = 60) -> dict:
    """
    Drop-in analog of lowfreq_v2_eval.py's generate_folds()+run_window() pairing --
    pass it the SAME fold list generate_folds() produces (this module doesn't need its
    own fold-boundary logic, only its own simulate()). lookback_days defaults to 60
    (vs. entries_v2's 220) since H4 order blocks warm up much faster than a 50-100
    period daily SMA -- no long moving-average history is needed here.
    """
    import statistics
    from datetime import timedelta

    def _slice(candles: List[Candle], start: datetime, end: datetime) -> List[Candle]:
        start_ms, end_ms = start.timestamp() * 1000, end.timestamp() * 1000
        return [c for c in candles if start_ms <= c.open_time <= end_ms]

    fold_sharpes = []
    all_trades = []
    for fold in folds:
        pad_start = fold["train_end"] - timedelta(days=lookback_days)
        h1_slice = _slice(h1_all, pad_start, fold["validate_end"])
        h4_slice = _slice(h4_all, pad_start, fold["validate_end"])
        trades = simulate(h1_slice, h4_slice, config, notional, DEFAULT_COSTS)
        window_trades = [t for t in trades if fold["train_end"] <= t.entry_time < fold["validate_end"]]
        m = compute_metrics(window_trades, notional)
        sharpe = m["sharpe"] if (m["sharpe"] is not None and m["n_trades"] >= 3) else 0.0
        fold_sharpes.append(sharpe)
        all_trades.extend(window_trades)

    pooled = compute_metrics(all_trades, notional)
    return {"median_sharpe": statistics.median(fold_sharpes), "fold_sharpes": fold_sharpes, "pooled": pooled}


if __name__ == "__main__":
    """Smoke test against whatever's currently cached (not a real edge test -- see
    module docstring; the real evaluation waits for the 2012-2018 expansion)."""
    from src.lowfreq_v2_eval import load_all_candles, TRAIN_START, TRAIN_END, generate_folds
    from src import data_dukascopy

    print("Smoke-testing entries_v4_session_ob against the currently-cached data "
          "(2018-present) -- NOT the real 2012-2018 out-of-sample evaluation, "
          "just confirming the mechanism runs end-to-end without error.\n")
    m5 = data_dukascopy.load_m5_candles("2018-01-01")
    h1_all = data_dukascopy.resample(m5, 60)
    h4_all = data_dukascopy.resample(m5, 240)
    print(f"Loaded {len(h1_all)} H1 bars, {len(h4_all)} H4 bars.")

    cfg = SessionOBConfig()
    folds = generate_folds(TRAIN_START, TRAIN_END)[:3]  # just the first 3 folds for a quick smoke test
    result = run_walk_forward_folds(h1_all, h4_all, cfg, folds)
    print(f"\n3-fold smoke test: median_sharpe={result['median_sharpe']:+.3f}  "
          f"fold_sharpes={[round(s, 2) for s in result['fold_sharpes']]}")
    p = result["pooled"]
    pf_str = f"{p['profit_factor']:.3f}" if p["profit_factor"] is not None else "undefined"
    print(f"Pooled: trades={p['n_trades']} win%={p['win_rate_pct']:.1f} PF={pf_str} "
          f"net%={p['net_pnl_pct']:+.2f}")
