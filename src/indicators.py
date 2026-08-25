# -*- coding: utf-8 -*-
"""
indicators.py

Minimal, dependency-free technical indicators used by the medfreq
strategy. Kept separate from the root-level structures.py (the
unconnected SMC engine) so src/ stays self-contained.
"""

from __future__ import annotations

from typing import List, Optional

from src.candle import Candle


def ema(values: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(values: List[float], period: int = 14) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period + 1:
        return out
    gains = [max(values[i] - values[i - 1], 0.0) for i in range(1, len(values))]
    losses = [max(values[i - 1] - values[i], 0.0) for i in range(1, len(values))]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs_val = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
        out[i + 1] = rs_val
    return out


def adx(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
    """
    Standard Wilder Average Directional Index -- an independent trend-
    strength measure, distinct from this project's ATR-expansion regime
    gate (regime_filter.atr_expansion_gate). That gate tests "is
    volatility expanding," which fires during whipsaw/ranging spikes too,
    not only genuine directional trends -- using it to classify "trending
    vs ranging" would be circular, since the strategy only ever trades
    when it's already True. ADX answers a different question (is price
    moving persistently in one direction), which is what a regime
    breakdown actually needs. Conventional reading: ADX > 25 trending,
    < 20 ranging, 20-25 transitional -- not a threshold defined here,
    left to the caller (a classification convention, not a strategy
    parameter to tune).

    +DM/-DM/TR are smoothed with the same Wilder recursive average as
    atr() above (seed = simple average of the first `period` values,
    then avg = (avg*(period-1) + new) / period) -- deliberately the same
    convention already used elsewhere in this project, not a different
    smoothing scheme invented for this function.
    """
    n = len(candles)
    out: List[Optional[float]] = [None] * n
    if n < 2 * period + 1:
        return out

    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, n):
        up_move = candles[i].high - candles[i - 1].high
        down_move = candles[i - 1].low - candles[i].low
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
        h, l, prev_c = candles[i].high, candles[i].low, candles[i - 1].close
        trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))

    def wilder_smooth(values: List[float], period: int) -> List[Optional[float]]:
        smoothed: List[Optional[float]] = [None] * len(values)
        avg = sum(values[:period]) / period
        smoothed[period - 1] = avg
        for i in range(period, len(values)):
            avg = (avg * (period - 1) + values[i]) / period
            smoothed[i] = avg
        return smoothed

    tr_s = wilder_smooth(trs, period)
    plus_dm_s = wilder_smooth(plus_dm, period)
    minus_dm_s = wilder_smooth(minus_dm, period)

    dx: List[Optional[float]] = [None] * len(trs)
    for i in range(len(trs)):
        if tr_s[i] is None or tr_s[i] == 0:
            continue
        plus_di = 100 * plus_dm_s[i] / tr_s[i]
        minus_di = 100 * minus_dm_s[i] / tr_s[i]
        denom = plus_di + minus_di
        if denom > 0:
            dx[i] = 100 * abs(plus_di - minus_di) / denom

    dx_valid = [v for v in dx if v is not None]
    if len(dx_valid) < period:
        return out
    first_dx_idx = next(i for i, v in enumerate(dx) if v is not None)
    adx_seed = sum(dx_valid[:period]) / period
    adx_val = adx_seed
    out[first_dx_idx + period] = adx_val  # +1 for the candles[1:] offset, -1+period for the seed window
    for i in range(first_dx_idx + period, len(dx)):
        if dx[i] is None:
            continue
        adx_val = (adx_val * (period - 1) + dx[i]) / period
        out[i + 1] = adx_val  # +1: dx/trs are indexed from candles[1:], out is indexed from candles[0:]
    return out


def atr(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
    n = len(candles)
    out: List[Optional[float]] = [None] * n
    if n < period + 1:
        return out
    trs = []
    for i in range(1, n):
        h, l, prev_c = candles[i].high, candles[i].low, candles[i - 1].close
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    avg = sum(trs[:period]) / period
    out[period] = avg
    for i in range(period, len(trs)):
        avg = (avg * (period - 1) + trs[i]) / period
        out[i + 1] = avg
    return out
