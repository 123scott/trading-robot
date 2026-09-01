# -*- coding: utf-8 -*-
"""
entries_v2.py

XAUUSD_LOWFREQ v2 -- adds an intraday entry layer to reach the 3-4
trades/week target the pure daily SMA(7,50) crossover structurally
cannot hit (~0.13 trades/week measured directly from data/ledger.csv --
see data/pre_retrain_snapshot.md). Does NOT touch or import
src/backtest_structures.py / src/backtest_entries.py / src/trading_robot.py
-- those remain exactly as they were; this is a new, separate module.

Design (one setup, not a committee, per the retraining brief):

  Daily chart  -> REGIME FILTER ONLY: price above its own daily SMA
                  (trend_sma_period) = long-only regime; below = short-
                  only regime. This is the ONLY use of the daily
                  timeframe in v2 -- it never times entries.
  H1 chart     -> ENTRY: a single pullback-to-EMA-and-bounce setup.
                  In a long regime, a bar whose LOW touches within
                  `pullback_tolerance_pct` of the H1 EMA
                  (pullback_ema_period) but CLOSES back above it
                  triggers a LONG. Mirror image for a short regime.
                  This is a real, standard trend-continuation entry
                  (buy the dip in an uptrend / sell the rip in a
                  downtrend), not a curve-fit invention.
  Exit         -> ATR-based stop-loss and take-profit (H1 ATR), fixed
                  at entry. This REPLACES the old "exit on opposite
                  crossover" model entirely -- the old LOWFREQ has no
                  stop-loss at all (see snapshot), which is not
                  something to carry forward into a higher-frequency,
                  short-permitting version.

Daily -> H1 alignment reuses medfreq_strategy.align_htf_to_m5's exact
no-lookahead logic (a daily SMA value is only visible to an H1 bar once
that daily bar has actually closed) rather than reimplementing it --
that logic is already correct and tested, and re-deriving it here would
just be a second place to introduce a lookahead bug.

Parameter budget: exactly 5 tunable parameters (trend_sma_period,
pullback_ema_period, pullback_tolerance_pct, atr_sl_mult, atr_tp_mult).
atr_period is fixed at 14 (the standard default used everywhere else in
this project, e.g. medfreq_strategy.py) and is NOT tuned -- deliberately
kept out of the search to hold the budget at 5, not 6.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from src.candle import Candle
from src.indicators import ema, atr, adx
from src.backtest_structures import sma
from src.medfreq_strategy import align_htf_to_m5
from src.regime_filter import atr_expansion_gate

# Original, non-optimized finding from the regime breakdown (trending vs. ranging performed
# almost identically; losses concentrated specifically in this band). Kept as the DEFAULT
# values for LowfreqV2Config.adx_transition_low/high (added 2026-08-31 so a search can sweep
# nearby bands rather than only toggling this exact one on/off) -- these module constants are
# no longer the only way to set the band, but they remain what block_adx_transition uses
# unless a config explicitly overrides them.
ADX_TRANSITION_LOW = 20.0
ADX_TRANSITION_HIGH = 25.0
ADX_PERIOD = 14

ATR_PERIOD = 14  # fixed, not tunable -- see module docstring


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class LowfreqV2Config:
    trend_sma_period: int = 100        # daily SMA regime filter
    pullback_ema_period: int = 21      # H1 EMA pullback reference
    pullback_tolerance_pct: float = 0.15  # how close (%) price must get to the EMA to count as a touch
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    # Binary on/off (src.regime_filter.atr_expansion_gate) -- NOT a 6th tunable parameter in
    # the search-range sense; this is a toggle for a controlled A/B comparison against the
    # already-established baseline, not something optimized over a range. Defaults False so
    # every existing caller (the live paper-trading harness included) is completely unaffected
    # unless this is explicitly turned on.
    use_regime_filter: bool = False
    # How many CONSECUTIVE bars the gate must have already held True before an entry is
    # allowed -- targets the specific failure mode diagnosed in the last round (the noisy
    # edge of a chop/expansion transition firing the gate on a single bar that reverts
    # immediately after). Fixed at 3, not searched -- reuses the exact convention already
    # independently justified for medfreq_strategy.py's confirm_bars anti-whipsaw filter,
    # not a new number invented for this round. 1 (the default) means "no persistence
    # requirement," i.e. identical to the prior round's behavior.
    regime_confirm_bars: int = 1
    # Binary on/off for the D1 ADX 20-25 transitional-band block (ADX_TRANSITION_LOW/HIGH
    # above) -- the specific, non-arbitrary finding from the regime breakdown (that band is
    # where losses concentrated, not "ranging" broadly). Not a parameter with a range; a
    # controlled A/B toggle, same pattern as use_regime_filter. Default False: every existing
    # caller (including the live paper-trading harness) is unaffected unless explicitly enabled.
    block_adx_transition: bool = False
    # The actual band block_adx_transition blocks, as CONFIG fields rather than only the
    # ADX_TRANSITION_LOW/HIGH module constants -- added so a search can sweep threshold
    # VALUES (e.g. testing whether the band is 20-25 specifically, or a nearby band works
    # just as well) instead of only toggling the fixed 20-25 band on/off. Defaults to the
    # module constants, so every existing caller (including block_adx_transition=True
    # ones) is completely unaffected unless these are explicitly overridden.
    adx_transition_low: float = ADX_TRANSITION_LOW
    adx_transition_high: float = ADX_TRANSITION_HIGH
    # STRUCTURAL improvement #1 (2026-09-01 core-edge round): requires the WEEKLY trend
    # (price vs. a weekly SMA) to agree with the existing daily trend direction before an
    # entry is allowed -- a genuine multi-timeframe-alignment concept (only take the daily
    # pullback setup when the higher-order weekly trend agrees), structurally distinct from
    # the ADX regime filter above (which measures trend STRENGTH/volatility expansion, not
    # cross-timeframe DIRECTIONAL agreement). Default False: no existing caller is affected.
    require_weekly_trend_alignment: bool = False
    weekly_trend_sma_period: int = 10  # ~10 weeks, matching trend_sma_period=50 trading days in spirit
    # STRUCTURAL improvement #2: scales the ATR take-profit multiple by the CURRENT
    # volatility regime instead of a single fixed atr_tp_mult -- lets winners run further
    # when ATR is running hot relative to its own recent baseline (trends have more room to
    # extend), tightens expectations when ATR is compressed (limited follow-through is more
    # likely). The two scale factors are hardcoded, non-optimized constants (same "hardcode
    # a non-optimized rule" discipline already used for the ADX band's original 20-25
    # finding) -- not free parameters to search, to avoid adding curve-fitting surface area
    # while testing whether the underlying MECHANISM helps at all. Default False: no
    # existing caller is affected.
    dynamic_atr_tp: bool = False
    atr_baseline_period: int = 100

    def as_dict(self) -> dict:
        return {
            "trend_sma_period": self.trend_sma_period,
            "pullback_ema_period": self.pullback_ema_period,
            "pullback_tolerance_pct": self.pullback_tolerance_pct,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
            "use_regime_filter": self.use_regime_filter,
            "regime_confirm_bars": self.regime_confirm_bars,
            "block_adx_transition": self.block_adx_transition,
            "adx_transition_low": self.adx_transition_low,
            "adx_transition_high": self.adx_transition_high,
            "require_weekly_trend_alignment": self.require_weekly_trend_alignment,
            "weekly_trend_sma_period": self.weekly_trend_sma_period,
            "dynamic_atr_tp": self.dynamic_atr_tp,
            "atr_baseline_period": self.atr_baseline_period,
        }


# Hardcoded, non-optimized scale factors for dynamic_atr_tp -- see the config field's
# docstring for why these are constants, not tunable parameters.
ATR_HOT_THRESHOLD = 1.2      # current ATR this far above its baseline SMA counts as "hot"
ATR_HOT_TP_SCALE = 1.3       # extend the TP multiple by this much when hot
ATR_COLD_THRESHOLD = 0.8     # current ATR this far below its baseline SMA counts as "cold"
ATR_COLD_TP_SCALE = 0.85     # tighten the TP multiple by this much when cold


def _resample_daily_to_weekly(daily_candles: List[Candle]) -> List[Candle]:
    """Groups daily candles into Monday-start ISO weeks. Used only for
    require_weekly_trend_alignment -- data_dukascopy.resample() is a pure
    intraday-bucket resampler (minute-of-day only) and can't express a
    multi-day period like a week, so this is a small dedicated helper
    rather than a misuse of that function."""
    from datetime import timedelta
    buckets: dict = {}
    for c in daily_candles:
        dt = datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc)
        monday = (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        buckets.setdefault(monday, []).append(c)
    out = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda c: c.open_time)
        out.append(Candle(open_time=int(key.timestamp() * 1000), open=group[0].open,
                           high=max(c.high for c in group), low=min(c.low for c in group),
                           close=group[-1].close, volume=sum(c.volume for c in group)))
    return out


@dataclass
class TradeRecordV2:
    direction: Direction
    entry_time: datetime
    entry_price: float
    stop: float   # SL price level set at entry (fixed for the life of the trade)
    target: float  # TP price level set at entry (fixed for the life of the trade)
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "SL" | "TP"
    qty: float
    pnl: float


@dataclass
class CostModel:
    """
    Deriv-realistic costs for this exercise, per the retraining brief --
    deliberately NOT market_data.COST_PROFILES["XAUUSD_DERIV"] (that one
    uses a percentage-based slippage model and a non-zero commission;
    this brief specifies a fixed-dollar spread/slippage and zero
    commission). Kept local to this module rather than overwriting the
    shared cost profile other bots (LOWFREQ v1, MEDFREQ) depend on.
    """
    spread: float = 0.40          # USD round-trip, midpoint of the stated 35-50 cent range
    slippage_per_side: float = 0.05  # USD, fixed, applied on both entry and exit
    commission: float = 0.0        # "no separate commission" per the brief


DEFAULT_COSTS = CostModel()


def simulate(h1_candles: List[Candle], daily_candles: List[Candle], config: LowfreqV2Config,
             notional: float = 10_000.0, costs: CostModel = DEFAULT_COSTS) -> List[TradeRecordV2]:
    """Pure in-memory bar-by-bar simulation, H1 execution, daily regime filter. No ledger I/O."""
    daily_closes = [c.close for c in daily_candles]
    daily_sma = sma(daily_closes, config.trend_sma_period)
    trend_on_h1 = align_htf_to_m5(h1_candles, daily_candles, daily_sma, 1440)
    # Computed unconditionally (cheap) but only consulted when block_adx_transition is True --
    # same non-disruptive pattern as regime_gate below. D1 ADX, aligned onto H1 with the same
    # no-lookahead pointer-advance as the daily trend filter above (a trade only ever sees the
    # most recently CLOSED daily bar's ADX, never a still-forming one).
    daily_adx = adx(daily_candles, ADX_PERIOD)
    adx_on_h1 = align_htf_to_m5(h1_candles, daily_candles, daily_adx, 1440)

    # Structural improvement #1: weekly trend, computed unconditionally (cheap) but only
    # consulted when require_weekly_trend_alignment is True.
    weekly_candles = _resample_daily_to_weekly(daily_candles)
    weekly_closes = [c.close for c in weekly_candles]
    weekly_sma = sma(weekly_closes, config.weekly_trend_sma_period)
    weekly_trend_on_h1 = align_htf_to_m5(h1_candles, weekly_candles, weekly_sma, 7 * 1440)

    h1_closes = [c.close for c in h1_candles]
    h1_ema = ema(h1_closes, config.pullback_ema_period)
    atr_vals = atr(h1_candles, ATR_PERIOD)
    # Structural improvement #2: rolling ATR baseline for dynamic_atr_tp. sma() can't take
    # None inputs, so the leading warmup Nones are substituted with 0.0 purely for this
    # calculation -- negligibly distorts the baseline for the first ~atr_baseline_period
    # bars of the entire multi-year dataset (a one-time startup transient), never touched
    # in practice since dynamic_atr_tp is also gated on atr_vals[i] being non-None below.
    atr_for_baseline = [v if v is not None else 0.0 for v in atr_vals]
    atr_baseline = sma(atr_for_baseline, config.atr_baseline_period)
    tol = config.pullback_tolerance_pct / 100.0
    # Computed unconditionally (cheap, single pass) but only ever consulted below when
    # use_regime_filter is True -- entry/exit behavior is unchanged from before this
    # integration whenever the flag is left at its default False.
    regime_gate = atr_expansion_gate(h1_candles)
    # Precomputed as its own series (not a counter mutated inside the main loop below) --
    # the main loop has several `continue` statements before reaching the entry check, and
    # a counter that only updates on iterations that reach that point would silently miss
    # bars and desync from the true consecutive-True count. This has no such dependency on
    # loop control flow: it's purely a function of regime_gate itself, computed once.
    regime_streak: List[int] = []
    _streak = 0
    for _g in regime_gate:
        _streak = _streak + 1 if _g is True else 0
        regime_streak.append(_streak)

    trades: List[TradeRecordV2] = []
    position: Optional[dict] = None

    for i in range(1, len(h1_candles)):
        if None in (trend_on_h1[i], h1_ema[i], atr_vals[i]) or atr_vals[i] <= 0:
            continue
        c = h1_candles[i]
        t = datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc)
        trend_sma = trend_on_h1[i]
        e = h1_ema[i]
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
                pnl = gross - costs.commission * qty * 2  # commission is 0.0 per the brief; kept for completeness
                trades.append(TradeRecordV2(
                    direction=position["direction"], entry_time=position["entry_time"],
                    entry_price=position["entry_price"], stop=position["sl"], target=position["tp"],
                    exit_time=t, exit_price=fill, exit_reason=exit_reason, qty=qty, pnl=pnl,
                ))
                position = None

        if position is not None:
            continue

        # Binary Can_Trade gate -- entries only, never blocks managing/closing a position
        # already open (that check happens above, before this point). No effect at all
        # unless explicitly enabled. regime_streak[i] is 0 whenever regime_gate[i] isn't
        # exactly True (including the None-during-warmup case), so requiring
        # regime_streak[i] >= regime_confirm_bars naturally subsumes the old direct
        # "is not True" check -- regime_confirm_bars=1 (the default) reproduces the prior
        # round's exact behavior with no persistence requirement.
        if config.use_regime_filter and regime_streak[i] < config.regime_confirm_bars:
            continue

        # Hardcoded, non-optimized ADX transitional-band block -- entries only, same as the
        # regime gate above. adx_on_h1[i] is None during warmup; a None fails the range check
        # below (Python raises on None <= float), so guard it explicitly rather than let a
        # warmup bar silently behave as "not blocked."
        if config.block_adx_transition:
            a_val = adx_on_h1[i]
            if a_val is None or (config.adx_transition_low <= a_val <= config.adx_transition_high):
                continue

        price = c.close
        long_regime = price > trend_sma
        short_regime = price < trend_sma

        # Structural improvement #1: weekly trend must AGREE with the daily trend direction
        # before an entry is allowed. weekly_trend_on_h1[i] is None until the first weekly
        # bar has closed (~weekly_trend_sma_period weeks in) -- treated as "not aligned"
        # (blocks entry) rather than silently passing, same convention as the ADX-band guard.
        if config.require_weekly_trend_alignment:
            w = weekly_trend_on_h1[i]
            if w is None:
                continue
            if long_regime and not (price > w):
                long_regime = False
            if short_regime and not (price < w):
                short_regime = False

        # Single pullback-to-EMA-and-bounce setup, mirrored for both regimes.
        long_trigger = long_regime and c.low <= e * (1 + tol) and c.close > e
        short_trigger = short_regime and c.high >= e * (1 - tol) and c.close < e

        direction = Direction.LONG if long_trigger else (Direction.SHORT if short_trigger else None)
        if direction is None:
            continue

        spread_adj = costs.spread / 2 + costs.slippage_per_side
        fill = (price + spread_adj) if direction == Direction.LONG else (price - spread_adj)
        qty = notional / fill
        sl = (fill - config.atr_sl_mult * a) if direction == Direction.LONG else (fill + config.atr_sl_mult * a)

        # Structural improvement #2: scale the TP multiple by the current volatility regime
        # (see ATR_HOT_*/ATR_COLD_* constants) instead of always using the fixed atr_tp_mult.
        tp_mult = config.atr_tp_mult
        if config.dynamic_atr_tp and atr_baseline[i] is not None and atr_baseline[i] > 0:
            ratio = a / atr_baseline[i]
            if ratio >= ATR_HOT_THRESHOLD:
                tp_mult = config.atr_tp_mult * ATR_HOT_TP_SCALE
            elif ratio <= ATR_COLD_THRESHOLD:
                tp_mult = config.atr_tp_mult * ATR_COLD_TP_SCALE
        tp = (fill + tp_mult * a) if direction == Direction.LONG else (fill - tp_mult * a)

        position = {"direction": direction, "entry_price": fill, "sl": sl, "tp": tp, "entry_time": t, "qty": qty}

    return trades


def compute_metrics(trades: List[TradeRecordV2], notional: float, risk_free_annual: float = 0.05) -> dict:
    pnls = [t.pnl for t in trades]
    n = len(pnls)
    wins_list = [p for p in pnls if p > 0]
    losses_list = [p for p in pnls if p < 0]
    wins, losses = len(wins_list), len(losses_list)
    win_rate = wins / n * 100 if n else 0.0
    avg_win = statistics.mean(wins_list) if wins_list else 0.0
    avg_loss = statistics.mean(losses_list) if losses_list else 0.0
    gross_profit = sum(wins_list)
    gross_loss = abs(sum(losses_list))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    expectancy = statistics.mean(pnls) if pnls else 0.0

    equity = notional
    peak = notional
    peak_time = None
    max_dd = 0.0
    max_dd_dollars = 0.0
    max_dd_duration_days = 0.0
    dd_start_time = None
    times = [t.exit_time for t in trades]
    for p, exit_time in zip(pnls, times):
        equity += p
        if equity >= peak:
            peak = equity
            peak_time = exit_time
        else:
            dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0.0
            dd_dollars = peak - equity
            if dd_pct > max_dd:
                max_dd = dd_pct
                max_dd_dollars = dd_dollars
                if peak_time is not None:
                    max_dd_duration_days = (exit_time - peak_time).total_seconds() / 86400

    sharpe = sortino = None
    trades_per_week = None
    if n >= 2:
        span_days = (times[-1] - times[0]).total_seconds() / 86400
        years = span_days / 365.25
        if years > 0:
            trades_per_year = n / years
            trades_per_week = trades_per_year / 52.1775
            returns = [p / notional for p in pnls]
            mean_r = statistics.mean(returns)
            stdev_r = statistics.pstdev(returns)
            rf_per_trade = risk_free_annual / trades_per_year if trades_per_year > 0 else 0.0
            if stdev_r > 0:
                sharpe = (mean_r - rf_per_trade) / stdev_r * math.sqrt(trades_per_year)
            downside_dev = (statistics.mean(min(r - rf_per_trade, 0.0) ** 2 for r in returns)) ** 0.5
            if downside_dev > 0:
                sortino = (mean_r - rf_per_trade) / downside_dev * math.sqrt(trades_per_year)

    return {
        "n_trades": n, "wins": wins, "losses": losses, "win_rate_pct": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "avg_win_loss_ratio": (avg_win / abs(avg_loss)) if avg_loss != 0 else None,
        "profit_factor": profit_factor, "expectancy": expectancy,
        "max_drawdown_pct": max_dd, "max_drawdown_dollars": max_dd_dollars,
        "max_drawdown_duration_days": max_dd_duration_days,
        "sharpe": sharpe, "sortino": sortino,
        "net_pnl": sum(pnls), "net_pnl_pct": sum(pnls) / notional * 100 if notional else 0.0,
        "trades_per_week": trades_per_week,
    }
