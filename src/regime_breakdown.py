# -*- coding: utf-8 -*-
"""
regime_breakdown.py

Per-regime validation of the N-bar persistence filter (entries_v2 with
use_regime_filter=True, regime_confirm_bars=3) -- TRAINING DATA ONLY
(2018-01-01 to 2025-07-31). The reserved 2025-08 to 2026-07 holdout is
never touched here; this answers "does the persistence gate's benefit
concentrate in genuinely trending conditions" using only data the model
selection already had access to.

Classifier: D1 ADX(14), the standard Wilder trend-strength measure --
independent of the strategy's own ATR-expansion entry gate (which tests
volatility expansion, not directional persistence, and would be
circular to reuse here since the strategy never trades when it's
False). Conventional thresholds, not fit to this data: ADX > 25
trending, < 20 ranging, 20-25 transitional. Verified correct against
hand-traceable synthetic cases (pure uptrend -> 100, pure chop -> ~3.6)
before trusting it on real data -- see the development session, or
re-run: a pure-trend/pure-chop synthetic check is cheap and worth
keeping in mind if this module is ever modified.

Each trade is classified by the D1 ADX value as of its own entry time,
aligned with the same no-lookahead pointer-advance pattern already used
for the daily trend filter (medfreq_strategy.align_htf_to_m5) -- a
trade only ever sees the most recently CLOSED daily bar's ADX, never a
still-forming one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from src.lowfreq_v2_eval import load_all_candles, TRAIN_START, TRAIN_END, NOTIONAL
from src.entries_v2 import LowfreqV2Config, TradeRecordV2, simulate, compute_metrics
from src.indicators import adx
from src.medfreq_strategy import align_htf_to_m5

ADX_PERIOD = 14
TRENDING_THRESHOLD = 25.0
RANGING_THRESHOLD = 20.0

FLAGSHIP_PARAMS = dict(trend_sma_period=50, pullback_ema_period=21, pullback_tolerance_pct=0.2,
                        atr_sl_mult=2.0, atr_tp_mult=2.5, use_regime_filter=True, regime_confirm_bars=3)


def classify_trades(trades: List[TradeRecordV2], daily_candles) -> dict:
    """Returns {'trending': [...], 'ranging': [...], 'transitional': [...]} by each trade's entry-time D1 ADX."""
    daily_adx = adx(daily_candles, ADX_PERIOD)
    # Reuse the exact no-lookahead alignment already used for the daily trend filter --
    # trades is a list of TradeRecordV2, not candles, so build a tiny synthetic "candle-like"
    # timeline of entry times to align against, matching align_htf_to_m5's expected shape.
    class _EntryPoint:
        def __init__(self, t):
            self.open_time = int(t.timestamp() * 1000)

    entry_points = [_EntryPoint(t.entry_time) for t in trades]
    aligned_adx = align_htf_to_m5(entry_points, daily_candles, daily_adx, 1440)

    buckets = {"trending": [], "ranging": [], "transitional": []}
    unclassified = 0
    for trade, a in zip(trades, aligned_adx):
        if a is None:
            unclassified += 1
            continue
        if a > TRENDING_THRESHOLD:
            buckets["trending"].append(trade)
        elif a < RANGING_THRESHOLD:
            buckets["ranging"].append(trade)
        else:
            buckets["transitional"].append(trade)
    if unclassified:
        print(f"[regime_breakdown] {unclassified} trades unclassified (ADX warmup not yet complete at entry time)")
    return buckets


def main() -> None:
    h1_all, daily_all = load_all_candles("2018-01-01")
    cfg = LowfreqV2Config(**FLAGSHIP_PARAMS)

    all_trades = simulate(h1_all, daily_all, cfg, NOTIONAL)
    train_trades = [t for t in all_trades if TRAIN_START <= t.entry_time < TRAIN_END]
    print(f"Training-period trades (flagship config, locked): {len(train_trades)}")
    print(f"Classifying by D1 ADX({ADX_PERIOD}) at each trade's entry time "
          f"(trending > {TRENDING_THRESHOLD}, ranging < {RANGING_THRESHOLD})\n")

    buckets = classify_trades(train_trades, daily_all)

    header = f"{'Regime':14} {'Trades':>7} {'Win%':>7} {'PF':>8} {'Expectancy':>11} {'Sharpe':>8} {'NetPnL%':>9}"
    print(header)
    print("-" * len(header))
    for label in ("trending", "ranging", "transitional"):
        seg = buckets[label]
        m = compute_metrics(seg, NOTIONAL)
        pf = f"{m['profit_factor']:.3f}" if m["profit_factor"] is not None else "undef"
        sh = f"{m['sharpe']:.3f}" if m["sharpe"] is not None else "--"
        print(f"{label:14} {m['n_trades']:7d} {m['win_rate_pct']:7.1f} {pf:>8} "
              f"{m['expectancy']:11.3f} {sh:>8} {m['net_pnl_pct']:9.2f}")

    print(f"\n(all figures training data only, 2018-01-01 to 2025-07-31 -- the holdout window was not touched)")


if __name__ == "__main__":
    main()
