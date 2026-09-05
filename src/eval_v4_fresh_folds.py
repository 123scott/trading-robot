# -*- coding: utf-8 -*-
"""
eval_v4_fresh_folds.py

Full walk-forward evaluation of entries_v4_session_ob (Hypothesis 2) on the
newly-expanded 2012-2017 XAUUSD data -- a fold set no strategy in this
project has ever been fit to, selected on, or evaluated against. Also runs
entries_v2's RAW core trigger (regime filter off, ADX block off -- the
exact configuration behind the -0.466 baseline figure) on the SAME folds,
because the baseline number was measured on 2018-2025 folds and comparing
two strategies across two different periods is apples-to-oranges. The
like-for-like comparison is both strategies on these same fresh folds.

Fold construction is identical to lowfreq_v2_eval.generate_folds (12-month
context / 3-month validate / slide 3 months), just bounded to 2012-01-01 ..
2017-12-31 so nothing after the original TRAIN_START (2018-01-01) leaks in.

Same scoring conventions as everywhere else in this project: per-fold
Sharpe is 0.0 (neutral) for folds with <3 trades, combo score is the
MEDIAN of fold Sharpes, pooled metrics come from concatenating every
fold's validate-window trades.

Usage:
    python3 -m src.eval_v4_fresh_folds
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone

from src import data_dukascopy
from src.entries_v2 import LowfreqV2Config, compute_metrics
from src.entries_v4_session_ob import SessionOBConfig, run_walk_forward_folds
from src.lowfreq_v2_eval import generate_folds, run_window

FRESH_START = datetime(2012, 1, 1, tzinfo=timezone.utc)
FRESH_END = datetime(2017, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
NOTIONAL = 10_000.0

# The exact raw-trigger config behind the -0.466 baseline (see performance_report.md,
# "Core Edge Diagnostic", 2026-09-01): locked structural params, every filter off.
RAW_BASELINE = LowfreqV2Config(trend_sma_period=50, pullback_ema_period=21, pullback_tolerance_pct=0.20,
                                atr_sl_mult=2.0, atr_tp_mult=2.5, use_regime_filter=False,
                                regime_confirm_bars=1, block_adx_transition=False)


def _fmt(label: str, median_sharpe: float, fold_sharpes: list, pooled: dict) -> None:
    pf = pooled["profit_factor"]
    pf_str = f"{pf:.3f}" if pf is not None else "undefined"
    scored = sum(1 for s in fold_sharpes if s != 0.0)
    print(f"\n{label}")
    print(f"  Median fold Sharpe:   {median_sharpe:+.3f}   ({scored}/{len(fold_sharpes)} folds had >=3 trades)")
    print(f"  Per-fold Sharpes:     {[round(s, 2) for s in fold_sharpes]}")
    print(f"  Pooled trades:        {pooled['n_trades']}")
    print(f"  Pooled win rate:      {pooled['win_rate_pct']:.1f}%")
    print(f"  Pooled profit factor: {pf_str}")
    print(f"  Pooled net P&L:       {pooled['net_pnl_pct']:+.2f}%")
    print(f"  Pooled max drawdown:  {pooled['max_drawdown_pct']:.2f}%")


def main() -> None:
    print("Loading XAUUSD M5 cache from 2012-01-01 and resampling to H1 / H4 / daily...")
    m5 = data_dukascopy.load_m5_candles("2012-01-01", "2018-01-01")
    h1 = data_dukascopy.resample(m5, 60)
    h4 = data_dukascopy.resample(m5, 240)
    daily = data_dukascopy.resample(m5, 1440)
    print(f"Loaded {len(m5)} M5 -> {len(h1)} H1, {len(h4)} H4, {len(daily)} daily bars.")

    folds = generate_folds(FRESH_START, FRESH_END)
    print(f"\n{len(folds)} walk-forward folds inside {FRESH_START.date()} .. {FRESH_END.date()} "
          f"(12mo context / 3mo validate / slide 3mo). First validate window: "
          f"{folds[0]['train_end'].date()} -> {folds[0]['validate_end'].date()}; last: "
          f"{folds[-1]['train_end'].date()} -> {folds[-1]['validate_end'].date()}.")

    # --- Hypothesis 2: session order blocks, no filters ---
    v4 = run_walk_forward_folds(h1, h4, SessionOBConfig(), folds, notional=NOTIONAL)
    _fmt("HYPOTHESIS 2 -- entries_v4_session_ob (session order blocks, zero filters)",
         v4["median_sharpe"], v4["fold_sharpes"], v4["pooled"])

    # --- Baseline: entries_v2 raw trigger on the SAME folds ---
    fold_sharpes = []
    all_trades = []
    for fold in folds:
        result = run_window(h1, daily, RAW_BASELINE, fold["train_end"], fold["validate_end"], notional=NOTIONAL)
        m = result["metrics"]
        fold_sharpes.append(m["sharpe"] if (m["sharpe"] is not None and m["n_trades"] >= 3) else 0.0)
        all_trades.extend(result["trades"])
    _fmt("BASELINE -- entries_v2 raw trigger (regime filter off, ADX block off), same folds",
         statistics.median(fold_sharpes), fold_sharpes, compute_metrics(all_trades, NOTIONAL))

    print("\nReference: the same raw baseline scored median fold Sharpe -0.466 on the 2018-2025-07 folds.")


if __name__ == "__main__":
    main()
