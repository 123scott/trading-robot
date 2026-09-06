# -*- coding: utf-8 -*-
"""
eval_v4_multi_asset.py

Cross-asset test of Hypothesis 2 (entries_v4_session_ob) on XAUUSD, EURUSD and
GBPUSD over the same 2012-2017 fold set, with IDENTICAL strategy parameters on
every instrument. Two purposes, and the second matters more than the first:

  1. Sample size. The single-asset XAUUSD result (32 trades, PF 1.860,
     p=0.117) is too sparse to conclude anything. Three instruments give
     roughly three times the trades without touching a single parameter.
  2. Falsification. Session order blocks are premised on institutional
     positioning around session opens -- a market-structure claim that is
     not specific to gold. If the effect is real it should appear in FX
     majors too. If it appears ONLY on XAUUSD 2012-2017, the most likely
     explanation is that the original result was noise. A negative result
     here is therefore genuinely informative, not a failed experiment.

COSTS -- the detail that would silently invalidate this entire comparison if
got wrong. entries_v4_session_ob defaults to entries_v2.DEFAULT_COSTS, which
is a GOLD model ($0.40 round-trip spread, $0.05/side slippage). Applied
unchanged to EURUSD at ~1.30, a $0.40 spread is ~30% of the instrument's
price -- every trade would be an instant catastrophic loss and the FX arms
would "fail" for reasons that have nothing to do with the hypothesis. So
each instrument gets its own CostModel, built from market_data.COST_PROFILES
(the project's existing illustrative retail assumptions) rather than invented
here. Read the resulting round-trip costs as a fraction of price before
interpreting anything -- they are printed at the top of every run:

    XAUUSD  ~0.036% of price per round trip (entries_v2.DEFAULT_COSTS)
    EURUSD  ~0.028%
    GBPUSD  ~0.030%

FX majors are genuinely cheaper to trade than gold, so the FX arms carry a
small structural cost advantage. That is realistic, not a thumb on the scale,
but it means a modest FX edge over XAUUSD should not be over-read.

commission_pct is deliberately NOT carried across: entries_v2's CostModel
charges commission per UNIT (`commission * qty * 2`), whereas COST_PROFILES
expresses it as a fraction of notional. Wiring the fractional figure into the
per-unit field would silently overcharge FX by ~5 orders of magnitude
(qty ~7,700 units for EURUSD vs ~7 oz for gold). Commission stays 0.0 for
every instrument here, exactly as it already is for the XAUUSD results this
compares against -- so the comparison stays apples-to-apples, and all three
arms are equally (slightly) optimistic on that one component.

Usage:
    python3 -m src.eval_v4_multi_asset
    python3 -m src.eval_v4_multi_asset --symbols XAUUSD EURUSD
"""

from __future__ import annotations

import argparse
import statistics
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
from scipy import stats

from src import data_dukascopy, market_data
from src.candle import Candle
from src.entries_v2 import CostModel, DEFAULT_COSTS, compute_metrics
from src.entries_v4_session_ob import SessionOBConfig, simulate
from src.lowfreq_v2_eval import generate_folds

FRESH_START = datetime(2012, 1, 1, tzinfo=timezone.utc)
FRESH_END = datetime(2017, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
NOTIONAL = 10_000.0
LOOKBACK_DAYS = 60

# Representative mid-window prices, used only to turn COST_PROFILES' fractional
# slippage_pct into the absolute per-side figure entries_v2.CostModel expects.
_REPRESENTATIVE_PRICE = {"XAUUSD": 1400.0, "EURUSD": 1.25, "GBPUSD": 1.50}


def cost_model_for(symbol: str) -> CostModel:
    """Per-instrument CostModel. See module docstring for why commission is not carried
    across.

    XAUUSD deliberately keeps entries_v2.DEFAULT_COSTS (the $0.40/$0.05 gold model) rather
    than being re-derived from COST_PROFILES: every prior XAUUSD result in this project --
    including the 32-trade / PF 1.860 single-asset figure this run is meant to be checked
    against -- was measured on DEFAULT_COSTS. Re-deriving it would make gold ~0.045% per
    round trip instead of ~0.036% and quietly break that comparability, so the one arm with
    an established convention keeps it. EURUSD/GBPUSD have no such convention and are
    derived from COST_PROFILES.
    """
    if symbol == "XAUUSD":
        return DEFAULT_COSTS
    profile = market_data.cost_profile_for(symbol)
    return CostModel(spread=profile.spread,
                      slippage_per_side=profile.slippage_pct * _REPRESENTATIVE_PRICE[symbol],
                      commission=0.0)


def _slice(candles: List[Candle], start: datetime, end: datetime) -> List[Candle]:
    start_ms, end_ms = start.timestamp() * 1000, end.timestamp() * 1000
    return [c for c in candles if start_ms <= c.open_time <= end_ms]


def evaluate_symbol(symbol: str, folds: List[dict], config: SessionOBConfig) -> Optional[dict]:
    """Runs the walk-forward folds for one instrument. Returns None (with a printed
    reason) if that symbol has no usable cached data yet, so a partially-fetched
    dataset degrades to 'fewer arms' rather than crashing the whole comparison."""
    try:
        m5 = data_dukascopy.load_m5_candles("2012-01-01", "2018-01-01", symbol=symbol)
    except FileNotFoundError:
        print(f"  {symbol}: no cache file yet -- skipping.")
        return None
    if len(m5) < 10_000:
        print(f"  {symbol}: only {len(m5)} M5 bars cached -- too sparse to evaluate, skipping.")
        return None

    h1 = data_dukascopy.resample(m5, 60)
    h4 = data_dukascopy.resample(m5, 240)
    costs = cost_model_for(symbol)
    rt_pct = (costs.spread + 2 * costs.slippage_per_side) / _REPRESENTATIVE_PRICE[symbol] * 100
    print(f"  {symbol}: {len(m5)} M5 -> {len(h1)} H1, {len(h4)} H4 bars | "
          f"round-trip cost {rt_pct:.3f}% of price")

    fold_sharpes, all_trades = [], []
    for fold in folds:
        pad_start = fold["train_end"] - timedelta(days=LOOKBACK_DAYS)
        trades = simulate(_slice(h1, pad_start, fold["validate_end"]),
                           _slice(h4, pad_start, fold["validate_end"]),
                           config, NOTIONAL, costs)
        window = [t for t in trades if fold["train_end"] <= t.entry_time < fold["validate_end"]]
        m = compute_metrics(window, NOTIONAL)
        fold_sharpes.append(m["sharpe"] if (m["sharpe"] is not None and m["n_trades"] >= 3) else 0.0)
        all_trades.extend(window)

    return {"symbol": symbol, "fold_sharpes": fold_sharpes, "trades": all_trades,
            "pooled": compute_metrics(all_trades, NOTIONAL),
            "median_sharpe": statistics.median(fold_sharpes)}


def significance(trades: list) -> dict:
    """One-sample t-test vs. a zero-mean null plus a bootstrap CI -- same methodology
    already used for the single-asset result, so the numbers are directly comparable."""
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    if n < 2:
        return {"n": n, "t": None, "p": None, "ci_low": None, "ci_high": None, "p_loss": None}
    t_stat, p_value = stats.ttest_1samp(pnls / NOTIONAL * 100, popmean=0.0)
    rng = np.random.default_rng(42)
    sims = np.array([rng.choice(pnls, n, replace=True).sum() for _ in range(5000)]) / NOTIONAL * 100
    return {"n": n, "t": float(t_stat), "p": float(p_value),
            "ci_low": float(np.percentile(sims, 2.5)), "ci_high": float(np.percentile(sims, 97.5)),
            "p_loss": float((sims < 0).mean() * 100)}


def _report(label: str, pooled: dict, sig: dict, median_sharpe: Optional[float] = None) -> None:
    pf = pooled["profit_factor"]
    print(f"\n{label}")
    print(f"  Trades:          {pooled['n_trades']}")
    print(f"  Win rate:        {pooled['win_rate_pct']:.1f}%")
    print(f"  Profit factor:   {pf:.3f}" if pf is not None else "  Profit factor:   undefined")
    print(f"  Net P&L:         {pooled['net_pnl_pct']:+.2f}%")
    print(f"  Max drawdown:    {pooled['max_drawdown_pct']:.2f}%")
    if median_sharpe is not None:
        print(f"  Median fold Sharpe: {median_sharpe:+.3f}")
    if sig["p"] is not None:
        print(f"  t={sig['t']:.3f}  p={sig['p']:.4f}  "
              f"bootstrap 95% CI [{sig['ci_low']:+.2f}%, {sig['ci_high']:+.2f}%]  "
              f"P(loss)={sig['p_loss']:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-asset session-order-block evaluation.")
    parser.add_argument("--symbols", nargs="+", default=["XAUUSD", "EURUSD", "GBPUSD"])
    args = parser.parse_args()

    folds = generate_folds(FRESH_START, FRESH_END)
    print(f"{len(folds)} walk-forward folds, {FRESH_START.date()} .. {FRESH_END.date()}, "
          f"identical SessionOBConfig on every instrument.\n")
    print("Loading instruments:")

    results = [r for r in (evaluate_symbol(s, folds, SessionOBConfig()) for s in args.symbols) if r]
    if not results:
        print("\nNo instruments had usable data -- nothing to evaluate.")
        return

    print("\n" + "=" * 70 + "\nPER-INSTRUMENT\n" + "=" * 70)
    for r in results:
        _report(r["symbol"], r["pooled"], significance(r["trades"]), r["median_sharpe"])

    if len(results) > 1:
        print("\n" + "=" * 70 + "\nPOOLED ACROSS INSTRUMENTS\n" + "=" * 70)
        pooled_trades = [t for r in results for t in r["trades"]]
        # Pooling is legitimate here because every arm uses the same $10,000 notional and
        # the same parameters, so per-trade P&L is denominated comparably across symbols.
        _report(" + ".join(r["symbol"] for r in results),
                compute_metrics(pooled_trades, NOTIONAL), significance(pooled_trades))
        per_symbol = ", ".join(
            "{} {:+.2f}%".format(r["symbol"], r["pooled"]["net_pnl_pct"]) for r in results)
        print(f"\n  Per-instrument net P&L: {per_symbol}")


if __name__ == "__main__":
    main()
