# -*- coding: utf-8 -*-
"""
lowfreq_v2_regime_search.py

Small, staged walk-forward search over the two remaining open parameters
(regime_confirm_bars, block_adx_transition) on top of the ALREADY-locked
structural params (trend_sma_period=50, pullback_ema_period=21,
pullback_tolerance_pct=0.20, atr_sl_mult=2.0, atr_tp_mult=2.5,
use_regime_filter=True) -- those five came out of lowfreq_v2_eval.py's own
48-combo structural grid search on the same TRAIN_START-TRAIN_END data in an
earlier round; re-running that exact search would reproduce the same answer
(the underlying pre-2025-08-01 cached data hasn't changed), so this script
does not repeat it -- it picks up exactly where that search left off,
matching this project's established staged-search convention (see
lowfreq_v2_eval.py / src/optimize.py: "searching a huge space over one
split is itself a form of overfitting" -- small, sequential grids, not one
combinatorial explosion).

Methodology (mirrors lowfreq_v2_eval.py exactly, reusing its fold/scoring
machinery):
  1. Grid search regime_confirm_bars in [1..5] x block_adx_transition in
     [False, True] (10 combos), scored ONLY on the 26 purged/embargoed
     12-month-context/3-month-validate rolling folds inside TRAIN_START
     (2018-01-01) - TRAIN_END (2025-07-31) -- the TEST window is never
     touched during selection.
  2. Stability/cliff check on the winning regime_confirm_bars value: its
     immediate neighbors (+/-1, clipped to [1,5]) must not collapse --
     a combo that only works at one exact integer and craters next door is
     rejected as overfit to fold-specific noise, per this round's explicit
     anti-overfitting requirement, even if it scored highest.
  3. ONE evaluation of the final locked combo on the untouched-by-THIS-
     SEARCH test window (2025-08-01 to 2026-07-31) -- but see the loud
     caveat printed before that section runs: this window has already been
     analyzed in at least two earlier rounds of this project (the original
     persistence-filter discovery, and the 2025/2026 ADX-transition IS/OOS
     round), so this is a RE-ANALYSIS with real accumulated leakage risk,
     not a fresh holdout. The result is reported as-is, whatever it is --
     this function runs exactly once and its result is never fed back into
     step 1 or 2, regardless of whether it clears any target threshold.
  4. Bootstrap Monte Carlo (resampling with replacement, matching
     monte_carlo.py's established methodology and rationale) + a one-sample
     t-test against zero mean return (matching alpha_test.py's established
     methodology) on the test-window trades, for a significance read.

Usage:
    python3 -m src.lowfreq_v2_regime_search
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np
from scipy import stats

from src.entries_v2 import LowfreqV2Config, compute_metrics, DEFAULT_COSTS
from src.lowfreq_v2_eval import (
    TRAIN_START, TRAIN_END, TEST_START, TEST_END, NOTIONAL,
    load_all_candles, generate_folds, run_window, score_combo_median_sharpe,
)

# The already-locked structural winner from lowfreq_v2_eval.py's own earlier
# 48-combo search -- held fixed here, not re-derived.
LOCKED_STRUCTURAL = dict(trend_sma_period=50, pullback_ema_period=21,
                          pullback_tolerance_pct=0.20, atr_sl_mult=2.0, atr_tp_mult=2.5)

REGIME_GRID = {
    "regime_confirm_bars": [1, 2, 3, 4, 5],
    "block_adx_transition": [False, True],
}

TARGET_WIN_RATE_PCT = 56.0
TARGET_SHARPE = 1.5


def _make_config(regime_confirm_bars: int, block_adx_transition: bool) -> LowfreqV2Config:
    return LowfreqV2Config(**LOCKED_STRUCTURAL, use_regime_filter=True,
                            regime_confirm_bars=regime_confirm_bars,
                            block_adx_transition=block_adx_transition)


def run_grid_search(h1_all, daily_all, folds) -> list:
    results = []
    for confirm_bars, adx_block in product(REGIME_GRID["regime_confirm_bars"], REGIME_GRID["block_adx_transition"]):
        cfg = _make_config(confirm_bars, adx_block)
        scored = score_combo_median_sharpe(h1_all, daily_all, cfg, folds)
        results.append({"regime_confirm_bars": confirm_bars, "block_adx_transition": adx_block,
                         "median_sharpe": scored["median_sharpe"], "fold_sharpes": scored["fold_sharpes"]})
    results.sort(key=lambda r: r["median_sharpe"], reverse=True)
    return results


def check_stability(results: list, winner: dict) -> dict:
    """
    Rejects the winner if its immediate regime_confirm_bars neighbors (same
    block_adx_transition setting) collapse -- a real cliff, not genuine signal.
    "Collapse" = neighbor's median_sharpe is both negative AND at least 1.0
    below the winner's (an absolute, not relative, threshold -- relative
    thresholds misbehave near zero/negative baselines).
    """
    by_key = {(r["regime_confirm_bars"], r["block_adx_transition"]): r for r in results}
    bars, adx = winner["regime_confirm_bars"], winner["block_adx_transition"]
    neighbors = [by_key.get((bars - 1, adx)), by_key.get((bars + 1, adx))]
    neighbors = [n for n in neighbors if n is not None]

    cliff_neighbors = [n for n in neighbors if n["median_sharpe"] < 0 and
                        (winner["median_sharpe"] - n["median_sharpe"]) >= 1.0]
    return {"is_cliff": len(cliff_neighbors) > 0, "neighbors_checked": neighbors,
            "cliff_neighbors": cliff_neighbors}


def select_stable_winner(results: list) -> dict:
    """Walks the ranked list, picking the first candidate that passes the cliff check."""
    for candidate in results:
        stability = check_stability(results, candidate)
        if not stability["is_cliff"]:
            return candidate, stability
    # Every candidate is cliff-sensitive -- report the top-ranked one but flag it loudly
    # rather than silently picking something the caller didn't ask for.
    return results[0], check_stability(results, results[0])


def _bootstrap_monte_carlo(pnls: np.ndarray, notional: float, iterations: int = 2000, seed: int = 42) -> dict:
    """Same methodology as src/monte_carlo.py (bootstrap WITH replacement, not
    a pure order permutation -- see that module's docstring for why: a pure
    shuffle of identical trades leaves total PnL mathematically invariant,
    it would only produce a distribution for max drawdown, not PnL)."""
    n = len(pnls)
    rng = np.random.default_rng(seed)
    sim_pnl_pct = np.empty(iterations)
    sim_max_dd_pct = np.empty(iterations)
    for i in range(iterations):
        sample = rng.choice(pnls, size=n, replace=True)
        sim_pnl_pct[i] = sample.sum() / notional * 100
        equity = notional + np.cumsum(sample)
        peak = np.maximum.accumulate(np.concatenate(([notional], equity)))[1:]
        sim_max_dd_pct[i] = ((peak - equity) / peak * 100).max()
    return {
        "pnl_pct_p2_5": float(np.percentile(sim_pnl_pct, 2.5)),
        "pnl_pct_p50": float(np.percentile(sim_pnl_pct, 50)),
        "pnl_pct_p97_5": float(np.percentile(sim_pnl_pct, 97.5)),
        "max_dd_pct_p50": float(np.percentile(sim_max_dd_pct, 50)),
        "max_dd_pct_p97_5": float(np.percentile(sim_max_dd_pct, 97.5)),
        "prob_of_loss_pct": float((sim_pnl_pct < 0).mean() * 100),
    }


def _significance_t_test(pnls: np.ndarray, notional: float) -> dict:
    """Same methodology as src/alpha_test.py's edge_t_test: one-sample t-test
    of per-trade return% against a zero-mean null."""
    returns_pct = pnls / notional * 100
    t_stat, p_value = stats.ttest_1samp(returns_pct, popmean=0.0)
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


def main() -> None:
    print("Loading full Dukascopy M5 cache and resampling to H1/daily...")
    t0 = time.time()
    h1_all, daily_all = load_all_candles("2018-01-01")
    print(f"Loaded {len(h1_all)} H1 bars, {len(daily_all)} daily bars in {time.time()-t0:.1f}s.\n")

    folds = generate_folds(TRAIN_START, TRAIN_END)
    print(f"Generated {len(folds)} purged/embargoed walk-forward folds inside "
          f"{TRAIN_START.date()}-{TRAIN_END.date()} (training only -- TEST window not touched yet).\n")

    print(f"--- Step 1: grid search over regime_confirm_bars x block_adx_transition "
          f"({len(REGIME_GRID['regime_confirm_bars']) * len(REGIME_GRID['block_adx_transition'])} combos), "
          f"structural params fixed at {LOCKED_STRUCTURAL} ---")
    t0 = time.time()
    results = run_grid_search(h1_all, daily_all, folds)
    print(f"Grid search done in {time.time()-t0:.1f}s. Ranked by median fold Sharpe (training only):")
    for r in results:
        print(f"  confirm_bars={r['regime_confirm_bars']}  adx_block={r['block_adx_transition']!s:5}  "
              f"median_sharpe={r['median_sharpe']:+.3f}")

    print(f"\n--- Step 2: stability/cliff check on the top candidate ---")
    winner, stability = select_stable_winner(results)
    print(f"Selected: confirm_bars={winner['regime_confirm_bars']} adx_block={winner['block_adx_transition']} "
          f"(median_sharpe={winner['median_sharpe']:+.3f})")
    if stability["neighbors_checked"]:
        for n in stability["neighbors_checked"]:
            print(f"  neighbor confirm_bars={n['regime_confirm_bars']}: median_sharpe={n['median_sharpe']:+.3f}"
                  + ("  <-- CLIFF" if n in stability["cliff_neighbors"] else ""))
    print(f"Cliff-sensitive: {stability['is_cliff']}"
          + ("  (WARNING: even the selected candidate is cliff-sensitive -- see above)" if stability["is_cliff"] else ""))

    final_config = _make_config(winner["regime_confirm_bars"], winner["block_adx_transition"])

    print(f"\n--- Step 3: ONE evaluation on the test window {TEST_START.date()} to {TEST_END.date()} ---")
    print("CAVEAT, read before trusting this number: this exact window has already been analyzed in at "
          "least two earlier rounds of this project (the original persistence-filter discovery, and the "
          "2025/2026 ADX-transition IS/OOS round) -- this is a RE-ANALYSIS of a window that has informed "
          "prior decisions, not a fresh, untainted out-of-sample test. Reported as-is regardless of outcome; "
          "this evaluation runs exactly once and is not fed back into step 1 or 2.")
    test_result = run_window(h1_all, daily_all, final_config, TEST_START, TEST_END)
    trades, metrics = test_result["trades"], test_result["metrics"]

    print(f"\nLocked final config: {final_config.as_dict()}")
    print(f"Test-window trades: {metrics['n_trades']}")
    print(f"Win rate:            {metrics['win_rate_pct']:.1f}%   (target: >= {TARGET_WIN_RATE_PCT:.0f}%)")
    pf_str = f"{metrics['profit_factor']:.3f}" if metrics["profit_factor"] is not None else "undefined"
    print(f"Profit factor:       {pf_str}")
    sharpe_str = f"{metrics['sharpe']:.3f}" if metrics["sharpe"] is not None else "undefined (too few trades)"
    print(f"Sharpe:              {sharpe_str}   (target: >= {TARGET_SHARPE})")
    print(f"Max drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"Net P&L:             {metrics['net_pnl_pct']:+.2f}%")

    hit_win_rate = metrics["win_rate_pct"] >= TARGET_WIN_RATE_PCT
    hit_sharpe = metrics["sharpe"] is not None and metrics["sharpe"] >= TARGET_SHARPE
    print(f"\nTarget met -- win rate: {hit_win_rate}   Sharpe: {hit_sharpe}")

    if trades:
        pnls = np.array([t.pnl for t in trades])
        print(f"\n--- Step 4: bootstrap Monte Carlo (2000 iterations) + significance t-test on these "
              f"{len(trades)} test-window trades ---")
        mc = _bootstrap_monte_carlo(pnls, NOTIONAL)
        print(f"  Net P&L 95% CI:    [{mc['pnl_pct_p2_5']:+.2f}%, {mc['pnl_pct_p97_5']:+.2f}%]  "
              f"(median {mc['pnl_pct_p50']:+.2f}%)")
        print(f"  Max DD 95th pctile: {mc['max_dd_pct_p97_5']:.2f}%  (median {mc['max_dd_pct_p50']:.2f}%)")
        print(f"  P(loss) across resamples: {mc['prob_of_loss_pct']:.1f}%")
        if len(trades) >= 2:
            sig = _significance_t_test(pnls, NOTIONAL)
            print(f"  One-sample t-test vs. 0: t={sig['t_stat']:.3f}, p={sig['p_value']:.4f}")
        else:
            print("  Too few trades for a t-test.")
    else:
        print("\nNo trades in the test window -- no Monte Carlo/significance test possible.")


if __name__ == "__main__":
    main()
