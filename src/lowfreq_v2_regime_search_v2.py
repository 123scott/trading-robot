# -*- coding: utf-8 -*-
"""
lowfreq_v2_regime_search_v2.py

Second-round regime-parameter search, built specifically to fix the three
concrete failure modes that got the previous round's "winner" rejected (see
data/performance_report.md, "Regime-Parameter WFO Search (2026-08-30)", and
this file's own module docstring items below):

  1. block_adx_transition was a boolean toggle for a FIXED 20-25 band --
     with only two possible values, a "cliff check" against its one
     neighbor was nearly meaningless (nothing to compare a binary flip
     against). Fixed: src/entries_v2.py's LowfreqV2Config now carries
     adx_transition_low/adx_transition_high as real config fields (this
     round's code change), so the band's actual THRESHOLD VALUES can be
     swept and checked for a genuine 2D neighborhood plateau, not just an
     on/off flip.
  2. regime_confirm_bars was searched at every adjacent integer (1,2,3,4,5)
     -- adjacent bar-counts on real trade data are highly correlated (they
     mostly overlap the same trades), so "not cliff-sensitive to its
     immediate neighbor" barely constrains anything. Fixed: this round
     searches a coarser, more separated grid ([2, 4, 6, 8]) where each step
     is a more meaningfully different persistence requirement.
  3. Selection scored purely by MEDIAN fold Sharpe -- a candidate can win
     by being flat-to-bad on most folds and spectacular on one or two
     without being penalized for that dispersion, which is exactly the
     "hyper-sensitive to a single input/window" pattern this round's task
     was told to reject. Fixed: scoring now penalizes cross-fold dispersion
     directly (median_sharpe - STABILITY_PENALTY * IQR(fold_sharpes)), so a
     consistently-mediocre candidate can outrank a spiky one -- exactly the
     tradeoff a real deployment cares about.

Structural params (trend_sma_period, pullback_ema_period,
pullback_tolerance_pct, atr_sl_mult, atr_tp_mult) stay fixed at the
already-locked values from lowfreq_v2_eval.py's own earlier search -- not
re-touched here, same staged-search rationale as the first regime search.

Still uses the SAME 26 purged/embargoed training folds (2018-01-01 to
2025-07-31) for selection -- these already span COVID, the 2022 rate-hike
bear run, multiple ranging years, and the recent rally, so they are
regime-diverse by construction; no new fold logic was needed to fix that
part.

IMPORTANT, read before trusting any TEST-window number this script prints:
the 2025-08-01/2026-07-31 test window has now been used to evaluate a
candidate in THREE separate prior rounds of this project, plus this one
would be a fourth. Every additional look at the same finite window raises
the odds that something clears an arbitrary bar by chance alone, regardless
of the scoring-methodology improvements above -- those fix how the
CANDIDATE is chosen, they do not fix how many times the SAME evaluation
window has been consulted. This script still runs the evaluation (asked
for), but the honest recommendation is to treat a pass here as, at best,
weak corroborating evidence, not confirmation -- and to stop reusing this
specific window for any future round.

Usage:
    python3 -m src.lowfreq_v2_regime_search_v2
    python3 -m src.lowfreq_v2_regime_search_v2 --confirm-bars 2 4 6 8 --stability-penalty 0.5
"""

from __future__ import annotations

import argparse
import statistics
import time
from itertools import product

import numpy as np
from scipy import stats

from src.entries_v2 import LowfreqV2Config, DEFAULT_COSTS
from src.lowfreq_v2_eval import (
    TRAIN_START, TRAIN_END, TEST_START, TEST_END, NOTIONAL,
    load_all_candles, generate_folds, run_window,
)

LOCKED_STRUCTURAL = dict(trend_sma_period=50, pullback_ema_period=21,
                          pullback_tolerance_pct=0.20, atr_sl_mult=2.0, atr_tp_mult=2.5)

DEFAULT_CONFIRM_BARS = [2, 4, 6, 8]
# (low, high) threshold pairs to sweep, plus None meaning "no block at all" (the existing
# baseline). Deliberately centered around the original 20-25 finding but wide enough to
# reveal whether nearby bands work comparably (real signal) or only this exact one does
# (curve-fit) -- see module docstring item 1.
DEFAULT_ADX_BANDS = [None, (15.0, 25.0), (20.0, 25.0), (20.0, 30.0), (15.0, 30.0)]
DEFAULT_STABILITY_PENALTY = 0.5  # weight on cross-fold IQR in the scoring objective

TARGET_WIN_RATE_PCT = 56.0
TARGET_SHARPE = 1.5


def _make_config(confirm_bars: int, adx_band) -> LowfreqV2Config:
    kwargs = dict(**LOCKED_STRUCTURAL, use_regime_filter=True, regime_confirm_bars=confirm_bars)
    if adx_band is None:
        kwargs["block_adx_transition"] = False
    else:
        low, high = adx_band
        kwargs.update(block_adx_transition=True, adx_transition_low=low, adx_transition_high=high)
    return LowfreqV2Config(**kwargs)


def score_combo_stability_penalized(h1_all, daily_all, config: LowfreqV2Config, folds: list,
                                     stability_penalty: float) -> dict:
    """Same per-fold scoring as lowfreq_v2_eval.score_combo_median_sharpe (Sharpe=0.0 for
    folds with <3 trades -- neutral, not a penalty invented to hit a target), but the
    combo-level score is median_sharpe - stability_penalty * IQR(fold_sharpes) instead of
    the bare median -- see module docstring item 3."""
    fold_sharpes = []
    for fold in folds:
        result = run_window(h1_all, daily_all, config, fold["train_end"], fold["validate_end"])
        m = result["metrics"]
        sharpe = m["sharpe"] if (m["sharpe"] is not None and m["n_trades"] >= 3) else 0.0
        fold_sharpes.append(sharpe)
    median_sharpe = statistics.median(fold_sharpes)
    q1, q3 = np.percentile(fold_sharpes, [25, 75])
    iqr = float(q3 - q1)
    stability_score = median_sharpe - stability_penalty * iqr
    return {"median_sharpe": median_sharpe, "iqr": iqr, "stability_score": stability_score,
            "fold_sharpes": fold_sharpes}


def _band_label(adx_band) -> str:
    return "none" if adx_band is None else f"{adx_band[0]:.0f}-{adx_band[1]:.0f}"


def run_grid_search(h1_all, daily_all, folds, confirm_bars_grid, adx_bands, stability_penalty) -> list:
    results = []
    for confirm_bars, adx_band in product(confirm_bars_grid, adx_bands):
        cfg = _make_config(confirm_bars, adx_band)
        scored = score_combo_stability_penalized(h1_all, daily_all, cfg, folds, stability_penalty)
        results.append({"regime_confirm_bars": confirm_bars, "adx_band": adx_band, **scored})
    results.sort(key=lambda r: r["stability_score"], reverse=True)
    return results


def check_stability(results: list, winner: dict, confirm_bars_grid: list, adx_bands: list) -> dict:
    """Neighbor = one step away on EITHER axis (confirm_bars grid step, or adjacent adx_band
    in the swept list) -- a real cliff, not genuine signal, is a neighbor whose stability_score
    is both negative AND at least 1.0 below the winner's."""
    by_key = {(r["regime_confirm_bars"], _band_label(r["adx_band"])): r for r in results}
    bars, band = winner["regime_confirm_bars"], winner["adx_band"]
    bars_idx = confirm_bars_grid.index(bars)
    band_idx = adx_bands.index(band)

    neighbor_keys = []
    if bars_idx > 0:
        neighbor_keys.append((confirm_bars_grid[bars_idx - 1], _band_label(band)))
    if bars_idx < len(confirm_bars_grid) - 1:
        neighbor_keys.append((confirm_bars_grid[bars_idx + 1], _band_label(band)))
    if band_idx > 0:
        neighbor_keys.append((bars, _band_label(adx_bands[band_idx - 1])))
    if band_idx < len(adx_bands) - 1:
        neighbor_keys.append((bars, _band_label(adx_bands[band_idx + 1])))

    neighbors = [by_key[k] for k in neighbor_keys if k in by_key]
    cliff_neighbors = [n for n in neighbors if n["stability_score"] < 0 and
                        (winner["stability_score"] - n["stability_score"]) >= 1.0]
    return {"is_cliff": len(cliff_neighbors) > 0, "neighbors_checked": neighbors,
            "cliff_neighbors": cliff_neighbors}


def select_stable_winner(results: list, confirm_bars_grid: list, adx_bands: list):
    for candidate in results:
        stability = check_stability(results, candidate, confirm_bars_grid, adx_bands)
        if not stability["is_cliff"]:
            return candidate, stability
    return results[0], check_stability(results, results[0], confirm_bars_grid, adx_bands)


def _bootstrap_monte_carlo(pnls: np.ndarray, notional: float, iterations: int = 2000, seed: int = 42) -> dict:
    n = len(pnls)
    rng = np.random.default_rng(seed)
    sim_pnl_pct = np.empty(iterations)
    for i in range(iterations):
        sample = rng.choice(pnls, size=n, replace=True)
        sim_pnl_pct[i] = sample.sum() / notional * 100
    return {"pnl_pct_p2_5": float(np.percentile(sim_pnl_pct, 2.5)),
            "pnl_pct_p50": float(np.percentile(sim_pnl_pct, 50)),
            "pnl_pct_p97_5": float(np.percentile(sim_pnl_pct, 97.5)),
            "prob_of_loss_pct": float((sim_pnl_pct < 0).mean() * 100)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-bars", type=int, nargs="+", default=DEFAULT_CONFIRM_BARS)
    parser.add_argument("--stability-penalty", type=float, default=DEFAULT_STABILITY_PENALTY)
    parser.add_argument("--evaluate-test-window", action="store_true",
                         help="Also run step 3 (the ONE evaluation on 2025-08-01/2026-07-31). Off by "
                              "default -- that window has already been used to evaluate a candidate in "
                              "THREE prior rounds; touching it a fourth time should be a deliberate, "
                              "explicit choice, not something that happens by default on every re-run "
                              "of the training-only search in steps 1-2.")
    args = parser.parse_args()

    confirm_bars_grid = sorted(args.confirm_bars)
    adx_bands = DEFAULT_ADX_BANDS

    print("Loading full Dukascopy M5 cache and resampling to H1/daily...")
    t0 = time.time()
    h1_all, daily_all = load_all_candles("2018-01-01")
    print(f"Loaded {len(h1_all)} H1 bars, {len(daily_all)} daily bars in {time.time()-t0:.1f}s.\n")

    folds = generate_folds(TRAIN_START, TRAIN_END)
    print(f"Generated {len(folds)} purged/embargoed walk-forward folds inside "
          f"{TRAIN_START.date()}-{TRAIN_END.date()} (training only).\n")

    n_combos = len(confirm_bars_grid) * len(adx_bands)
    print(f"--- Step 1: grid search, confirm_bars={confirm_bars_grid} x adx_band="
          f"{[_band_label(b) for b in adx_bands]} ({n_combos} combos), "
          f"stability_penalty={args.stability_penalty} ---")
    print(f"Structural params fixed at {LOCKED_STRUCTURAL}")
    t0 = time.time()
    results = run_grid_search(h1_all, daily_all, folds, confirm_bars_grid, adx_bands, args.stability_penalty)
    print(f"Done in {time.time()-t0:.1f}s. Ranked by stability_score (median_sharpe - "
          f"{args.stability_penalty}*IQR):")
    for r in results:
        print(f"  confirm_bars={r['regime_confirm_bars']}  adx_band={_band_label(r['adx_band']):>8}  "
              f"median_sharpe={r['median_sharpe']:+.3f}  iqr={r['iqr']:.3f}  "
              f"stability_score={r['stability_score']:+.3f}")

    print(f"\n--- Step 2: stability/cliff check on the top candidate ---")
    winner, stability = select_stable_winner(results, confirm_bars_grid, adx_bands)
    print(f"Selected: confirm_bars={winner['regime_confirm_bars']} adx_band={_band_label(winner['adx_band'])} "
          f"(stability_score={winner['stability_score']:+.3f})")
    for n in stability["neighbors_checked"]:
        flag = "  <-- CLIFF" if n in stability["cliff_neighbors"] else ""
        print(f"  neighbor confirm_bars={n['regime_confirm_bars']} adx_band={_band_label(n['adx_band'])}: "
              f"stability_score={n['stability_score']:+.3f}{flag}")
    print(f"Cliff-sensitive: {stability['is_cliff']}")

    final_config = _make_config(winner["regime_confirm_bars"], winner["adx_band"])
    print(f"\nLocked candidate from training-only selection: {final_config.as_dict()}")

    if not args.evaluate_test_window:
        print("\nStopping here -- test-window evaluation (step 3) was not requested "
              "(pass --evaluate-test-window to run it). See this module's docstring for why "
              "that's opt-in, not automatic.")
        return

    print(f"\n--- Step 3: ONE evaluation on the test window {TEST_START.date()} to {TEST_END.date()} ---")
    print("CAVEAT: this window has now been used to evaluate a candidate in THREE prior rounds of this "
          "project -- this would be the FOURTH. Treat a pass here as weak corroborating evidence at best, "
          "not confirmation, regardless of the scoring-methodology improvements above. Reported as-is; "
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
        print(f"  P(loss) across resamples: {mc['prob_of_loss_pct']:.1f}%")
        if len(trades) >= 2:
            returns_pct = pnls / NOTIONAL * 100
            t_stat, p_value = stats.ttest_1samp(returns_pct, popmean=0.0)
            print(f"  One-sample t-test vs. 0: t={t_stat:.3f}, p={p_value:.4f}")
    else:
        print("\nNo trades in the test window -- no Monte Carlo/significance test possible.")


if __name__ == "__main__":
    main()
