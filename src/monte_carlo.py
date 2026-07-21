# -*- coding: utf-8 -*-
"""
monte_carlo.py

Monte Carlo stress test on a symbol/mode's REAL historical trade PnL
sequence (read from data/ledger.csv -- no synthetic trade data is
generated, only the resampling order/composition is randomized).

Methodology: bootstrap resampling WITH replacement. Each of N simulations
draws a synthetic trade sequence of the same length as the real trade
count, sampling from the historical per-trade PnL distribution with
replacement, then builds a synthetic equity curve from a fixed starting
notional to compute that simulation's Max Drawdown % and Total PnL %.

Why bootstrap-with-replacement rather than a pure order permutation: a
pure shuffle of the exact same trades leaves total PnL mathematically
invariant (same numbers, different order) -- it would only produce a
distribution for Max Drawdown, not for PnL. Bootstrapping lets both
requested outputs (95% CI for Max Drawdown AND for Expected PnL) carry
real variance, which is what "test for sequence risk" needs to be useful
for -- at the cost of also reflecting trade-frequency/composition risk,
not pure ordering alone. This is stated explicitly because it's a
methodology choice that affects the numbers.

Usage:
    python -m src.monte_carlo --symbol XAUUSD --mode memory --iterations 5000
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import List

import numpy as np

from src import memory as memory_mod


@dataclass
class MonteCarloResult:
    symbol: str
    mode: str
    iterations: int
    n_trades: int
    notional: float
    actual_total_pnl_pct: float
    actual_max_dd_pct: float
    pnl_pct_p2_5: float
    pnl_pct_p50: float
    pnl_pct_p97_5: float
    max_dd_pct_p2_5: float
    max_dd_pct_p50: float
    max_dd_pct_p97_5: float
    prob_of_loss: float  # fraction of simulations that ended net negative


def _read_trade_pnls(symbol: str, mode: str) -> List[float]:
    with open(memory_mod.LEDGER_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        float(r["pnl"]) for r in rows
        if r["symbol"] == symbol and r["mode"] == mode and r["action"] == "SELL"
    ]


def run_monte_carlo(symbol: str, mode: str, notional: float = 10_000.0,
                     iterations: int = 2000, seed: int = 42) -> MonteCarloResult:
    pnls = _read_trade_pnls(symbol, mode)
    n = len(pnls)
    if n < 2:
        raise ValueError(f"Not enough closed trades for {symbol}/{mode} to run a Monte Carlo simulation (found {n}).")

    pnls_arr = np.array(pnls)
    actual_total_pnl_pct = pnls_arr.sum() / notional * 100
    actual_max_dd_pct = _max_drawdown_pct(pnls_arr, notional)

    rng = np.random.default_rng(seed)
    sim_pnl_pct = np.empty(iterations)
    sim_max_dd_pct = np.empty(iterations)

    for i in range(iterations):
        sample = rng.choice(pnls_arr, size=n, replace=True)
        sim_pnl_pct[i] = sample.sum() / notional * 100
        sim_max_dd_pct[i] = _max_drawdown_pct(sample, notional)

    return MonteCarloResult(
        symbol=symbol, mode=mode, iterations=iterations, n_trades=n, notional=notional,
        actual_total_pnl_pct=actual_total_pnl_pct, actual_max_dd_pct=actual_max_dd_pct,
        pnl_pct_p2_5=float(np.percentile(sim_pnl_pct, 2.5)),
        pnl_pct_p50=float(np.percentile(sim_pnl_pct, 50)),
        pnl_pct_p97_5=float(np.percentile(sim_pnl_pct, 97.5)),
        max_dd_pct_p2_5=float(np.percentile(sim_max_dd_pct, 2.5)),
        max_dd_pct_p50=float(np.percentile(sim_max_dd_pct, 50)),
        max_dd_pct_p97_5=float(np.percentile(sim_max_dd_pct, 97.5)),
        prob_of_loss=float((sim_pnl_pct < 0).mean() * 100),
    )


def _max_drawdown_pct(pnls: np.ndarray, notional: float) -> float:
    equity = notional + np.cumsum(pnls)
    peak = np.maximum.accumulate(np.concatenate(([notional], equity)))[1:]
    drawdown = (peak - equity) / peak * 100
    return float(drawdown.max()) if len(drawdown) else 0.0


def print_result(r: MonteCarloResult) -> None:
    print(f"Monte Carlo stress test: {r.symbol} ({r.mode} mode), {r.n_trades} real closed trades, "
          f"{r.iterations} bootstrap resamples (with replacement), notional ${r.notional:,.2f}")
    print("-" * 78)
    print(f"Actual realized result:      Total PnL {r.actual_total_pnl_pct:+.2f}%   Max DD {r.actual_max_dd_pct:.2f}%")
    print(f"Simulated median:            Total PnL {r.pnl_pct_p50:+.2f}%   Max DD {r.max_dd_pct_p50:.2f}%")
    print(f"95% CI, Total PnL %:         [{r.pnl_pct_p2_5:+.2f}%, {r.pnl_pct_p97_5:+.2f}%]")
    print(f"95% CI, Max Drawdown %:      [{r.max_dd_pct_p2_5:.2f}%, {r.max_dd_pct_p97_5:.2f}%]")
    print(f"P(net loss) across resamples: {r.prob_of_loss:.1f}%")
    if r.n_trades < 30:
        print(f"\nCaveat: only {r.n_trades} real trades feed this simulation -- the resampled")
        print("distribution reflects those trades' composition, not independent new evidence.")
        print("Treat this as a sequence-risk sanity check, not a statistically robust forecast.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo stress test on a symbol/mode's real trade history.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--mode", choices=["raw", "memory"], required=True)
    parser.add_argument("--notional", type=float, default=10_000.0)
    parser.add_argument("--iterations", type=int, default=2000, help="1000-10000 recommended.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_monte_carlo(args.symbol, args.mode, args.notional, args.iterations, args.seed)
    print_result(result)


if __name__ == "__main__":
    main()
