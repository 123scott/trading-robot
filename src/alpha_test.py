# -*- coding: utf-8 -*-
"""
alpha_test.py

Two independent statistical tests for whether a strategy has genuine edge,
not just noise or free exposure to the underlying's own drift:

1. edge_t_test(): one-sample t-test on real closed-trade returns against
   a null hypothesis of zero mean return. Answers "is the average trade
   significantly different from breakeven?"

2. alpha_beta_vs_buy_and_hold(): regresses the strategy's daily returns
   against the underlying's own daily buy-and-hold returns (OLS). Answers
   the more literal question: after accounting for the fact that this is
   a long-only strategy that's sometimes just holding the same asset a
   buy-and-hold investor holds, is there residual (alpha) return left
   over -- i.e. does the *timing* (entries/exits/memory filtering) add
   value, or is the strategy's edge fully explained by beta (being in the
   market some fraction of the time during a rising asset)?

   Since this strategy is long-only, unleveraged, all-in-or-flat, its
   daily return equals the benchmark's daily return whenever a position
   is open, and 0 while flat -- so the daily strategy-return series is
   reconstructed directly from the ledger's BUY/SELL timestamps + the
   benchmark's own daily closes, not simulated separately.

Both tests read real data only: data/ledger.csv for trade history,
real market data (via market_data.fetch_candles) for the benchmark
series. No synthetic returns are generated.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from scipy import stats

from src import market_data
from src import memory as memory_mod

TRADING_DAYS_PER_YEAR = 252


@dataclass
class EdgeTTestResult:
    symbol: str
    mode: str
    n_trades: int
    mean_return_pct: float
    stdev_return_pct: float
    t_stat: Optional[float]
    p_value: Optional[float]
    significant_5pct: bool


@dataclass
class AlphaBetaResult:
    symbol: str
    mode: str
    n_days: int
    pct_time_in_market: float
    alpha_annualized_pct: float
    alpha_t_stat: float
    alpha_p_value: float
    alpha_significant_5pct: bool
    beta: float
    r_squared: float
    strategy_cagr_pct: float
    benchmark_cagr_pct: float


def _read_ledger() -> List[dict]:
    with open(memory_mod.LEDGER_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def edge_t_test(symbol: str, mode: str, notional: float = 10_000.0) -> EdgeTTestResult:
    """One-sample t-test: is the mean trade return significantly different from zero?"""
    rows = _read_ledger()
    pnls = [float(r["pnl"]) for r in rows if r["symbol"] == symbol and r["mode"] == mode and r["action"] == "SELL"]
    n = len(pnls)

    if n < 2:
        return EdgeTTestResult(symbol=symbol, mode=mode, n_trades=n, mean_return_pct=0.0, stdev_return_pct=0.0,
                                t_stat=None, p_value=None, significant_5pct=False)

    returns_pct = [p / notional * 100 for p in pnls]
    mean_r = sum(returns_pct) / n
    stdev_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns_pct) / (n - 1))

    if stdev_r == 0:
        return EdgeTTestResult(symbol=symbol, mode=mode, n_trades=n, mean_return_pct=mean_r, stdev_return_pct=0.0,
                                t_stat=None, p_value=None, significant_5pct=False)

    t_stat, p_value = stats.ttest_1samp(returns_pct, popmean=0.0)
    return EdgeTTestResult(
        symbol=symbol, mode=mode, n_trades=n, mean_return_pct=mean_r, stdev_return_pct=stdev_r,
        t_stat=float(t_stat), p_value=float(p_value), significant_5pct=bool(p_value < 0.05),
    )


def _position_intervals(rows: List[dict], symbol: str, mode: str) -> List[tuple]:
    """Returns [(buy_datetime, sell_datetime_or_None), ...] sorted chronologically."""
    trades = sorted(
        (r for r in rows if r["symbol"] == symbol and r["mode"] == mode and r["action"] in ("BUY", "SELL")),
        key=lambda r: r["timestamp"],
    )
    intervals = []
    open_ts = None
    for r in trades:
        ts = datetime.fromisoformat(r["timestamp"])
        if r["action"] == "BUY":
            open_ts = ts
        elif r["action"] == "SELL" and open_ts is not None:
            intervals.append((open_ts, ts))
            open_ts = None
    if open_ts is not None:
        intervals.append((open_ts, None))  # still open at end of backtest
    return intervals


def alpha_beta_vs_buy_and_hold(symbol: str, mode: str, notional: float = 10_000.0,
                                start: str = "2018-01-01") -> AlphaBetaResult:
    """OLS regression of daily strategy returns on daily benchmark (buy-and-hold) returns."""
    rows = _read_ledger()
    intervals = _position_intervals(rows, symbol, mode)

    candles = market_data.fetch_candles(symbol=symbol, interval="1d", start=start)
    closes = [(c.open_time, c.close) for c in candles]

    bench_returns: List[float] = []
    strat_returns: List[float] = []
    in_market_days = 0

    for i in range(1, len(closes)):
        t_prev_ms, c_prev = closes[i - 1]
        t_ms, c = closes[i]
        if c_prev == 0 or c != c or c_prev != c_prev:  # skip NaN/zero
            continue
        bench_r = (c - c_prev) / c_prev
        day = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)

        in_position = any(
            buy_ts <= day and (sell_ts is None or day <= sell_ts)
            for buy_ts, sell_ts in intervals
        )
        strat_r = bench_r if in_position else 0.0
        if in_position:
            in_market_days += 1

        bench_returns.append(bench_r)
        strat_returns.append(strat_r)

    n = len(bench_returns)
    if n < 10:
        raise ValueError(f"Not enough daily data ({n} days) to run an alpha/beta regression for {symbol}/{mode}.")

    mean_b = sum(bench_returns) / n
    mean_s = sum(strat_returns) / n
    var_b = sum((b - mean_b) ** 2 for b in bench_returns)
    cov_sb = sum((strat_returns[i] - mean_s) * (bench_returns[i] - mean_b) for i in range(n))

    beta = cov_sb / var_b if var_b > 0 else 0.0
    alpha_daily = mean_s - beta * mean_b

    predicted = [alpha_daily + beta * b for b in bench_returns]
    residuals = [strat_returns[i] - predicted[i] for i in range(n)]
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((s - mean_s) ** 2 for s in strat_returns)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard error of the OLS intercept (alpha), for a t-test that alpha != 0.
    dof = n - 2
    mse = ss_res / dof if dof > 0 else 0.0
    se_alpha = math.sqrt(mse * (1.0 / n + mean_b ** 2 / var_b)) if var_b > 0 and dof > 0 else 0.0

    if se_alpha > 0:
        t_alpha = alpha_daily / se_alpha
        p_alpha = float(2 * stats.t.sf(abs(t_alpha), df=dof))
    else:
        t_alpha, p_alpha = 0.0, 1.0

    strategy_total_return = 1.0
    benchmark_total_return = 1.0
    for r in strat_returns:
        strategy_total_return *= (1 + r)
    for r in bench_returns:
        benchmark_total_return *= (1 + r)

    years = n / TRADING_DAYS_PER_YEAR
    strategy_cagr = (strategy_total_return ** (1 / years) - 1) * 100 if years > 0 and strategy_total_return > 0 else 0.0
    benchmark_cagr = (benchmark_total_return ** (1 / years) - 1) * 100 if years > 0 and benchmark_total_return > 0 else 0.0

    return AlphaBetaResult(
        symbol=symbol, mode=mode, n_days=n, pct_time_in_market=in_market_days / n * 100,
        alpha_annualized_pct=alpha_daily * TRADING_DAYS_PER_YEAR * 100,
        alpha_t_stat=t_alpha, alpha_p_value=p_alpha, alpha_significant_5pct=p_alpha < 0.05,
        beta=beta, r_squared=r_squared,
        strategy_cagr_pct=strategy_cagr, benchmark_cagr_pct=benchmark_cagr,
    )


def print_alpha_report(symbol: str, mode: str, notional: float = 10_000.0) -> None:
    tt = edge_t_test(symbol, mode, notional)
    print(f"--- Edge t-test: {symbol} ({mode}) ---")
    if tt.t_stat is None:
        print(f"  Not enough closed trades ({tt.n_trades}) or zero variance -- cannot compute.")
    else:
        sig = "YES" if tt.significant_5pct else "no"
        print(f"  n={tt.n_trades} trades, mean return {tt.mean_return_pct:+.3f}%, "
              f"t={tt.t_stat:.2f}, p={tt.p_value:.4f}  =>  statistically significant edge at 5%? {sig}")

    print(f"--- Alpha/Beta vs. buy-and-hold: {symbol} ({mode}) ---")
    try:
        ab = alpha_beta_vs_buy_and_hold(symbol, mode, notional)
    except ValueError as e:
        print(f"  {e}")
        return
    sig = "YES" if ab.alpha_significant_5pct else "no"
    print(f"  {ab.n_days} trading days, in market {ab.pct_time_in_market:.1f}% of the time")
    print(f"  Beta: {ab.beta:.3f} (exposure to the underlying while in a position)")
    print(f"  Alpha (annualized): {ab.alpha_annualized_pct:+.2f}%  (t={ab.alpha_t_stat:.2f}, p={ab.alpha_p_value:.4f})  "
          f"=>  statistically significant alpha at 5%? {sig}")
    print(f"  R-squared: {ab.r_squared:.3f}")
    print(f"  Strategy CAGR: {ab.strategy_cagr_pct:.2f}%  |  Buy-and-hold CAGR: {ab.benchmark_cagr_pct:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Statistical edge/alpha testing against real ledger data.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--mode", choices=["raw", "memory"], required=True)
    parser.add_argument("--notional", type=float, default=10_000.0)
    args = parser.parse_args()
    print_alpha_report(args.symbol, args.mode, args.notional)
