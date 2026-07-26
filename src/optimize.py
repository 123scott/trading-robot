# -*- coding: utf-8 -*-
"""
optimize.py

Train/test split parameter search for the SMA crossover strategy, built
specifically to answer "did tuning the strategy actually help, or did we
just overfit the training window?"

Methodology:
  1. Grid-search fast/slow SMA periods using ONLY the training period's
     real candles (never the test period -- that would be look-ahead bias
     and defeat the entire point of the split). Scoring is training-period
     Sharpe with a minimum-trade-count floor, so a lucky low-n combo can't
     win by chance. The grid is deliberately small (a few dozen combos,
     not thousands) -- searching a huge space over one train/test split is
     itself a form of overfitting (data snooping), so this is a first-pass
     sanity check, not an exhaustive optimizer.
  2. The chosen parameters, AND the original baseline (9/21) parameters,
     are then each run as a single continuous backtest over the FULL
     history (train+test), in both raw and memory mode, each tagged with
     its own ledger_symbol so memory mode's SKIP decisions never leak
     between experiments (see trading_robot.run_replay's ledger_symbol
     param). A continuous run is equivalent to running train-then-test
     with carried-over memory (a live-deployed bot doesn't forget its
     training-period losses when the calendar flips) -- so metrics are
     computed by slicing ONE continuous run's trade log at the train/test
     boundary, not by re-running twice.
  3. Reports train vs test metrics for both baseline and optimized
     parameters side by side. The overfitting tell: optimized params doing
     much better than baseline IN TRAINING but not (or worse) OUT OF
     TEST is the signature of overfitting; holding up in test is the
     signature of a real (if modest) improvement.

No synthetic data anywhere -- grid search and both final backtests run on
real market data via market_data.fetch_candles.

Usage:
    python -m src.optimize --symbol XAUUSD --train-end 2025-07-01
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from src import market_data
from src import memory as memory_mod
from src.backtest_structures import detect_crossovers
from src.backtest_entries import Action, PositionTracker
from src.trading_robot import run_replay

TRADING_DAYS_PER_YEAR = 252


@dataclass
class GridResult:
    fast: int
    slow: int
    n_trades: int
    sharpe: Optional[float]
    total_pnl_pct: float


def _simulate_raw_pnls(candles, fast_period: int, slow_period: int, notional: float,
                        costs) -> List[Tuple[datetime, float]]:
    """Pure in-memory raw-mode simulation -- no ledger I/O, no memory system. Returns [(exit_time, pnl), ...]."""
    signals = detect_crossovers(candles, fast_period=fast_period, slow_period=slow_period)
    tracker = PositionTracker()
    open_entry_price = None
    open_qty = None
    open_commission = None
    closed: List[Tuple[datetime, float]] = []

    for signal in signals:
        intent = tracker.next_intent(signal)
        if intent is None:
            continue
        if intent.action == Action.BUY:
            fill = intent.price + costs.spread / 2 + intent.price * costs.slippage_pct
            tracker.apply(Action.BUY)
            open_entry_price = fill
            open_qty = notional / fill
            open_commission = notional * costs.commission_pct
        else:
            fill = intent.price - costs.spread / 2 - intent.price * costs.slippage_pct
            exit_commission = fill * open_qty * costs.commission_pct
            gross = (fill - open_entry_price) * open_qty if open_entry_price is not None else 0.0
            pnl = gross - (open_commission or 0.0) - exit_commission
            tracker.apply(Action.SELL)
            closed.append((datetime.fromtimestamp(intent.open_time / 1000, tz=timezone.utc), pnl))
            open_entry_price = open_qty = open_commission = None

    return closed


def _sharpe(pnls_with_time: List[Tuple[datetime, float]], notional: float) -> Optional[float]:
    if len(pnls_with_time) < 5:
        return None
    returns = [p / notional for _, p in pnls_with_time]
    times = sorted(t for t, _ in pnls_with_time)
    years = (times[-1] - times[0]).total_seconds() / (365.25 * 24 * 3600)
    if years <= 0:
        return None
    trades_per_year = len(returns) / years
    mean_r = statistics.mean(returns)
    stdev_r = statistics.pstdev(returns)
    if stdev_r == 0:
        return None
    return mean_r / stdev_r * math.sqrt(trades_per_year)


def grid_search(symbol: str, train_start: str, train_end: str, notional: float = 10_000.0,
                 fast_range=(5, 7, 9, 12, 15), slow_range=(15, 20, 21, 26, 30, 40, 50),
                 min_gap: int = 8, min_trades: int = 15, log=print) -> List[GridResult]:
    """Grid search on TRAINING DATA ONLY. Never touches the test period."""
    log(f"Fetching {symbol} training candles ({train_start} -> {train_end}) for grid search...")
    candles = market_data.fetch_candles(symbol=symbol, interval="1d", start=train_start, end=train_end)
    log(f"Fetched {len(candles)} training candles.\n")
    costs = market_data.cost_profile_for(symbol)

    results = []
    combos = [(f, s) for f in fast_range for s in slow_range if s - f >= min_gap]
    log(f"Testing {len(combos)} (fast, slow) SMA combinations on training data only...")
    for fast, slow in combos:
        closed = _simulate_raw_pnls(candles, fast, slow, notional, costs)
        total_pnl_pct = sum(p for _, p in closed) / notional * 100
        sharpe = _sharpe(closed, notional)
        results.append(GridResult(fast=fast, slow=slow, n_trades=len(closed), sharpe=sharpe, total_pnl_pct=total_pnl_pct))

    eligible = [r for r in results if r.n_trades >= min_trades and r.sharpe is not None]
    eligible.sort(key=lambda r: r.sharpe, reverse=True)
    log(f"{len(eligible)}/{len(results)} combinations had >= {min_trades} training trades (required to be eligible).\n")
    return eligible


def _read_ledger() -> List[dict]:
    with open(memory_mod.LEDGER_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _clear_ledger_symbol(tag: str) -> None:
    """Removes any pre-existing rows for `tag` so reruns of this experiment don't duplicate (a bug we hit before)."""
    rows = _read_ledger()
    kept = [r for r in rows if r["symbol"] != tag]
    if len(kept) == len(rows):
        return
    with open(memory_mod.LEDGER_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=memory_mod.LEDGER_HEADER)
        w.writeheader()
        w.writerows(kept)
    with open(memory_mod.LEARNINGS_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    kept_lines = [ln for ln in lines if not (ln.startswith("- WARNING") and f"WARNING: {tag} " in ln)]
    with open(memory_mod.LEARNINGS_PATH, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)


def _metrics_for_slice(rows: List[dict], tag: str, mode: str, notional: float,
                        since: Optional[str], before: Optional[str]) -> dict:
    sells = [
        r for r in rows
        if r["symbol"] == tag and r["mode"] == mode and r["action"] == "SELL"
        and (since is None or r["timestamp"] >= since)
        and (before is None or r["timestamp"] < before)
    ]
    skips = [
        r for r in rows
        if r["symbol"] == tag and r["mode"] == mode and r["action"] == "SKIP"
        and (since is None or r["timestamp"] >= since)
        and (before is None or r["timestamp"] < before)
    ]
    pnls = [float(r["pnl"]) for r in sells]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    closed = len(pnls)
    win_rate = wins / closed * 100 if closed else 0.0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    net_pnl = sum(pnls)

    equity = notional
    peak = notional
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)

    time_pairs = [(datetime.fromisoformat(r["timestamp"]), float(r["pnl"])) for r in sells]
    sharpe = _sharpe(time_pairs, notional)

    return {
        "trades": closed, "skips": len(skips), "wins": wins, "losses": losses,
        "win_rate_pct": win_rate, "profit_factor": profit_factor, "net_pnl": net_pnl,
        "net_pnl_pct": net_pnl / notional * 100, "max_drawdown_pct": max_dd, "sharpe": sharpe,
    }


def run_experiment(symbol: str = "XAUUSD", train_start: str = "2018-01-01",
                    train_end: str = "2025-07-01", notional: float = 10_000.0, log=print) -> dict:
    grid = grid_search(symbol, train_start, train_end, notional, log=log)
    if not grid:
        raise RuntimeError("Grid search found no eligible (fast, slow) combination -- try lowering min_trades.")

    log("Top 5 training-period candidates by Sharpe:")
    for r in grid[:5]:
        log(f"  SMA({r.fast},{r.slow}): {r.n_trades} trades, Sharpe {r.sharpe:.2f}, train PnL {r.total_pnl_pct:+.2f}%")
    best = grid[0]
    log(f"\nSelected SMA({best.fast},{best.slow}) -- best training Sharpe ({best.sharpe:.2f}) among eligible combos.\n")

    experiments = [
        ("XAUUSD_BASE", 9, 21, "Baseline (original, untuned)"),
        ("XAUUSD_OPT", best.fast, best.slow, f"Optimized (SMA {best.fast}/{best.slow}, selected on training data only)"),
    ]

    for tag, fast, slow, label in experiments:
        _clear_ledger_symbol(tag)
        for mode in ("raw", "memory"):
            log(f"=== {label} -- {mode} mode, full history ===")
            run_replay(symbol=symbol, start=train_start, mode=mode, notional=notional,
                       fast_period=fast, slow_period=slow, ledger_symbol=tag, log=log)

    rows = _read_ledger()
    report = {}
    for tag, fast, slow, label in experiments:
        report[tag] = {"label": label, "fast": fast, "slow": slow}
        for mode in ("raw", "memory"):
            report[tag][mode] = {
                "train": _metrics_for_slice(rows, tag, mode, notional, since=None, before=train_end),
                "test": _metrics_for_slice(rows, tag, mode, notional, since=train_end, before=None),
            }
    return report


def print_report(report: dict, train_end: str) -> None:
    print("\n" + "=" * 100)
    print(f"TRAIN/TEST SPLIT RESULTS (train < {train_end} <= test)")
    print("=" * 100)
    for tag, data in report.items():
        print(f"\n--- {data['label']} [{tag}] ---")
        header = f"{'Mode':8} {'Period':6} {'Trades':7} {'Win%':>7} {'PF':>8} {'MaxDD%':>8} {'Sharpe':>8} {'NetPnL%':>9}"
        print(header)
        print("-" * len(header))
        for mode in ("raw", "memory"):
            for period in ("train", "test"):
                m = data[mode][period]
                pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "undef"
                sh = f"{m['sharpe']:.2f}" if m["sharpe"] is not None else "--"
                print(f"{mode:8} {period:6} {m['trades']:7} {m['win_rate_pct']:7.1f} {pf:>8} "
                      f"{m['max_drawdown_pct']:8.2f} {sh:>8} {m['net_pnl_pct']:9.2f}")

    print("\n--- Overfitting check (memory mode: test Sharpe vs train Sharpe) ---")
    for tag, data in report.items():
        tr = data["memory"]["train"]["sharpe"]
        te = data["memory"]["test"]["sharpe"]
        if tr is None or te is None:
            print(f"{tag}: insufficient trades in train or test to compare.")
            continue
        verdict = "held up" if te >= tr * 0.5 else "DEGRADED SUBSTANTIALLY (overfitting signal)"
        print(f"{tag}: train Sharpe {tr:.2f} -> test Sharpe {te:.2f}  [{verdict}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test split parameter search for the SMA crossover strategy.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--train-end", default="2025-07-01")
    parser.add_argument("--notional", type=float, default=10_000.0)
    args = parser.parse_args()

    report = run_experiment(args.symbol, args.train_start, args.train_end, args.notional)
    print_report(report, args.train_end)


if __name__ == "__main__":
    main()
