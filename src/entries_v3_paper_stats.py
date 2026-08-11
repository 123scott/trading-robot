# -*- coding: utf-8 -*-
"""
entries_v3_paper_stats.py

Small script: reads data/entries_v3_paper_trades.csv and prints current
forward-test stats next to the backtest reference points from this
round's report (Train/Validation/Holdout). Given the backtested
frequency is only ~3-4 trades/year, this will likely show 0 trades for
a long time -- that's expected, not broken.

Usage:
    python -m src.entries_v3_paper_stats
"""

from __future__ import annotations

import csv
import os

from src import memory

PAPER_LOG_PATH = os.path.join(memory.DATA_DIR, "entries_v3_paper_trades.csv")

# From this round's report (data/performance_report.md) -- updated by hand if revised.
REFERENCES = [
    {"label": "Train (2018 - 2023-02, 17 trades)", "n": 17, "win_rate_pct": 29.4, "pf": 0.707, "net_pnl_pct": -0.39},
    {"label": "Validation (2023-02 - 2024-11, 4 trades)", "n": 4, "win_rate_pct": 25.0, "pf": 1.204, "net_pnl_pct": 0.07},
    {"label": "Holdout (2024-11 - 2026-08, 8 trades)", "n": 8, "win_rate_pct": 12.5, "pf": 0.146, "net_pnl_pct": -1.01},
]


def load_paper_trades() -> list:
    if not os.path.exists(PAPER_LOG_PATH):
        return []
    with open(PAPER_LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_live_stats(rows: list, notional: float = 10_000.0) -> dict:
    pnls = [float(r["pnl"]) for r in rows]
    n = len(pnls)
    if n == 0:
        return {"n": 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    return {"n": n, "win_rate_pct": len(wins) / n * 100,
            "pf": (gross_profit / gross_loss) if gross_loss > 0 else None,
            "net_pnl_pct": sum(pnls) / notional * 100}


def _fmt(v, fmt="{:.2f}"):
    return fmt.format(v) if v is not None else "--"


def main() -> None:
    rows = load_paper_trades()
    live = compute_live_stats(rows)

    print("=== XAUUSD_LOWFREQ v3 -- forward-test vs. backtest ===\n")
    print(f"{'':40} {'Trades':>7} {'Win%':>7} {'PF':>7} {'NetPnL%':>9}")
    if live["n"] == 0:
        print(f"{'LIVE (paper, so far)':40} {0:>7} {'--':>7} {'--':>7} {'--':>9}")
        print("\nNo closed paper trades yet -- backtested frequency is ~3-4/year, so long gaps between "
              "entries are expected. Run `python -m src.entries_v3_paper` to keep collecting them.")
    else:
        print(f"{'LIVE (paper, so far)':40} {live['n']:7d} {_fmt(live['win_rate_pct']):>7} "
              f"{_fmt(live['pf']):>7} {_fmt(live['net_pnl_pct']):>9}")
    for ref in REFERENCES:
        print(f"{ref['label']:40.40} {ref['n']:7d} {_fmt(ref['win_rate_pct']):>7} "
              f"{_fmt(ref['pf']):>7} {_fmt(ref['net_pnl_pct']):>9}")

    print("\nEvery backtest segment above has a small sample (4-17 trades) -- none of these numbers, "
          "live or backtest, should be treated as statistically settled.")


if __name__ == "__main__":
    main()
