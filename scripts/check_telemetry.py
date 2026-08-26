# -*- coding: utf-8 -*-
"""
check_telemetry.py

Lightweight, read-only reporting utility over scripts/run_paper_daemon.py's
persisted state (data/paper_state.json). Prints a one-shot summary: running
days, total simulated fills, realized win rate / profit factor / max
drawdown, and average observed fill spread vs. the $0.40 spread / $0.05
slippage backtest assumption baked into entries_v2.DEFAULT_COSTS.

The spread comparison uses ONLY trades that actually captured a live
Deriv quote (the daemon's best-effort quoted_bid/quoted_ask enrichment,
via sample_current_spread_resilient) -- never estimated or interpolated.
If Deriv's tick-subscribe endpoint is unavailable (a real, currently-
confirmed outage for the default app_id -- see data/performance_report.md
and src/data_deriv.py), this script says so plainly rather than printing
a fabricated number.

Usage:
  python3 scripts/check_telemetry.py [--state data/paper_state.json] [--json]

Reads only -- never modifies data/paper_state.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_PATH = REPO_ROOT / "data" / "paper_state.json"

# entries_v2.DEFAULT_COSTS' values, kept as literals here rather than imported --
# same rationale as spread_telemetry.py: this script's whole point is checking the
# assumption against reality, not silently inheriting whatever it currently is.
ASSUMED_SPREAD = 0.40
ASSUMED_SLIPPAGE_PER_SIDE = 0.05


def _max_drawdown_pct(equity_curve: list) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]["equity"]
    max_dd = 0.0
    for point in equity_curve:
        eq = point["equity"]
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak * 100)
    return max_dd


def build_report(state: dict) -> dict:
    trades = state.get("trades", [])
    metrics = state.get("metrics", {})

    running_days: Optional[float] = None
    started_at_str = state.get("started_at")
    if started_at_str:
        started_at = datetime.fromisoformat(started_at_str)
        running_days = (datetime.now(timezone.utc) - started_at).total_seconds() / 86400

    max_dd = _max_drawdown_pct(state.get("equity_curve", []))

    quoted = [t for t in trades if t.get("quoted_bid") is not None and t.get("quoted_ask") is not None]
    if quoted:
        observed_spreads = [t["quoted_ask"] - t["quoted_bid"] for t in quoted]
        avg_observed_spread = sum(observed_spreads) / len(observed_spreads)
        spread_delta_vs_assumed = avg_observed_spread - ASSUMED_SPREAD
    else:
        avg_observed_spread = None
        spread_delta_vs_assumed = None

    return {
        "running_days": running_days,
        "total_simulated_fills": len(trades),
        "win_rate_pct": metrics.get("win_rate_pct", 0.0),
        "profit_factor": metrics.get("pf"),
        "max_drawdown_pct": max_dd,
        "net_pnl_pct": metrics.get("net_pnl_pct", 0.0),
        "avg_observed_spread": avg_observed_spread,
        "assumed_spread": ASSUMED_SPREAD,
        "assumed_slippage_per_side": ASSUMED_SLIPPAGE_PER_SIDE,
        "spread_delta_vs_assumed": spread_delta_vs_assumed,
        "quotes_available_for": len(quoted),
        "quotes_available_out_of": len(trades),
    }


def print_report(report: dict) -> None:
    print("=== AMARO Paper Daemon Telemetry ===")
    if report["running_days"] is not None:
        print(f"Running days:           {report['running_days']:.2f}")
    else:
        print("Running days:           n/a (no started_at in state file)")
    print(f"Total simulated fills:   {report['total_simulated_fills']}")
    print()

    pf = report["profit_factor"]
    pf_str = f"{pf:.3f}" if pf is not None else "undefined (no losing trades yet)"
    print(f"Realized win rate:       {report['win_rate_pct']:.1f}%")
    print(f"Profit factor:           {pf_str}")
    print(f"Max drawdown:            {report['max_drawdown_pct']:.2f}%")
    print(f"Net P&L:                 {report['net_pnl_pct']:+.2f}%")
    print()

    n_q, n_tot = report["quotes_available_for"], report["quotes_available_out_of"]
    if report["avg_observed_spread"] is not None:
        print(f"Avg observed spread:    ${report['avg_observed_spread']:.4f}  ({n_q}/{n_tot} fills had a live quote)")
        print(f"Backtest assumption:    ${report['assumed_spread']:.2f} spread / "
              f"${report['assumed_slippage_per_side']:.2f} slippage per side")
        print(f"Delta vs. assumption:   {report['spread_delta_vs_assumed']:+.4f}")
    else:
        print(f"Avg observed spread:    no live quote data available (0/{n_tot} fills)")
        print(f"                        Deriv's tick-subscribe endpoint is currently rejecting real-time "
              f"quotes for this app_id (see data/performance_report.md / src/data_deriv.py) -- fills still "
              f"use the ${report['assumed_spread']:.2f} spread / ${report['assumed_slippage_per_side']:.2f} "
              f"slippage backtest assumption, this just can't be checked against a real observed spread yet.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only telemetry summary for run_paper_daemon.py's state file.")
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of the text report.")
    args = parser.parse_args()

    state_path = Path(args.state)
    if not state_path.exists():
        print(f"No state file found at {state_path} -- the daemon hasn't run yet, or hasn't closed a trade.",
              file=sys.stderr)
        sys.exit(1)

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    report = build_report(state)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
