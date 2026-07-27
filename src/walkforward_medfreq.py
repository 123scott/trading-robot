# -*- coding: utf-8 -*-
"""
walkforward_medfreq.py

Validation runner for the XAUUSD_MEDFREQ bot (src/medfreq_strategy.py).

The strategy's parameters (EMA 8/21/200, RSI 14 with 40-65/35-60 bands,
ATR 14 with 1.5x SL / 2.75x TP) are fully specified, not fit to data --
so there is no parameter search to protect against overfitting here (see
src/optimize.py for that, used by the lowfreq bot). What "walk-forward"
means for a fixed-rule strategy like this is: run it continuously across
the full real history (2018-present, real Dukascopy-derived H1 candles)
so it's tested across genuinely different market regimes (2018 range,
2020 COVID crash, 2022 inflation, 2024-2025 rally), then report:
  - year-by-year performance (regime breakdown, nothing averaged away)
  - a development-period vs. out-of-sample split at 2025-07-01, matching
    the split boundary used for the lowfreq bot's validation, so the two
    bots' results are comparable

No synthetic data: candles come from src.data_dukascopy's real,
Yahoo-cross-validated cache. If the cache doesn't yet cover the full
requested range, this reports exactly how much real data it found and
proceeds on that -- it does not fill gaps.

Usage:
    python -m src.walkforward_medfreq --start 2018-01-01 --split 2025-07-01
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from typing import List

from src import data_dukascopy
from src.medfreq_strategy import MedFreqConfig, simulate, compute_metrics, TradeRecord

SYMBOL = "XAUUSD"


def run(start: str, split: str, notional: float = 10_000.0, config: MedFreqConfig = None, log=print) -> dict:
    config = config or MedFreqConfig()
    candles = data_dukascopy.load_h1_candles(start)
    if not candles:
        raise RuntimeError(f"No cached Dukascopy H1 candles found from {start}. Run src.data_dukascopy first.")

    first = datetime.fromtimestamp(candles[0].open_time / 1000, tz=timezone.utc)
    last = datetime.fromtimestamp(candles[-1].open_time / 1000, tz=timezone.utc)
    log(f"Loaded {len(candles)} real H1 candles ({first.date()} -> {last.date()}).")
    log(f"Config: EMA({config.ema_fast},{config.ema_slow},trend={config.ema_trend}), "
        f"RSI({config.rsi_period}) long={config.rsi_long} short={config.rsi_short}, "
        f"ATR({config.atr_period}) SL={config.atr_sl_mult}x TP={config.atr_tp_mult}x\n")

    trades = simulate(candles, config, SYMBOL, notional)
    log(f"Simulated {len(trades)} total trades over the full real history.\n")

    split_dt = datetime.fromisoformat(split).replace(tzinfo=timezone.utc)
    dev_trades = [t for t in trades if t.exit_time < split_dt]
    oos_trades = [t for t in trades if t.exit_time >= split_dt]

    by_year: dict = defaultdict(list)
    for t in trades:
        by_year[t.exit_time.year].append(t)

    return {
        "candles": len(candles), "first": first, "last": last,
        "all_trades": trades, "dev_trades": dev_trades, "oos_trades": oos_trades,
        "by_year": dict(sorted(by_year.items())),
        "all_metrics": compute_metrics(trades, notional),
        "dev_metrics": compute_metrics(dev_trades, notional),
        "oos_metrics": compute_metrics(oos_trades, notional),
        "split": split_dt,
    }


def _fmt_metrics_row(label: str, m: dict) -> str:
    pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "undef"
    sh = f"{m['sharpe']:.2f}" if m["sharpe"] is not None else "--"
    tpy = f"{m['trades_per_year']:.1f}" if m["trades_per_year"] is not None else "--"
    return (f"{label:14} {m['n_trades']:7} {tpy:>9} {m['win_rate_pct']:7.1f} {pf:>8} "
            f"{m['max_drawdown_pct']:8.2f} {sh:>8} {m['net_pnl_pct']:10.2f}")


def print_report(result: dict) -> None:
    header = f"{'Period':14} {'Trades':7} {'Trd/Yr':>9} {'Win%':>7} {'PF':>8} {'MaxDD%':>8} {'Sharpe':>8} {'NetPnL%':>10}"
    print("\n" + "=" * len(header))
    print("MEDFREQ (XAUUSD, H1) -- FULL HISTORY + DEV/OOS SPLIT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    print(_fmt_metrics_row("Full history", result["all_metrics"]))
    print(_fmt_metrics_row(f"Dev (<{result['split'].date()})", result["dev_metrics"]))
    print(_fmt_metrics_row(f"OOS (>={result['split'].date()})", result["oos_metrics"]))

    print("\nYear-by-year (regime breakdown):")
    print(header)
    print("-" * len(header))
    for year, yr_trades in result["by_year"].items():
        m = compute_metrics(yr_trades, 10_000.0)
        print(_fmt_metrics_row(str(year), m))

    oos = result["oos_metrics"]
    print("\n--- Target check (out-of-sample) ---")
    tpy = oos["trades_per_year"] or 0
    pf = oos["profit_factor"] or 0
    sh = oos["sharpe"] or 0
    print(f"Trades/year: {tpy:.1f}  (target 50-75)  {'PASS' if 50 <= tpy <= 75 else 'MISS'}")
    print(f"Profit Factor: {pf:.2f}  (target >1.5)   {'PASS' if pf > 1.5 else 'MISS'}")
    print(f"Sharpe: {sh:.2f}  (target >0.8)          {'PASS' if sh > 0.8 else 'MISS'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation for the XAUUSD_MEDFREQ bot.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--split", default="2025-07-01")
    parser.add_argument("--notional", type=float, default=10_000.0)
    args = parser.parse_args()
    result = run(args.start, args.split, args.notional)
    print_report(result)


if __name__ == "__main__":
    main()
