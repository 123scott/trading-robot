# -*- coding: utf-8 -*-
"""
spread_telemetry.py

Periodically samples a REAL bid/ask spread from Deriv's live tick feed
and logs it next to this project's illustrative cost-model assumption
($0.40 spread / $0.05 slippage, see entries_v2.CostModel /
entries_v3.CostModelV3), so that assumption can eventually be checked
against reality rather than trusted indefinitely.

Read this before running it: as of this module's creation, Deriv's live
tick-SUBSCRIBE endpoint for frxXAUUSD is rejecting the symbol with
InvalidSymbol -- confirmed NOT a bug in this code, since the identical
failure was reproduced in the already-established, previously-working
stream_deriv_ticks() function (the same one live_monitor.py's --paper
mode has used successfully throughout this project). Historical CANDLE
data (fetch_deriv_candles_async, what entries_v2_paper.py/
entries_v3_paper.py actually poll) is unaffected -- only the raw
tick-subscribe path is currently blocked. This module's logging/stats
logic is verified correct against a mocked sample (see the development
session); the live sampling call itself will simply return no data
until Deriv's tick-subscribe access is restored. Not something more
retry logic fixes -- an external service state, checked and reported
honestly rather than worked around.

No synthetic spread data is ever written here -- a failed sample logs
nothing, never a fabricated or estimated value.
"""

from __future__ import annotations

import asyncio
import csv
import os
from datetime import datetime, timezone
from typing import Optional

from src import memory
from src.data_deriv import sample_current_spread_resilient, deriv_ticker

TELEMETRY_LOG_PATH = os.path.join(memory.DATA_DIR, "spread_telemetry.csv")
TELEMETRY_HEADER = ["timestamp", "symbol", "observed_bid", "observed_ask", "observed_spread",
                     "assumed_spread", "delta_vs_assumed"]

# The illustrative assumptions actually used elsewhere in this project (entries_v2.CostModel /
# entries_v3.CostModelV3) -- kept as a literal here rather than imported, since this module's
# whole point is checking that assumption against reality, not inheriting it silently.
ASSUMED_SPREAD = 0.40


def _ensure_log() -> None:
    os.makedirs(memory.DATA_DIR, exist_ok=True)
    if not os.path.exists(TELEMETRY_LOG_PATH):
        with open(TELEMETRY_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(TELEMETRY_HEADER)


def log_sample(symbol: str, sample: dict) -> None:
    _ensure_log()
    ts = datetime.now(timezone.utc).isoformat()
    delta = sample["spread"] - ASSUMED_SPREAD
    with open(TELEMETRY_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, symbol, f"{sample['bid']:.5f}", f"{sample['ask']:.5f}",
                                 f"{sample['spread']:.5f}", f"{ASSUMED_SPREAD:.5f}", f"{delta:+.5f}"])


async def run(symbol: str = "XAUUSD_DERIV", poll_seconds: float = 300.0,
              max_iterations: Optional[int] = None, log=print) -> None:
    ticker = deriv_ticker(symbol)
    log(f"[SPREAD TELEMETRY] Sampling real {ticker} bid/ask every {poll_seconds:.0f}s, "
        f"comparing against the assumed ${ASSUMED_SPREAD:.2f} spread. Logging to {TELEMETRY_LOG_PATH}\n")
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        # sample_current_spread_resilient already retries transient failures and tries a
        # dynamic-symbol fallback on a permanent rejection internally -- a None return here
        # means all of that was exhausted this poll, not a single bare failure.
        sample = await sample_current_spread_resilient(symbol)
        if sample is not None:
            log_sample(symbol, sample)
            delta = sample["spread"] - ASSUMED_SPREAD
            log(f"[SPREAD TELEMETRY] bid={sample['bid']:.5f} ask={sample['ask']:.5f} "
                f"observed_spread=${sample['spread']:.5f} (assumed ${ASSUMED_SPREAD:.2f}, "
                f"delta {delta:+.5f})")
        else:
            log(f"[SPREAD TELEMETRY] Sample failed after retries + dynamic-symbol fallback -- "
                f"logging nothing (never fabricating a spread value), will retry next poll.")
        if max_iterations is None or iteration < max_iterations:
            await asyncio.sleep(poll_seconds)
    log("[SPREAD TELEMETRY] Stopped.")


def print_summary(notional_context: str = "") -> None:
    """Reads the accumulated log and reports observed-vs-assumed spread statistics."""
    if not os.path.exists(TELEMETRY_LOG_PATH):
        print("No telemetry samples logged yet.")
        return
    with open(TELEMETRY_LOG_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No telemetry samples logged yet.")
        return
    spreads = [float(r["observed_spread"]) for r in rows]
    n = len(spreads)
    mean_spread = sum(spreads) / n
    print(f"=== Spread telemetry summary ({n} real samples) ===")
    print(f"  Observed mean spread: ${mean_spread:.4f}")
    print(f"  Observed min/max:     ${min(spreads):.4f} / ${max(spreads):.4f}")
    print(f"  Assumed (cost model): ${ASSUMED_SPREAD:.4f}")
    print(f"  Mean delta:           {mean_spread - ASSUMED_SPREAD:+.4f}")


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Real vs. assumed Deriv spread telemetry logger.")
    parser.add_argument("--symbol", default="XAUUSD_DERIV")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--summary", action="store_true", help="Print accumulated summary and exit, no sampling.")
    args = parser.parse_args()
    if args.summary:
        print_summary()
        return
    asyncio.run(run(symbol=args.symbol, poll_seconds=args.poll_seconds, max_iterations=args.max_iterations))


if __name__ == "__main__":
    _cli()
