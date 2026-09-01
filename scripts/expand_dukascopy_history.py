# -*- coding: utf-8 -*-
"""
expand_dukascopy_history.py

Extends the XAUUSD Dukascopy M5 cache backward, from the current earliest
cached date (2018-01-01) toward `--start` (default 2012-01-01), so a
genuinely independent, never-analyzed pre-2018 fold set becomes available
for future strategy research -- see data/performance_report.md's
2026-09-01 entry: every existing historical slice (2018-2025-07 for
training, 2025-08-2026-07 reused 3x as a test window) is now used up, and
there is no way to manufacture a fresh holdout from data already in hand.

This is a THIN wrapper around the already-existing, already-hardened
fetcher in src/data_dukascopy.py (circuit breaker, resumable via
dukascopy_hours_done.csv, Retry-After-aware) -- it adds no new fetching
logic of its own, on purpose: that fetcher's circuit-breaker behavior is a
deliberate, previously-established policy decision (see its own module
docstring) to respect Dukascopy's rate limiting rather than evade it
(no proxy rotation, no header spoofing, no aggressive retry hammering).
This script does not change that.

IMPORTANT, checked directly before writing this script: Dukascopy is
CURRENTLY rate-limiting requests to this endpoint (a plain, single-request
check for 2012-01-02 returned HTTP 429 in ~0.1s, checked 2026-09-01).
Running this script right this moment will very likely trip the circuit
breaker on its first chunk and exit cleanly via CircuitBreakerTripped --
which is the correct, designed behavior, not a bug. Wait for the rate
limit to clear (hours, possibly longer -- there's no way to know from the
client side) before running this for real. It's also unconfirmed whether
Dukascopy actually serves XAUUSD tick data all the way back to 2012 --
this script's job is to find out safely (it stops on sustained failure
rather than grinding), not to assume the answer.

Given the scope (~6 years, roughly 35,000+ hours to check), a full run is
a long-lived operation even under good conditions -- expect it to take
hours to days respecting the server's pace, not minutes. Safe to
interrupt (Ctrl+C) and re-run: already-processed hours are skipped via
dukascopy_hours_done.csv, so it resumes exactly where it left off.

Usage:
    python3 scripts/expand_dukascopy_history.py
    python3 scripts/expand_dukascopy_history.py --start 2012-01-01 --end 2018-01-01
    python3 scripts/expand_dukascopy_history.py --start 2016-01-01 --end 2018-01-01  # smaller first bite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_dukascopy import fetch_and_cache_range, CircuitBreakerTripped

DEFAULT_START = "2012-01-01"
DEFAULT_END = "2018-01-01"  # the current earliest cached date -- fills the gap with no overlap, no hole


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend the XAUUSD Dukascopy M5 cache backward.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    print(f"Extending {args.symbol} cache: {args.start} -> {args.end}")
    print("Respecting the existing circuit breaker (150-hour chunks, escalating cooldown up to 30min, "
          "gives up cleanly after 3 consecutive blocked chunks). Safe to Ctrl+C and re-run.\n")
    try:
        result = fetch_and_cache_range(args.symbol, args.start, args.end, max_workers=args.max_workers, log=print)
        print(f"\nDone: {result}")
    except CircuitBreakerTripped as e:
        print(f"\nStopped by the circuit breaker (sustained blocking detected) -- this is the intended "
              f"behavior, not a crash: {e}")
        print("Whatever hours succeeded before the trip are already saved in the cache and "
              "dukascopy_hours_done.csv -- re-run this same command later to resume from there.")
        sys.exit(1)


if __name__ == "__main__":
    main()
