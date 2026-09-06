# -*- coding: utf-8 -*-
"""Clears ERROR-marked hours from a symbol's tracker, then re-fetches exactly those
hours (everything already fetched is skipped). Same circuit breaker as the main
expansion script. ERROR rows are transient failures (rate-limit exhaustion, hanging
files); genuinely market-closed hours are recorded as "0" and are left alone.

Usage:
    python3 scripts/refill_dukascopy_errors.py                      # XAUUSD, 2012-2018
    python3 scripts/refill_dukascopy_errors.py --symbol EURUSD
    python3 scripts/refill_dukascopy_errors.py --symbol GBPUSD --start 2012-01-01 --end 2018-01-01
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_dukascopy import clear_error_hours, fetch_and_cache_range, CircuitBreakerTripped

parser = argparse.ArgumentParser(description="Retry only the ERROR-marked hours for a symbol.")
parser.add_argument("--symbol", default="XAUUSD")
parser.add_argument("--start", default="2012-01-01")
parser.add_argument("--end", default="2018-01-01")
parser.add_argument("--max-workers", type=int, default=4)
args = parser.parse_args()

cleared = clear_error_hours(args.symbol)
print(f"Cleared {cleared} ERROR hours from {args.symbol}'s tracker; re-fetching those only.", flush=True)
try:
    print(fetch_and_cache_range(args.symbol, args.start, args.end,
                                 max_workers=args.max_workers, log=print))
except CircuitBreakerTripped as e:
    print(f"Circuit breaker stopped the retry: {e} -- re-run later to resume.")
    sys.exit(1)
