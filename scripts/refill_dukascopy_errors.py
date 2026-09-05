# -*- coding: utf-8 -*-
"""Clears ERROR-marked hours from the tracker, then re-fetches exactly those hours
(everything else is skipped). Same circuit breaker as the main expansion script."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_dukascopy import clear_error_hours, fetch_and_cache_range, CircuitBreakerTripped

cleared = clear_error_hours("XAUUSD")
print(f"Cleared {cleared} ERROR hours from the tracker; re-fetching those only.", flush=True)
try:
    print(fetch_and_cache_range("XAUUSD", "2012-01-01", "2018-01-01", max_workers=4, log=print))
except CircuitBreakerTripped as e:
    print(f"Circuit breaker stopped the retry: {e} -- re-run later to resume.")
    sys.exit(1)
