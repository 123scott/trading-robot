# -*- coding: utf-8 -*-
"""
fetch_fx_2012_2017.py

Fetches EURUSD then GBPUSD M5 history for 2012-01-01..2018-01-01 into their
own caches (data/dukascopy_m5_cache_EURUSD.csv etc. -- the symbol-suffixed
paths data_dukascopy already supports; XAUUSD's unsuffixed files are never
touched).

Run SEQUENTIALLY, not in parallel, deliberately: Dukascopy was responding in
~11-15s per request when this was written, which already indicates heavy
throttling. Two concurrent symbol fetches would double the request rate
against a server that is visibly straining, and this project's standing
policy is to respect that rather than push through it (same reasoning as the
circuit breaker itself). max_workers stays at the established 4.

Known and expected: some individual hourly files hang rather than 404 --
verified directly, e.g. GBPUSD 2014/05/16/09h times out reproducibly while
2016/02/10/14h returns 75KB reliably. Those become ERROR rows and can be
retried afterwards with scripts/refill_dukascopy_errors.py (which takes a
--symbol argument), exactly as was done for XAUUSD.

Expect this to run for a long time -- on the order of a day or more per
symbol at the latency observed at launch. Safe to interrupt and re-run;
already-fetched hours are skipped.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_dukascopy import fetch_and_cache_range, CircuitBreakerTripped

START, END = "2012-01-01", "2018-01-01"


def main() -> None:
    for symbol in ["EURUSD", "GBPUSD"]:
        print(f"\n{'='*70}\n=== {symbol}: {START} -> {END} ===\n{'='*70}", flush=True)
        t0 = time.time()
        try:
            result = fetch_and_cache_range(symbol, START, END, max_workers=4, log=print)
            print(f"{symbol} done in {(time.time()-t0)/3600:.1f}h: {result}", flush=True)
        except CircuitBreakerTripped as e:
            # Don't abandon the second symbol because the first hit a sustained block --
            # report it and move on; both are resumable.
            print(f"{symbol} stopped by circuit breaker after {(time.time()-t0)/3600:.1f}h: {e}", flush=True)
        except Exception as e:
            print(f"{symbol} failed with {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
