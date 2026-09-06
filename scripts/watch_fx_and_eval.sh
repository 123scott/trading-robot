#!/usr/bin/env bash
#
# watch_fx_and_eval.sh -- waits for scripts/fetch_fx_2012_2017.py to finish each
# symbol, then runs the matching entries_v4_session_ob evaluation automatically:
#
#   EURUSD terminal  -> two-instrument interim eval (XAUUSD + EURUSD)
#   GBPUSD terminal  -> full three-instrument eval
#
# Emits ONE compact line per milestone on stdout (each becomes a notification);
# full evaluation output goes to logs/eval_2instrument.log and
# logs/eval_3instrument.log for reading afterwards.
#
# "Terminal" deliberately means done OR circuit-breaker-stopped OR failed, and
# the fetch process dying is also treated as terminal -- a watcher that only
# matched the success line would sit silent through a crash, and silence would
# be indistinguishable from "still fetching".

set -uo pipefail

REPO="/Users/admin/Desktop - admin’s MacBook Pro/AMARO"
cd "$REPO" || exit 1

FETCH_LOG="logs/fx_fetch.log"
PY="venv/bin/python3"

# Terminal lines written by fetch_fx_2012_2017.py's own handlers, plus the
# process-gone case.
terminal_for() {
  grep -aE "^$1 (done|stopped by circuit breaker|failed)" "$FETCH_LOG" 2>/dev/null | tail -1
}

fetch_alive() {
  pgrep -f "fetch_fx_2012_2017.py" >/dev/null 2>&1
}

# Reports how many 2012-2017 M5 bars a symbol actually has, so a "completed"
# fetch that produced little usable data in the target window is visible rather
# than silently turning into a skipped arm inside the evaluation.
coverage() {
  "$PY" - "$1" <<'PY' 2>/dev/null || echo "unknown"
import csv, sys, os
sym = sys.argv[1]
path = f"data/dukascopy_m5_cache_{sym}.csv"
if not os.path.exists(path):
    print("0"); raise SystemExit
n = 0
with open(path) as f:
    r = csv.reader(f); next(r, None)
    for row in r:
        if "2012" <= row[0][:4] <= "2017":
            n += 1
print(n)
PY
}

wait_for() {
  local sym="$1"
  while true; do
    local line
    line="$(terminal_for "$sym")"
    if [ -n "$line" ]; then
      echo "$sym fetch terminal: $line"
      return 0
    fi
    if ! fetch_alive; then
      echo "$sym: fetch process is GONE before any terminal line appeared -- check logs/fx_fetch.log"
      return 1
    fi
    sleep 60
  done
}

# ---------------------------------------------------------------- EURUSD
wait_for EURUSD
echo "EURUSD 2012-2017 M5 bars cached: $(coverage EURUSD)"
echo "Running interim two-instrument evaluation (XAUUSD + EURUSD)..."
"$PY" -u -m src.eval_v4_multi_asset --symbols XAUUSD EURUSD > logs/eval_2instrument.log 2>&1
echo "INTERIM EVAL COMPLETE (XAUUSD+EURUSD) -> logs/eval_2instrument.log"

# ---------------------------------------------------------------- GBPUSD
wait_for GBPUSD
echo "GBPUSD 2012-2017 M5 bars cached: $(coverage GBPUSD)"
echo "Running full three-instrument evaluation..."
"$PY" -u -m src.eval_v4_multi_asset --symbols XAUUSD EURUSD GBPUSD > logs/eval_3instrument.log 2>&1
echo "FULL EVAL COMPLETE (XAUUSD+EURUSD+GBPUSD) -> logs/eval_3instrument.log"
