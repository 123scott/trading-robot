# Pre-Retrain Snapshot -- XAUUSD_LOWFREQ (before picture)

Written before any modification, per STEP 0. Covers current parameters,
current entry/exit logic in plain English, current memory/cost
behavior, and one factual correction to the task brief.

## Correction to file paths in the task brief

The task brief says "src/ contains the strategy logic, entries.py has
entry rules, structures.py has data structures." That's not accurate for
this repo: `entries.py` and `structures.py` are **root-level** files
belonging to a completely separate, never-connected Smart Money Concepts
(SMC) engine (BOS/CHoCH, FVG, PD arrays) -- there is no `src/entries.py`
or `src/structures.py`, and the SMC engine has no SMA, no memory system,
and has never been run as part of any bot. The actual XAUUSD_LOWFREQ
logic the task describes ("daily SMA crossover + memory system") lives
in:
- `src/backtest_structures.py` -- the SMA crossover signal detector
- `src/backtest_entries.py` -- the long-only position state machine
- `src/trading_robot.py` -- orchestration, costs, memory integration
- `src/bot.py` -- the `run_lowfreq()` entry point and parameter constants
- `src/memory.py` -- the two-file memory system

This snapshot, and all subsequent work in this round, targets those
files. `entries_v2.py` will be a new module in `src/` (not root) that
composes with these, not a modification of the root SMC files.

## Current parameters and their values

| Parameter | Value | Where |
|---|---:|---|
| Fast SMA period | 7 | `src/bot.py: LOWFREQ_FAST` |
| Slow SMA period | 50 | `src/bot.py: LOWFREQ_SLOW` |
| Candle interval | 1d (daily) | `market_data.default_interval_for` |
| Position sizing | notional / fill_price (fixed $ exposure) | `trading_robot.run_replay` |
| Default notional | $10,000 | `bot.run_lowfreq` default |
| Memory price-match tolerance | +/-1.0% | `memory.PRICE_TOLERANCE_PCT` |
| Memory decay window | 365 days | `memory.MEMORY_DECAY_DAYS` |
| Consecutive-skip alert threshold | 10 | `memory.CONSECUTIVE_SKIP_ALERT_THRESHOLD` |
| XAUUSD spread (round-trip) | $0.35 | `market_data.COST_PROFILES["XAUUSD"]` |
| XAUUSD slippage | 0.01% of price | `market_data.COST_PROFILES["XAUUSD"]` |
| XAUUSD commission | 0.010% of notional/side | `market_data.COST_PROFILES["XAUUSD"]` |
| XAUUSD_DERIV spread | $0.30 (`DEFAULT_SPREAD`) | `data_deriv.py` |

The 7/50 values were **not chosen for this snapshot** -- they came from
an earlier `src/optimize.py` grid search (35 combinations, trained on
2018-01-01 to 2025-07-01, scored by training-Sharpe with a 15-trade
minimum) under separate ledger tags `XAUUSD_BASE`/`XAUUSD_OPT`, not
under `XAUUSD_LOWFREQ` itself.

**Note on the SMA crossover itself:** `detect_crossovers` takes
`fast_period`/`slow_period` as its only tunable inputs -- there is no
stop-loss, no take-profit, and no ATR or volatility input anywhere in
the current LOWFREQ path. Exits are purely the opposite crossover
signal. That means today's "parameter count" is effectively 2 (fast,
slow) plus the 2 memory parameters (decay window, price tolerance) --
nowhere near a stop/target/lookback set, which is why this round's
5-parameter budget has room for a real entry-quality addition.

## Current entry/exit logic, in plain English

1. Fetch daily XAUUSD candles (Yahoo Finance `GC=F` proxy).
2. Compute a 7-day and a 50-day simple moving average of the close.
3. Whenever the 7-day average crosses **above** the 50-day average
   (a "golden cross") **while flat**, that's a BUY signal.
4. Whenever the 7-day average crosses **below** the 50-day average
   (a "death cross") **while holding a long position**, that's a SELL
   (close) signal. There is no short-selling anywhere in this path --
   position state is strictly FLAT or LONG.
5. Before every BUY/SELL is actually executed, the two-file memory
   system checks: has a crossover in roughly this direction, within
   +/-1% of this price, lost money within the last 365 days (per
   `data/ledger.csv` and `data/learnings.md`)? If yes, the action is
   downgraded to SKIP -- the signal is logged but no trade happens and
   position state doesn't change.
6. Every executed fill pays cost: BUY fills above the signal price,
   SELL fills below it (spread/2 + slippage), plus commission on both
   sides. All of it comes out of the trade's recorded PnL.
7. Every realized LOSS on a SELL writes a new `learnings.md` warning
   for that price zone/direction, feeding step 5 on future signals.

**Trade frequency, measured directly from what's actually in the
ledger:** across the 218 `XAUUSD` raw-mode rows in `data/ledger.csv`
(2018-present), 55 were realized round-trip trades -- roughly **6-7
trades/year, ~0.13/week**, an order of magnitude below the 3-4/week
target. This confirms the task brief's own framing: a daily-only SMA
crossover cannot reach that frequency by parameter tweaking alone, an
intraday entry layer is structurally required.

## Data coverage check (before touching anything else)

`data/dukascopy_m5_cache.csv` (608,791 M5 bars) checked directly, not
assumed: **100.0000% of expected forex market hours present for
2018-01-01 through 2026-07-31** (verified by comparing against every
expected Mon-Fri market hour, Friday 22:00 UTC - Sunday 22:00 UTC
excluded). The only zero-bar weekdays in range are genuine market
holidays (Good Friday, Christmas, New Year's Day) -- not gaps. No
Dukascopy fetch was needed this round.

## One more thing worth knowing before retraining

`XAUUSD_LOWFREQ` -- the exact ledger tag `src/bot.py`'s `run_lowfreq()`
writes to and reads memory from -- **has zero rows in `data/ledger.csv`
right now.** It's never been run under its own tag; all prior LOWFREQ-
adjacent history lives under `XAUUSD`/`XAUUSD_BASE`/`XAUUSD_OPT` instead,
which the memory system (exact-symbol match) does NOT share with
`XAUUSD_LOWFREQ`. This round's walk-forward/test work uses isolated,
purpose-tagged ledger symbols of its own (see performance_report.md),
so this doesn't block anything -- just flagging it as a pre-existing
gap between the documented profile name and what's actually been run.
