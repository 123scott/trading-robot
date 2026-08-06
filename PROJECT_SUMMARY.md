# AMARO Trading Bot -- Project Summary

Repo: https://github.com/123scott/trading-robot

This project contains two separate, independent systems. They share no
code and were built for different purposes -- don't confuse one for the
other.

## 1. SMC Market-Structure Engine (root-level: `structures.py`, `entries.py`)

A mechanical Smart Money Concepts / ICT-style engine for MT5 forex/gold
(XAUUSD, GBPUSD, USDJPY):
- `structures.py` -- swing/ITH-ITL detection, BOS/CHoCH classification,
  weak/strong swing labeling, three-candle FVG engine (mitigation/fill/
  inversion tracking), Premium/Discount arrays.
- `entries.py` -- risk-entry vs. confirmation-entry execution engine,
  three entry-trigger techniques (candle-close, 50%-engulfing,
  break-of-candle).

This is analysis/signal code only -- no data fetching, no execution, no
memory system. It has not been wired into the `src/` system below.

## 2. Multi-Asset Replay Backtester + Memory System (`src/`, `data/`)

A simpler, different strategy (fast/slow SMA crossover, long-only) built
specifically to test a two-file "memory" concept: before every BUY/SELL,
check whether a similar setup has lost money before, and if so, downgrade
to SKIP.

**Active model scope: XAUUSD and USDJPY only.** GBPUSD was removed from
the default/client-facing scope (no statistically significant edge found
across any test) -- its data/code aren't deleted, just no longer part of
the default reporting or the client deliverable.

### Data sources (`src/market_data.py` dispatches by symbol)
- **Binance** (`src/data_binance.py`) -- BTCUSDT, public klines API.
- **Yahoo Finance** (`src/data_yfinance.py`) -- USDJPY, XAUUSD (via `GC=F`
  gold futures proxy; GBPUSD support still exists but is out of active
  scope), full 2018-present daily history. **This is the dataset of
  record for all statistical work** (Sharpe, Sortino, Monte Carlo) --
  it's the only source with a deep enough sample.
- **Deriv** (`src/data_deriv.py`) -- `XAUUSD_DERIV` (`frxXAUUSD`), public
  WebSocket API, no auth needed. Historical depth capped at ~1 year
  (verified by direct probing -- not a bug, a real API limit). Used
  exclusively for live execution validation, not stress testing.

### Strategy + memory (`src/backtest_structures.py`, `backtest_entries.py`, `memory.py`)
- SMA(9)/SMA(21) crossover generates BUY/SELL signals; a `PositionTracker`
  makes it long-only and state-aware.
- `memory.py` reads `data/ledger.csv` (every trade + outcome) and
  `data/learnings.md` (plain-English loss warnings) before every
  BUY/SELL, and downgrades to SKIP on a match. Matches decay after
  `MEMORY_DECAY_DAYS` (365, tunable) -- without this, one old loss can
  permanently veto a price zone forever, which is a real failure mode we
  hit and fixed (see `data/performance_report.md`).

### Orchestration + CLI (`src/trading_robot.py`, `src/replay.py`)
```
python -m src.replay --raw    --symbol XAUUSD --start 2018-01-01   # no memory, baseline
python -m src.replay --memory --symbol XAUUSD --start 2018-01-01   # memory-filtered
python -m src.replay --reset                                       # clear ledger + learnings
python -m src.replay --paper  --symbol XAUUSD_DERIV --notional 100 # live forward-test, no orders
```
Position sizing is notional-based (fixed $ exposure per trade) so results
are comparable across assets with very different price scales.

### Transaction costs (`market_data.COST_PROFILES`)
Every fill pays spread + slippage (worsens the fill price) and commission
(% of notional, both sides), applied in `trading_robot.py`. Illustrative
retail-ish assumptions, not a real broker's fee schedule -- see
`data/performance_report.md` for the exact figures and why they matter
(GBPUSD raw mode flips from profitable to a net loss once costs are
applied).

### Reporting & statistics (`src/report.py`, `src/monte_carlo.py`, `src/alpha_test.py`)
```
python -m src.report --symbols XAUUSD,USDJPY        # comparison table, Sharpe/Sortino (default scope)
python -m src.monte_carlo --symbol XAUUSD --mode memory --iterations 5000
python -m src.alpha_test --symbol XAUUSD --mode memory     # edge t-test + alpha/beta vs buy-and-hold
```
Full current results: **`data/performance_report.md`** and the
client-facing dashboard (equity curves, Monte Carlo histograms, stat
tiles, evaluation/recommendations) published as an Artifact. Headline
finding: **this strategy does not currently show statistically
significant alpha vs. a simple buy-and-hold benchmark on either active
asset.** XAUUSD memory mode has a significant trade-level edge (p=0.037)
but its alpha vs. buy-and-hold is not significant (p=0.68) and its CAGR
(10.87%) is below just holding gold (14.24%) over the same period. USDJPY
raw mode has significant alpha (p=0.029) but that doesn't replicate in
the trade-level test, and memory mode makes USDJPY worse on every metric.
Read the full alpha section before treating any of this as proven edge.

### Train/test parameter search (`src/optimize.py`)
```
python -m src.optimize --symbol XAUUSD --train-end 2025-07-01
```
Grid-searches SMA fast/slow periods on training data ONLY (2018 to a cutoff
date), then validates the chosen parameters on the untouched holdout
period. Each experiment gets an isolated `ledger_symbol` tag
(`trading_robot.run_replay`'s `ledger_symbol` param) so memory systems
never cross-contaminate between experiments. Result for XAUUSD
(train 2018-2025-07, test 2025-07-present): selected SMA(7,50) over the
baseline SMA(9,21), which improved training win rate (52% vs 34.7%) and
profit factor (4.34 vs 1.87), and held up directionally out-of-sample --
**but the 1-year test window only produced 1-6 closed trades, too few to
trust statistically.** Full writeup in `data/performance_report.md`.

### Two bot profiles: XAUUSD_LOWFREQ and XAUUSD_MEDFREQ (`src/bot.py`)
```
python -m src.bot --profile lowfreq --mode memory --start 2018-01-01
python -m src.bot --profile medfreq --start 2018-01-01
```
One engine, two isolated ledger-tagged profiles (each usable directly with
`report.py`/`monte_carlo.py`/`alpha_test.py`):

- **XAUUSD_LOWFREQ** -- the daily SMA(7,50) + memory system above. This is
  the validated, working bot.
- **XAUUSD_MEDFREQ** -- Top-Down Multi-Timeframe model (`medfreq_strategy.py`):
  H4 200 EMA trend + H1 RSI(14) momentum forward-filled onto M5 (no
  lookahead -- `align_htf_to_m5`), M5 EMA(8,21) crossover trigger, M5
  ATR-based SL/TP. Real M5/H1/H4 data via `src/data_dukascopy.py`
  (Dukascopy tick feed -- Yahoo/Deriv can't supply 2018-present intraday
  history; H1/H4 are pure resamples of the cached M5 data, so no extra
  fetch cost). **Verdict: fails decisively.** Original spec produced
  ~1,123 trades/year (vs. 50-75 target) with catastrophic whipsaw losses,
  diagnosed to repeated same-direction re-entries within minutes of being
  stopped out during chop. Added confirmation+cooldown anti-whipsaw
  filters (33% trade reduction) -- helped but did not fix the core
  problem: Profit Factor 0.44, Sharpe -7.41, Monte Carlo 100% probability
  of loss across 5,000 resamples, edge t-test p<0.000001 (statistically
  certain negative edge). Root cause and concrete redesign
  recommendations (pullback entry instead of crossover, chop/volatility
  filter, SL:TP math, slower execution timeframe) in
  `data/performance_report.md` -- this needs a different entry mechanism,
  not more parameter tuning on the current one.

### Live monitoring (`src/live_monitor.py`)
```
python -m src.live_monitor --paper --symbol XAUUSD_DERIV --notional 100
```
Connects to Deriv's live tick WebSocket, aggregates into daily candles,
runs the same strategy+memory logic, logs decisions to
`data/paper_trades.csv`. **Places no real orders.** Reconnects
automatically with exponential backoff on connection drops (a real
failure observed in testing). Currently running in the background of the
development session that produced this summary.

### MT5 execution -- ZeroMQ bridge (native macOS), untested end-to-end
The `MetaTrader5` Python package has no macOS/Linux build (verified:
won't even install here), so live execution runs through a ZeroMQ socket
bridge instead: `src/mt5_zmq_bridge.py` (Python side, runs natively on
this Mac) talks JSON over a local REQ/REP socket to
`mt5_bridge_ea/AmaroZmqBridge.mq5` (an Expert Advisor running inside an
actual MT5 terminal -- MetaQuotes doesn't ship a native macOS terminal
either, so that piece still needs Windows/a VM/a broker's Mac MT5 build;
see `MT5_SETUP.md`). `src/mt5_live.py` wires the existing crossover +
memory decision logic (`detect_crossovers`/`PositionTracker`/
`memory.check_memory`, same as `replay.py`/`live_monitor.py`) to this
bridge -- on a memory-approved signal it places an ATR-based-SL/TP order
sized by risk percentage (not notional -- the backtested strategy has no
stop-loss, so this adds one specifically for real order placement; see
the "no explicit stop-loss" caveat below). Logs every decision to
`data/mt5_live_trades.csv`, kept isolated from `data/ledger.csv` the
same way `live_monitor.py`'s paper trades are.

Safety is enforced in two independent places: the EA refuses to even
initialize on a non-demo account (a real-money account can never load
it), and `mt5_zmq_bridge.connect()` checks again as defense in depth --
no `--allow-live` flag exists anywhere in this path. Real side benefit:
no MT5 credentials ever pass through Python or a `.env` file at all --
you log into the terminal manually via its own GUI. The Python↔EA wire
protocol (every command, plus timeout/reconnect handling) is verified
end-to-end against a dummy responder; the EA's own MQL5 trade calls are
reviewed against MQL5's documented API but unverified against a real
terminal (none available in this dev environment) -- validate on a demo
account before trusting it. An older Windows-native path
(`src/mt5_executor.py`, direct `MetaTrader5` package calls) remains in
the repo, unused by default, as a reference for anyone who does end up
on native Windows.

## Known Limitations / Honest Caveats

0. **No statistically significant alpha vs. buy-and-hold on any asset yet**
   -- see the alpha-test section above. This is not a proven-edge system.
1. **USDJPY memory mode underperforms raw mode** at every tested decay
   window -- don't trust it live without further work.
2. **Every backtest's "Max Drawdown" is realized-PnL only** -- it doesn't
   account for intra-trade floating drawdown, and several runs end with
   an open position whose unrealized PnL is reported separately (see
   "Unrealized Open-Position Marks" in the performance report).
3. **Deriv's XAUUSD_DERIV backtest sample is small** (~1 year of data,
   capped by the API itself) -- treat it as an execution/spread sanity
   check, not a statistical result.
4. **`mt5_executor.py` is unverified** -- written carefully, never run.
5. **The two systems (SMC engine vs. replay backtester) are unconnected.**
   Combining them (e.g. running the SMC entry logic through the memory
   filter) would be a substantial follow-on project, not something
   already wired up.

## Setup

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m src.replay --raw --symbol XAUUSD --start 2018-01-01
```
