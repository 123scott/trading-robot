# -*- coding: utf-8 -*-
"""
run_paper_daemon.py

Headless, restart-safe paper-trading daemon for XAUUSD_LOWFREQ v2
(flagship: regime filter + 3-bar persistence). Reuses the exact,
already-validated strategy engine (src.entries_v2.simulate) and cost
model (src.entries_v2.DEFAULT_COSTS -- spread=$0.40, slippage=$0.05,
already exactly matching this task's mandatory assumption, not
re-hardcoded here) -- this script is a production-hardening WRAPPER
around proven logic, not a second implementation of the strategy.

Price source: H1/daily candles from Deriv (src.data_deriv.
fetch_deriv_candles_async), the same path src/entries_v2_paper.py
already uses successfully -- confirmed still working in this session.
Each newly-closed trade's fill is additionally enriched with a real
live bid/ask sample (src.data_deriv.sample_current_spread) when
available, purely for telemetry (see the logged "quoted_bid"/
"quoted_ask" fields) -- the fill price and PnL themselves always come
from entries_v2.simulate()'s own cost-model math, so a temporarily
unavailable live-quote endpoint (confirmed blocked as of this session --
Deriv's tick-subscribe endpoint is currently rejecting frxXAUUSD,
verified NOT a bug in this code, see data/performance_report.md)
degrades telemetry richness only, never trading logic.

Places no real orders. No broker credentials anywhere in this file.

Production hardening this script adds on top of the proven core loop:
  - Structured logging (stdlib `logging`, daily-rotating file handler)
    to logs/paper_trader.log, instead of print().
  - A persisted state file (data/paper_state.json) -- last-seen bar
    time, running equity, and running metrics survive a process
    restart, so `kill` + relaunch resumes cleanly rather than
    re-scanning from the same fixed lookback window every time.
  - Exponential backoff with jitter on any fetch failure (network
    drop, Deriv 503, subscription rejection) -- same reconnect
    philosophy already established in src/live_monitor.py, generalized
    here to the candle-polling path.
  - SIGTERM/SIGINT handling: flushes state and logs a clean shutdown
    line before exiting, so `kill -15 <PID>` (not `kill -9`) is safe.

Usage: see the CLI block at the bottom of this file, or run
`python3 scripts/run_paper_daemon.py --help`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import random
import signal
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `python3 scripts/run_paper_daemon.py` finds the src package

from src.data_deriv import fetch_deriv_candles_async, sample_current_spread_resilient, DerivApiError
from src.entries_v2 import LowfreqV2Config, simulate, compute_metrics, DEFAULT_COSTS

LOG_DIR = REPO_ROOT / "logs"
LOG_PATH = LOG_DIR / "paper_trader.log"
STATE_PATH = REPO_ROOT / "data" / "paper_state.json"

H1_WINDOW = 500     # ~3 weeks of H1 bars -- same window entries_v2_paper.py already uses successfully
DAILY_WINDOW = 100

# The originally, most rigorously validated candidate (full 7.5yr walk-forward + a significant
# holdout) -- deliberately NOT block_adx_transition=True by default: this round's own IS/OOS
# check on 2025/2026 data found that filter roughly neutral-to-mildly-negative on this shorter,
# more recent window (see data/performance_report.md), so it isn't the safer default yet.
DEFAULT_CONFIG = LowfreqV2Config(trend_sma_period=50, pullback_ema_period=21, pullback_tolerance_pct=0.20,
                                  atr_sl_mult=2.0, atr_tp_mult=2.5, use_regime_filter=True,
                                  regime_confirm_bars=3, block_adx_transition=False)

RECONNECT_BASE_DELAY = 5.0
RECONNECT_MAX_DELAY = 300.0

log = logging.getLogger("paper_daemon")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.handlers.TimedRotatingFileHandler(
        LOG_PATH, when="midnight", backupCount=30, utc=True, encoding="utf-8")
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)  # visible under `nohup ... > logs/nohup_paper.log`
    stream_handler.setFormatter(fmt)
    log.addHandler(stream_handler)


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                state = json.load(f)
            last_seen_ms = state.get("last_seen_open_time_ms")
            last_seen_str = (datetime.fromtimestamp(last_seen_ms / 1000, tz=timezone.utc).isoformat()
                              if last_seen_ms else None)
            log.info(f"Loaded prior state: {state['metrics']['n_trades']} trades, "
                     f"last_seen_bar={last_seen_str}")
            return state
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"State file unreadable ({e}) -- starting fresh rather than crashing on a corrupt file.")
    return {"last_seen_open_time_ms": None, "trades": [], "equity_curve": [],
            "metrics": {"n_trades": 0, "win_rate_pct": 0.0, "pf": None, "net_pnl_pct": 0.0},
            "config": DEFAULT_CONFIG.as_dict(), "started_at": datetime.now(timezone.utc).isoformat()}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp_path, STATE_PATH)  # atomic on POSIX -- a crash mid-write never leaves a truncated state file


def _running_metrics(trades: list, notional: float) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate_pct": 0.0, "pf": None, "net_pnl_pct": 0.0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    return {"n_trades": n, "win_rate_pct": len(wins) / n * 100,
            "pf": (gross_profit / gross_loss) if gross_loss > 0 else None,
            "net_pnl_pct": sum(pnls) / notional * 100}


async def poll_once(symbol: str, notional: float, config: LowfreqV2Config, state: dict) -> None:
    h1 = await fetch_deriv_candles_async(symbol=symbol, interval="1h", limit=H1_WINDOW)
    daily = await fetch_deriv_candles_async(symbol=symbol, interval="1d", limit=DAILY_WINDOW)
    all_trades = simulate(h1, daily, config, notional, DEFAULT_COSTS)

    last_seen_ms = state.get("last_seen_open_time_ms")
    new_trades = [t for t in all_trades if last_seen_ms is None or t.entry_time.timestamp() * 1000 > last_seen_ms]

    if not new_trades:
        log.info("Poll complete: no newly-closed trades.")
        return

    # Best-effort live-quote enrichment for telemetry only -- never affects the fill price or
    # PnL, which already come from simulate()'s own cost-model math. Confirmed as of this
    # session that Deriv's tick-subscribe endpoint is currently rejecting frxXAUUSD -- this is
    # expected to fail right now and that's fine, it's explicitly best-effort.
    # Resilient sampler already retries transient failures and tries a dynamic-symbol
    # fallback internally, returning None (never raising) once that's exhausted -- so this
    # stays purely best-effort with no try/except needed here.
    quote = await sample_current_spread_resilient(symbol, max_attempts=2)

    for t in sorted(new_trades, key=lambda t: t.exit_time):
        state["trades"].append({
            "direction": t.direction.value, "entry_time": t.entry_time.isoformat(), "entry_price": t.entry_price,
            "stop": t.stop, "target": t.target, "exit_time": t.exit_time.isoformat(), "exit_price": t.exit_price,
            "exit_reason": t.exit_reason, "qty": t.qty, "pnl": t.pnl,
            "quoted_bid": quote["bid"] if quote else None, "quoted_ask": quote["ask"] if quote else None,
        })
        state["equity_curve"].append({"time": t.exit_time.isoformat(),
                                       "equity": notional + sum(x["pnl"] for x in state["trades"])})
        log.info(f"NEW closed trade: {t.direction.value} {t.entry_time.date()} @ {t.entry_price:.2f} -> "
                 f"{t.exit_time.date()} @ {t.exit_price:.2f} ({t.exit_reason}), pnl {t.pnl:+.2f}")

    state["last_seen_open_time_ms"] = max(t.entry_time.timestamp() * 1000 for t in new_trades)
    state["metrics"] = _running_metrics(state["trades"], notional)
    save_state(state)
    m = state["metrics"]
    pf_str = f"{m['pf']:.2f}" if m["pf"] is not None else "undef"
    log.info(f"Running: n={m['n_trades']} win%={m['win_rate_pct']:.1f} pf={pf_str} net%={m['net_pnl_pct']:+.2f}")


async def run(symbol: str, notional: float, config: LowfreqV2Config, poll_seconds: float,
              max_iterations: Optional[int] = None) -> None:
    state = load_state()
    stop_event = asyncio.Event()

    def _handle_signal(sig_name: str) -> None:
        log.info(f"Received {sig_name} -- shutting down cleanly (state already saved after each poll; "
                 f"nothing lost).")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: _handle_signal(signal.Signals(s).name))

    log.info(f"Starting paper daemon: symbol={symbol} notional={notional} poll_seconds={poll_seconds}")
    log.info(f"Config: {config.as_dict()}")
    log.info(f"Cost model: spread=${DEFAULT_COSTS.spread:.2f}, slippage=${DEFAULT_COSTS.slippage_per_side:.2f}/side "
             f"(mandatory, per task spec)")

    attempt = 0
    iteration = 0
    while not stop_event.is_set() and (max_iterations is None or iteration < max_iterations):
        iteration += 1
        try:
            await poll_once(symbol, notional, config, state)
            attempt = 0  # reset backoff after any successful poll
        except (DerivApiError, asyncio.TimeoutError, OSError) as e:
            attempt += 1
            delay = min(RECONNECT_MAX_DELAY, RECONNECT_BASE_DELAY * (2 ** (attempt - 1)))
            delay *= 0.8 + 0.4 * random.random()  # +/-20% jitter, avoids thundering-herd reconnects
            log.warning(f"Poll failed (attempt {attempt}): {type(e).__name__}: {e} -- retrying in {delay:.1f}s.")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            continue
        except Exception as e:
            log.exception(f"Unexpected error in poll loop -- state already saved as of the last successful "
                          f"poll, continuing rather than crashing the daemon: {e}")
            attempt += 1

        if stop_event.is_set() or (max_iterations is not None and iteration >= max_iterations):
            break
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
        except asyncio.TimeoutError:
            pass

    save_state(state)
    log.info("Daemon stopped. Final state saved.")


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Headless paper-trading daemon for XAUUSD_LOWFREQ v2 (no real orders).")
    parser.add_argument("--symbol", default="XAUUSD_DERIV")
    parser.add_argument("--notional", type=float, default=10_000.0)
    parser.add_argument("--poll-seconds", type=float, default=1800.0)
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop after N polls (omit to run forever).")
    args = parser.parse_args()

    setup_logging()
    try:
        asyncio.run(run(symbol=args.symbol, notional=args.notional, config=DEFAULT_CONFIG,
                        poll_seconds=args.poll_seconds, max_iterations=args.max_iterations))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    _cli()
