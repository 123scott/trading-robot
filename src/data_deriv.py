# -*- coding: utf-8 -*-
"""
data_deriv.py

Real historical and live market data for Deriv's XAUUSD (Gold) instrument,
via Deriv's public WebSocket API (no account/API token required for market
data). No generated or fixture data is used anywhere in this module.

Deriv symbol: "frxXAUUSD" (their forex-style Gold-vs-USD quote).

IMPORTANT, VERIFIED DATA LIMITATION: Deriv's `ticks_history` endpoint for
frxXAUUSD daily candles only serves roughly the trailing ~1 year of
history -- requesting an earlier `start` (tested 2018, 2020, 2023, 2024)
silently returns the same ~258 most-recent daily candles regardless. This
is a real constraint of the public feed, not a bug in this module and not
something this module works around with synthetic data. Callers wanting
the full 2018-present range should use src/data_yfinance.py (the XAUUSD
symbol backed by GC=F) instead; this module is for genuine Deriv-sourced
comparison over the window Deriv actually provides.

Spread handling: the history/ticks endpoints return a single quote price
(no per-candle bid/ask), so this module can't read Deriv's *actual*
historical spread. `DEFAULT_SPREAD` is an explicit, documented
approximation (not a live-fetched figure) applied at the execution layer
(trading_robot.py) so round-trip trades pay a realistic spread cost:
buys fill at quote + spread/2, sells fill at quote - spread/2.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import ssl
from typing import AsyncIterator, Callable, List, Optional

import certifi
import websockets

from src.candle import Candle

# Overridable via env var so a user who registers their OWN Deriv app_id (the
# real fix for the tick-subscription entitlement issue documented below) can
# switch to it without touching code: DERIV_APP_ID=123456 python3 ...
DERIV_APP_ID = os.environ.get("DERIV_APP_ID", "1089")
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
DERIV_SYMBOLS = {"XAUUSD_DERIV": "frxXAUUSD"}
# Keyword matched against active_symbols' display_name for dynamic resolution
# (see resolve_symbol_dynamic) -- kept separate from the static ticker above
# so a rename on Deriv's side doesn't require a code change to recover from.
_SYMBOL_DISPLAY_HINTS = {"XAUUSD_DERIV": "gold"}
DEFAULT_SPREAD = 0.30  # approximate USD spread for gold; see module docstring

# Error codes Deriv returns that mean "retrying THIS exact request will never
# succeed" (wrong/unavailable symbol) as opposed to a transient network/server
# hiccup that a reconnect can plausibly recover from.
_PERMANENT_ERROR_CODES = {"InvalidSymbol"}

RECONNECT_BASE_DELAY = 3.0
RECONNECT_MAX_DELAY = 120.0
MAX_RECONNECT_ATTEMPTS = 20  # bounded -- see stream_deriv_ticks_resilient docstring

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def deriv_ticker(symbol: str) -> str:
    if symbol not in DERIV_SYMBOLS:
        raise ValueError(f"Unsupported Deriv symbol: {symbol}. Supported: {list(DERIV_SYMBOLS)}")
    return DERIV_SYMBOLS[symbol]


async def fetch_candles_raw_async(ticker: str, granularity: int, start: Optional[int],
                                   end: str, count: int) -> List[dict]:
    async with websockets.connect(DERIV_WS_URL, ssl=_SSL_CTX, open_timeout=15) as ws:
        req = {
            "ticks_history": ticker,
            "adjust_start_time": 1,
            "count": count,
            "end": end,
            "start": start or 1,
            "style": "candles",
            "granularity": granularity,
        }
        await ws.send(json.dumps(req))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
        if "error" in resp:
            raise RuntimeError(f"Deriv API error for {ticker}: {resp['error']}")
        return resp.get("candles", [])


def fetch_deriv_candles(symbol: str = "XAUUSD_DERIV", interval: str = "1d",
                         start: Optional[str] = None, end: Optional[str] = None,
                         limit: int = 5000) -> List[Candle]:
    """
    Pull real historical candles for `symbol` from Deriv. `interval` accepts
    "1d"/"1h"/"1m" (mapped to Deriv's granularity in seconds). `start`/`end`
    are accepted for API-shape compatibility with the other data sources,
    but per the module-level note, Deriv currently only honors the trailing
    ~1 year window for this symbol regardless of what's requested here.
    """
    import datetime as _dt

    granularity = {"1m": 60, "1h": 3600, "1d": 86400}.get(interval, 86400)
    ticker = deriv_ticker(symbol)

    start_epoch = None
    if start:
        start_epoch = int(_dt.datetime.fromisoformat(start).replace(tzinfo=_dt.timezone.utc).timestamp())
    end_param = "latest"
    if end:
        end_param = str(int(_dt.datetime.fromisoformat(end).replace(tzinfo=_dt.timezone.utc).timestamp()))

    rows = asyncio.run(fetch_candles_raw_async(ticker, granularity, start_epoch, end_param, limit))
    return _rows_to_candles(rows)


def _rows_to_candles(rows: List[dict]) -> List[Candle]:
    return [
        Candle(
            open_time=int(row["epoch"]) * 1000,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=0.0,  # Deriv candles don't carry volume
        )
        for row in rows
    ]


async def fetch_deriv_candles_async(symbol: str = "XAUUSD_DERIV", interval: str = "1d",
                                     limit: int = 60) -> List[Candle]:
    """Async variant of fetch_deriv_candles, safe to call from within a running event loop (e.g. live_monitor.py)."""
    granularity = {"1m": 60, "1h": 3600, "1d": 86400}.get(interval, 86400)
    ticker = deriv_ticker(symbol)
    rows = await fetch_candles_raw_async(ticker, granularity, None, "latest", limit)
    return _rows_to_candles(rows)


class DerivApiError(RuntimeError):
    """Raised when Deriv's API responds to a request with an {"error": ...} payload."""


class DerivPermanentError(DerivApiError):
    """
    A DerivApiError whose error code (see _PERMANENT_ERROR_CODES) means retrying
    the identical request will never succeed -- e.g. InvalidSymbol. Callers use
    this to distinguish "reconnect and try again" from "stop, this needs a
    different symbol or a different app_id" (still a DerivApiError, so existing
    `except DerivApiError` call sites keep working unchanged).
    """


def _classify_error(code: Optional[str], message: str, context: str) -> "DerivApiError":
    text = f"Deriv rejected {context}: {code}: {message}"
    if code in _PERMANENT_ERROR_CODES:
        return DerivPermanentError(text)
    return DerivApiError(text)


_active_symbols_cache: Optional[List[dict]] = None


async def _fetch_active_symbols(timeout: float = 10.0) -> List[dict]:
    """
    Queries Deriv's active_symbols list once per process (cached) -- used by
    resolve_symbol_dynamic below. NOTE: as of this writing, this call returns
    an EMPTY list under the default public app_id (1089), for every product
    type tried -- confirmed live, not assumed. That's consistent with the
    account/app itself having lost real-time-data entitlement (ticks_history
    for historical candles still works fine on the same app_id; active_symbols
    and tick-subscribe do not), not a bug in this function. It still runs the
    real query every time (no hardcoded stand-in list) so it self-heals the
    moment entitlement is restored -- e.g. after DERIV_APP_ID is set to a
    self-registered app_id.
    """
    global _active_symbols_cache
    if _active_symbols_cache is not None:
        return _active_symbols_cache
    try:
        async with websockets.connect(DERIV_WS_URL, ssl=_SSL_CTX, open_timeout=15) as ws:
            await ws.send(json.dumps({"active_symbols": "brief", "product_type": "basic"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if "error" in resp:
            return []
        _active_symbols_cache = resp.get("active_symbols", [])
        return _active_symbols_cache
    except (asyncio.TimeoutError, OSError, websockets.exceptions.WebSocketException):
        return []  # best-effort; caller falls back to the static DERIV_SYMBOLS mapping


async def resolve_symbol_dynamic(symbol: str) -> str:
    """
    Looks up the CURRENT real Deriv ticker for `symbol` (a short name from
    DERIV_SYMBOLS, e.g. "XAUUSD_DERIV") by matching _SYMBOL_DISPLAY_HINTS
    against active_symbols' live display_name field, instead of trusting the
    static DERIV_SYMBOLS mapping could go stale if Deriv renames/relists an
    instrument. Falls back to the static mapping (deriv_ticker(symbol)) if
    active_symbols is unavailable/empty (the current, confirmed state -- see
    _fetch_active_symbols) or has no match. Never raises -- a failed dynamic
    lookup is not itself an error, just a fallback to the known-good default.
    """
    static_ticker = deriv_ticker(symbol)
    hint = _SYMBOL_DISPLAY_HINTS.get(symbol)
    if hint is None:
        return static_ticker
    symbols = await _fetch_active_symbols()
    for s in symbols:
        if hint.lower() in s.get("display_name", "").lower():
            return s["symbol"]
    return static_ticker


async def sample_current_spread(ticker: str, timeout: float = 10.0) -> dict:
    """
    Opens a short-lived subscription, takes exactly the first real tick's bid/ask, then
    unsubscribes and closes -- for telemetry (comparing a real observed spread against the
    project's illustrative cost-model assumption), not for continuous streaming (that's
    stream_deriv_ticks). Returns {"bid", "ask", "spread", "epoch"}. Raises DerivApiError on
    an API-level rejection, or asyncio.TimeoutError if no tick arrives within `timeout`.
    """
    async with websockets.connect(DERIV_WS_URL, ssl=_SSL_CTX, open_timeout=15) as ws:
        await ws.send(json.dumps({"ticks": ticker, "subscribe": 1}))
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        msg = json.loads(raw)
        if "error" in msg:
            raise _classify_error(msg["error"].get("code"), msg["error"].get("message", ""),
                                   f"tick subscription for {ticker!r}")
        if msg.get("msg_type") != "tick" or "tick" not in msg:
            raise DerivApiError(f"Unexpected response sampling {ticker!r}: {msg}")
        tick = msg["tick"]
        try:
            await ws.send(json.dumps({"forget_all": "ticks"}))
        except Exception:
            pass  # best-effort unsubscribe -- the `async with` close() below tears the connection down regardless
        bid, ask = float(tick["bid"]), float(tick["ask"])
        return {"bid": bid, "ask": ask, "spread": ask - bid, "epoch": int(tick["epoch"])}


async def stream_deriv_ticks(ticker: str, on_tick: Callable[[dict], None],
                              stop_event: Optional[asyncio.Event] = None) -> None:
    """
    Subscribes to Deriv's live tick stream for the raw Deriv `ticker`
    (e.g. "frxXAUUSD" -- use deriv_ticker(symbol) to resolve from the
    short symbol name first) and calls `on_tick` with each raw tick dict
    ({"quote": ..., "bid": ..., "ask": ..., "epoch": ...}) as it arrives.
    Runs until `stop_event` is set (or the connection drops). Used by the
    --paper forward-testing mode; this function only reads the live feed,
    it never places orders.

    Raises DerivApiError immediately if Deriv rejects the subscription
    (e.g. InvalidSymbol, or the app_id losing streaming authorization --
    a real failure mode observed in production: the previous version of
    this function only checked for msg_type=="tick", so an {"error":...}
    response with msg_type=="tick" and no "tick" key was silently
    ignored -- the loop just sat there forever waiting for a tick that
    would never come, and the caller's only symptom was the connection
    eventually timing out and reconnecting into the exact same silent
    failure, forever. Surface it instead so the caller (and its
    reconnect-with-backoff logic) can actually see what's wrong.
    """
    async with websockets.connect(DERIV_WS_URL, ssl=_SSL_CTX, open_timeout=15) as ws:
        await ws.send(json.dumps({"ticks": ticker, "subscribe": 1}))
        first = json.loads(await ws.recv())
        if "error" in first:
            raise _classify_error(first["error"].get("code"), first["error"].get("message", ""),
                                   f"tick subscription for {ticker!r}")
        if first.get("msg_type") == "tick" and "tick" in first:
            on_tick(first["tick"])
        while stop_event is None or not stop_event.is_set():
            raw = await ws.recv()
            msg = json.loads(raw)
            if "error" in msg:
                raise _classify_error(msg["error"].get("code"), msg["error"].get("message", ""),
                                       f"mid-stream for {ticker!r}")
            if msg.get("msg_type") == "tick" and "tick" in msg:
                on_tick(msg["tick"])


async def stream_deriv_ticks_resilient(symbol: str, on_tick: Callable[[dict], None],
                                        stop_event: Optional[asyncio.Event] = None,
                                        max_attempts: int = MAX_RECONNECT_ATTEMPTS) -> None:
    """
    Resilience wrapper around stream_deriv_ticks with three concrete additions
    over the raw function: (1) dynamic symbol fallback -- on a permanent
    rejection (e.g. InvalidSymbol) it re-resolves the ticker via
    resolve_symbol_dynamic and retries ONCE with whatever that returns;
    (2) a bounded reconnect-with-backoff loop for transient errors (dropped
    socket, timeout, temporary server error), instead of live_monitor.py's
    previous unbounded retry, which would spin forever with no way to
    distinguish "will recover on its own" from "permanently broken"; (3) after
    `max_attempts` transient failures, or after the dynamic-fallback retry ALSO
    fails permanently, raises a clear final error instead of looping silently.

    Honest limitation, confirmed live in this session: under the current
    default app_id (1089), active_symbols returns zero symbols and every
    tick-subscribe attempt (across multiple real symbols tested, not just
    frxXAUUSD) is rejected with InvalidSymbol, while historical candle
    fetching on the SAME app_id still works. That pattern indicates the
    app_id has lost real-time-data entitlement, not that the symbol string is
    wrong -- no dynamic symbol fallback or retry count can restore access
    that the server isn't granting. This wrapper is the correct, complete fix
    for genuine transient drops and for a stale/renamed symbol string; the
    documented recovery step for the entitlement issue itself is registering
    your own app_id at https://developers.deriv.com and setting the
    DERIV_APP_ID environment variable -- at which point this code path
    starts working with zero further changes.
    """
    ticker = deriv_ticker(symbol)
    attempt = 0
    tried_fallback = False
    while stop_event is None or not stop_event.is_set():
        try:
            await stream_deriv_ticks(ticker, on_tick, stop_event)
            return  # only returns normally once stop_event is set
        except asyncio.CancelledError:
            raise
        except DerivPermanentError as e:
            if not tried_fallback:
                tried_fallback = True
                fallback = await resolve_symbol_dynamic(symbol)
                if fallback and fallback != ticker:
                    ticker = fallback
                    continue
            raise DerivPermanentError(
                f"{e} -- dynamic symbol fallback exhausted (tried {ticker!r}), this is not "
                f"recoverable by retrying or resubscribing; see this function's docstring."
            ) from e
        except Exception as e:
            if stop_event is not None and stop_event.is_set():
                return
            attempt += 1
            if attempt > max_attempts:
                raise RuntimeError(
                    f"Gave up reconnecting to Deriv tick stream for {ticker!r} after "
                    f"{max_attempts} attempts. Last error: {type(e).__name__}: {e}"
                ) from e
            delay = min(RECONNECT_MAX_DELAY, RECONNECT_BASE_DELAY * (2 ** (attempt - 1)))
            delay *= 0.8 + 0.4 * random.random()  # +/-20% jitter, avoids thundering-herd reconnects
            if stop_event is not None:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(delay)


async def sample_current_spread_resilient(symbol: str, max_attempts: int = 3,
                                           timeout: float = 10.0) -> Optional[dict]:
    """
    One-shot version of the same resilience for sample_current_spread: tries
    the dynamic-fallback symbol once on a permanent rejection, retries
    transient failures up to `max_attempts` times with backoff, and returns
    None (never raises) if every attempt is exhausted -- appropriate for
    best-effort telemetry callers (spread_telemetry.py, run_paper_daemon.py)
    that must never let a Deriv-side outage crash their own loop.
    """
    ticker = deriv_ticker(symbol)
    tried_fallback = False
    for attempt in range(1, max_attempts + 1):
        try:
            return await sample_current_spread(ticker, timeout=timeout)
        except DerivPermanentError:
            if not tried_fallback:
                tried_fallback = True
                fallback = await resolve_symbol_dynamic(symbol)
                if fallback and fallback != ticker:
                    ticker = fallback
                    continue
            return None
        except (DerivApiError, asyncio.TimeoutError, OSError):
            if attempt == max_attempts:
                return None
            delay = min(RECONNECT_MAX_DELAY, RECONNECT_BASE_DELAY * (2 ** (attempt - 1)))
            delay *= 0.8 + 0.4 * random.random()
            await asyncio.sleep(delay)
    return None
