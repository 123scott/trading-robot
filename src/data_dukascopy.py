# -*- coding: utf-8 -*-
"""
data_dukascopy.py

Real historical tick data from Dukascopy's public historical datafeed
(datafeed.dukascopy.com), used to build multi-year H1 candles for XAUUSD --
neither Yahoo Finance (~2yr H1 cap) nor Deriv (~7mo H1 cap) can supply
2018-present intraday history, and this project does not fabricate
missing history to fill that gap. This is the same raw feed used by
Dukascopy's own JForex historical-data downloader and by open-source
tools like dukascopy-node; no authentication or API key is required for
historical ticks.

Format: one file per (symbol, hour), LZMA-compressed, 20 bytes/tick:
  >i  time offset in ms from the top of the hour
  >i  ask price * point_value
  >i  bid price * point_value
  >f  ask volume
  >f  bid volume
point_value for XAUUSD is 1000 (verified against known real gold prices).

Ticks are aggregated into H1 candles (mid price = (bid+ask)/2) and cached
to data/dukascopy_h1_cache.csv incrementally, so a fetch can resume
without re-downloading hours already retrieved -- this matters because a
multi-year pull is tens of thousands of individual HTTP requests.

No synthetic/generated data: any hour with zero real ticks (market
closed) is recorded as such and skipped when building candles, never
interpolated or filled in.
"""

from __future__ import annotations

import csv
import lzma
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from src.candle import Candle

DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year:04d}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
POINT_VALUE = {"XAUUSD": 1000.0}

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_PATH = os.path.join(CACHE_DIR, "dukascopy_h1_cache.csv")
CACHE_HEADER = ["hour_start_utc", "open", "high", "low", "close", "tick_count"]


@dataclass
class TickHour:
    hour_start: datetime
    ticks: List[tuple]  # (ms_offset, ask, bid)


class RateLimited(Exception):
    pass


def _fetch_one_hour(symbol: str, dt: datetime, session: requests.Session, timeout: float = 20.0,
                     max_retries: int = 6) -> Optional[bytes]:
    """Returns raw bytes (possibly empty = genuinely closed market), or raises after exhausting retries on 429/5xx."""
    url = DUKASCOPY_URL.format(symbol=symbol, year=dt.year, month=dt.month - 1, day=dt.day, hour=dt.hour)
    delay = 1.0
    for attempt in range(max_retries):
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.content  # empty body here is a REAL signal: market closed that hour
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        raise RuntimeError(f"Unexpected status {resp.status_code} for {url}")
    raise RateLimited(f"Exhausted retries (still rate-limited) for {url}")


def _parse_bi5(raw: bytes) -> List[tuple]:
    if not raw:
        return []
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return []
    n = len(data) // 20
    ticks = []
    for i in range(n):
        ms, ask, bid, _askvol, _bidvol = struct.unpack_from(">iiiff", data, i * 20)
        ticks.append((ms, ask, bid))
    return ticks


def _ensure_cache() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CACHE_HEADER)


def _cached_hours() -> set:
    _ensure_cache()
    with open(CACHE_PATH, newline="", encoding="utf-8") as f:
        return {row["hour_start_utc"] for row in csv.DictReader(f)}


def _hour_range(start: datetime, end: datetime):
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        if cur.weekday() != 5:  # skip Saturday -- gold/FX fully closed, no point requesting it
            yield cur
        cur += timedelta(hours=1)


def fetch_and_cache_range(symbol: str, start: str, end: Optional[str] = None,
                           max_workers: int = 16, log=print) -> dict:
    """
    Fetches every missing hour in [start, end) from Dukascopy, builds H1
    OHLC candles from real ticks, and appends them to the persistent cache
    (data/dukascopy_h1_cache.csv). Safe to interrupt and re-run -- already
    -cached hours are skipped. Returns a small summary dict.
    """
    if symbol not in POINT_VALUE:
        raise ValueError(f"No Dukascopy point value configured for {symbol}.")
    point = POINT_VALUE[symbol]

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)

    _ensure_cache()
    already = _cached_hours()
    todo = [h for h in _hour_range(start_dt, end_dt) if h.isoformat() not in already]
    log(f"{len(already)} hours already cached. {len(todo)} hours to fetch for {symbol} "
        f"({start_dt.date()} -> {end_dt.date()})...")

    fetched = empty = errors = 0
    session = requests.Session()
    lock_buffer = []

    def _flush():
        nonlocal lock_buffer
        if not lock_buffer:
            return
        with open(CACHE_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(lock_buffer)
        lock_buffer = []

    def _work(hour_dt):
        try:
            raw = _fetch_one_hour(symbol, hour_dt, session)
            ticks = _parse_bi5(raw) if raw else []
            return hour_dt, ticks, None
        except Exception as e:
            return hour_dt, None, e

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_work, h): h for h in todo}
        done_count = 0
        for fut in as_completed(futures):
            hour_dt, ticks, err = fut.result()
            done_count += 1
            if err is not None:
                errors += 1
                lock_buffer.append([hour_dt.isoformat(), "", "", "", "", "ERROR"])
            elif not ticks:
                empty += 1
                lock_buffer.append([hour_dt.isoformat(), "", "", "", "", 0])
            else:
                prices = [(a + b) / 2.0 / point for _, a, b in ticks]
                lock_buffer.append([hour_dt.isoformat(), prices[0], max(prices), min(prices), prices[-1], len(ticks)])
                fetched += 1

            if len(lock_buffer) >= 200:
                _flush()
            if done_count % 2000 == 0:
                log(f"  ...{done_count}/{len(todo)} hours processed "
                    f"({fetched} with data, {empty} empty, {errors} errors)")
    _flush()

    log(f"Done. {fetched} hours with real ticks, {empty} empty (market closed), {errors} errors.")
    return {"fetched": fetched, "empty": empty, "errors": errors, "total_requested": len(todo)}


def load_h1_candles(start: str, end: Optional[str] = None) -> List[Candle]:
    """Reads H1 candles back out of the cache (only hours with real ticks -- empties are gaps, not zero-filled)."""
    _ensure_cache()
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)

    candles = []
    with open(CACHE_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["tick_count"] in ("0", "ERROR", ""):
                continue
            ts = datetime.fromisoformat(row["hour_start_utc"])
            if ts < start_dt or ts >= end_dt:
                continue
            candles.append(Candle(
                open_time=int(ts.timestamp() * 1000),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]), volume=0.0,
            ))
    candles.sort(key=lambda c: c.open_time)
    return candles


def resample_to_h4(h1_candles: List[Candle]) -> List[Candle]:
    """Groups H1 candles into 4-hour buckets aligned to 00:00 UTC (standard H4 boundaries: 00,04,08,...)."""
    buckets = {}
    for c in h1_candles:
        dt = datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc)
        bucket_hour = (dt.hour // 4) * 4
        key = dt.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)
        buckets.setdefault(key, []).append(c)

    h4 = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda c: c.open_time)
        h4.append(Candle(
            open_time=int(key.timestamp() * 1000),
            open=group[0].open, high=max(c.high for c in group),
            low=min(c.low for c in group), close=group[-1].close, volume=0.0,
        ))
    return h4


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and cache real Dukascopy H1 tick-derived candles.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    fetch_and_cache_range(args.symbol, args.start, args.end, max_workers=args.workers)
