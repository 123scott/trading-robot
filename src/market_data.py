# -*- coding: utf-8 -*-
"""
market_data.py

Single entry point the replay/trading_robot layer calls to get candles,
regardless of which underlying source a symbol actually comes from. Keeps
strategy/memory code fully data-source-agnostic.
"""

from __future__ import annotations

from typing import List, Optional

from src.candle import Candle
from src.data_binance import fetch_klines
from src.data_yfinance import fetch_yfinance_candles, YFINANCE_TICKERS

BINANCE_SYMBOLS = {"BTCUSDT"}
YFINANCE_SYMBOLS = set(YFINANCE_TICKERS)
SUPPORTED_SYMBOLS = BINANCE_SYMBOLS | YFINANCE_SYMBOLS


def source_for(symbol: str) -> str:
    if symbol in BINANCE_SYMBOLS:
        return "binance"
    if symbol in YFINANCE_SYMBOLS:
        return "yfinance"
    raise ValueError(f"Unsupported symbol: {symbol}. Supported: {sorted(SUPPORTED_SYMBOLS)}")


def default_interval_for(symbol: str) -> str:
    """Binance defaults to 1h (short recent windows); yfinance defaults to
    1d since Yahoo doesn't retain multi-year intraday history."""
    return "1h" if source_for(symbol) == "binance" else "1d"


def fetch_candles(symbol: str, interval: Optional[str] = None, limit: int = 500,
                   start: Optional[str] = None, end: Optional[str] = None) -> List[Candle]:
    """
    Fetches real candles for `symbol`, dispatching to Binance or Yahoo
    Finance as appropriate. If `start` is given, pulls the full date range;
    otherwise pulls the `limit` most recent candles (Binance only -- yfinance
    always requires a start date and defaults to 2018-01-01 if none given).
    """
    source = source_for(symbol)
    interval = interval or default_interval_for(symbol)

    if source == "binance":
        return fetch_klines(symbol=symbol, interval=interval, limit=limit, start=start, end=end)

    return fetch_yfinance_candles(symbol=symbol, interval=interval, start=start or "2018-01-01", end=end)
