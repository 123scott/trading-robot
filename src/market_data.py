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
from src.data_deriv import fetch_deriv_candles, DERIV_SYMBOLS, DEFAULT_SPREAD

BINANCE_SYMBOLS = {"BTCUSDT"}
YFINANCE_SYMBOLS = set(YFINANCE_TICKERS)
DERIV_SYMBOL_NAMES = set(DERIV_SYMBOLS)
SUPPORTED_SYMBOLS = BINANCE_SYMBOLS | YFINANCE_SYMBOLS | DERIV_SYMBOL_NAMES

# Approximate spread (price units) applied at execution for spread-aware
# sources. 0.0 = fills exactly at signal price (the original behavior).
SPREAD_BY_SYMBOL = {"XAUUSD_DERIV": DEFAULT_SPREAD}


def source_for(symbol: str) -> str:
    if symbol in BINANCE_SYMBOLS:
        return "binance"
    if symbol in YFINANCE_SYMBOLS:
        return "yfinance"
    if symbol in DERIV_SYMBOL_NAMES:
        return "deriv"
    raise ValueError(f"Unsupported symbol: {symbol}. Supported: {sorted(SUPPORTED_SYMBOLS)}")


def default_interval_for(symbol: str) -> str:
    """Binance/Deriv default to 1h/1d respectively for recent windows;
    yfinance defaults to 1d since Yahoo doesn't retain multi-year
    intraday history."""
    if source_for(symbol) == "binance":
        return "1h"
    return "1d"


def spread_for(symbol: str) -> float:
    """Approximate round-trip spread cost applied at execution (0.0 if unmodeled for this symbol)."""
    return SPREAD_BY_SYMBOL.get(symbol, 0.0)


def fetch_candles(symbol: str, interval: Optional[str] = None, limit: int = 500,
                   start: Optional[str] = None, end: Optional[str] = None) -> List[Candle]:
    """
    Fetches real candles for `symbol`, dispatching to Binance, Yahoo
    Finance, or Deriv as appropriate. If `start` is given, pulls the full
    date range; otherwise pulls the `limit` most recent candles (Binance
    only -- yfinance/Deriv always fetch by date range and default to
    2018-01-01 if no start is given, though Deriv's actual history for
    XAUUSD_DERIV is capped to roughly the trailing year regardless -- see
    src/data_deriv.py).
    """
    source = source_for(symbol)
    interval = interval or default_interval_for(symbol)

    if source == "binance":
        return fetch_klines(symbol=symbol, interval=interval, limit=limit, start=start, end=end)
    if source == "deriv":
        return fetch_deriv_candles(symbol=symbol, interval=interval, start=start, end=end, limit=max(limit, 5000))

    return fetch_yfinance_candles(symbol=symbol, interval=interval, start=start or "2018-01-01", end=end)
