# -*- coding: utf-8 -*-
"""
data_binance.py

Fetches real historical candles from Binance's public klines REST endpoint.
No API key is required -- this is a public market-data endpoint. Every
replay run pulls a live candle set; no generated or fixture data is used
anywhere in this module.

Endpoint: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass
class Candle:
    open_time: int  # epoch millis
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500) -> List[Candle]:
    """Pull the `limit` most recent candles for `symbol`/`interval` from Binance."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    rows = resp.json()

    return [
        Candle(
            open_time=row[0],
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in rows
    ]
