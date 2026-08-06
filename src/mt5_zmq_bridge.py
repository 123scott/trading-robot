# -*- coding: utf-8 -*-
"""
mt5_zmq_bridge.py

Native macOS/Linux replacement for mt5_executor.py's direct MetaTrader5
API calls. The MetaTrader5 Python package only ships for Windows (see
mt5_executor.py's docstring) -- this module instead talks to a companion
MQL5 Expert Advisor (mt5_bridge_ea/AmaroZmqBridge.mq5) over a local
ZeroMQ REQ/REP socket. The strategy/memory logic (all of it: everything
in src/) now runs entirely natively on this machine; the only thing that
still needs an actual MT5 runtime is the terminal + EA itself -- MT5
terminal isn't a native macOS binary either, so that still means a
Windows machine/VM, or your broker's own Mac MT5 build if they ship one.
That's a much smaller dependency than before (just the "hands" that
place orders, not the "brain" that decides what to trade).

Genuine side benefit: this path never touches MT5 login credentials at
all. You log into the terminal once, manually, via its own GUI, and
attach the EA to a chart -- the EA operates against that already-
authenticated session. No .env, no MT5_LOGIN/MT5_PASSWORD anywhere in
this module.

Protocol: flat (no nested objects/arrays in *requests* -- deliberately,
since MQL5 has no built-in JSON library and hand-rolled parsing of a
flat schema is far more tractable than a general one) JSON request/
response over a single REQ/REP socket pair. REQ/REP enforces strict
send-then-recv alternation, so there's no need for our own sequencing or
request IDs. Every request has a "cmd" field; every reply has an "ok"
boolean plus either the payload or an "error" string.

    -> {"cmd": "PING"}
    <- {"ok": true, "account": {"login":.., "server":.., "balance":..,
                                 "currency":.., "trade_mode": 0}}

    -> {"cmd": "SYMBOL_INFO", "symbol": "XAUUSD"}
    <- {"ok": true, "tick_value":.., "tick_size":.., "volume_min":..,
        "volume_max":.., "volume_step":..}

    -> {"cmd": "RATES", "symbol": "XAUUSD", "timeframe": "D1", "start_pos": 1, "count": 60}
    <- {"ok": true, "rates": [{"time":.., "open":.., "high":.., "low":..,
                                "close":.., "tick_volume":..}, ...]}

    -> {"cmd": "TICK", "symbol": "XAUUSD"}
    <- {"ok": true, "bid":.., "ask":.., "time":..}

    -> {"cmd": "ORDER", "symbol": "XAUUSD", "direction": "buy", "volume": 0.10,
        "sl": 2340.0, "tp": 2410.0, "magic": 202607, "comment": "amaro-bot",
        "deviation": 20}
    <- {"ok": true, "retcode":.., "ticket":.., "price":.., "volume":.., "comment":..}

    -> {"cmd": "POSITIONS", "symbol": "XAUUSD", "magic": 202607}
    <- {"ok": true, "positions": [{"ticket":.., "volume":.., "price_open":..,
                                    "sl":.., "tp":.., "magic":..}, ...]}
       (magic: 0 means "no filter" -- this project's real magic number, 202607,
        is never 0, so 0 is safe to reserve as the sentinel.)

    -> {"cmd": "CLOSE", "symbol": "XAUUSD", "ticket": 123456, "volume": 0.10,
        "deviation": 20, "comment": "amaro-bot-close"}
    <- {"ok": true, "retcode":.., "ticket":.., "price":.., "volume":.., "comment":..}

SECURITY: this socket has no authentication or encryption of its own.
Only bind/connect it on 127.0.0.1 (same machine) or a private/host-only
VM network -- never expose this port on a public interface or the open
internet; anyone who can reach it can place orders on the connected
account.

Demo-only enforcement now exists in TWO independent places:
  1. The EA itself refuses to initialize at all (OnInit returns
     INIT_FAILED) if the attached account isn't a demo account -- the
     stronger guarantee, since a live account can literally never load
     the EA regardless of what any Python code does.
  2. connect() below also checks the account info this bridge receives
     and raises LiveAccountBlockedError if it's somehow not a demo
     account, as defense in depth. Neither this module nor mt5_live.py
     exposes any way to bypass either check.

*** UNTESTED end-to-end (no MT5 terminal available in this development
environment to run the EA against) -- the Python half below is ordinary,
testable code, but validate the full round-trip against a real EA/demo
account before trusting it. ***
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import zmq

from src.candle import Candle

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555
DEFAULT_TIMEOUT_MS = 5000


class BridgeNotConnectedError(RuntimeError):
    """Raised when a call is made before connect()."""


class BridgeTimeoutError(RuntimeError):
    """Raised when the EA doesn't reply within the timeout -- it may not be attached/running."""


class BridgeError(RuntimeError):
    """Raised when the EA replies with ok: false."""


class LiveAccountBlockedError(RuntimeError):
    """Raised when connect() detects a real-money account (defense in depth; the EA should already refuse to load)."""


@dataclass
class OrderResult:
    success: bool
    retcode: Optional[int]
    ticket: Optional[int]
    price: Optional[float]
    volume: Optional[float]
    comment: str


_context: Optional["zmq.Context"] = None
_socket: Optional["zmq.Socket"] = None
_endpoint: Optional[str] = None
_timeout_ms: int = DEFAULT_TIMEOUT_MS

_TIMEFRAME_MAP = {"M1": "M1", "M5": "M5", "M15": "M15", "H1": "H1", "H4": "H4", "D1": "D1"}


def _new_socket() -> "zmq.Socket":
    sock = _context.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, _timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, _timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(_endpoint)
    return sock


def _send(payload: dict) -> dict:
    global _socket
    if _socket is None:
        raise BridgeNotConnectedError("Not connected -- call connect() first.")
    try:
        _socket.send_json(payload)
        resp = _socket.recv_json()
    except zmq.error.Again:
        # A REQ socket that times out is stuck until it gets its matching reply --
        # the only clean recovery is to discard it and reconnect fresh.
        _socket.close(0)
        _socket = _new_socket()
        raise BridgeTimeoutError(
            f"No response from the MT5 EA within {_timeout_ms}ms (cmd={payload.get('cmd')}). "
            "Is AmaroZmqBridge.mq5 attached to a chart, running (smiley face, AutoTrading "
            "enabled), and bound to the same host/port?"
        )
    if not resp.get("ok", False):
        raise BridgeError(f"EA reported an error for {payload.get('cmd')}: {resp.get('error')}")
    return resp


def connect(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> dict:
    """Connects to the EA's REP socket and returns its account info dict. Refuses non-demo accounts."""
    global _context, _socket, _endpoint, _timeout_ms
    _timeout_ms = timeout_ms
    _endpoint = f"tcp://{host}:{port}"
    _context = zmq.Context.instance()
    _socket = _new_socket()

    resp = _send({"cmd": "PING"})
    account = resp["account"]
    # ACCOUNT_TRADE_MODE_DEMO == 0 in MQL5's ENUM_ACCOUNT_TRADE_MODE (matches the
    # MetaTrader5 Python package's mt5.ACCOUNT_TRADE_MODE_DEMO, also 0).
    if account.get("trade_mode") != 0:
        disconnect()
        raise LiveAccountBlockedError(
            f"Account {account.get('login')} on server {account.get('server')} is NOT a demo "
            f"account (trade_mode={account.get('trade_mode')}). Refusing to proceed. This should "
            f"be unreachable -- the EA itself is supposed to refuse to even load on a non-demo "
            f"account -- so if you see this, something about the EA's own guard didn't fire."
        )
    return account


def disconnect() -> None:
    global _socket
    if _socket is not None:
        _socket.close(0)
        _socket = None


def get_balance() -> float:
    resp = _send({"cmd": "PING"})
    return resp["account"].get("balance", 0.0)


def get_symbol_info(symbol: str) -> dict:
    """Raw SYMBOL_INFO reply: tick_value, tick_size, volume_min/max/step."""
    return _send({"cmd": "SYMBOL_INFO", "symbol": symbol})


def calc_lot_size(symbol: str, risk_amount: float, stop_loss_distance: float) -> float:
    """Same math as mt5_executor.calc_lot_size, sourced from the EA's SYMBOL_INFO reply instead of the native API."""
    if stop_loss_distance <= 0:
        raise ValueError("stop_loss_distance must be positive.")
    info = get_symbol_info(symbol)
    pip_value = info["tick_value"] / info["tick_size"]
    raw_lot = risk_amount / (stop_loss_distance * pip_value)
    step = info["volume_step"] if info["volume_step"] > 0 else 0.01
    lot = round(round(raw_lot / step) * step, 2)
    return max(info["volume_min"], min(info["volume_max"], lot))


def get_rates(symbol: str, timeframe: str = "D1", start_pos: int = 1, count: int = 60) -> List[Candle]:
    """Fetches `count` completed bars (start_pos=1 skips the still-forming current bar)."""
    tf = _TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {sorted(_TIMEFRAME_MAP)}")
    resp = _send({"cmd": "RATES", "symbol": symbol, "timeframe": tf, "start_pos": start_pos, "count": count})
    rates = resp.get("rates", [])
    if not rates:
        raise RuntimeError(f"EA returned no rates for {symbol}/{timeframe}.")
    return [Candle(open_time=int(r["time"]) * 1000, open=float(r["open"]), high=float(r["high"]),
                    low=float(r["low"]), close=float(r["close"]), volume=float(r["tick_volume"]))
            for r in rates]


def get_tick(symbol: str) -> dict:
    resp = _send({"cmd": "TICK", "symbol": symbol})
    return {"bid": resp["bid"], "ask": resp["ask"], "time": resp["time"]}


def get_open_position(symbol: str, magic: Optional[int] = None) -> Optional[dict]:
    resp = _send({"cmd": "POSITIONS", "symbol": symbol, "magic": magic or 0})
    for p in resp.get("positions", []):
        if magic is None or p.get("magic") == magic:
            return p
    return None


def place_market_order(symbol: str, direction: str, lot: float, sl: Optional[float] = None,
                        tp: Optional[float] = None, magic: int = 202607, comment: str = "amaro-bot",
                        deviation: int = 20) -> OrderResult:
    resp = _send({"cmd": "ORDER", "symbol": symbol, "direction": direction, "volume": lot,
                   "sl": sl or 0.0, "tp": tp or 0.0, "magic": magic, "comment": comment, "deviation": deviation})
    return OrderResult(success=True, retcode=resp.get("retcode"), ticket=resp.get("ticket"),
                        price=resp.get("price"), volume=resp.get("volume"), comment=resp.get("comment", ""))


def close_position(symbol: str, ticket: int, volume: float, direction: str,
                    magic: int = 202607, comment: str = "amaro-bot-close", deviation: int = 20) -> OrderResult:
    resp = _send({"cmd": "CLOSE", "symbol": symbol, "ticket": ticket, "volume": volume,
                   "deviation": deviation, "comment": comment})
    return OrderResult(success=True, retcode=resp.get("retcode"), ticket=resp.get("ticket"),
                        price=resp.get("price"), volume=resp.get("volume"), comment=resp.get("comment", ""))


def _connectivity_smoke_test() -> None:
    """
    Run this against a running EA before relying on anything else:

        python -m src.mt5_zmq_bridge --host 127.0.0.1 --port 5555

    Connects, prints account info (refusing to proceed if it's not a demo
    account), and disconnects. Places no orders.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ZeroMQ bridge connectivity smoke test.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    print(f"Connecting to AmaroZmqBridge EA at tcp://{args.host}:{args.port} (refuses non-demo accounts)...")
    try:
        account = connect(host=args.host, port=args.port)
    except (BridgeTimeoutError, LiveAccountBlockedError, BridgeError) as e:
        print(f"FAILED: {e}")
        return

    print("Connected successfully.")
    print(f"  Login:    {account.get('login')}")
    print(f"  Server:   {account.get('server')}")
    print(f"  Balance:  {account.get('balance')} {account.get('currency')}")
    print(f"  Trade mode: {account.get('trade_mode')} (0 = demo)")
    disconnect()
    print("Disconnected. No orders were placed.")


if __name__ == "__main__":
    _connectivity_smoke_test()
