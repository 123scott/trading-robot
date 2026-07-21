# -*- coding: utf-8 -*-
"""
mt5_executor.py

MetaTrader 5 connection and order execution. Talks to a running MT5
terminal via the official `MetaTrader5` Python package.

*** THIS MODULE IS WINDOWS-ONLY AND UNTESTED IN THIS DEVELOPMENT SESSION ***
The `MetaTrader5` package has no macOS/Linux distribution (verified: `pip
install MetaTrader5` fails outright on this dev machine with "no matching
distribution"). This code was written carefully against MetaTrader5's
documented Python API, but it has never actually been executed -- there
was no environment available to run it in. Validate every function here
against a real MT5 terminal on Windows, on a DEMO account, before trusting
it with anything. Treat this as a reviewed draft, not a tested module.

Credentials come from a .env file (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER),
loaded via python-dotenv -- never hardcode them, never commit .env (it's
gitignored; see .env.example for the template).

Safety guard: connect() refuses to proceed on a REAL-money account unless
you explicitly pass allow_live=True. Default posture is demo-only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # allows this module to be imported (e.g. for type-checking) on non-Windows


class MT5NotAvailableError(RuntimeError):
    """Raised when the MetaTrader5 package isn't installed (i.e. not on Windows)."""


class LiveAccountBlockedError(RuntimeError):
    """Raised when connect() detects a real-money account and allow_live wasn't set."""


@dataclass
class OrderResult:
    success: bool
    retcode: Optional[int]
    ticket: Optional[int]
    price: Optional[float]
    volume: Optional[float]
    comment: str


def _require_mt5() -> None:
    if mt5 is None:
        raise MT5NotAvailableError(
            "The MetaTrader5 package is not installed. It only ships for Windows -- "
            "this must run on a Windows machine/VPS with an MT5 terminal installed."
        )


def load_credentials(env_path: Optional[str] = None) -> dict:
    load_dotenv(env_path)  # loads .env from cwd if env_path is None
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")
    missing = [name for name, val in [("MT5_LOGIN", login), ("MT5_PASSWORD", password), ("MT5_SERVER", server)] if not val]
    if missing:
        raise ValueError(f"Missing required .env values: {', '.join(missing)}. Copy .env.example to .env and fill it in.")
    return {"login": int(login), "password": password, "server": server}


def connect(env_path: Optional[str] = None, allow_live: bool = False) -> dict:
    """
    Initializes the MT5 terminal connection and logs in. Returns the
    account_info as a dict. Raises LiveAccountBlockedError if the account
    is not a demo account and allow_live wasn't explicitly set.
    """
    _require_mt5()
    creds = load_credentials(env_path)

    if not mt5.initialize(login=creds["login"], password=creds["password"], server=creds["server"]):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    info = mt5.account_info()
    if info is None:
        mt5.shutdown()
        raise RuntimeError(f"MT5 connected but account_info() returned None: {mt5.last_error()}")

    info_dict = info._asdict()
    is_demo = info_dict.get("trade_mode") == mt5.ACCOUNT_TRADE_MODE_DEMO
    if not is_demo and not allow_live:
        mt5.shutdown()
        raise LiveAccountBlockedError(
            f"Account {info_dict.get('login')} on server {info_dict.get('server')} is NOT a demo "
            f"account (trade_mode={info_dict.get('trade_mode')}). Refusing to proceed -- this module "
            f"defaults to demo-only. Pass allow_live=True only if you have deliberately decided to "
            f"trade real money with fully-tested, reviewed code."
        )

    return info_dict


def disconnect() -> None:
    if mt5 is not None:
        mt5.shutdown()


def get_balance() -> float:
    _require_mt5()
    info = mt5.account_info()
    return info.balance if info else 0.0


def calc_lot_size(symbol: str, risk_amount: float, stop_loss_distance: float) -> float:
    """
    Converts a dollar risk amount + stop-loss distance (in price units) into
    a valid MT5 lot size for `symbol`, respecting the broker's volume_min/
    volume_max/volume_step. Mirrors the standard MT5 EA lot-sizing pattern
    (risk / (sl_distance * pip_value)), clipped and rounded to the step.
    """
    _require_mt5()
    if stop_loss_distance <= 0:
        raise ValueError("stop_loss_distance must be positive.")

    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        raise ValueError(f"Unknown MT5 symbol: {symbol}. Broker symbol names vary (e.g. XAUUSD vs XAUUSD.a) -- "
                          f"check Market Watch in the terminal for the exact name.")

    pip_value = sym_info.trade_tick_value / sym_info.trade_tick_size
    raw_lot = risk_amount / (stop_loss_distance * pip_value)

    step = sym_info.volume_step if sym_info.volume_step > 0 else 0.01
    lot = round(round(raw_lot / step) * step, 2)
    return max(sym_info.volume_min, min(sym_info.volume_max, lot))


def get_open_position(symbol: str, magic: Optional[int] = None) -> Optional[dict]:
    _require_mt5()
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for p in positions:
        if magic is None or p.magic == magic:
            return p._asdict()
    return None


def place_market_order(symbol: str, direction: str, lot: float, sl: Optional[float] = None,
                        tp: Optional[float] = None, magic: int = 202607, comment: str = "amaro-bot",
                        deviation: int = 20) -> OrderResult:
    """direction: "buy" or "sell". Sends a market order via TRADE_ACTION_DEAL."""
    _require_mt5()
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return OrderResult(success=False, retcode=None, ticket=None, price=None, volume=None,
                            comment=f"No tick data for {symbol}")

    order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
    price = tick.ask if direction == "buy" else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if sl is not None:
        request["sl"] = sl
    if tp is not None:
        request["tp"] = tp

    result = mt5.order_send(request)
    if result is None:
        return OrderResult(success=False, retcode=None, ticket=None, price=None, volume=None,
                            comment=f"order_send returned None: {mt5.last_error()}")

    success = result.retcode == mt5.TRADE_RETCODE_DONE
    return OrderResult(success=success, retcode=result.retcode, ticket=getattr(result, "order", None),
                        price=getattr(result, "price", None), volume=getattr(result, "volume", None),
                        comment=result.comment)


def _connectivity_smoke_test() -> None:
    """
    Run this on the actual Windows/MT5 machine to validate the connection
    end-to-end before relying on anything else in this module:

        python -m src.mt5_executor

    Connects, prints account info (refusing to proceed if it's not a demo
    account), and disconnects. Places no orders.
    """
    print("Connecting to MT5 (refuses non-demo accounts by default)...")
    try:
        info = connect()
    except (MT5NotAvailableError, LiveAccountBlockedError, ValueError, RuntimeError) as e:
        print(f"FAILED: {e}")
        return

    print("Connected successfully.")
    print(f"  Login:    {info.get('login')}")
    print(f"  Server:   {info.get('server')}")
    print(f"  Company:  {info.get('company')}")
    print(f"  Balance:  {info.get('balance')} {info.get('currency')}")
    print(f"  Trade mode: {info.get('trade_mode')} (0 = demo)")
    disconnect()
    print("Disconnected. No orders were placed.")


def close_position(symbol: str, ticket: int, volume: float, direction: str,
                    magic: int = 202607, comment: str = "amaro-bot-close", deviation: int = 20) -> OrderResult:
    """Closes an existing position by sending the opposite-side deal against its ticket."""
    _require_mt5()
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return OrderResult(success=False, retcode=None, ticket=None, price=None, volume=None,
                            comment=f"No tick data for {symbol}")

    close_type = mt5.ORDER_TYPE_SELL if direction == "buy" else mt5.ORDER_TYPE_BUY
    price = tick.bid if direction == "buy" else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None:
        return OrderResult(success=False, retcode=None, ticket=None, price=None, volume=None,
                            comment=f"order_send returned None: {mt5.last_error()}")

    success = result.retcode == mt5.TRADE_RETCODE_DONE
    return OrderResult(success=success, retcode=result.retcode, ticket=getattr(result, "order", None),
                        price=getattr(result, "price", None), volume=getattr(result, "volume", None),
                        comment=result.comment)


if __name__ == "__main__":
    _connectivity_smoke_test()
