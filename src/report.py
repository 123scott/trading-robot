# -*- coding: utf-8 -*-
"""
report.py

Computes performance metrics from data/ledger.csv for one or more symbols,
comparing raw vs memory mode, and prints a comparison table plus a simple
risk-adjusted "optimal structural fit" call.

All numbers come directly from ledger rows written by actual replay runs
(python -m src.replay --raw / --memory for each symbol) -- this module does
no simulation of its own, it only aggregates what already happened.

Conventions (stated explicitly since they affect the numbers):
  - Position sizing is notional-based: quantity = notional / entry_price,
    the same fixed dollar notional reused for every trade (simple,
    non-compounding), so PnL $ is directly comparable across assets.
  - Net PnL % = Net PnL $ / notional * 100 (return on that fixed notional).
  - Max Drawdown % is computed off the realized-PnL equity curve
    (base = notional, marked only at each closed trade) -- it does not
    account for intra-trade floating drawdown.
  - Memory Efficiency = % of raw-mode's realized LOSS trades that
    memory-mode converted to SKIP at that same signal timestamp. Requires
    a --raw run to exist for the symbol first; otherwise reported as "--".

Usage:
    python -m src.report --symbols GBPUSD,XAUUSD,USDJPY
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import List, Optional

from src import market_data
from src import memory as memory_mod


@dataclass
class ModeMetrics:
    symbol: str
    mode: str
    trades: int
    skips: int
    wins: int
    losses: int
    breakeven: int
    net_pnl: float
    net_pnl_pct: float
    win_rate_pct: float
    profit_factor: Optional[float]
    max_drawdown_pct: float
    memory_efficiency_pct: Optional[float]
    notional: float = 10_000.0
    open_position: bool = False
    unrealized_pnl: Optional[float] = None

    @property
    def total_pnl(self) -> float:
        """Realized + unrealized (if any open position) -- the honest bottom line."""
        return self.net_pnl + (self.unrealized_pnl or 0.0)

    @property
    def total_pnl_pct(self) -> float:
        return (self.total_pnl / self.notional * 100) if self.notional else 0.0


_LAST_PRICE_CACHE: dict = {}


def _last_known_price(symbol: str) -> Optional[float]:
    """Fetches the single most recent close for `symbol`, cached per report run."""
    if symbol in _LAST_PRICE_CACHE:
        return _LAST_PRICE_CACHE[symbol]
    try:
        if market_data.source_for(symbol) == "binance":
            candles = market_data.fetch_candles(symbol, interval="1d", limit=5)
        else:
            candles = market_data.fetch_candles(symbol, interval="1d", start="2026-01-01")
        # Today's daily candle can still be in progress (close=NaN) if the
        # trading day hasn't closed yet -- walk back to the last complete one.
        price = None
        for c in reversed(candles):
            if c.close == c.close:  # NaN != NaN
                price = c.close
                break
    except Exception:
        price = None
    _LAST_PRICE_CACHE[symbol] = price
    return price


def _read_ledger() -> List[dict]:
    with open(memory_mod.LEDGER_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _memory_efficiency(symbol: str, all_rows: List[dict]) -> Optional[float]:
    raw_by_ts = {r["timestamp"]: r for r in all_rows if r["symbol"] == symbol and r["mode"] == "raw"}
    memory_by_ts = {r["timestamp"]: r for r in all_rows if r["symbol"] == symbol and r["mode"] == "memory"}

    raw_losses = [ts for ts, r in raw_by_ts.items() if r["action"] == "SELL" and r["outcome"] == "LOSS"]
    if not raw_losses:
        return None

    converted = sum(1 for ts in raw_losses if memory_by_ts.get(ts, {}).get("action") == "SKIP")
    return converted / len(raw_losses) * 100


def compute_metrics(symbol: str, mode: str, notional: float, all_rows: List[dict]) -> ModeMetrics:
    rows = [r for r in all_rows if r["symbol"] == symbol and r["mode"] == mode]
    buys = [r for r in rows if r["action"] == "BUY"]
    sells = [r for r in rows if r["action"] == "SELL"]
    skips = [r for r in rows if r["action"] == "SKIP"]

    wins = [r for r in sells if r["outcome"] == "WIN"]
    losses = [r for r in sells if r["outcome"] == "LOSS"]
    breakeven = [r for r in sells if r["outcome"] == "BREAKEVEN"]

    pnls = [float(r["pnl"]) for r in sells]
    net_pnl = sum(pnls)
    net_pnl_pct = (net_pnl / notional * 100) if notional else 0.0

    closed = len(wins) + len(losses) + len(breakeven)
    win_rate = (len(wins) / closed * 100) if closed else 0.0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    equity = notional
    peak = notional
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)

    mem_eff = _memory_efficiency(symbol, all_rows) if mode == "memory" else None

    # A dangling BUY with no matching SELL means the backtest ended still
    # holding a position -- its PnL is unrealized and NOT in net_pnl above.
    # Mark it to the most recent known price so the comparison isn't
    # silently flattering runs that just never got a chance to lose.
    open_position = len(buys) > len(sells)
    unrealized_pnl = None
    if open_position:
        last_buy = buys[-1]
        current_price = _last_known_price(symbol)
        if current_price is not None:
            unrealized_pnl = (current_price - float(last_buy["price"])) * float(last_buy["quantity"])

    return ModeMetrics(
        symbol=symbol, mode=mode, trades=len(buys) + len(sells), skips=len(skips),
        wins=len(wins), losses=len(losses), breakeven=len(breakeven),
        net_pnl=net_pnl, net_pnl_pct=net_pnl_pct, win_rate_pct=win_rate,
        profit_factor=profit_factor, max_drawdown_pct=max_dd, memory_efficiency_pct=mem_eff,
        notional=notional, open_position=open_position, unrealized_pnl=unrealized_pnl,
    )


def print_comparison_table(symbols: List[str], notional: float = 10_000.0) -> List[ModeMetrics]:
    all_rows = _read_ledger()
    results: List[ModeMetrics] = []

    header = (
        f"{'Symbol':8} {'Mode':8} {'Trades':7} {'Skips':6} {'Net PnL $':>12} {'Net PnL %':>10} "
        f"{'Win %':>7} {'ProfitFactor':>13} {'Max DD %':>9} {'MemEff %':>9}"
    )
    print(header)
    print("-" * len(header))

    for symbol in symbols:
        for mode in ("raw", "memory"):
            m = compute_metrics(symbol, mode, notional, all_rows)
            results.append(m)
            pf = f"{m.profit_factor:.2f}" if m.profit_factor is not None else "undef"
            me = f"{m.memory_efficiency_pct:.1f}" if m.memory_efficiency_pct is not None else "--"
            print(
                f"{m.symbol:8} {m.mode:8} {m.trades:7} {m.skips:6} {m.net_pnl:12,.2f} {m.net_pnl_pct:10.2f} "
                f"{m.win_rate_pct:7.1f} {pf:>13} {m.max_drawdown_pct:9.2f} {me:>9}"
            )
            if m.open_position:
                mark = f"{m.unrealized_pnl:+,.2f}" if m.unrealized_pnl is not None else "unknown (price fetch failed)"
                print(f"         ^ WARNING: backtest ended holding an OPEN position, not reflected above. "
                      f"Unrealized PnL (marked to last known price): {mark}")
        print()

    return results


def highlight_optimal(results: List[ModeMetrics]) -> Optional[ModeMetrics]:
    """
    Simple Calmar-style risk-adjusted ranking (Total PnL % / Max Drawdown %)
    among memory-mode results. Uses TOTAL PnL (realized + any unrealized
    open-position mark) so a run can't look artificially good just because
    it's still sitting on an unclosed position.
    """
    candidates = [r for r in results if r.mode == "memory" and r.max_drawdown_pct > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.total_pnl_pct / r.max_drawdown_pct)


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance comparison report across symbols/modes.")
    parser.add_argument("--symbols", default="GBPUSD,XAUUSD,USDJPY")
    parser.add_argument("--notional", type=float, default=10_000.0)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    results = print_comparison_table(symbols, notional=args.notional)

    best = highlight_optimal(results)
    if best:
        note = " (includes an unrealized open-position mark)" if best.open_position else ""
        print(
            f"Optimal structural fit (memory mode, total-return/drawdown ratio): {best.symbol} "
            f"(Total PnL {best.total_pnl_pct:.2f}%{note}, Max DD {best.max_drawdown_pct:.2f}%, "
            f"ratio {best.total_pnl_pct / best.max_drawdown_pct:.2f})"
        )


if __name__ == "__main__":
    main()
