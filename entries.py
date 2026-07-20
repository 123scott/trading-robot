# -*- coding: utf-8 -*-
"""
entries.py

Execution engine: turns a HTF StructureSnapshot (see structures.py) into a
concrete entry plan, implementing the two entry models and three
entry-trigger techniques from the mechanical playbook:

  Risk Entry         -- a limit/instant order placed directly on the HTF POI
                         the moment price reaches it. No LTF confirmation.
  Confirmation Entry -- price must first sweep into the HTF POI, then print
                         an LTF CHoCH in the trade direction before the
                         entry is considered valid (HTF Intention / LTF
                         Execution).

Trigger techniques (confirmation entries only, applied to the LTF candle
that confirms the CHoCH):
  candle_close         -- enter on that candle's close / next candle's open.
  fifty_pct_engulfing  -- enter at the 50% midpoint of the confirming candle.
  break_of_candle      -- enter only once price re-breaks the confirming
                           candle's own high/low.

The HTF POI is always pulled from the correct side of the dealing range
(discount for buys, premium for sells) via find_htf_poi() -- the bot is
mechanically restricted from buying premium or selling discount by
construction, it never has to be told.

Depends only on structures.py -- no MT5/network/I-O.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import pandas as pd

from structures import (
    Bias, FVG, StructureSnapshot, StructureEvent, StructureEventType,
    ExecutionState, get_swing_points, alternate_swings, label_swing_strength,
    detect_structure_events, build_structure,
)


class EntryMode(Enum):
    RISK = "risk"
    CONFIRMATION = "confirmation"


class EntryTechnique(Enum):
    CANDLE_CLOSE = "candle_close"
    FIFTY_PCT_ENGULFING = "fifty_pct_engulfing"
    BREAK_OF_CANDLE = "break_of_candle"


@dataclass
class EntrySignal:
    state: ExecutionState
    direction: Bias
    mode: EntryMode
    htf_poi: FVG
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    technique: Optional[EntryTechnique] = None
    confirmed_at: Optional[pd.Timestamp] = None
    note: str = ""


def find_htf_poi(htf_snapshot: StructureSnapshot, direction: Bias) -> Optional[FVG]:
    """
    Nearest-to-equilibrium valid FVG on the correct side of the dealing
    range: discount for buys, premium for sells.
    """
    if htf_snapshot.pd_array is None:
        return None
    pool = htf_snapshot.ranked["discount"] if direction == Bias.BULLISH else htf_snapshot.ranked["premium"]
    return pool[0] if pool else None


def risk_entry(htf_poi: FVG, direction: Bias, stop_buffer: float) -> EntrySignal:
    """
    Instant/limit order placed directly on the HTF zone -- no LTF
    confirmation required. Entry sits at the near edge of the zone (the
    boundary price reaches first); stop sits stop_buffer beyond the far edge.
    """
    if direction == Bias.BULLISH:
        entry, stop = htf_poi.top, htf_poi.bottom - stop_buffer
        state = ExecutionState.BUY
    else:
        entry, stop = htf_poi.bottom, htf_poi.top + stop_buffer
        state = ExecutionState.SELL

    return EntrySignal(
        state=state, direction=direction, mode=EntryMode.RISK, htf_poi=htf_poi,
        entry_price=entry, stop_price=stop,
        note="Risk entry: no LTF confirmation, filled directly off the HTF POI.",
    )


def _price_in_zone(low: float, high: float, poi: FVG) -> bool:
    return low <= poi.top and high >= poi.bottom


def _resolve_trigger(
    ltf_slice: pd.DataFrame, confirm_idx: int, direction: Bias,
    technique: EntryTechnique, stop_buffer: float,
) -> Tuple[float, float]:
    confirm_candle = ltf_slice.iloc[confirm_idx]

    if technique == EntryTechnique.CANDLE_CLOSE:
        entry = confirm_candle["close"]
    elif technique == EntryTechnique.BREAK_OF_CANDLE:
        entry = confirm_candle["high"] if direction == Bias.BULLISH else confirm_candle["low"]
    elif technique == EntryTechnique.FIFTY_PCT_ENGULFING:
        entry = (confirm_candle["high"] + confirm_candle["low"]) / 2.0
    else:
        raise ValueError(f"Unknown entry technique: {technique}")

    # Invalidation = the wick that initiated the reversal move, i.e. the
    # sweep extreme since price entered the HTF POI up to the CHoCH itself.
    window = ltf_slice.iloc[: confirm_idx + 1]
    if direction == Bias.BULLISH:
        stop = window["low"].min() - stop_buffer
    else:
        stop = window["high"].max() + stop_buffer

    return entry, stop


def confirmation_entry(
    ltf_df: pd.DataFrame,
    htf_poi: FVG,
    direction: Bias,
    technique: EntryTechnique = EntryTechnique.CANDLE_CLOSE,
    swing_strength: int = 1,
    stop_buffer: float = 0.0,
) -> EntrySignal:
    """
    Confirmation entry per the HTF-Intention/LTF-Execution model: wait for
    price to tap the HTF POI, then require an LTF CHoCH in the trade
    direction before the setup is valid. Returns a WAIT signal if the zone
    hasn't been reached yet or no CHoCH has printed within it.
    """
    highs, lows = ltf_df["high"].values, ltf_df["low"].values

    zone_entry_idx = next(
        (i for i in range(len(ltf_df)) if _price_in_zone(lows[i], highs[i], htf_poi)), None,
    )
    if zone_entry_idx is None:
        return EntrySignal(
            state=ExecutionState.WAIT, direction=direction, mode=EntryMode.CONFIRMATION,
            htf_poi=htf_poi, note="Price hasn't reached the HTF POI yet.",
        )

    ltf_slice = ltf_df.iloc[zone_entry_idx:]
    if len(ltf_slice) < 2 * swing_strength + 3:
        return EntrySignal(
            state=ExecutionState.WAIT, direction=direction, mode=EntryMode.CONFIRMATION,
            htf_poi=htf_poi, note="Inside the HTF POI, awaiting enough LTF bars for a CHoCH read.",
        )

    swings = get_swing_points(ltf_slice, strength=swing_strength)
    alt = alternate_swings(swings)
    label_swing_strength(alt)
    events, _ = detect_structure_events(ltf_slice, alt)

    chochs = [e for e in events if e.type == StructureEventType.CHOCH and e.direction == direction]
    if not chochs:
        return EntrySignal(
            state=ExecutionState.WAIT, direction=direction, mode=EntryMode.CONFIRMATION,
            htf_poi=htf_poi, note="Inside the HTF POI, no LTF CHoCH in the trade direction yet.",
        )

    choch: StructureEvent = chochs[-1]
    confirm_idx = ltf_slice.index.get_loc(choch.time)
    entry, stop = _resolve_trigger(ltf_slice, confirm_idx, direction, technique, stop_buffer)
    state = ExecutionState.BUY if direction == Bias.BULLISH else ExecutionState.SELL

    return EntrySignal(
        state=state, direction=direction, mode=EntryMode.CONFIRMATION, htf_poi=htf_poi,
        entry_price=entry, stop_price=stop, technique=technique, confirmed_at=choch.time,
        note=f"LTF CHoCH confirmed at {choch.time}, broke {choch.broken_level:.5f}.",
    )


def generate_signal(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    direction: Bias,
    mode: EntryMode = EntryMode.CONFIRMATION,
    technique: EntryTechnique = EntryTechnique.CANDLE_CLOSE,
    stop_buffer: float = 0.0,
) -> EntrySignal:
    """Single entry point tying the HTF structure read to an LTF trade plan."""
    htf_snapshot = build_structure(htf_df)
    poi = find_htf_poi(htf_snapshot, direction)
    if poi is None:
        return EntrySignal(
            state=ExecutionState.INVALIDATED, direction=direction, mode=mode, htf_poi=None,
            note="No valid HTF POI on the correct side of the dealing range.",
        )

    if mode == EntryMode.RISK:
        return risk_entry(poi, direction, stop_buffer)
    return confirmation_entry(ltf_df, poi, direction, technique, stop_buffer=stop_buffer)


if __name__ == "__main__":
    import numpy as np

    # --- HTF (H4): same synthetic generator as structures.py's own smoke
    # test, which is already known to produce a confirmed dealing range.
    htf_rng = pd.date_range("2026-01-01", periods=60, freq="4h")
    rs = np.random.default_rng(7)
    htf_close_arr = 2000 + np.cumsum(rs.normal(0, 3, size=len(htf_rng)))
    htf_open = htf_close_arr + rs.normal(0, 0.5, size=len(htf_rng))
    htf_high = np.maximum(htf_open, htf_close_arr) + rs.uniform(0.5, 2, size=len(htf_rng))
    htf_low = np.minimum(htf_open, htf_close_arr) - rs.uniform(0.5, 2, size=len(htf_rng))
    htf = pd.DataFrame({"open": htf_open, "high": htf_high, "low": htf_low, "close": htf_close_arr}, index=htf_rng)

    htf_snapshot = build_structure(htf)
    poi = find_htf_poi(htf_snapshot, Bias.BULLISH)
    print(f"HTF trend: {htf_snapshot.trend} | discount POIs available: {len(htf_snapshot.ranked['discount'])}")
    if poi:
        print(f"Selected HTF discount POI: {poi.bottom:.2f} - {poi.top:.2f} ({poi.state.value})")

        risk = risk_entry(poi, Bias.BULLISH, stop_buffer=0.5)
        print(f"\nRisk entry -> entry={risk.entry_price:.2f} stop={risk.stop_price:.2f}")

        # --- LTF (M15): approaches from above, prints a bearish leg (LH/LL)
        # that sweeps below the POI, then reverses hard enough to close back
        # above the last swing high -- a textbook bullish CHoCH off the zone.
        ltf_rng = pd.date_range("2026-01-01", periods=11, freq="15min")
        ltf_close = [1985, 1970, 1978, 1966, 1973, 1960, 1975, 1990, 1985, 1995, 2005]
        ltf = pd.DataFrame({
            "open": [c - 0.3 for c in ltf_close],
            "high": [c + 0.5 for c in ltf_close],
            "low": [c - 0.5 for c in ltf_close],
            "close": ltf_close,
        }, index=ltf_rng)

        for technique in EntryTechnique:
            signal = confirmation_entry(ltf, poi, Bias.BULLISH, technique=technique, stop_buffer=0.2)
            print(f"\n[{technique.value}] state={signal.state.value} "
                  f"entry={signal.entry_price} stop={signal.stop_price}")
            print(f"  {signal.note}")
    else:
        print("No discount POI formed in this synthetic HTF sample.")
