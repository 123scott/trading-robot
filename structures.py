# -*- coding: utf-8 -*-
"""
structures.py

Mechanical market-structure engine: swing/ITH-ITL detection, the three-candle
FVG parser (with mitigation / inversion / fill state tracking), and dynamic
Premium/Discount (PD) array construction.

Input contract: every function takes a pandas DataFrame with columns
["open", "high", "low", "close"] and a monotonically increasing datetime
index (oldest -> newest), e.g. the output of mt5.copy_rates_from_pos() run
through pd.DataFrame(...).set_index("time").

This module is pure analysis — it has no MT5, network, or I/O dependency,
so it can be unit tested on static OHLC data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


# ============================================================
# Enums
# ============================================================

class Bias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class Zone(Enum):
    PREMIUM = "premium"
    DISCOUNT = "discount"
    EQUILIBRIUM = "equilibrium"


class FVGState(Enum):
    UNMITIGATED = "unmitigated"   # untouched void, full PD-array priority
    MITIGATED = "mitigated"       # wick has tapped into the zone at least once
    FILLED = "filled"             # price has wicked all the way through the zone
    INVERTED = "inverted"         # a candle body closed through the far edge (iFVG)


class ExecutionState(Enum):
    BUY = "buy"
    SELL = "sell"
    WAIT = "wait"
    INVALIDATED = "invalidated"


class SwingStrength(Enum):
    """
    Per the weak-high/strong-low (and mirrored strong-high/weak-low) model:
    in a bullish leg every low is STRONG (real demand, protects the trend)
    and every high is WEAK (temporary liquidity, expected to be swept as
    the trend continues) -- and vice versa in a bearish leg.
    """
    WEAK = "weak"
    STRONG = "strong"


class StructureEventType(Enum):
    BOS = "bos"      # break of structure -- continues the existing trend
    CHOCH = "choch"  # change of character -- reverses the trend


# ============================================================
# Data containers
# ============================================================

@dataclass
class FVG:
    """A single Fair Value Gap, anchored to the middle (displacement) candle."""
    index: int                 # positional index of candle 2 in the source df
    time: pd.Timestamp
    direction: Bias
    top: float
    bottom: float
    state: FVGState = FVGState.UNMITIGATED
    mitigated_at: Optional[pd.Timestamp] = None
    filled_at: Optional[pd.Timestamp] = None
    inverted_at: Optional[pd.Timestamp] = None

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def is_tradeable(self) -> bool:
        """Only unmitigated / mitigated gaps retain their original polarity as PD arrays."""
        return self.state in (FVGState.UNMITIGATED, FVGState.MITIGATED)


@dataclass
class SwingPoint:
    index: int
    time: pd.Timestamp
    price: float
    kind: str                       # "high" or "low"
    is_intermediate: bool = False   # confirmed ITH / ITL
    swept_at: Optional[pd.Timestamp] = None
    strength: Optional[SwingStrength] = None


@dataclass
class PDArray:
    range_high: float
    range_low: float
    equilibrium: float = field(init=False)
    premium_bottom: float = field(init=False)
    premium_top: float = field(init=False)
    discount_bottom: float = field(init=False)
    discount_top: float = field(init=False)

    def __post_init__(self):
        self.equilibrium = (self.range_high + self.range_low) / 2.0
        self.premium_bottom = self.equilibrium
        self.premium_top = self.range_high
        self.discount_bottom = self.range_low
        self.discount_top = self.equilibrium

    def zone_of(self, price: float) -> Zone:
        if price > self.equilibrium:
            return Zone.PREMIUM
        if price < self.equilibrium:
            return Zone.DISCOUNT
        return Zone.EQUILIBRIUM

    def can_buy(self, price: float) -> bool:
        return self.zone_of(price) == Zone.DISCOUNT

    def can_sell(self, price: float) -> bool:
        return self.zone_of(price) == Zone.PREMIUM


# ============================================================
# Swing points / ITH / ITL
# ============================================================

def get_swing_points(df: pd.DataFrame, strength: int = 1) -> List[SwingPoint]:
    """
    Fractal swing detection: bar i is a swing high if its high is strictly
    greater than the `strength` bars on both sides; symmetric for lows.
    Ordered oldest -> newest, alternating is NOT enforced here (raw fractals).
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swings: List[SwingPoint] = []

    for i in range(strength, n - strength):
        window_h = highs[i - strength:i + strength + 1]
        if highs[i] == window_h.max() and np.argmax(window_h) == strength:
            swings.append(SwingPoint(index=i, time=df.index[i], price=highs[i], kind="high"))
        window_l = lows[i - strength:i + strength + 1]
        if lows[i] == window_l.min() and np.argmin(window_l) == strength:
            swings.append(SwingPoint(index=i, time=df.index[i], price=lows[i], kind="low"))

    swings.sort(key=lambda s: s.index)
    return swings


def get_intermediate_points(df: pd.DataFrame, swings: List[SwingPoint]) -> List[SwingPoint]:
    """
    Confirm Intermediate Term Highs (ITH) / Intermediate Term Lows (ITL).

    Mechanical definition applied here:
      ITH = a swing high flanked by a lower swing high before it and a lower
            swing high after it (i.e. locally the highest point of its
            cluster — it swept the internal liquidity resting above the two
            flanking highs), AND price subsequently produces a body-close
            break below the swing LOW that sits between the ITH and its
            prior swing high (confirms the sweep led to a structural shift
            down, not just a higher-high continuation).
      ITL = mirror image.

    Returns only the confirmed subset, each flagged is_intermediate=True with
    swept_at set to the timestamp of the confirming structural break.
    """
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    closes = df["close"].values
    confirmed: List[SwingPoint] = []

    # --- ITH: flanked by two lower highs, then a body close below the
    # internal low that separates it from the prior high confirms the shift.
    for j in range(1, len(highs) - 1):
        prev_h, this_h, next_h = highs[j - 1], highs[j], highs[j + 1]
        if not (this_h.price > prev_h.price and this_h.price > next_h.price):
            continue
        internal_lows = [l for l in lows if prev_h.index < l.index < this_h.index]
        if not internal_lows:
            continue
        sweep_level = min(l.price for l in internal_lows)
        for k in range(this_h.index + 1, len(closes)):
            if closes[k] < sweep_level:
                confirmed.append(SwingPoint(
                    index=this_h.index, time=this_h.time, price=this_h.price,
                    kind="high", is_intermediate=True, swept_at=df.index[k],
                ))
                break

    # --- ITL: mirror image, flanked by two higher lows, confirmed by a body
    # close above the internal high that separates it from the prior low.
    for j in range(1, len(lows) - 1):
        prev_l, this_l, next_l = lows[j - 1], lows[j], lows[j + 1]
        if not (this_l.price < prev_l.price and this_l.price < next_l.price):
            continue
        internal_highs = [h for h in highs if prev_l.index < h.index < this_l.index]
        if not internal_highs:
            continue
        sweep_level = max(h.price for h in internal_highs)
        for k in range(this_l.index + 1, len(closes)):
            if closes[k] > sweep_level:
                confirmed.append(SwingPoint(
                    index=this_l.index, time=this_l.time, price=this_l.price,
                    kind="low", is_intermediate=True, swept_at=df.index[k],
                ))
                break

    confirmed.sort(key=lambda s: s.index)
    return confirmed


# ============================================================
# BOS / CHoCH engine + weak/strong swing labeling
# ============================================================

@dataclass
class StructureEvent:
    type: StructureEventType
    direction: Bias           # direction of the break: BULLISH = closed above a high
    time: pd.Timestamp
    broken_level: float
    broken_swing: SwingPoint  # the prior same-kind swing that got taken out


def alternate_swings(swings: List[SwingPoint]) -> List[SwingPoint]:
    """
    Collapse raw fractals into a strict zigzag (high, low, high, low, ...) by
    keeping only the most extreme point of any run of same-kind swings.
    BOS/CHoCH and weak/strong labeling both require this clean alternation.
    """
    if not swings:
        return []
    alt = [swings[0]]
    for s in swings[1:]:
        if s.kind == alt[-1].kind:
            if (s.kind == "high" and s.price > alt[-1].price) or \
               (s.kind == "low" and s.price < alt[-1].price):
                alt[-1] = s
        else:
            alt.append(s)
    return alt


def label_swing_strength(alt_swings: List[SwingPoint]) -> None:
    """
    Mutates each swing in place. A high is WEAK if it's higher than the
    previous same-kind high (HH -> bullish leg continuing -> this high is
    just fuel waiting to be swept) and STRONG if it's lower (LH -> bearish
    leg -> this high is the protected structural level). Mirror for lows:
    a higher low (HL) is STRONG, a lower low (LL) is WEAK.
    """
    for i in range(2, len(alt_swings)):
        prev_same, curr = alt_swings[i - 2], alt_swings[i]
        if curr.kind == "high":
            curr.strength = SwingStrength.WEAK if curr.price > prev_same.price else SwingStrength.STRONG
        else:
            curr.strength = SwingStrength.STRONG if curr.price > prev_same.price else SwingStrength.WEAK


def detect_structure_events(df: pd.DataFrame, alt_swings: List[SwingPoint]) -> tuple:
    """
    Mechanical BOS/CHoCH: for each new same-kind swing (e.g. a fresh high),
    find the first candle body CLOSE beyond the previous same-kind swing
    (two steps back in the zigzag) — that close is the actual break event,
    not the wick that formed the new fractal. A break in the direction of
    the prevailing trend is a BOS (continuation); a break against it flips
    the trend and is a CHoCH. Wick-only reaches that never close through
    (the "short/long trap" case) never fire an event.

    Returns (events, final_trend).
    """
    closes = df["close"].values
    events: List[StructureEvent] = []
    trend: Optional[Bias] = None

    for i in range(2, len(alt_swings)):
        prev_same, curr = alt_swings[i - 2], alt_swings[i]
        for k in range(prev_same.index + 1, curr.index + 1):
            broke = (curr.kind == "high" and closes[k] > prev_same.price) or \
                    (curr.kind == "low" and closes[k] < prev_same.price)
            if not broke:
                continue
            direction = Bias.BULLISH if curr.kind == "high" else Bias.BEARISH
            event_type = StructureEventType.BOS if (trend is None or direction == trend) else StructureEventType.CHOCH
            trend = direction
            events.append(StructureEvent(
                type=event_type, direction=direction, time=df.index[k],
                broken_level=prev_same.price, broken_swing=prev_same,
            ))
            break

    return events, trend


# ============================================================
# FVG engine (three-candle sequence + lifecycle state)
# ============================================================

def detect_fvgs(df: pd.DataFrame) -> List[FVG]:
    """
    Scan every consecutive candle triplet (C1, C2, C3). A gap exists when
    C1's wick and C3's wick do not overlap, leaving C2 as an open void:
      Bullish FVG: C1.high < C3.low   -> void = [C1.high, C3.low]
      Bearish FVG: C1.low  > C3.high  -> void = [C3.high, C1.low]
    The FVG is anchored to C2 (the displacement candle).
    """
    fvgs: List[FVG] = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    for i in range(2, len(df)):
        c1_high, c1_low = h[i - 2], l[i - 2]
        c3_high, c3_low = h[i], l[i]

        if c1_high < c3_low:
            fvgs.append(FVG(
                index=i - 1, time=df.index[i - 1], direction=Bias.BULLISH,
                top=c3_low, bottom=c1_high,
            ))
        elif c1_low > c3_high:
            fvgs.append(FVG(
                index=i - 1, time=df.index[i - 1], direction=Bias.BEARISH,
                top=c1_low, bottom=c3_high,
            ))

    return fvgs


def update_fvg_states(df: pd.DataFrame, fvgs: List[FVG]) -> None:
    """
    Walk price forward from each FVG's formation candle and advance its
    lifecycle state in place. State only ever moves forward
    (UNMITIGATED -> MITIGATED -> FILLED, with INVERTED overriding as a
    terminal state the moment a body close breaks the far edge) — an FVG
    that has flipped polarity never reverts to acting as its original zone.
    """
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values

    for gap in fvgs:
        start = gap.index + 2  # first candle fully formed after the gap
        for k in range(start, len(df)):
            if gap.direction == Bias.BULLISH:
                if closes[k] < gap.bottom:
                    gap.state = FVGState.INVERTED
                    gap.inverted_at = df.index[k]
                    break
                if lows[k] <= gap.bottom:
                    gap.state = FVGState.FILLED
                    gap.filled_at = df.index[k]
                elif lows[k] <= gap.top and gap.state == FVGState.UNMITIGATED:
                    gap.state = FVGState.MITIGATED
                    gap.mitigated_at = df.index[k]
            else:  # BEARISH
                if closes[k] > gap.top:
                    gap.state = FVGState.INVERTED
                    gap.inverted_at = df.index[k]
                    break
                if highs[k] >= gap.top:
                    gap.state = FVGState.FILLED
                    gap.filled_at = df.index[k]
                elif highs[k] >= gap.bottom and gap.state == FVGState.UNMITIGATED:
                    gap.state = FVGState.MITIGATED
                    gap.mitigated_at = df.index[k]


def active_fvgs(fvgs: List[FVG], direction: Optional[Bias] = None) -> List[FVG]:
    """Unmitigated/mitigated gaps only — the ones still valid as PD arrays."""
    pool = [g for g in fvgs if g.is_tradeable]
    if direction is not None:
        pool = [g for g in pool if g.direction == direction]
    return pool


# ============================================================
# PD Array construction
# ============================================================

def htf_dealing_range(intermediate_points: List[SwingPoint]) -> Optional[PDArray]:
    """
    Build the current dealing range from the most recent confirmed ITH and
    ITL (regardless of order), i.e. the last leg the market actually
    respects. Returns None until both an ITH and an ITL have printed.
    """
    iths = [p for p in intermediate_points if p.kind == "high"]
    itls = [p for p in intermediate_points if p.kind == "low"]
    if not iths or not itls:
        return None
    return PDArray(range_high=iths[-1].price, range_low=itls[-1].price)


def rank_pd_arrays(pd_array: PDArray, fvgs: List[FVG]) -> dict:
    """
    Partition the still-valid (unmitigated/mitigated) FVGs into the
    discount and premium halves of the current dealing range, ordered
    nearest-to-equilibrium first (highest priority reaction zones).
    """
    in_range = [
        g for g in fvgs
        if g.is_tradeable and pd_array.range_low <= g.midpoint <= pd_array.range_high
    ]
    discount = sorted(
        (g for g in in_range if pd_array.zone_of(g.midpoint) == Zone.DISCOUNT),
        key=lambda g: -g.midpoint,
    )
    premium = sorted(
        (g for g in in_range if pd_array.zone_of(g.midpoint) == Zone.PREMIUM),
        key=lambda g: g.midpoint,
    )
    return {"discount": discount, "premium": premium}


# ============================================================
# Composite structure snapshot
# ============================================================

@dataclass
class StructureSnapshot:
    pd_array: Optional[PDArray]
    fvgs: List[FVG]
    swings: List[SwingPoint]
    intermediate_points: List[SwingPoint]
    ranked: dict
    alt_swings: List[SwingPoint] = field(default_factory=list)
    structure_events: List[StructureEvent] = field(default_factory=list)
    trend: Optional[Bias] = None

    @property
    def last_choch(self) -> Optional[StructureEvent]:
        chochs = [e for e in self.structure_events if e.type == StructureEventType.CHOCH]
        return chochs[-1] if chochs else None


def build_structure(df: pd.DataFrame, swing_strength: int = 1) -> StructureSnapshot:
    """Single entry point: run the full structure pipeline over an OHLC frame."""
    swings = get_swing_points(df, strength=swing_strength)
    itps = get_intermediate_points(df, swings)
    fvgs = detect_fvgs(df)
    update_fvg_states(df, fvgs)
    pd_array = htf_dealing_range(itps)
    ranked = rank_pd_arrays(pd_array, fvgs) if pd_array else {"discount": [], "premium": []}

    alt_swings = alternate_swings(swings)
    label_swing_strength(alt_swings)
    structure_events, trend = detect_structure_events(df, alt_swings)

    return StructureSnapshot(
        pd_array=pd_array, fvgs=fvgs, swings=swings,
        intermediate_points=itps, ranked=ranked,
        alt_swings=alt_swings, structure_events=structure_events, trend=trend,
    )


if __name__ == "__main__":
    # Minimal smoke test on synthetic data so the module can be sanity
    # checked without an MT5 connection.
    rng = pd.date_range("2026-01-01", periods=60, freq="4h")
    rs = np.random.default_rng(7)
    close = 2000 + np.cumsum(rs.normal(0, 3, size=len(rng)))
    o = close + rs.normal(0, 0.5, size=len(rng))
    h = np.maximum(o, close) + rs.uniform(0.5, 2, size=len(rng))
    l = np.minimum(o, close) - rs.uniform(0.5, 2, size=len(rng))
    demo = pd.DataFrame({"open": o, "high": h, "low": l, "close": close}, index=rng)

    snap = build_structure(demo)
    print(f"Swings: {len(snap.swings)} | ITH/ITL confirmed: {len(snap.intermediate_points)}")
    print(f"FVGs detected: {len(snap.fvgs)}")
    for state in FVGState:
        print(f"  {state.value}: {sum(1 for g in snap.fvgs if g.state == state)}")
    if snap.pd_array:
        pa = snap.pd_array
        print(f"Dealing range: {pa.range_low:.2f} - {pa.range_high:.2f} (EQ {pa.equilibrium:.2f})")
        print(f"Discount FVGs: {len(snap.ranked['discount'])} | Premium FVGs: {len(snap.ranked['premium'])}")
    else:
        print("No confirmed HTF dealing range yet.")

    print(f"\nAlternating zigzag swings: {len(snap.alt_swings)}")
    weak = sum(1 for s in snap.alt_swings if s.strength == SwingStrength.WEAK)
    strong = sum(1 for s in snap.alt_swings if s.strength == SwingStrength.STRONG)
    print(f"  weak: {weak} | strong: {strong}")
    print(f"Structure events: {len(snap.structure_events)} | current trend: {snap.trend}")
    for e in snap.structure_events[-5:]:
        print(f"  {e.time} {e.type.value.upper():5s} {e.direction.value:8s} broke {e.broken_level:.2f}")
