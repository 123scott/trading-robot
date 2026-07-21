# -*- coding: utf-8 -*-
"""
backtest_entries.py

Turns a stream of raw crossover signals into position-aware trade intents
using a simple long-only state machine: BUY on a golden cross while flat,
SELL (close the long) on a death cross while holding. Memory-based
downgrading to SKIP happens one layer up in trading_robot.py -- this module
only knows about strategy position state, never about memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.backtest_structures import CrossDirection, CrossSignal


class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionState(Enum):
    FLAT = "flat"
    LONG = "long"


@dataclass
class TradeIntent:
    action: Action
    price: float
    reason: str
    open_time: int


class PositionTracker:
    """Tracks long-only strategy state across the replay."""

    def __init__(self) -> None:
        self.state = PositionState.FLAT

    def next_intent(self, signal: CrossSignal) -> Optional[TradeIntent]:
        """Only golden crosses while flat and death crosses while long produce an intent."""
        if signal.direction == CrossDirection.GOLDEN and self.state == PositionState.FLAT:
            return TradeIntent(action=Action.BUY, price=signal.price, reason=signal.reason, open_time=signal.open_time)
        if signal.direction == CrossDirection.DEATH and self.state == PositionState.LONG:
            return TradeIntent(action=Action.SELL, price=signal.price, reason=signal.reason, open_time=signal.open_time)
        return None

    def apply(self, action: Action) -> None:
        """Commit a FINAL action (after any memory downgrade) to position state. SKIP never calls this."""
        if action == Action.BUY:
            self.state = PositionState.LONG
        elif action == Action.SELL:
            self.state = PositionState.FLAT
