"""Independent X2 ATR trailing exit; native execution remains L3-owned."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from rles_route_a.contracts import PointInTimeObservation

from .contracts import TeacherSignal


@dataclass(slots=True)
class AtrTrailingExit:
    atr_multiplier: float = 3.0
    policy_id: str = "X2_ATR_TRAIL_3_V1"
    side: int = 0
    favorable_extreme: float | None = None
    trail: float | None = None
    last_close_time_ns: int | None = None

    required_fields = (
        "bar_high_1m",
        "bar_low_1m",
        "bar_close_1m",
        "atr_14",
    )

    def __post_init__(self) -> None:
        if (isinstance(self.atr_multiplier, bool)
                or not isinstance(self.atr_multiplier, (int, float))
                or not isfinite(self.atr_multiplier)
                or self.atr_multiplier <= 0):
            raise ValueError("ATR multiplier must be positive and finite")
        if not self.policy_id:
            raise ValueError("ATR exit policy id is required")

    def reset(self) -> None:
        self.side = 0
        self.favorable_extreme = None
        self.trail = None
        self.last_close_time_ns = None

    def evaluate(
        self, observation: PointInTimeObservation, current_exposure: float
    ) -> TeacherSignal | None:
        if abs(current_exposure) <= 1e-8:
            self.reset()
            return None
        if any(name not in observation.market for name in self.required_fields):
            return None
        side = 1 if current_exposure > 0 else -1
        close_time_ns = observation.market_clock_ns("latest_1m_close_ts_ns")
        high = observation.market_value("bar_high_1m")
        low = observation.market_value("bar_low_1m")
        close = observation.market_value("bar_close_1m")
        atr = observation.market_value("atr_14")
        if atr <= 0:
            raise ValueError("ATR exit requires a positive mature ATR")
        if self.side != side:
            self.side = side
            self.last_close_time_ns = close_time_ns
            self.favorable_extreme = high if side > 0 else low
            self.trail = (
                self.favorable_extreme - self.atr_multiplier * atr
                if side > 0
                else self.favorable_extreme + self.atr_multiplier * atr
            )
            return None
        if self.last_close_time_ns is not None and close_time_ns <= self.last_close_time_ns:
            raise ValueError("ATR exit must advance on a new completed 1m bar")
        self.last_close_time_ns = close_time_ns
        assert self.favorable_extreme is not None and self.trail is not None
        if side > 0:
            self.favorable_extreme = max(self.favorable_extreme, high)
            self.trail = max(
                self.trail,
                self.favorable_extreme - self.atr_multiplier * atr,
            )
            if close < self.trail:
                return TeacherSignal("EXIT_LONG", 0, "ATR_TRAIL_EXIT_LONG")
        else:
            self.favorable_extreme = min(self.favorable_extreme, low)
            self.trail = min(
                self.trail,
                self.favorable_extreme + self.atr_multiplier * atr,
            )
            if close > self.trail:
                return TeacherSignal("EXIT_SHORT", 0, "ATR_TRAIL_EXIT_SHORT")
        return None

