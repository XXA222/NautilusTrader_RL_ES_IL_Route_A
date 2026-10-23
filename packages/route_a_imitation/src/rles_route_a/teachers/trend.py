"""Causal multi-timeframe trend candidate; NOT a proven profitable teacher."""

from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from .protocol import advice, classify_target


@dataclass(frozen=True)
class TrendTeacher:
    size: float = 0.5
    threshold: float = 0.002
    max_volatility: float = 0.08
    expert_id: str = "trend_v1"
    hold_on_neutral: bool = False
    direction: int = 1
    regime_strength_threshold: float | None = None

    def __post_init__(self) -> None:
        if (
            not all(isfinite(x) for x in (self.size, self.threshold, self.max_volatility))
            or not 0 < self.size <= 1
            or self.threshold < 0
            or self.max_volatility <= 0
            or type(self.direction) is not int
            or self.direction not in {-1, 1}
            or type(self.hold_on_neutral) is not bool
            or (
                self.regime_strength_threshold is not None
                and (
                    not isfinite(self.regime_strength_threshold)
                    or self.regime_strength_threshold <= 0
                )
            )
        ):
            raise ValueError("invalid frozen trend parameters")

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        hourly = observation.market_value("trend_1h")
        daily = observation.market_value("trend_1d")
        volatility = observation.market_value("vol_1m")
        current = observation.account_value("net_exposure")
        direction = self.direction
        if self.regime_strength_threshold is not None:
            aligned_strength = min(abs(hourly), abs(daily))
            direction = 1 if aligned_strength >= self.regime_strength_threshold else -1
        if volatility > self.max_volatility:
            target, reason = 0.0, "VOLATILITY_LIMIT"
        elif hourly > self.threshold and daily > self.threshold:
            target = self.size * direction
            reason = "MTF_TREND_UP" if direction == 1 else "MTF_TREND_DOWN"
        elif hourly < -self.threshold and daily < -self.threshold:
            target = -self.size * direction
            reason = "MTF_TREND_DOWN" if direction == 1 else "MTF_TREND_UP"
        elif self.hold_on_neutral and current != 0.0:
            target, reason = current, "HOLD_NEUTRAL_TREND"
        else:
            target, reason = 0.0, "NO_TREND_ALIGNMENT"
        return advice(observation, self.expert_id, classify_target(current, target), target, reason)
