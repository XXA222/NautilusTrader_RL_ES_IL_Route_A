"""Candidate breakout expert requiring externally computed causal features."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from .protocol import advice, classify_target


@dataclass(frozen=True)
class BreakoutTeacher:
    threshold: float = 0.015
    size: float = 0.3
    expert_id: str = "breakout_v1"

    def __post_init__(self) -> None:
        if (not all(isfinite(x) for x in (self.threshold, self.size))
                or self.threshold < 0 or not 0 < self.size <= 1):
            raise ValueError("invalid breakout parameters")

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        signal = observation.market_value("breakout_1h")
        current = observation.account_value("net_exposure")
        target = self.size if signal > self.threshold else -self.size if signal < -self.threshold else 0.
        return advice(observation, self.expert_id, classify_target(current, target), target,
                      "BREAKOUT_POS" if target > 0 else "BREAKOUT_NEG" if target < 0 else "NO_BREAKOUT")
