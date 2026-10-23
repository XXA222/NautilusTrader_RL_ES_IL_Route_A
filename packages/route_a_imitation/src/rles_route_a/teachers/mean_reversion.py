"""Candidate range-only reversal expert, gated by causal regime score."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from .protocol import advice, classify_target


@dataclass(frozen=True)
class MeanReversionTeacher:
    z_entry: float = 2.
    size: float = 0.2
    range_max_abs_trend: float = 0.001
    expert_id: str = "mean_reversion_v1"

    def __post_init__(self) -> None:
        if (not all(isfinite(x) for x in (self.z_entry, self.size, self.range_max_abs_trend))
                or self.z_entry <= 0 or not 0 < self.size <= 1 or self.range_max_abs_trend <= 0):
            raise ValueError("invalid mean reversion parameters")

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        z = observation.market_value("zscore_1m")
        hourly = observation.market_value("trend_1h")
        current = observation.account_value("net_exposure")
        target = 0.
        if abs(hourly) < self.range_max_abs_trend:
            target = (-self.size if z >= self.z_entry else self.size if z <= -self.z_entry else 0.)
        return advice(observation, self.expert_id, classify_target(current, target), target,
                      "RANGE_REVERSION" if target != 0 else "NO_RANGE_EDGE")
