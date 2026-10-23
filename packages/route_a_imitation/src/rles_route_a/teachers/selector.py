"""Frozen, causal teacher selection; NEVER select using future realized return."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from rles_route_a.contracts import ExpertAdvice, ExpertTeacher, PointInTimeObservation
from .recovery import RecoveryTeacher


@dataclass(frozen=True)
class FrozenTeacherSelector:
    primary: ExpertTeacher
    recovery: RecoveryTeacher
    max_active_vol: float = 0.08
    wait: ExpertTeacher | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.max_active_vol) or self.max_active_vol <= 0:
            raise ValueError("invalid teacher selector volatility threshold")

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        override = self.recovery.intervention(observation)
        if override is not None:
            return override
        vol = observation.market_value("vol_1m")
        if self.wait is not None and vol > self.max_active_vol:
            return self.wait.advise(observation)
        return self.primary.advise(observation)
