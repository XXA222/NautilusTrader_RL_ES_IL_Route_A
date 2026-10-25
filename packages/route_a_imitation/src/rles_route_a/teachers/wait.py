"""Explicit flat / no-alpha teacher. Valid for risk and no-trade baselines."""
from __future__ import annotations
from dataclasses import dataclass
from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from .protocol import advice, classify_target


@dataclass(frozen=True)
class WaitTeacher:
    expert_id: str = "wait_v1"

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        current = observation.account_value("net_exposure")
        return advice(observation, self.expert_id, classify_target(current, 0.), 0., "NO_TRUSTED_ALPHA")
