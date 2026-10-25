"""Position-level corrective teacher; may override alpha teacher."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from .protocol import advice, classify_target


@dataclass(frozen=True)
class RecoveryTeacher:
    max_abs_exposure: float = 0.8
    stop_loss_fraction: float = 0.04
    max_drawdown: float = 0.15
    expert_id: str = "recovery_v1"

    def __post_init__(self) -> None:
        if (not all(isfinite(x) for x in (self.max_abs_exposure, self.stop_loss_fraction, self.max_drawdown))
                or not 0 < self.max_abs_exposure <= 1
                or not 0 < self.stop_loss_fraction < 1 or not 0 < self.max_drawdown < 1):
            raise ValueError("invalid frozen risk limits")

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        current = observation.account_value("net_exposure")
        dd = observation.account_value("drawdown")
        unreal = observation.account_value("unrealized_pnl_fraction")
        if dd >= self.max_drawdown:
            target, reason = 0., "MAX_DRAWDOWN"
        elif unreal <= -self.stop_loss_fraction and abs(current) > 1e-8:
            target, reason = 0., "POSITION_STOP_LOSS"
        elif abs(current) > self.max_abs_exposure:
            target, reason = (self.max_abs_exposure if current > 0 else -self.max_abs_exposure), "POSITION_OVERSIZE"
        else:
            target, reason = current, "POSITION_SAFE"
        return advice(
            observation, self.expert_id, classify_target(current, target), target,
            reason, hard_risk_override=reason != "POSITION_SAFE",
        )

    def intervention(self, observation: PointInTimeObservation) -> ExpertAdvice | None:
        result = self.advise(observation)
        if result.reason_codes == ("POSITION_SAFE",):
            return None
        return result
