"""Frozen, causal teacher selection; NEVER select using future realized return."""
from __future__ import annotations
from dataclasses import dataclass, replace
from math import isfinite
from rles_route_a.contracts import ExpertAdvice, ExpertTeacher, Intent, PointInTimeObservation
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
        current = observation.account_value("net_exposure")
        if observation.pending_order_ids:
            # Native order state wins over a fresh signal. Keep the current
            # filled exposure target until the order/cancel acknowledgement is
            # observed and the account projection is refreshed.
            intent: Intent = "HOLD" if abs(current) > 1e-8 else "FLAT"
            return ExpertAdvice(
                "selector_v1", observation.decision_ts_ns, intent, current,
                ("PENDING_ACK",),
            )
        vol = observation.market_value("vol_1m")
        if self.wait is not None and vol > self.max_active_vol and abs(current) <= 1e-8:
            # This is an entry veto, not a position management decision.
            # Existing positions remain under the primary teacher (and the
            # recovery override above).
            veto = self.wait.advise(observation)
            return replace(veto, reason_codes=("ENTRY_VETO_HIGH_VOL",))
        return self.primary.advise(observation)
