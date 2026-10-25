"""Teacher decision helpers; the true account and execution remain L3-owned."""
from __future__ import annotations
from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation, Intent


def advice(obs: PointInTimeObservation, teacher: str, intent: Intent,
           target: float, *reason: str, horizon: int | None = None,
           hard_risk_override: bool = False) -> ExpertAdvice:
    current = obs.account_value("net_exposure")
    if obs.pending_order_ids and not hard_risk_override:
        # All teachers share the same pending-order guard, including when they
        # are called directly without FrozenTeacherSelector.
        intent = "HOLD" if abs(current) > 1e-8 else "FLAT"
        target = current if abs(current) > 1e-8 else 0.0
        reason = ("PENDING_ACK",)
    if current * target < -1e-8:
        # A netting-account reversal is two distinct native transitions.
        # First request a confirmed close; a later barrier may open the new
        # side after the account projection reports flat.
        intent, target = "EXIT", 0.0
        reason = ("REVERSE_CLOSE_FIRST", *reason)
    return ExpertAdvice(teacher, obs.decision_ts_ns, intent, target,
                        tuple(reason), horizon_hint_bars=horizon)


def classify_target(current: float, target: float, *, eps: float = 1e-6) -> Intent:
    if abs(target) <= eps:
        return "EXIT" if abs(current) > eps else "FLAT"
    if abs(current) <= eps or current * target < 0:
        return "OPEN_LONG" if target > 0 else "OPEN_SHORT"
    if abs(target - current) <= eps:
        return "HOLD"
    return "ADD" if abs(target) > abs(current) else "REDUCE"
