"""Teacher decision helpers; the true account and execution remain L3-owned."""
from __future__ import annotations
from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation, Intent


def advice(obs: PointInTimeObservation, teacher: str, intent: Intent,
           target: float, *reason: str, horizon: int | None = None) -> ExpertAdvice:
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
