"""Policy adapter sends ONLY a normalized target to existing L3 execution.

No local planner, order matching, risk guard, or account mutation is permitted.
The host owns zero_target_policy=close and the cancel/ACK behavior.
"""
from __future__ import annotations
from typing import Protocol
import math
from rles_route_a.contracts import PointInTimeObservation, RouteAPolicy


class NativePolicySink(Protocol):
    def submit_normalized_target(self, *, decision_ts_ns: int, target_exposure: float,
                                 policy_id: str, action_schema: str) -> object: ...


class L3ActorAdapter:
    def __init__(self, *, model: RouteAPolicy, sink: NativePolicySink,
                 policy_id: str, zero_target_policy: str):
        if zero_target_policy != "close":
            raise ValueError("Route A requires explicit policy flat-close semantics in host L3")
        if not policy_id:
            raise ValueError("policy identity required")
        self.model, self.sink, self.policy_id = model, sink, policy_id

    def reset(self, episode_id: str) -> None:
        self.model.reset(episode_id)

    def step(self, observation: PointInTimeObservation) -> object:
        action = float(self.model.predict_exposure(observation))
        if not math.isfinite(action) or not -1 <= action <= 1:
            raise ValueError("actor proposed invalid exposure")
        return self.sink.submit_normalized_target(
            decision_ts_ns=observation.decision_ts_ns,
            target_exposure=action, policy_id=self.policy_id,
            action_schema="single_exposure_v1")
