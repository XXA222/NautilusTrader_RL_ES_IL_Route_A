"""Deterministic candidate events from already-causal Route A observations."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.contracts import ExpertAdvice, NativeActionTrace

CandidateFamily = Literal["breakout", "pullback"]


@dataclass(frozen=True, slots=True)
class OpportunityEvent:
    """A point-in-time candidate, not an order or a realized trade label."""

    event_id: str
    decision_ts_ns: int
    feature_available_at_ns: int
    side: Literal[-1, 1]
    family: CandidateFamily
    fold_id: str
    source_identity: str
    rule_version: str
    snapshot_id: str
    feature_spec_hash: str

    def __post_init__(self) -> None:
        if (not isinstance(self.event_id, str) or len(self.event_id) != 64
                or any(ch not in "0123456789abcdef" for ch in self.event_id)):
            raise ValueError("candidate event id must be a lowercase SHA-256 identity")
        if (type(self.decision_ts_ns) is not int or self.decision_ts_ns <= 0
                or type(self.feature_available_at_ns) is not int
                or not 0 <= self.feature_available_at_ns <= self.decision_ts_ns):
            raise ValueError("candidate event timestamps are invalid or noncausal")
        if self.side not in {-1, 1} or self.family not in {"breakout", "pullback"}:
            raise ValueError("candidate side or family is invalid")
        if any(not isinstance(value, str) or not value.strip() for value in (
                self.fold_id, self.source_identity, self.rule_version)):
            raise ValueError("candidate fold/source/rule identity is required")
        for name, value in (("snapshot_id", self.snapshot_id),
                            ("feature_spec_hash", self.feature_spec_hash)):
            if (not isinstance(value, str) or len(value) != 64
                    or any(ch not in "0123456789abcdef" for ch in value)):
                raise ValueError(f"candidate {name} must be a lowercase SHA-256 identity")


@dataclass(frozen=True, slots=True)
class CausalOpportunityGenerator:
    """Small frozen G-A/G-B candidate generator with causal feature checks.

    Breakout uses a completed-bar breakout feature aligned with the slower
    1d trend. Pullback requires aligned 1h/1d trends plus a signed z-score and
    efficiency. It emits the raw candidate universe; position, volatility,
    risk, cost, and order-state vetoes belong downstream so rejected
    candidates remain measurable.
    """

    breakout_threshold: float = 0.015
    pullback_z_threshold: float = 1.0
    minimum_efficiency: float = 0.2
    rule_version: str = "route_a_candidates_v2"
    require_recovery_confirmation: bool = False

    def __post_init__(self) -> None:
        from math import isfinite

        values = (self.breakout_threshold, self.pullback_z_threshold,
                  self.minimum_efficiency)
        if (not all(isfinite(value) for value in values)
                or self.breakout_threshold < 0
                or self.pullback_z_threshold <= 0
                or not 0 <= self.minimum_efficiency <= 1
                or not self.rule_version.strip()):
            raise ValueError("invalid frozen opportunity-generation recipe")
        if (type(self.require_recovery_confirmation) is not bool
                or (self.require_recovery_confirmation and self.rule_version == "route_a_candidates_v2")):
            raise ValueError("recovery confirmation requires a separately versioned candidate recipe")

    def candidates(self, observation: PointInTimeObservation) -> tuple[OpportunityEvent, ...]:
        # Missing/stale features fail closed through market_value() and the
        # PointInTimeObservation constructor's availability-time validation.
        # Volatility is intentionally not a generator filter: consumers can
        # preserve candidate coverage and record a downstream risk veto.

        if (not observation.source_id or not observation.snapshot_id
                or not observation.feature_spec_hash):
            raise ValueError(
                "candidate provenance requires source, snapshot, and feature-spec identities"
            )
        found: list[tuple[CandidateFamily, Literal[-1, 1], tuple[str, ...]]] = []
        breakout = observation.market_value("breakout_1h")
        trend_1d = observation.market_value("trend_1d")
        if breakout > self.breakout_threshold:
            if trend_1d > 0:
                found.append(("breakout", 1, ("breakout_1h", "trend_1d")))
        elif breakout < -self.breakout_threshold:
            if trend_1d < 0:
                found.append(("breakout", -1, ("breakout_1h", "trend_1d")))

        trend = observation.market_value("trend_1h")
        zscore = observation.market_value("zscore_1m")
        efficiency = observation.market_value("efficiency_15m")
        # v2 remains reproducible. The versioned recovery mechanism treats the
        # host's signed efficiency as a magnitude and requires a countertrend
        # 15m move followed by an aligned 5m recovery, both already closed.
        recovery_long = recovery_short = True
        recovery_names: tuple[str, ...] = ()
        if self.require_recovery_confirmation:
            recent = observation.market_value("return_5m")
            pullback = observation.market_value("return_15m")
            recovery_long = pullback < 0 < recent
            recovery_short = recent < 0 < pullback
            recovery_names = ("return_5m", "return_15m")
            efficiency = abs(efficiency)
        if efficiency >= self.minimum_efficiency:
            # Buy a pullback inside a positive trend; sell a rally inside a
            # negative trend. The feature definitions/units remain host-owned.
            if (trend > 0 and trend_1d > 0
                    and zscore <= -self.pullback_z_threshold and recovery_long):
                found.append(("pullback", 1,
                              ("trend_1h", "trend_1d", "zscore_1m",
                               "efficiency_15m") + recovery_names))
            elif (trend < 0 and trend_1d < 0
                    and zscore >= self.pullback_z_threshold and recovery_short):
                found.append(("pullback", -1,
                              ("trend_1h", "trend_1d", "zscore_1m",
                               "efficiency_15m") + recovery_names))

        events: list[OpportunityEvent] = []
        for family, side, feature_names in found:
            available = max(observation.feature_available_at_ns[name]
                            for name in feature_names)
            identity = (f"{observation.fold_id}:{observation.source_id}:"
                        f"{observation.snapshot_id}:{observation.feature_spec_hash}:"
                        f"{observation.decision_ts_ns}:"
                        f"{family}:{side}:{self.rule_version}")
            event_id = sha256(identity.encode("utf-8")).hexdigest()
            events.append(OpportunityEvent(
                event_id=event_id,
                decision_ts_ns=observation.decision_ts_ns,
                feature_available_at_ns=available,
                side=side,
                family=family,
                fold_id=observation.fold_id,
                source_identity=observation.source_id,
                rule_version=self.rule_version,
                snapshot_id=observation.snapshot_id,
                feature_spec_hash=observation.feature_spec_hash,
            ))
        return tuple(events)


def audit_candidate_decision(event: OpportunityEvent, advice: ExpertAdvice,
                             trace: NativeActionTrace) -> dict[str, object]:
    """Join one raw candidate to the actual decision and native action trace.

    The fields intentionally distinguish teacher intent, risk/planner targets,
    and actual post-fill exposure. This is provenance, not a realized trade
    outcome or proof that the candidate was economically good.
    """
    if advice.decision_ts_ns != event.decision_ts_ns or trace.decision_ts_ns != event.decision_ts_ns:
        raise ValueError("candidate, advice, and native trace clocks do not match")
    side_selected = (
        advice.intent in {"OPEN_LONG", "OPEN_SHORT"}
        and (1 if advice.requested_exposure > 0 else -1) == event.side
    )
    return {
        "event_id": event.event_id,
        "decision_ts_ns": event.decision_ts_ns,
        "feature_available_at_ns": event.feature_available_at_ns,
        "side": event.side,
        "family": event.family,
        "fold_id": event.fold_id,
        "source_identity": event.source_identity,
        "snapshot_id": event.snapshot_id,
        "feature_spec_hash": event.feature_spec_hash,
        "rule_version": event.rule_version,
        "teacher_expert_id": advice.expert_id,
        "teacher_intent": advice.intent,
        "teacher_reason_codes": list(advice.reason_codes),
        "teacher_requested_exposure": advice.requested_exposure,
        "teacher_selected_candidate_side": side_selected,
        "risk_adjusted_target": trace.risk_adjusted_target,
        "planner_target": trace.planner_target,
        "actual_exposure_after": trace.actual_exposure_after,
        "submitted_order_ids": list(trace.submitted_order_ids),
        "filled_order_ids": list(trace.filled_order_ids),
        "pending_order_ids_after": list(trace.pending_order_ids_after),
        "scenario_only": trace.scenario_only,
    }
