"""Meta-gated teacher using the existing native account and risk lifecycle."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from rles_route_a.contracts import ExpertAdvice, NativeActionTrace, PointInTimeObservation
from rles_route_a.teachers.protocol import advice
from rles_route_a.teachers_1m.registry import build_teacher_1m
from .candidate_collector import OneMinuteCandidateCollector
from .candidate_event_union import CandidateGroup
from .entry_meta_policy import EntryScore, FrozenEntryMetaPolicy


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    group: CandidateGroup
    score: EntryScore | None
    selected: bool
    rejection_override: str | None = None

    @property
    def disposition(self) -> str:
        if self.rejection_override is not None:
            return "REJECTED_" + self.rejection_override
        if self.score is None:
            return "REJECTED_UNSCORED"
        if self.selected:
            return "ACCEPTED_SELECTED"
        if self.score.action in ("LONG", "SHORT"):
            return "REJECTED_BY_ARBITRATION"
        return "REJECTED_" + self.score.reason


@dataclass(frozen=True, slots=True)
class MetaDecision:
    observation: PointInTimeObservation
    candidates: tuple[CandidateDecision, ...]
    action: ExpertAdvice
    owner_id: str | None
    candidate_error: str | None = None
    teacher_signals: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class NativeDecisionReceipt:
    decision_ts_ns: int
    action: ExpertAdvice
    trace: NativeActionTrace
    native_order_facts: Any


class MetaGatedTeacher:
    """One position, original teacher exits, host risk/execution and native receipts.

    Recovery intervention is mandatory and takes precedence over entry or hold.
    The host continues to own hard risk, order state, fills and the account.
    """

    expert_id = "meta_gated_teacher_v1"

    def __init__(self, policy: FrozenEntryMetaPolicy, *, recovery: Any):
        if not isinstance(policy, FrozenEntryMetaPolicy) or policy.margin_bps <= 0:
            raise ValueError("a frozen policy with positive edge margin is required")
        if policy.label_recipe[0] != "X1":
            raise ValueError("entry policy exit recipe must match the original X1 teachers")
        if not callable(getattr(recovery, "intervention", None)):
            raise TypeError("native recovery intervention is required")
        self.policy = policy
        self.recovery = recovery
        self.collector = OneMinuteCandidateCollector()
        self._teachers = {name: build_teacher_1m(name) for name in self.collector.teacher_ids}
        self._owner: str | None = None
        self._scope: tuple[str, str, str] | None = None
        self.decisions: list[MetaDecision] = []
        self.native_receipts: list[NativeDecisionReceipt] = []

    def reset(self, episode_id: str = "") -> None:
        self.collector.reset(episode_id)
        for teacher in self._teachers.values():
            teacher.reset(episode_id)
        self._owner = None
        self._scope = None
        self.decisions.clear()
        self.native_receipts.clear()

    def _missing_model_features(self, observation: PointInTimeObservation) -> tuple[str, ...]:
        return tuple(name for name in self.policy.feature_names
                     if (name.startswith("market.") and name[7:] not in observation.market)
                     or (name.startswith("account.") and name[8:] not in observation.account))

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        scope = (observation.episode_id, observation.source_id, observation.fold_id)
        current = observation.account_value("net_exposure")
        # Risk is evaluated before model, feature, scope, or ownership checks.
        override = self.recovery.intervention(observation)
        if override is not None:
            # Diagnostics must never prevent a hard-risk action. Record failures
            # explicitly instead of silently treating them as no candidates.
            failures = []
            try:
                collection = self.collector.collect(observation)
            except Exception as exc:
                collection = None
                failures.append(f"collection {type(exc).__name__}: {exc}")
            for name, teacher in self._teachers.items():
                try:
                    teacher.advise(observation)
                except Exception as exc:
                    failures.append(f"{name} {type(exc).__name__}: {exc}")
            scored = []
            for group in collection.groups if collection is not None else ():
                try:
                    score = (None if self._missing_model_features(observation)
                             else self.policy.score(group))
                except Exception as exc:
                    score = None
                    failures.append(f"score {group.event_id} {type(exc).__name__}: {exc}")
                scored.append(CandidateDecision(group, score, False, "HARD_RISK_OVERRIDE"))
            self.decisions.append(MetaDecision(observation, tuple(scored), override, self._owner,
                "; ".join(failures) or None, collection.teacher_signals if collection is not None else ()))
            return override
        if self._scope is not None and self._scope != scope:
            if abs(current) > 1e-8 or observation.pending_order_ids:
                raise ValueError("cannot change scope with an active native position or order")
            self._owner = None
        self._scope = scope
        if abs(current) > 1e-8 and self._owner is None:
            raise ValueError("native position has no verified originating teacher")
        collection = self.collector.collect(observation)
        native_advice = {name: teacher.advise(observation)
                         for name, teacher in self._teachers.items()}
        missing = self._missing_model_features(observation)
        scored = [(group, None if missing else self.policy.score(group)) for group in collection.groups]
        selected = None
        owner_not_ready = set()
        if observation.pending_order_ids:
            action = advice(observation, self.expert_id,
                            "HOLD" if abs(current) > 1e-8 else "FLAT",
                            current, "PENDING_ACK")
        elif abs(current) > 1e-8:
            action = native_advice[self._owner]
        else:
            self._owner = None
            eligible = [(group, score) for group, score in scored
                        if score is not None and score.action in ("LONG", "SHORT")]
            for group, score in eligible:
                expected = "OPEN_LONG" if group.side > 0 else "OPEN_SHORT"
                if native_advice[group.teachers[0][0]].intent != expected:
                    owner_not_ready.add(group.event_id)
            eligible = [(group, score) for group, score in eligible
                        if group.event_id not in owner_not_ready]
            if eligible:
                # Deterministic arbitration selects one group, never both directions.
                selected, score = max(eligible, key=lambda item:
                    (item[1].expected_net_bps, item[1].probability_positive, item[0].side))
                self._owner = selected.teachers[0][0]
                underlying = native_advice[self._owner]
                action = underlying
            else:
                action = advice(observation, self.expert_id, "FLAT", 0., "NO_ACCEPTED_CANDIDATE")
        self.decisions.append(MetaDecision(observation, tuple(
            CandidateDecision(group, score, selected is not None and group.event_id == selected.event_id,
                              "PENDING_ORDER" if observation.pending_order_ids else
                              "POSITION_OPEN" if abs(current) > 1e-8 else
                              "OWNER_NOT_READY" if group.event_id in owner_not_ready else
                              "FEATURE_NOT_READY" if score is None else None)
            for group, score in scored), action, self._owner, teacher_signals=collection.teacher_signals))
        return action

    def observe_native_action(self, action: ExpertAdvice, trace: NativeActionTrace,
                              native_order_facts: Any) -> None:
        if (not self.decisions or self.decisions[-1].observation.decision_ts_ns != trace.decision_ts_ns
                or action != self.decisions[-1].action
                or len(self.native_receipts) != len(self.decisions) - 1
                or (self.native_receipts
                    and self.native_receipts[-1].decision_ts_ns >= trace.decision_ts_ns)):
            raise ValueError("native receipt has no unique matching decision")
        self.native_receipts.append(NativeDecisionReceipt(trace.decision_ts_ns, action, trace,
                                                          native_order_facts))
        if self._owner is not None:
            self._teachers[self._owner].observe_native_action(action, trace, native_order_facts)


def write_meta_trace(teacher: MetaGatedTeacher, path: Any) -> None:
    """Persist full candidate, decision, native order/fill, and equity receipts."""
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass
    from decimal import Decimal
    from pathlib import Path
    import json

    def primitive(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, Mapping):
            return {str(k): primitive(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [primitive(v) for v in value]
        if is_dataclass(value):
            return {field.name: primitive(getattr(value, field.name)) for field in fields(value)}
        raise TypeError(f"unsupported native trace value: {type(value).__name__}")

    records = []
    for decision in teacher.decisions:
        row = primitive(decision)
        for candidate, encoded in zip(decision.candidates, row["candidates"]):
            encoded["disposition"] = candidate.disposition
        records.append({"record_type": "decision", **row})
    records.extend({"record_type": "native_receipt", **primitive(receipt)}
                   for receipt in teacher.native_receipts)
    encoded = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                  sort_keys=True) + "\n" for row in records)
    Path(path).write_text(encoded, encoding="utf-8")
