"""Meta-gated teacher using the existing native account and risk lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rles_route_a.contracts import ExpertAdvice, NativeActionTrace, PointInTimeObservation
from rles_route_a.teachers.protocol import advice
from rles_route_a.teachers_1m.registry import build_teacher_1m
from .candidate_collector import OneMinuteCandidateCollector
from .candidate_event_union import CandidateGroup
from .entry_meta_policy import EntryScore, FrozenEntryMetaPolicy, missing_candidate_features


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    group: CandidateGroup
    score: EntryScore | None
    selected: bool
    rejection_override: str | None = None
    missing_features: tuple[str, ...] = ()
    error_detail: str | None = None

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
    teacher_errors: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class NativeDecisionReceipt:
    decision_ts_ns: int
    action: ExpertAdvice
    trace: NativeActionTrace
    native_order_facts: Any
    selected_candidate_event_id: str | None = None
    selected_teacher_id: str | None = None


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
        self._scope: tuple[str, str, str, str] | None = None
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

    def _reset_signal_states(self, episode_id: str) -> None:
        """Reset rolling teacher state without discarding the native audit trace."""
        self.collector.reset(episode_id)
        for teacher in self._teachers.values():
            teacher.reset(episode_id)
        self._owner = None

    def _missing_model_features(self, observation: PointInTimeObservation) -> tuple[str, ...]:
        return missing_candidate_features(observation, self.policy.feature_names)

    def _model_rejection(self, group: CandidateGroup,
                         missing: tuple[str, ...]) -> str | None:
        observation = group.observation
        if (observation.source_id, observation.fold_id, observation.feature_spec_hash) != (
                self.policy.source_id, self.policy.fold_id, self.policy.feature_spec_hash):
            return "MODEL_SCOPE_MISMATCH"
        if observation.decision_ts_ns <= self.policy.available_after_ns:
            return "MODEL_NOT_AVAILABLE"
        if missing:
            return "FEATURE_NOT_READY"
        return None

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        scope = (observation.episode_id, observation.source_id,
                 observation.fold_id, observation.feature_spec_hash)
        current = observation.account_value("net_exposure")
        # Risk is evaluated before scope and ownership checks so a scope error
        # cannot suppress a mandatory native recovery action.
        override = self.recovery.intervention(observation)
        scope_changed = self._scope is not None and self._scope != scope
        has_native_state = abs(current) > 1e-8 or bool(observation.pending_order_ids)
        if scope_changed and has_native_state:
            if override is None:
                raise ValueError("cannot change episode/source/fold/schema with an active native position or order")
            # Never advance rolling teacher state with a different feature or
            # data identity while a native position/order still belongs to the
            # old scope. Preserve the risk action and make the skipped trace
            # diagnostics explicit.
            self.decisions.append(MetaDecision(
                observation, (), override, self._owner,
                "scope transition with active native state; candidate diagnostics skipped",
            ))
            return override
        if scope_changed:
            self._reset_signal_states(observation.episode_id)
        self._scope = scope
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
            risk_scored: list[CandidateDecision] = []
            missing = self._missing_model_features(observation)
            for group in collection.groups if collection is not None else ():
                try:
                    score = (None if (collection is None or collection.teacher_errors
                                      or self._model_rejection(group, missing) is not None)
                             else self.policy.score(group))
                except Exception as exc:
                    score = None
                    failures.append(f"score {group.event_id} {type(exc).__name__}: {exc}")
                risk_scored.append(CandidateDecision(group, score, False, "HARD_RISK_OVERRIDE", missing))
            self.decisions.append(MetaDecision(observation, tuple(risk_scored), override, self._owner,
                "; ".join(failures) or None,
                collection.teacher_signals if collection is not None else (),
                collection.teacher_errors if collection is not None else ()))
            return override
        if abs(current) > 1e-8 and self._owner is None:
            raise ValueError("native position has no verified originating teacher")
        collection = None
        collection_error = None
        try:
            collection = self.collector.collect(observation)
        except Exception as exc:
            # Candidate generation is required before opening a new position,
            # but a research-side failure must not suppress an existing exit.
            collection_error = f"{type(exc).__name__}: {exc}"
        native_advice = {}
        owner_errors = {}
        for name, teacher in self._teachers.items():
            try:
                native_advice[name] = teacher.advise(observation)
            except Exception as exc:
                owner_errors[name] = f"{type(exc).__name__}: {exc}"
        missing = self._missing_model_features(observation)
        scored: list[tuple[CandidateGroup, EntryScore | None]] = []
        pre_score_rejections: dict[str, str] = {}
        scoring_errors: dict[str, str] = {}
        for group in collection.groups if collection is not None else ():
            if collection is not None and collection.teacher_errors:
                scored.append((group, None))
                pre_score_rejections[group.event_id] = "TEACHER_SIGNAL_INCOMPLETE"
                continue
            rejection = self._model_rejection(group, missing)
            if rejection is not None:
                scored.append((group, None))
                pre_score_rejections[group.event_id] = rejection
                continue
            try:
                score = self.policy.score(group)
            except Exception as exc:
                # Keep the failed candidate in the denominator, reject it, and
                # allow unrelated candidates and native exits to proceed.
                score = None
                pre_score_rejections[group.event_id] = "SCORING_ERROR"
                scoring_errors[group.event_id] = f"{type(exc).__name__}: {exc}"
            scored.append((group, score))
        selected: CandidateGroup | None = None
        owner_not_ready: set[str] = set()
        for group, score in scored:
            expected_score = "LONG" if group.side > 0 else "SHORT"
            if (score is not None and score.action in ("LONG", "SHORT")
                    and score.action != expected_score):
                pre_score_rejections[group.event_id] = "MODEL_DIRECTION_MISMATCH"
        if observation.pending_order_ids:
            action = advice(observation, self.expert_id,
                            "HOLD" if abs(current) > 1e-8 else "FLAT",
                            current, "PENDING_ACK")
        elif abs(current) > 1e-8:
            owner = self._owner
            if owner is None:
                raise ValueError("native position has no verified originating teacher")
            if owner in owner_errors:
                raise RuntimeError(
                    f"originating teacher {owner} failed while managing a native position: "
                    f"{owner_errors[owner]}"
                )
            action = native_advice[owner]
        else:
            self._owner = None
            eligible = [(group, score) for group, score in scored
                        if score is not None
                        and score.action == ("LONG" if group.side > 0 else "SHORT")]
            executable_owner_by_event: dict[str, str] = {}
            for group, score in eligible:
                expected = "OPEN_LONG" if group.side > 0 else "OPEN_SHORT"
                ready_owners = tuple(
                    teacher_id for teacher_id, _reason in group.teachers
                    if (teacher_id in native_advice
                        and native_advice[teacher_id].intent == expected)
                )
                if not ready_owners:
                    owner_not_ready.add(group.event_id)
                else:
                    # Consensus is a single candidate, but any contributing
                    # teacher whose native lifecycle is ready may own execution.
                    executable_owner_by_event[group.event_id] = ready_owners[0]
            eligible = [(group, score) for group, score in eligible
                        if group.event_id not in owner_not_ready]
            if collection is None:
                action = advice(observation, self.expert_id, "FLAT", 0.,
                                "CANDIDATE_COLLECTION_FAILED")
            elif collection.teacher_errors:
                action = advice(observation, self.expert_id, "FLAT", 0.,
                                "TEACHER_SIGNAL_INCOMPLETE")
            elif eligible:
                # Deterministic arbitration selects one group, never both directions.
                selected, score = max(eligible, key=lambda item:
                    (item[1].expected_net_bps, item[1].probability_positive, item[0].side))
                self._owner = executable_owner_by_event[selected.event_id]
                underlying = native_advice[self._owner]
                action = underlying
            else:
                action = advice(observation, self.expert_id, "FLAT", 0., "NO_ACCEPTED_CANDIDATE")
        decisions = []
        for group, score in scored:
            owner_failures = tuple(
                f"{teacher_id}: {owner_errors[teacher_id]}"
                for teacher_id, _reason in group.teachers if teacher_id in owner_errors
            )
            signal_failures = tuple(
                f"{teacher_id}: {detail}"
                for teacher_id, detail in (collection.teacher_errors if collection is not None else ())
            )
            rejection = pre_score_rejections.get(group.event_id)
            if rejection is None and observation.pending_order_ids:
                rejection = "PENDING_ORDER"
            elif rejection is None and abs(current) > 1e-8:
                rejection = "POSITION_OPEN"
            elif rejection is None and group.event_id in owner_not_ready:
                rejection = "OWNER_EVALUATION_ERROR" if owner_failures else "OWNER_NOT_READY"
            elif rejection is None and score is None:
                rejection = "FEATURE_NOT_READY"
            decisions.append(CandidateDecision(
                group, score, selected is not None and group.event_id == selected.event_id,
                rejection, missing,
                scoring_errors.get(group.event_id) or "; ".join(owner_failures or signal_failures) or None,
            ))
        self.decisions.append(MetaDecision(
            observation, tuple(decisions), action, self._owner,
            candidate_error=collection_error,
            teacher_signals=collection.teacher_signals if collection is not None else (),
            teacher_errors=collection.teacher_errors if collection is not None else (),
        ))
        return action

    def observe_native_action(self, action: ExpertAdvice, trace: NativeActionTrace,
                              native_order_facts: Any) -> None:
        decision = self.decisions[-1] if self.decisions else None
        if (decision is None or decision.observation.decision_ts_ns != trace.decision_ts_ns
                or action != decision.action
                or len(self.native_receipts) != len(self.decisions) - 1
                or (self.native_receipts
                    and self.native_receipts[-1].decision_ts_ns >= trace.decision_ts_ns)):
            raise ValueError("native receipt has no unique matching decision")
        selected = tuple(candidate for candidate in decision.candidates if candidate.selected)
        if len(selected) > 1:
            raise ValueError("native decision cannot bind multiple selected candidates")
        opening_side = 1 if action.intent == "OPEN_LONG" else -1 if action.intent == "OPEN_SHORT" else None
        if ((opening_side is None and selected)
                or (opening_side is not None
                    and (len(selected) != 1 or selected[0].group.side != opening_side))):
            raise ValueError("native entry receipt must bind its single direction-matched candidate")
        selected_id = selected[0].group.event_id if selected else None
        selected_teacher = decision.owner_id if selected else None
        if (selected and (selected_teacher is None
                or selected_teacher not in {name for name, _reason in selected[0].group.teachers})):
            raise ValueError("native entry owner must be one of the selected candidate's teachers")
        self.native_receipts.append(NativeDecisionReceipt(trace.decision_ts_ns, action, trace,
            native_order_facts, selected_id, selected_teacher))
        decision_scope = (
            decision.observation.episode_id, decision.observation.source_id,
            decision.observation.fold_id, decision.observation.feature_spec_hash,
        )
        if self._owner is not None and self._scope == decision_scope:
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
