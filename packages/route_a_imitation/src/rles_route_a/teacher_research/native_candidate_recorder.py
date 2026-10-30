"""Record all eight causal candidates alongside an existing native teacher."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rles_route_a.contracts import ExpertAdvice, NativeActionTrace, PointInTimeObservation
from rles_route_a.teachers_1m.registry import ALL_TEACHER_RECIPE_BY_ID
from .candidate_collector import CandidateCollection, OneMinuteCandidateCollector


@dataclass(frozen=True, slots=True)
class RecordedCandidateBarrier:
    observation: PointInTimeObservation
    collection: CandidateCollection | None
    action: ExpertAdvice
    collection_error: str | None = None


class NativeCandidateRecorder:
    """A research sidecar; the delegated teacher and host retain all execution."""

    def __init__(self, teacher: Any):
        if not callable(getattr(teacher, "advise", None)):
            raise TypeError("an existing native teacher is required")
        self.teacher = teacher
        self.collector = OneMinuteCandidateCollector()
        self.records: list[RecordedCandidateBarrier] = []

    def reset(self, episode_id: str = "") -> None:
        self.collector.reset(episode_id)
        reset = getattr(self.teacher, "reset", None)
        if callable(reset):
            reset(episode_id)
        self.records.clear()

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        # Never prevent an existing risk/exit decision from reaching the host.
        action = self.teacher.advise(observation)
        try:
            collection = self.collector.collect(observation)
            error = None
        except Exception as exc:
            collection = None
            error = f"{type(exc).__name__}: {exc}"
        self.records.append(RecordedCandidateBarrier(observation, collection, action, error))
        return action

    def observe_native_action(self, action: ExpertAdvice, trace: NativeActionTrace, facts: Any) -> None:
        callback = getattr(getattr(self.teacher, "primary", self.teacher), "observe_native_action", None)
        if callable(callback):
            callback(action, trace, facts)

    def require_complete(self) -> tuple[RecordedCandidateBarrier, ...]:
        if not self.records or any(record.collection_error is not None or record.collection is None
                                   for record in self.records):
            raise ValueError("candidate collection has missing or failed native barriers")
        return tuple(self.records)


def summarize_teacher_signal_coverage(records, *, start_ns: int, end_ns: int):
    """Count all per-minute teacher signals, including indicator unavailability."""
    from collections import Counter
    if type(start_ns) is not int or type(end_ns) is not int or start_ns >= end_ns:
        raise ValueError("valid signal coverage interval required")
    counters = {}
    signal_counts = Counter()
    error_counts = Counter()
    barriers = 0
    for record in records:
        if not start_ns <= record.observation.decision_ts_ns < end_ns:
            continue
        if record.collection is None or record.collection_error is not None:
            raise ValueError("signal coverage requires complete candidate records")
        barriers += 1
        for teacher_id, raw, reason in record.collection.teacher_signals:
            raw_counts, reason_counts = counters.setdefault(teacher_id, (Counter(), Counter()))
            raw_counts[raw] += 1
            reason_counts[reason] += 1
            signal_counts[teacher_id] += 1
        for teacher_id, _error in record.collection.teacher_errors:
            error_counts[teacher_id] += 1
    expected_teachers = set(ALL_TEACHER_RECIPE_BY_ID)
    complete = (barriers > 0 and set(counters) == expected_teachers
                and all(signal_counts[name] == barriers for name in expected_teachers)
                and not error_counts)
    return {"barriers": barriers, "teachers": {teacher_id: {
        "raw_signal_counts": dict(sorted(raw.items())), "reason_counts": dict(sorted(reasons.items()))}
        for teacher_id, (raw, reasons) in sorted(counters.items())},
        "teacher_error_counts": dict(sorted(error_counts.items())),
        "complete_eight_teacher_coverage": complete}



def run_native_candidate_collection(
    *, provider, recorder: NativeCandidateRecorder, factory, host_rollout, grant,
    verify_grant, seed: int, rollout_id: str, episode_steps: int, reset_options=None,
):
    """Run the recorder through the host and require a complete M2.2 episode."""
    from rles_route_a.integration.gym_rollout import NativeGymTeacherProvider
    from rles_route_a.integration.scope import HostTrainGrant

    if not isinstance(provider, NativeGymTeacherProvider) or provider.teacher is not recorder:
        raise TypeError("native provider must own the candidate recorder")
    if (not isinstance(grant, HostTrainGrant) or not callable(verify_grant)
            or verify_grant(grant) is not True
            or (provider.fold_id, provider.source_identity) != (grant.fold_id, grant.source_identity)):
        raise PermissionError("candidate collection requires the matching verified Train grant")
    if (getattr(factory, "transition_hash", None) != provider.execution_recipe
            or not callable(host_rollout) or type(seed) is not int
            or type(episode_steps) is not int or episode_steps <= 0
            or not isinstance(rollout_id, str) or not rollout_id.strip()):
        raise ValueError("native candidate collection run identity is invalid")
    options = {"episode_steps": episode_steps}
    if reset_options is not None:
        options["reset_options"] = dict(reset_options)
    barriers, native = provider.rollout_with_m22_trace(
        factory=factory, host_rollout=host_rollout, grant=grant, seed=seed,
        decision_policy=None, rollout_id=rollout_id, **options)
    verification = native.get("route_a_native_order_verification")
    records = recorder.require_complete()
    if (len(barriers) != episode_steps or len(records) != len(barriers)
            or not isinstance(verification, dict)
            or verification.get("barriers") != len(barriers)
            or verification.get("authority") != "retained_m22_session_snapshots"
            or any(row.observation.decision_ts_ns != barrier.trace.decision_ts_ns
                   for row, barrier in zip(records, barriers))):
        raise ValueError("native candidate collection lacks complete matching M2.2 barriers")
    return barriers, native, records



def write_candidate_records(recorder: NativeCandidateRecorder, path) -> None:
    """Persist every causal snapshot, all teacher states, and every candidate."""
    import json
    from pathlib import Path

    rows = []
    for record in recorder.require_complete():
        obs = record.observation
        assert record.collection is not None
        rows.append({
            "evidence_type": "NATIVE_OBSERVATION_RESEARCH_CANDIDATES",
            "episode_id": obs.episode_id, "decision_ts_ns": obs.decision_ts_ns,
            "source_id": obs.source_id, "fold_id": obs.fold_id, "snapshot_id": obs.snapshot_id,
            "feature_spec_hash": obs.feature_spec_hash, "market": dict(obs.market),
            "feature_available_at_ns": dict(obs.feature_available_at_ns),
            "exact_market_clocks_ns": obs.exact_market_clocks_ns, "account": dict(obs.account),
            "pending_order_ids": obs.pending_order_ids,
            "teacher_signals": record.collection.teacher_signals,
            "teacher_errors": record.collection.teacher_errors,
            "candidates": [{"event_id": group.event_id, "side": group.side,
                            "teachers": group.teachers} for group in record.collection.groups],
            "native_advice": {"expert_id": record.action.expert_id, "intent": record.action.intent,
                "requested_exposure": record.action.requested_exposure,
                "reason_codes": record.action.reason_codes},
        })
    encoded = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                  allow_nan=False, separators=(",", ":")) + "\n" for row in rows)
    with Path(path).open("x", encoding="utf-8", newline="\n") as output:
        output.write(encoded)
