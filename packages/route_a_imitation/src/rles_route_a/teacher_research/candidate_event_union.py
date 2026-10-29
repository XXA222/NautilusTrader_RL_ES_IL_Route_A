"""Deduplicate causal one-minute teacher proposals without losing attribution."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal, Sequence

from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.teachers_1m.contracts import TeacherSignal


@dataclass(frozen=True, slots=True)
class TeacherProposal:
    observation: PointInTimeObservation
    teacher_id: str
    side: Literal[-1, 1]
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.observation, PointInTimeObservation):
            raise TypeError("causal observation required")
        if (not isinstance(self.teacher_id, str) or not self.teacher_id.strip()
                or type(self.side) is not int or self.side not in (-1, 1)
                or not isinstance(self.reason_code, str) or not self.reason_code.strip()):
            raise ValueError("teacher, direction, and reason are required")
        o = self.observation
        if not o.fold_id or not o.source_id or not o.snapshot_id or not o.feature_spec_hash:
            raise ValueError("candidate source/fold/snapshot/schema identities required")


def candidate_event_id(observation: PointInTimeObservation, side: int) -> str:
    """Bind the complete point-in-time market and portfolio state."""
    material = {
        "fold_id": observation.fold_id, "source_id": observation.source_id,
        "feature_spec_hash": observation.feature_spec_hash,
        "snapshot_id": observation.snapshot_id, "episode_id": observation.episode_id,
        "decision_ts_ns": observation.decision_ts_ns, "side": side,
        "market": dict(observation.market),
        "feature_available_at_ns": dict(observation.feature_available_at_ns),
        "account": dict(observation.account), "pending_order_ids": observation.pending_order_ids,
        "exact_market_clocks_ns": observation.exact_market_clocks_ns,
    }
    return sha256(json.dumps(material, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateGroup:
    event_id: str
    observation: PointInTimeObservation
    side: Literal[-1, 1]
    teachers: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.observation, PointInTimeObservation)
                or type(self.side) is not int or self.side not in (-1, 1)
                or self.event_id != candidate_event_id(self.observation, self.side)):
            raise ValueError("candidate group identity does not match its observation and side")
        if (not isinstance(self.teachers, tuple) or not self.teachers
                or any(not isinstance(item, tuple) or len(item) != 2
                       or any(not isinstance(value, str) or not value.strip() for value in item)
                       for item in self.teachers)
                or len({item[0] for item in self.teachers}) != len(self.teachers)):
            raise ValueError("candidate group requires unique teacher reasons")


def group_teacher_proposals(proposals: Sequence[TeacherProposal]) -> tuple[CandidateGroup, ...]:
    """Group equal source/fold/snapshot/minute/side; reject mixed source scope."""
    rows = tuple(proposals)
    if any(not isinstance(row, TeacherProposal) for row in rows):
        raise TypeError("teacher proposals required")
    if not rows:
        return ()
    scopes = {(row.observation.fold_id, row.observation.source_id,
               row.observation.feature_spec_hash) for row in rows}
    if len(scopes) != 1:
        raise ValueError("candidate batch cannot mix source, fold, or feature schema")
    grouped: dict[tuple[int, Literal[-1, 1], str], list[TeacherProposal]] = {}
    snapshots: dict[int, str] = {}
    for row in rows:
        o = row.observation
        if snapshots.setdefault(o.decision_ts_ns, o.snapshot_id) != o.snapshot_id:
            raise ValueError("same decision clock has conflicting snapshot identities")
        grouped.setdefault((o.decision_ts_ns, row.side, o.snapshot_id), []).append(row)
    result = []
    for (timestamp, side, snapshot), members in sorted(grouped.items()):
        o = members[0].observation
        if any(member.observation != o for member in members):
            raise ValueError("same snapshot has conflicting observations")
        teachers = tuple(sorted((member.teacher_id, member.reason_code) for member in members))
        if len({member.teacher_id for member in members}) != len(members):
            raise ValueError("teacher proposed the same candidate more than once")
        event_id = candidate_event_id(o, side)
        result.append(CandidateGroup(event_id, o, side, teachers))
    return tuple(result)


def proposal_from_signal(
    observation: PointInTimeObservation, teacher_id: str, signal: TeacherSignal,
) -> TeacherProposal | None:
    """Convert an already evaluated 1m teacher signal without imposing HTF filters."""
    if not isinstance(signal, TeacherSignal):
        raise TypeError("validated teacher signal required")
    if signal.raw_signal not in ("LONG_ENTRY", "SHORT_ENTRY"):
        return None
    expected: Literal[-1, 1] = 1 if signal.raw_signal == "LONG_ENTRY" else -1
    if signal.side != expected:
        raise ValueError("teacher signal side disagrees with raw entry")
    return TeacherProposal(observation, teacher_id, expected, signal.reason_code)
