"""Causal, account-independent collection of the eight original 1m entry signals.

The flat view is exclusively an input to isolated signal state machines; it is
never submitted to execution. Every returned candidate retains the real native
account, pending orders, market snapshot and identities.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.teachers_1m.registry import ALL_TEACHER_RECIPE_BY_ID, build_teacher_1m
from .candidate_event_union import CandidateGroup, group_teacher_proposals, proposal_from_signal


@dataclass(frozen=True, slots=True)
class CandidateCollection:
    groups: tuple[CandidateGroup, ...]
    teacher_signals: tuple[tuple[str, str, str], ...]


class OneMinuteCandidateCollector:
    """Run independent original X1 signal states without simulating account fills."""

    def __init__(self) -> None:
        self._teachers = tuple(
            build_teacher_1m(teacher_id) for teacher_id in sorted(ALL_TEACHER_RECIPE_BY_ID)
        )
        self._scope: tuple[str, str, str, str] | None = None

    @property
    def teacher_ids(self) -> tuple[str, ...]:
        return tuple(teacher.expert_id for teacher in self._teachers)

    def reset(self, episode_id: str = "") -> None:
        for teacher in self._teachers:
            teacher.reset(episode_id)
        self._scope = None

    def collect(self, observation: PointInTimeObservation) -> CandidateCollection:
        if not isinstance(observation, PointInTimeObservation):
            raise TypeError("point-in-time observation required")
        if not all((observation.fold_id, observation.source_id, observation.snapshot_id,
                    observation.feature_spec_hash)):
            raise ValueError("candidate source/fold/snapshot/schema identities required")
        scope = (observation.episode_id, observation.source_id, observation.fold_id,
                 observation.feature_spec_hash)
        if self._scope is not None and self._scope != scope:
            self.reset(observation.episode_id)
        self._scope = scope
        observation.account_value("net_exposure")
        # Flat is a signal-evaluation convention only; the proposal is bound to
        # observation, never this derived view, so blocked opportunities remain visible.
        account = dict(observation.account)
        account["net_exposure"] = 0.0
        signal_view = replace(observation, account=account, pending_order_ids=())
        proposals = []
        signals = []
        for teacher in self._teachers:
            teacher.advise(signal_view)
            signal = teacher.last_signal
            signals.append((teacher.expert_id, signal.raw_signal, signal.reason_code))
            proposal = proposal_from_signal(observation, teacher.expert_id, signal)
            if proposal is not None:
                proposals.append(proposal)
        return CandidateCollection(group_teacher_proposals(proposals), tuple(signals))
