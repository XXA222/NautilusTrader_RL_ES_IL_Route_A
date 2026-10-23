"""DAgger: label STUDENT-VISITED native states, not the original teacher path."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol
from math import isfinite
from rles_route_a.contracts import (
    ExpertAdvice, ExpertTeacher, NativeActionTrace, PointInTimeObservation,
)
from rles_route_a.datasets.sequence_dataset import LabeledStep
from rles_route_a.datasets.daggerset import AggregatedLabel, DAggerSet
from rles_route_a.integration.scope import HostTrainGrant


@dataclass(frozen=True)
class VisitedStudentState:
    """Produced by existing native L3 rollout, not a generated financial ledger."""
    observation: PointInTimeObservation
    native_action_trace: NativeActionTrace
    student_policy_id: str
    source_identity: str
    split: str = "train"
    teacher_label: ExpertAdvice | None = None
    projected_teacher_target: ProjectedTeacherTarget | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation, PointInTimeObservation):
            raise TypeError("student state requires a native point-in-time observation")
        if not isinstance(self.native_action_trace, NativeActionTrace):
            raise TypeError("student state requires a native action trace")
        if not isinstance(self.student_policy_id, str) or not self.student_policy_id.strip():
            raise ValueError("student policy identity required")
        if not isinstance(self.source_identity, str) or not self.source_identity.strip():
            raise ValueError("student state must bind a registered source identity")
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("student state split identity required")
        if (self.teacher_label is None) != (self.projected_teacher_target is None):
            raise ValueError("precomputed teacher label and native projection must be supplied together")
        if self.teacher_label is not None and not isinstance(self.teacher_label, ExpertAdvice):
            raise TypeError("precomputed teacher advice is invalid")
        if (self.projected_teacher_target is not None
                and not isinstance(self.projected_teacher_target, ProjectedTeacherTarget)):
            raise TypeError("precomputed native target projection is invalid")


class TrainStudentProvider(Protocol):
    """Return complete visits, binding labels before native step and traces after it.

    A native rollout should ask the teacher and project its command synchronously
    at the student's pre-action barrier, then attach the real M2.2 trace once the
    native action completes. The two precomputed fields avoid projecting against
    a later account snapshot.
    """

    def collect_train_states(self, *, grant: HostTrainGrant,
                             student_policy_id: str) -> Iterable[VisitedStudentState]: ...


@dataclass(frozen=True)
class ProjectedTeacherTarget:
    """Host L3 projection, NOT a claimed fill or observed account transition."""
    target: float
    host_execution_recipe: str
    disposition: str = "PROJECTED"

    def __post_init__(self) -> None:
        if not isfinite(self.target) or not -1 <= self.target <= 1:
            raise ValueError("invalid risk-projected target")
        if not self.host_execution_recipe or self.disposition not in {"PROJECTED", "NOOP", "WAIT", "CANCEL"}:
            raise ValueError("native projection recipe and status required")


class NativeExpertProjection(Protocol):
    """Host-provided projection of teacher command through L3 risk/planner.

    It must use state that student ACTUALLY visited and not mutate the student's
    recorded trajectory. For CANCEL/WAIT/NOOP, return reached exposure, not a
    fictional fill. This is a counterfactual *label*, not a PnL measurement.
    """

    def project_native_target(self, observation: PointInTimeObservation,
                              requested_exposure: float) -> ProjectedTeacherTarget: ...


class DAggerCollector:
    def __init__(self, teacher: ExpertTeacher, projector: NativeExpertProjection,
                 train_grant_verifier: Callable[[HostTrainGrant], bool]):
        if not callable(train_grant_verifier):
            raise TypeError("host registered Train grant verifier required")
        self.teacher = teacher
        self.projector = projector
        self.train_grant_verifier = train_grant_verifier

    def authorize_train_grant(self, grant: HostTrainGrant) -> None:
        """Verify L6 registration before a student-state provider is accessed."""
        if (not isinstance(grant, HostTrainGrant) or grant.split != "train"
                or self.train_grant_verifier(grant) is not True):
            raise PermissionError("host authority did not verify the registered Train grant")

    def collect(self, *, provider: TrainStudentProvider, grant: HostTrainGrant,
                student_policy_id: str, iteration: int, aggregate: DAggerSet) -> int:
        self.authorize_train_grant(grant)
        if (grant.split != "train" or grant.fold_id != aggregate.fold_id
                or aggregate.source_identity != grant.source_identity
                or iteration < 1 or not student_policy_id):
            raise PermissionError("DAgger collection must be registered Train-only")
        candidates: list[AggregatedLabel] = []
        execution_recipe: str | None = None
        for visit in provider.collect_train_states(grant=grant,
                                                   student_policy_id=student_policy_id):
            if visit.split != "train" or visit.observation.fold_id != grant.fold_id:
                raise PermissionError("student state outside granted fold or Train scope")
            if visit.source_identity != grant.source_identity:
                raise PermissionError("student state source differs from registered Train grant")
            if visit.student_policy_id != student_policy_id:
                raise ValueError("student actor identity mismatch")
            if visit.observation.decision_ts_ns != visit.native_action_trace.decision_ts_ns:
                raise ValueError("native decision mismatch")
            # IMPORTANT: current account snapshot is the STUDENT's position and
            # pending orders, not the historical teacher account snapshot.
            label = (visit.teacher_label if visit.teacher_label is not None
                     else self.teacher.advise(visit.observation))
            if label.decision_ts_ns != visit.observation.decision_ts_ns:
                raise ValueError("expert label cannot use a different decision timestamp")
            result = (visit.projected_teacher_target
                      if visit.projected_teacher_target is not None else
                      self.projector.project_native_target(
                          visit.observation, label.requested_exposure,
                      ))
            if not isinstance(result, ProjectedTeacherTarget):
                raise TypeError("native projection evidence required, not a bare float")
            if execution_recipe is not None and result.host_execution_recipe != execution_recipe:
                raise ValueError("DAgger projection used changing execution recipes")
            execution_recipe = result.host_execution_recipe
            # A NOOP caused by deadband or constraints is NOT a counterfactual
            # execution of the teacher's requested change. The original student
            # position cannot become a positive EXIT supervision target.
            supervised = result.disposition not in {"WAIT", "CANCEL"} and not (
                result.disposition == "NOOP"
                and abs(result.target - label.requested_exposure) > 1e-6
            )
            sample = LabeledStep(visit.observation, label, result.target,
                                 action_status=result.disposition, split="train",
                                 supervision_weight=1.0 if supervised else 0.0,
                                 source_identity=grant.source_identity,
                                 intent_supervision_weight=1.0)
            candidates.append(AggregatedLabel(iteration, student_policy_id, sample,
                                              result.host_execution_recipe))
        aggregate.extend(candidates)
        return len(candidates)
