"""Evaluate host-provided native recovery facts; no synthetic PnL invented here."""
from __future__ import annotations
from dataclasses import dataclass
from math import inf, isfinite
from statistics import mean
from typing import Callable, Literal, Protocol, Sequence
from rles_route_a.datasets.daggerset import DAggerSet
from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.training.dagger_collector import (
    DAggerCollector, VisitedStudentState,
)
from rles_route_a.contracts import PointInTimeObservation


RecoveryKind = Literal["wrong_direction", "oversized_position", "missed_exit"]
RECOVERY_KINDS: tuple[RecoveryKind, ...] = (
    "wrong_direction", "oversized_position", "missed_exit",
)


@dataclass(frozen=True)
class RecoveryScenarioSpec:
    """Reference a host native prefix and fault recipe; never fabricate account state."""

    scenario_id: str
    kind: RecoveryKind
    native_prefix_id: str
    fault_recipe_id: str

    def __post_init__(self) -> None:
        if (not all(isinstance(value, str) and value.strip() for value in
                    (self.scenario_id, self.native_prefix_id, self.fault_recipe_id))
                or self.kind not in RECOVERY_KINDS):
            raise ValueError("recovery scenario needs a known fault and native recipe references")


@dataclass(frozen=True)
class NativeRecoveryEpisode:
    """Host-produced native recovery path plus the student states to relabel."""

    scenario_id: str
    fold_id: str
    source_identity: str
    execution_recipe: str
    native_receipt_id: str
    case: "RecoveryCase"
    visited_states: tuple[VisitedStudentState, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in
                   (self.scenario_id, self.fold_id, self.source_identity,
                    self.execution_recipe, self.native_receipt_id)):
            raise ValueError("native recovery episode is missing provenance")
        if not isinstance(self.case, RecoveryCase) or self.case.case_id != self.scenario_id:
            raise ValueError("recovery summary facts must bind the requested scenario")
        if (not isinstance(self.visited_states, tuple) or not self.visited_states
                or len(self.visited_states) != self.case.steps_observed
                or any(not isinstance(state, VisitedStudentState)
                       for state in self.visited_states)):
            raise ValueError("native recovery facts must cover every visited decision")


class NativeRecoveryRunner(Protocol):
    """Use the passed student actor in L3 and bind its ID into native receipts.

    The runner must reset this exact policy for the scenario episode before
    its first prediction; it must not resolve a different actor by ID.
    """

    def run_recovery_scenario(self, *, grant: HostTrainGrant,
                              scenario: RecoveryScenarioSpec,
                              student_policy: "StudentExposurePolicy",
                              student_policy_id: str, seed: int
                              ) -> NativeRecoveryEpisode: ...


class StudentExposurePolicy(Protocol):
    """Exact student actor used for the host native policy rollout."""

    def reset(self, episode_id: str) -> None: ...
    def predict_exposure(self, observation: PointInTimeObservation) -> float: ...


@dataclass(frozen=True)
class RecoveryIterationReport:
    scenarios: tuple[NativeRecoveryEpisode, ...]
    scenario_specs: tuple[RecoveryScenarioSpec, ...]
    summary: dict[str, float | int]
    dagger_samples: int
    fold_id: str
    source_identity: str
    execution_recipe: str
    student_policy_id: str
    seed: int
    iteration: int


@dataclass(frozen=True)
class RecoveryGateThresholds:
    """Pre-registered maximum regressions for a paired RA3 comparison."""

    max_safe_fraction_regression: float
    max_drawdown_regression: float
    max_recovery_cost_regression: float
    max_steps_to_safe_regression: float
    max_normal_metric_degradation: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative pre-registered bound")


@dataclass(frozen=True)
class RecoveryGateResult:
    passed: bool
    baseline_policy_id: str
    candidate_policy_id: str
    scenario_count: int
    deltas: dict[str, float | int | None]
    normal_metric_name: str
    normal_metric_degradation: float
    reason_codes: tuple[str, ...]


def compare_recovery_iterations(
        *, baseline: RecoveryIterationReport, candidate: RecoveryIterationReport,
        thresholds: RecoveryGateThresholds, normal_metric_name: str,
        baseline_normal_metric: float, candidate_normal_metric: float,
        normal_metric_direction: Literal["higher_is_better", "lower_is_better"],
) -> RecoveryGateResult:
    """Apply a paired RA3 promotion gate to two host-backed recovery runs.

    Recovery scenarios must share their fold, source, execution recipe, seed,
    and exact scenario identities. Thresholds are explicit inputs so the caller
    can freeze them before collecting candidate results. Positive normal-metric
    degradation means the candidate value is worse than the baseline.
    """
    if not isinstance(baseline, RecoveryIterationReport) or not isinstance(
            candidate, RecoveryIterationReport):
        raise TypeError("paired native recovery iteration reports required")
    if not isinstance(thresholds, RecoveryGateThresholds):
        raise TypeError("pre-registered recovery thresholds required")
    if not isinstance(normal_metric_name, str) or not normal_metric_name.strip():
        raise ValueError("normal-trajectory metric name required")
    if normal_metric_direction not in {"higher_is_better", "lower_is_better"}:
        raise ValueError("normal metric direction must be explicitly frozen")
    for name, value in (("baseline normal metric", baseline_normal_metric),
                        ("candidate normal metric", candidate_normal_metric)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not isfinite(value)):
            raise ValueError(f"{name} must be finite")
    baseline_scope = (baseline.fold_id, baseline.source_identity,
                      baseline.execution_recipe, baseline.seed)
    candidate_scope = (candidate.fold_id, candidate.source_identity,
                       candidate.execution_recipe, candidate.seed)
    if baseline_scope != candidate_scope:
        raise PermissionError("RA3 comparison reports must share Fold, source, recipe, and seed")
    if (baseline.student_policy_id == candidate.student_policy_id
            or candidate.iteration <= baseline.iteration):
        raise ValueError("candidate must be a later DAgger iteration with a distinct policy")

    for report in (baseline, candidate):
        specs_by_id = {spec.scenario_id: spec for spec in report.scenario_specs}
        if (len(specs_by_id) != len(report.scenario_specs)
                or specs_by_id.keys() != {episode.scenario_id for episode in report.scenarios}):
            raise ValueError("recovery report must retain one spec for every native scenario")
        if any(
                episode.fold_id != report.fold_id
                or episode.source_identity != report.source_identity
                or episode.execution_recipe != report.execution_recipe
                or episode.scenario_id not in specs_by_id
                or any(state.student_policy_id != report.student_policy_id
                       or state.source_identity != report.source_identity
                       or state.observation.fold_id != report.fold_id
                       for state in episode.visited_states)
                for episode in report.scenarios):
            raise PermissionError("recovery report metadata differs from its native episode evidence")

    baseline_specs = {spec.scenario_id: spec for spec in baseline.scenario_specs}
    candidate_specs = {spec.scenario_id: spec for spec in candidate.scenario_specs}
    if baseline_specs != candidate_specs:
        raise ValueError("RA3 paired reports must use identical kinds, native prefixes, and fault recipes")

    baseline_cases = {row.scenario_id: row.case for row in baseline.scenarios}
    candidate_cases = {row.scenario_id: row.case for row in candidate.scenarios}
    if (not baseline_cases or len(baseline_cases) != len(baseline.scenarios)
            or len(candidate_cases) != len(candidate.scenarios)
            or baseline_cases.keys() != candidate_cases.keys()):
        raise ValueError("RA3 reports must contain the same unique paired scenarios")

    def metrics(cases: dict[str, "RecoveryCase"]) -> dict[str, float | int]:
        values = tuple(cases.values())
        safe_steps = tuple(case.safe_at_step for case in values
                           if case.safe_at_step is not None)
        return {
            "safe_fraction": len(safe_steps) / len(values),
            "mean_steps_to_safe": (sum(safe_steps) / len(safe_steps)
                                   if safe_steps else inf),
            "mean_recovery_cost": mean(case.additional_native_cost for case in values),
            "max_peak_drawdown": max(case.peak_drawdown for case in values),
            "hard_risk_violations": sum(case.hard_risk_violations for case in values),
        }

    before = metrics(baseline_cases)
    after = metrics(candidate_cases)
    safe_delta = float(after["safe_fraction"]) - float(before["safe_fraction"])
    steps_before = float(before["mean_steps_to_safe"])
    steps_after = float(after["mean_steps_to_safe"])
    steps_delta = (steps_after - steps_before
                   if isfinite(steps_before) and isfinite(steps_after)
                   else 0.0 if not isfinite(steps_before) and not isfinite(steps_after)
                   else None)
    cost_delta = float(after["mean_recovery_cost"]) - float(before["mean_recovery_cost"])
    drawdown_delta = float(after["max_peak_drawdown"]) - float(before["max_peak_drawdown"])
    violation_delta = int(after["hard_risk_violations"]) - int(before["hard_risk_violations"])
    normal_degradation = (
        float(baseline_normal_metric) - float(candidate_normal_metric)
        if normal_metric_direction == "higher_is_better"
        else float(candidate_normal_metric) - float(baseline_normal_metric)
    )
    deltas: dict[str, float | int | None] = {
        "safe_fraction": safe_delta,
        "mean_steps_to_safe": steps_delta,
        "mean_recovery_cost": cost_delta,
        "max_peak_drawdown": drawdown_delta,
        "hard_risk_violations": violation_delta,
    }
    failures: list[str] = []
    if int(after["hard_risk_violations"]) > 0:
        failures.append("CANDIDATE_HARD_RISK_VIOLATION")
    if safe_delta < -thresholds.max_safe_fraction_regression:
        failures.append("SAFE_RECOVERY_REGRESSED")
    if drawdown_delta > thresholds.max_drawdown_regression:
        failures.append("RECOVERY_DRAWDOWN_REGRESSED")
    if cost_delta > thresholds.max_recovery_cost_regression:
        failures.append("RECOVERY_COST_REGRESSED")
    steps_regressed = (
        isfinite(steps_before) and not isfinite(steps_after)
        or (steps_delta is not None
            and steps_delta > thresholds.max_steps_to_safe_regression)
    )
    if steps_regressed:
        failures.append("TIME_TO_SAFE_REGRESSED")
    if normal_degradation > thresholds.max_normal_metric_degradation:
        failures.append("NORMAL_TRAJECTORY_DEGRADED")
    strict_improvement = (
        safe_delta > 0 or drawdown_delta < 0 or cost_delta < 0
        or (not isfinite(steps_before) and isfinite(steps_after))
        or (steps_delta is not None and steps_delta < 0) or violation_delta < 0
    )
    if not strict_improvement:
        failures.append("NO_STRICT_RECOVERY_IMPROVEMENT")
    return RecoveryGateResult(
        passed=not failures,
        baseline_policy_id=baseline.student_policy_id,
        candidate_policy_id=candidate.student_policy_id,
        scenario_count=len(baseline_cases),
        deltas=deltas,
        normal_metric_name=normal_metric_name,
        normal_metric_degradation=normal_degradation,
        reason_codes=tuple(failures),
    )


def run_native_recovery_iteration(*, runner: NativeRecoveryRunner,
                                  collector: DAggerCollector,
                                  grant: HostTrainGrant,
                                  scenarios: Sequence[RecoveryScenarioSpec],
                                  student_policy: StudentExposurePolicy,
                                  student_policy_id: str, seed: int, iteration: int,
                                  aggregate: DAggerSet, execution_recipe: str,
                                  native_receipt_verifier: Callable[
                                      [NativeRecoveryEpisode, RecoveryScenarioSpec,
                                       HostTrainGrant], bool
                                  ]) -> RecoveryIterationReport:
    """Run the three initial failure modes through the existing native host.

    Prefixes, fault actions, market replay, account facts and recovery outcomes
    remain owned by the host. Route A only checks their binding, verifies the
    host receipt, and asks the expert to relabel states the student visited.
    """
    if not isinstance(collector, DAggerCollector):
        raise TypeError("Route A DAgger collector with host grant verifier required")
    collector.authorize_train_grant(grant)
    if (grant.split != "train" or grant.fold_id != aggregate.fold_id
            or aggregate.source_identity != grant.source_identity):
        raise PermissionError("recovery DAgger must match the registered Train source and Fold")
    if not isinstance(execution_recipe, str) or not execution_recipe.strip():
        raise ValueError("frozen native recovery execution recipe required")
    if not callable(native_receipt_verifier):
        raise TypeError("host native recovery receipt verifier required")
    if (not callable(getattr(student_policy, "reset", None))
            or not callable(getattr(student_policy, "predict_exposure", None))):
        raise TypeError("the exact Route A student policy must be passed to the host runner")
    if (not isinstance(student_policy_id, str) or not student_policy_id.strip()
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(iteration) is not int or iteration < 1):
        raise ValueError("student identity, seed and positive DAgger iteration required")
    specs = tuple(scenarios)
    if (len(specs) != len(RECOVERY_KINDS)
            or len({s.scenario_id for s in specs}) != len(specs)
            or {s.kind for s in specs} != set(RECOVERY_KINDS)):
        raise ValueError("one unique native scenario for each initial recovery failure mode is required")

    episodes: list[NativeRecoveryEpisode] = []
    all_states: list[VisitedStudentState] = []
    for spec in specs:
        student_policy.reset(spec.scenario_id)
        episode = runner.run_recovery_scenario(
            grant=grant, scenario=spec, student_policy=student_policy,
            student_policy_id=student_policy_id, seed=seed,
        )
        if not isinstance(episode, NativeRecoveryEpisode):
            raise TypeError("host recovery runner must return native episode evidence")
        if (episode.scenario_id != spec.scenario_id or episode.fold_id != grant.fold_id
                or episode.source_identity != grant.source_identity
                or episode.execution_recipe != execution_recipe):
            raise PermissionError("native recovery episode differs from its Train grant or frozen recipe")
        if native_receipt_verifier(episode, spec, grant) is not True:
            raise PermissionError("host authority rejected native recovery receipt")
        for state in episode.visited_states:
            if (state.split != "train" or state.observation.fold_id != grant.fold_id
                    or state.source_identity != episode.source_identity
                    or state.student_policy_id != student_policy_id
                    or state.observation.episode_id != spec.scenario_id
                    or state.observation.decision_ts_ns != state.native_action_trace.decision_ts_ns):
                raise PermissionError("recovery replay contains an unbound or non-Train student state")
        episodes.append(episode)
        all_states.extend(episode.visited_states)

    class _VisitedRecoveryStates:
        def collect_train_states(self, *, grant: HostTrainGrant,
                                 student_policy_id: str):
            yield from all_states

    # One collector call keeps DAggerSet's batch insertion atomic across all
    # three scenarios if any projected label fails validation.
    count = collector.collect(
        provider=_VisitedRecoveryStates(), grant=grant,
        student_policy_id=student_policy_id, iteration=iteration, aggregate=aggregate,
    )
    return RecoveryIterationReport(
        tuple(episodes), specs, summarize_recovery(tuple(episode.case for episode in episodes)),
        count, grant.fold_id, grant.source_identity, execution_recipe,
        student_policy_id, seed, iteration,
    )


@dataclass(frozen=True)
class RecoveryCase:
    case_id: str
    safe_at_step: int | None
    steps_observed: int
    additional_native_cost: float
    peak_drawdown: float
    hard_risk_violations: int
    closed_normally: bool

    def __post_init__(self) -> None:
        if not self.case_id or self.steps_observed < 1 or self.hard_risk_violations < 0:
            raise ValueError("invalid recovery case facts")
        if self.safe_at_step is not None and not 0 <= self.safe_at_step < self.steps_observed:
            raise ValueError("safe boundary outside observed episode")
        if (not isfinite(self.additional_native_cost) or self.additional_native_cost < 0
                or not isfinite(self.peak_drawdown) or not 0 <= self.peak_drawdown <= 1):
            raise ValueError("invalid native recovery cost or drawdown")


def summarize_recovery(cases: Sequence[RecoveryCase]) -> dict[str, float | int]:
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("need uniquely identified native recovery scenarios")
    return {
        "n": len(cases),
        "safe_fraction": sum(c.safe_at_step is not None for c in cases) / len(cases),
        "mean_steps_to_safe_if_reached": mean(c.safe_at_step for c in cases if c.safe_at_step is not None)
                                       if any(c.safe_at_step is not None for c in cases) else float("nan"),
        "mean_extra_cost": mean(c.additional_native_cost for c in cases),
        "max_peak_drawdown": max(c.peak_drawdown for c in cases),
        "hard_risk_violations": sum(c.hard_risk_violations for c in cases),
        "normal_close_fraction": sum(c.closed_normally for c in cases) / len(cases),
    }
