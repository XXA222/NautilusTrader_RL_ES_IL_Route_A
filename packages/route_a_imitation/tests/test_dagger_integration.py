from dataclasses import replace
import pytest
from rles_route_a.contracts import NativeActionTrace
from rles_route_a.teachers.wait import WaitTeacher
from rles_route_a.datasets.sequence_dataset import LabeledStep
from rles_route_a.datasets.daggerset import DAggerSet
from rles_route_a.training.dagger_collector import DAggerCollector, VisitedStudentState, ProjectedTeacherTarget
from rles_route_a.integration.scope import HostTrainGrant, read_train_only
from rles_route_a.integration.l3_policy_adapter import L3ActorAdapter
from rles_route_a.integration.native_rollout import TeacherFact, native_teacher_step
from rles_route_a.evaluation.diagnostics import summarize_actions
from rles_route_a.evaluation.recovery_benchmark import (
    NativeRecoveryEpisode, RecoveryCase, RecoveryScenarioSpec,
    RecoveryGateThresholds, RecoveryIterationReport, compare_recovery_iterations,
    run_native_recovery_iteration, summarize_recovery,
)
from rles_route_a.evaluation.experiment_matrix import (
    PREREGISTERED_ARMS, PREREGISTERED_SEEDS, build_matrix,
    build_preregistered_matrix,
)


def trace(ts, req=0., actual=0., action="NOOP"):
    return NativeActionTrace(ts, req, req, req, actual, 1000., 999., action)


class StaticStudentPolicy:
    def __init__(self):
        self.resets = []

    def reset(self, episode_id):
        self.resets.append(episode_id)

    def predict_exposure(self, observation):
        return observation.account_value("net_exposure")


def test_teacher_native_label_tracks_unfilled_not_requested(observed):
    o = observed(pos=.5)
    a = WaitTeacher().advise(o)
    step = native_teacher_step(TeacherFact(o, a, trace(o.decision_ts_ns, actual=.5)))
    assert step.executable_exposure == .5  # not fictional zero fill
    assert step.action_status == "NOOP"


def test_teacher_native_noop_label_accepts_executable_risk_adjusted_target(observed):
    o = observed(pos=.1)
    a = replace(WaitTeacher().advise(o), requested_exposure=.2)
    native = replace(
        trace(o.decision_ts_ns, req=.2, actual=.1002),
        risk_adjusted_target=.1, planner_target=.1,
        planner_reason="deadband",
    )

    step = native_teacher_step(TeacherFact(o, a, native))

    assert step.action_status == "NOOP"
    assert step.executable_exposure == .1002
    assert step.supervision_weight == 1.
    from rles_route_a.datasets.sequence_dataset import EpisodeSequenceDataset
    dataset = EpisodeSequenceDataset((step,), feature_names=("account.net_exposure",), context_bars=1)
    assert dataset[0]["exposure"].item() == pytest.approx(.2)


def test_dagger_relabels_actual_student_account(observed):
    step_o = observed(pos=0.)
    start = LabeledStep(step_o, WaitTeacher().advise(step_o), 0.,
                        source_identity="market-snapshot-1")
    data = DAggerSet((start,))
    visited = observed(pos=-.6, episode="student-e2", ts=2_000_000_000)
    class Provider:
        def collect_train_states(self, *, grant, student_policy_id):
            assert grant.registered_before_read
            yield VisitedStudentState(visited, trace(visited.decision_ts_ns, actual=-.6),
                                      student_policy_id, grant.source_identity)
    class Projector:
        def project_native_target(self, obs, requested_exposure):
            assert obs.account_value("net_exposure") == -.6
            return ProjectedTeacherTarget(0., "snapshot80194a-exec-recipe", "PROJECTED")
    grant = HostTrainGrant("f1", "route_a", "reserve-1", "market-snapshot-1", "train", True)
    collector = DAggerCollector(WaitTeacher(), Projector(), lambda _grant: True)
    count = collector.collect(
        provider=Provider(), grant=grant, student_policy_id="student-1", iteration=1,
        aggregate=data)
    assert count == 1 and data.collected == 1
    assert data.training_steps()[-1].advice.intent == "EXIT"
    assert data.training_steps()[-1].action_status == "PROJECTED"
    with pytest.raises(ValueError, match="duplicate"):
        collector.collect(
            provider=Provider(), grant=grant, student_policy_id="student-1", iteration=1,
            aggregate=data)


def test_dagger_rejects_student_state_from_another_source(observed):
    obs = observed(episode="foreign-source", ts=2_000_000_000)
    grant = HostTrainGrant("f1", "route_a", "reserve-1", "market-snapshot-1", "train", True)
    original = observed()
    data = DAggerSet((LabeledStep(original, WaitTeacher().advise(original), 0.,
                                  source_identity="market-snapshot-1"),))

    class Provider:
        def collect_train_states(self, *, grant, student_policy_id):
            yield VisitedStudentState(obs, trace(obs.decision_ts_ns), student_policy_id, "other-source")

    class Projector:
        def project_native_target(self, obs, requested_exposure):
            return ProjectedTeacherTarget(requested_exposure, "native-exec-v1")

    with pytest.raises(PermissionError, match="source differs"):
        DAggerCollector(WaitTeacher(), Projector(), lambda _grant: True).collect(
            provider=Provider(), grant=grant, student_policy_id="student-1", iteration=1,
            aggregate=data)


def test_dagger_verifies_registered_grant_before_reading_student_states(observed):
    obs = observed()
    grant = HostTrainGrant("f1", "route_a", "reserve-1", "market-snapshot-1", "train", True)
    data = DAggerSet((LabeledStep(obs, WaitTeacher().advise(obs), 0.,
                                  source_identity="market-snapshot-1"),))

    class Provider:
        def collect_train_states(self, *, grant, student_policy_id):
            raise AssertionError("student-state source accessed before grant verification")

    class Projector:
        def project_native_target(self, obs, requested_exposure):
            return ProjectedTeacherTarget(requested_exposure, "native-exec-v1")

    collector = DAggerCollector(WaitTeacher(), Projector(), lambda _grant: False)
    with pytest.raises(PermissionError, match="did not verify"):
        collector.collect(provider=Provider(), grant=grant, student_policy_id="student-1",
                          iteration=1, aggregate=data)


def test_recovery_verifies_registered_grant_before_calling_host_runner(observed):
    grant = HostTrainGrant("f1", "route_a", "reserve-1", "market-snapshot-1", "train", True)
    obs = observed()
    aggregate = DAggerSet((LabeledStep(obs, WaitTeacher().advise(obs), 0.,
                                       source_identity=grant.source_identity),))

    class Runner:
        def run_recovery_scenario(self, *, grant, scenario, student_policy,
                                  student_policy_id, seed):
            raise AssertionError("native recovery runner called before grant verification")

    class Projector:
        def project_native_target(self, observation, requested_exposure):
            return ProjectedTeacherTarget(requested_exposure, "native-exec-v1")

    collector = DAggerCollector(WaitTeacher(), Projector(), lambda _grant: False)
    scenarios = (
        RecoveryScenarioSpec("scenario-1", "wrong_direction", "prefix-1", "fault-1"),
        RecoveryScenarioSpec("scenario-2", "oversized_position", "prefix-2", "fault-2"),
        RecoveryScenarioSpec("scenario-3", "missed_exit", "prefix-3", "fault-3"),
    )
    with pytest.raises(PermissionError, match="did not verify"):
        run_native_recovery_iteration(
            runner=Runner(), collector=collector, grant=grant, scenarios=scenarios,
            student_policy=StaticStudentPolicy(),
            student_policy_id="student-1", seed=7, iteration=1, aggregate=aggregate,
            execution_recipe="native-exec-v1", native_receipt_verifier=lambda *_: True,
        )


def test_host_grants_fail_closed():
    with pytest.raises(PermissionError):
        HostTrainGrant("f", "a", "r", "source", "validation", True)
    class Reader:
        def read_registered_train(self, grant):
            return grant.source_identity
    grant = HostTrainGrant("f", "a", "r", "source", "train", True)
    assert read_train_only(Reader(), grant) == "source"


def test_l3_requires_explicit_close_and_only_calls_sink(observed):
    class Model:
        def reset(self, episode_id):
            self.episode_id = episode_id
        def predict_exposure(self, observation):
            return 0.
    class Sink:
        def submit_normalized_target(self, **kwargs):
            return kwargs
    with pytest.raises(ValueError, match="flat-close"):
        L3ActorAdapter(model=Model(), sink=Sink(), policy_id="p", zero_target_policy="legacy")
    x = L3ActorAdapter(model=Model(), sink=Sink(), policy_id="p", zero_target_policy="close")
    x.reset("e1")
    event = x.step(observed())
    assert event["target_exposure"] == 0. and event["action_schema"] == "single_exposure_v1"


def test_reporting_no_native_pnl_invention():
    cases = [RecoveryCase("a", 2, 5, 0.2, 0.06, 0, True),
             RecoveryCase("b", None, 3, 0.4, 0.1, 1, False)]
    report = summarize_recovery(cases)
    assert report["n"] == 2 and report["safe_fraction"] == .5
    assert report["normal_close_fraction"] == .5
    rows = [trace(1), trace(2, .4, .3, "SUBMIT")]
    a = summarize_actions(rows)
    assert a["submit_fraction"] == .5 and a["economic_status"] == "SCENARIO_ONLY"
    matrix = build_matrix(recipes={"teacher_trend": "r1", "bc_gru_multiscale": "r2"},
                          seeds=(11,), scenarios=("base", "stress"))
    assert len(matrix) == 4


def test_formal_experiment_matrix_requires_all_preregistered_arms_and_protocols():
    recipes = {name: f"frozen:{name}" for name in PREREGISTERED_ARMS}
    matrix = build_preregistered_matrix(recipes=recipes)

    assert len(matrix) == len(PREREGISTERED_ARMS) * len(PREREGISTERED_SEEDS) * 3
    assert {row.seed for row in matrix} == set(PREREGISTERED_SEEDS)
    assert {row.cost_scenario for row in matrix} == {"zero", "base", "stress"}
    with pytest.raises(ValueError, match="incomplete preregistered"):
        build_preregistered_matrix(recipes={key: value for key, value in recipes.items()
                                            if key != "random"})
    with pytest.raises(ValueError, match="frozen five-seed"):
        build_preregistered_matrix(recipes=recipes, seeds=(11,))


def test_native_recovery_iteration_verifies_all_three_modes_then_daggers(observed):
    grant = HostTrainGrant("f1", "route_a", "reserve-1", "market-snapshot-1", "train", True)
    original_obs = observed(episode="teacher-e1")
    original = LabeledStep(original_obs, WaitTeacher().advise(original_obs), 0.,
                           source_identity=grant.source_identity)
    aggregate = DAggerSet((original,))
    scenarios = (
        RecoveryScenarioSpec("wrong-way-1", "wrong_direction", "prefix-1", "fault-1"),
        RecoveryScenarioSpec("oversize-1", "oversized_position", "prefix-2", "fault-2"),
        RecoveryScenarioSpec("missed-exit-1", "missed_exit", "prefix-3", "fault-3"),
    )
    policy = StaticStudentPolicy()

    class Runner:
        def run_recovery_scenario(self, *, grant, scenario, student_policy,
                                  student_policy_id, seed):
            assert student_policy is policy
            obs = observed(episode=scenario.scenario_id, ts=2_000_000_000)
            state = VisitedStudentState(
                obs, trace(obs.decision_ts_ns), student_policy_id, grant.source_identity,
            )
            case = RecoveryCase(scenario.scenario_id, 0, 1, .01, .01, 0, True)
            return NativeRecoveryEpisode(
                scenario.scenario_id, grant.fold_id, grant.source_identity,
                "native-exec-v1", f"receipt:{scenario.scenario_id}", case, (state,),
            )

    class Projector:
        def project_native_target(self, observation, requested_exposure):
            return ProjectedTeacherTarget(requested_exposure, "native-exec-v1")

    verified = []
    report = run_native_recovery_iteration(
        runner=Runner(), collector=DAggerCollector(WaitTeacher(), Projector(), lambda _grant: True),
        grant=grant, scenarios=scenarios, student_policy=policy,
        student_policy_id="student-1", seed=7,
        iteration=1, aggregate=aggregate, execution_recipe="native-exec-v1",
        native_receipt_verifier=lambda episode, spec, source: (
            verified.append(episode.native_receipt_id) or True
        ),
    )

    assert report.dagger_samples == 3 and aggregate.collected == 3
    assert report.summary["n"] == 3 and report.summary["normal_close_fraction"] == 1.
    assert len(verified) == 3
    assert policy.resets == [scenario.scenario_id for scenario in scenarios]


def test_native_recovery_rejects_state_from_another_source(observed):
    grant = HostTrainGrant("f1", "route_a", "reserve-1", "market-snapshot-1", "train", True)
    class Runner:
        def run_recovery_scenario(self, *, grant, scenario, student_policy,
                                  student_policy_id, seed):
            obs = observed(episode=scenario.scenario_id, ts=2_000_000_000)
            state = VisitedStudentState(obs, trace(obs.decision_ts_ns), student_policy_id, "other-source")
            case = RecoveryCase(scenario.scenario_id, 0, 1, .01, .01, 0, True)
            return NativeRecoveryEpisode(
                scenario.scenario_id, grant.fold_id, grant.source_identity,
                "native-exec-v1", f"receipt:{scenario.scenario_id}", case, (state,),
            )

    class Projector:
        def project_native_target(self, observation, requested_exposure):
            return ProjectedTeacherTarget(requested_exposure, "native-exec-v1")

    scenarios = (
        RecoveryScenarioSpec("scenario-1", "wrong_direction", "prefix-1", "fault-1"),
        RecoveryScenarioSpec("scenario-2", "oversized_position", "prefix-2", "fault-2"),
        RecoveryScenarioSpec("scenario-3", "missed_exit", "prefix-3", "fault-3"),
    )
    original_obs = observed()
    aggregate = DAggerSet((LabeledStep(original_obs, WaitTeacher().advise(original_obs), 0.,
                                       source_identity=grant.source_identity),))
    with pytest.raises(PermissionError, match="unbound or non-Train"):
        run_native_recovery_iteration(
            runner=Runner(), collector=DAggerCollector(WaitTeacher(), Projector(), lambda _grant: True),
            grant=grant, scenarios=scenarios, student_policy=StaticStudentPolicy(),
            student_policy_id="student-1", seed=7,
            iteration=1, aggregate=aggregate, execution_recipe="native-exec-v1",
            native_receipt_verifier=lambda *_: True,
        )


def test_recovery_gate_compares_paired_scenarios_and_normal_degradation(observed):
    def report(policy_id, iteration, *, cost, drawdown, safe_at, violations=0):
        episodes = []
        for index, scenario_id in enumerate(("wrong-way", "oversized", "missed-exit")):
            obs = observed(episode=f"{policy_id}-{scenario_id}", ts=2_000_000_000 + index)
            state = VisitedStudentState(
                obs, trace(obs.decision_ts_ns), policy_id, "market-snapshot-1",
            )
            case = RecoveryCase(
                scenario_id, safe_at, 1, cost, drawdown, violations, True,
            )
            episodes.append(NativeRecoveryEpisode(
                scenario_id, "f1", "market-snapshot-1", "native-exec-v1",
                f"receipt:{policy_id}:{scenario_id}", case, (state,),
            ))
        cases = tuple(row.case for row in episodes)
        specs = (
            RecoveryScenarioSpec("wrong-way", "wrong_direction", "prefix-1", "fault-1"),
            RecoveryScenarioSpec("oversized", "oversized_position", "prefix-2", "fault-2"),
            RecoveryScenarioSpec("missed-exit", "missed_exit", "prefix-3", "fault-3"),
        )
        return RecoveryIterationReport(
            tuple(episodes), specs, summarize_recovery(cases), 3, "f1",
            "market-snapshot-1", "native-exec-v1", policy_id, 7, iteration,
        )

    baseline = report("student-v1", 0, cost=.03, drawdown=.10, safe_at=0)
    candidate = report("student-v2", 1, cost=.02, drawdown=.08, safe_at=0)
    thresholds = RecoveryGateThresholds(
        max_safe_fraction_regression=0.,
        max_drawdown_regression=0.,
        max_recovery_cost_regression=0.,
        max_steps_to_safe_regression=0.,
        max_normal_metric_degradation=.03,
    )

    result = compare_recovery_iterations(
        baseline=baseline, candidate=candidate, thresholds=thresholds,
        normal_metric_name="net_return", baseline_normal_metric=.20,
        candidate_normal_metric=.18, normal_metric_direction="higher_is_better",
    )

    assert result.passed
    assert result.scenario_count == 3
    assert result.deltas["mean_recovery_cost"] == pytest.approx(-.01)
    assert result.normal_metric_degradation == pytest.approx(.02)

    rejected = compare_recovery_iterations(
        baseline=baseline, candidate=candidate,
        thresholds=RecoveryGateThresholds(0., 0., 0., 0., .01),
        normal_metric_name="max_drawdown", baseline_normal_metric=.10,
        candidate_normal_metric=.12, normal_metric_direction="lower_is_better",
    )
    assert not rejected.passed
    assert "NORMAL_TRAJECTORY_DEGRADED" in rejected.reason_codes

    unsafe_candidate = report(
        "student-v3", 2, cost=.01, drawdown=.01, safe_at=0, violations=1,
    )
    unsafe = compare_recovery_iterations(
        baseline=baseline, candidate=unsafe_candidate, thresholds=thresholds,
        normal_metric_name="net_return", baseline_normal_metric=.20,
        candidate_normal_metric=.20, normal_metric_direction="higher_is_better",
    )
    assert not unsafe.passed
    assert "CANDIDATE_HARD_RISK_VIOLATION" in unsafe.reason_codes

    altered_spec = replace(candidate.scenario_specs[0], native_prefix_id="different-prefix")
    mismatched = replace(
        candidate, scenario_specs=(altered_spec, *candidate.scenario_specs[1:]),
    )
    with pytest.raises(ValueError, match="identical kinds, native prefixes, and fault recipes"):
        compare_recovery_iterations(
            baseline=baseline, candidate=mismatched, thresholds=thresholds,
            normal_metric_name="net_return", baseline_normal_metric=.20,
            candidate_normal_metric=.20, normal_metric_direction="higher_is_better",
        )


def test_action_summary_separates_current_submission_from_prior_order_fills():
    rows = [
        NativeActionTrace(
            1, .5, .4, .4, 0., 1000., 999., "SUBMIT",
            submitted_order_ids=("old-order",),
            soft_risk_target=.4,
        ),
        NativeActionTrace(
            2, .2, .2, .2, .1, 999., 998., "SUBMIT",
            submitted_order_ids=("new-order",),
            filled_order_ids=("old-order",),
            related_existing_order_ids=("old-order",),
            soft_risk_target=.1,
        ),
    ]

    summary = summarize_actions(rows)

    assert summary["nonzero_request_fraction"] == 1.
    assert summary["soft_risk_target_known_fraction"] == 1.
    assert summary["nonzero_soft_risk_target_fraction"] == 1.
    assert summary["mean_request_to_soft_risk_gap"] == pytest.approx(.1)
    assert summary["mean_soft_to_hard_risk_gap"] == pytest.approx(.05)
    assert summary["nonzero_risk_adjusted_fraction"] == 1.
    assert summary["nonzero_planner_target_fraction"] == 1.
    assert summary["submit_barriers_with_current_fill_fraction"] == 0.
    assert summary["current_submitted_order_fill_fraction"] == 0.
