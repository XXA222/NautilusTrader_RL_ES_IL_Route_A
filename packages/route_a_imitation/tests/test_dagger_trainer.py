import torch
import pytest

from rles_route_a.config import BCConfig, DAggerConfig
from rles_route_a.contracts import ExpertAdvice, NativeActionTrace
from rles_route_a.datasets.daggerset import AggregatedLabel, DAggerSet
from rles_route_a.datasets.sequence_dataset import LabeledStep
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.evaluation.recovery_benchmark import (
    NativeRecoveryEpisode, RecoveryCase, RecoveryScenarioSpec,
)
from rles_route_a.training.dagger_collector import (
    DAggerCollector, ProjectedTeacherTarget, VisitedStudentState,
)
from rles_route_a.training.dagger_round import (
    DAggerIterationsError, run_native_dagger_iterations, run_native_dagger_round,
)
from rles_route_a.teachers.wait import WaitTeacher
from rles_route_a.training.dagger_trainer import fine_tune_bc_on_daggerset
from rles_route_a.training.dagger_trainer import DAggerMixtureSampler
from rles_route_a.training.factories import new_bc_policy


def test_native_dagger_round_connects_recovery_collection_and_warm_start(observed):
    config = BCConfig(
        feature_names=("market.trend_1h", "account.net_exposure"),
        context_bars=2, hidden_size=8, layers=1, batch_size=2, epochs=1, seed=17,
    )
    policy = BCInferencePolicy(new_bc_policy(config), config)
    expert_observation = observed(episode="expert")
    expert = LabeledStep(
        expert_observation, WaitTeacher().advise(expert_observation), 0.,
        source_identity="source-1",
    )
    aggregate = DAggerSet((expert,))
    grant = HostTrainGrant(expert_observation.fold_id, "owner", "reservation-1",
                           "source-1", "train", True)
    scenarios = (
        RecoveryScenarioSpec("wrong-way", "wrong_direction", "prefix-1", "fault-1"),
        RecoveryScenarioSpec("oversized", "oversized_position", "prefix-2", "fault-2"),
        RecoveryScenarioSpec("missed-exit", "missed_exit", "prefix-3", "fault-3"),
    )

    class Runner:
        def __init__(self):
            self.seen_policies = []

        def run_recovery_scenario(self, *, grant, scenario, student_policy,
                                  student_policy_id, seed):
            self.seen_policies.append(student_policy)
            obs = observed(episode=scenario.scenario_id, ts=2_000_000_000)
            trace = NativeActionTrace(
                obs.decision_ts_ns, 0., 0., 0., 0., 100., 100., "NOOP",
            )
            state = VisitedStudentState(
                obs, trace, student_policy_id, grant.source_identity,
            )
            case = RecoveryCase(scenario.scenario_id, 0, 1, .01, .01, 0, True)
            return NativeRecoveryEpisode(
                scenario.scenario_id, grant.fold_id, grant.source_identity,
                "native-exec-v1", f"receipt:{scenario.scenario_id}", case, (state,),
            )

    class Projector:
        def project_native_target(self, observation, requested_exposure):
            return ProjectedTeacherTarget(requested_exposure, "native-exec-v1")

    runner = Runner()
    result = run_native_dagger_iterations(
        runner=runner,
        collector=DAggerCollector(WaitTeacher(), Projector(), lambda value: value == grant),
        policy=policy,
        aggregate=aggregate,
        grant=grant,
        scenarios=scenarios,
        student_policy_ids=("student-v1", "student-v2", "student-v3"),
        seed=7,
        execution_recipe="native-exec-v1",
        native_receipt_verifier=lambda episode, scenario, _grant: (
            episode.native_receipt_id == f"receipt:{scenario.scenario_id}"
        ),
        train_grant_verifier=lambda value: value == grant,
        dagger_config=DAggerConfig(iterations=2, expert_fraction=.9),
    )

    assert len(result.rounds) == 2
    assert [row.recovery.dagger_samples for row in result.rounds] == [3, 3]
    assert aggregate.collected == 6
    assert [row.fine_tune.dagger_sample_count for row in result.rounds] == [3, 6]
    assert result.rounds[0].previous_student_policy_id == "student-v1"
    assert result.rounds[1].previous_student_policy_id == "student-v2"
    assert result.rounds[1].next_student_policy_id == "student-v3"
    assert result.final_student_policy_id == "student-v3"
    assert result.rounds[0].fine_tune.policy is not policy
    assert result.final_policy is result.rounds[1].fine_tune.policy
    assert runner.seen_policies[0] is policy
    assert runner.seen_policies[3] is result.rounds[0].fine_tune.policy
    assert all(row.fine_tune.history for row in result.rounds)
    assert all(row.fine_tune.requested_expert_fraction == pytest.approx(.9)
               for row in result.rounds)


def test_native_dagger_round_rejects_policy_identity_reuse():
    with pytest.raises(ValueError, match="distinct identities"):
        run_native_dagger_round(
            runner=None, collector=None, policy=None, aggregate=None, grant=None,
            scenarios=(), student_policy_id="student-v1",
            next_student_policy_id="student-v1", seed=1, iteration=1,
            execution_recipe="recipe", native_receipt_verifier=lambda *_: True,
            train_grant_verifier=lambda _: True,
        )


def test_native_dagger_iterations_resume_failed_round_without_partial_labels(observed):
    config = BCConfig(
        feature_names=("market.trend_1h", "account.net_exposure"),
        context_bars=2, hidden_size=8, layers=1, batch_size=2, epochs=1, seed=17,
    )
    policy = BCInferencePolicy(new_bc_policy(config), config)
    expert_observation = observed(episode="expert")
    expert = LabeledStep(
        expert_observation, WaitTeacher().advise(expert_observation), 0.,
        source_identity="source-1",
    )
    aggregate = DAggerSet((expert,))
    grant = HostTrainGrant(expert_observation.fold_id, "owner", "reservation-1",
                           "source-1", "train", True)
    scenarios = tuple(
        RecoveryScenarioSpec(scenario_id, kind, f"prefix-{index}", f"fault-{index}")
        for index, (scenario_id, kind) in enumerate((
            ("wrong-way", "wrong_direction"),
            ("oversized", "oversized_position"),
            ("missed-exit", "missed_exit"),
        ))
    )

    class Runner:
        def __init__(self):
            self.seen_policies = []

        def run_recovery_scenario(self, *, grant, scenario, student_policy,
                                  student_policy_id, seed):
            self.seen_policies.append(student_policy)
            obs = observed(episode=scenario.scenario_id, ts=2_000_000_000)
            trace = NativeActionTrace(
                obs.decision_ts_ns, 0., 0., 0., 0., 100., 100., "NOOP",
            )
            state = VisitedStudentState(
                obs, trace, student_policy_id, grant.source_identity,
            )
            case = RecoveryCase(scenario.scenario_id, 0, 1, .01, .01, 0, True)
            return NativeRecoveryEpisode(
                scenario.scenario_id, grant.fold_id, grant.source_identity,
                "native-exec-v1", f"receipt:{scenario.scenario_id}", case, (state,),
            )

    class Projector:
        def project_native_target(self, observation, requested_exposure):
            return ProjectedTeacherTarget(requested_exposure, "native-exec-v1")

    runner = Runner()
    collector = DAggerCollector(WaitTeacher(), Projector(), lambda value: value == grant)
    common = dict(
        runner=runner,
        collector=collector,
        grant=grant,
        scenarios=scenarios,
        student_policy_ids=("student-v1", "student-v2", "student-v3"),
        seed=7,
        execution_recipe="native-exec-v1",
        native_receipt_verifier=lambda episode, scenario, _grant: (
            episode.native_receipt_id == f"receipt:{scenario.scenario_id}"
        ),
        dagger_config=DAggerConfig(iterations=2, expert_fraction=.9),
        patience=1,
    )
    verifier_calls = 0

    def fail_second_fine_tune(value):
        nonlocal verifier_calls
        verifier_calls += 1
        return value == grant and verifier_calls != 4

    with pytest.raises(DAggerIterationsError, match="iteration 2 failed") as caught:
        run_native_dagger_iterations(
            **common,
            policy=policy,
            aggregate=aggregate,
            train_grant_verifier=fail_second_fine_tune,
        )

    progress = caught.value.progress
    assert progress.next_iteration == 2
    assert len(progress.completed_rounds) == 1
    assert progress.current_policy is progress.completed_rounds[0].fine_tune.policy
    assert aggregate.collected == 3
    assert progress.aggregate is aggregate
    assert len(runner.seen_policies) == 6
    assert all(item is progress.current_policy for item in runner.seen_policies[3:])

    resumed = run_native_dagger_iterations(
        **common,
        train_grant_verifier=lambda value: value == grant,
        progress=progress,
    )

    assert len(resumed.rounds) == 2
    assert resumed.rounds[0] is progress.completed_rounds[0]
    assert resumed.aggregate is aggregate
    assert aggregate.collected == 6
    assert resumed.rounds[1].fine_tune.dagger_sample_count == 6
    assert runner.seen_policies[6] is progress.current_policy

    with pytest.raises(ValueError, match="resume inputs differ"):
        run_native_dagger_iterations(
            **{**common, "seed": 8},
            train_grant_verifier=lambda value: value == grant,
            progress=progress,
        )


def test_native_dagger_round_denied_grant_does_not_start_recovery(observed):
    config = BCConfig(
        feature_names=("market.trend_1h", "account.net_exposure"),
        context_bars=2, hidden_size=8, layers=1, batch_size=2, epochs=1,
    )
    policy = BCInferencePolicy(new_bc_policy(config), config)
    observation = observed()
    sample = LabeledStep(
        observation, WaitTeacher().advise(observation), 0., source_identity="source-1",
    )
    aggregate = DAggerSet((sample,))
    grant = HostTrainGrant(
        observation.fold_id, "owner", "reservation-1", "source-1", "train", True,
    )
    runner_calls = []

    class Runner:
        def run_recovery_scenario(self, **_kwargs):
            runner_calls.append(True)
            raise AssertionError("recovery must not start after Train grant denial")

    with pytest.raises(PermissionError, match="did not verify"):
        run_native_dagger_round(
            runner=Runner(),
            collector=DAggerCollector(WaitTeacher(), object(), lambda _: True),
            policy=policy,
            aggregate=aggregate,
            grant=grant,
            scenarios=(),
            student_policy_id="student-v1",
            next_student_policy_id="student-v2",
            seed=1,
            iteration=1,
            execution_recipe="recipe",
            native_receipt_verifier=lambda *_: True,
            train_grant_verifier=lambda _: False,
        )

    assert runner_calls == []
    assert aggregate.collected == 0


def test_dagger_fine_tune_warm_starts_and_preserves_live_policy(observed):
    config = BCConfig(
        feature_names=("market.trend_1h", "account.net_exposure"),
        context_bars=2, hidden_size=8, layers=1, batch_size=2, epochs=1, seed=17,
    )
    policy = BCInferencePolicy(new_bc_policy(config), config)
    initial_weights = {key: value.detach().clone() for key, value in policy.model.state_dict().items()}

    expert_observation = observed(episode="expert")
    expert = LabeledStep(
        expert_observation, WaitTeacher().advise(expert_observation), 0.,
        source_identity="source-1",
    )
    student_observation = observed(episode="student", ts=2_000_000_000, pos=.4)
    student_label = LabeledStep(
        student_observation,
        ExpertAdvice("expert", student_observation.decision_ts_ns, "EXIT", 0., ("FLAT",)),
        0., source_identity="source-1",
    )
    aggregate = DAggerSet((expert,))
    aggregate.append(AggregatedLabel(1, "student-policy-1", student_label, "recipe-1"))
    grant = HostTrainGrant(expert_observation.fold_id, "owner", "reservation-1",
                           "source-1", "train", True)

    result = fine_tune_bc_on_daggerset(
        policy=policy, aggregate=aggregate, grant=grant,
        train_grant_verifier=lambda verified: verified == grant,
        expert_fraction=.7, samples_per_epoch=10,
    )

    assert result.sample_count == 2 and result.dagger_sample_count == 1
    assert result.requested_expert_fraction == pytest.approx(.7)
    assert result.sampled_expert_fraction == pytest.approx(.7)
    assert result.samples_per_epoch == 10
    assert result.fold_id == grant.fold_id and result.source_identity == grant.source_identity
    assert result.history and result.policy is not policy
    assert len(result.policy._series) == 0 and result.policy._episode is None
    assert all(torch.equal(policy.model.state_dict()[key], value)
               for key, value in initial_weights.items())
    assert any(not torch.equal(result.policy.model.state_dict()[key], value)
               for key, value in initial_weights.items())


@pytest.mark.parametrize("fraction", [.7, .9, .5])
def test_dagger_mixture_sampler_realizes_preregistered_ratios(observed, fraction):
    obs = observed()
    expert = LabeledStep(
        obs, WaitTeacher().advise(obs), 0., source_identity="source-1",
    )
    student_obs = observed(episode="student", ts=2_000_000_000)
    student = LabeledStep(
        student_obs, WaitTeacher().advise(student_obs), 0.,
        source_identity="source-1", trajectory_id="dagger:1:student:student",
    )
    sampler = DAggerMixtureSampler(
        (expert, student), expert_count=1, expert_fraction=fraction,
        samples_per_epoch=100, seed=17,
    )

    indices = tuple(sampler)

    assert len(indices) == 100
    assert sum(index == 0 for index in indices) == round(100 * fraction)
    replica = DAggerMixtureSampler(
        (expert, student), expert_count=1, expert_fraction=fraction,
        samples_per_epoch=100, seed=17,
    )
    assert tuple(replica) == indices


def test_dagger_fine_tune_requires_verified_matching_train_scope(observed):
    config = BCConfig(
        feature_names=("market.trend_1h", "account.net_exposure"),
        context_bars=2, hidden_size=8, layers=1, batch_size=2, epochs=1,
    )
    policy = BCInferencePolicy(new_bc_policy(config), config)
    obs = observed()
    sample = LabeledStep(obs, WaitTeacher().advise(obs), 0., source_identity="source-1")
    aggregate = DAggerSet((sample,))
    grant = HostTrainGrant(obs.fold_id, "owner", "reservation-1", "source-1", "train", True)

    with pytest.raises(PermissionError, match="did not verify"):
        fine_tune_bc_on_daggerset(
            policy=policy, aggregate=aggregate, grant=grant,
            train_grant_verifier=lambda _grant: False,
        )
    foreign_grant = HostTrainGrant(obs.fold_id, "owner", "reservation-2", "source-2", "train", True)
    with pytest.raises(PermissionError, match="differs"):
        fine_tune_bc_on_daggerset(
            policy=policy, aggregate=aggregate, grant=foreign_grant,
            train_grant_verifier=lambda _grant: True,
        )
