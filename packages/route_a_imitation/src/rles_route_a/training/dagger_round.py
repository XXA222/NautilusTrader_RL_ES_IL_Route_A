"""Run one native recovery DAgger round and warm-start its student update."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from rles_route_a.datasets.daggerset import DAggerSet
from rles_route_a.config import DAggerConfig
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.integration.scope import HostTrainGrant

if TYPE_CHECKING:
    from rles_route_a.evaluation.recovery_benchmark import (
        NativeRecoveryEpisode,
        NativeRecoveryRunner,
        RecoveryIterationReport,
        RecoveryScenarioSpec,
    )
    from rles_route_a.training.dagger_collector import DAggerCollector
    from rles_route_a.training.dagger_trainer import DAggerFineTuneResult


@dataclass(frozen=True)
class DAggerRoundResult:
    """Native recovery evidence and the resulting warm-started student."""

    recovery: RecoveryIterationReport
    fine_tune: DAggerFineTuneResult
    previous_student_policy_id: str
    next_student_policy_id: str


@dataclass(frozen=True)
class DAggerIterationsResult:
    """All completed rounds and the policy produced by the configured K loop."""

    rounds: tuple[DAggerRoundResult, ...]
    aggregate: DAggerSet
    final_policy: BCInferencePolicy
    final_student_policy_id: str


@dataclass(frozen=True)
class DAggerIterationsProgress:
    """Resumable state captured before a failed configured iteration."""

    completed_rounds: tuple[DAggerRoundResult, ...]
    aggregate: DAggerSet
    current_policy: BCInferencePolicy
    student_policy_ids: tuple[str, ...]
    next_iteration: int
    dagger_config: DAggerConfig
    grant: HostTrainGrant
    scenarios: tuple[RecoveryScenarioSpec, ...]
    seed: int
    execution_recipe: str
    device: str
    patience: int


class DAggerIterationsError(RuntimeError):
    """An iteration failed without losing the completed actors or aggregate."""

    def __init__(self, progress: DAggerIterationsProgress):
        super().__init__(f"DAgger iteration {progress.next_iteration} failed; resumable progress attached")
        self.progress = progress


def run_native_dagger_round(
        *, runner: NativeRecoveryRunner, collector: DAggerCollector,
        policy: BCInferencePolicy, aggregate: DAggerSet, grant: HostTrainGrant,
        scenarios: Sequence[RecoveryScenarioSpec], student_policy_id: str,
        next_student_policy_id: str, seed: int, iteration: int,
        execution_recipe: str,
        native_receipt_verifier: Callable[
            [NativeRecoveryEpisode, RecoveryScenarioSpec, HostTrainGrant], bool
        ],
        train_grant_verifier: Callable[[HostTrainGrant], bool],
        device: str = "cpu", patience: int = 8,
        dagger_config: DAggerConfig | None = None,
) -> DAggerRoundResult:
    """Collect authorized, host-replayed recovery labels and update the BC actor.

    The recovery iteration verifies its L6 Train grant and native receipts
    before appending labels. Fitting then re-verifies the grant and aggregate
    provenance. The input policy is left untouched; callers promote the
    returned policy and its new identity only after this function succeeds.
    """
    from rles_route_a.evaluation.recovery_benchmark import run_native_recovery_iteration
    from rles_route_a.training.dagger_trainer import fine_tune_bc_on_daggerset
    from rles_route_a.models.bc import BCGRUPolicy

    active_config = DAggerConfig() if dagger_config is None else dagger_config
    if not isinstance(active_config, DAggerConfig):
        raise TypeError("validated Route A DAgger config required")

    if (not isinstance(student_policy_id, str) or not student_policy_id.strip()
            or not isinstance(next_student_policy_id, str)
            or not next_student_policy_id.strip()
            or student_policy_id == next_student_policy_id):
        raise ValueError("recovery and updated student policies need distinct identities")
    if type(iteration) is not int or not 1 <= iteration <= active_config.iterations:
        raise ValueError("DAgger iteration falls outside the frozen configuration")
    if active_config.sample_split != "train":
        raise PermissionError("DAgger round may only collect Train samples")
    if not isinstance(policy, BCInferencePolicy) or not isinstance(policy.model, BCGRUPolicy):
        raise TypeError("an existing Route A GRU BC student policy is required")
    if not isinstance(aggregate, DAggerSet):
        raise TypeError("provenance-bound DAggerSet required")
    if not isinstance(grant, HostTrainGrant) or grant.split != "train":
        raise PermissionError("registered Train grant required for DAgger recovery")
    if (aggregate.fold_id != grant.fold_id
            or aggregate.source_identity != grant.source_identity):
        raise PermissionError("DAgger aggregate differs from the registered Train Fold or source")
    if (not callable(train_grant_verifier)
            or train_grant_verifier(grant) is not True):
        raise PermissionError("host authority did not verify the Train grant for DAgger recovery")

    aggregate_checkpoint = aggregate.collected
    working_aggregate = aggregate.fork()
    recovery = run_native_recovery_iteration(
        runner=runner,
        collector=collector,
        grant=grant,
        scenarios=scenarios,
        student_policy=policy,
        student_policy_id=student_policy_id,
        seed=seed,
        iteration=iteration,
        aggregate=working_aggregate,
        execution_recipe=execution_recipe,
        native_receipt_verifier=native_receipt_verifier,
    )
    if recovery.dagger_samples <= 0:
        raise ValueError("native recovery produced no DAgger labels; student was not updated")

    fine_tune = fine_tune_bc_on_daggerset(
        policy=policy,
        aggregate=working_aggregate,
        grant=grant,
        train_grant_verifier=train_grant_verifier,
        device=device,
        patience=patience,
        expert_fraction=active_config.expert_fraction,
    )
    aggregate.extend(working_aggregate.added_since(aggregate_checkpoint))
    return DAggerRoundResult(
        recovery=recovery,
        fine_tune=fine_tune,
        previous_student_policy_id=student_policy_id,
        next_student_policy_id=next_student_policy_id,
    )


def run_native_dagger_iterations(
        *, runner: NativeRecoveryRunner, collector: DAggerCollector,
        grant: HostTrainGrant,
        scenarios: Sequence[RecoveryScenarioSpec], student_policy_ids: Sequence[str],
        seed: int, execution_recipe: str,
        native_receipt_verifier: Callable[
            [NativeRecoveryEpisode, RecoveryScenarioSpec, HostTrainGrant], bool
        ],
        train_grant_verifier: Callable[[HostTrainGrant], bool],
        dagger_config: DAggerConfig,
        device: str = "cpu", patience: int = 8,
        policy: BCInferencePolicy | None = None, aggregate: DAggerSet | None = None,
        progress: DAggerIterationsProgress | None = None,
) -> DAggerIterationsResult:
    """Execute the configured K-round DAgger loop, warm-starting each actor.

    The caller supplies a distinct registered identity for the initial actor
    and each resulting actor. Every native runner receives the exact preceding
    round's policy object along with that identity.
    """
    if not isinstance(dagger_config, DAggerConfig):
        raise TypeError("validated Route A DAgger config required")
    identities = tuple(student_policy_ids)
    if (len(identities) != dagger_config.iterations + 1
            or any(not isinstance(item, str) or not item.strip() for item in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError("provide one distinct student policy identity for the initial actor and each round")

    if progress is None:
        if not isinstance(policy, BCInferencePolicy) or not isinstance(aggregate, DAggerSet):
            raise TypeError("initial BC policy and DAgger aggregate are required")
        current_policy, active_aggregate = policy, aggregate
        results: list[DAggerRoundResult] = []
        start_iteration = 1
    else:
        if not isinstance(progress, DAggerIterationsProgress):
            raise TypeError("valid DAgger iteration progress required")
        if (progress.student_policy_ids != identities
                or progress.dagger_config != dagger_config
                or progress.grant != grant
                or progress.scenarios != tuple(scenarios)
                or progress.seed != seed
                or progress.execution_recipe != execution_recipe
                or progress.device != device
                or progress.patience != patience
                or (policy is not None and policy is not progress.current_policy)
                or (aggregate is not None and aggregate is not progress.aggregate)):
            raise ValueError("resume inputs differ from the failed DAgger run")
        if not 1 <= progress.next_iteration <= dagger_config.iterations:
            raise ValueError("DAgger progress has no remaining iteration")
        current_policy, active_aggregate = progress.current_policy, progress.aggregate
        results = list(progress.completed_rounds)
        start_iteration = progress.next_iteration

    for iteration in range(start_iteration, dagger_config.iterations + 1):
        try:
            result = run_native_dagger_round(
                runner=runner,
                collector=collector,
                policy=current_policy,
                aggregate=active_aggregate,
                grant=grant,
                scenarios=scenarios,
                student_policy_id=identities[iteration - 1],
                next_student_policy_id=identities[iteration],
                seed=seed,
                iteration=iteration,
                execution_recipe=execution_recipe,
                native_receipt_verifier=native_receipt_verifier,
                train_grant_verifier=train_grant_verifier,
                device=device,
                patience=patience,
                dagger_config=dagger_config,
            )
        except Exception as error:
            failed_progress = DAggerIterationsProgress(
                completed_rounds=tuple(results),
                aggregate=active_aggregate,
                current_policy=current_policy,
                student_policy_ids=identities,
                next_iteration=iteration,
                dagger_config=dagger_config,
                grant=grant,
                scenarios=tuple(scenarios),
                seed=seed,
                execution_recipe=execution_recipe,
                device=device,
                patience=patience,
            )
            raise DAggerIterationsError(failed_progress) from error
        results.append(result)
        current_policy = result.fine_tune.policy
    return DAggerIterationsResult(
        rounds=tuple(results), aggregate=active_aggregate, final_policy=current_policy,
        final_student_policy_id=identities[-1],
    )
