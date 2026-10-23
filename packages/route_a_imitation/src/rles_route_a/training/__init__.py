"""Train-only Route A model fitting entry points."""

from .native_bc import (
    NativeBCBehaviorSummary, NativeBCIndependentEvaluation,
    NativeBCTrainingResult, evaluate_bc_on_registered_train_inner_rollout,
    train_bc_from_registered_native_rollout,
)
from .dagger_trainer import (
    DAggerFineTuneResult, DAggerMixtureSampler, fine_tune_bc_on_daggerset,
)
from .dagger_round import (
    DAggerIterationsError, DAggerIterationsProgress, DAggerIterationsResult,
    DAggerRoundResult,
    run_native_dagger_iterations, run_native_dagger_round,
)

__all__ = [
    "NativeBCBehaviorSummary", "NativeBCIndependentEvaluation",
    "NativeBCTrainingResult", "evaluate_bc_on_registered_train_inner_rollout",
    "train_bc_from_registered_native_rollout",
    "DAggerFineTuneResult", "fine_tune_bc_on_daggerset",
    "DAggerMixtureSampler",
    "DAggerRoundResult", "run_native_dagger_round",
    "DAggerIterationsProgress", "DAggerIterationsError",
    "DAggerIterationsResult", "run_native_dagger_iterations",
]
