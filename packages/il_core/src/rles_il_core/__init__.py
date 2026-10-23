"""Route A offline imitation learning built around native Nautilus episodes."""

from .bc import (
    BCBehaviorReport, BCCandidateResult, BCCandidateSweepResult, BCNativeRolloutResult, BCComparisonPoint,
    BCComparisonResult, BCCandidateSweepBlocked, BCFinetuneBlocked, BCFinetuneResult,
    FourArmNativeComparisonResult, FrozenBCPolicy,
    behavior_report, compare_bc_to_random_ppo, compare_four_native_arms,
    evaluate_bc_candidates, evaluate_bc_native_rollouts,
    finetune_bc_actor, prepare_bc_pairs,
    publish_bc_artifact, train_behavioral_clone,
)
from .collector import collect_episode, collect_episodes
from .diagnostics import native_rollout_report
from .dataset import (
    EpisodeTrajectory, Transition, assign_time_blocks, build_dataset_manifest,
    export_d3rlpy_npz, read_parquet_dataset, to_offline_episode_arrays, validate_trajectory,
    verify_parquet_dataset, write_parquet_dataset,
)
from .experts import (
    E1TrendExpert, E2Config, E2ReversionExpert, E3Config, E3CostAwareExpert,
    ExpertPolicy, FixedExposureExpert, FlatExpert, FrozenVisibleContext, RecoveryExpert,
)
from .iql import (
    FrozenIQLPolicy, IQLNativeRolloutResult, evaluate_iql_native_rollouts,
    verify_iql_export_parity,
)

__all__ = [
    "BCBehaviorReport", "BCCandidateResult", "BCCandidateSweepBlocked", "BCCandidateSweepResult", "BCNativeRolloutResult", "BCComparisonPoint", "BCComparisonResult", "BCFinetuneBlocked", "BCFinetuneResult", "FourArmNativeComparisonResult", "E1TrendExpert", "EpisodeTrajectory", "ExpertPolicy", "FlatExpert",
    "E2Config", "E2ReversionExpert", "E3Config", "E3CostAwareExpert",
    "FixedExposureExpert", "FrozenBCPolicy", "FrozenVisibleContext", "RecoveryExpert", "Transition", "assign_time_blocks",
    "native_rollout_report",
    "FrozenIQLPolicy", "IQLNativeRolloutResult", "evaluate_iql_native_rollouts",
    "verify_iql_export_parity",
    "behavior_report", "build_dataset_manifest", "compare_bc_to_random_ppo", "compare_four_native_arms", "evaluate_bc_candidates", "evaluate_bc_native_rollouts", "finetune_bc_actor", "collect_episode", "collect_episodes", "export_d3rlpy_npz",
    "prepare_bc_pairs", "publish_bc_artifact", "read_parquet_dataset",
    "to_offline_episode_arrays", "train_behavioral_clone", "validate_trajectory",
    "verify_parquet_dataset", "write_parquet_dataset",
]


