"""Train-inner behavioral cloning on reviewed native teacher trajectories."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np

from .dataset import EpisodeTrajectory, validate_trajectory
from .experts import FrozenVisibleContext


@dataclass(frozen=True, slots=True)
class BCBehaviorReport:
    """Offline fit and action-coverage diagnostics; never an economic result."""

    episodes: int
    transitions: int
    observation_dim: int
    mse: float
    mae: float
    direction_accuracy: float
    label_nonzero_fraction: float
    prediction_nonzero_fraction: float
    label_deadband_fraction: float
    prediction_deadband_fraction: float
    label_mean: float
    prediction_mean: float
    label_std: float
    prediction_std: float
    target_flip_count: int
    prediction_flip_count: int
    deadband: float
    collapsed_to_flat: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BCCandidateResult:
    """One frozen recipe evaluated on the reserved Train-inner time block."""

    candidate_id: str
    seed: int
    epochs: int
    learning_rate: float
    config_hash: str
    behavior_clone_parity: Any
    fit_behavior: BCBehaviorReport
    selection_behavior: BCBehaviorReport


@dataclass(frozen=True, slots=True)
class BCCandidateSweepResult:
    """Finite Train-inner BC sweep with an immutable selected actor artifact."""

    candidates: tuple[BCCandidateResult, ...]
    selected_candidate_id: str
    selection_rule: str
    selection_window_ns: tuple[int, int]
    artifact_path: str
    selection_plan_path: str
    selection_plan_sha256: str


@dataclass(frozen=True, slots=True)
class BCNativeRolloutResult:
    """Hashed immutable behavior and native account summary from independent replay."""

    actor_sha256: str
    report_path: str
    report_sha256: str
    episode_count: int
    native_report: Mapping[str, Any]


class BCCandidateSweepBlocked(ValueError):
    """The preregistered BC sweep produced no non-collapsed candidate."""

    def __init__(self, candidates: tuple[BCCandidateResult, ...], report_path: Path) -> None:
        super().__init__("all frozen BC candidates collapsed to the flat-action deadband on selection data")
        self.candidates = candidates
        self.report_path = report_path


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-volume rename is atomic and non-overwriting on Windows, where
        # hard-link creation may be unavailable (for example, restricted
        # desktop sandboxes). POSIX uses link's atomic no-replace semantics.
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(encoded).hexdigest()


def prepare_bc_pairs(
    trajectories: tuple[EpisodeTrajectory, ...] | list[EpisodeTrajectory],
    *,
    deadband: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, tuple[EpisodeTrajectory, ...]]:
    """Flatten only complete Train-inner episodes without filtering any rows."""
    if not trajectories:
        raise ValueError("BC requires at least one complete Train-inner episode")
    if not math.isfinite(deadband) or not 0 <= deadband < 1:
        raise ValueError("deadband must be finite and in [0, 1)")
    rows = tuple(trajectories)
    frozen_identities: dict[str, set[str]] = {
        key: set() for key in ("dataset_hash", "fold_hash", "config_hash", "transition_hash", "schema_hash")
    }
    dims: set[int] = set()
    obs_rows: list[tuple[float, ...]] = []
    action_rows: list[float] = []
    for trajectory in rows:
        validate_trajectory(trajectory)
        if trajectory.episode_metadata.get("split") != "train_inner":
            raise ValueError("BC accepts Train-inner episodes only; validation/test leakage blocked")
        if trajectory.episode_metadata.get("expert_id") != trajectory.expert_id:
            raise ValueError("episode expert metadata does not match trajectory identity")
        identity = trajectory.transitions[0].native_identity
        for key, value in identity.items():
            frozen_identities[key].add(value)
        dims.add(len(trajectory.observations[0]))
        for transition in trajectory.transitions:
            # The canonical Transition.action is the expert's requested target,
            # before the environment applies risk or execution constraints.
            obs_rows.append(transition.observation)
            action_rows.append(float(transition.action))
    stable_keys = ("dataset_hash", "config_hash", "schema_hash")
    if any(len(frozen_identities[key]) != 1 for key in stable_keys) or len(dims) != 1:
        raise ValueError("BC cannot mix dataset/config/schema identities or dimensions")
    window_identities = (frozen_identities["fold_hash"],
                         frozen_identities["transition_hash"])
    recipes = {row.episode_metadata.get("execution_recipe_hash") for row in rows}
    if (len(recipes) != 1 or (None in recipes and any(len(values) != 1
                                                    for values in window_identities))):
        raise ValueError("BC mixed windows require one frozen execution recipe hash")
    recipe = next(iter(recipes))
    if recipe is not None and (not isinstance(recipe, str) or len(recipe) != 64
                               or any(c not in "0123456789abcdef" for c in recipe)):
        raise ValueError("BC execution recipe hash is malformed")
    observations = np.asarray(obs_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.float32).reshape((-1, 1))
    if (not len(observations) or not np.isfinite(observations).all()
            or not np.isfinite(actions).all() or (np.abs(actions) > 1).any()):
        raise ValueError("BC input contains empty, non-finite, or out-of-range samples")
    return observations, actions, rows


def _reset_ppo_arm_random_stream(provider: Any) -> None:
    """Start each fresh PPO arm from its own declared algorithm/replay seeds."""
    model = getattr(provider, "model", None)
    seeds = getattr(getattr(provider, "config", None), "seeds", None)
    if (model is None or int(getattr(model, "num_timesteps", -1)) != 0
            or seeds is None or not callable(getattr(model, "set_random_seed", None))):
        raise ValueError("fresh seeded PPO provider required for an independent arm")
    model.set_random_seed(seeds.algorithm)
    np.random.seed(seeds.replay)


def behavior_report(
    trajectories: tuple[EpisodeTrajectory, ...] | list[EpisodeTrajectory],
    predictions: np.ndarray,
    *,
    deadband: float = 0.05,
) -> BCBehaviorReport:
    """Report fit and activity separately so a low-MSE flat actor is visible."""
    observations, labels, rows = prepare_bc_pairs(trajectories, deadband=deadband)
    pred = np.asarray(predictions, dtype=np.float32)
    if pred.shape != labels.shape or not np.isfinite(pred).all() or (np.abs(pred) > 1.000001).any():
        raise ValueError("predictions must be finite bounded float32 actions aligned with labels")
    error = pred.astype(np.float64) - labels.astype(np.float64)
    label_flat, pred_flat = labels[:, 0], pred[:, 0]
    warnings: list[str] = []
    collapsed = bool(np.all(np.abs(pred_flat) <= deadband))
    if collapsed:
        warnings.append("actor is within the configured deadband for every training state")
    if float(np.mean(np.abs(label_flat) > deadband)) < 0.05:
        warnings.append("teacher labels contain less than 5% non-deadband actions")
    pred_sign = np.where(np.abs(pred_flat) <= deadband, 0, np.sign(pred_flat))
    label_sign = np.where(np.abs(label_flat) <= deadband, 0, np.sign(label_flat))
    flips_label = flips_pred = 0
    offset = 0
    for trajectory in rows:
        count = len(trajectory.transitions)
        flips_label += int(np.count_nonzero(np.diff(label_sign[offset:offset + count])))
        flips_pred += int(np.count_nonzero(np.diff(pred_sign[offset:offset + count])))
        offset += count
    return BCBehaviorReport(
        episodes=len(rows), transitions=len(labels), observation_dim=observations.shape[1],
        mse=float(np.mean(np.square(error))), mae=float(np.mean(np.abs(error))),
        direction_accuracy=float(np.mean(np.sign(label_flat) == np.sign(pred_flat))),
        label_nonzero_fraction=float(np.mean(np.abs(label_flat) > deadband)),
        prediction_nonzero_fraction=float(np.mean(np.abs(pred_flat) > deadband)),
        label_deadband_fraction=float(np.mean(np.abs(label_flat) <= deadband)),
        prediction_deadband_fraction=float(np.mean(np.abs(pred_flat) <= deadband)),
        label_mean=float(np.mean(label_flat)), prediction_mean=float(np.mean(pred_flat)),
        label_std=float(np.std(label_flat)), prediction_std=float(np.std(pred_flat)),
        target_flip_count=flips_label, prediction_flip_count=flips_pred,
        deadband=float(deadband), collapsed_to_flat=collapsed, warnings=tuple(warnings),
    )


def train_behavioral_clone(
    provider: Any,
    trajectories: tuple[EpisodeTrajectory, ...] | list[EpisodeTrajectory],
    *,
    epochs: int,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    tolerance: float = 1e-3,
    early_stop: bool = False,
    deadband: float = 0.05,
) -> tuple[Any, BCBehaviorReport]:
    """Fit the pristine existing L4 actor in the frozen policy-input domain."""
    config = getattr(provider, "config", None)
    model = getattr(provider, "model", None)
    if config is None or model is None:
        raise ValueError("a created, pristine RLProvider is required for BC")
    if bool(getattr(config, "normalize_observation", True)):
        raise ValueError("BC requires frozen policy-domain observations; disable L4 VecNormalize")
    if int(getattr(model, "num_timesteps", -1)) != 0:
        raise ValueError("BC must initialize an actor before any RL environment updates")
    observations, labels, rows = prepare_bc_pairs(trajectories, deadband=deadband)
    if observations.shape[1] != config.policy.observation_dim:
        raise ValueError("BC observation width differs from the frozen L4 policy schema")
    parity = provider.distill(
        observations, labels, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, tolerance=tolerance, early_stop=early_stop,
    )
    predicted, _ = model.predict(observations, deterministic=True)
    report = behavior_report(rows, np.asarray(predicted, dtype=np.float32).reshape((-1, 1)),
                             deadband=deadband)
    return parity, report


def evaluate_bc_candidates(
    provider_factory: Callable[[int], Any],
    fit_trajectories: tuple[EpisodeTrajectory, ...] | list[EpisodeTrajectory],
    selection_trajectories: tuple[EpisodeTrajectory, ...] | list[EpisodeTrajectory],
    *,
    selection_window_ns: tuple[int, int],
    artifact_path: str | Path,
    dataset_manifest: Mapping[str, Any],
    epochs_candidates: tuple[int, ...] = (10, 30, 60),
    learning_rate_candidates: tuple[float, ...] = (1e-4, 3e-4),
    seeds: tuple[int, ...] = (11, 23, 37),
    deadband: float = 0.05,
) -> BCCandidateSweepResult:
    """Run the preregistered finite BC grid and publish its Train-inner winner.

    The fit episodes must precede the reserved selection block. Selection uses
    validation-block MAE among non-collapsed actors, with MSE and stable
    candidate identity as tie breakers; all behavior diagnostics are retained.
    This chooses an imitation candidate, not an economic strategy.
    """
    if not callable(provider_factory):
        raise ValueError("a seed-indexed L4 provider factory is required")
    if (type(selection_window_ns) is not tuple or len(selection_window_ns) != 2
            or any(type(value) is not int for value in selection_window_ns)
            or selection_window_ns[0] <= 0 or selection_window_ns[1] <= selection_window_ns[0]):
        raise ValueError("a valid reserved Train-inner [start_ns, end_ns) selection window is required")
    allowed_epochs = (10, 30, 60)
    allowed_rates = (1e-4, 3e-4)
    allowed_seeds = (11, 23, 37)
    if (not epochs_candidates or len(set(epochs_candidates)) != len(epochs_candidates)
            or any(type(value) is not int or value not in allowed_epochs for value in epochs_candidates)
            or not learning_rate_candidates
            or len(set(learning_rate_candidates)) != len(learning_rate_candidates)
            or any(type(value) not in (int, float) or value not in allowed_rates
                   for value in learning_rate_candidates)
            or not seeds or len(set(seeds)) != len(seeds)
            or any(type(value) is not int or value not in allowed_seeds for value in seeds)):
        raise ValueError("BC sweep may use only unique candidates from the frozen RFC grid")
    if not math.isfinite(deadband) or not 0 <= deadband < 1:
        raise ValueError("deadband must be finite and in [0, 1)")
    fit_rows = tuple(fit_trajectories)
    selection_rows = tuple(selection_trajectories)
    if not fit_rows or not selection_rows:
        raise ValueError("both fit and reserved Train-inner selection episodes are required")
    # Validate every episode and all frozen source/schema identities together.
    fit_obs, fit_actions, _ = prepare_bc_pairs(fit_rows, deadband=deadband)
    selection_obs, _, _ = prepare_bc_pairs(selection_rows, deadband=deadband)
    _, _, all_rows = prepare_bc_pairs((*fit_rows, *selection_rows), deadband=deadband)
    identity = dict(all_rows[0].transitions[0].native_identity)
    expected_folds = {row.transitions[0].native_identity["fold_hash"] for row in all_rows}
    expected_transitions = {row.transitions[0].native_identity["transition_hash"]
                            for row in all_rows}
    expected_recipe = all_rows[0].episode_metadata.get("execution_recipe_hash")
    manifest_range = dataset_manifest.get("allowed_time_range_ns") if isinstance(dataset_manifest, Mapping) else None
    manifest_seeds = dataset_manifest.get("seeds") if isinstance(dataset_manifest, Mapping) else None
    manifest_experts = dataset_manifest.get("expert_ids") if isinstance(dataset_manifest, Mapping) else None
    fit_windows: set[tuple[int, int]] = set()
    selection_windows: set[tuple[int, int]] = set()
    for row in fit_rows:
        window = row.transitions[0].episode_window
        start, end = window["start_ns"], window["end_ns"]
        if end > selection_window_ns[0]:
            raise ValueError("BC fit episodes must end before the reserved selection block")
        fit_windows.add((start, end))
    for row in selection_rows:
        window = row.transitions[0].episode_window
        start, end = window["start_ns"], window["end_ns"]
        if not (selection_window_ns[0] <= start < end < selection_window_ns[1]):
            raise ValueError("selection episodes must lie entirely inside the frozen Train-inner block")
        selection_windows.add((start, end))
    if fit_windows & selection_windows:
        raise ValueError("fit and selection blocks contain a shared market window")
    fold_hashes = dataset_manifest.get("fold_hashes") if isinstance(dataset_manifest, Mapping) else None
    if (not isinstance(dataset_manifest, Mapping)
            or not isinstance(fold_hashes, (tuple, list))
            or any(not isinstance(value, str) for value in fold_hashes)
            or dataset_manifest.get("split") != "train_inner"
            or dataset_manifest.get("account_truth") != "nautilus_native"
            or dataset_manifest.get("source_data_hash") != identity["dataset_hash"]
            or dataset_manifest.get("source_environment_config_hash") != identity["config_hash"]
            or (set(dataset_manifest.get("transition_hashes", ())) != expected_transitions
                if "transition_hashes" in dataset_manifest else
                (len(expected_transitions) != 1
                 or dataset_manifest.get("transition_hash") not in expected_transitions))
            or (dataset_manifest.get("execution_recipe_hash") != expected_recipe
                if expected_recipe is not None else len(expected_transitions) != 1)
            or dataset_manifest.get("observation_schema_hash") != identity["schema_hash"]
            or set(fold_hashes) != expected_folds
            or dataset_manifest.get("all_episodes_retained") is not True
            or not isinstance(manifest_range, (tuple, list)) or len(manifest_range) != 2
            or any(type(value) is not int for value in manifest_range)
            or not isinstance(manifest_seeds, (tuple, list))
            or any(type(value) is not int for value in manifest_seeds)
            or not isinstance(manifest_experts, (tuple, list))
            or any(not isinstance(value, str) for value in manifest_experts)):
        raise ValueError("dataset manifest does not identify the frozen Train-inner BC trajectories")
    episode_seeds = {row.episode_metadata.get("seed") for row in all_rows}
    window_start = min(row.transitions[0].episode_window["start_ns"] for row in all_rows)
    window_end = max(row.transitions[0].episode_window["end_ns"] for row in all_rows)
    if (any(type(seed) is not int for seed in episode_seeds)
            or not episode_seeds <= set(manifest_seeds)
            or not {row.expert_id for row in all_rows} <= set(manifest_experts)
            or manifest_range[0] > window_start or manifest_range[1] < window_end):
        raise ValueError("dataset manifest omits BC episode seed, expert, or time-range coverage")
    if fit_obs.shape[1] != selection_obs.shape[1]:
        raise ValueError("fit and selection observations must share the frozen policy schema")

    target = Path(artifact_path).resolve()
    metadata_path = target.with_suffix(".json")
    selection_plan_path = target.with_suffix(".selection.json")
    if target.suffix != ".zip":
        raise ValueError("selected BC actor artifact path must end in .zip")
    if target.exists() or metadata_path.exists() or selection_plan_path.exists():
        raise FileExistsError("BC candidate plans and actor artifacts are immutable")
    target.parent.mkdir(parents=True, exist_ok=True)

    recipe_hash: str | None = None
    selection_plan_hash: str | None = None
    candidate_rows: list[BCCandidateResult] = []
    best_key: tuple[float, float, str] | None = None
    best_candidate_id: str | None = None
    best_state: dict[str, Any] | None = None
    best_fit_report: BCBehaviorReport | None = None
    best_parity: Any = None
    for seed in seeds:
        for epochs in epochs_candidates:
            for learning_rate in learning_rate_candidates:
                provider = provider_factory(seed)
                try:
                    provider.create()
                    config, model = provider.config, provider.model
                    if (config.algorithm not in {"PPO", "SAC"} or config.normalize_observation
                            or config.seeds.algorithm != seed or model is None
                            or int(model.num_timesteps) != 0
                            or config.policy.observation_dim != fit_obs.shape[1]):
                        raise ValueError("BC candidates require pristine PPO/SAC actors in the frozen raw policy domain")
                    recipe = {
                        "algorithm": config.algorithm, "policy": asdict(config.policy),
                        "fingerprint": asdict(config.fingerprint),
                        "hyperparameters": dict(config.hyperparameters), "device": config.device,
                        "noise_sigma": config.noise_sigma,
                        "cuda_allocator_fraction": config.cuda_allocator_fraction,
                        "normalize_observation": config.normalize_observation,
                        "normalize_reward": config.normalize_reward, "version": config.version,
                    }
                    observed_recipe_hash = sha256(json.dumps(
                        recipe, sort_keys=True, separators=(",", ":"), allow_nan=False,
                    ).encode("utf-8")).hexdigest()
                    if recipe_hash is None:
                        recipe_hash = observed_recipe_hash
                        preregistered = {
                            "schema": "rles.il.bc_sweep_plan.v1",
                            "dataset_manifest_sha256": sha256(json.dumps(
                                dataset_manifest, sort_keys=True, separators=(",", ":"),
                                allow_nan=False,
                            ).encode("utf-8")).hexdigest(),
                            "dataset_hash": identity["dataset_hash"],
                            "fold_hash": identity["fold_hash"],
                            "transition_hash": identity["transition_hash"],
                            "observation_schema_hash": identity["schema_hash"],
                            "l4_recipe_hash_excluding_seed": observed_recipe_hash,
                            "algorithm": config.algorithm,
                            "fit_window_ns": [
                                min(start for start, _ in fit_windows),
                                max(end for _, end in fit_windows),
                            ],
                            "selection_window_ns": list(selection_window_ns),
                            "epochs_candidates": list(epochs_candidates),
                            "learning_rate_candidates": [float(value) for value in learning_rate_candidates],
                            "seeds": list(seeds), "batch_size": 256,
                            "action_validation_tolerance": 1e-3, "deadband": float(deadband),
                            "selection_rule": (
                                "minimum_selection_mae_among_noncollapsed; then_mse; then_candidate_id"
                            ),
                            "economic_performance_claim": False,
                        }
                        selection_plan_hash = _write_immutable_json(selection_plan_path, preregistered)
                    elif observed_recipe_hash != recipe_hash:
                        raise ValueError("BC candidate provider recipes differ beyond the frozen seed")
                    parity, fit_report = train_behavioral_clone(
                        provider, fit_rows, epochs=epochs, batch_size=256,
                        learning_rate=float(learning_rate), tolerance=1e-3,
                        early_stop=False, deadband=deadband,
                    )
                    predicted, _ = model.predict(selection_obs, deterministic=True)
                    selection_report = behavior_report(
                        selection_rows, np.asarray(predicted, dtype=np.float32).reshape((-1, 1)),
                        deadband=deadband,
                    )
                    identity_payload = {
                        "recipe_hash": observed_recipe_hash, "config_hash": config.hash,
                        "seed": seed, "epochs": epochs, "learning_rate": float(learning_rate),
                        "dataset_hash": identity["dataset_hash"], "fold_hash": identity["fold_hash"],
                        "selection_window_ns": selection_window_ns,
                    }
                    candidate_id = sha256(json.dumps(
                        identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
                    ).encode("utf-8")).hexdigest()
                    result = BCCandidateResult(
                        candidate_id=candidate_id, seed=seed, epochs=epochs,
                        learning_rate=float(learning_rate), config_hash=config.hash,
                        behavior_clone_parity=parity, fit_behavior=fit_report,
                        selection_behavior=selection_report,
                    )
                    candidate_rows.append(result)
                    if not selection_report.collapsed_to_flat:
                        score = (selection_report.mae, selection_report.mse, candidate_id)
                        if best_key is None or score < best_key:
                            best_key = score
                            best_candidate_id = candidate_id
                            best_state = {
                                key: tensor.detach().cpu().clone()
                                for key, tensor in model.policy.state_dict().items()
                            }
                            best_fit_report, best_parity = fit_report, parity
                finally:
                    provider.close()
    if best_candidate_id is None or best_state is None or best_fit_report is None:
        failure_path = target.with_suffix(".failed.json")
        _write_immutable_json(failure_path, {
            "schema": "rles.il.bc_sweep_failure.v1", "status": "FAILED",
            "reason": "all_candidates_collapsed_to_flat_on_reserved_selection_block",
            "selection_plan_sha256": selection_plan_hash,
            "candidates": [asdict(row) for row in candidate_rows],
            "economic_performance_claim": False,
        })
        raise BCCandidateSweepBlocked(tuple(candidate_rows), failure_path)

    selected = next(row for row in candidate_rows if row.candidate_id == best_candidate_id)
    final_provider = provider_factory(selected.seed)
    try:
        final_provider.create()
        if (final_provider.config.hash != selected.config_hash or final_provider.model is None
                or int(final_provider.model.num_timesteps) != 0):
            raise RuntimeError("selected BC artifact provider differs from its frozen sweep recipe")
        final_provider.model.policy.load_state_dict(best_state, strict=True)
        path = publish_bc_artifact(
            final_provider, best_fit_report, target, parity=best_parity,
            dataset_manifest=dataset_manifest,
            training={
                "method": "train_inner_finite_candidate_sweep",
                "selection_plan_sha256": selection_plan_hash,
                "selection_rule": "minimum_selection_mae_among_noncollapsed; then_mse; then_candidate_id",
                "selection_window_ns": list(selection_window_ns),
                "selected_candidate_id": best_candidate_id,
                "fit_transition_count": int(len(fit_actions)),
                "selection_transition_count": int(len(selection_obs)),
                "batch_size": 256, "action_validation_tolerance": 1e-3,
                "candidates": [asdict(row) for row in candidate_rows],
                "economic_performance_claim": False,
            },
        )
    finally:
        final_provider.close()
    return BCCandidateSweepResult(
        candidates=tuple(candidate_rows), selected_candidate_id=best_candidate_id,
        selection_rule="minimum_selection_mae_among_noncollapsed; then_mse; then_candidate_id",
        selection_window_ns=selection_window_ns, artifact_path=str(path),
        selection_plan_path=str(selection_plan_path),
        selection_plan_sha256=str(selection_plan_hash),
    )


@dataclass(frozen=True, slots=True)
class BCFinetuneResult:
    """BC initialization and a bounded L4 RL continuation, without economics claims."""

    behavior_clone_parity: Any
    behavior_before_rl: BCBehaviorReport
    behavior_after_rl: BCBehaviorReport
    training_performance: Any
    rl_steps: int


class BCFinetuneBlocked(ValueError):
    """BC failed a prerequisite action-quality gate before RL updates."""

    def __init__(self, parity: Any, report: BCBehaviorReport,
                 *, reason: str = "BC actor collapsed into the configured flat-action deadband") -> None:
        super().__init__(f"{reason}; RL fine-tuning refused")
        self.parity = parity
        self.report = report


def _require_bc_action_gate(parity: Any, report: BCBehaviorReport, *, arm: str) -> None:
    if report.collapsed_to_flat:
        raise BCFinetuneBlocked(parity, report)
    if not getattr(parity, "passed", False):
        raise BCFinetuneBlocked(
            parity, report,
            reason=f"{arm} action parity did not meet the frozen tolerance",
        )


@dataclass(frozen=True, slots=True)
class BCComparisonPoint:
    """Matched sample-efficiency point; diagnostics are not economic returns."""

    samples: int
    random_performance: Any
    bc_performance: Any
    random_checkpoint: str
    bc_checkpoint: str
    random_checkpoint_max_abs_error: float
    bc_checkpoint_max_abs_error: float
    random_behavior: BCBehaviorReport
    bc_behavior: BCBehaviorReport


@dataclass(frozen=True, slots=True)
class BCComparisonResult:
    """Paired random-actor and BC-actor PPO experiment on one frozen L4 recipe."""

    config_hash: str
    transition_hash: str
    seed: int
    behavior_clone_parity: Any
    behavior_before_rl: BCBehaviorReport
    points: tuple[BCComparisonPoint, ...]


@dataclass(frozen=True, slots=True)
class FourArmNativeComparisonResult:
    """Immutable scenario-only comparison of the required Route A policy arms."""

    sample_budget: int
    config_hash: str
    transition_hash: str
    report_path: str
    report_sha256: str
    paired_episode_identity_sha256: str
    native_report: Mapping[str, Any]


def compare_four_native_arms(
    provider_factory: Callable[[], Any],
    trajectories: tuple[EpisodeTrajectory, ...] | list[EpisodeTrajectory],
    *,
    selection_trajectories: tuple[EpisodeTrajectory, ...] | list[EpisodeTrajectory],
    selection_window_ns: tuple[int, int],
    selection_purge_ns: int | None = None,
    rule_teacher_factory: Callable[[], Any],
    native_env_factory: Any,
    native_seeds: tuple[int, ...],
    report_path: str | Path,
    rl_steps: int,
    bc_epochs: int,
    bc_batch_size: int = 256,
    bc_learning_rate: float = 1e-4,
    parity_tolerance: float = 1e-3,
    deadband: float = 0.05,
    episode_options: tuple[dict[str, Any] | None, ...] | None = None,
) -> FourArmNativeComparisonResult:
    """Run rule-teacher, pure-BC, pure-PPO, and BC-to-PPO under one native plan.

    BC quality gates run before either PPO arm receives an environment sample.
    The two PPO arms then receive the exact same sample budget.  Every policy is
    finally replayed on fresh native accounts with the same seeds and episode
    options.  The result is descriptive Train-inner/scenario evidence only.
    """
    if (not callable(provider_factory) or not callable(rule_teacher_factory)
            or not callable(getattr(native_env_factory, "make_env", None))):
        raise ValueError("provider, rule teacher, and native environment factories are required")
    if getattr(native_env_factory, "split", None) != "train_inner":
        raise ValueError("four-arm comparison is restricted to Train-inner native environments")
    if (type(native_seeds) is not tuple or not native_seeds
            or any(type(seed) is not int or seed < 0 for seed in native_seeds)
            or len(set(native_seeds)) != len(native_seeds)):
        raise ValueError("unique nonnegative native comparison seeds are required")
    if type(rl_steps) is not int or rl_steps <= 0:
        raise ValueError("positive equal PPO sample budget required")
    destination = Path(report_path).resolve()
    if destination.suffix != ".json":
        raise ValueError("four-arm comparison report path must end in .json")

    # Validate disjoint Train-inner fit and selection trajectories before any
    # provider is created or any environment is sampled.
    observations, _actions, rows = prepare_bc_pairs(trajectories, deadband=deadband)
    selection_observations, selection_actions, selection_rows = prepare_bc_pairs(
        selection_trajectories, deadband=deadband,
    )
    all_rows = prepare_bc_pairs((*rows, *selection_rows), deadband=deadband)[2]
    if (type(selection_window_ns) is not tuple or len(selection_window_ns) != 2
            or any(type(value) is not int for value in selection_window_ns)
            or selection_window_ns[0] <= 0
            or selection_window_ns[1] <= selection_window_ns[0]):
        raise ValueError("a frozen Train-inner BC selection window is required")
    factory_features = getattr(native_env_factory, "features", None)
    required_purge = getattr(factory_features, "max_lookback_ns", None)
    if required_purge is not None:
        if type(required_purge) is not int or required_purge < 0:
            raise ValueError("native Train-inner feature lookback must be nonnegative nanoseconds")
        if selection_purge_ns is not None and selection_purge_ns != required_purge:
            raise ValueError("BC selection purge differs from the frozen feature lookback")
        selection_purge_ns = required_purge
    if type(selection_purge_ns) is not int or selection_purge_ns < 0:
        raise ValueError("the independent BC selection feature-lookback purge is required")
    fit_windows = [row.transitions[0].episode_window for row in rows]
    selection_windows = [row.transitions[0].episode_window for row in selection_rows]
    if (max(window["end_ns"] for window in fit_windows)
            + selection_purge_ns > selection_window_ns[0]
            or any(not (selection_window_ns[0] <= window["start_ns"]
                        < window["end_ns"] <= selection_window_ns[1])
                   for window in selection_windows)
            or {row.expert_id for row in rows} != {row.expert_id for row in selection_rows}):
        raise ValueError(
            "BC fit and selection trajectories overlap, escape the reserved window, "
            "or use different expert identities"
        )
    if (selection_observations.shape[1] != observations.shape[1]
            or len(all_rows) != len(rows) + len(selection_rows)):
        raise ValueError("fit and selection trajectories differ in frozen input identity")
    factory_start, factory_end = (getattr(native_env_factory, "start_ns", None),
                                  getattr(native_env_factory, "end_ns", None))
    if factory_start is not None or factory_end is not None:
        if (factory_start, factory_end) != selection_window_ns:
            raise ValueError("native four-arm replay factory must match the reserved BC selection window")
    # The host may expose its exact frozen Train-inner window range separately.
    allowed_range = getattr(native_env_factory, "allowed_train_inner_range_ns", None)
    if (allowed_range is not None
            and (not isinstance(allowed_range, (tuple, list)) or len(allowed_range) != 2
                 or selection_window_ns[0] < allowed_range[0]
                 or selection_window_ns[1] > allowed_range[1])):
        raise PermissionError("BC selection window escapes the native factory's Train-inner scope")
    providers = {arm: provider_factory() for arm in ("pure_bc", "pure_ppo", "bc_ppo")}
    if len({id(provider) for provider in providers.values()}) != len(providers):
        raise ValueError("four-arm comparison requires independent provider instances")
    created: list[Any] = []
    try:
        for provider in providers.values():
            provider.create()
            created.append(provider)
        configs = {arm: provider.config for arm, provider in providers.items()}
        first = configs["pure_bc"]
        if (any(config.algorithm != "PPO" or config.normalize_observation
                or config.hash != first.hash or config.fingerprint.hash != first.fingerprint.hash
                for config in configs.values())
                or any(provider.model.num_timesteps != 0 for provider in providers.values())):
            raise ValueError(
                "four-arm comparison requires pristine identical PPO recipes in the raw policy domain"
            )
        fit_identity = rows[0].transitions[0].native_identity
        native_recipe = getattr(native_env_factory, "route_a_execution_recipe_hash", None)
        if native_recipe is None:
            if (getattr(native_env_factory, "transition_hash", None) != first.fingerprint.hash
                    or fit_identity["transition_hash"] != first.fingerprint.hash):
                raise ValueError("native comparison environment differs from PPO transition identity")
        else:
            from rles_rl_sb3.config import digest  # type: ignore[import-untyped]
            factory_identity = {
                "dataset": native_env_factory.store.snapshot_hash,
                "feature": native_env_factory.features.hash,
                "feature_scaler": native_env_factory.scaler.hash,
                "observation": native_env_factory.schema.hash,
                "execution": native_env_factory.execution.fingerprint(),
                "hard_risk": native_env_factory.hard.signature(),
                "soft_risk": digest({name: str(value) for name, value in
                                     asdict(native_env_factory.soft).items()}),
                "fold": digest([native_env_factory.start_ns, native_env_factory.end_ns]),
            }
            if (any(getattr(first.fingerprint, name) != value
                    for name, value in factory_identity.items())
                    or native_recipe != selection_rows[0].episode_metadata.get("execution_recipe_hash")
                    or native_recipe != rows[0].episode_metadata.get("execution_recipe_hash")
                    or selection_rows[0].transitions[0].native_identity["transition_hash"]
                        != native_env_factory.transition_hash):
                raise ValueError("native selection factory differs from the frozen PPO recipe")
        if fit_identity["schema_hash"] != first.policy.observation_schema_hash:
            raise ValueError("BC trajectories differ from the frozen L4 input domain")

        pure_bc_parity, pure_bc_behavior = train_behavioral_clone(
            providers["pure_bc"], rows, epochs=bc_epochs, batch_size=bc_batch_size,
            learning_rate=bc_learning_rate, tolerance=parity_tolerance,
            early_stop=False, deadband=deadband,
        )
        bc_ppo_parity, bc_ppo_behavior = train_behavioral_clone(
            providers["bc_ppo"], rows, epochs=bc_epochs, batch_size=bc_batch_size,
            learning_rate=bc_learning_rate, tolerance=parity_tolerance,
            early_stop=False, deadband=deadband,
        )
        from rles_rl_sb3.actor import action_parity

        selection_gates = {}
        for arm in ("pure_bc", "bc_ppo"):
            provider = providers[arm]
            predicted, _ = provider.model.predict(
                selection_observations, deterministic=True,
            )
            predicted = np.asarray(predicted, dtype=np.float32).reshape((-1, 1))
            behavior = behavior_report(selection_rows, predicted, deadband=deadband)
            parity = action_parity(
                provider.model, provider.config, selection_observations,
                selection_actions, tolerance=parity_tolerance,
            )
            selection_gates[arm] = (parity, behavior)
        _require_bc_action_gate(pure_bc_parity, pure_bc_behavior, arm="pure-BC")
        _require_bc_action_gate(
            *selection_gates["pure_bc"], arm="pure-BC independent Train-inner selection",
        )
        _require_bc_action_gate(bc_ppo_parity, bc_ppo_behavior, arm="BC-to-PPO")
        _require_bc_action_gate(
            *selection_gates["bc_ppo"], arm="BC-to-PPO independent Train-inner selection",
        )
        pure_bc_actions, _ = providers["pure_bc"].model.predict(
            observations, deterministic=True,
        )
        bc_ppo_actions, _ = providers["bc_ppo"].model.predict(
            observations, deterministic=True,
        )
        initial_error = float(np.max(np.abs(
            np.asarray(pure_bc_actions, dtype=np.float64)
            - np.asarray(bc_ppo_actions, dtype=np.float64)
        )))
        if initial_error > parity_tolerance:
            raise ValueError("pure-BC and BC-to-PPO actors differ before PPO updates")

        # A completed native rollout may auto-reset and hold the process-global
        # Nautilus engine session.  Freeze each model, close its provider, then
        # start the next arm so account ownership remains strictly serial.
        models = {arm: provider.model for arm, provider in providers.items()}
        _reset_ppo_arm_random_stream(providers["pure_ppo"])
        pure_ppo_training = providers["pure_ppo"].train(rl_steps)
        models["pure_ppo"] = providers["pure_ppo"].model
        providers["pure_ppo"].close()
        created.remove(providers["pure_ppo"])
        _reset_ppo_arm_random_stream(providers["bc_ppo"])
        bc_ppo_training = providers["bc_ppo"].train(rl_steps)
        models["bc_ppo"] = providers["bc_ppo"].model
        if (getattr(pure_ppo_training, "samples", None) != rl_steps
                or getattr(bc_ppo_training, "samples", None) != rl_steps
                or models["pure_ppo"].num_timesteps != rl_steps
                or models["bc_ppo"].num_timesteps != rl_steps
                or models["pure_bc"].num_timesteps != 0):
            raise RuntimeError("four-arm comparison failed its frozen PPO sample budgets")
        for provider in reversed(created):
            provider.close()
        created.clear()

        from .collector import collect_episodes
        from .diagnostics import native_rollout_report

        policy_factories: dict[str, Callable[[], Any]] = {
            "rule_teacher": rule_teacher_factory,
            "pure_bc": lambda: FrozenBCPolicy(
                models["pure_bc"], expert_id="pure_bc",
            ),
            "pure_ppo": lambda: FrozenBCPolicy(
                models["pure_ppo"], expert_id="pure_ppo",
            ),
            "bc_ppo": lambda: FrozenBCPolicy(
                models["bc_ppo"], expert_id="bc_ppo",
            ),
        }
        arm_episodes = {
            arm: collect_episodes(
                native_env_factory, policy_factory, seeds=native_seeds,
                episode_options=episode_options,
            )
            for arm, policy_factory in policy_factories.items()
        }
        expert_ids = tuple(
            arm_episodes[arm][0].expert_id
            for arm in ("rule_teacher", "pure_bc", "pure_ppo", "bc_ppo")
        )
        if len(set(expert_ids)) != len(expert_ids):
            raise RuntimeError("four-arm comparison requires a unique expert ID for every arm")
        for episodes in arm_episodes.values():
            for episode in episodes:
                identity = episode.transitions[0].native_identity
                expected_native_transition = (first.fingerprint.hash if native_recipe is None
                                              else native_env_factory.transition_hash)
                if (identity["transition_hash"] != expected_native_transition
                        or identity["schema_hash"] != first.policy.observation_schema_hash):
                    raise RuntimeError(
                        "native comparison replay differs from the frozen L4 input domain"
                    )
        _validate_paired_native_arms(arm_episodes, native_seeds)
        combined = tuple(
            episode for arm in ("rule_teacher", "pure_bc", "pure_ppo", "bc_ppo")
            for episode in arm_episodes[arm]
        )
        native_report = native_rollout_report(combined)
        paired_identity = _paired_native_identity(arm_episodes, native_seeds)
        payload = {
            "schema": "rles.il.four_arm_native_comparison.v1",
            "scope": "train_inner_scenario_only",
            "arms": {
                arm: arm_episodes[arm][0].expert_id
                for arm in ("rule_teacher", "pure_bc", "pure_ppo", "bc_ppo")
            },
            "native_seeds": list(native_seeds),
            "sample_budget": rl_steps,
            "config_hash": first.hash,
            "transition_hash": first.fingerprint.hash,
            "native_selection_transition_hash": (
                first.fingerprint.hash if native_recipe is None
                else native_env_factory.transition_hash),
            "execution_recipe_hash": native_recipe,
            "training_identity": dict(fit_identity),
            "fit_episode_windows": sorted({
                (int(episode.transitions[0].episode_window["start_ns"]),
                 int(episode.transitions[0].episode_window["end_ns"]))
                for episode in rows
            }),
            "policy_input_domain": {
                "observation_schema_hash": first.policy.observation_schema_hash,
                "l4_vecnormalize_enabled": False,
            },
            "bc_selection": {
                "window_ns": list(selection_window_ns),
                "episode_windows": sorted({
                    (int(episode.transitions[0].episode_window["start_ns"]),
                     int(episode.transitions[0].episode_window["end_ns"]))
                    for episode in selection_rows
                }),
                "purge_ns": selection_purge_ns,
                "pure_bc_behavior": asdict(selection_gates["pure_bc"][1]),
                "pure_bc_parity": _parity_receipt(selection_gates["pure_bc"][0]),
                "bc_ppo_behavior": asdict(selection_gates["bc_ppo"][1]),
                "bc_ppo_parity": _parity_receipt(selection_gates["bc_ppo"][0]),
            },
            "bc_initialization": {
                "pure_bc": asdict(pure_bc_behavior),
                "bc_ppo": asdict(bc_ppo_behavior),
                "pure_bc_parity": _parity_receipt(pure_bc_parity),
                "bc_ppo_parity": _parity_receipt(bc_ppo_parity),
                "pure_bc_to_bc_ppo_max_abs_error": initial_error,
            },
            "ppo_training": {
                "pure_ppo": _training_receipt(pure_ppo_training),
                "bc_ppo": _training_receipt(bc_ppo_training),
            },
            "paired_episode_identity_sha256": paired_identity,
            "native_report": native_report,
            "economic_performance_claim": False,
        }
        report_sha256 = _write_immutable_json(destination, payload)
        return FourArmNativeComparisonResult(
            sample_budget=rl_steps,
            config_hash=first.hash,
            transition_hash=first.fingerprint.hash,
            report_path=str(destination),
            report_sha256=report_sha256,
            paired_episode_identity_sha256=paired_identity,
            native_report=native_report,
        )
    finally:
        for provider in reversed(created):
            provider.close()


def _training_receipt(performance: Any) -> dict[str, Any]:
    diagnostics = getattr(performance, "diagnostics", None)
    return {
        "samples": int(performance.samples),
        "updates": performance.updates,
        "diagnostics": asdict(diagnostics) if diagnostics is not None else None,
    }


def _parity_receipt(parity: Any) -> dict[str, Any]:
    """Retain the exact frozen behavioral-cloning action-parity result."""
    fields = (
        "samples", "max_abs_error", "mean_abs_error", "p50_abs_error",
        "p95_abs_error", "tolerance", "passed", "method", "epochs_completed",
    )
    if any(not hasattr(parity, name) for name in fields):
        raise TypeError("behavioral_clone must return the frozen actor ParityReport")
    result = {name: getattr(parity, name) for name in fields}
    if (type(result["samples"]) is not int or result["samples"] <= 0
            or type(result["passed"]) is not bool
            or type(result["epochs_completed"]) is not int
            or not isinstance(result["method"], str) or not result["method"]
            or any(not math.isfinite(result[name]) for name in (
                "max_abs_error", "mean_abs_error", "p50_abs_error",
                "p95_abs_error", "tolerance",
            ))):
        raise ValueError("behavioral_clone returned an invalid action-parity receipt")
    return result


def _validate_paired_native_arms(
    arms: Mapping[str, tuple[EpisodeTrajectory, ...]], seeds: tuple[int, ...],
) -> None:
    required = ("rule_teacher", "pure_bc", "pure_ppo", "bc_ppo")
    if set(arms) != set(required) or any(len(arms[arm]) != len(seeds) for arm in required):
        raise RuntimeError("four-arm native replay did not preserve the frozen arm/seed matrix")
    for arm in required:
        for episode in arms[arm]:
            validate_trajectory(episode)
    for episode_index, seed in enumerate(seeds):
        reference = arms[required[0]][episode_index]
        reference_rows = reference.transitions
        reference_clock = tuple((row.decision_ts_ns, row.next_decision_ts_ns)
                                for row in reference_rows)
        for arm in required:
            episode = arms[arm][episode_index]
            if (episode.episode_metadata.get("seed") != seed
                    or episode.transitions[0].native_identity
                    != reference_rows[0].native_identity
                    or episode.transitions[0].episode_window
                    != reference_rows[0].episode_window
                    or tuple((row.decision_ts_ns, row.next_decision_ts_ns)
                             for row in episode.transitions) != reference_clock):
                raise RuntimeError("four-arm native policies did not replay identical episode conditions")


def _paired_native_identity(
    arms: Mapping[str, tuple[EpisodeTrajectory, ...]], seeds: tuple[int, ...],
) -> str:
    reference = arms["rule_teacher"]
    value = {
        "seeds": list(seeds),
        "episodes": [{
            "native_identity": dict(episode.transitions[0].native_identity),
            "episode_window": dict(episode.transitions[0].episode_window),
            "decision_clock": [
                [row.decision_ts_ns, row.next_decision_ts_ns]
                for row in episode.transitions
            ],
        } for episode in reference],
    }
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def compare_bc_to_random_ppo(
    provider_factory: Callable[[], Any],
    trajectories: tuple[EpisodeTrajectory, ...] | list[EpisodeTrajectory],
    *,
    sample_milestones: tuple[int, ...],
    checkpoint_writer: Callable[[str, Any, int], str | Path],
    checkpoint_replayer: Callable[[str, Path, np.ndarray], np.ndarray],
    attempt_ids: Mapping[str, str],
    bc_epochs: int,
    bc_batch_size: int = 256,
    bc_learning_rate: float = 1e-4,
    parity_tolerance: float = 1e-3,
    checkpoint_parity_tolerance: float = 1e-5,
    deadband: float = 0.05,
) -> BCComparisonResult:
    """Run matched random-PPO/BC-PPO arms with L1-owned durable checkpoints.

    ``checkpoint_writer(arm, provider, samples)`` must save an L1 checkpoint
    and return its path. ``checkpoint_replayer(arm, path, observations)`` must
    restore its exact native boundary through L1 and return deterministic
    predictions. Milestones are cumulative environment samples and must each
    be complete PPO rollout budgets.
    """
    if (not callable(provider_factory) or not callable(checkpoint_writer)
            or not callable(checkpoint_replayer)):
        raise ValueError("provider factory and L1 checkpoint save/replay adapters are required")
    arms = ("random_ppo", "bc_ppo")
    if (not isinstance(attempt_ids, Mapping) or set(attempt_ids) != set(arms)
            or any(not isinstance(value, str) or not value or value in {".", ".."}
                   or "/" in value or "\\" in value
                   for value in attempt_ids.values())
            or attempt_ids[arms[0]] == attempt_ids[arms[1]]):
        raise ValueError("paired arms require distinct valid frozen L1 attempt IDs")
    if (not math.isfinite(checkpoint_parity_tolerance)
            or checkpoint_parity_tolerance < 0):
        raise ValueError("checkpoint action parity tolerance must be finite and nonnegative")
    milestones = tuple(sample_milestones)
    if (not milestones or any(type(step) is not int or step <= 0 for step in milestones)
            or tuple(sorted(set(milestones))) != milestones):
        raise ValueError("sample milestones must be strictly increasing positive integers")
    random_provider = provider_factory()
    bc_provider = provider_factory()
    if random_provider is bc_provider:
        raise ValueError("paired arms require independent provider instances")
    created: list[Any] = []
    try:
        random_provider.create()
        created.append(random_provider)
        bc_provider.create()
        created.append(bc_provider)
        random_config, bc_config = random_provider.config, bc_provider.config
        if (random_config.algorithm != "PPO" or bc_config.algorithm != "PPO"
                or random_config.normalize_observation or bc_config.normalize_observation
                or random_config.hash != bc_config.hash
                or random_config.fingerprint.hash != bc_config.fingerprint.hash
                or random_provider.model.num_timesteps != 0
                or bc_provider.model.num_timesteps != 0):
            raise ValueError("F1 arms require pristine PPO actors with identical frozen configs and no normalization")
        parity, before = train_behavioral_clone(
            bc_provider, trajectories, epochs=bc_epochs, batch_size=bc_batch_size,
            learning_rate=bc_learning_rate, tolerance=parity_tolerance,
            early_stop=False, deadband=deadband,
        )
        _require_bc_action_gate(parity, before, arm="BC-to-PPO comparison")
        observations, _, rows = prepare_bc_pairs(trajectories, deadband=deadband)
        from rles_rl_sb3.checkpoint import (  # type: ignore[import-untyped]
            capture_rng, checkpoint_verify, restore_rng,
        )
        caller_rng = capture_rng()
        streams: dict[str, dict[str, Any]] = {}
        for arm, provider in (("random_ppo", random_provider), ("bc_ppo", bc_provider)):
            _reset_ppo_arm_random_stream(provider)
            streams[arm] = capture_rng()
        restore_rng(caller_rng)
        points: list[BCComparisonPoint] = []
        previous = 0
        for milestone in milestones:
            delta = milestone - previous
            restore_rng(streams["random_ppo"])
            random_perf = random_provider.train(delta, resumed=previous > 0)
            streams["random_ppo"] = capture_rng()
            restore_rng(streams["bc_ppo"])
            bc_perf = bc_provider.train(delta, resumed=previous > 0)
            streams["bc_ppo"] = capture_rng()
            if (random_perf.samples != delta or bc_perf.samples != delta
                    or random_provider.model.num_timesteps != milestone
                    or bc_provider.model.num_timesteps != milestone):
                raise RuntimeError("paired F1 arms failed their exact equal sample budget")
            random_actions, _ = random_provider.model.predict(observations, deterministic=True)
            bc_actions, _ = bc_provider.model.predict(observations, deterministic=True)
            saved = []
            for arm, provider, current_actions in (
                ("random_ppo", random_provider, random_actions),
                ("bc_ppo", bc_provider, bc_actions),
            ):
                path_raw = checkpoint_writer(arm, provider, milestone)
                if not isinstance(path_raw, (str, Path)) or not str(path_raw):
                    raise RuntimeError("L1 checkpoint writer must return its durable checkpoint path")
                # Normalize relative components without dereferencing a final
                # symlink; checkpoint_verify must see and reject link roots.
                checkpoint_path = Path(os.path.abspath(path_raw))
                manifest = checkpoint_verify(checkpoint_path, provider.config, trusted_source=True)
                if (manifest["attempt_id"] != attempt_ids[arm]
                        or manifest["num_timesteps"] != milestone):
                    raise RuntimeError("L1 checkpoint attempt or sample cursor differs from paired F1 arm")
                rng_before_restore = capture_rng()
                try:
                    restored_actions = checkpoint_replayer(arm, checkpoint_path, observations.copy())
                finally:
                    # Checkpoint load restores process-global learner RNG. A
                    # parity audit must not perturb either live training arm.
                    restore_rng(rng_before_restore)
                current = np.asarray(current_actions, dtype=np.float32).reshape((-1, 1))
                if (not isinstance(restored_actions, np.ndarray) or restored_actions.dtype != np.float32
                        or restored_actions.shape != current.shape or not np.isfinite(restored_actions).all()):
                    raise RuntimeError("L1 checkpoint replay returned malformed actor actions")
                error = float(np.max(np.abs(restored_actions.astype(np.float64) - current.astype(np.float64))))
                if error > checkpoint_parity_tolerance:
                    raise RuntimeError("L1 checkpoint restore changed deterministic policy actions")
                saved.append((str(checkpoint_path), error))
            points.append(BCComparisonPoint(
                samples=milestone, random_performance=random_perf, bc_performance=bc_perf,
                random_checkpoint=saved[0][0], bc_checkpoint=saved[1][0],
                random_checkpoint_max_abs_error=saved[0][1],
                bc_checkpoint_max_abs_error=saved[1][1],
                random_behavior=behavior_report(rows, np.asarray(random_actions, dtype=np.float32).reshape((-1, 1)),
                                               deadband=deadband),
                bc_behavior=behavior_report(rows, np.asarray(bc_actions, dtype=np.float32).reshape((-1, 1)),
                                            deadband=deadband),
            ))
            previous = milestone
        return BCComparisonResult(
            config_hash=random_config.hash, transition_hash=random_config.fingerprint.hash,
            seed=int(random_config.seeds.algorithm), behavior_clone_parity=parity,
            behavior_before_rl=before, points=tuple(points),
        )
    finally:
        for provider in reversed(created):
            provider.close()


def finetune_bc_actor(
    provider: Any,
    trajectories: tuple[EpisodeTrajectory, ...] | list[EpisodeTrajectory],
    *,
    rl_steps: int,
    bc_epochs: int,
    bc_batch_size: int = 256,
    bc_learning_rate: float = 1e-4,
    parity_tolerance: float = 1e-3,
    deadband: float = 0.05,
) -> BCFinetuneResult:
    """F1: initialize a pristine PPO/SAC actor from BC, then continue via L4.

    L1 remains responsible for its ordinary audited checkpoint boundaries; this
    helper performs only the BC gate and a fixed RL budget through RLProvider.
    """
    config = getattr(provider, "config", None)
    model = getattr(provider, "model", None)
    if config is None or model is None:
        raise ValueError("a created L4 RLProvider is required")
    if config.algorithm not in {"PPO", "SAC"}:
        raise ValueError("RFC F1 first candidates are PPO and SAC")
    if bool(config.normalize_observation):
        raise ValueError("F1 requires the frozen policy observation domain with L4 VecNormalize disabled")
    if type(rl_steps) is not int or rl_steps <= 0:
        raise ValueError("positive fixed RL continuation budget required")
    if int(model.num_timesteps) != 0:
        raise ValueError("F1 BC initialization must precede all L4 RL environment updates")
    parity, before = train_behavioral_clone(
        provider, trajectories, epochs=bc_epochs, batch_size=bc_batch_size,
        learning_rate=bc_learning_rate, tolerance=parity_tolerance,
        early_stop=False, deadband=deadband,
    )
    _require_bc_action_gate(parity, before, arm="BC-to-PPO fine-tuning")
    performance = provider.train(rl_steps)
    if int(model.num_timesteps) != rl_steps or getattr(performance, "samples", None) != rl_steps:
        raise RuntimeError("L4 did not execute the exact frozen F1 RL step budget")
    observations, _, rows = prepare_bc_pairs(trajectories, deadband=deadband)
    predictions, _ = model.predict(observations, deterministic=True)
    after = behavior_report(rows, np.asarray(predictions, dtype=np.float32).reshape((-1, 1)),
                            deadband=deadband)
    return BCFinetuneResult(parity, before, after, performance, rl_steps)

class FrozenBCPolicy:
    """Deterministic SB3 actor adapter accepted by the native episode collector."""

    def __init__(self, model: Any, *, expert_id: str = "bc_sb3_v1") -> None:
        if not expert_id:
            raise ValueError("expert_id is required")
        self.model = model
        self.expert_id = expert_id

    def reset(self, episode_metadata: Mapping[str, object]) -> None:
        if not isinstance(episode_metadata, Mapping):
            raise TypeError("episode metadata mapping required")

    def act(self, *, observation: np.ndarray, context: FrozenVisibleContext) -> float:
        del context  # BC consumes the same frozen policy observation used for training.
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim != 1 or not len(obs) or not np.isfinite(obs).all():
            raise ValueError("BC requires one finite policy-domain observation")
        action, _ = self.model.predict(obs, deterministic=True)
        value = np.asarray(action, dtype=np.float32).reshape(-1)
        if value.shape != (1,) or not np.isfinite(value).all() or abs(float(value[0])) > 1.000001:
            raise ValueError("SB3 BC actor returned an invalid target exposure")
        return float(np.clip(value[0], -1.0, 1.0))


def evaluate_bc_native_rollouts(
    actor_path: str | Path,
    env_factory: Any,
    *,
    seeds: tuple[int, ...],
    report_path: str | Path,
    episode_options: tuple[dict[str, Any] | None, ...] | None = None,
) -> BCNativeRolloutResult:
    """Load a published BC actor, replay it on Train-inner native accounts, and bind results.

    The actor sidecar's dataset identities and checkpoint hash are checked before
    creating the native model or environment. Reports are immutable JSON artifacts.
    """
    actor = Path(actor_path).resolve()
    sidecar = actor.with_suffix(".json")
    destination = Path(report_path).resolve()
    if actor.suffix != ".zip" or not actor.is_file() or not sidecar.is_file():
        raise ValueError("published BC checkpoint and provenance sidecar are required")
    if destination.suffix != ".json":
        raise ValueError("native rollout report path must end in .json")
    if type(seeds) is not tuple or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("one or more deterministic nonnegative replay seeds are required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("native replay seeds must be unique")
    if getattr(env_factory, "split", None) != "train_inner":
        raise ValueError("BC native candidate replay accepts Train-inner environments only")
    if not callable(getattr(env_factory, "make_env", None)):
        raise TypeError("native Train-inner environment factory is required")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("BC actor provenance sidecar is unreadable") from exc
    actor_digest = sha256(actor.read_bytes()).hexdigest()
    manifest = metadata.get("dataset_manifest") if isinstance(metadata, dict) else None
    mapping = metadata.get("actor_mapping") if isinstance(metadata, dict) else None
    if (not isinstance(metadata, dict) or metadata.get("schema") != "rles.il.bc_actor.v1"
            or metadata.get("account_truth") != "nautilus_native"
            or metadata.get("split") != "train_inner"
            or metadata.get("checkpoint_sha256") != actor_digest
            or not isinstance(manifest, dict) or manifest.get("split") != "train_inner"
            or manifest.get("account_truth") != "nautilus_native"
            or not isinstance(mapping, dict) or mapping.get("schema") != "rles.l4.actor.mapping.v1"):
        raise ValueError("BC actor, hash-bound metadata, or native Train-inner identity is invalid")
    manifest_digest = sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    recorded_manifest_digest = sha256(json.dumps(
        metadata.get("dataset_manifest"), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if metadata.get("dataset_manifest_sha256") != recorded_manifest_digest:
        raise ValueError("BC actor sidecar dataset-manifest hash mismatch")
    algorithm = mapping.get("algorithm")
    actor_signature = mapping.get("actor_signature")
    fold_hashes = manifest.get("fold_hashes")
    if (not isinstance(algorithm, str) or not isinstance(actor_signature, str)
            or not actor_signature or not isinstance(fold_hashes, list) or not fold_hashes
            or any(not isinstance(value, str) for value in fold_hashes)):
        raise ValueError("BC actor mapping and fold identities are incomplete")
    try:
        from stable_baselines3 import DDPG, PPO, SAC, TD3
    except ImportError as exc:
        raise RuntimeError("SB3 is required to load the published BC actor") from exc
    model_classes = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "DDPG": DDPG}
    model_class: Any = model_classes.get(algorithm)
    if model_class is None:
        raise ValueError("published BC actor uses an unsupported SB3 algorithm")
    model = model_class.load(str(actor), device="cpu")

    from .collector import collect_episodes
    from .diagnostics import native_rollout_report

    expert_id = f"bc_{algorithm.lower()}_{actor_signature[:12]}"
    trajectories = collect_episodes(
        env_factory,
        lambda: FrozenBCPolicy(model, expert_id=expert_id),
        seeds=seeds,
        episode_options=episode_options,
    )
    expected_fold = set(fold_hashes)
    for episode in trajectories:
        identity = episode.transitions[0].native_identity
        if (identity["dataset_hash"] != manifest.get("source_data_hash")
                or identity["config_hash"] != manifest.get("source_environment_config_hash")
                or identity["transition_hash"] != manifest.get("transition_hash")
                or identity["schema_hash"] != manifest.get("observation_schema_hash")
                or identity["fold_hash"] not in expected_fold):
            raise ValueError("native replay identity differs from the frozen BC training manifest")
    from .dataset import validate_trajectory
    for episode in trajectories:
        validate_trajectory(episode)
    native_report = native_rollout_report(trajectories)
    payload = {
        "schema": "rles.il.bc_native_rollout.v1",
        "actor_sha256": actor_digest,
        "actor_signature": mapping.get("actor_signature"),
        "algorithm": algorithm,
        "dataset_manifest_sha256": manifest_digest,
        "seeds": list(seeds),
        "transition_hashes": sorted({row.transitions[0].native_identity["transition_hash"]
                                      for row in trajectories}),
        "episode_windows": [dict(row.transitions[0].episode_window) for row in trajectories],
        "episode_count": len(trajectories),
        "native_report": native_report,
        "economic_performance_claim": False,
    }
    report_digest = _write_immutable_json(destination, payload)
    return BCNativeRolloutResult(
        actor_sha256=actor_digest,
        report_path=str(destination),
        report_sha256=report_digest,
        episode_count=len(trajectories),
        native_report=native_report,
    )


def publish_bc_artifact(
    provider: Any,
    report: BCBehaviorReport,
    destination: str | Path,
    *,
    parity: Any,
    dataset_manifest: Mapping[str, Any],
    training: Mapping[str, Any],
) -> Path:
    """Publish a frozen SB3 actor and provenance sidecar without overwriting."""
    model = getattr(provider, "model", None)
    if model is None or int(getattr(model, "num_timesteps", -1)) != 0:
        raise ValueError("only a BC-initialized actor before RL updates can be published")
    if dataset_manifest.get("split") != "train_inner" or dataset_manifest.get("account_truth") != "nautilus_native":
        raise ValueError("BC artifact requires an auditable Train-inner native dataset manifest")
    target = Path(destination).resolve()
    if target.suffix != ".zip":
        raise ValueError("actor destination must end in .zip")
    metadata_path = target.with_suffix(".json")
    if target.exists() or metadata_path.exists():
        raise FileExistsError("BC artifacts are immutable and cannot overwrite prior runs")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{target.stem}.", dir=target.parent))
    staged_model = temp_dir / target.name
    try:
        model.save(str(staged_model))
        produced = staged_model if staged_model.exists() else staged_model.with_suffix(".zip")
        if not produced.is_file():
            raise RuntimeError("SB3 did not produce the requested BC checkpoint")
        digest = sha256(produced.read_bytes()).hexdigest()
        metadata = {
            "schema": "rles.il.bc_actor.v1",
            "account_truth": "nautilus_native",
            "split": "train_inner",
            "dataset_manifest": dict(dataset_manifest),
            "dataset_manifest_sha256": sha256(json.dumps(
                dataset_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "actor_mapping": provider.mapping(),
            "checkpoint_sha256": digest,
            "behavior_report": asdict(report),
            "parity_report": asdict(parity),
            "training": dict(training),
            "economic_performance_claim": False,
        }
        staged_metadata = temp_dir / metadata_path.name
        staged_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                   encoding="utf-8")
        # Both payloads are fully staged before publication; destinations are new.
        os.replace(produced, target)
        try:
            os.replace(staged_metadata, metadata_path)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    finally:
        for child in temp_dir.iterdir():
            child.unlink(missing_ok=True)
        temp_dir.rmdir()
    return target


