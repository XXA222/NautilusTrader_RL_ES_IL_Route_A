"""Load isolated IQL TorchScript actors for independent Train-inner native replay."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bc import _write_immutable_json
from .collector import collect_episodes
from .dataset import validate_trajectory
from .diagnostics import native_rollout_report
from .experts import FrozenVisibleContext


@dataclass(frozen=True, slots=True)
class IQLNativeRolloutResult:
    actor_sha256: str
    report_path: str
    report_sha256: str
    episode_count: int
    native_report: Mapping[str, Any]


class FrozenIQLPolicy:
    """Deterministic bounded adapter for a d3rlpy-exported TorchScript actor."""

    def __init__(self, module: Any, *, expert_id: str) -> None:
        if not expert_id:
            raise ValueError("IQL expert identity is required")
        self.module = module
        self.expert_id = expert_id

    def reset(self, episode_metadata: Mapping[str, object]) -> None:
        if not isinstance(episode_metadata, Mapping):
            raise TypeError("episode metadata mapping required")

    def act(self, *, observation: np.ndarray, context: FrozenVisibleContext) -> float:
        del context
        obs = np.array(observation, dtype=np.float32, copy=True)
        if obs.ndim != 1 or not len(obs) or not np.isfinite(obs).all():
            raise ValueError("IQL actor requires one finite frozen policy observation")
        import torch
        with torch.no_grad():
            output = self.module(torch.as_tensor(obs[None, :], dtype=torch.float32))
        action = np.asarray(output.detach().cpu().numpy(), dtype=np.float32)
        if action.shape != (1, 1) or not np.isfinite(action).all() or abs(float(action[0, 0])) > 1.000001:
            raise ValueError("IQL actor must produce one finite target exposure in [-1,1]")
        return float(np.clip(action[0, 0], -1.0, 1.0))


def _manifest_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify_iql_export_parity(
    run_directory: str | Path,
    *,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Verify at least 10k worker-produced IQL actions against TorchScript in this process."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive finite cross-process parity tolerance required")
    run = Path(run_directory).resolve()
    manifest_path, policy_path = run / "manifest.json", run / "policy.pt"
    parity_path = run / "parity_inputs.npz"
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("IQL parity manifest is unreadable") from exc
    artifacts = metadata.get("artifacts_sha256") if isinstance(metadata, dict) else None
    if (not isinstance(artifacts, dict) or not policy_path.is_file() or not parity_path.is_file()
            or artifacts.get("policy.pt") != sha256(policy_path.read_bytes()).hexdigest()
            or artifacts.get("parity_inputs.npz") != sha256(parity_path.read_bytes()).hexdigest()):
        raise ValueError("IQL policy or fixed parity input hash differs from the run manifest")
    with np.load(parity_path, allow_pickle=False) as archive:
        if set(archive.files) != {"observations", "expected_actions"}:
            raise ValueError("IQL cross-process parity payload fields are invalid")
        observations = np.array(archive["observations"], copy=True)
        expected = np.array(archive["expected_actions"], copy=True)
    if (observations.dtype != np.float32 or observations.ndim != 2
            or expected.dtype != np.float32 or expected.shape != (len(observations), 1)
            or len(observations) < 10_000 or not np.isfinite(observations).all()
            or not np.isfinite(expected).all() or (np.abs(expected) > 1.000001).any()):
        raise ValueError("IQL cross-process parity requires 10,000 finite fixed states and bounded actions")
    import torch
    module = torch.jit.load(str(policy_path), map_location="cpu")
    module.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(observations), 2048):
            output = module(torch.from_numpy(observations[start:start + 2048]))
            chunks.append(output.detach().cpu().numpy())
    actual = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("main-process TorchScript actions have invalid shape or values")
    errors = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    maximum = float(errors.max())
    if maximum > tolerance:
        raise ValueError("IQL d3rlpy-to-TorchScript cross-process action parity failed")
    return {
        "samples": int(len(observations)), "tolerance": tolerance,
        "max_abs_error": maximum, "mean_abs_error": float(errors.mean()),
        "passed": True,
        "source": "isolated_d3rlpy_expected_actions_vs_main_process_torchscript",
    }


def _assert_identity(dataset_manifest: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, str]:
    identity = dataset_manifest.get("native_identity")
    other_identity = other.get("native_identity")
    source = dataset_manifest.get("source_manifest")
    other_source = other.get("source_manifest")
    keys = {"dataset_hash", "fold_hash", "config_hash", "transition_hash", "schema_hash"}
    if (not isinstance(identity, dict) or set(identity) != keys or identity != other_identity
            or not isinstance(source, dict) or not isinstance(other_source, dict)
            or source.get("account_truth") != "nautilus_native"
            or other_source.get("account_truth") != "nautilus_native"
            or source.get("split") != "train_inner" or other_source.get("split") != "train_inner"):
        raise ValueError("IQL fit and selection datasets must bind the same native Train-inner identity")
    return identity


def _validate_windows(windows: Any, *, label: str) -> list[tuple[int, int]]:
    if not isinstance(windows, list) or not windows:
        raise ValueError(f"IQL run manifest lacks {label} episode time windows")
    output: list[tuple[int, int]] = []
    for row in windows:
        if (not isinstance(row, list) or len(row) != 2 or type(row[0]) is not int
                or type(row[1]) is not int or row[0] <= 0 or row[1] <= row[0]):
            raise ValueError(f"IQL {label} episode window is invalid")
        output.append((row[0], row[1]))
    return output


def evaluate_iql_native_rollouts(
    run_directory: str | Path,
    env_factory: Any,
    *,
    seeds: tuple[int, ...],
    report_path: str | Path,
    episode_options: tuple[dict[str, Any] | None, ...] | None = None,
) -> IQLNativeRolloutResult:
    """Evaluate a completed native-research IQL run on fresh Train-inner windows."""
    run = Path(run_directory).resolve()
    destination = Path(report_path).resolve()
    model_path, manifest_path = run / "policy.pt", run / "manifest.json"
    if (not (run / "COMPLETED").is_file() or (run / "FAILED").exists()
            or not model_path.is_file() or not manifest_path.is_file()):
        raise ValueError("completed IQL run, policy, and manifest are required")
    if destination.suffix != ".json":
        raise ValueError("native IQL report path must use .json")
    if type(seeds) is not tuple or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("one or more deterministic nonnegative native replay seeds are required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("native replay seeds must be unique")
    if getattr(env_factory, "split", None) != "train_inner" or not callable(
        getattr(env_factory, "make_env", None),
    ):
        raise ValueError("IQL native replay requires the Train-inner native environment factory")
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("IQL run manifest is unreadable") from exc
    if (not isinstance(metadata, dict) or metadata.get("schema") != "rles.il.iql_run.v1"
            or metadata.get("purpose") != "native_research"
            or metadata.get("historical_economic_claim") is not False):
        raise ValueError("synthetic or unsupported IQL output cannot enter native research replay")
    artifacts = metadata.get("artifacts_sha256")
    actor_digest = sha256(model_path.read_bytes()).hexdigest()
    if not isinstance(artifacts, dict) or artifacts.get("policy.pt") != actor_digest:
        raise ValueError("IQL TorchScript policy hash differs from the completed run manifest")
    parity_report = verify_iql_export_parity(run)
    training = metadata.get("dataset_manifest")
    selection = metadata.get("selection_dataset_manifest")
    if not isinstance(training, dict) or not isinstance(selection, dict):
        raise ValueError("native IQL run must retain fit and selection dataset manifests")
    identity = _assert_identity(training, selection)
    fit_windows = _validate_windows(metadata.get("training_episode_windows_ns"), label="fit")
    selection_windows = _validate_windows(metadata.get("selection_episode_windows_ns"), label="selection")
    if max(end for _start, end in fit_windows) > min(start for start, _end in selection_windows):
        raise ValueError("IQL selection block must follow all fit episodes")
    if any(start < fit_end and fit_start < end
           for fit_start, fit_end in fit_windows for start, end in selection_windows):
        raise ValueError("IQL fit and selection blocks overlap")

    import torch
    module = torch.jit.load(str(model_path), map_location="cpu")
    module.eval()
    policy = FrozenIQLPolicy(module, expert_id=f"iql_{actor_digest[:12]}")
    trajectories = collect_episodes(
        env_factory, lambda: FrozenIQLPolicy(module, expert_id=policy.expert_id),
        seeds=seeds, episode_options=episode_options,
    )
    evaluation_windows: list[tuple[int, int]] = []
    for episode in trajectories:
        validate_trajectory(episode)
        row = episode.transitions[0]
        if row.native_identity != identity:
            raise ValueError("IQL native replay identity differs from fit/selection data")
        window = (row.episode_window["start_ns"], row.episode_window["end_ns"])
        if max(end for _start, end in selection_windows) > window[0]:
            raise ValueError("IQL native replay window must follow the frozen selection block")
        if any(window[0] < used_end and used_start < window[1]
               for used_start, used_end in (*fit_windows, *selection_windows, *evaluation_windows)):
            raise ValueError("IQL fit, selection, and native replay windows must be disjoint")
        evaluation_windows.append(window)

    native_report = native_rollout_report(trajectories)
    payload = {
        "schema": "rles.il.iql_native_rollout.v1",
        "iql_run_manifest_sha256": _manifest_digest(manifest_path),
        "policy_sha256": actor_digest,
        "torchscript_cross_process_parity": parity_report,
        "dataset_native_identity": identity,
        "seeds": list(seeds),
        "fit_windows_ns": [list(window) for window in fit_windows],
        "selection_windows_ns": [list(window) for window in selection_windows],
        "evaluation_windows_ns": [list(window) for window in evaluation_windows],
        "episode_count": len(trajectories),
        "native_report": native_report,
        "economic_performance_claim": False,
    }
    report_digest = _write_immutable_json(destination, payload)
    return IQLNativeRolloutResult(
        actor_sha256=actor_digest, report_path=str(destination), report_sha256=report_digest,
        episode_count=len(trajectories), native_report=native_report,
    )
