"""Bounded IQL training command for the isolated d3rlpy runtime."""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import traceback

import numpy as np

from .data import WorkerDataset, build_d3rlpy_replay, load_worker_dataset


_ALLOWED_CONFIG = {
    "schema", "purpose", "run_label", "dataset", "selection_dataset", "output_dir", "device", "seed",
    "n_steps", "n_steps_per_epoch", "batch_size", "gamma", "expectile",
    "weight_temp", "max_weight", "reward_scale", "truncation_policy",
}


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearest_support_distance(predictions: np.ndarray, support: np.ndarray) -> np.ndarray:
    candidates = np.sort(np.asarray(support, dtype=np.float32).reshape(-1))
    values = np.asarray(predictions, dtype=np.float32).reshape(-1)
    right = np.searchsorted(candidates, values, side="left")
    right = np.clip(right, 0, len(candidates) - 1)
    left = np.clip(right - 1, 0, len(candidates) - 1)
    return np.minimum(np.abs(values - candidates[left]), np.abs(values - candidates[right]))


def _metrics(algo, data: WorkerDataset, *, reward_scale: float, source: str) -> dict:
    policy_actions = np.asarray(algo.predict(data.observations), dtype=np.float32).reshape((-1, 1))
    if policy_actions.shape != data.actions.shape or not np.isfinite(policy_actions).all():
        raise RuntimeError("IQL policy returned invalid action batch")
    if (np.abs(policy_actions) > 1.000001).any():
        raise RuntimeError("IQL policy output escaped bounded target-exposure action space")
    q_data = np.asarray(algo.predict_value(data.observations, data.actions), dtype=np.float64).reshape(-1)
    q_policy = np.asarray(algo.predict_value(data.observations, policy_actions), dtype=np.float64).reshape(-1)
    if not np.isfinite(q_data).all() or not np.isfinite(q_policy).all():
        raise RuntimeError("IQL Q diagnostics contain non-finite values")
    distances = _nearest_support_distance(policy_actions, data.actions)
    return {
        "source": source,
        "transitions": len(data.actions),
        "reward_scale": reward_scale,
        "action_mean": float(policy_actions.mean()),
        "action_std": float(policy_actions.std()),
        "action_zero_fraction_0_05": float(np.mean(np.abs(policy_actions[:, 0]) <= 0.05)),
        "action_min": float(policy_actions.min()),
        "action_max": float(policy_actions.max()),
        "nearest_supported_action_distance_p50": float(np.percentile(distances, 50)),
        "nearest_supported_action_distance_p95": float(np.percentile(distances, 95)),
        "nearest_supported_action_distance_max": float(distances.max()),
        "q_on_data_mean": float(q_data.mean()),
        "q_on_data_std": float(q_data.std()),
        "q_on_policy_mean": float(q_policy.mean()),
        "q_on_policy_std": float(q_policy.std()),
        "q_all_finite": bool(np.isfinite(q_data).all() and np.isfinite(q_policy).all()),
        "historical_economic_claim": False,
    }


def _q_calibration(algo, data: WorkerDataset, *, gamma: float, reward_scale: float) -> dict:
    """Compare held-out dataset-action Q estimates with terminal realized discounted returns."""
    if np.any(data.truncated) or not all(
        bool(data.terminated[int(end) - 1])
        for end in data.episode_offsets[1:]
    ):
        raise ValueError("held-out Q calibration requires fully settled terminated episodes")
    realized = np.empty(len(data.actions), dtype=np.float64)
    for begin, end in zip(data.episode_offsets[:-1], data.episode_offsets[1:]):
        lo, hi = int(begin), int(end)
        value = 0.0
        for index in range(hi - 1, lo - 1, -1):
            value = float(data.rewards[index]) * reward_scale + gamma * value
            realized[index] = value
    q_values = np.asarray(algo.predict_value(data.observations, data.actions), dtype=np.float64).reshape(-1)
    if q_values.shape != realized.shape or not np.isfinite(q_values).all():
        raise RuntimeError("held-out IQL Q predictions are invalid")
    errors = q_values - realized
    if not np.isfinite(realized).all() or not np.isfinite(errors).all():
        raise RuntimeError("held-out return calibration overflowed its finite numeric domain")
    order = np.argsort(q_values, kind="stable")
    bins = []
    for indices in np.array_split(order, min(5, len(order))):
        if len(indices):
            bins.append({
                "count": int(len(indices)),
                "q_mean": float(q_values[indices].mean()),
                "realized_discounted_return_mean": float(realized[indices].mean()),
                "bias_q_minus_return": float(errors[indices].mean()),
                "mae": float(np.abs(errors[indices]).mean()),
            })
    return {
        "source": "reserved_train_inner_selection_dataset",
        "discount": gamma,
        "reward_scale": reward_scale,
        "transitions": int(len(realized)),
        "episodes": int(len(data.episode_offsets) - 1),
        "q_mean": float(q_values.mean()),
        "realized_discounted_return_mean": float(realized.mean()),
        "bias_q_minus_return": float(errors.mean()),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "quintile_bins": bins,
        "interpretation": "descriptive_calibration; not economic evaluation",
    }


def _validate_selection_dataset(training: WorkerDataset, selection: WorkerDataset) -> None:
    train_identity = training.manifest.get("native_identity")
    selection_identity = selection.manifest.get("native_identity")
    train_source = training.manifest.get("source_manifest", {})
    selection_source = selection.manifest.get("source_manifest", {})
    if (train_source.get("account_truth") != "nautilus_native"
            or selection_source.get("account_truth") != "nautilus_native"
            or train_source.get("split") != "train_inner"
            or selection_source.get("split") != "train_inner"
            or not isinstance(train_identity, dict) or train_identity != selection_identity):
        raise ValueError("IQL selection data must share frozen native recipe identity with fit data")
    train_windows = set(map(tuple, np.asarray(training.episode_windows_ns).tolist()))
    selection_windows = set(map(tuple, np.asarray(selection.episode_windows_ns).tolist()))
    if not train_windows or not selection_windows or train_windows & selection_windows:
        raise ValueError("IQL fit and selection episodes must use distinct market windows")
    if any(left[0] < right[1] and right[0] < left[1]
           for left in train_windows for right in selection_windows):
        raise ValueError("IQL fit and selection time windows overlap")
    if max(end for _start, end in train_windows) > min(start for start, _end in selection_windows):
        raise ValueError("IQL fit windows must precede the reserved selection time block")
    if training.observations.shape[1] != selection.observations.shape[1]:
        raise ValueError("IQL fit and selection observation dimensions differ")
    if np.any(selection.truncated):
        raise ValueError("Q calibration selection episodes must be natively settled, not truncated")
    if any(not bool(selection.terminated[int(end) - 1]) for end in selection.episode_offsets[1:]):
        raise ValueError("every Q calibration episode must end in native termination")


def train_iql(config_path: str | Path) -> Path:
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != _ALLOWED_CONFIG:
        missing = _ALLOWED_CONFIG - set(config) if isinstance(config, dict) else _ALLOWED_CONFIG
        extra = set(config) - _ALLOWED_CONFIG if isinstance(config, dict) else set()
        raise ValueError(f"IQL config fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    if config["schema"] != "rles.il.iql.v1" or config["device"] not in {"cpu:0", "cuda:0"}:
        raise ValueError("unsupported IQL schema/device")
    if config["purpose"] not in {"native_research", "synthetic_smoke"}:
        raise ValueError("explicit native_research or synthetic_smoke purpose required")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("non-negative integer seed required")
    for key in ("n_steps", "n_steps_per_epoch", "batch_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["n_steps"] % config["n_steps_per_epoch"] != 0:
        raise ValueError("n_steps must be divisible by n_steps_per_epoch; no updates may be silently dropped")
    if (not 0 < float(config["gamma"]) < 1 or not 0 < float(config["expectile"]) < 1
            or not float(config["weight_temp"]) > 0 or not float(config["max_weight"]) > 0
            or not math.isfinite(float(config["reward_scale"])) or float(config["reward_scale"]) <= 0):
        raise ValueError("invalid IQL discount, expectile, weight, or explicit reward-scale value")
    if config["truncation_policy"] not in {"terminal_no_bootstrap", "timeout_bootstrap"}:
        raise ValueError("truncation policy must be an explicit reviewed choice")
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError("IQL run output is immutable and already exists")
    data = load_worker_dataset(config["dataset"])
    selection_data = None
    if config["purpose"] == "native_research":
        if not isinstance(config["selection_dataset"], str) or not config["selection_dataset"]:
            raise ValueError("native IQL research requires a separate reserved Train-inner selection dataset")
        selection_data = load_worker_dataset(config["selection_dataset"])
        _validate_selection_dataset(data, selection_data)
    elif config["selection_dataset"] is not None:
        raise ValueError("synthetic smoke runs do not accept a research selection dataset")
    source_truth = data.manifest["source_manifest"].get("account_truth")
    if config["purpose"] == "native_research" and source_truth != "nautilus_native":
        raise ValueError("native research run requires Nautilus-native account facts")
    if config["purpose"] == "synthetic_smoke" and (
        source_truth != "synthetic_mechanism" or config["run_label"] != "IQL_TOY_SMOKE"
    ):
        raise ValueError("synthetic payload is restricted to the named smoke run")
    replay = build_d3rlpy_replay(
        data, reward_scale=float(config["reward_scale"]),
        truncation_policy=config["truncation_policy"],
    )
    output.mkdir(parents=True)
    try:
        import d3rlpy
        import torch
        version = importlib.metadata.version("d3rlpy")
        if version != "2.8.1":
            raise RuntimeError(f"worker requires d3rlpy 2.8.1; found {version}")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if config["device"] == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        learner = d3rlpy.algos.IQLConfig(
            batch_size=config["batch_size"], gamma=float(config["gamma"]),
            expectile=float(config["expectile"]), weight_temp=float(config["weight_temp"]),
            max_weight=float(config["max_weight"]),
            observation_scaler=None, action_scaler=None, reward_scaler=None,
            compile_graph=False,
        ).create(device=config["device"])
        log_dir = output / "logs"
        log_dir.mkdir()
        previous_cwd = Path.cwd()
        try:
            os.chdir(log_dir)
            history = learner.fit(
                replay, n_steps=config["n_steps"],
                n_steps_per_epoch=config["n_steps_per_epoch"],
                experiment_name=str(config["run_label"]), with_timestamp=False,
                show_progress=False, save_interval=1,
            )
        finally:
            os.chdir(previous_cwd)
        if int(learner.grad_step) != config["n_steps"]:
            raise RuntimeError("IQL gradient update count differs from requested fixed budget")
        model_path = output / "iql_model.d3"
        policy_path = output / "policy.pt"
        learner.save(str(model_path))
        learner.save_policy(str(policy_path))
        reloaded = d3rlpy.load_learnable(str(model_path), device=config["device"])
        parity_data = selection_data if selection_data is not None else data
        fixed = parity_data.observations[:min(10_000, len(parity_data.observations))]
        if selection_data is None:
            # Toy compatibility only: repeat the tiny three-step fixture so the
            # cross-process contract checks the required batch size without
            # pretending that duplicate states are market/position coverage.
            repeats = math.ceil(10_000 / len(fixed))
            fixed = np.tile(fixed, (repeats, 1))[:10_000].copy()
        expected = np.asarray(learner.predict(fixed), dtype=np.float32)
        restored = np.asarray(reloaded.predict(fixed), dtype=np.float32)
        model_error = np.abs(expected.astype(np.float64) - restored.astype(np.float64))
        if not np.allclose(expected, restored, rtol=0, atol=1e-6):
            raise RuntimeError("d3rlpy save/load action parity failed")
        scripted = torch.jit.load(str(policy_path), map_location=config["device"])
        with torch.no_grad():
            scripted_actions = scripted(torch.as_tensor(fixed, device=config["device"])).cpu().numpy()
        export_error = np.abs(expected.astype(np.float64) - scripted_actions.astype(np.float64))
        if scripted_actions.shape != expected.shape or not np.allclose(expected, scripted_actions, rtol=1e-5, atol=1e-5):
            raise RuntimeError("TorchScript policy export parity failed")
        parity_path = output / "parity_inputs.npz"
        with parity_path.open("xb") as stream:
            np.savez_compressed(stream, observations=fixed, expected_actions=expected)
        metrics = {
            "training": _metrics(learner, data, reward_scale=float(config["reward_scale"]),
                                 source="fit_dataset_diagnostic_only"),
            "selection": None,
        }
        if selection_data is not None:
            metrics["selection"] = _metrics(
                learner, selection_data, reward_scale=float(config["reward_scale"]),
                source="reserved_train_inner_selection_dataset",
            )
            metrics["selection"]["q_calibration"] = _q_calibration(
                learner, selection_data, gamma=float(config["gamma"]),
                reward_scale=float(config["reward_scale"]),
            )
        metrics["train_fit_last_epoch"] = history[-1][1] if history else {}
        metrics["gradient_steps"] = int(learner.grad_step)
        metrics["save_load_parity_max_abs_error"] = float(model_error.max())
        metrics["torchscript_parity_max_abs_error"] = float(export_error.max())
        metrics["cross_process_parity_source"] = (
            "reserved_train_inner_selection_dataset" if selection_data is not None
            else "synthetic_smoke_training_dataset")
        metrics["cross_process_parity_samples"] = int(len(fixed))
        metrics["cross_process_parity_10k_complete"] = bool(len(fixed) >= 10_000)
        metadata = {
            "schema": "rles.il.iql_run.v1", "purpose": config["purpose"],
            "run_label": config["run_label"],
            "dataset_manifest": data.manifest, "config": config,
            "dataset_payload_sha256": data.manifest.get("payload_sha256"),
            "training_episode_windows_ns": data.episode_windows_ns.tolist(),
            "selection_dataset_manifest": None if selection_data is None else selection_data.manifest,
            "selection_episode_windows_ns": (
                None if selection_data is None else selection_data.episode_windows_ns.tolist()),
            "selection_dataset_payload_sha256": (
                None if selection_data is None else selection_data.manifest.get("payload_sha256")),
            "runtime": {"python": platform.python_version(), "d3rlpy": version,
                        "torch": torch.__version__, "numpy": np.__version__},
            "metrics": metrics,
            "artifacts_sha256": {path.name: _hash(path)
                                 for path in (model_path, policy_path, parity_path)},
            "historical_economic_claim": False,
        }
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True,
                                                          allow_nan=False) + "\n", encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True,
                                                           allow_nan=False) + "\n", encoding="utf-8")
        (output / "COMPLETED").write_text("IQL engineering run complete; native economic evaluation not implied.\n",
                                          encoding="utf-8")
    except BaseException:
        (output / "FAILED").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(train_iql(args.config))


if __name__ == "__main__":
    main()





