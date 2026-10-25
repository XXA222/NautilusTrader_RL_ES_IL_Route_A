"""Generate and train a synthetic d3rlpy compatibility smoke episode set."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .train_iql import train_iql


def create_toy_run(destination: str | Path) -> Path:
    root = Path(destination).resolve()
    if root.exists():
        raise FileExistsError("toy run destination is immutable")
    root.mkdir(parents=True)
    data_path = root / "toy_dataset.npz"
    config_path = root / "toy_config.json"
    run_path = root / "run"
    episodes, steps, obs_dim = 4, 3, 4
    total = episodes * steps
    observations = np.zeros((total, obs_dim), dtype=np.float32)
    next_observations = np.zeros_like(observations)
    actions = np.zeros((total, 1), dtype=np.float32)
    rewards = np.zeros(total, dtype=np.float32)
    terminated = np.zeros(total, dtype=np.bool_)
    truncated = np.zeros(total, dtype=np.bool_)
    offsets = [0]
    for episode in range(episodes):
        start = episode * steps
        for step in range(steps):
            index = start + step
            observations[index] = (episode, step, episode % 2, step / steps)
            next_observations[index] = (episode, step + 1, episode % 2, (step + 1) / steps)
            actions[index, 0] = (-0.5 if episode % 2 == 0 else 0.5) * (1 if step < 2 else 0.5)
            rewards[index] = np.float32((1 if episode % 2 else -1) * (step + 1) * 0.001)
        terminated[start + steps - 1] = True
        offsets.append(start + steps)
    with data_path.open("xb") as stream:
        np.savez_compressed(
            stream, observations=observations, next_observations=next_observations,
            actions=actions, rewards=rewards, terminated=terminated,
            truncated=truncated, episode_offsets=np.asarray(offsets, dtype=np.int64),
            episode_windows_ns=np.asarray([
                (episode * 100 + 1, episode * 100 + 91) for episode in range(episodes)
            ], dtype=np.int64),
            native_identity_hashes=np.asarray([], dtype="U64"),
        )
    manifest = {
        "schema": "rles.il.worker_dataset.v1", "payload_file": data_path.name,
        "payload_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "episode_count": episodes, "transition_count": total,
        "observation_dim": obs_dim,
        "source_manifest": {"split": "train_inner", "account_truth": "synthetic_mechanism",
                             "purpose": "toy_smoke", "run_label": "synthetic-mechanism-only"},
        "native_identity": {},
        "episode_terminal_semantics": "separate_terminated_and_truncated",
    }
    data_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    config = {
        "schema": "rles.il.iql.v1", "purpose": "synthetic_smoke",
        "run_label": "IQL_TOY_SMOKE", "selection_dataset": None,
        "dataset": str(data_path), "output_dir": str(run_path),
        "device": "cpu:0", "seed": 11, "n_steps": 4,
        "n_steps_per_epoch": 2, "batch_size": 2, "gamma": 0.99,
        "expectile": 0.7, "weight_temp": 1.0, "max_weight": 20.0,
        "reward_scale": 1.0, "truncation_policy": "terminal_no_bootstrap",
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        return train_iql(config_path)
    except BaseException:
        if not run_path.exists():
            run_path.mkdir()
            (run_path / "FAILED").write_text("Toy config rejected before worker run.\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(create_toy_run(args.output))


if __name__ == "__main__":
    main()




