"""Safe NPZ dataset transport and exact d3rlpy episode adaptation."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WorkerDataset:
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_offsets: np.ndarray
    episode_windows_ns: np.ndarray
    manifest: dict[str, Any]


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_worker_dataset(path: str | Path) -> WorkerDataset:
    target = Path(path).resolve()
    sidecar = target.with_suffix(".json")
    if not target.is_file() or not sidecar.is_file():
        raise ValueError("worker dataset NPZ and JSON sidecar are both required")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("schema") != "rles.il.worker_dataset.v1":
        raise ValueError("unsupported worker dataset schema")
    if metadata.get("payload_file") != target.name or metadata.get("payload_sha256") != _file_hash(target):
        raise ValueError("worker dataset filename/hash mismatch")
    source = metadata.get("source_manifest", {})
    allowed_source = (source.get("account_truth") == "nautilus_native"
        or (source.get("account_truth") == "synthetic_mechanism"
            and source.get("purpose") == "toy_smoke"))
    if source.get("split") != "train_inner" or not allowed_source:
        raise ValueError("worker accepts native Train-inner data or explicitly tagged toy smoke data")
    native_identity = metadata.get("native_identity")
    if source.get("account_truth") == "nautilus_native":
        expected_identity_keys = {
            "dataset_hash", "fold_hash", "config_hash", "transition_hash", "schema_hash",
        }
        folds = source.get("fold_hashes")
        time_range = source.get("allowed_time_range_ns")
        if (not isinstance(native_identity, dict) or set(native_identity) != expected_identity_keys
                or source.get("source_data_hash") != native_identity.get("dataset_hash")
                or source.get("source_environment_config_hash") != native_identity.get("config_hash")
                or source.get("transition_hash") != native_identity.get("transition_hash")
                or source.get("observation_schema_hash") != native_identity.get("schema_hash")
                or not isinstance(folds, list) or native_identity.get("fold_hash") not in folds
                or not isinstance(time_range, list) or len(time_range) != 2
                or type(time_range[0]) is not int or type(time_range[1]) is not int):
            raise ValueError("native worker manifest lacks a complete source, fold, and time identity")
    required = {"observations", "next_observations", "actions", "rewards", "terminated", "truncated",
                "episode_offsets", "episode_windows_ns", "native_identity_hashes"}
    with np.load(target, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("worker NPZ fields differ from the frozen schema")
        arrays = {name: np.array(archive[name], copy=True) for name in required}
    obs = arrays["observations"]
    next_obs = arrays["next_observations"]
    actions = arrays["actions"]
    rewards = arrays["rewards"]
    terminated = arrays["terminated"]
    truncated = arrays["truncated"]
    offsets = arrays["episode_offsets"]
    windows = arrays["episode_windows_ns"]
    identity_hashes = arrays["native_identity_hashes"]
    count = len(actions)
    if (obs.ndim != 2 or next_obs.shape != obs.shape or actions.shape != (count, 1)
            or rewards.shape != (count,) or terminated.shape != (count,)
            or truncated.shape != (count,) or offsets.ndim != 1 or len(offsets) < 2
            or offsets.dtype != np.int64 or offsets[0] != 0 or offsets[-1] != count
            or windows.shape != (len(offsets) - 1, 2) or windows.dtype != np.int64
            or identity_hashes.dtype.kind != "U"
            or np.any(windows[:, 0] <= 0) or np.any(windows[:, 1] <= windows[:, 0])
            or np.any(np.diff(offsets) <= 0)):
        raise ValueError("worker dataset array shapes/episode offsets are invalid")
    if source.get("account_truth") == "nautilus_native":
        expected_hashes = [native_identity[key] for key in sorted(native_identity)]
        if identity_hashes.shape != (len(expected_hashes),) or identity_hashes.tolist() != expected_hashes:
            raise ValueError("native source identity differs from the checksummed worker payload")
    elif identity_hashes.size != 0:
        raise ValueError("synthetic smoke payload cannot claim a native source identity")
    previous_window: tuple[int, int] | None = None
    for raw_start, raw_end in windows:
        window = (int(raw_start), int(raw_end))
        if previous_window is not None and window != previous_window and window[0] < previous_window[1]:
            raise ValueError("worker episode windows overlap without being an exact shared market block")
        previous_window = window
    if source.get("account_truth") == "nautilus_native":
        if (int(windows[:, 0].min()) < time_range[0]
                or int(windows[:, 1].max()) > time_range[1]):
            raise ValueError("native episode windows escape the declared source time coverage")
    if (obs.dtype != np.float32 or next_obs.dtype != np.float32 or actions.dtype != np.float32
            or rewards.dtype != np.float32 or terminated.dtype != np.bool_ or truncated.dtype != np.bool_
            or not np.isfinite(obs).all() or not np.isfinite(next_obs).all()
            or not np.isfinite(actions).all() or not np.isfinite(rewards).all()
            or (np.abs(actions) > 1).any() or np.any(terminated & truncated)):
        raise ValueError("worker arrays have invalid dtype, non-finite values, or invalid labels")
    for begin, end in zip(offsets[:-1], offsets[1:]):
        lo, hi = int(begin), int(end)
        if not (terminated[hi - 1] or truncated[hi - 1]):
            raise ValueError("each worker episode must end with an explicit terminal/timeout flag")
        if np.any(terminated[lo:hi - 1] | truncated[lo:hi - 1]):
            raise ValueError("terminal flags may occur only on the final transition of an episode")
        if hi - lo > 1 and not np.array_equal(next_obs[lo:hi - 1], obs[lo + 1:hi]):
            raise ValueError("T+1/T state alignment failed inside an episode")
    if (len(offsets) - 1 != int(metadata.get("episode_count", -1))
            or len(obs) != int(metadata.get("transition_count", -1))
            or obs.shape[1] != int(metadata.get("observation_dim", -1))):
        raise ValueError("worker dataset sidecar counts/schema do not match payload")
    return WorkerDataset(obs, next_obs, actions, rewards, terminated, truncated, offsets, windows, metadata)


def build_d3rlpy_replay(dataset: WorkerDataset, *, reward_scale: float, truncation_policy: str):
    """Build explicit Episode objects while preserving each native transition."""
    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("reward_scale must be finite and positive")
    if truncation_policy not in {"terminal_no_bootstrap", "timeout_bootstrap"}:
        raise ValueError("explicit truncation_policy required")
    from d3rlpy.constants import ActionSpace
    from d3rlpy.dataset import Episode, InfiniteBuffer, ReplayBuffer

    episodes = []
    for begin, end in zip(dataset.episode_offsets[:-1], dataset.episode_offsets[1:]):
        lo, hi = int(begin), int(end)
        last = hi - 1
        is_timeout = bool(dataset.truncated[last])
        terminal = bool(dataset.terminated[last]) or (
            is_timeout and truncation_policy == "terminal_no_bootstrap"
        )
        actions = dataset.actions[lo:hi]
        rewards = (dataset.rewards[lo:hi] * np.float32(reward_scale)).reshape((-1, 1))
        if terminal:
            # d3rlpy's terminal picker masks the next value and uses a zero next
            # state, so the native post-settlement state is intentionally not bootstrapped.
            states = dataset.observations[lo:hi]
        else:
            # d3rlpy Episode represents a timeout as a nonterminal sequence whose
            # last array row is a next-state sentinel. Append unused dummy a/r so
            # T actual transitions retain the original T+1 observations.
            states = np.concatenate((dataset.observations[lo:hi], dataset.next_observations[last:last + 1]))
            actions = np.concatenate((actions, np.zeros((1, 1), dtype=np.float32)))
            rewards = np.concatenate((rewards, np.zeros((1, 1), dtype=np.float32)))
        episode = Episode(
            observations=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminated=terminal,
        )
        if episode.transition_count != hi - lo:
            raise ValueError("d3rlpy adapter changed the number of real transitions")
        episodes.append(episode)
    replay = ReplayBuffer(
        buffer=InfiniteBuffer(), episodes=episodes,
        action_space=ActionSpace.CONTINUOUS, action_size=1,
    )
    if replay.transition_count != len(dataset.actions):
        raise ValueError("d3rlpy replay transition count differs from native dataset")
    # Golden-check first, last, and all transitions before allowing training.
    cursor = 0
    for episode in replay.episodes:
        for index in range(episode.transition_count):
            transition = replay.transition_picker(episode, index)
            source = cursor + index
            if (not np.array_equal(transition.observation, dataset.observations[source])
                    or not np.array_equal(transition.action, dataset.actions[source])
                    or not np.isclose(float(transition.reward[0]), float(dataset.rewards[source] * reward_scale))):
                raise ValueError("d3rlpy transition conversion failed golden comparison")
            terminal_expected = bool(dataset.terminated[source]) or (
                bool(dataset.truncated[source]) and truncation_policy == "terminal_no_bootstrap"
            )
            if bool(transition.terminal) != terminal_expected:
                raise ValueError("d3rlpy terminal mask differs from frozen timeout policy")
            if not terminal_expected and not np.array_equal(transition.next_observation, dataset.next_observations[source]):
                raise ValueError("d3rlpy timeout next-observation alignment failed")
        cursor += episode.transition_count
    return replay








