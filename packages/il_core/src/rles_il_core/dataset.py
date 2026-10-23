"""Validated T+1/T trajectories and immutable Parquet dataset publication."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


_IDENTITY_KEYS = {"dataset_hash", "fold_hash", "config_hash", "transition_hash", "schema_hash"}


@dataclass(frozen=True, slots=True)
class Transition:
    observation: tuple[float, ...]
    action: float
    next_observation: tuple[float, ...]
    r_econ: float
    r_shape: float
    r_train: float
    terminated: bool
    truncated: bool
    decision_ts_ns: int
    next_decision_ts_ns: int
    native_equity_before: str
    native_equity_after: str
    expert_id: str
    native_identity: Mapping[str, str]
    episode_window: Mapping[str, int]
    execution: Mapping[str, Any]
    native_fills: tuple[Mapping[str, Any], ...] = ()
    native_funding: tuple[Mapping[str, Any], ...] = ()
    terminal_settlement: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("observation", "next_observation"):
            value = tuple(float(x) for x in getattr(self, name))
            if not value or not all(math.isfinite(x) for x in value):
                raise ValueError(f"{name} must be a finite non-empty vector")
            object.__setattr__(self, name, value)
        if len(self.observation) != len(self.next_observation):
            raise ValueError("observation schema changed within a transition")
        scalars = (self.action, self.r_econ, self.r_shape, self.r_train)
        if not all(math.isfinite(float(x)) for x in scalars) or not -1 <= self.action <= 1:
            raise ValueError("non-finite or out-of-range transition scalar")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("termination flags must be Boolean")
        if self.terminated and self.truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        if (type(self.decision_ts_ns) is not int or type(self.next_decision_ts_ns) is not int
                or self.next_decision_ts_ns <= self.decision_ts_ns):
            raise ValueError("decision clock must advance strictly")
        if type(self.expert_id) is not str or not self.expert_id:
            raise ValueError("expert source is required")
        identity = dict(self.native_identity)
        if (set(identity) != _IDENTITY_KEYS
                or any(type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None
                       for v in identity.values())):
            raise ValueError("complete frozen native and observation identity required")
        window = dict(self.episode_window)
        required_window = {"start_ns", "end_ns", "decision_steps", "episode_index", "start_offset"}
        if (set(window) != required_window
                or any(type(window[key]) is not int for key in required_window)
                or window["start_ns"] <= 0 or window["end_ns"] <= window["start_ns"]
                or window["decision_steps"] <= 0 or window["episode_index"] < 0
                or window["start_offset"] < 0):
            raise ValueError("exact historical episode window required")
        if type(self.execution) is not dict and not isinstance(self.execution, Mapping):
            raise ValueError("execution plan and risk facts required")
        object.__setattr__(self, "native_identity", identity)
        object.__setattr__(self, "episode_window", window)
        object.__setattr__(self, "execution", dict(self.execution))
        object.__setattr__(self, "native_fills", tuple(dict(x) for x in self.native_fills))
        object.__setattr__(self, "native_funding", tuple(dict(x) for x in self.native_funding))
        if self.terminal_settlement is not None:
            object.__setattr__(self, "terminal_settlement", dict(self.terminal_settlement))


@dataclass(frozen=True, slots=True)
class EpisodeTrajectory:
    expert_id: str
    observations: tuple[tuple[float, ...], ...]
    transitions: tuple[Transition, ...]
    episode_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(tuple(map(float, row)) for row in self.observations))
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "episode_metadata", dict(self.episode_metadata))
        validate_trajectory(self)


def validate_trajectory(trajectory: EpisodeTrajectory) -> None:
    """Fail closed unless a full, finite T+1/T episode is internally aligned."""
    if type(trajectory.expert_id) is not str or not trajectory.expert_id:
        raise ValueError("trajectory expert identity missing")
    obs, transitions = trajectory.observations, trajectory.transitions
    if not transitions or len(obs) != len(transitions) + 1:
        raise ValueError("episode must contain exactly T+1 observations and T transitions")
    dim = len(obs[0])
    if dim == 0 or any(len(row) != dim or not all(math.isfinite(x) for x in row) for row in obs):
        raise ValueError("trajectory observations have inconsistent shape or non-finite values")
    for index, transition in enumerate(transitions):
        if type(transition) is not Transition or transition.expert_id != trajectory.expert_id:
            raise ValueError("transition source differs from episode identity")
        if transition.observation != obs[index] or transition.next_observation != obs[index + 1]:
            raise ValueError("transition state is not aligned with its episode observation sequence")
        if index < len(transitions) - 1 and (transition.terminated or transition.truncated):
            raise ValueError("terminal flags are only permitted on the last transition")
    last = transitions[-1]
    if not (last.terminated or last.truncated):
        raise ValueError("episode must record an explicit termination or truncation")
    if last.terminal_settlement is None or not last.terminal_settlement:
        raise ValueError("finite native episodes must retain forced settlement facts")
    identities = {tuple(sorted(t.native_identity.items())) for t in transitions}
    windows = {tuple(sorted(t.episode_window.items())) for t in transitions}
    if len(identities) != 1 or len(windows) != 1:
        raise ValueError("native identity and episode window cannot change within an episode")


def assign_time_blocks(
    trajectories: Sequence[EpisodeTrajectory], blocks: Mapping[str, tuple[int, int]],
) -> dict[str, tuple[EpisodeTrajectory, ...]]:
    """Assign whole window groups to predeclared non-overlapping half-open UTC ns blocks."""
    if not blocks:
        raise ValueError("at least one frozen time block is required")
    ordered = sorted((int(start), int(end), name) for name, (start, end) in blocks.items())
    if (any(start <= 0 or end <= start for start, end, _ in ordered)
            or any(left[1] > right[0] for left, right in zip(ordered, ordered[1:]))):
        raise ValueError("time blocks must be valid, disjoint and ordered")
    result: dict[str, list[EpisodeTrajectory]] = {name: [] for _, _, name in ordered}
    grouped: dict[tuple[int, int], list[EpisodeTrajectory]] = {}
    for trajectory in trajectories:
        validate_trajectory(trajectory)
        window = trajectory.transitions[0].episode_window
        key = (int(window["start_ns"]), int(window["end_ns"]))
        grouped.setdefault(key, []).append(trajectory)
    for (start, end), group in grouped.items():
        matches = [name for block_start, block_end, name in ordered
                   if block_start <= start and end < block_end]
        if len(matches) != 1:
            raise ValueError("an episode window crosses or falls outside frozen Train/Val/Test blocks")
        result[matches[0]].extend(group)
    return {name: tuple(rows) for name, rows in result.items()}


def export_d3rlpy_npz(
    destination: str | Path,
    trajectories: Sequence[EpisodeTrajectory],
    manifest: Mapping[str, Any],
) -> Path:
    """Export a safe, checksummed episode payload for the isolated Python worker."""
    target = Path(destination).resolve()
    if target.suffix != ".npz":
        raise ValueError("isolated worker payload must use the .npz suffix")
    sidecar = target.with_suffix(".json")
    if target.exists() or sidecar.exists():
        raise FileExistsError("isolated worker payloads are immutable")
    if manifest.get("split") != "train_inner" or manifest.get("account_truth") != "nautilus_native":
        raise ValueError("isolated worker only accepts Train-inner Nautilus-native manifests")
    rows = tuple(sorted(
        trajectories,
        key=lambda item: (item.transitions[0].episode_window["start_ns"],
                          item.transitions[0].episode_window["end_ns"], item.expert_id,
                          int(item.episode_metadata.get("seed", 0))),
    ))
    if not rows:
        raise ValueError("cannot export an empty worker dataset")
    identity_sets: dict[str, set[str]] = {key: set() for key in _IDENTITY_KEYS}
    offsets = [0]
    observations: list[tuple[float, ...]] = []
    next_observations: list[tuple[float, ...]] = []
    actions: list[float] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    episode_windows: list[tuple[int, int]] = []
    for trajectory in rows:
        validate_trajectory(trajectory)
        if trajectory.episode_metadata.get("split") != "train_inner":
            raise ValueError("isolated worker export refuses non-Train-inner episodes")
        for key, value in trajectory.transitions[0].native_identity.items():
            identity_sets[key].add(value)
        window = trajectory.transitions[0].episode_window
        episode_windows.append((window["start_ns"], window["end_ns"]))
        for transition in trajectory.transitions:
            observations.append(transition.observation)
            next_observations.append(transition.next_observation)
            actions.append(transition.action)
            rewards.append(transition.r_econ)
            terminated.append(transition.terminated)
            truncated.append(transition.truncated)
        offsets.append(len(actions))
    if any(len(values) != 1 for values in identity_sets.values()):
        raise ValueError("one isolated worker dataset cannot mix frozen identities")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.stem}.worker-staging-", dir=target.parent))
    staged_payload = staging / target.name
    staged_sidecar = staging / sidecar.name
    try:
        with staged_payload.open("xb") as stream:
            np.savez_compressed(
                stream,
                observations=np.asarray(observations, dtype=np.float32),
                next_observations=np.asarray(next_observations, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32).reshape((-1, 1)),
                rewards=np.asarray(rewards, dtype=np.float32),
                terminated=np.asarray(terminated, dtype=np.bool_),
                truncated=np.asarray(truncated, dtype=np.bool_),
                episode_offsets=np.asarray(offsets, dtype=np.int64),
                episode_windows_ns=np.asarray(episode_windows, dtype=np.int64),
                native_identity_hashes=np.asarray(
                    [next(iter(identity_sets[key])) for key in sorted(identity_sets)], dtype="U64",
                ),
            )
        payload_hash = _hash(staged_payload)
        staged_sidecar.write_text(json.dumps({
            "schema": "rles.il.worker_dataset.v1",
            "payload_file": target.name,
            "payload_sha256": payload_hash,
            "episode_count": len(rows),
            "transition_count": len(actions),
            "observation_dim": len(rows[0].observations[0]),
            "source_manifest": dict(manifest),
            "native_identity": {key: next(iter(values)) for key, values in identity_sets.items()},
            "episode_terminal_semantics": "separate_terminated_and_truncated",
        }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(staged_payload, target)
        try:
            os.replace(staged_sidecar, sidecar)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    finally:
        for child in staging.iterdir():
            child.unlink(missing_ok=True)
        staging.rmdir()
    return target

def to_offline_episode_arrays(trajectory: EpisodeTrajectory) -> dict[str, np.ndarray]:
    """Return dependency-neutral arrays for the isolated d3rlpy worker adapter."""
    validate_trajectory(trajectory)
    return {
        "observations": np.asarray(trajectory.observations, dtype=np.float32),
        "actions": np.asarray([row.action for row in trajectory.transitions], dtype=np.float32)[:, None],
        "rewards": np.asarray([row.r_econ for row in trajectory.transitions], dtype=np.float32),
        "terminated": np.asarray([row.terminated for row in trajectory.transitions], dtype=np.bool_),
        "truncated": np.asarray([row.truncated for row in trajectory.transitions], dtype=np.bool_),
    }


def build_dataset_manifest(factory: Any, trajectories: Sequence[EpisodeTrajectory], *,
                           run_label: str, dependency_versions: Mapping[str, str]) -> dict[str, Any]:
    """Bind a collected Train-inner dataset to frozen source, fold, scaler and feature identity."""
    if getattr(factory, "split", None) != "train_inner":
        raise ValueError("offline IL dataset manifests may only be created from Train-inner")
    rows = tuple(trajectories)
    if not rows or not run_label or not dependency_versions:
        raise ValueError("non-empty trajectories, run label and dependency versions required")
    for item in rows:
        validate_trajectory(item)
        if item.episode_metadata.get("split") != "train_inner":
            raise ValueError("dataset contains a non-Train-inner episode")
    identities = [item.transitions[0].native_identity for item in rows]
    transition_hashes = {identity["transition_hash"] for identity in identities}
    schema_hashes = {identity["schema_hash"] for identity in identities}
    dataset_hashes = {identity["dataset_hash"] for identity in identities}
    fold_hashes = {identity["fold_hash"] for identity in identities}
    config_hashes = {identity["config_hash"] for identity in identities}
    recipes = {item.episode_metadata.get("execution_recipe_hash") for item in rows}
    if any(len(values) != 1 for values in (schema_hashes, dataset_hashes, config_hashes)):
        raise ValueError("one immutable experiment dataset cannot mix source identities")
    if (len(recipes) != 1 or (len(transition_hashes) > 1 and None in recipes)
            or (next(iter(recipes)) is not None
                and next(iter(recipes)) != getattr(factory, "route_a_execution_recipe_hash", None))):
        raise ValueError("mixed native windows require one factory-bound execution recipe")
    windows = [item.transitions[0].episode_window for item in rows]
    metadata = factory.metadata()
    return {
        "schema": "rles.il.dataset.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "split": "train_inner",
        "account_truth": "nautilus_native",
        "instrument": factory.instrument,
        "market_timeframe": "1m",
        "feature_profile": factory.feature_profile,
        "source_data_hash": next(iter(dataset_hashes)),
        "fold_hashes": sorted(fold_hashes),
        "source_environment_config_hash": next(iter(config_hashes)),
        "transition_hash": (next(iter(transition_hashes)) if len(transition_hashes) == 1
                            else None),
        "transition_hashes": sorted(transition_hashes),
        "execution_recipe_hash": next(iter(recipes)),
        "observation_schema_hash": next(iter(schema_hashes)),
        "scaler_hash": metadata["identities"]["scaler"],
        "dependencies": dict(dependency_versions),
        "seeds": [item.episode_metadata["seed"] for item in rows],
        "allowed_time_range_ns": [min(w["start_ns"] for w in windows),
                                  max(w["end_ns"] for w in windows)],
        "expert_ids": sorted({item.expert_id for item in rows}),
        "all_episodes_retained": True,
        "reward_source": "native_equity_delta",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet_dataset(
    destination: str | Path, trajectories: Sequence[EpisodeTrajectory], manifest: Mapping[str, Any],
) -> Path:
    """Atomically publish immutable episodes/transitions Parquet plus checksummed manifest."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("install rles-il-core[parquet] in the active environment") from exc
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError("dataset destination is immutable and already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = tuple(trajectories)
    if not rows:
        raise ValueError("cannot publish an empty offline dataset")
    required_manifest = {
        "run_label", "split", "account_truth", "instrument", "market_timeframe",
        "feature_profile", "source_data_hash", "fold_hashes", "source_environment_config_hash",
        "transition_hash", "observation_schema_hash", "scaler_hash", "dependencies", "seeds",
        "allowed_time_range_ns", "all_episodes_retained", "reward_source",
    }
    if not required_manifest <= set(manifest):
        raise ValueError(f"dataset manifest missing {sorted(required_manifest - set(manifest))}")
    if manifest["split"] != "train_inner" or manifest["account_truth"] != "nautilus_native":
        raise ValueError("dataset must be Train-inner and Nautilus-native")
    for item in rows:
        validate_trajectory(item)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        episode_rows = [{
            "expert_id": item.expert_id,
            "observations": [list(row) for row in item.observations],
            "transition_count": len(item.transitions),
            "episode_metadata_json": json.dumps(item.episode_metadata, sort_keys=True,
                                                 separators=(",", ":"), default=_json_default),
        } for item in rows]
        transition_rows = []
        for trajectory_index, item in enumerate(rows):
            for index, transition in enumerate(item.transitions):
                row = {name: getattr(transition, name) for name in Transition.__dataclass_fields__}
                row["trajectory_index"] = trajectory_index
                row["step_index"] = index
                row["native_identity"] = dict(transition.native_identity)
                row["episode_window"] = dict(transition.episode_window)
                row["execution"] = dict(transition.execution)
                row["native_fills"] = [dict(x) for x in transition.native_fills]
                row["native_funding"] = [dict(x) for x in transition.native_funding]
                row["terminal_settlement"] = (None if transition.terminal_settlement is None
                                               else dict(transition.terminal_settlement))
                transition_rows.append(row)
        for name, data in (("episodes.parquet", episode_rows), ("transitions.parquet", transition_rows)):
            table = pa.Table.from_pylist(data)
            path = staging / name
            pq.write_table(table, path, compression="zstd")
        from .diagnostics import behavior_support_report
        (staging / "support.json").write_text(
            json.dumps(behavior_support_report(rows), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        hashes = {name: _hash(staging / name) for name in (
            "episodes.parquet", "transitions.parquet", "support.json",
        )}
        payload = dict(manifest)
        payload.update({"schema": "rles.il.dataset.v1", "files_sha256": hashes,
                        "episode_count": len(rows),
                        "transition_count": sum(len(item.transitions) for item in rows)})
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=_json_default) + "\n",
                                 encoding="utf-8")
        (staging / "COMPLETED").write_text("validated\n", encoding="ascii")
        staging.rename(target)
        for path in target.iterdir():
            path.chmod(0o444)
        target.chmod(0o555)
        return target
    except BaseException as exc:
        (staging / "FAILED").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        if not target.exists():
            staging.rename(target)
        raise


def verify_parquet_dataset(path: str | Path) -> dict[str, Any]:
    """Verify completion marker, manifest schema and independent file hashes."""
    root = Path(path).resolve()
    if not (root / "COMPLETED").is_file() or (root / "FAILED").exists():
        raise ValueError("dataset is not marked completed")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "rles.il.dataset.v1":
        raise ValueError("unsupported offline dataset manifest")
    checksums = manifest.get("files_sha256")
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("dataset checksums missing")
    for name, digest in checksums.items():
        file = root / name
        if not file.is_file() or _hash(file) != digest:
            raise ValueError(f"dataset file checksum mismatch: {name}")
    try:
        import pyarrow.parquet as pq
        if pq.read_metadata(root / "episodes.parquet").num_rows != manifest["episode_count"]:
            raise ValueError("episode Parquet count differs from manifest")
        if pq.read_metadata(root / "transitions.parquet").num_rows != manifest["transition_count"]:
            raise ValueError("transition Parquet count differs from manifest")
    except ImportError as exc:
        raise RuntimeError("install pyarrow to verify the offline dataset") from exc
    return manifest


def read_parquet_dataset(path: str | Path) -> tuple[tuple[EpisodeTrajectory, ...], dict[str, Any]]:
    """Load and revalidate every immutable episode after checksum verification."""
    root = Path(path).resolve()
    manifest = verify_parquet_dataset(root)
    import pyarrow.parquet as pq

    episodes = pq.read_table(root / "episodes.parquet").to_pylist()
    transitions = pq.read_table(root / "transitions.parquet").to_pylist()
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in transitions:
        by_episode.setdefault(int(row["trajectory_index"]), []).append(row)
    output = []
    for index, row in enumerate(episodes):
        ordered = sorted(by_episode.get(index, []), key=lambda item: int(item["step_index"]))
        data = []
        for item in ordered:
            transition_fields: dict[str, Any] = {
                field: item.get(field) for field in Transition.__dataclass_fields__
            }
            data.append(Transition(**transition_fields))
        metadata = json.loads(row["episode_metadata_json"])
        output.append(EpisodeTrajectory(
            row["expert_id"], tuple(tuple(state) for state in row["observations"]),
            tuple(data), metadata,
        ))
    if (len(output) != manifest["episode_count"]
            or sum(len(item.transitions) for item in output) != manifest["transition_count"]):
        raise ValueError("loaded Parquet counts differ from the verified manifest")
    return tuple(output), manifest




