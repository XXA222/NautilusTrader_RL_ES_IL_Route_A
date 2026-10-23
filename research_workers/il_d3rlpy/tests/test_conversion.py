from __future__ import annotations

import unittest
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import numpy as np

from il_worker.data import WorkerDataset, build_d3rlpy_replay, load_worker_dataset
from il_worker.train_iql import _q_calibration, _validate_selection_dataset


class D3rlpyConversionTests(unittest.TestCase):
    def setUp(self):
        self.dataset = WorkerDataset(
            observations=np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
            next_observations=np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
            actions=np.asarray([[0.25], [0.5]], dtype=np.float32),
            rewards=np.asarray([0.1, -0.2], dtype=np.float32),
            terminated=np.asarray([False, False], dtype=np.bool_),
            truncated=np.asarray([False, True], dtype=np.bool_),
            episode_offsets=np.asarray([0, 2], dtype=np.int64),
            episode_windows_ns=np.asarray([[60, 180]], dtype=np.int64),
            manifest={"split": "train_inner", "account_truth": "nautilus_native"},
        )

    def test_timeout_bootstrap_preserves_the_final_native_next_state(self):
        replay = build_d3rlpy_replay(
            self.dataset, reward_scale=10.0, truncation_policy="timeout_bootstrap",
        )
        self.assertEqual(replay.transition_count, 2)
        self.assertEqual(replay.episodes[0].transition_count, 2)
        last = replay.transition_picker(replay.episodes[0], 1)
        np.testing.assert_array_equal(last.next_observation, self.dataset.next_observations[-1])
        self.assertEqual(float(last.terminal), 0.0)
        self.assertAlmostEqual(float(last.reward[0]), -2.0)

    def test_terminalized_timeout_is_explicitly_non_bootstrapable(self):
        replay = build_d3rlpy_replay(
            self.dataset, reward_scale=1.0, truncation_policy="terminal_no_bootstrap",
        )
        last = replay.transition_picker(replay.episodes[0], 1)
        self.assertEqual(float(last.terminal), 1.0)
        np.testing.assert_array_equal(last.next_observation, np.zeros((2,), dtype=np.float32))

    def test_worker_refuses_implicit_timeout_semantics(self):
        with self.assertRaisesRegex(ValueError, "explicit truncation_policy"):
            build_d3rlpy_replay(self.dataset, reward_scale=1.0, truncation_policy="")

    def test_native_worker_payload_binds_episode_windows_and_source_identity(self):
        identity = {
            "dataset_hash": "a" * 64, "fold_hash": "b" * 64, "config_hash": "c" * 64,
            "transition_hash": "d" * 64, "schema_hash": "e" * 64,
        }
        source = {
            "split": "train_inner", "account_truth": "nautilus_native",
            "source_data_hash": identity["dataset_hash"], "fold_hashes": [identity["fold_hash"]],
            "source_environment_config_hash": identity["config_hash"],
            "transition_hash": identity["transition_hash"],
            "observation_schema_hash": identity["schema_hash"],
            "allowed_time_range_ns": [60, 180],
        }
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "native.npz"
            np.savez_compressed(
                payload, observations=np.asarray([[0, 1]], dtype=np.float32),
                next_observations=np.asarray([[1, 2]], dtype=np.float32),
                actions=np.asarray([[0.25]], dtype=np.float32),
                rewards=np.asarray([0.1], dtype=np.float32),
                terminated=np.asarray([True], dtype=np.bool_),
                truncated=np.asarray([False], dtype=np.bool_),
                episode_offsets=np.asarray([0, 1], dtype=np.int64),
                episode_windows_ns=np.asarray([[60, 180]], dtype=np.int64),
                native_identity_hashes=np.asarray([identity[key] for key in sorted(identity)], dtype="U64"),
            )
            payload.with_suffix(".json").write_text(json.dumps({
                "schema": "rles.il.worker_dataset.v1", "payload_file": payload.name,
                "payload_sha256": sha256(payload.read_bytes()).hexdigest(),
                "episode_count": 1, "transition_count": 1, "observation_dim": 2,
                "source_manifest": source, "native_identity": identity,
            }), encoding="utf-8")
            restored = load_worker_dataset(payload)
            np.testing.assert_array_equal(restored.episode_windows_ns, [[60, 180]])

    def test_q_calibration_uses_held_out_settled_realized_returns(self):
        settled = WorkerDataset(
            observations=self.dataset.observations,
            next_observations=self.dataset.next_observations,
            actions=self.dataset.actions, rewards=self.dataset.rewards,
            terminated=np.asarray([False, True], dtype=np.bool_),
            truncated=np.asarray([False, False], dtype=np.bool_),
            episode_offsets=self.dataset.episode_offsets,
            episode_windows_ns=self.dataset.episode_windows_ns,
            manifest=self.dataset.manifest,
        )

        class _Algo:
            def predict_value(self, observations, actions):
                return np.zeros(len(actions), dtype=np.float32)

        result = _q_calibration(_Algo(), settled, gamma=0.9, reward_scale=1.0)
        self.assertEqual(result["source"], "reserved_train_inner_selection_dataset")
        self.assertAlmostEqual(result["realized_discounted_return_mean"], -0.14)
        self.assertAlmostEqual(result["bias_q_minus_return"], 0.14)
        self.assertEqual(sum(row["count"] for row in result["quintile_bins"]), 2)

    def test_iql_fit_and_selection_reject_overlapping_episode_windows(self):
        identity = {name: name[0] * 64 for name in (
            "dataset_hash", "fold_hash", "config_hash", "transition_hash", "schema_hash",
        )}
        source = {"split": "train_inner", "account_truth": "nautilus_native"}
        training = WorkerDataset(
            self.dataset.observations, self.dataset.next_observations, self.dataset.actions,
            self.dataset.rewards, self.dataset.terminated, self.dataset.truncated,
            self.dataset.episode_offsets, np.asarray([[60, 180]], dtype=np.int64),
            {"source_manifest": source, "native_identity": identity},
        )
        selection = WorkerDataset(
            self.dataset.observations, self.dataset.next_observations, self.dataset.actions,
            self.dataset.rewards, self.dataset.terminated, self.dataset.truncated,
            self.dataset.episode_offsets, np.asarray([[150, 270]], dtype=np.int64),
            {"source_manifest": source, "native_identity": identity},
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            _validate_selection_dataset(training, selection)


if __name__ == "__main__":
    unittest.main()
