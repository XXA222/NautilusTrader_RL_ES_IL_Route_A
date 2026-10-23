from __future__ import annotations

import json

import numpy as np
import pytest

from rles_il_core.dataset import export_d3rlpy_npz
from test_core import trajectory


def test_worker_npz_export_is_train_only_checksummed_and_preserves_t_plus_one_t(tmp_path):
    manifest = {
        "schema": "rles.il.dataset.v1", "run_label": "toy",
        "split": "train_inner", "account_truth": "nautilus_native",
    }
    path = export_d3rlpy_npz(tmp_path / "worker.npz", (trajectory(),), manifest)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["schema"] == "rles.il.worker_dataset.v1"
    assert sidecar["episode_count"] == 1
    assert sidecar["transition_count"] == 2
    assert len(sidecar["payload_sha256"]) == 64
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["observations"].shape == (2, 2)
        assert arrays["next_observations"].shape == (2, 2)
        assert arrays["actions"].shape == (2, 1)
        assert arrays["episode_offsets"].tolist() == [0, 2]
        assert arrays["episode_windows_ns"].tolist() == [[60, 240]]
        assert arrays["native_identity_hashes"].tolist() == [
            "c" * 64, "a" * 64, "b" * 64, "e" * 64, "d" * 64,
        ]
        assert arrays["terminated"].tolist() == [False, True]
    with pytest.raises(FileExistsError):
        export_d3rlpy_npz(path, (trajectory(),), manifest)


def test_worker_npz_export_refuses_validation_split(tmp_path):
    with pytest.raises(ValueError, match="Train-inner"):
        export_d3rlpy_npz(
            tmp_path / "validation.npz", (trajectory(),),
            {"split": "validation", "account_truth": "nautilus_native"},
        )
