import json
from datetime import datetime
from pathlib import Path

import pytest

from run_route_a_meta_study_20261029 import _registered_screen_split_evidence


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((ROOT / "configs/route_a_meta_x1_train_prereg_20261029.json").read_text())


def test_registered_screen_records_temporal_separation_without_claiming_independent_trials():
    config = _config()
    window = config["intervals"]["native_screen"][0]
    available_after = int(datetime.fromisoformat(
        config["intervals"]["train_calibration"][1].replace("Z", "+00:00")
    ).timestamp() * 1_000_000_000)

    evidence = _registered_screen_split_evidence(
        config, window, model_available_after_ns=available_after
    )

    assert evidence["status"] == "PASS_REGISTERED_TEMPORAL_SEPARATION"
    assert evidence["independent_trials_claimed"] is False
    assert evidence["screen_interval_ns"][0] == available_after


def test_registered_screen_rejects_unregistered_window_and_late_model():
    config = _config()
    registered = config["intervals"]["native_screen"][0]
    invalid_window = [registered[0], "2026-12-15T00:00:00Z"]

    with pytest.raises(ValueError, match="not present"):
        _registered_screen_split_evidence(config, invalid_window)
    with pytest.raises(ValueError, match="not available"):
        _registered_screen_split_evidence(
            config, registered, model_available_after_ns=2**63 - 1
        )
