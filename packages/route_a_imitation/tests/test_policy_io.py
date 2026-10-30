import json

import pytest

from rles_route_a.teacher_research.candidate_event_union import TeacherProposal, group_teacher_proposals
from rles_route_a.teacher_research.policy_io import (
    load_frozen_entry_policy, policy_document, policy_from_document, save_frozen_entry_policy,
)
from test_candidate_collector import observation
from test_meta_gated_teacher import trained_policy


def test_policy_roundtrip_preserves_predictions_and_costs(tmp_path):
    original = trained_policy()
    path = tmp_path / "policy.json"
    save_frozen_entry_policy(original, path)
    restored = load_frozen_entry_policy(path)
    obs = observation(10, ema_9=101.)
    group, = group_teacher_proposals([TeacherProposal(obs, "T01_EMA_9_21", 1, "cross")])
    assert restored.score(group) == original.score(group)
    assert (restored.fee_bps_per_leg, restored.slippage_bps_per_leg) == (5., 1.)
    assert restored.fit_effective_label_count == original.fit_effective_label_count == 4.
    assert restored.calibration_effective_label_count == 4.
    assert restored.fit_weighting_id == "AVERAGE_INVERSE_LABEL_CONCURRENCY_V1"
    with pytest.raises(ValueError):
        restored.mean.setflags(write=True)
    with pytest.raises(FileExistsError):
        save_frozen_entry_policy(original, path)


def test_invalid_policy_is_rejected(tmp_path):
    doc = policy_document(trained_policy())
    legacy = {**doc, "schema": "rles.route-a.frozen-entry-meta-policy.v1"}
    with pytest.raises(ValueError, match="invalid frozen policy schema"):
        policy_from_document(legacy)
    doc["policy"]["scale"][0] = 0.
    with pytest.raises(ValueError, match="scaling"):
        policy_from_document(doc)
    doc = policy_document(trained_policy())
    doc["policy"]["fee_bps_per_leg"] = 4.
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="transaction costs"):
        load_frozen_entry_policy(path)
    doc = policy_document(trained_policy())
    doc["policy"]["fit_weighting_id"] = "UNWEIGHTED"
    with pytest.raises(ValueError, match="weighting recipe"):
        policy_from_document(doc)
