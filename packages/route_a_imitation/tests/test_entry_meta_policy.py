from dataclasses import replace

import pytest

from rles_route_a.teacher_research.candidate_event_union import TeacherProposal, group_teacher_proposals
from rles_route_a.teacher_research.conditional_outcome_labels import GroupOutcome
from rles_route_a.teacher_research.entry_meta_policy import (EntryScore, fit_entry_meta_policy,
                                                              summarize_entry_scores)


def test_gate_diagnostics_count_overlapping_filters_without_changing_threshold():
    scores = [EntryScore("NO_TRADE", .8, 6., False, "OUT_OF_SUPPORT"),
              EntryScore("NO_TRADE", .4, 2., True, "NET_EDGE_TOO_SMALL"),
              EntryScore("NO_TRADE", .8, 8., True, "PENDING_ORDER")]
    result = summarize_entry_scores(scores, margin_bps=5.)
    assert result["accepted_count"] == 0
    assert result["gate_counts"]["support_pass"] == 2
    assert result["gate_counts"]["edge_pass"] == 2
    assert result["gate_counts"]["support_and_edge_pass"] == 1
    assert result["gate_counts"]["account_ineligible"] == 1
    assert summarize_entry_scores([], margin_bps=5.)["coverage"] is None


def _samples(observed):
    groups = {}
    labels = []
    for i in range(8):
        obs = observed(ts=(i + 1) * 1000000000, hourly=0.01 if i % 2 else -0.01)
        group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
        groups[group.event_id] = group
        labels.append(GroupOutcome(group.event_id, obs.decision_ts_ns, obs.source_id,
                                   obs.fold_id, "X1", "native", "base",
                                   obs.decision_ts_ns + 1, None, 10. if i % 2 else -10.))
    return labels[:4], labels[4:], groups


def test_frozen_entry_model_is_causal_and_respects_pending(observed):
    fit, calibration, groups = _samples(observed)
    model = fit_entry_meta_policy(fit, calibration, groups,
                                  feature_names=("market.trend_1h",), margin_bps=0.)
    with pytest.raises(ValueError, match="before calibration"):
        model.score(groups[calibration[-1].event_id])
    next_obs = observed(ts=10_000_000_000, hourly=0.01, pending=("order",))
    group, = group_teacher_proposals([TeacherProposal(next_obs, "EMA", 1, "cross")])
    result = model.score(group)
    assert result.action == "NO_TRADE" and result.reason == "PENDING_ORDER"
    clean = replace(next_obs, pending_order_ids=())
    group, = group_teacher_proposals([TeacherProposal(clean, "EMA", 1, "cross")])
    assert model.score(group).action == "LONG"


def test_overlap_and_changed_exit_policy_rejected(observed):
    fit, calibration, groups = _samples(observed)
    fit[0] = replace(fit[0], label_end_ts_ns=calibration[0].decision_ts_ns)
    with pytest.raises(ValueError, match="overlap"):
        fit_entry_meta_policy(fit, calibration, groups,
                              feature_names=("market.trend_1h",), margin_bps=0.)
    fit, calibration, groups = _samples(observed)
    calibration[0] = replace(calibration[0], exit_policy_id="X2")
    with pytest.raises(ValueError, match="recipes"):
        fit_entry_meta_policy(fit, calibration, groups,
                              feature_names=("market.trend_1h",), margin_bps=0.)
