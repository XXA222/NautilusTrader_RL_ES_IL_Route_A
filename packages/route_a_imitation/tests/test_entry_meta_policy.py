from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from rles_route_a.teacher_research.candidate_event_union import TeacherProposal, group_teacher_proposals
from rles_route_a.teacher_research.conditional_outcome_labels import GroupOutcome
from rles_route_a.teacher_research.entry_meta_policy import (EntryScore, fit_entry_meta_policy,
                                                              overlap_uniqueness_weights,
                                                              summarize_entry_scores)
from rles_route_a.teacher_research.opportunity_models import _fit_logistic, _fit_platt
from rles_route_a.teacher_research.opportunity_models import uniqueness_effective_label_count


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


def test_gate_funnel_uses_all_candidates_and_assigns_first_failed_stage():
    scores = [
        EntryScore("LONG", .8, 10., True, "POSITIVE_NET_EDGE"),
        EntryScore("NO_TRADE", .8, 11., False, "OUT_OF_SUPPORT", ("market.ema_9",)),
        EntryScore("NO_TRADE", .8, 12., True, "PENDING_ORDER"),
    ]
    result = summarize_entry_scores(scores, margin_bps=5., unscored_count=2,
                                    unscored_reason_counts={"REJECTED_FEATURE_NOT_READY": 1,
                                                            "REJECTED_SCORING_ERROR": 1})

    assert result["candidate_count"] == 5
    assert result["scored_count"] == 3 and result["unscored_count"] == 2
    assert result["accepted_count"] == 1 and result["coverage"] == pytest.approx(.2)
    funnel = result["sequential_gate_funnel"]
    assert funnel["stage_counts"] == {
        "total_candidates": 5, "scored_candidates": 3, "account_eligible": 2,
        "in_support": 1, "above_frozen_net_edge_margin": 1, "model_gate_accepted": 1,
        "above_margin_after_calibration_error": 1,
        "native_eligible_after_overrides": 1,
    }
    assert funnel["rejected_at_stage"] == {
        "unscored": 2, "account": 1, "support": 1, "net_edge": 0, "calibration_error": 0,
        "post_gate_native_constraints": 0,
        "native_arbitration_or_owner_readiness": None,
    }
    assert result["unscored_reason_counts"] == {
        "REJECTED_FEATURE_NOT_READY": 1, "REJECTED_SCORING_ERROR": 1,
    }


def test_calibration_error_gate_is_visible_in_coverage_diagnostics():
    score = EntryScore(
        "NO_TRADE", .8, 10., True, "CALIBRATION_ERROR_EXCEEDS_EDGE",
        calibration_mae_bps=6.,
    )
    result = summarize_entry_scores([score], margin_bps=5.)
    assert result["accepted_count"] == 0
    assert result["reason_counts"] == {"CALIBRATION_ERROR_EXCEEDS_EDGE": 1}
    assert result["expected_net_after_calibration_mae_bps"] == {
        "min": 4., "p25": 4., "median": 4., "p75": 4., "max": 4.,
    }
    assert result["sequential_gate_funnel"]["rejected_at_stage"]["calibration_error"] == 1


def test_model_acceptance_is_separate_from_single_direction_native_selection():
    scores = [EntryScore("LONG", .8, 8., True, "POSITIVE_NET_EDGE"),
              EntryScore("SHORT", .7, 7., True, "POSITIVE_NET_EDGE")]

    result = summarize_entry_scores(scores, margin_bps=5., native_selected_count=1)

    assert result["model_gate_accepted_count"] == 2
    assert result["native_selected_count"] == 1
    assert result["coverage"] == 1.
    assert result["native_selected_coverage"] == .5
    funnel = result["sequential_gate_funnel"]
    assert funnel["stage_counts"]["model_gate_accepted"] == 2
    assert funnel["stage_counts"]["native_eligible_after_overrides"] == 2
    assert funnel["stage_counts"]["native_selected_after_arbitration"] == 1
    assert funnel["rejected_at_stage"]["post_gate_native_constraints"] == 0
    assert funnel["rejected_at_stage"]["native_arbitration_or_owner_readiness"] == 1


def test_post_gate_native_override_is_separate_from_model_rejection():
    scores = [EntryScore("LONG", .8, 8., True, "POSITIVE_NET_EDGE"),
              EntryScore("SHORT", .7, 7., True, "POSITIVE_NET_EDGE")]
    result = summarize_entry_scores(scores, margin_bps=5., native_selected_count=1,
                                    post_gate_override_counts={"HARD_RISK_OVERRIDE": 1})
    funnel = result["sequential_gate_funnel"]
    assert result["model_gate_accepted_count"] == 2
    assert result["post_gate_override_reason_counts"] == {"HARD_RISK_OVERRIDE": 1}
    assert funnel["stage_counts"]["native_eligible_after_overrides"] == 1
    assert funnel["rejected_at_stage"]["post_gate_native_constraints"] == 1
    assert funnel["rejected_at_stage"]["native_arbitration_or_owner_readiness"] == 0
    with pytest.raises(ValueError, match="post-override"):
        summarize_entry_scores(scores, margin_bps=5., native_selected_count=2,
                               post_gate_override_counts={"HARD_RISK_OVERRIDE": 1})


def test_unscored_reason_counts_must_match_denominator():
    with pytest.raises(ValueError, match="cover the unscored"):
        summarize_entry_scores([], margin_bps=5., unscored_count=2,
                               unscored_reason_counts={"REJECTED_FEATURE_NOT_READY": 1})


def _samples(observed):
    groups = {}
    labels = []
    for i in range(8):
        obs = observed(ts=(i + 1) * 1000000000, hourly=0.01 if i % 2 else -0.01)
        group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
        groups[group.event_id] = group
        labels.append(GroupOutcome(group.event_id, obs.decision_ts_ns, obs.source_id,
                                   obs.fold_id, "X1", "native", "base",
                                   obs.decision_ts_ns + 1, None, 10. if i % 2 else -10.,
                                   simulation_receipt_id=f"fit-replay-{i}"))
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


def test_entry_requires_edge_to_cover_calibration_prediction_error(observed):
    fit, calibration, groups = _samples(observed)
    model = fit_entry_meta_policy(
        fit, calibration, groups, feature_names=("market.trend_1h",), margin_bps=0.1,
    )
    candidate = observed(ts=10_000_000_000, hourly=0.01, pending=())
    group, = group_teacher_proposals([TeacherProposal(candidate, "EMA", 1, "cross")])
    original = model.score(group)
    # Keep the same calibrated prediction while injecting only the empirically
    # measured calibration error into a frozen-policy copy.
    cautious = replace(model, calibration_mae_bps=max(original.expected_net_bps, 0.1))
    rejected = cautious.score(group)
    assert rejected.action == "NO_TRADE"
    assert rejected.reason == "CALIBRATION_ERROR_EXCEEDS_EDGE"
    assert rejected.calibration_mae_bps >= rejected.expected_net_bps


def test_fitted_meta_policy_does_not_double_weight_duplicate_future_interval(observed):
    fit, calibration, groups = _samples(observed)
    reference = fit_entry_meta_policy(fit, calibration, groups,
        feature_names=("market.trend_1h",), margin_bps=0.)
    original = groups[fit[0].event_id]
    duplicate_observation = replace(original.observation,
                                    snapshot_id=sha256(b"same-minute-duplicate").hexdigest())
    duplicate, = group_teacher_proposals([TeacherProposal(
        duplicate_observation, original.teachers[0][0], original.side,
        original.teachers[0][1])])
    duplicate_label = replace(fit[0], event_id=duplicate.event_id)
    groups[duplicate.event_id] = duplicate
    assert overlap_uniqueness_weights((*fit, duplicate_label)) == pytest.approx(
        [.5, 1., 1., 1., .5])
    assert uniqueness_effective_label_count(
        overlap_uniqueness_weights((*fit, duplicate_label))) == pytest.approx(4.)
    weighted = fit_entry_meta_policy((*fit, duplicate_label), calibration, groups,
        feature_names=("market.trend_1h",), margin_bps=0.)
    assert weighted.logistic == pytest.approx(reference.logistic)
    assert weighted.value_weights == pytest.approx(reference.value_weights)
    assert weighted.calibrator == pytest.approx(reference.calibrator)


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


def test_fit_rejects_candidates_blocked_by_positions_or_pending_orders(observed):
    from dataclasses import replace

    fit, calibration, groups = _samples(observed)
    original = groups[fit[0].event_id]
    blocked_observation = replace(original.observation, pending_order_ids=("native-order",))
    blocked_group, = group_teacher_proposals([TeacherProposal(
        blocked_observation, "EMA", 1, "cross")])
    groups.pop(original.event_id)
    groups[blocked_group.event_id] = blocked_group
    fit[0] = replace(fit[0], event_id=blocked_group.event_id)

    with pytest.raises(ValueError, match="only executable flat candidates"):
        fit_entry_meta_policy(fit, calibration, groups,
                              feature_names=("market.trend_1h",), margin_bps=0.)


def test_overlap_uniqueness_weights_share_only_the_concurrent_label_time():
    labels = [
        GroupOutcome(sha256(b"a").hexdigest(), 1, "source", "fold", "X1", "native", "base", 11, None, 1., "replay-a"),
        GroupOutcome(sha256(b"b").hexdigest(), 6, "source", "fold", "X1", "native", "base", 16, None, 1., "replay-b"),
        GroupOutcome(sha256(b"c").hexdigest(), 21, "source", "fold", "X1", "native", "base", 31, None, 1., "replay-c"),
    ]
    assert overlap_uniqueness_weights(labels) == pytest.approx([.75, .75, 1.])
    fully_overlapped = [replace(labels[0], event_id=sha256(b"d").hexdigest(), label_end_ts_ns=10),
                        replace(labels[0], event_id=sha256(b"e").hexdigest(), label_end_ts_ns=10)]
    assert overlap_uniqueness_weights(fully_overlapped) == pytest.approx([.5, .5])
    assert uniqueness_effective_label_count(
        overlap_uniqueness_weights(fully_overlapped)) == pytest.approx(1.)


def test_overlap_weights_reject_censored_or_duplicate_labels():
    label = GroupOutcome(sha256(b"a").hexdigest(), 1, "source", "fold", "X1", "native", "base", 2, None, 1., "replay")
    with pytest.raises(ValueError, match="complete future labels"):
        overlap_uniqueness_weights([replace(label, label_end_ts_ns=None, censor_ts_ns=3,
                                            net_return_bps=None, outcome_status="RIGHT_CENSORED")])
    with pytest.raises(ValueError, match="complete future labels"):
        overlap_uniqueness_weights([label, label])


def test_uniqueness_weights_prevent_duplicate_overlap_from_inflating_model_fit():
    design = np.asarray([[1., -1.], [1., 0.], [1., 1.], [1., 2.]])
    target = np.asarray([0., 0., 1., 1.])
    duplicated = np.asarray([design[0], design[0], *design[1:]])
    duplicated_target = np.asarray([target[0], target[0], *target[1:]])
    weights = np.asarray([.5, .5, 1., 1., 1.])
    assert _fit_logistic(design, target, .2, .1, 400) == pytest.approx(
        _fit_logistic(duplicated, duplicated_target, .2, .1, 400,
                      sample_weights=weights))
    logits = design[:, 1] * 2
    duplicated_logits = np.asarray([logits[0], logits[0], *logits[1:]])
    assert _fit_platt(logits, target, .01, 400) == pytest.approx(
        _fit_platt(duplicated_logits, duplicated_target, .01, 400,
                   sample_weights=weights))
