from hashlib import sha256

import pytest

from rles_route_a.teacher_research.candidate_events import OpportunityEvent
from rles_route_a.teacher_research.opportunity_diagnostics import (
    DiagnosticAblationSet,
    audit_screen_ablation_outcomes_after_prediction,
    audit_screen_outcomes_after_prediction,
    build_diagnostic_ablation_sets,
    build_opportunity_prediction_report,
)
from rles_route_a.teacher_research.opportunity_models import NetValueEstimate, OpportunityOutcome


def _event(number: int) -> OpportunityEvent:
    digest = sha256(f"event-{number}".encode()).hexdigest()
    return OpportunityEvent(
        event_id=digest,
        decision_ts_ns=100 + number,
        feature_available_at_ns=100 + number,
        side=1,
        family="breakout",
        fold_id="train-inner",
        source_identity="synthetic-source",
        rule_version="route-a-v1",
        snapshot_id=sha256(b"snapshot").hexdigest(),
        feature_spec_hash=sha256(b"feature-spec").hexdigest(),
    )


def _estimate(event: OpportunityEvent, *, probability: float = 0.8,
              expected: float = 3.0, residual: float = -2.0,
              support: bool = True) -> NetValueEstimate:
    return NetValueEstimate(
        event_id=event.event_id,
        feature_vector_sha256=sha256(f"features-{event.event_id}".encode()).hexdigest(),
        fold_id=event.fold_id,
        source_identity=event.source_identity,
        exit_policy_id="dynamic-test",
        execution_recipe="next-open-v1",
        cost_scenario_id="base",
        return_basis_id="LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS",
        rule_version=event.rule_version,
        outcome_method="synthetic-test-v1",
        feature_spec_hash=event.feature_spec_hash,
        expected_net_increment_bps=expected,
        global_residual_floor_proxy_bps=expected + residual,
        epistemic_uncertainty_bps=1.5,
        probability_positive=probability,
        fold_calibration_id="calibration-v1",
        in_support=support,
        economic_status="SCENARIO_ONLY",
        evidence_type="MODEL_COUNTERFACTUAL",
    )


def _outcome(event: OpportunityEvent, *, net: float | None, censor: int | None = None,
             cost: str = "base"):
    return OpportunityOutcome(
        event_id=event.event_id,
        candidate_family=event.family,
        candidate_side=event.side,
        decision_ts_ns=event.decision_ts_ns,
        feature_available_at_ns=event.feature_available_at_ns,
        rule_version=event.rule_version,
        snapshot_id=event.snapshot_id,
        feature_spec_hash=event.feature_spec_hash,
        feature_vector_sha256=sha256(f"features-{event.event_id}".encode()).hexdigest(),
        label_end_ts_ns=(event.decision_ts_ns + 10 if net is not None else None),
        net_return_bps=net,
        fold_id=event.fold_id,
        source_identity=event.source_identity,
        split="train_inner_screen",
        independent_block_id="screen-block-1",
        independent_block_start_ts_ns=100,
        independent_block_end_ts_ns=1000,
        exit_policy_id="dynamic-test",
        execution_recipe="next-open-v1",
        cost_scenario_id=cost,
        return_basis_id="LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS",
        simulation_receipt_id=sha256(f"receipt-{event.event_id}".encode()).hexdigest(),
        outcome_method="synthetic-test-v1",
        censor_ts_ns=censor,
    )


def test_prediction_report_preserves_missing_denominator_and_frozen_gate_reasons():
    events = (_event(1), _event(2), _event(3))
    estimates = {
        events[0].event_id: _estimate(events[0]),
        events[1].event_id: _estimate(events[1], probability=0.5, expected=-1.0),
    }
    report = build_opportunity_prediction_report(
        events, estimates,
        {event.event_id: "screen-1" for event in events},
        {event.event_id: "train_inner_screen" for event in events},
    )
    assert report.candidate_count == 3
    assert report.prediction_count == 2
    assert report.missing_prediction_ids == (events[2].event_id,)
    rows_by_id = {row.event_id: row for row in report.rows}
    assert rows_by_id[events[0].event_id].first_rejection_code == "SELECTED"
    assert rows_by_id[events[1].event_id].first_rejection_code == "CALIBRATED_PROBABILITY_NOT_ABOVE_0_5"
    assert rows_by_id[events[2].event_id].first_rejection_code == "PREDICTION_MISSING"
    assert rows_by_id[events[2].event_id].expected_net_increment_bps is None
    assert len(report.score_distributions) == 4


def test_prediction_report_audits_fit_extrema_and_quantiles_by_group():
    event = _event(1)
    report = build_opportunity_prediction_report(
        (event,), {event.event_id: _estimate(event)},
        {event.event_id: "screen-1"}, {event.event_id: "train_inner_screen"},
        feature_rows={event.event_id: {"feature-x": 0.5}},
        fit_extrema={"feature-x": (0.0, 1.0)},
    )
    assert report.support_features[0].out_of_support_count == 0
    distribution = report.score_distributions[0]
    assert distribution.count == 1
    assert distribution.minimum == distribution.maximum

    with pytest.raises(ValueError, match="in-support flag differs"):
        build_opportunity_prediction_report(
            (event,), {event.event_id: _estimate(event, support=False)},
            {event.event_id: "screen-1"}, {event.event_id: "train_inner_screen"},
            feature_rows={event.event_id: {"feature-x": 0.5}},
            fit_extrema={"feature-x": (0.0, 1.0)},
        )


def test_outcome_audit_is_separate_and_preserves_complete_censor_and_calibration():
    events = (_event(1), _event(2), _event(3))
    estimates = {
        events[0].event_id: _estimate(events[0]),
        events[1].event_id: _estimate(events[1], probability=0.4, expected=-1.0),
    }
    report = build_opportunity_prediction_report(
        events, estimates,
        {event.event_id: "screen-1" for event in events},
        {event.event_id: "train_inner_screen" for event in events},
    )
    audit = audit_screen_outcomes_after_prediction(
        report,
        {
            events[0].event_id: _outcome(events[0], net=2.0),
            events[1].event_id: _outcome(events[1], net=-4.0),
            events[2].event_id: _outcome(events[2], net=None,
                                         censor=events[2].decision_ts_ns + 5),
        },
        calibration_residual_p10_offset_bps=-2.0,
    )
    assert (audit.complete_count, audit.censored_count, audit.missing_outcome_count) == (2, 1, 0)
    assert audit.selected_complete_net_bps == (2.0,)
    assert audit.rejected_complete_net_bps == (-4.0,)
    assert audit.brier_score == pytest.approx(0.1)
    assert len(audit.calibration_bins) == 10
    assert sum(row.count for row in audit.calibration_bins) == 2
    assert audit.calibration_bins[4].count == 1
    assert audit.calibration_bins[8].count == 1
    assert audit.global_residual_floor_proxy_identity_max_abs_error_bps == 0.0
    assert audit.actual_below_global_residual_floor_proxy_rate == pytest.approx(0.5)


def test_frozen_f0_to_f5_ablation_sets_are_deterministic_and_diagnostic_only():
    events = (_event(1), _event(2), _event(3))
    estimates = {
        events[0].event_id: _estimate(events[0]),
        events[1].event_id: _estimate(events[1], probability=0.4, expected=-1.0),
    }
    report = build_opportunity_prediction_report(
        events, estimates,
        {event.event_id: "screen-1" for event in events},
        {event.event_id: "train_inner_screen" for event in events},
    )
    first = build_diagnostic_ablation_sets(report, negative_control_count=2, frozen_seed=7)
    second = build_diagnostic_ablation_sets(report, negative_control_count=2, frozen_seed=7)
    assert tuple(row.ablation_id for row in first) == ("F0", "F1", "F2", "F3", "F4", "F5")
    assert first == second
    assert first[0].selected_ids == (events[0].event_id,)
    assert first[1].selected_ids == (events[0].event_id,)
    assert first[4].selected_ids == tuple(sorted(event.event_id for event in events))
    assert len(first[5].selected_ids) == 2
    assert {row.evidence_status for row in first} == {"DIAGNOSTIC_REUSE_NOT_OOS"}


def test_ablation_outcome_audit_keeps_paired_costs_and_censors_post_selection():
    events = (_event(1), _event(2), _event(3))
    report = build_opportunity_prediction_report(
        events, {event.event_id: _estimate(event) for event in events},
        {event.event_id: "screen-1" for event in events},
        {event.event_id: "train_inner_screen" for event in events},
    )
    ablations = (
        DiagnosticAblationSet("F0", (events[0].event_id,)),
        DiagnosticAblationSet("F1", (events[0].event_id, events[1].event_id)),
        DiagnosticAblationSet("F2", (events[1].event_id,)),
        DiagnosticAblationSet("F3", ()),
        DiagnosticAblationSet("F4", tuple(sorted(event.event_id for event in events))),
        DiagnosticAblationSet("F5", (events[2].event_id,)),
    )
    stress_outcomes = {
        events[0].event_id: _outcome(events[0], net=2.0, cost="stress"),
        events[1].event_id: _outcome(events[1], net=-4.0, cost="stress"),
        events[2].event_id: _outcome(events[2], net=None,
                                     censor=events[2].decision_ts_ns + 5, cost="stress"),
    }

    with pytest.raises(ValueError, match="paired cost scenario must be explicitly identified"):
        audit_screen_ablation_outcomes_after_prediction(report, stress_outcomes, ablations)
    audits = audit_screen_ablation_outcomes_after_prediction(
        report, stress_outcomes, ablations, expected_outcome_cost_scenario_id="stress",
    )

    assert [audit.ablation_id for audit in audits] == ["F0", "F1", "F2", "F3", "F4", "F5"]
    assert all(audit.evidence_status == "DIAGNOSTIC_REUSE_NOT_OOS" for audit in audits)
    assert all(audit.prediction_cost_scenario_id == "base" for audit in audits)
    assert all(audit.outcome_cost_scenario_id == "stress" for audit in audits)
    complete_by_id = {events[0].event_id: 2.0, events[1].event_id: -4.0}
    selected_f1 = tuple(complete_by_id[event_id] for event_id in sorted(complete_by_id))
    rejected_f2 = (complete_by_id[events[0].event_id],)
    assert audits[1].selected_complete_net_bps == selected_f1
    assert audits[1].selected_censored_count == 0
    assert audits[2].rejected_complete_net_bps == rejected_f2
    assert audits[3].rejected_complete_count == 2
    assert audits[4].selected_censored_count == 1
    assert audits[5].selected_censored_count == 1
