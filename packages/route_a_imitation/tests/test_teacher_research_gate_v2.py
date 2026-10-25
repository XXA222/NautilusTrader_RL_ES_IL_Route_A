from dataclasses import replace
from hashlib import sha256

import pytest

from rles_route_a.evaluation import (
    CandidateBlockResult,
    evaluate_teacher_research_gate_v2,
)


def _rows(*, blocks=4, candidate=10.0, control=-1.0, selected=True):
    rows = []
    for block in range(blocks):
        universe = tuple(
            sha256(f"event-{block}-{suffix}".encode()).hexdigest()
            for suffix in ("a", "b")
        )
        selected_ids = (universe[0],) if selected else ()
        for scenario, offset in (("base", 0.0), ("stress", -1.0)):
            rows.append(CandidateBlockResult(
                preregistration_id="prereg-g1-v1",
                trial_id="frozen-trial-a",
                independent_block_id=f"week-{block}",
                block_start_ts_ns=block * 1_000,
                block_end_ts_ns=block * 1_000 + 999,
                split="train_inner_screen",
                cost_scenario_id=scenario,
                candidate_universe_ids=universe,
                selected_candidate_ids=selected_ids,
                candidate_outcomes_bps=(
                    (universe[0], -2.0 + offset),
                    (universe[1], 4.0 + offset),
                ),
                candidate_rejection_reasons=(
                    (universe[0], None if selected else "ALL_FLAT_NO_TRADE"),
                    (universe[1], "EDGE_BELOW_FROZEN_CUTOFF"),
                ),
                baseline_net_return_bps=0.0,
                candidate_net_return_bps=candidate + offset,
                negative_control_net_return_bps=control + offset,
                baseline_risk_adjusted_excess_bps=1.0,
                candidate_risk_adjusted_excess_bps=3.0 + offset,
                baseline_tail_loss_bps=8.0,
                candidate_tail_loss_bps=6.0 + offset,
                baseline_max_drawdown_bps=15.0,
                candidate_max_drawdown_bps=12.0 + offset,
                baseline_calibration_log_loss=0.70,
                candidate_calibration_log_loss=0.60 + offset / 100.0,
                baseline_turnover_bps=30.0,
                candidate_turnover_bps=35.0 + offset,
                candidate_fees_bps=12.0 if scenario == "base" else 30.0,
                candidate_funding_bps=-0.25,
                candidate_filled_orders=2,
                candidate_exposure_minutes=90.0,
                risk_budget_id="risk-025-v1",
                exit_policy_id="fixed-exit-p0-v1",
                execution_recipe="market-like-v1",
            ))
    return rows


def test_g1_gate_bootstraps_independent_blocks_and_beats_a_positive_control():
    result = evaluate_teacher_research_gate_v2(
        _rows(control=8.0), bootstrap_replicates=1_000, bootstrap_seed=17,
    )

    assert result.status == "G1_TRAIN_INNER_SIGNAL_SCENARIO_ONLY"
    assert result.independent_blocks == 4
    assert result.candidate_events == 8
    assert result.selected_events == 4
    assert result.selection_rate == 0.5
    assert result.increment_ci95_bps[0] > 0
    assert result.negative_control_delta_ci95_bps[0] > 0
    assert result.candidate_vs_negative_control_ci95_bps[0] > 0
    assert result.risk_adjusted_excess_ci95_bps[0] > 0
    assert result.tail_loss_reduction_ci95_bps[0] > 0
    assert result.max_drawdown_reduction_ci95_bps[0] > 0
    assert result.calibration_log_loss_reduction_ci95[0] > 0
    assert result.turnover_change_ci95_bps[0] > 0
    execution = {row[0]: row[1:] for row in result.scenario_execution_diagnostics}
    assert execution["base"] == (12.0, -0.25, 8, 90.0)
    assert execution["stress"] == (30.0, -0.25, 8, 90.0)
    coverage = {row.scenario_id: row for row in result.scenario_candidate_coverage}
    assert coverage["base"].candidate_count == 8
    assert coverage["base"].complete_count == 8
    assert coverage["base"].rejected_positive_count == 4
    assert coverage["base"].rejected_negative_count == 0
    assert coverage["base"].missed_positive_opportunity_bps == 16.0
    assert coverage["base"].mean_selected_outcome_bps == -2.0
    assert coverage["base"].q10_selected_outcome_bps == -2.0
    assert coverage["base"].mean_rejected_outcome_bps == 4.0
    assert coverage["base"].q10_rejected_outcome_bps == 4.0
    assert coverage["base"].rejection_reason_counts == (("EDGE_BELOW_FROZEN_CUTOFF", 4),)
    assert coverage["stress"].missed_positive_opportunity_bps == 12.0
    assert {row[0] for row in result.scenario_increment_ci95_bps} == {"base", "stress"}


def test_g1_metric_sign_conventions_and_cost_decomposition_are_explicit():
    rows = _rows()
    with pytest.raises(ValueError, match="nonnegative magnitude"):
        replace(rows[0], candidate_tail_loss_bps=-1.0)
    with pytest.raises(ValueError, match="nonnegative integer"):
        replace(rows[0], candidate_filled_orders=-1)
    with pytest.raises(ValueError, match="complete candidate denominator"):
        replace(rows[0], candidate_outcomes_bps=(("event-only-one", 1.0),))
    with pytest.raises(ValueError, match="every rejected candidate"):
        replace(rows[0], candidate_rejection_reasons=(
            (rows[0].candidate_universe_ids[0], None),
            (rows[0].candidate_universe_ids[1], None),
        ))


def test_g1_never_counts_scenarios_as_independent_blocks():
    result = evaluate_teacher_research_gate_v2(
        _rows(blocks=3), bootstrap_replicates=1_000,
    )
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.increment_ci95_bps is None


def test_all_flat_and_live_negative_control_signal_do_not_pass():
    all_flat = evaluate_teacher_research_gate_v2(
        _rows(selected=False), bootstrap_replicates=1_000,
    )
    assert all_flat.status == "NO_ELIGIBLE_ALPHA_TEACHER"
    assert "ALL_FLAT_NO_TRADE" in all_flat.reason_codes

    negative_control = evaluate_teacher_research_gate_v2(
        _rows(control=10.0), bootstrap_replicates=1_000,
    )
    assert negative_control.status == "NO_PREDICTABLE_INCREMENT"
    assert "NEGATIVE_CONTROL_NOT_BEATEN" in negative_control.reason_codes


def test_g1_refuses_test_scope_and_unmatched_candidate_denominator():
    rows = _rows()
    with pytest.raises(PermissionError, match="Train-inner"):
        replace(rows[0], split="test")
    with pytest.raises(ValueError, match="every independent block"):
        evaluate_teacher_research_gate_v2(rows[:-1], bootstrap_replicates=1_000)
    mismatched = [*rows]
    selected_event = mismatched[-1].selected_candidate_ids[0]
    mismatched[-1] = replace(
        mismatched[-1], candidate_universe_ids=(
            sha256(b"different-event").hexdigest(), selected_event,
        ),
        candidate_outcomes_bps=(
            (sha256(b"different-event").hexdigest(), -2.0), (selected_event, 3.0),
        ),
        candidate_rejection_reasons=(
            (sha256(b"different-event").hexdigest(), "EDGE_BELOW_FROZEN_CUTOFF"),
            (selected_event, None),
        ),
    )
    with pytest.raises(ValueError, match="same candidate universe"):
        evaluate_teacher_research_gate_v2(mismatched, bootstrap_replicates=1_000)


def test_g1_requires_nonoverlapping_time_blocks_and_matching_scenario_boundaries():
    rows = _rows()
    overlapping = [*rows]
    overlapping[2] = replace(
        overlapping[2], block_start_ts_ns=500, block_end_ts_ns=1_500,
    )
    overlapping[3] = replace(
        overlapping[3], block_start_ts_ns=500, block_end_ts_ns=1_500,
    )
    with pytest.raises(ValueError, match="must not overlap"):
        evaluate_teacher_research_gate_v2(overlapping, bootstrap_replicates=1_000)

    mismatched = [*rows]
    mismatched[1] = replace(mismatched[1], block_end_ts_ns=900)
    with pytest.raises(ValueError, match="identical time boundaries"):
        evaluate_teacher_research_gate_v2(mismatched, bootstrap_replicates=1_000)
