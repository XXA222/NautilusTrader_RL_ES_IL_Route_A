from hashlib import sha256
from itertools import product
from dataclasses import replace

import pytest

from rles_route_a.teacher_research.gate_funnel import summarize_frozen_gate_funnel
from rles_route_a.teacher_research.opportunity_models import NetValueEstimate


def _estimate(event_id: str, passed: tuple[bool, bool, bool, bool]) -> NetValueEstimate:
    support, probability, expected, floor_proxy = passed
    return NetValueEstimate(
        event_id=event_id,
        feature_vector_sha256=sha256(f"features-{event_id}".encode()).hexdigest(),
        fold_id="train-inner",
        source_identity="synthetic-source",
        exit_policy_id="fixed-exit-diagnostic",
        execution_recipe="next-open-v1",
        cost_scenario_id="base",
        return_basis_id="entry-notional-log-return-bps-v1",
        rule_version="candidates-v2",
        outcome_method="synthetic-test-v1",
        feature_spec_hash=sha256(b"feature-spec").hexdigest(),
        expected_net_increment_bps=1.0 if expected else 0.0,
        global_residual_floor_proxy_bps=1.0 if floor_proxy else 0.0,
        epistemic_uncertainty_bps=0.0,
        probability_positive=0.6 if probability else 0.5,
        fold_calibration_id="calibration-v1",
        in_support=support,
        economic_status="SCENARIO_ONLY",
        evidence_type="MODEL_COUNTERFACTUAL",
    )


def test_frozen_funnel_reports_all_sixteen_combinations_and_sequential_counts():
    cases = tuple(product((False, True), repeat=4))
    estimates = {
        sha256(f"candidate-{index}".encode()).hexdigest(): _estimate(
            sha256(f"candidate-{index}".encode()).hexdigest(), gates,
        )
        for index, gates in enumerate(cases)
    }
    groups = {
        event_id: (
            f"block-{index // 8}",
            "breakout" if index // 4 % 2 else "pullback",
            1 if index // 2 % 2 else -1,
        )
        for index, event_id in enumerate(estimates)
    }

    result = summarize_frozen_gate_funnel(estimates, groups)

    assert result.candidate_count == 16
    assert result.selected_count == 1
    assert len(result.combinations) == 16
    assert [row.count for row in result.combinations] == [1] * 16
    assert [count for _gate, count in result.sequential_funnel] == [16, 8, 4, 2, 1]
    assert len(result.by_block_family_side) == 8
    assert sum(row.candidate_count for row in result.by_block_family_side) == 16


def test_frozen_funnel_preserves_first_failed_gate_reason():
    event_id = sha256(b"candidate").hexdigest()
    estimate = _estimate(event_id, (False, False, False, False))
    result = summarize_frozen_gate_funnel(
        {event_id: estimate}, {event_id: ("block-1", "breakout", 1)},
    )
    assert result.rejection_reasons == ((event_id, "OUT_OF_FIT_SUPPORT"),)


def test_frozen_funnel_rejects_nonfinite_score_and_missing_group():
    event_id = sha256(b"candidate").hexdigest()
    estimate = _estimate(event_id, (True, True, True, True))
    invalid = replace(estimate, global_residual_floor_proxy_bps=float("nan"))
    group = {event_id: ("block-1", "breakout", 1)}
    with pytest.raises(ValueError, match="nonfinite"):
        summarize_frozen_gate_funnel({event_id: invalid}, group)
    with pytest.raises(ValueError, match="identical event ids"):
        summarize_frozen_gate_funnel({event_id: estimate}, {})


def test_frozen_funnel_preserves_empty_denominator_without_indexing():
    result = summarize_frozen_gate_funnel({}, {})
    assert result.candidate_count == 0
    assert result.selected_ids == ()
    assert [row.count for row in result.combinations] == [0] * 16
    assert [count for _stage, count in result.sequential_funnel] == [0] * 5
