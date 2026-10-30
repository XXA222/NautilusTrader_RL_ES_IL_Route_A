from dataclasses import replace
from hashlib import sha256
from math import log

import pytest

from rles_route_a.teacher_research import (
    CounterfactualTradeReplay,
    OpportunityEvent,
    fit_opportunity_model,
    label_counterfactual_candidates,
)


FEATURE_SPEC = sha256(b"causal-feature-spec").hexdigest()


def test_return_basis_id_must_match_the_formula_used_for_net_labels():
    with pytest.raises(ValueError, match="return basis ID must match"):
        label_counterfactual_candidates(
            (), lambda *_: None, {}, {}, {}, feature_names=("x",),
            split="train_fit", exit_policy_id="fixed-exit-v1",
            execution_recipe="next-open-v1", cost_scenario_id="base",
            return_basis_id="LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS",
        )


def _event(index, *, side=1, family="breakout"):
    event_id = sha256(f"candidate-{index}".encode()).hexdigest()
    return OpportunityEvent(
        event_id=event_id,
        decision_ts_ns=1_000_000 + index * 60,
        feature_available_at_ns=1_000_000 + index * 60 - 1,
        side=side,
        family=family,
        fold_id="fold-train-a",
        source_identity="registered-train-source",
        rule_version="candidate-rule-v2",
        snapshot_id=sha256(f"snapshot-{index}".encode()).hexdigest(),
        feature_spec_hash=FEATURE_SPEC,
    )


def test_fixed_exit_labeler_preserves_winners_losers_and_censored_candidates():
    events = (_event(1), _event(2, side=-1, family="pullback"), _event(3))
    features = {event.event_id: (float(i),) for i, event in enumerate(events)}
    block_ids = {event.event_id: "train-week-1" for event in events}
    interval = (events[0].decision_ts_ns, events[1].decision_ts_ns + 121)
    block_intervals = {event.event_id: interval for event in events}
    replay_rows = {
        events[0].event_id: CounterfactualTradeReplay(
            event_id=events[0].event_id,
            entry_ts_ns=events[0].decision_ts_ns + 1,
            entry_price=100.0,
            exit_ts_ns=events[0].decision_ts_ns + 60,
            exit_price=101.0,
            entry_fee_bps=5,
            exit_fee_bps=5,
            entry_slippage_bps=2,
            exit_slippage_bps=2,
            funding_cashflow_bps=1,
            censor_ts_ns=None,
            simulation_receipt_id="sim-long",
        ),
        events[1].event_id: CounterfactualTradeReplay(
            event_id=events[1].event_id,
            entry_ts_ns=events[1].decision_ts_ns + 1,
            entry_price=100.0,
            exit_ts_ns=events[1].decision_ts_ns + 120,
            exit_price=99.0,
            entry_fee_bps=55,
            exit_fee_bps=55,
            entry_slippage_bps=2,
            exit_slippage_bps=2,
            funding_cashflow_bps=-5,
            censor_ts_ns=None,
            simulation_receipt_id="sim-short",
        ),
        events[2].event_id: CounterfactualTradeReplay(
            event_id=events[2].event_id,
            entry_ts_ns=events[2].decision_ts_ns + 1,
            entry_price=100.0,
            exit_ts_ns=None,
            exit_price=None,
            entry_fee_bps=5,
            exit_fee_bps=0,
            entry_slippage_bps=2,
            exit_slippage_bps=0,
            funding_cashflow_bps=-3,
            censor_ts_ns=events[2].decision_ts_ns + 600,
            simulation_receipt_id="sim-open-prefix",
        ),
    }
    callback_calls = []

    def replay(event, exit_policy_id, execution_recipe, scenario_id):
        callback_calls.append((event.event_id, exit_policy_id, execution_recipe, scenario_id))
        return replay_rows[event.event_id]

    batch = label_counterfactual_candidates(
        events,
        replay,
        features,
        block_ids,
        block_intervals,
        feature_names=("causal.edge",),
        split="train_fit",
        exit_policy_id="fixed-trend-exit-v1",
        execution_recipe="market-like-v1",
        cost_scenario_id="stress-30bp",
        return_basis_id="LOG_REFERENCE_ENTRY_NOTIONAL_BPS",
    )

    long_label, short_label, censored_label = batch.outcomes
    assert len(callback_calls) == len(events)
    assert all(call[1:] == (
        "fixed-trend-exit-v1", "market-like-v1", "stress-30bp",
    ) for call in callback_calls)
    assert long_label.net_return_bps == pytest.approx(log(101 / 100) * 10_000 - 13)
    assert short_label.net_return_bps == pytest.approx(-log(99 / 100) * 10_000 - 119)
    assert long_label.gross_return_bps == pytest.approx(log(101 / 100) * 10_000)
    assert (long_label.fees_bps, long_label.slippage_bps,
            long_label.funding_cashflow_bps) == (10, 4, 1)
    assert (short_label.fees_bps, short_label.slippage_bps,
            short_label.funding_cashflow_bps) == (110, 4, -5)
    assert long_label.net_return_bps > 0
    assert short_label.net_return_bps < 0
    assert censored_label.net_return_bps is None
    assert censored_label.label_end_ts_ns is None
    assert censored_label.censor_ts_ns == interval[1]
    assert all(value is None for value in (
        censored_label.gross_return_bps, censored_label.fees_bps,
        censored_label.slippage_bps, censored_label.funding_cashflow_bps,
    ))
    assert batch.complete_count == 2
    assert batch.censored_count == 1
    assert {row.event_id for row in batch.outcomes} == {event.event_id for event in events}
    assert all(row.evidence_type == "MODEL_COUNTERFACTUAL" for row in batch.outcomes)
    assert all(row.economic_status == "SCENARIO_ONLY" for row in batch.outcomes)
    assert set(batch.event_map()) == set(batch.feature_map()) == {
        event.event_id for event in events
    }


def test_labeler_fails_closed_on_missing_denominator_or_wrong_replay_event():
    events = (_event(10), _event(11))
    features = {event.event_id: (0.1,) for event in events}
    blocks = {event.event_id: "week-a" for event in events}
    intervals = {event.event_id: (events[0].decision_ts_ns,
                                  events[-1].decision_ts_ns + 1)
                 for event in events}
    output = CounterfactualTradeReplay(
        event_id=events[0].event_id,
        entry_ts_ns=events[0].decision_ts_ns + 1,
        entry_price=100,
        exit_ts_ns=events[0].decision_ts_ns + 60,
        exit_price=100.1,
        entry_fee_bps=5,
        exit_fee_bps=5,
        entry_slippage_bps=1,
        exit_slippage_bps=1,
        funding_cashflow_bps=0,
        censor_ts_ns=None,
        simulation_receipt_id="sim-one",
    )
    with pytest.raises(ValueError, match="cover every candidate"):
        label_counterfactual_candidates(
            events, lambda *_: output, {events[0].event_id: (0.1,)}, blocks,
            intervals,
            feature_names=("x",), split="train_fit", exit_policy_id="exit-v1",
            execution_recipe="market-like-v1", cost_scenario_id="base-12bp",
            return_basis_id="LOG_REFERENCE_ENTRY_NOTIONAL_BPS",
        )
    with pytest.raises(ValueError, match="does not match"):
        label_counterfactual_candidates(
            events, lambda *_: output, features, blocks, intervals,
            feature_names=("x",), split="train_fit", exit_policy_id="exit-v1",
            execution_recipe="market-like-v1", cost_scenario_id="base-12bp",
            return_basis_id="LOG_REFERENCE_ENTRY_NOTIONAL_BPS",
        )


def test_outcome_crossing_independent_block_end_is_censored_at_boundary():
    event = _event(40)
    interval = (event.decision_ts_ns, event.decision_ts_ns + 50)
    replay = CounterfactualTradeReplay(
        event_id=event.event_id,
        entry_ts_ns=event.decision_ts_ns + 1,
        entry_price=100.0,
        exit_ts_ns=event.decision_ts_ns + 60,
        exit_price=101.0,
        entry_fee_bps=5.0,
        exit_fee_bps=5.0,
        entry_slippage_bps=1.0,
        exit_slippage_bps=1.0,
        funding_cashflow_bps=0.0,
        censor_ts_ns=None,
        simulation_receipt_id="crosses-independent-block",
    )
    batch = label_counterfactual_candidates(
        (event,), lambda *_: replay, {event.event_id: (0.5,)},
        {event.event_id: "short-block"}, {event.event_id: interval},
        feature_names=("edge",), split="train_fit",
        exit_policy_id="fixed-exit-v1", execution_recipe="market-like-v1",
        cost_scenario_id="base-12bp", return_basis_id="LOG_REFERENCE_ENTRY_NOTIONAL_BPS",
    )

    assert batch.outcomes[0].net_return_bps is None
    assert batch.outcomes[0].label_end_ts_ns is None
    assert batch.outcomes[0].censor_ts_ns == interval[1]


def test_complete_outcome_rejects_mismatched_economic_decomposition():
    event = _event(41)
    replay = CounterfactualTradeReplay(
        event_id=event.event_id,
        entry_ts_ns=event.decision_ts_ns + 1,
        entry_price=100.0,
        exit_ts_ns=event.decision_ts_ns + 10,
        exit_price=101.0,
        entry_fee_bps=2.0,
        exit_fee_bps=3.0,
        entry_slippage_bps=1.0,
        exit_slippage_bps=1.5,
        funding_cashflow_bps=-0.5,
        censor_ts_ns=None,
        simulation_receipt_id="decomposition-check",
    )
    batch = label_counterfactual_candidates(
        (event,), lambda *_: replay, {event.event_id: (0.5,)},
        {event.event_id: "block"},
        {event.event_id: (event.decision_ts_ns, event.decision_ts_ns + 20)},
        feature_names=("edge",), split="train_fit",
        exit_policy_id="fixed-exit-v1", execution_recipe="market-like-v1",
        cost_scenario_id="base", return_basis_id="LOG_REFERENCE_ENTRY_NOTIONAL_BPS",
    )
    with pytest.raises(ValueError, match="do not reconcile"):
        replace(batch.outcomes[0], fees_bps=batch.outcomes[0].fees_bps + 1.0)


def test_full_denominator_label_to_later_calibrated_model_chain():
    def make_batch(split, *, first_decision_ns, id_prefix):
        events = tuple(
            OpportunityEvent(
                event_id=sha256(f"{id_prefix}-{index}".encode()).hexdigest(),
                decision_ts_ns=first_decision_ns + index * 60_000_000_000,
                feature_available_at_ns=first_decision_ns + index * 60_000_000_000 - 1,
                side=1 if index % 2 else -1,
                family="breakout" if index % 3 else "pullback",
                fold_id="fold-train-a",
                source_identity="registered-train-source",
                rule_version="candidate-rule-v2",
                snapshot_id=sha256(f"{id_prefix}-snapshot-{index}".encode()).hexdigest(),
                feature_spec_hash=FEATURE_SPEC,
            )
            for index in range(12)
        )
        features = {event.event_id: (float(index % 3),) for index, event in enumerate(events)}
        blocks = {event.event_id: f"{id_prefix}-block-{index // 4}"
                  for index, event in enumerate(events)}
        grouped_events = {}
        for event in events:
            grouped_events.setdefault(blocks[event.event_id], []).append(event)
        intervals = {
            event.event_id: (
                grouped_events[blocks[event.event_id]][0].decision_ts_ns,
                grouped_events[blocks[event.event_id]][-1].decision_ts_ns + 101,
            )
            for event in events
        }

        def replay(event, _exit, _execution, _scenario):
            return CounterfactualTradeReplay(
                event_id=event.event_id,
                entry_ts_ns=event.decision_ts_ns + 1,
                entry_price=100.0,
                exit_ts_ns=event.decision_ts_ns + 100,
                exit_price=100.5,
                entry_fee_bps=5.0,
                exit_fee_bps=5.0,
                entry_slippage_bps=1.0,
                exit_slippage_bps=1.0,
                funding_cashflow_bps=0.0,
                censor_ts_ns=None,
                simulation_receipt_id=f"{id_prefix}-receipt-{event.event_id}",
            )

        return label_counterfactual_candidates(
            events, replay, features, blocks, intervals,
            feature_names=("activity",), split=split,
            exit_policy_id="fixed-exit-v1",
            execution_recipe="market-like-v1",
            cost_scenario_id="base-12bp",
            return_basis_id="LOG_REFERENCE_ENTRY_NOTIONAL_BPS",
        )

    fit_batch = make_batch(
        "train_fit", first_decision_ns=1_000_000_000_000, id_prefix="fit",
    )
    calibration_batch = make_batch(
        "train_calibration", first_decision_ns=100_000_000_000_000,
        id_prefix="calibration",
    )
    model = fit_opportunity_model(
        fit_batch.outcomes,
        calibration_batch.outcomes,
        {**fit_batch.feature_map(), **calibration_batch.feature_map()},
        candidate_events={**fit_batch.event_map(), **calibration_batch.event_map()},
        feature_names=("activity",),
    )
    long_event = OpportunityEvent(
        event_id=sha256(b"post-calibration-long").hexdigest(),
        decision_ts_ns=200_000_000_000_000,
        feature_available_at_ns=199_999_999_999_999,
        side=1,
        family="breakout",
        fold_id="fold-train-a",
        source_identity="registered-train-source",
        rule_version="candidate-rule-v2",
        snapshot_id=sha256(b"post-calibration-snapshot").hexdigest(),
        feature_spec_hash=FEATURE_SPEC,
    )
    prediction = model.predict(long_event, (1.0,))

    assert fit_batch.complete_count == calibration_batch.complete_count == 12
    assert model.fit_sample_count == model.calibration_sample_count == 12
    assert prediction.probability_positive > 0.5
    assert prediction.expected_net_increment_bps > 0
    assert prediction.cost_scenario_id == "base-12bp"
    assert prediction.economic_status == "SCENARIO_ONLY"
