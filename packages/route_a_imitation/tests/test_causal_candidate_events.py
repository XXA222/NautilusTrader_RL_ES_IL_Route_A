from dataclasses import replace

import pytest

from rles_route_a.contracts import ExpertAdvice, NativeActionTrace
from rles_route_a.teacher_research import CausalOpportunityGenerator
from rles_route_a.teacher_research import audit_candidate_decision


def test_candidate_generator_records_causal_provenance_and_is_deterministic(observed):
    generator = CausalOpportunityGenerator()
    obs = observed(hourly=.01)
    # Independently construct a valid flat observation with an actual breakout.
    obs = replace(obs, market={**obs.market, "breakout_1h": .02})
    event, = generator.candidates(obs)
    assert event.family == "breakout"
    assert event.side == 1
    assert event.feature_available_at_ns <= event.decision_ts_ns
    assert event.feature_available_at_ns == max(
        obs.feature_available_at_ns["breakout_1h"],
        obs.feature_available_at_ns["trend_1d"],
    )
    assert event.source_identity == obs.source_id
    assert event.snapshot_id == obs.snapshot_id
    assert event.feature_spec_hash == obs.feature_spec_hash
    assert generator.candidates(obs) == (event,)


def test_breakout_candidate_requires_slower_trend_alignment(observed):
    generator = CausalOpportunityGenerator()
    obs = observed(hourly=.01, daily=-.02)
    obs = replace(obs, market={**obs.market, "breakout_1h": .02})
    assert generator.candidates(obs) == ()


def test_pullback_candidate_requires_trend_retrace_and_efficiency(observed):
    generator = CausalOpportunityGenerator()
    obs = observed(hourly=.01)
    obs = replace(obs, market={**obs.market, "zscore_1m": -1.5})
    event, = generator.candidates(obs)
    assert (event.family, event.side) == ("pullback", 1)

    countertrend = replace(obs, market={**obs.market, "trend_1d": -.02})
    assert generator.candidates(countertrend) == ()

    low_efficiency = replace(obs, market={**obs.market, "efficiency_15m": .1})
    assert generator.candidates(low_efficiency) == ()


def test_raw_candidate_universe_survives_position_and_volatility_vetoes(observed):
    generator = CausalOpportunityGenerator()
    breakout = replace(observed(pos=0.2), market={
        **observed(pos=0.2).market, "breakout_1h": .5,
    })
    assert generator.candidates(breakout)[0].family == "breakout"

    volatile = replace(observed(vol=.09), market={
        **observed(vol=.09).market, "breakout_1h": .5,
    })
    # Preserve the candidate denominator; a downstream selector/risk layer
    # records the later volatility veto.
    assert generator.candidates(volatile)[0].family == "breakout"


def test_missing_required_candidate_feature_fails_closed_by_error(observed):
    generator = CausalOpportunityGenerator()
    obs = observed()
    with pytest.raises(KeyError, match="efficiency_15m"):
        generator.candidates(replace(obs, market={
            key: value for key, value in obs.market.items()
            if key != "efficiency_15m"
        }, feature_available_at_ns={
            key: value for key, value in obs.feature_available_at_ns.items()
            if key != "efficiency_15m"
        }))


def test_candidate_generation_requires_auditable_source_snapshot_and_feature_spec(observed):
    generator = CausalOpportunityGenerator()
    obs = observed()
    with pytest.raises(ValueError, match="candidate provenance requires"):
        generator.candidates(replace(obs, snapshot_id=""))


def test_candidate_audit_keeps_intent_planner_and_actual_fill_distinct(observed):
    obs = observed()
    obs = replace(obs, market={**obs.market, "breakout_1h": .02})
    event, = CausalOpportunityGenerator().candidates(obs)
    advice = ExpertAdvice("breakout", obs.decision_ts_ns, "OPEN_LONG", .3, ("BREAKOUT_POS",))
    trace = NativeActionTrace(
        decision_ts_ns=obs.decision_ts_ns,
        requested_action=.3,
        risk_adjusted_target=.3,
        planner_target=.3,
        actual_exposure_after=0.0,
        native_equity_before=1000.0,
        native_equity_after=1000.0,
        planner_action="SUBMIT",
        submitted_order_ids=("open-1",),
        pending_order_ids_after=("open-1",),
        scenario_only=True,
        execution_barrier_ts_ns=obs.decision_ts_ns + 1,
    )
    record = audit_candidate_decision(event, advice, trace)
    assert record["teacher_selected_candidate_side"] is True
    assert record["planner_target"] == .3
    assert record["actual_exposure_after"] == 0.0
    assert record["pending_order_ids_after"] == ["open-1"]
    assert record["scenario_only"] is True


def test_candidate_audit_rejects_mismatched_decision_clock(observed):
    obs = observed()
    obs = replace(obs, market={**obs.market, "breakout_1h": .02})
    event, = CausalOpportunityGenerator().candidates(obs)
    advice = ExpertAdvice("breakout", obs.decision_ts_ns + 1, "OPEN_LONG", .3, ("BREAKOUT_POS",))
    trace = NativeActionTrace(
        decision_ts_ns=obs.decision_ts_ns,
        requested_action=.3,
        risk_adjusted_target=.3,
        planner_target=.3,
        actual_exposure_after=0.0,
        native_equity_before=1000.0,
        native_equity_after=1000.0,
        planner_action="SUBMIT",
    )
    with pytest.raises(ValueError, match="clocks do not match"):
        audit_candidate_decision(event, advice, trace)
