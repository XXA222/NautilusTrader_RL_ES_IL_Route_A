from dataclasses import replace
from hashlib import sha256

import pytest

from rles_route_a.teacher_research.candidate_events import OpportunityEvent
from rles_route_a.teacher_research.causal_exit import CausalExitSchedule, TrendInvalidationExit
from rles_route_a.teacher_research.causal_exit_replay import (
    attach_funding_mark_proxies, make_causal_exit_replay,
)
from rles_route_a.teacher_research.opportunity_labels import label_counterfactual_candidates
from rles_route_a.teachers.causal_position import CausalPositionTeacher


MINUTE = 60_000_000_000
START = 10_000 * MINUTE


def _observations(observed, *, count=600, exit_minute=None, side=1):
    for minute in range(count):
        trend = -side * .01 if exit_minute is not None and minute >= exit_minute else side * .01
        yield replace(
            observed(ts=START + minute * MINUTE + 1, pos=side * .3, hourly=trend, daily=trend),
            snapshot_id="a" * 64,
        )


def _schedule(observed, **kwargs):
    return CausalExitSchedule.from_observations(
        _observations(observed, **kwargs), rule=TrendInvalidationExit(),
        start_ns=START, end_ns=START + 600 * MINUTE, cadence_ns=MINUTE,
    )


def _event(observed, side=1):
    obs = next(_observations(observed))
    return OpportunityEvent(
        event_id=sha256(b"candidate").hexdigest(), decision_ts_ns=obs.decision_ts_ns,
        feature_available_at_ns=obs.decision_ts_ns, side=side, family="breakout",
        fold_id=obs.fold_id, source_identity=obs.source_id, rule_version="v2",
        snapshot_id=obs.snapshot_id, feature_spec_hash=obs.feature_spec_hash,
    )


def _replay(observed, schedule, *, side=1, funding=()):
    event = _event(observed, side)
    callback = make_causal_exit_replay(
        schedule=schedule,
        opens=tuple((START + minute * MINUTE, 100 + minute / 100) for minute in range(601)),
        funding=funding, block_ends={event.event_id: schedule.end_ns},
        fee_bps=5, slippage_bps=1, execution_recipe="next-open-v1", scenario="base",
    )
    return event, callback


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("exit_minute", [15, 241, 480, None])
def test_label_and_runtime_share_exact_ordinary_exit_without_time_stop(observed, side, exit_minute):
    schedule = _schedule(observed, exit_minute=exit_minute, side=side)
    teacher = CausalPositionTeacher()
    runtime_exits = []
    for minute, obs in enumerate(_observations(observed, exit_minute=exit_minute, side=side)):
        obs = replace(obs, account={**obs.account, "position_duration_ns": minute * MINUTE})
        if teacher.advise(obs).intent == "EXIT":
            runtime_exits.append(obs.decision_ts_ns)
    expected = runtime_exits[0] if runtime_exits else None
    assert schedule.next_exit_decision(
        side=side, entry_ts_ns=START + MINUTE, block_end_ns=schedule.end_ns,
    ) == expected
    event, replay = _replay(observed, schedule, side=side)
    result = replay(event, schedule.policy_id, "next-open-v1", "base")
    if expected is None:
        assert result.exit_ts_ns is None
        assert result.censor_ts_ns == schedule.end_ns
    else:
        assert result.exit_ts_ns == START + (exit_minute + 1) * MINUTE
        assert result.exit_ts_ns > expected


@pytest.mark.parametrize("defect", ["gap", "duplicate", "truncated", "provenance"])
def test_exit_schedule_rejects_incomplete_or_mixed_path(observed, defect):
    rows = list(_observations(observed))
    if defect == "gap":
        del rows[10]
    elif defect == "duplicate":
        rows.insert(10, rows[10])
    elif defect == "truncated":
        rows = rows[:-2]
    else:
        rows[10] = replace(rows[10], snapshot_id="b" * 64)
    with pytest.raises(ValueError):
        CausalExitSchedule.from_observations(
            rows, rule=TrendInvalidationExit(), start_ns=START,
            end_ns=START + 600 * MINUTE, cadence_ns=MINUTE,
        )


def test_boundary_exit_fill_is_censored_and_not_a_zero_label(observed):
    schedule = _schedule(observed, exit_minute=599)
    event, replay = _replay(observed, schedule)
    result = replay(event, schedule.policy_id, "next-open-v1", "base")
    assert result.exit_ts_ns is None  # decision inside; next open at excluded boundary
    batch = label_counterfactual_candidates(
        [event], replay, {event.event_id: (1.,)}, {event.event_id: "fit-1"},
        {event.event_id: (START, schedule.end_ns)}, feature_names=["trend"],
        split="train_fit", exit_policy_id=schedule.policy_id,
        execution_recipe="next-open-v1", cost_scenario_id="base",
        return_basis_id="LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS", gross_return_method="simple",
    )
    assert batch.outcomes[0].net_return_bps is None
    assert batch.censored_count == 1


def test_simple_cashflow_fees_and_funding_share_entry_notional_basis(observed):
    schedule = _schedule(observed, exit_minute=15)
    funding = ((START + MINUTE, .1, 100.),  # excluded: same time as entry
               (START + 8 * MINUTE, .001, 120.),
               (START + 16 * MINUTE, .002, 110.),  # included: exit time
               (START + 17 * MINUTE, .9, 150.))  # excluded: after exit
    event, replay = _replay(observed, schedule, funding=funding)
    result = replay(event, schedule.policy_id, "next-open-v1", "base")
    expected_funding = -(.001 * 120 + .002 * 110) / 100.01 * 10_000
    assert result.funding_cashflow_bps == pytest.approx(expected_funding)
    batch = label_counterfactual_candidates(
        [event], replay, {event.event_id: (1.,)}, {event.event_id: "fit-1"},
        {event.event_id: (START, schedule.end_ns)}, feature_names=["trend"],
        split="train_fit", exit_policy_id=schedule.policy_id,
        execution_recipe="next-open-v1", cost_scenario_id="base",
        return_basis_id="LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS", gross_return_method="simple",
    )
    expected = (100.16 / 100.01 - 1) * 10_000 - 6 * (1 + 100.16 / 100.01) + expected_funding
    assert batch.outcomes[0].net_return_bps == pytest.approx(expected)
    with pytest.raises(ValueError, match="mismatch"):
        replay(event, "fixed-240m", "next-open-v1", "base")


def test_no_in_block_entry_fails_instead_of_dropping_candidate(observed):
    schedule = _schedule(observed)
    event, replay = _replay(observed, schedule)
    event = replace(event, decision_ts_ns=schedule.end_ns - 1)
    with pytest.raises(ValueError, match="in-block entry"):
        replay(event, schedule.policy_id, "next-open-v1", "base")


def test_exit_threshold_equality_is_hold_and_nonfinite_rejected():
    rule = TrendInvalidationExit()
    assert not rule.invalidated(1, -.002, -.003)
    with pytest.raises(ValueError):
        rule.invalidated(1, float("nan"), -.003)
    with pytest.raises(ValueError):
        TrendInvalidationExit(float("inf"))


@pytest.mark.parametrize("side", (-1, 1))
def test_either_scale_policy_matches_runtime_and_research_endpoints(observed, side):
    rule = TrendInvalidationExit(scale_policy="either")
    assert rule.policy_id != TrendInvalidationExit().policy_id
    assert rule.invalidated(side, -side * .003, side * .01)
    assert not TrendInvalidationExit().invalidated(side, -side * .003, side * .01)
    teacher = CausalPositionTeacher(exit_scale_policy="either")
    rows = tuple(replace(
        observed(ts=START + minute * MINUTE + 1, pos=side * .3,
                 hourly=(-side * .003 if minute >= 15 else side * .01),
                 daily=side * .01), snapshot_id="a" * 64,
    ) for minute in range(600))
    schedule = CausalExitSchedule.from_observations(
        rows, rule=rule, start_ns=START, end_ns=START + 600 * MINUTE,
        cadence_ns=MINUTE,
    )
    assert schedule.next_exit_decision(
        side=side, entry_ts_ns=rows[1].decision_ts_ns,
        block_end_ns=schedule.end_ns,
    ) == rows[15].decision_ts_ns
    assert teacher.advise(rows[15]).intent == "EXIT"
    assert CausalPositionTeacher().advise(rows[15]).intent == "HOLD"
    with pytest.raises(ValueError, match="scale policy"):
        TrendInvalidationExit(scale_policy="unknown")


def test_funding_clock_jitter_uses_past_mark_not_future_or_stale():
    marks = ((START, 100.), (START + MINUTE, 200.))
    rates = ((START + 1_000_000, .001),)
    assert attach_funding_mark_proxies(rates, marks) == ((rates[0][0], .001, 100.),)
    with pytest.raises(ValueError, match="recent closed mark"):
        attach_funding_mark_proxies(((START - 1, .001),), marks)
    with pytest.raises(ValueError, match="recent closed mark"):
        attach_funding_mark_proxies(((START + 2 * MINUTE, .001),), marks)
