from hashlib import sha256
import math
from dataclasses import replace
from typing import Literal

import pytest

from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.teacher_research.candidate_events import OpportunityEvent
from rles_route_a.teacher_research.causal_exit import TrendInvalidationExit
from rles_route_a.teacher_research.opportunity_models import OpportunityOutcome
from rles_route_a.teacher_research.opportunity_labels import CounterfactualTradeReplay
from rles_route_a.teacher_research.portfolio_replay import (
    MINUTE_NS,
    ResearchFundingSettlement,
    ResearchMinuteBar,
    ResearchPortfolioRecipe,
    admit_single_position_candidates,
    replay_single_position_portfolio,
    simulate_causal_research_portfolio,
)


START = 1_000_000_000
END = 100_000_000_000


def _event(number: int, decision: int) -> OpportunityEvent:
    return OpportunityEvent(
        event_id=sha256(f"candidate-{number}".encode()).hexdigest(),
        decision_ts_ns=decision,
        feature_available_at_ns=decision,
        side=1,
        family="breakout",
        fold_id="train-inner",
        source_identity="synthetic-source",
        rule_version="test-v1",
        snapshot_id=sha256(b"snapshot").hexdigest(),
        feature_spec_hash=sha256(b"features").hexdigest(),
    )


def _replay(event: OpportunityEvent, *, entry: int, exit: int | None,
            censor: int | None = None) -> CounterfactualTradeReplay:
    return CounterfactualTradeReplay(
        event_id=event.event_id,
        entry_ts_ns=entry,
        entry_price=100.0,
        exit_ts_ns=exit,
        exit_price=101.0 if exit is not None else None,
        entry_fee_bps=1.0,
        exit_fee_bps=1.0,
        entry_slippage_bps=0.5,
        exit_slippage_bps=0.5,
        funding_cashflow_bps=0.0,
        censor_ts_ns=censor,
        simulation_receipt_id=sha256(f"receipt-{event.event_id}".encode()).hexdigest(),
    )


def _outcome(event: OpportunityEvent, *, net: float | None, end: int | None,
             censor: int | None, receipt: str, gross_return_bps=None,
             fees_bps=None, slippage_bps=None, funding_cashflow_bps=None) -> OpportunityOutcome:
    if net is not None:
        gross_return_bps = net if gross_return_bps is None else gross_return_bps
        fees_bps = 0.0 if fees_bps is None else fees_bps
        slippage_bps = 0.0 if slippage_bps is None else slippage_bps
        funding_cashflow_bps = 0.0 if funding_cashflow_bps is None else funding_cashflow_bps
    return OpportunityOutcome(
        event_id=event.event_id,
        candidate_family=event.family,
        candidate_side=event.side,
        decision_ts_ns=event.decision_ts_ns,
        feature_available_at_ns=event.feature_available_at_ns,
        rule_version=event.rule_version,
        snapshot_id=event.snapshot_id,
        feature_spec_hash=event.feature_spec_hash,
        feature_vector_sha256=sha256(b"features").hexdigest(),
        label_end_ts_ns=end,
        net_return_bps=net,
        gross_return_bps=gross_return_bps,
        fees_bps=fees_bps,
        slippage_bps=slippage_bps,
        funding_cashflow_bps=funding_cashflow_bps,
        fold_id=event.fold_id,
        source_identity=event.source_identity,
        split="train_inner_screen",
        independent_block_id="screen-1",
        independent_block_start_ts_ns=START,
        independent_block_end_ts_ns=END,
        exit_policy_id="test-exit-v1",
        execution_recipe="next-open-v1",
        cost_scenario_id="base",
        return_basis_id="LOG_REFERENCE_ENTRY_NOTIONAL_BPS",
        simulation_receipt_id=receipt,
        outcome_method="synthetic-test-v1",
        censor_ts_ns=censor,
    )


def test_censored_position_is_admitted_then_blocks_later_candidates():
    first, later = _event(1, START), _event(2, START + 20)
    replays = {
        first.event_id: _replay(first, entry=START + 1, exit=None, censor=END),
        later.event_id: _replay(later, entry=START + 21, exit=START + 30),
    }

    result = admit_single_position_candidates(
        {first.event_id, later.event_id}, (first, later), replays, block_end_ns=END,
    )

    assert result.accepted_ids == (first.event_id,)
    assert result.busy_ids == (later.event_id,)
    assert result.no_in_block_entry_ids == ()


def test_closed_position_releases_ledger_only_after_exit():
    first, overlapping, after = _event(1, START), _event(2, START + 10), _event(3, START + 30)
    replays = {
        first.event_id: _replay(first, entry=START + 1, exit=START + 20),
        overlapping.event_id: _replay(overlapping, entry=START + 11, exit=START + 25),
        after.event_id: _replay(after, entry=START + 31, exit=START + 40),
    }

    result = admit_single_position_candidates(
        {first.event_id, overlapping.event_id, after.event_id},
        (first, overlapping, after), replays, block_end_ns=END,
    )

    assert result.accepted_ids == (first.event_id, after.event_id)
    assert result.busy_ids == (overlapping.event_id,)


def test_post_block_entry_is_not_counted_as_a_fill():
    event = _event(1, END - 1)
    result = admit_single_position_candidates(
        {event.event_id}, (event,),
        {event.event_id: _replay(event, entry=END, exit=None, censor=END + 1)},
        block_end_ns=END,
    )
    assert result.accepted_ids == ()
    assert result.no_in_block_entry_ids == (event.event_id,)


def test_selection_requires_an_explicit_in_block_replay():
    event = _event(1, START)
    with pytest.raises(ValueError, match="missing its replay"):
        admit_single_position_candidates({event.event_id}, (event,), {}, block_end_ns=END)


def test_closed_ledger_reconciles_fees_slippage_and_funding_once():
    event = _event(1, START)
    replay = CounterfactualTradeReplay(
        event_id=event.event_id,
        entry_ts_ns=START + 1,
        entry_price=100.0,
        exit_ts_ns=START + 10,
        exit_price=110.0,
        entry_fee_bps=2.0,
        exit_fee_bps=1.0,
        entry_slippage_bps=0.75,
        exit_slippage_bps=0.25,
        funding_cashflow_bps=0.5,
        censor_ts_ns=None,
        simulation_receipt_id=sha256(b"closed").hexdigest(),
    )
    net = 10_000 * math.log(1.1) - 4.0 + 0.5
    result = replay_single_position_portfolio(
        {event.event_id}, (event,), {event.event_id: replay},
        {event.event_id: _outcome(
            event, net=net, end=START + 10, censor=None,
            receipt=replay.simulation_receipt_id,
            gross_return_bps=10_000 * math.log(1.1), fees_bps=3.0,
            slippage_bps=1.0, funding_cashflow_bps=0.5,
        )},
        {}, block_end_ns=END,
    )
    assert result.realized_net_bps == pytest.approx(net)
    assert result.terminal_mark_only_bps == 0.0
    assert result.total_net_bps == pytest.approx(net)
    assert result.fees_bps == 3.0
    assert result.slippage_bps == 1.0
    assert result.funding_bps == 0.5
    assert result.filled_orders == 2
    assert result.trades[0].status == "REALIZED"


@pytest.mark.parametrize("terminal_price", (103.0, 97.0))
def test_open_position_uses_mark_only_and_does_not_fabricate_exit(
    terminal_price: float,
):
    event = _event(1, START)
    replay = CounterfactualTradeReplay(
        event_id=event.event_id,
        entry_ts_ns=START + 1,
        entry_price=100.0,
        exit_ts_ns=None,
        exit_price=None,
        entry_fee_bps=2.0,
        exit_fee_bps=5.0,
        entry_slippage_bps=0.75,
        exit_slippage_bps=3.0,
        funding_cashflow_bps=0.5,
        censor_ts_ns=END,
        simulation_receipt_id=sha256(b"open").hexdigest(),
    )
    result = replay_single_position_portfolio(
        {event.event_id}, (event,), {event.event_id: replay},
        {event.event_id: _outcome(
            event, net=None, end=None, censor=END,
            receipt=replay.simulation_receipt_id,
        )},
        {event.event_id: (END - 1, terminal_price)},
        block_end_ns=END,
    )
    expected = 10_000 * math.log(terminal_price / 100.0) - 2.75 + 0.5
    assert result.realized_net_bps == 0.0
    assert result.terminal_mark_only_bps == pytest.approx(expected)
    assert result.total_net_bps == pytest.approx(expected)
    assert result.fees_bps == 2.0
    assert result.slippage_bps == 0.75
    assert result.funding_bps == 0.5
    assert result.filled_orders == 1
    assert result.open_censored_count == 1
    assert result.trades[0].status == "TERMINAL_MARK_ONLY"


def test_closed_label_cashflow_mismatch_fails_closed():
    event = _event(1, START)
    replay = _replay(event, entry=START + 1, exit=START + 10)
    with pytest.raises(ValueError, match="cashflows differ"):
        replay_single_position_portfolio(
            {event.event_id}, (event,), {event.event_id: replay},
            {event.event_id: _outcome(
                event, net=999.0, end=START + 10, censor=None,
                receipt=replay.simulation_receipt_id,
            )},
            {}, block_end_ns=END,
        )


def _account_path(*, minutes: int, side: Literal[-1, 1] = 1, final_price: float = 100.0,
                  invalidation_minute: int | None = None):
    start = 10 * MINUTE_NS
    end = start + (minutes + 1) * MINUTE_NS
    bars = tuple(ResearchMinuteBar(
        open_ts_ns=start + index * MINUTE_NS,
        close_available_at_ns=start + (index + 1) * MINUTE_NS,
        open_price=100.0,
        close_price=(100.0 if index < minutes - 1 else final_price),
    ) for index in range(minutes))
    observations = []
    for index, bar in enumerate(bars):
        trend = -side * 0.01 if invalidation_minute == index else side * 0.01
        available = bar.close_available_at_ns
        observations.append(PointInTimeObservation(
            episode_id="synthetic-account",
            decision_ts_ns=available,
            market={"trend_1h": trend, "trend_1d": trend},
            feature_available_at_ns={"trend_1h": available, "trend_1d": available},
            account={},
            fold_id="synthetic",
            source_id="synthetic-source",
            snapshot_id=sha256(f"snapshot-{index}".encode()).hexdigest(),
            feature_spec_hash=sha256(b"account-features").hexdigest(),
        ))
    event = OpportunityEvent(
        event_id=sha256(b"account-candidate").hexdigest(),
        decision_ts_ns=observations[0].decision_ts_ns,
        feature_available_at_ns=observations[0].decision_ts_ns,
        side=side,
        family="breakout",
        fold_id="synthetic",
        source_identity="synthetic-source",
        rule_version="synthetic-v1",
        snapshot_id=observations[0].snapshot_id,
        feature_spec_hash=observations[0].feature_spec_hash,
    )
    return start, end, bars, tuple(observations), event


def _simulate_account(*, minutes: int, side: Literal[-1, 1] = 1, final_price: float = 100.0,
                      funding=(), invalidation_minute: int | None = None,
                      terminal_policy="mark_only"):
    from rles_route_a.teacher_research.portfolio_replay import simulate_causal_research_portfolio

    start, end, bars, observations, event = _account_path(
        minutes=minutes, side=side, final_price=final_price,
        invalidation_minute=invalidation_minute,
    )
    return simulate_causal_research_portfolio(
        ResearchPortfolioRecipe(
            scenario_id="synthetic-account-v1", target_fraction=0.3,
            starting_equity=10_000.0, fee_bps_per_leg=1.0,
            slippage_bps_per_leg=2.0, entry_cadence_minutes=1,
            terminal_policy=terminal_policy,
        ),
        (event,), {event.event_id}, bars, observations, funding,
        TrendInvalidationExit(threshold=0.002, scale_policy="both"),
        start_ns=start, end_ns=end,
    )


def test_account_replay_is_causal_and_independent_of_truncated_future():
    full = _simulate_account(minutes=12, final_price=120.0)
    truncated = _simulate_account(minutes=7, final_price=120.0)
    cutoff = truncated.equity_timeline[-1].ts_ns
    full_prefix = tuple(row for row in full.equity_timeline if row.ts_ns < cutoff)
    truncated_prefix = tuple(row for row in truncated.equity_timeline if row.ts_ns < cutoff)
    assert full_prefix == truncated_prefix
    assert full.candidate_decision_trace[0].decision_ts_ns == truncated.candidate_decision_trace[0].decision_ts_ns
    assert full.candidate_decision_trace[0].execution_ts_ns == truncated.candidate_decision_trace[0].execution_ts_ns


@pytest.mark.parametrize("minutes", (239, 241, 480, 1440, 6400))
def test_account_holds_beyond_240_minutes_without_a_fabricated_timeout(minutes: int):
    result = _simulate_account(minutes=minutes)
    assert result.realized_trade_count == 0
    assert result.open_censored_count == 1
    assert result.terminal_force_flat_count == 0
    assert result.exposure_minutes == pytest.approx(minutes - 2)
    if minutes > 242:
        assert result.exposure_minutes > 240


@pytest.mark.parametrize(("side", "expected_sign"), ((1, -1), (-1, 1)))
def test_account_funding_is_settled_once_with_correct_long_short_sign(
    side: Literal[-1, 1], expected_sign: Literal[-1, 1],
):
    start, *_ = _account_path(minutes=8, side=side)
    settlement = ResearchFundingSettlement(
        ts_ns=start + 5 * MINUTE_NS,
        rate=0.001,
        mark_ts_ns=start + 4 * MINUTE_NS + 1,
        mark_price=100.0,
    )
    result = _simulate_account(minutes=8, side=side, funding=(settlement,))
    assert result.funding_cashflow_signed_usdt == pytest.approx(expected_sign * 3.0)
    funding_changes = [
        right.cumulative_funding_cashflow_usdt - left.cumulative_funding_cashflow_usdt
        for left, right in zip(result.equity_timeline, result.equity_timeline[1:])
    ]
    assert sum(change != 0 for change in funding_changes) == 1


def test_account_equity_conserves_long_short_pnl_costs_and_terminal_policy():
    long = _simulate_account(minutes=8, side=1, final_price=110.0)
    short = _simulate_account(minutes=8, side=-1, final_price=110.0)
    assert long.end_equity_usdt == pytest.approx(10_299.1)
    assert short.end_equity_usdt == pytest.approx(9_699.1)
    assert long.fees_paid_usdt == pytest.approx(0.3)
    assert long.slippage_cost_usdt == pytest.approx(0.6)
    assert long.total_turnover_usdt == pytest.approx(3_000.0)
    assert long.economic_status == "SCENARIO_ONLY"
    flat = _simulate_account(minutes=8, final_price=110.0, terminal_policy="scenario_force_flat")
    assert flat.open_censored_count == 0
    assert flat.terminal_force_flat_count == 1
    assert flat.filled_order_count == 2
    assert flat.fees_paid_usdt == pytest.approx(0.63)
    assert flat.slippage_cost_usdt == pytest.approx(1.26)
    assert flat.total_turnover_usdt == pytest.approx(6_300.0)


def test_account_exit_uses_shared_dynamic_rule_and_next_open():
    result = _simulate_account(minutes=8, invalidation_minute=4)
    trace = result.candidate_decision_trace[0]
    assert trace.execution_ts_ns is not None
    assert result.realized_trade_count == 1
    assert result.open_censored_count == 0
    assert result.filled_order_count == 2


def test_account_rejects_duplicate_or_reversed_bar_clock_before_replay():
    start, end, bars, observations, event = _account_path(minutes=8)
    with pytest.raises(ValueError, match="missing or duplicate"):
        simulate_causal_research_portfolio(
            ResearchPortfolioRecipe("bad-bars", 0.3, 10_000, 1, 2, 1),
            (event,), {event.event_id}, (bars[0], bars[0], *bars[2:]), observations,
            (), TrendInvalidationExit(), start_ns=start, end_ns=end,
        )


def test_account_rejects_candidate_outside_declared_block():
    start, end, bars, observations, event = _account_path(minutes=8)
    outside = replace(event, decision_ts_ns=end, feature_available_at_ns=end)
    with pytest.raises(ValueError, match="noncausal or out of window"):
        simulate_causal_research_portfolio(
            ResearchPortfolioRecipe("outside-event", 0.3, 10_000, 1, 2, 1),
            (outside,), {outside.event_id}, bars, observations, (),
            TrendInvalidationExit(), start_ns=start, end_ns=end,
        )


def test_opposite_candidates_at_same_clock_are_rejected_as_conflict():
    start, end, bars, observations, long_event = _account_path(minutes=8)
    short_event = replace(
        long_event,
        event_id=sha256(b"opposite-account-candidate").hexdigest(),
        side=-1,
    )
    result = simulate_causal_research_portfolio(
        ResearchPortfolioRecipe("conflicting-events", 0.3, 10_000, 1, 2, 1),
        (long_event, short_event), {long_event.event_id, short_event.event_id},
        bars, observations, (), TrendInvalidationExit(), start_ns=start, end_ns=end,
    )
    assert result.filled_order_count == 0
    assert {row.status for row in result.candidate_decision_trace} == {
        "CONFLICTING_CANDIDATE_SIDES",
    }


def test_account_rejects_funding_mark_at_or_outside_freshness_limit():
    start, end, bars, observations, event = _account_path(minutes=8)
    settlement = ResearchFundingSettlement(
        ts_ns=start + 5 * MINUTE_NS,
        rate=0.001,
        mark_ts_ns=start + 4 * MINUTE_NS,
        mark_price=100.0,
    )
    with pytest.raises(ValueError, match="strictly fresh"):
        simulate_causal_research_portfolio(
            ResearchPortfolioRecipe("stale-funding", 0.3, 10_000, 1, 2, 1),
            (event,), {event.event_id}, bars, observations, (settlement,),
            TrendInvalidationExit(), start_ns=start, end_ns=end,
        )
