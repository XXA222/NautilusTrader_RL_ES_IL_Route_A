from math import log

import pytest

from rles_route_a.teacher_research.candidate_event_union import TeacherProposal, group_teacher_proposals
from rles_route_a.teacher_research.conditional_outcome_labels import (
    FrozenReplay,
    GroupOutcome,
    NoEntryReplay,
    label_grouped_candidates,
)
from rles_route_a.teacher_research.native_reconciliation import reconcile_native_trade
from rles_route_a.teacher_research.opportunity_labels import CounterfactualTradeReplay


@pytest.mark.parametrize("status", ["UNFILLED", "CANCELLED", "MISSED", "RIGHT_CENSORED"])
def test_no_entry_is_never_a_zero_return(observed, status):
    obs = observed()
    group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
    def replay(g, policy, execution, cost):
        return FrozenReplay(NoEntryReplay(g.event_id, status, obs.decision_ts_ns + 5, "r"),
                            policy, execution, cost)
    result, = label_grouped_candidates([group], replay, exit_policy_id="X1",
        execution_recipe="limit", cost_scenario_id="base", block_end_ns=obs.decision_ts_ns + 10)
    assert result.net_return_bps is None and result.outcome_status == status
    assert (result.censor_ts_ns is not None) == (status == "RIGHT_CENSORED")
    assert result.simulation_receipt_id == "r"
    assert result.observed_until_ns == obs.decision_ts_ns + 5


def test_linear_and_log_returns_have_explicit_bases(observed):
    obs = observed()
    group, = group_teacher_proposals([TeacherProposal(obs, "EMA", -1, "cross")])
    def replay(g, policy, execution, cost):
        return FrozenReplay(CounterfactualTradeReplay(g.event_id, obs.decision_ts_ns+1, 100.,
            obs.decision_ts_ns+2, 90., 1., 1., 1., 1., -2., None, "r"), policy, execution, cost)
    kwargs = dict(exit_policy_id="X1", execution_recipe="market", cost_scenario_id="base",
                  block_end_ns=obs.decision_ts_ns+10)
    linear, = label_grouped_candidates([group], replay, gross_return_method="simple", **kwargs)
    logarithmic, = label_grouped_candidates([group], replay, **kwargs)
    assert linear.net_return_bps == pytest.approx(994.)
    assert logarithmic.net_return_bps == pytest.approx(-log(.9)*10000 - 6)
    assert linear.return_basis_id == "LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS"
    assert linear.simulation_receipt_id == "r"
    assert linear.observed_until_ns is None
    assert linear.gross_return_bps == pytest.approx(1000.)
    assert linear.fees_bps == pytest.approx(2.)
    assert linear.slippage_bps == pytest.approx(2.)
    assert linear.funding_cashflow_bps == pytest.approx(-2.)
    assert linear.net_return_bps == pytest.approx(
        linear.gross_return_bps - linear.fees_bps - linear.slippage_bps
        + linear.funding_cashflow_bps)
    assert logarithmic.gross_return_bps == pytest.approx(-log(.9) * 10000)


def test_group_outcome_requires_reconciled_complete_economic_components():
    with pytest.raises(ValueError, match="do not reconcile"):
        GroupOutcome("a" * 64, 1, "source", "fold", "X1", "execution", "base",
                     2, None, 1., "receipt", gross_return_bps=4., fees_bps=1.,
                     slippage_bps=1., funding_cashflow_bps=0.)


def test_group_outcome_rejects_ambiguous_return_basis():
    with pytest.raises(ValueError, match="explicit reference basis"):
        GroupOutcome("a" * 64, 1, "source", "fold", "X1", "execution", "base",
                     2, None, 1., "replay", return_basis_id="entry-notional-bps-v1")


def test_group_outcome_requires_candidate_identity_and_positive_decision_clock():
    with pytest.raises(ValueError, match="lowercase SHA-256 candidate"):
        GroupOutcome("short", 1, "source", "fold", "X1", "execution", "base",
                     2, None, 1., "receipt")
    with pytest.raises(ValueError, match="positive integer timestamp"):
        GroupOutcome("a" * 64, 0, "source", "fold", "X1", "execution", "base",
                     2, None, 1., "receipt")


def test_group_outcome_rejects_native_evidence_and_status_return_conflicts():
    common = ("a" * 64, 1, "source", "fold", "X1", "execution", "base", 2, None, 1., "receipt")
    with pytest.raises(ValueError, match="scenario-only model counterfactuals"):
        GroupOutcome(*common, evidence_type="OBSERVED_NATIVE_CASE")
    with pytest.raises(ValueError, match="incomplete group outcome"):
        GroupOutcome("b" * 64, 1, "source", "fold", "X1", "execution", "base",
                     None, 2, 1., "receipt", outcome_status="RIGHT_CENSORED")
    with pytest.raises(ValueError, match="observation end"):
        GroupOutcome("c" * 64, 1, "source", "fold", "X1", "execution", "base",
                     None, None, None, "receipt", outcome_status="MISSED",
                     observed_until_ns=1)


def test_group_outcome_requires_finite_complete_return_and_bound_replay_receipt():
    with pytest.raises(ValueError, match="complete group outcome"):
        GroupOutcome("d" * 64, 1, "source", "fold", "X1", "execution", "base",
                     2, None, float("nan"), "receipt")
    with pytest.raises(ValueError, match="simulation receipt identities"):
        GroupOutcome("e" * 64, 1, "source", "fold", "X1", "execution", "base",
                     2, None, 1.0, " ")
    with pytest.raises(ValueError, match="matching future observation clock"):
        GroupOutcome("f" * 64, 1, "source", "fold", "X1", "execution", "base",
                     None, 3, None, "receipt", outcome_status="RIGHT_CENSORED",
                     observed_until_ns=4)


def test_retained_actual_native_trade_cashflow():
    # Archived Train diagnostic: native-case-047b9e12-31d2-4dfc-931f-e32b441a5c14.
    row = dict(status="ORDINARY_COMPLETE", trade_case_id="native-case-047b9e12-31d2-4dfc-931f-e32b441a5c14",
               side=1, entry_fill_price=2285.97, exit_fill_price=2284.37,
               entry_notional=249.17073, fees=.24908352, funding=0.,
               gross_cash_pnl_before_fees_and_funding=-.17440000000044542,
               realized_pnl=-.4234835200004454, slippage_cost=.05014)
    reconciled = reconcile_native_trade(row)
    assert reconciled["net_cash"] == pytest.approx(row["realized_pnl"])
    assert reconciled["net_bps"] == pytest.approx(-16.995716952807637)
    with pytest.raises(ValueError, match="does not reconcile"):
        reconcile_native_trade({**row, "realized_pnl": row["realized_pnl"] - row["slippage_cost"]})



def test_same_native_case_and_research_label_discrepancy_are_accounted():
    from hashlib import sha256
    from rles_route_a.teacher_research.native_reconciliation import reconcile_counterfactual_to_native
    case = dict(status="ORDINARY_COMPLETE",
        trade_case_id="native-case-047b9e12-31d2-4dfc-931f-e32b441a5c14",
        entry_decision_ts_ns=1706745780001000000, side=1,
        entry_fill_ts_ns=1706745840001000000, exit_fill_ts_ns=1706747160001000000,
        entry_fill_price=2285.97, exit_fill_price=2284.37, entry_notional=249.17073,
        fees=.24908352, funding=0., slippage_cost=.05014,
        gross_cash_pnl_before_fees_and_funding=-.1744,
        realized_pnl=-.42348352, economic_status="SCENARIO_ONLY")
    entry, exit_price = 2285.73, 2284.60
    ratio = exit_price / entry
    replay = CounterfactualTradeReplay(sha256(b"real-native-case").hexdigest(),
        case["entry_fill_ts_ns"]-1_000_000, entry,
        case["exit_fill_ts_ns"]-1_000_000, exit_price,
        5., 5.*ratio, 1., 1.*ratio, 0., None, "historical-candidate-replay")
    result = reconcile_counterfactual_to_native(replay=replay, native_case=case,
        decision_ts_ns=case["entry_decision_ts_ns"], side=1)
    assert result["native_net_cash"] == pytest.approx(case["realized_pnl"])
    assert result["native_net_cash"] == pytest.approx(-.42348352)
    assert result["native_reported_slippage_cash"] == pytest.approx(.05014)
    assert result["native_slippage_treatment"] == "included_in_actual_fill_prices_not_subtracted_again"
    assert result["modeled_net_cash"] == pytest.approx(-.422069582)
    assert result["native_minus_modeled_cash"] == pytest.approx(-.001413938)
    assert result["native_minus_modeled_cash"] == pytest.approx(
        result["native_fill_vs_reference_gross_cash"] + result["modeled_fee_minus_native_fee_cash"]
        + result["native_minus_modeled_funding_cash"] + result["modeled_slippage_cash"])
    assert result["native_minus_modeled_cash"] != 0
    with pytest.raises(ValueError, match="same trade"):
        reconcile_counterfactual_to_native(replay=replay, native_case=case,
            decision_ts_ns=case["entry_decision_ts_ns"]+1, side=1)
