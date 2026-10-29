from math import log

import pytest

from rles_route_a.teacher_research.candidate_event_union import TeacherProposal, group_teacher_proposals
from rles_route_a.teacher_research.conditional_outcome_labels import FrozenReplay, NoEntryReplay, label_grouped_candidates
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
