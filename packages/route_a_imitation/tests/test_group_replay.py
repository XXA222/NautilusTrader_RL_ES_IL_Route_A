import pytest

from rles_route_a.teacher_research.candidate_event_union import TeacherProposal, group_teacher_proposals
from rles_route_a.teacher_research.causal_exit import CausalExitSchedule
from rles_route_a.teacher_research.conditional_outcome_labels import label_grouped_candidates
from rles_route_a.teacher_research.group_replay import make_group_market_replay

M = 60_000_000_000


def test_group_uses_next_open_and_censors_missing_in_block_entry(observed):
    obs = observed(ts=2*M+1)
    late = observed(ts=5*M-1)
    group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
    late_group, = group_teacher_proposals([TeacherProposal(late, "EMA", 1, "cross")])
    schedule = CausalExitSchedule("fixed", M, 5*M, (3*M+1,), ())
    replay = make_group_market_replay(
        schedule=schedule, opens=tuple((i*M, 100.+i) for i in range(1, 6)),
        funding=(), block_ends={group.event_id: 5*M, late_group.event_id: 5*M},
        fee_bps=5., slippage_bps=1., execution_recipe="next-open", scenario="base")
    result = label_grouped_candidates([group, late_group], replay, exit_policy_id="fixed",
        execution_recipe="next-open", cost_scenario_id="base", block_end_ns=5*M,
        gross_return_method="simple")
    assert result[0].label_end_ts_ns == 4*M
    assert result[0].net_return_bps == pytest.approx((104/103-1)*10000 - 6 - 6*104/103)
    assert result[1].outcome_status == "RIGHT_CENSORED"
    assert result[1].net_return_bps is None
