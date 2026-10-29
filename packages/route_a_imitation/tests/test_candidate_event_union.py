from dataclasses import replace
from math import log

import pytest

from rles_route_a.teacher_research.candidate_event_union import (
    TeacherProposal, group_teacher_proposals, proposal_from_signal,
)
from rles_route_a.teacher_research.conditional_outcome_labels import (
    FrozenReplay, label_grouped_candidates, purge_before,
)
from rles_route_a.teacher_research.opportunity_labels import CounterfactualTradeReplay


def test_same_minute_side_deduplicates_but_preserves_teacher_reasons(observed):
    obs = observed()
    rows = [TeacherProposal(obs, "EMA", 1, "cross"),
            TeacherProposal(obs, "ROC", 1, "momentum"),
            TeacherProposal(obs, "RSI", -1, "reversion")]
    groups = group_teacher_proposals(rows)
    assert len(groups) == 2
    long = next(group for group in groups if group.side == 1)
    assert long.teachers == (("EMA", "cross"), ("ROC", "momentum"))
    with pytest.raises(ValueError, match="more than once"):
        group_teacher_proposals([rows[0], rows[0]])
    switched = replace(obs, source_id="other")
    with pytest.raises(ValueError, match="source"):
        group_teacher_proposals([rows[0], TeacherProposal(switched, "ROC", 1, "x")])


def test_frozen_exit_label_censor_and_purge(observed):
    obs = observed()
    group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
    def replay(candidate, policy, execution, cost):
        assert (policy, execution, cost) == ("X1", "native-recipe", "base")
        return FrozenReplay(CounterfactualTradeReplay(
            candidate.event_id, obs.decision_ts_ns + 1, 100.,
            obs.decision_ts_ns + 10, 101., 2., 2., 1., 1., -0.5, None, "receipt",
        ), policy, execution, cost)
    labels = label_grouped_candidates(
        [group], replay, exit_policy_id="X1", execution_recipe="native-recipe",
        cost_scenario_id="base", block_end_ns=obs.decision_ts_ns + 100,
    )
    assert labels[0].net_return_bps == pytest.approx(log(1.01) * 10000 - 6.5)
    assert labels[0].evidence_type == "MODEL_COUNTERFACTUAL"
    assert purge_before(labels, next_start_ns=obs.decision_ts_ns + 10) == ()
    assert purge_before(labels, next_start_ns=obs.decision_ts_ns + 11) == labels
    censored = label_grouped_candidates(
        [group], replay, exit_policy_id="X1", execution_recipe="native-recipe",
        cost_scenario_id="base", block_end_ns=obs.decision_ts_ns + 5,
    )
    assert censored[0].net_return_bps is None
    assert censored[0].censor_ts_ns == obs.decision_ts_ns + 5


def test_mismatched_replay_recipe_is_rejected(observed):
    obs = observed()
    group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
    def replay(candidate, policy, execution, cost):
        fact = CounterfactualTradeReplay(
            candidate.event_id, obs.decision_ts_ns + 1, 100.,
            obs.decision_ts_ns + 10, 101., 2., 2., 1., 1., 0., None, "receipt",
        )
        return FrozenReplay(fact, "X2", execution, cost)
    with pytest.raises(ValueError, match="recipe mismatch"):
        label_grouped_candidates([group], replay, exit_policy_id="X1",
                                 execution_recipe="native", cost_scenario_id="base",
                                 block_end_ns=obs.decision_ts_ns + 100)


def test_existing_teacher_signal_adapter_does_not_force_htf_consensus(observed):
    from rles_route_a.teachers_1m.contracts import TeacherSignal
    obs = observed(hourly=-0.1, daily=-0.2)
    proposal = proposal_from_signal(obs, "EMA", TeacherSignal("LONG_ENTRY", 1, "cross"))
    assert proposal is not None and proposal.side == 1
    assert proposal_from_signal(obs, "EMA", TeacherSignal("NONE", 0, "quiet")) is None
    with pytest.raises(ValueError, match="disagrees"):
        proposal_from_signal(obs, "EMA", TeacherSignal("LONG_ENTRY", -1, "bad"))


def test_future_feature_is_rejected_before_proposal(observed):
    obs = observed()
    available = dict(obs.feature_available_at_ns)
    available["trend_1h"] = obs.decision_ts_ns + 1
    with pytest.raises(ValueError, match="future"):
        replace(obs, feature_available_at_ns=available)


def test_candidate_identity_includes_account_and_pending_state(observed):
    obs = observed()
    group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
    modified = replace(obs, account={**dict(obs.account), "net_exposure": 0.2})
    other, = group_teacher_proposals([TeacherProposal(modified, "EMA", 1, "cross")])
    assert group.event_id != other.event_id
    with pytest.raises(ValueError, match="identity"):
        replace(group, observation=modified)
    pending = replace(obs, pending_order_ids=("order",))
    pending_group, = group_teacher_proposals([TeacherProposal(pending, "EMA", 1, "cross")])
    assert group.event_id != pending_group.event_id
