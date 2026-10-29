import pytest

from rles_route_a.teacher_research.candidate_collector import OneMinuteCandidateCollector
from rles_route_a.teacher_research.conditional_outcome_labels import label_grouped_candidates
from rles_route_a.teacher_research.x1_group_replay import make_x1_group_replay
from test_candidate_collector import observation
from rles_route_a.indicators_1m import MINUTE_NS as M


def test_x1_replay_uses_original_teacher_exit_and_next_opens():
    rows = tuple(observation(i, ema_9=fast) for i, fast in enumerate((99., 101., 101., 99., 99.), 1))
    collector = OneMinuteCandidateCollector()
    collector.collect(rows[0])
    group, = collector.collect(rows[1]).groups
    assert group.teachers[0][0] == "T01_EMA_9_21"
    replay = make_x1_group_replay(observations=rows,
        opens=tuple((i*M, 100.+i) for i in range(1, 7)), funding=(),
        block_ends={group.event_id: 6*M}, fee_bps=5., slippage_bps=1.,
        execution_recipe="next-open", scenario="base")
    result, = label_grouped_candidates([group], replay, exit_policy_id="X1",
        execution_recipe="next-open", cost_scenario_id="base", block_end_ns=6*M,
        gross_return_method="simple")
    assert result.label_end_ts_ns == 5*M
    assert result.net_return_bps == pytest.approx((105/103-1)*10000 - 6 - 6*105/103)
    assert result.evidence_type == "MODEL_COUNTERFACTUAL"
    again, = label_grouped_candidates([group], replay, exit_policy_id="X1",
        execution_recipe="next-open", cost_scenario_id="base", block_end_ns=6*M,
        gross_return_method="simple")
    assert again == result


def test_x1_replay_rejects_gaps_and_wrong_exit_recipe():
    rows = (observation(1), observation(3))
    with pytest.raises(ValueError, match="coverage"):
        make_x1_group_replay(observations=rows, opens=((M,100.), (2*M,100.)),
            funding=(), block_ends={}, fee_bps=5., slippage_bps=1.,
            execution_recipe="native", scenario="base")



def test_limit_touch_integrates_with_x1_exit_and_censor():
    from rles_route_a.teacher_research.limit_entry import LimitBar, make_limit_entry_provider
    rows = tuple(observation(i, ema_9=fast)
                 for i, fast in enumerate((99., 101., 101., 101., 99., 99.), 1))
    collector = OneMinuteCandidateCollector()
    collector.collect(rows[0])
    group, = collector.collect(rows[1]).groups
    common = dict(observations=rows, opens=tuple((i*M, 100.+i) for i in range(1, 8)),
                  funding=(), block_ends={group.event_id: 7*M}, fee_bps=5.,
                  slippage_bps=1., execution_recipe="limit", scenario="base")
    entry = make_limit_entry_provider(bars=(LimitBar(3*M, 4*M, 100., 105.),),
                                      price_for_group=lambda group: 102.)
    replay = make_x1_group_replay(**common, limit_entry=entry)
    result, = label_grouped_candidates([group], replay, exit_policy_id="X1",
        execution_recipe="limit", cost_scenario_id="base", block_end_ns=7*M,
        gross_return_method="simple")
    assert result.label_end_ts_ns == 6*M
    assert result.net_return_bps == pytest.approx((106/102-1)*10000 - 6 - 6*106/102)
    no_touch = make_limit_entry_provider(bars=(LimitBar(3*M, 4*M, 100., 105.),),
                                         price_for_group=lambda group: 90.)
    censored, = label_grouped_candidates([group],
        make_x1_group_replay(**common, limit_entry=no_touch), exit_policy_id="X1",
        execution_recipe="limit", cost_scenario_id="base", block_end_ns=7*M)
    assert censored.outcome_status == "RIGHT_CENSORED" and censored.net_return_bps is None

    complete_no_touch = make_limit_entry_provider(
        bars=(LimitBar(3*M, 4*M, 100., 105.), LimitBar(4*M, 7*M, 100., 105.)),
        price_for_group=lambda group: 90.)
    missed, = label_grouped_candidates([group],
        make_x1_group_replay(**common, limit_entry=complete_no_touch), exit_policy_id="X1",
        execution_recipe="limit", cost_scenario_id="base", block_end_ns=7*M)
    assert missed.outcome_status == "MISSED" and missed.net_return_bps is None



def test_incomplete_tail_is_censored_at_observed_coverage():
    rows = tuple(observation(i, ema_9=fast)
                 for i, fast in enumerate((99., 101., 101.), 1))
    collector = OneMinuteCandidateCollector()
    collector.collect(rows[0])
    group, = collector.collect(rows[1]).groups
    replay = make_x1_group_replay(observations=rows,
        opens=tuple((i*M, 100.+i) for i in range(1, 7)), funding=(),
        block_ends={group.event_id: 6*M}, fee_bps=5., slippage_bps=1.,
        execution_recipe="next-open", scenario="base")
    outcome, = label_grouped_candidates([group], replay, exit_policy_id="X1",
        execution_recipe="next-open", cost_scenario_id="base", block_end_ns=6*M)
    assert outcome.outcome_status == "RIGHT_CENSORED"
    assert outcome.censor_ts_ns == 4*M and outcome.net_return_bps is None
