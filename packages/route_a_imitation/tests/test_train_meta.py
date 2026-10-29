import pytest

from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.teacher_research.candidate_event_union import TeacherProposal, group_teacher_proposals
from rles_route_a.teacher_research.conditional_outcome_labels import FrozenReplay
from rles_route_a.teacher_research.opportunity_labels import CounterfactualTradeReplay
from rles_route_a.teacher_research.train_meta import train_meta_on_registered_candidates


def test_registered_meta_training_reports_coverage_and_delays_inference(observed):
    groups = []
    for i in range(1, 9):
        obs = observed(ts=i*1_000_000_000, hourly=.01 if i % 2 else -.01)
        group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
        groups.append(group)
    def replay(group, policy, execution, scenario):
        obs = group.observation
        end = 101. if obs.market["trend_1h"] > 0 else 99.
        fact = CounterfactualTradeReplay(group.event_id, obs.decision_ts_ns+1, 100.,
            obs.decision_ts_ns+2, end, 1., 1., 1., 1., 0., None, "synthetic")
        return FrozenReplay(fact, policy, execution, scenario)
    grant = HostTrainGrant("f1", "owner", "reservation", "fixture-source", "train", True)
    kwargs = dict(grant=grant, fit_groups=groups[:4], calibration_groups=groups[4:],
                  fit_interval=(1, 4_500_000_000),
                  calibration_interval=(4_500_000_000, 9_000_000_000),
                  replay=replay, execution_recipe="native", scenario="base",
                  feature_names=("market.trend_1h", "teacher.EMA", "consensus.count",
                                 "cost.round_trip_bps"), margin_bps=1., cost_bps=4.)
    with pytest.raises(PermissionError):
        train_meta_on_registered_candidates(**kwargs, verify_grant=lambda grant: False)
    result = train_meta_on_registered_candidates(**kwargs, verify_grant=lambda grant: True)
    assert result.diagnostics["fit_complete_used"] == 4
    assert result.diagnostics["calibration_complete_used"] == 4
    assert result.policy.return_basis_id == "LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS"
    assert result.policy.available_after_ns == 8_000_000_002
    with pytest.raises(ValueError, match="before calibration"):
        result.policy.score(groups[-1])
