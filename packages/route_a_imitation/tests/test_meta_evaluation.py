import pytest

from rles_route_a.evaluation.teacher_case_report_1m import NativeTradeCase1m
from rles_route_a.teacher_research.meta_evaluation import summarize_native_economics, summarize_candidate_quality
from rles_route_a.teacher_research.meta_gated_teacher import CandidateDecision
from rles_route_a.teacher_research.entry_meta_policy import EntryScore
from rles_route_a.teacher_research.candidate_event_union import TeacherProposal, group_teacher_proposals
from rles_route_a.teacher_research.conditional_outcome_labels import GroupOutcome


def test_native_economics_counts_open_censor_separately():
    case = NativeTradeCase1m(
        trade_case_id="native", signal_id="signal", teacher_id="EMA", exit_policy_id="X1",
        scenario_id="base", block_id="train", side=1, status="ORDINARY_COMPLETE",
        entry_decision_ts_ns=1, entry_signal_price=100., entry_fill_ts_ns=2,
        entry_fill_price=100., entry_notional=1000., exit_signal_price=101.,
        exit_fill_ts_ns=4, exit_fill_price=101., fees=2., fee_currency="USDT",
        equity_currency="USDT", slippage_cost=1., funding=0., realized_pnl=8.,
        equity_start=1000., equity_end=1008., mfe_bps=100., mae_bps=-10.,
        holding_minutes=2., exit_reason="exit", economic_status="SCENARIO_ONLY")
    report = summarize_native_economics((case,), (1000., 990., 1008.))
    assert report["completed_trade_gross_before_costs"] == pytest.approx(11.)
    assert report["completed_trade_net_pnl"] == 8.
    assert report["max_drawdown"] == pytest.approx(.01)
    assert report["long"]["win_rate"] == 1. and report["short"]["win_rate"] is None


def test_quality_reports_selected_rejected_and_censor(observed):
    decisions, labels = [], []
    for i in range(3):
        obs = observed(ts=(i+1)*1_000_000_000)
        group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
        decisions.append(CandidateDecision(group, EntryScore("LONG", .7, 5., True, "EDGE"), i == 0))
        labels.append(GroupOutcome(group.event_id, obs.decision_ts_ns, obs.source_id, obs.fold_id,
             "X1", "native", "base", obs.decision_ts_ns+1 if i != 2 else None,
             obs.decision_ts_ns+2 if i == 2 else None, 5. if i == 0 else (-2. if i == 1 else None),
             outcome_status="RIGHT_CENSORED" if i == 2 else "COMPLETE"))
    result = summarize_candidate_quality(tuple(decisions), tuple(labels))
    assert result["accepted"]["mean_net_bps"] == 5.
    assert result["rejected"]["mean_net_bps"] == -2.
    assert result["rejected"]["without_outcome"] == 1
    assert result["quality_by_disposition"]["REJECTED_BY_ARBITRATION"]["complete"] == 1
    assert result["quality_by_disposition"]["REJECTED_BY_ARBITRATION"]["without_outcome"] == 1
    assert result["prediction_quality"]["paired_complete_scored"] == 2
    assert result["prediction_quality"]["scored_without_complete_outcome"] == 1
    assert result["prediction_quality"]["mean_absolute_error_bps"] == pytest.approx(3.5)
    assert result["prediction_quality"]["mean_prediction_error_bps"] == pytest.approx(3.5)
    assert result["prediction_quality"]["brier_score"] == pytest.approx(.29)
    assert result["evidence_type"] == "MODEL_COUNTERFACTUAL"
    from dataclasses import replace
    no_pairs = summarize_candidate_quality(
        (replace(decisions[0], score=None, selected=False), decisions[2]), (labels[0], labels[2]))
    assert no_pairs["prediction_quality"]["paired_complete_scored"] == 0
    assert no_pairs["prediction_quality"]["unscored_candidates"] == 1
    assert no_pairs["prediction_quality"]["brier_score"] is None
    with pytest.raises(ValueError, match="cannot mix"):
        summarize_candidate_quality(tuple(decisions),
            (labels[0], replace(labels[1], cost_scenario_id="stress"), labels[2]))
    with pytest.raises(ValueError, match="causal candidate"):
        summarize_candidate_quality(tuple(decisions),
            (labels[0], replace(labels[1], source_identity="different"), labels[2]))



def test_native_comparison_reports_missing_cells_without_model_substitution():
    from rles_route_a.teacher_research.meta_evaluation import compare_preregistered_native_runs
    config = {"teachers": ["EMA"], "controls": ["FLAT", "FIXED_LONG_0.3", "FIXED_SHORT_0.3", "ORIGINAL_8_X1"],
              "cost_scenarios": {"base": {}, "stress": {}},
              "intervals": {"native_screen": [["a", "b"], ["b", "c"]]}}
    run = {"strategy_id": "FLAT", "scenario": "base", "window": ("a", "b"),
           "cases": (), "equity_curve": (1000., 1000.),
           "native_order_verification": {"barriers": 10, "authority": "retained_m22_session_snapshots"}}
    result = compare_preregistered_native_runs(config, (run,))
    assert result["expected_cells"] == 20 and result["available_cells"] == 1
    assert result["status"] == "INCOMPLETE" and result["qualification"] == "NOT_ESTABLISHED"
    with pytest.raises(ValueError, match="duplicate"):
        compare_preregistered_native_runs(config, (run, run))
    with pytest.raises(ValueError, match="M2.2"):
        compare_preregistered_native_runs(config, ({**run, "native_order_verification": {}},))
