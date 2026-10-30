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
    assert report["winning_trades"] == 1 and report["losing_trades"] == 0
    assert report["breakeven_trades"] == 0
    assert report["average_winning_trade_pnl"] == 8.
    assert report["average_losing_trade_pnl"] is None


def test_native_economics_separates_average_wins_losses_and_breakeven():
    from dataclasses import replace

    winner = NativeTradeCase1m(
        trade_case_id="win", signal_id="win-signal", teacher_id="EMA", exit_policy_id="X1",
        scenario_id="base", block_id="train", side=1, status="ORDINARY_COMPLETE",
        entry_decision_ts_ns=1, entry_signal_price=100., entry_fill_ts_ns=2,
        entry_fill_price=100., entry_notional=1000., exit_signal_price=101.,
        exit_fill_ts_ns=4, exit_fill_price=101., fees=2., fee_currency="USDT",
        equity_currency="USDT", slippage_cost=1., funding=0., realized_pnl=8.,
        equity_start=1000., equity_end=1008., mfe_bps=100., mae_bps=-10.,
        holding_minutes=2., exit_reason="exit", economic_status="SCENARIO_ONLY")
    loser = replace(winner, trade_case_id="loss", signal_id="loss-signal",
        exit_signal_price=99., exit_fill_price=99., realized_pnl=-12.)
    flat = replace(winner, trade_case_id="flat", signal_id="flat-signal",
        exit_signal_price=100.2, exit_fill_price=100.2, realized_pnl=0.)

    report = summarize_native_economics((winner, loser, flat), (1000., 1010.))

    assert report["winning_trades"] == report["losing_trades"] == report["breakeven_trades"] == 1
    assert report["average_winning_trade_pnl"] == 8.
    assert report["average_losing_trade_pnl"] == -12.


def test_quality_reports_selected_rejected_and_censor(observed):
    decisions, labels = [], []
    for i in range(3):
        obs = observed(ts=(i+1)*1_000_000_000)
        group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
        score = (EntryScore("NO_TRADE", .7, 5., False, "OUT_OF_SUPPORT", ("market.foo",)) if i == 1
                 else EntryScore("LONG", .7, 5., True, "EDGE"))
        decisions.append(CandidateDecision(group, score, i == 0))
        labels.append(GroupOutcome(group.event_id, obs.decision_ts_ns, obs.source_id, obs.fold_id,
             "X1", "native", "base", obs.decision_ts_ns+1 if i != 2 else None,
             obs.decision_ts_ns+2 if i == 2 else None, 5. if i == 0 else (-2. if i == 1 else None),
             outcome_status="RIGHT_CENSORED" if i == 2 else "COMPLETE"))
    result = summarize_candidate_quality(tuple(decisions), tuple(labels))
    assert result["accepted"]["mean_net_bps"] == 5.
    assert result["rejected"]["mean_net_bps"] == -2.
    assert result["accepted"]["candidates"] == 2
    assert result["accepted"]["without_outcome"] == 1
    assert result["accepted_not_native_selected"]["candidates"] == 1
    assert result["accepted_not_native_selected"]["without_outcome"] == 1
    assert result["native_selected"]["candidates"] == 1
    assert result["rejected"]["without_outcome"] == 0
    assert result["accepted"]["positive_complete_outcomes"] == 1
    assert result["accepted"]["average_positive_net_bps"] == 5.
    assert result["accepted"]["negative_complete_outcomes"] == 0
    assert result["accepted"]["average_negative_net_bps"] is None
    assert result["rejected"]["negative_complete_outcomes"] == 1
    assert result["rejected"]["average_negative_net_bps"] == -2.
    assert result["rejected"]["positive_complete_outcomes"] == 0
    assert result["quality_by_side"]["long"]["accepted"]["mean_net_bps"] == 5.
    assert result["quality_by_side"]["short"]["all"]["candidates"] == 0
    assert result["quality_by_teacher"]["EMA"]["accepted"]["candidates"] == 2
    assert result["quality_by_teacher"]["EMA"]["accepted"]["complete"] == 1
    assert result["quality_by_teacher"]["EMA"]["rejected"]["mean_net_bps"] == -2.
    assert result["quality_by_consensus_count"]["1"]["all"]["candidates"] == 3
    assert result["slice_interpretation"]["teacher_slices_overlap"] is True
    assert result["quality_by_disposition"]["REJECTED_BY_ARBITRATION"]["complete"] == 0
    assert result["quality_by_disposition"]["REJECTED_BY_ARBITRATION"]["without_outcome"] == 1
    assert result["prediction_quality"]["paired_complete_scored"] == 2
    assert result["prediction_quality"]["scored_without_complete_outcome"] == 1
    assert result["prediction_quality"]["mean_absolute_error_bps"] == pytest.approx(3.5)
    assert result["prediction_quality"]["mean_prediction_error_bps"] == pytest.approx(3.5)
    assert result["prediction_quality"]["brier_score"] == pytest.approx(.29)
    bins = result["prediction_quality"]["probability_bins"]
    assert sum(item["count"] for item in bins) == 2
    assert bins[3]["mean_predicted_probability"] == pytest.approx(.7)
    assert bins[3]["observed_positive_fraction"] == pytest.approx(.5)
    assert result["evidence_type"] == "MODEL_COUNTERFACTUAL"
    flat_obs = observed(ts=4_000_000_000)
    flat_group, = group_teacher_proposals([TeacherProposal(flat_obs, "EMA", 1, "cross")])
    flat_decision = CandidateDecision(flat_group,
        EntryScore("NO_TRADE", .5, 0., True, "NET_EDGE_TOO_SMALL"), False)
    flat_label = GroupOutcome(flat_group.event_id, flat_obs.decision_ts_ns, flat_obs.source_id,
        flat_obs.fold_id, "X1", "native", "base", flat_obs.decision_ts_ns + 1, None, 0.)
    with_breakeven = summarize_candidate_quality(
        (*decisions, flat_decision), (*labels, flat_label))
    assert with_breakeven["rejected"]["breakeven_complete_outcomes"] == 1
    assert with_breakeven["rejected"]["negative_complete_outcomes"] == 1
    assert with_breakeven["rejected"]["average_negative_net_bps"] == -2.
    from dataclasses import replace
    from rles_route_a.teacher_research.meta_evaluation import paired_cost_candidate_quality
    pair = paired_cost_candidate_quality(tuple(decisions), (labels[0],),
        (replace(labels[0], cost_scenario_id="stress", net_return_bps=-3.),))
    assert pair["paired_complete"] == 1 and pair["positive_under_both"] == 0
    assert pair["base_mean_net_bps"] == 5. and pair["stress_mean_net_bps"] == -3.
    with pytest.raises(ValueError, match="same modeled path"):
        paired_cost_candidate_quality(tuple(decisions), (labels[0],),
            (replace(labels[0], cost_scenario_id="stress", label_end_ts_ns=999),))
    no_pairs = summarize_candidate_quality(
        (replace(decisions[0], score=None, selected=False), decisions[2]), (labels[0], labels[2]))
    assert no_pairs["prediction_quality"]["paired_complete_scored"] == 0
    assert no_pairs["prediction_quality"]["unscored_candidates"] == 1
    assert no_pairs["prediction_quality"]["brier_score"] is None
    assert all(item["observed_positive_fraction"] is None
               for item in no_pairs["prediction_quality"]["probability_bins"])
    with pytest.raises(ValueError, match="cannot mix"):
        summarize_candidate_quality(tuple(decisions),
            (labels[0], replace(labels[1], cost_scenario_id="stress"), labels[2]))
    with pytest.raises(ValueError, match="causal candidate"):
        summarize_candidate_quality(tuple(decisions),
            (labels[0], replace(labels[1], source_identity="different"), labels[2]))


def test_quality_keeps_hard_risk_override_out_of_model_rejection(observed):
    obs = observed(ts=10_000_000_000)
    group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
    decision = CandidateDecision(
        group, EntryScore("LONG", .8, 8., True, "POSITIVE_NET_EDGE"), False,
        "HARD_RISK_OVERRIDE")
    outcome = GroupOutcome(group.event_id, obs.decision_ts_ns, obs.source_id, obs.fold_id,
        "X1", "native", "base", obs.decision_ts_ns + 1, None, 8.)

    result = summarize_candidate_quality((decision,), (outcome,))

    assert result["accepted"]["candidates"] == 1
    assert result["rejected"]["candidates"] == 0
    assert result["accepted_blocked_by_override"]["mean_net_bps"] == 8.
    assert result["native_selected"]["candidates"] == 0



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
    assert all(item["all_windows_improved"] is None for item in result["window_consistency"])
    with pytest.raises(ValueError, match="duplicate"):
        compare_preregistered_native_runs(config, (run, run))
    with pytest.raises(ValueError, match="M2.2"):
        compare_preregistered_native_runs(config, ({**run, "native_order_verification": {}},))


def test_window_consistency_does_not_infer_missing_or_tied_improvements():
    from rles_route_a.teacher_research.meta_evaluation import compare_preregistered_native_runs
    config = {"teachers": ["EMA"], "controls": ["FLAT"], "cost_scenarios": {"base": {}},
              "intervals": {"native_screen": [["a", "b"], ["b", "c"]]}}
    def row(strategy, window, final):
        return {"strategy_id": strategy, "scenario": "base", "window": window,
                "cases": (), "equity_curve": (1000., final),
                "native_order_verification": {"barriers": 2, "authority": "retained_m22_session_snapshots"}}
    rows = (row("META_X1", ("a", "b"), 1002.), row("FLAT", ("a", "b"), 1000.),
            row("META_X1", ("b", "c"), 1000.), row("FLAT", ("b", "c"), 1000.))
    result = compare_preregistered_native_runs(config, rows)
    flat = next(x for x in result["window_consistency"] if x["baseline"] == "FLAT")
    ema = next(x for x in result["window_consistency"] if x["baseline"] == "EMA")
    assert flat["positive_improvement_windows"] == 1 and flat["all_windows_improved"] is False
    assert ema["status"] == "INCOMPLETE" and ema["all_windows_improved"] is None


def test_cost_resilience_does_not_treat_flat_or_missing_as_profit():
    from rles_route_a.teacher_research.meta_evaluation import compare_preregistered_native_runs
    config = {"teachers": [], "controls": ["FLAT"], "cost_scenarios": {"base": {}, "stress": {}},
              "intervals": {"native_screen": [["a", "b"]]}}
    base = {"strategy_id": "META_X1", "scenario": "base", "window": ("a", "b"),
            "cases": (), "equity_curve": (1000., 1000.),
            "native_order_verification": {"barriers": 2, "authority": "retained_m22_session_snapshots"}}
    result = compare_preregistered_native_runs(config, (base,))
    assert result["cost_resilience"][0]["status"] == "INCOMPLETE"
    stress = {**base, "scenario": "stress"}
    report = compare_preregistered_native_runs(config, (base, stress))["cost_resilience"][0]
    assert report["status"] == "NO_COMPLETED_TRADE"
    assert report["both_positive_after_costs"] is None
