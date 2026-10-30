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
                 else EntryScore("LONG", .7, 5., True, "POSITIVE_NET_EDGE"))
        decisions.append(CandidateDecision(group, score, i == 0))
        labels.append(GroupOutcome(group.event_id, obs.decision_ts_ns, obs.source_id, obs.fold_id,
             "X1", "native", "base", obs.decision_ts_ns+1 if i != 2 else None,
             obs.decision_ts_ns+2 if i == 2 else None, 5. if i == 0 else (-2. if i == 1 else None),
             f"replay-{i}",
             outcome_status="RIGHT_CENSORED" if i == 2 else "COMPLETE",
             gross_return_bps=(10. if i == 0 else 0. if i == 1 else None),
             fees_bps=(2. if i == 0 else 1. if i == 1 else None),
             slippage_bps=(1. if i == 0 else 1. if i == 1 else None),
             funding_cashflow_bps=(-2. if i == 0 else 0. if i == 1 else None)))
    result = summarize_candidate_quality(tuple(decisions), tuple(labels))
    assert result["accepted"]["mean_net_bps"] == 5.
    assert result["rejected"]["mean_net_bps"] == -2.
    assert result["accepted"]["candidates"] == 2
    assert result["accepted"]["without_outcome"] == 1
    assert result["accepted"]["complete_coverage"] == .5
    assert result["accepted"]["outcome_status_counts"] == {
        "COMPLETE": 1, "RIGHT_CENSORED": 1,
    }
    assert result["accepted"]["incomplete_outcome_status_counts"] == {
        "RIGHT_CENSORED": 1,
    }
    assert result["accepted"]["decomposed_complete_outcomes"] == 1
    assert result["accepted"]["mean_gross_return_bps"] == 10.
    assert result["accepted"]["mean_fees_bps"] == 2.
    assert result["accepted"]["mean_slippage_bps"] == 1.
    assert result["accepted"]["mean_funding_cashflow_bps"] == -2.
    assert result["accepted_not_native_selected"]["candidates"] == 1
    assert result["accepted_not_native_selected"]["without_outcome"] == 1
    assert result["native_selected"]["candidates"] == 1
    assert result["rejected"]["without_outcome"] == 0
    assert result["rejected"]["complete_coverage"] == 1.
    assert result["rejected"]["incomplete_outcome_status_counts"] == {}
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
    assert result["quality_by_teacher"]["EMA"]["rejection_disposition_counts"] == {
        "REJECTED_BY_ARBITRATION": 1, "REJECTED_OUT_OF_SUPPORT": 1}
    assert result["quality_by_side"]["long"]["disposition_counts"]["ACCEPTED_SELECTED"] == 1
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
        flat_obs.fold_id, "X1", "native", "base", flat_obs.decision_ts_ns + 1, None, 0., "replay-flat",
        gross_return_bps=0., fees_bps=0., slippage_bps=0., funding_cashflow_bps=0.)
    with_breakeven = summarize_candidate_quality(
        (*decisions, flat_decision), (*labels, flat_label))
    assert with_breakeven["rejected"]["breakeven_complete_outcomes"] == 1
    assert with_breakeven["rejected"]["negative_complete_outcomes"] == 1
    assert with_breakeven["rejected"]["average_negative_net_bps"] == -2.
    from dataclasses import replace
    from rles_route_a.teacher_research.meta_evaluation import paired_cost_candidate_quality
    stress_complete = replace(labels[0], cost_scenario_id="stress", net_return_bps=-3.,
                               fees_bps=5., slippage_bps=8., funding_cashflow_bps=0.)
    stress_censored = replace(labels[2], cost_scenario_id="stress")
    pair = paired_cost_candidate_quality(tuple(decisions), (labels[0], labels[2]),
                                         (stress_complete, stress_censored))
    assert pair["paired_complete"] == 1 and pair["positive_under_both"] == 0
    assert pair["base_mean_net_bps"] == 5. and pair["stress_mean_net_bps"] == -3.
    assert pair["model_accepted_candidates"] == 2
    assert pair["model_accepted_paired_complete"] == 1
    assert pair["model_accepted_without_complete_outcome"] == 1
    assert pair["native_selected_candidates"] == 1
    assert pair["native_selected_base_mean_net_bps"] == 5.
    assert pair["outcomes"][1]["native_selected"] is False
    assert pair["base_mean_gross_bps"] == pair["stress_mean_gross_bps"] == 10.
    assert pair["base_mean_fees_bps"] == 2. and pair["stress_mean_fees_bps"] == 5.
    assert pair["base_mean_slippage_bps"] == 1. and pair["stress_mean_slippage_bps"] == 8.
    assert pair["base_mean_funding_bps"] == -2. and pair["stress_mean_funding_bps"] == 0.
    with pytest.raises(ValueError, match="same modeled path"):
        paired_cost_candidate_quality(tuple(decisions), (labels[0], labels[2]),
            (replace(stress_complete, label_end_ts_ns=labels[0].label_end_ts_ns + 1),
             stress_censored))
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


def test_screen_prediction_quality_separates_executable_and_account_blocked_candidates(observed):
    from dataclasses import replace

    rows = []
    labels = []
    for index, (pending, probability, expected, actual) in enumerate((
            ((), .8, 4., 3.), (("native-order",), .9, 5., -2.)), start=1):
        obs = replace(observed(ts=index * 1_000_000_000), pending_order_ids=pending)
        group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
        score = EntryScore("LONG" if not pending else "NO_TRADE", probability,
                           expected, True, "POSITIVE_NET_EDGE" if not pending else "PENDING_ORDER")
        rows.append(CandidateDecision(group, score, not pending))
        labels.append(GroupOutcome(
            group.event_id, obs.decision_ts_ns, obs.source_id, obs.fold_id,
            "X1", "native", "base", obs.decision_ts_ns + 1, None, actual,
            f"screen-{index}", gross_return_bps=actual, fees_bps=0.,
            slippage_bps=0., funding_cashflow_bps=0.,
        ))

    report = summarize_candidate_quality(tuple(rows), tuple(labels))
    assert report["prediction_quality"]["paired_complete_scored"] == 2
    assert report["prediction_quality"]["uniqueness_weighted_metrics"]["net_prediction_rmse_bps"] >= 0.
    assert report["prediction_quality"]["net_return_bins"]
    by_account = report["prediction_quality_by_account_state"]
    assert by_account["executable_flat"]["paired_complete_scored"] == 1
    assert by_account["position_or_pending_order_blocked"]["paired_complete_scored"] == 1
    assert by_account["position_or_pending_order_blocked"]["scored_outcome_status_counts"] == {
        "COMPLETE": 1,
    }
    assert by_account["executable_flat"]["independence_status"] == "UNVERIFIED_BY_GENERIC_SUMMARIZER"
    assert by_account["executable_flat"]["thresholds_adjusted"] is False


def test_quality_keeps_hard_risk_override_out_of_model_rejection(observed):
    obs = observed(ts=10_000_000_000)
    group, = group_teacher_proposals([TeacherProposal(obs, "EMA", 1, "cross")])
    decision = CandidateDecision(
        group, EntryScore("LONG", .8, 8., True, "POSITIVE_NET_EDGE"), False,
        "HARD_RISK_OVERRIDE")
    outcome = GroupOutcome(group.event_id, obs.decision_ts_ns, obs.source_id, obs.fold_id,
        "X1", "native", "base", obs.decision_ts_ns + 1, None, 8., "replay-risk",
        gross_return_bps=10., fees_bps=1., slippage_bps=1., funding_cashflow_bps=0.)

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
           "account_metrics": {"total_fee": 0., "total_slippage_cost": 0., "total_funding": 0.},
           "native_order_verification": {"barriers": 10, "authority": "retained_m22_session_snapshots"}}
    result = compare_preregistered_native_runs(config, (run,))
    assert result["expected_cells"] == 24 and result["available_cells"] == 1
    assert result["status"] == "INCOMPLETE" and result["qualification"] == "NOT_ESTABLISHED"
    assert all(item["all_windows_improved"] is None for item in result["window_consistency"])
    with pytest.raises(ValueError, match="duplicate"):
        compare_preregistered_native_runs(config, (run, run))
    with pytest.raises(ValueError, match="M2.2"):
        compare_preregistered_native_runs(config, ({**run, "native_order_verification": {}},))
    with pytest.raises(ValueError, match="requires complete native account costs"):
        compare_preregistered_native_runs(config, ({k:v for k,v in run.items()
                                                    if k != "account_metrics"},))


def test_native_comparison_includes_the_executed_original_eight_teacher_control():
    from rles_route_a.teacher_research.meta_evaluation import compare_preregistered_native_runs

    config = {"teachers": [], "controls": ["ORIGINAL_8_X1"],
              "cost_scenarios": {"base": {}}, "intervals": {"native_screen": [["a", "b"]]}}
    run = {"strategy_id": "ORIGINAL_8_X1", "scenario": "base", "window": ("a", "b"),
           "cases": (), "equity_curve": (1000., 1000.),
           "account_metrics": {"total_fee": 0., "total_slippage_cost": 0., "total_funding": 0.},
           "native_order_verification": {"barriers": 2,
               "authority": "retained_m22_session_snapshots"}}

    result = compare_preregistered_native_runs(config, (run,))

    assert result["expected_cells"] == 2 and result["available_cells"] == 1
    original = next(row for row in result["rows"] if row["strategy_id"] == "ORIGINAL_8_X1")
    assert original["status"] == "AVAILABLE"


def test_complete_preregistered_meta_study_matrix_accepts_all_52_cells():
    from rles_route_a.teacher_research.meta_evaluation import compare_preregistered_native_runs

    teachers = [f"T{i:02d}" for i in range(1, 9)]
    controls = ["FLAT", "FIXED_LONG_0.3", "FIXED_SHORT_0.3", "ORIGINAL_8_X1"]
    scenarios = ("base", "stress")
    windows = (("a", "b"), ("b", "c"))
    config = {"teachers": teachers, "controls": controls,
              "cost_scenarios": {name: {} for name in scenarios},
              "intervals": {"native_screen": [list(window) for window in windows]}}
    strategies = (*teachers, "META_X1", *controls)
    runs = tuple({"strategy_id": strategy, "scenario": scenario, "window": window,
                  "cases": (), "equity_curve": (1000., 1000.),
                  "account_metrics": {"total_fee": 0., "total_slippage_cost": 0.,
                                      "total_funding": 0.},
                  "native_order_verification": {"barriers": 2,
                      "authority": "retained_m22_session_snapshots"}}
                 for strategy in strategies for scenario in scenarios for window in windows)

    result = compare_preregistered_native_runs(config, runs)

    assert result["expected_cells"] == result["available_cells"] == 52
    assert result["status"] == "COMPLETE"
    assert len(result["comparisons"]) == 48
    assert all(row["status"] == "COMPLETE" for row in result["window_consistency"])
    assert all(row["status"] == "NO_COMPLETED_TRADE" for row in result["cost_resilience"])
    assert result["qualification"] == "NOT_ESTABLISHED"
    assert result["preregistered_controls_outside_active_matrix"] == ["ORIGINAL_8_X1"]


def test_unimplemented_control_is_reported_and_unknown_control_fails_closed():
    from rles_route_a.teacher_research.meta_evaluation import (
        compare_preregistered_native_runs, registered_control_scope,
    )

    config = {"teachers": [], "controls": ["ORIGINAL_8_X1", "FLAT"],
              "cost_scenarios": {"base": {}}, "intervals": {"native_screen": [["a", "b"]]}}
    active, planned_only = registered_control_scope(config)
    assert active == ("FLAT",) and planned_only == ("ORIGINAL_8_X1",)
    report = compare_preregistered_native_runs(config, ())
    assert report["preregistered_controls_outside_active_matrix"] == ["ORIGINAL_8_X1"]

    with pytest.raises(ValueError, match="unrecognized preregistered control"):
        registered_control_scope({**config, "controls": ["HIDDEN_CONTROL"]})
    with pytest.raises(ValueError, match="unique"):
        registered_control_scope({**config, "controls": ["FLAT", "FLAT"]})


def test_native_comparison_rejects_mixed_registered_source_receipts():
    from rles_route_a.teacher_research.meta_evaluation import compare_preregistered_native_runs

    window = ("1970-01-01T00:01:00Z", "1970-01-01T00:02:00Z")
    config = {"protocol_id": "synthetic-protocol", "teachers": [], "controls": ["FLAT"],
              "cost_scenarios": {"base": {}}, "intervals": {"native_screen": [list(window)]}}

    def run(strategy, *, source_marker="b"*64, with_receipt=True):
        row = {"strategy_id": strategy, "scenario": "base", "window": window,
               "cases": (), "equity_curve": (1000., 1000.),
               "account_metrics": {"total_fee": 0., "total_slippage_cost": 0., "total_funding": 0.},
               "native_order_verification": {"barriers": 1,
                   "authority": "retained_m22_session_snapshots"}}
        if with_receipt:
            start = 60_000_000_000
            row["registered_screen_split"] = {
                "status": "PASS_REGISTERED_TEMPORAL_SEPARATION",
                "independent_trials_claimed": False,
                "protocol_id": "synthetic-protocol",
                "preregistration_sha256": "a"*64,
                "source_marker_sha256": source_marker,
                "market_snapshot_sha256": "c"*64,
                "import_manifest_sha256": "d"*64,
                "strategy_id": strategy, "scenario": "base", "seed": 7,
                "screen_interval_ns": [start, start + 60_000_000_000],
                "model_available_after_ns": start if strategy == "META_X1" else None,
            }
        return row

    verified = compare_preregistered_native_runs(config, (run("META_X1"), run("FLAT")))
    assert verified["run_receipt_status"] == "VERIFIED_SHARED_PROTOCOL_SOURCE_AND_SEED"
    with pytest.raises(ValueError, match="mixes preregistrations, sources, or seeds"):
        compare_preregistered_native_runs(
            config, (run("META_X1"), run("FLAT", source_marker="e"*64))
        )
    with pytest.raises(ValueError, match="mix runs with and without"):
        compare_preregistered_native_runs(config, (run("META_X1"), run("FLAT", with_receipt=False)))


def test_native_comparison_requires_trade_case_cashflows_to_match_account():
    from rles_route_a.teacher_research.meta_evaluation import compare_preregistered_native_runs

    config = {"teachers": ["EMA"], "controls": ["FLAT"],
              "cost_scenarios": {"base": {}},
              "intervals": {"native_screen": [["a", "b"]]}}
    case = NativeTradeCase1m(
        trade_case_id="native", signal_id="signal", teacher_id="EMA", exit_policy_id="X1",
        scenario_id="base", block_id="a_b", side=1, status="ORDINARY_COMPLETE",
        entry_decision_ts_ns=1, entry_signal_price=100., entry_fill_ts_ns=2,
        entry_fill_price=100., entry_notional=1000., exit_signal_price=101.,
        exit_fill_ts_ns=4, exit_fill_price=101., fees=2., fee_currency="USDT",
        equity_currency="USDT", slippage_cost=1., funding=0., realized_pnl=8.,
        equity_start=1000., equity_end=1008., mfe_bps=100., mae_bps=-10.,
        holding_minutes=2., exit_reason="X1", economic_status="SCENARIO_ONLY")
    run = {"strategy_id": "EMA", "scenario": "base", "window": ("a", "b"),
           "cases": (case,), "equity_curve": (1000., 1008.),
           "account_metrics": {"total_fee": 2., "total_slippage_cost": 1., "total_funding": 0.},
           "native_order_verification": {"barriers": 2,
               "authority": "retained_m22_session_snapshots"}}

    report = compare_preregistered_native_runs(config, (run,))
    metrics = next(row["metrics"] for row in report["rows"] if row["status"] == "AVAILABLE")
    assert metrics["native_case_cashflow_reconciliation"]["status"] == "RECONCILED"
    assert metrics["native_case_cashflow_reconciliation"]["case_totals"]["net_pnl"] == 8.

    flat = {"strategy_id": "FLAT", "scenario": "base", "window": ("a", "b"),
            "cases": (), "equity_curve": (1000., 1000.),
            "account_metrics": {"total_fee": 0., "total_slippage_cost": 0., "total_funding": 0.},
            "native_order_verification": {"barriers": 2,
                "authority": "retained_m22_session_snapshots"}}
    meta = {**run, "strategy_id": "META_X1"}
    economic_comparison = compare_preregistered_native_runs(config, (meta, run, flat))
    meta_vs_flat = next(row for row in economic_comparison["comparisons"]
                        if row["baseline"] == "FLAT")
    assert meta_vs_flat["completed_trade_count_delta"] == 1
    assert meta_vs_flat["completed_trade_fees_delta"] == 2.
    assert meta_vs_flat["completed_trade_slippage_delta"] == 1.
    assert meta_vs_flat["completed_trade_net_pnl_delta"] == 8.
    assert meta_vs_flat["average_holding_minutes_delta"] is None
    assert meta_vs_flat["long_count_delta"] == 1
    assert meta_vs_flat["long_net_pnl_delta"] == 8.
    assert meta_vs_flat["long_win_rate_delta"] is None
    meta_vs_ema = next(row for row in economic_comparison["comparisons"]
                       if row["baseline"] == "EMA")
    assert meta_vs_ema["average_holding_minutes_delta"] == 0.
    assert meta_vs_ema["long_win_rate_delta"] == 0.

    bad_run = {**run, "account_metrics": {**run["account_metrics"], "total_fee": 2.1}}
    with pytest.raises(ValueError, match="do not reconcile"):
        compare_preregistered_native_runs(config, (bad_run,))


def test_window_consistency_does_not_infer_missing_or_tied_improvements():
    from rles_route_a.teacher_research.meta_evaluation import compare_preregistered_native_runs
    config = {"teachers": ["EMA"], "controls": ["FLAT"], "cost_scenarios": {"base": {}},
              "intervals": {"native_screen": [["a", "b"], ["b", "c"]]}}
    def row(strategy, window, final):
        cases = ()
        if final != 1000.:
            cases = (NativeTradeCase1m(
                trade_case_id=f"{strategy}-{window[0]}", signal_id=f"{strategy}-{window[0]}",
                teacher_id=strategy, exit_policy_id="X1", scenario_id="base",
                block_id=f"{window[0]}_{window[1]}", side=1, status="ORDINARY_COMPLETE",
                entry_decision_ts_ns=1, entry_signal_price=100., entry_fill_ts_ns=2,
                entry_fill_price=100., entry_notional=1000., exit_signal_price=100.2,
                exit_fill_ts_ns=4, exit_fill_price=100.2, fees=0., fee_currency="USDT",
                equity_currency="USDT", slippage_cost=0., funding=0., realized_pnl=final-1000.,
                equity_start=1000., equity_end=final, mfe_bps=20., mae_bps=0.,
                holding_minutes=2., exit_reason="X1", economic_status="SCENARIO_ONLY"),)
        return {"strategy_id": strategy, "scenario": "base", "window": window,
                "cases": cases, "equity_curve": (1000., final),
                "account_metrics": {"total_fee": 0., "total_slippage_cost": 0., "total_funding": 0.},
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
            "account_metrics": {"total_fee": 0., "total_slippage_cost": 0., "total_funding": 0.},
            "native_order_verification": {"barriers": 2, "authority": "retained_m22_session_snapshots"}}
    result = compare_preregistered_native_runs(config, (base,))
    assert result["cost_resilience"][0]["status"] == "INCOMPLETE"
    stress = {**base, "scenario": "stress"}
    report = compare_preregistered_native_runs(config, (base, stress))["cost_resilience"][0]
    assert report["status"] == "NO_COMPLETED_TRADE"
    assert report["both_positive_after_costs"] is None
