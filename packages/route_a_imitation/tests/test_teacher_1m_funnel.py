from __future__ import annotations

import csv
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from rles_route_a.evaluation.teacher_case_report_1m import (
    NativeTradeCase1m,
    summarize_native_trade_cases_1m,
    write_teacher_research_artifacts_1m,
)
from rles_route_a.evaluation.teacher_funnel_1m import (
    TeacherDecisionRecord,
    TeacherExecutionRecord,
    TeacherPlannerRecord,
    TeacherPreDecision,
    build_teacher_funnel_1m,
    group_teacher_funnels_1m,
    validate_staged_teacher_decision,
)
from rles_route_a.indicators_1m import MINUTE_NS


def _pre(
    minute: int,
    *,
    raw: str = "NONE",
    position: float = 0.0,
    risk_veto: str | None = None,
    order_veto: str | None = None,
    teacher: str = "T01_EMA_9_21",
    exit_policy: str = "X1_EMA_REVERSE_CROSS_V1",
    scenario: str = "base",
) -> TeacherPreDecision:
    close_time = minute * MINUTE_NS
    return TeacherPreDecision(
        teacher_id=teacher,
        recipe_id="route_a_1m_multiteacher_v1",
        fold_id="synthetic-block",
        source_identity="D0_SYNTHETIC",
        decision_ts_ns=close_time + 1_000_000,
        latest_1m_close_ts_ns=close_time,
        latest_1m_available_at_ns=close_time + 1_000_000,
        bar_status="READY",
        indicator_values={"ema_9": 101.0, "ema_21": 100.0},
        raw_signal=raw,  # type: ignore[arg-type]
        raw_signal_reason="SYNTHETIC",
        position_before=position,
        native_pending_orders=1 if order_veto else 0,
        risk_veto_reason=risk_veto,
        order_state_veto_reason=order_veto,
        exit_policy_id=exit_policy,
        scenario_id=scenario,
        economic_status="SYNTHETIC_ENGINEERING_ONLY",
        block_id="synthetic-block",
    )


def _planner(intent: str, *, order: str | None = None) -> TeacherPlannerRecord:
    return TeacherPlannerRecord(
        advice_intent=intent,  # type: ignore[arg-type]
        requested_exposure=0.3 if intent == "OPEN_LONG" else -0.3 if intent == "OPEN_SHORT" else 0.0,
        risk_adjusted_target=0.3 if intent == "OPEN_LONG" else -0.3 if intent == "OPEN_SHORT" else 0.0,
        planner_target=0.3 if intent == "OPEN_LONG" else -0.3 if intent == "OPEN_SHORT" else 0.0,
        planner_status="SUBMIT" if order else "NOOP",
        planner_reason="SYNTHETIC_SUBMIT" if order else "SYNTHETIC_NOOP",
        submitted_order_ids=() if order is None else (order,),
    )


def _execution(
    minute: int, *, order: str | None = None, exposure: float = 0.0,
    case_status: str = "NONE", fee: float = 0.0,
) -> TeacherExecutionRecord:
    return TeacherExecutionRecord(
        execution_ts_ns=minute * MINUTE_NS + 2_000_000,
        native_filled_order_ids=() if order is None else (order,),
        native_exposure_after=exposure,
        fee=fee,
        fee_currency="USDT",
        equity_currency="USDT",
        slippage_cost=0.1 if order else 0.0,
        funding=0.0,
        account_equity=10_000.0,
        mfe_bps=20.0,
        mae_bps=-10.0,
        position_age_minutes=5.0,
        case_status=case_status,  # type: ignore[arg-type]
        case_id=None if case_status == "NONE" else f"case-{minute}",
    )


def test_zero_signal_funnel_is_explicit_not_a_trade():
    rows = tuple(
        TeacherDecisionRecord(_pre(minute), _planner("FLAT")) for minute in range(1, 4)
    )
    funnel = build_teacher_funnel_1m(rows)
    assert funnel.decisions == 3
    assert funnel.eligible_decision_bars == 3
    assert funnel.raw_long_signals == funnel.raw_short_signals == 0
    assert funnel.submitted_entries == funnel.filled_entries == 0
    assert funnel.unfilled_or_missed_cases == 0


def test_signal_many_zero_fill_risk_veto_and_pending_veto_are_distinct():
    no_fill = TeacherDecisionRecord(
        _pre(1, raw="LONG_ENTRY"),
        _planner("OPEN_LONG", order="entry-1"),
        _execution(1, exposure=0.0),
    )
    risk = TeacherDecisionRecord(
        _pre(2, raw="LONG_ENTRY", risk_veto="RISK_HARD_VETO"), _planner("FLAT")
    )
    pending = TeacherDecisionRecord(
        _pre(3, raw="SHORT_ENTRY", order_veto="PENDING_ACK"), _planner("FLAT")
    )
    funnels = {row.side: row for row in group_teacher_funnels_1m((no_fill, risk, pending))}
    long_funnel = funnels["LONG"]
    short_funnel = funnels["SHORT"]
    assert long_funnel.raw_long_signals == 2
    assert long_funnel.entry_signals_while_flat == 1
    assert long_funnel.submitted_entries == 1
    assert long_funnel.filled_entries == 0
    assert long_funnel.unfilled_or_missed_cases == 1
    assert long_funnel.veto_counts == {"RISK_HARD_VETO": 1}
    assert short_funnel.raw_short_signals == 1
    assert short_funnel.entry_signals_while_flat == 0
    assert short_funnel.veto_counts == {"PENDING_ACK": 1}


def test_submitted_fill_exit_and_case_counts_match_staged_native_facts():
    entry = TeacherDecisionRecord(
        _pre(1, raw="LONG_ENTRY"),
        _planner("OPEN_LONG", order="entry"),
        _execution(1, order="entry", exposure=0.3, fee=0.5),
    )
    hold = TeacherDecisionRecord(_pre(2, position=0.3), _planner("HOLD"))
    exit_row = TeacherDecisionRecord(
        _pre(3, raw="EXIT_LONG", position=0.3),
        _planner("EXIT", order="exit"),
        _execution(3, order="exit", exposure=0.0, case_status="ORDINARY_COMPLETE", fee=0.5),
    )
    funnel = build_teacher_funnel_1m((entry, hold, exit_row))
    assert funnel.submitted_entries == funnel.filled_entries == 1
    assert funnel.submitted_exits == funnel.filled_exits == 1
    assert funnel.ordinary_flat_to_flat_cases == 1
    assert funnel.exit_signals == 1


def test_later_native_fill_completes_the_original_entry_order_count():
    submitted_pending = TeacherDecisionRecord(
        _pre(1, raw="LONG_ENTRY"),
        _planner("OPEN_LONG", order="entry"),
        _execution(1),
    )
    later_fill = TeacherDecisionRecord(
        _pre(2, position=0.0),
        replace(_planner("FLAT"), related_existing_order_ids=("entry",)),
        _execution(2, order="entry", exposure=0.3),
    )
    funnels = {row.side: row for row in group_teacher_funnels_1m((submitted_pending, later_fill))}
    funnel = funnels["LONG"]
    assert funnel.submitted_entries == funnel.filled_entries == 1
    assert funnel.unfilled_or_missed_cases == 0


def test_funnels_split_long_and_short_by_teacher_exit_scenario_and_block():
    long_row = TeacherDecisionRecord(
        _pre(1, raw="LONG_ENTRY"),
        _planner("OPEN_LONG", order="long-entry"),
        _execution(1, order="long-entry", exposure=0.3),
    )
    short_row = TeacherDecisionRecord(
        _pre(2, raw="SHORT_ENTRY"),
        _planner("OPEN_SHORT", order="short-entry"),
        _execution(2, order="short-entry", exposure=-0.3),
    )
    long_funnel, short_funnel = group_teacher_funnels_1m((long_row, short_row))
    assert long_funnel.side == "LONG"
    assert long_funnel.raw_long_signals == long_funnel.submitted_entries == 1
    assert short_funnel.side == "SHORT"
    assert short_funnel.raw_short_signals == short_funnel.submitted_entries == 1


def test_predecision_cannot_contain_future_execution_or_unproven_fill():
    pre = _pre(1, raw="LONG_ENTRY")
    planner = _planner("OPEN_LONG", order="entry")
    with pytest.raises(ValueError, match="cannot precede or equal"):
        TeacherDecisionRecord(
            pre,
            planner,
            replace(_execution(1, order="entry", exposure=0.3), execution_ts_ns=pre.decision_ts_ns),
        )
    with pytest.raises(ValueError, match="submitted order provenance"):
        TeacherDecisionRecord(
            pre,
            planner,
            _execution(2, order="different", exposure=0.3),
        )


def test_staged_row_must_match_the_same_native_planner_fill_and_equity_facts():
    ts = 120_000_000_000
    observation = PointInTimeObservation(
        episode_id="native-check",
        decision_ts_ns=ts,
        market={"ema_9": 101.0},
        feature_available_at_ns={"ema_9": ts - 1},
        account={"net_exposure": 0.0},
        exact_market_clocks_ns=(("latest_1m_close_ts_ns", ts - MINUTE_NS),),
    )
    advice = ExpertAdvice("T01_EMA_9_21", ts, "OPEN_LONG", 0.3, ("EMA_CROSS",))
    pre = TeacherPreDecision(
        teacher_id=advice.expert_id,
        recipe_id="route_a_1m_multiteacher_v1",
        fold_id="fold",
        source_identity="source",
        decision_ts_ns=ts,
        latest_1m_close_ts_ns=ts - MINUTE_NS,
        latest_1m_available_at_ns=ts - 1,
        bar_status="READY",
        indicator_values={"ema_9": 101.0},
        raw_signal="LONG_ENTRY",
        raw_signal_reason="EMA_CROSS",
        position_before=0.0,
        native_pending_orders=0,
        risk_veto_reason=None,
        order_state_veto_reason=None,
        exit_policy_id="X1",
        scenario_id="base",
        economic_status="SCENARIO_ONLY",
        block_id="train-block-1",
    )
    planner = TeacherPlannerRecord(
        advice_intent="OPEN_LONG",
        requested_exposure=0.3,
        risk_adjusted_target=0.3,
        planner_target=0.3,
        planner_status="SUBMIT",
        planner_reason="TARGET_CHANGED",
        submitted_order_ids=("entry-1",),
    )
    execution = _execution(3, order="entry-1", exposure=0.3, fee=0.5)
    record = TeacherDecisionRecord(pre, planner, execution)
    trace = SimpleNamespace(
        risk_adjusted_target=0.3,
        planner_target=0.3,
        fee_currency="USDT",
        planner_action="SUBMIT",
        planner_reason="TARGET_CHANGED",
        submitted_order_ids=("entry-1",),
        related_existing_order_ids=(),
        pending_order_ids_after=(),
        execution_barrier_ts_ns=execution.execution_ts_ns,
        native_equity_after=execution.account_equity,
    )
    validate_staged_teacher_decision(
        record,
        observation=observation,
        advice=advice,
        gym_info={"native_funding": ()},
        barrier=SimpleNamespace(trace=trace),
        order_facts=SimpleNamespace(
            filled_order_ids=("entry-1",),
            native_policy_fills=({"commission": "0.5 USDT"},),
        ),
    )
    with pytest.raises(ValueError, match="same-step native L3/L6 facts"):
        validate_staged_teacher_decision(
            replace(record, execution=replace(execution, account_equity=9_999.0)),
            observation=observation,
            advice=advice,
            gym_info={"native_funding": ()},
            barrier=SimpleNamespace(trace=trace),
            order_facts=SimpleNamespace(
                filled_order_ids=("entry-1",),
                native_policy_fills=({"commission": "0.5 USDT"},),
            ),
        )
    with pytest.raises(ValueError, match="fee differs"):
        validate_staged_teacher_decision(
            replace(record, execution=replace(execution, fee=0.4)),
            observation=observation,
            advice=advice,
            gym_info={"native_funding": ()},
            barrier=SimpleNamespace(trace=trace),
            order_facts=SimpleNamespace(
                filled_order_ids=("entry-1",),
                native_policy_fills=({"commission": "0.5 USDT"},),
            ),
        )
    with pytest.raises(ValueError, match="funding differs"):
        validate_staged_teacher_decision(
            record,
            observation=observation,
            advice=advice,
            gym_info={"native_funding": ({"pnl_change": "-0.1 USDT"},)},
            barrier=SimpleNamespace(trace=trace),
            order_facts=SimpleNamespace(
                filled_order_ids=("entry-1",),
                native_policy_fills=({"commission": "0.5 USDT"},),
            ),
        )
    with pytest.raises(ValueError, match="unfilled submitted order lacks"):
        validate_staged_teacher_decision(
            TeacherDecisionRecord(pre, planner, _execution(3)),
            observation=observation,
            advice=advice,
            gym_info={"native_funding": ()},
            barrier=SimpleNamespace(trace=trace),
            order_facts=SimpleNamespace(
                filled_order_ids=(),
                native_policy_fills=(),
            ),
        )
def test_grouping_never_mixes_teacher_exit_or_scenario_accounts():
    rows = (
        TeacherDecisionRecord(_pre(1), _planner("FLAT")),
        TeacherDecisionRecord(_pre(1, teacher="T02_ROC_20_60"), _planner("FLAT")),
        TeacherDecisionRecord(_pre(1, scenario="stress"), _planner("FLAT")),
    )
    funnels = group_teacher_funnels_1m(rows)
    assert len(funnels) == 3
    with pytest.raises(ValueError, match="cannot mix"):
        build_teacher_funnel_1m(rows)


def _case(
    case_id: str, *, pnl: float, equity_start: float, equity_end: float,
    status: str = "ORDINARY_COMPLETE",
) -> NativeTradeCase1m:
    complete = status == "ORDINARY_COMPLETE"
    return NativeTradeCase1m(
        trade_case_id=case_id,
        signal_id=f"signal-{case_id}",
        teacher_id="T01_EMA_9_21",
        exit_policy_id="X1_EMA_REVERSE_CROSS_V1",
        scenario_id="base",
        block_id="synthetic-block",
        side=1,
        status=status,  # type: ignore[arg-type]
        entry_decision_ts_ns=MINUTE_NS,
        entry_signal_price=100.0,
        entry_fill_ts_ns=MINUTE_NS + 1,
        entry_fill_price=100.1,
        entry_notional=300.0,
        exit_signal_price=101.0 if complete else None,
        exit_fill_ts_ns=2 * MINUTE_NS if complete else None,
        exit_fill_price=100.9 if complete else None,
        fees=0.3,
        fee_currency="USDT",
        equity_currency="USDT",
        slippage_cost=0.06,
        funding=-0.01,
        realized_pnl=pnl,
        equity_start=equity_start,
        equity_end=equity_end,
        mfe_bps=100.0,
        mae_bps=-50.0,
        holding_minutes=1.0,
        exit_reason="INDICATOR_EXIT" if complete else "EPISODE_TERMINAL_FLATTEN",
        economic_status="SYNTHETIC_ENGINEERING_ONLY",
    )


def test_case_summary_retains_winner_loser_and_terminal_censor_with_costs_once():
    cases = (
        _case("win", pnl=2.0, equity_start=10_000.0, equity_end=10_002.0),
        _case("loss", pnl=-3.0, equity_start=10_002.0, equity_end=9_999.0),
        _case(
            "terminal", pnl=-1.0, equity_start=9_999.0, equity_end=9_998.0,
            status="TERMINAL_CENSORED",
        ),
    )
    summary = summarize_native_trade_cases_1m(cases)
    assert summary.total_cases == 3
    assert summary.ordinary_complete == 2
    assert (summary.winners, summary.losers, summary.terminal_censored) == (1, 1, 1)
    assert summary.gross_profit == 2.0
    assert summary.gross_loss == -3.0
    assert summary.profit_factor == pytest.approx(2 / 3)
    assert summary.fees == pytest.approx(0.9)
    assert summary.slippage_cost == pytest.approx(0.18)
    assert summary.funding == pytest.approx(-0.03)
    assert summary.net_pnl == -2.0
    assert summary.account_return == pytest.approx(9_998 / 10_000 - 1)


def test_terminal_flatten_cannot_be_labeled_ordinary_complete():
    with pytest.raises(ValueError, match="terminal flatten"):
        replace(_case("bad", pnl=1.0, equity_start=10_000, equity_end=10_001),
                exit_reason="EPISODE_TERMINAL_FLATTEN")


def test_s05_artifact_writer_reconciles_ids_and_emits_pinned_csv_json_outputs(tmp_path):
    entry = TeacherDecisionRecord(
        _pre(1, raw="LONG_ENTRY"),
        _planner("OPEN_LONG", order="entry"),
        _execution(1, order="entry", exposure=0.3),
    )
    exit_row = TeacherDecisionRecord(
        _pre(2, raw="EXIT_LONG", position=0.3),
        _planner("EXIT", order="exit"),
        _execution(2, order="exit", exposure=0.0, case_status="ORDINARY_COMPLETE"),
    )
    cases = (_case("case-2", pnl=2.0, equity_start=10_000.0, equity_end=10_002.0),)
    output = tmp_path / "native-report"
    write_teacher_research_artifacts_1m(
        output, (entry, exit_row), cases,
        expected_submitted_order_ids=("entry", "exit"),
        expected_filled_order_ids=("entry", "exit"),
        expected_case_ids=("case-2",),
    )
    funnel = json.loads((output / "teacher-funnel.json").read_text(encoding="utf-8"))
    assert funnel["execution_classification"] == "M22_VERIFIED_NATIVE_FACTS"
    assert funnel["groups"][0]["submitted_entries"] == 1
    assert funnel["groups"][0]["filled_exits"] == 1
    with (output / "decision-rows.csv").open(encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    assert tuple(row["planner_intent"] for row in rows) == ("OPEN_LONG", "EXIT")
    assert len((output / "native-trades.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    summary = json.loads((output / "case-summary.json").read_text(encoding="utf-8"))
    assert summary["economic_classifications"] == ["SYNTHETIC_ENGINEERING_ONLY"]


def test_native_accounting_mismatch_fails_before_creating_output(tmp_path):
    rows = (TeacherDecisionRecord(_pre(1), _planner("FLAT"), _execution(1)),)
    output = tmp_path / "rejected-report"
    with pytest.raises(ValueError, match="INCONSISTENT_NATIVE_ACCOUNTING"):
        write_teacher_research_artifacts_1m(
            output, rows, (),
            expected_submitted_order_ids=(),
            expected_filled_order_ids=("unobserved-fill",),
            expected_case_ids=(),
        )
    assert not output.exists()

