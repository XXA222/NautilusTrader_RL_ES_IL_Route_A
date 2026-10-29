from dataclasses import replace

import pytest

from rles_route_a.evaluation.teacher_case_report_1m import NativeTradeCase1m
from rles_route_a.teacher_research.meta_gated_teacher import MetaGatedTeacher
from rles_route_a.teacher_research.native_meta_funnel import MetaNativeFunnelAdapter
from test_candidate_collector import observation
from test_meta_gated_teacher import Recovery, trained_policy


class NativeHook:
    cases = ()
    def capture_pre_decision(self, obs, action, visible, teacher):
        return teacher.last_signal, teacher.recipe.indicator_recipe_id
    def capture_post_decision(self, *args):
        return args
    def build_native_cases(self, records):
        return self.cases
    def commit(self, records, cases):
        return records, cases


def test_meta_funnel_projects_selected_signal_to_native_case_hook():
    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    hook = MetaNativeFunnelAdapter(teacher=teacher, native_hook=NativeHook())
    first = observation(10)
    action = teacher.advise(first)
    signal, recipe = hook.capture_pre_decision(first, action, {}, teacher)
    assert signal.raw_signal == "NONE" and recipe == "route_a.meta_gated_x1.v1"
    second = observation(11, ema_9=101.)
    action = teacher.advise(second)
    signal, _ = hook.capture_pre_decision(second, action, {}, teacher)
    assert signal.raw_signal == "LONG_ENTRY" and signal.reason_code == "EMA_CROSS_LONG"
    third = observation(12, ema_9=99., exposure=.3)
    action = teacher.advise(third)
    signal, _ = hook.capture_pre_decision(third, action, {}, teacher)
    assert signal.raw_signal == "EXIT_LONG" and signal.reason_code == "EMA_REVERSE_CROSS"
    with pytest.raises(ValueError, match="corresponding"):
        hook.capture_pre_decision(second, action, {}, MetaGatedTeacher(trained_policy(), recovery=Recovery()))


def test_meta_native_cases_retain_exact_selected_candidate_provenance():
    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    hook = NativeHook()
    adapter = MetaNativeFunnelAdapter(teacher=teacher, native_hook=hook)
    teacher.advise(observation(10))
    entry = observation(11, ema_9=101.)
    teacher.advise(entry)
    selected = next(item.group for item in teacher.decisions[-1].candidates if item.selected)
    case = NativeTradeCase1m(
        trade_case_id="native-case", signal_id="signal", teacher_id=teacher.expert_id,
        exit_policy_id="X1", scenario_id="base", block_id="window", side=1,
        status="ORDINARY_COMPLETE", entry_decision_ts_ns=entry.decision_ts_ns,
        entry_signal_price=100., entry_fill_ts_ns=entry.decision_ts_ns+1,
        entry_fill_price=100., entry_notional=30., exit_signal_price=101.,
        exit_fill_ts_ns=entry.decision_ts_ns+2, exit_fill_price=101., fees=.1,
        fee_currency="USDT", equity_currency="USDT", slippage_cost=.01,
        funding=0., realized_pnl=.19, equity_start=1000., equity_end=1000.19,
        mfe_bps=100., mae_bps=0., holding_minutes=1., exit_reason="X1",
        economic_status="SCENARIO_ONLY")
    hook.cases = (case,)
    result, = adapter.build_native_cases(())
    assert result.candidate_event_id == selected.event_id
    assert result.candidate_teacher_ids == ("T01_EMA_9_21",)

    from rles_route_a.teacher_research.native_reconciliation import reconcile_selected_native_cases
    decision = teacher.decisions[-1].candidates
    mismatch = reconcile_selected_native_cases(decision, (replace(result, side=-1),),
        lambda *_: pytest.fail("mismatched native case must not reach price replay"),
        execution_recipe="native", scenario="base")
    assert mismatch["cases"][0]["status"] == "NOT_COMPARABLE"

    hook.cases = (replace(case, candidate_event_id="f"*64,
                          candidate_teacher_ids=("T02_ROC_20_60",)),)
    with pytest.raises(ValueError, match="conflicts"):
        adapter.build_native_cases(())
