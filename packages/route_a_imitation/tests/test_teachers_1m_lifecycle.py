from __future__ import annotations

import pytest

from rles_route_a.contracts import NativeActionTrace, PointInTimeObservation
from rles_route_a.indicators_1m import MINUTE_NS
from rles_route_a.integration.exit_lifecycle import verify_native_flat_lifecycle
from rles_route_a.teachers.recovery import RecoveryTeacher
from rles_route_a.teachers.selector import FrozenTeacherSelector
from rles_route_a.teachers_1m import AtrTrailingExit, build_teacher_1m


DELAY_NS = 1_000_000


def _obs(
    minute: int,
    *,
    fast: float = 101.0,
    slow: float = 100.0,
    high: float = 102.0,
    low: float = 100.0,
    close: float = 101.0,
    atr: float = 1.0,
    exposure: float = 0.0,
    pending: tuple[str, ...] = (),
    drawdown: float = 0.0,
    unrealized: float = 0.0,
) -> PointInTimeObservation:
    close_time = minute * MINUTE_NS
    decision = close_time + DELAY_NS
    market = {
        "latest_1m_close_ts_ns": close_time,
        "ema_9": fast,
        "ema_21": slow,
        "bar_high_1m": high,
        "bar_low_1m": low,
        "bar_close_1m": close,
        "atr_14": atr,
        "vol_1m": 0.01,
    }
    return PointInTimeObservation(
        episode_id="lifecycle",
        decision_ts_ns=decision,
        market=market,
        feature_available_at_ns={name: decision for name in market},
        account={
            "net_exposure": exposure,
            "equity": 10_000.0,
            "drawdown": drawdown,
            "unrealized_pnl_fraction": unrealized,
        },
        pending_order_ids=pending,
    )


def test_x1_and_x2_share_the_exact_entry_event_and_target():
    x1 = build_teacher_1m("T01_EMA_9_21", exit_variant="X1")
    x2 = build_teacher_1m("T01_EMA_9_21", exit_variant="X2")
    for teacher in (x1, x2):
        assert teacher.advise(_obs(1, fast=99.0)).intent == "FLAT"
    entry_x1 = x1.advise(_obs(2, fast=101.0))
    entry_x2 = x2.advise(_obs(2, fast=101.0))
    assert (entry_x1.intent, entry_x1.requested_exposure, entry_x1.reason_codes) == (
        entry_x2.intent, entry_x2.requested_exposure, entry_x2.reason_codes
    ) == ("OPEN_LONG", 0.3, ("EMA_CROSS_LONG",))
    assert x1.recipe.risk_recipe_id == x2.recipe.risk_recipe_id
    assert x1.recipe.scenario_cost_ids == x2.recipe.scenario_cost_ids
    assert x2.recipe.indicator_exit_rule_id == "X2_ATR_TRAIL_3_V1"


def test_x2_uses_exact_epoch_nanosecond_close_clock():
    close_ts = (1_700_000_000_000_000_000 // MINUTE_NS) * MINUTE_NS
    decision_ts = close_ts + DELAY_NS
    market = {
        "bar_high_1m": 102.0,
        "bar_low_1m": 100.0,
        "bar_close_1m": 101.0,
        "atr_14": 1.0,
        "ema_9": 101.0,
        "ema_21": 100.0,
    }
    observation = PointInTimeObservation(
        episode_id="epoch-ns-x2",
        decision_ts_ns=decision_ts,
        market=market,
        feature_available_at_ns={name: decision_ts for name in market},
        account={"net_exposure": 0.3, "equity": 10_000.0},
        exact_market_clocks_ns=(("latest_1m_close_ts_ns", close_ts),),
    )
    policy = AtrTrailingExit()
    assert policy.evaluate(observation, 0.3) is None
    assert policy.last_close_time_ns == close_ts


def test_x2_atr_trail_updates_only_on_completed_bars_and_never_loosens():
    policy = AtrTrailingExit(3.0)
    assert policy.evaluate(_obs(1, high=101, low=99, close=100), 0.3) is None
    first_trail = policy.trail
    assert first_trail == 98.0
    assert policy.evaluate(_obs(2, high=105, low=103, close=104), 0.3) is None
    second_trail = policy.trail
    assert second_trail == 102.0
    signal = policy.evaluate(_obs(3, high=103, low=100, close=101, atr=2.0), 0.3)
    assert policy.trail == second_trail
    assert signal is not None and signal.reason_code == "ATR_TRAIL_EXIT_LONG"

    policy = AtrTrailingExit(3.0)
    assert policy.evaluate(_obs(1, high=101, low=99, close=100), -0.3) is None
    first_short = policy.trail
    assert first_short == 102.0
    assert policy.evaluate(_obs(2, high=97, low=95, close=96), -0.3) is None
    second_short = policy.trail
    assert second_short == 98.0
    signal = policy.evaluate(_obs(3, high=100, low=97, close=99, atr=2.0), -0.3)
    assert policy.trail == second_short
    assert signal is not None and signal.reason_code == "ATR_TRAIL_EXIT_SHORT"

    with pytest.raises(ValueError, match="new completed"):
        policy.evaluate(_obs(3, high=100, low=97, close=99, atr=2.0), -0.3)


def test_x2_teacher_holds_then_exits_on_close_beyond_trail():
    teacher = build_teacher_1m("T01_EMA_9_21", exit_variant="X2")
    teacher.advise(_obs(1, fast=99.0))
    assert teacher.advise(_obs(2, fast=101.0)).intent == "OPEN_LONG"
    assert teacher.advise(
        _obs(3, fast=102.0, high=102, low=100, close=101, exposure=0.3)
    ).intent == "HOLD"
    assert teacher.advise(
        _obs(4, fast=102.0, high=105, low=103, close=104, exposure=0.3)
    ).intent == "HOLD"
    exited = teacher.advise(
        _obs(5, fast=102.0, high=102, low=100, close=101, exposure=0.3)
    )
    assert exited.intent == "EXIT"
    assert exited.reason_codes == ("ATR_TRAILING_EXIT", "ATR_TRAIL_EXIT_LONG")


def test_no_ordinary_max_holding_time_through_1441_minutes():
    teacher = build_teacher_1m("T01_EMA_9_21")
    teacher.advise(_obs(1, fast=99.0))
    assert teacher.advise(_obs(2, fast=101.0)).intent == "OPEN_LONG"
    checkpoints = {}
    for minute in range(3, 1444):
        result = teacher.advise(_obs(minute, fast=102.0, exposure=0.3))
        if minute in (243, 483, 1443):
            checkpoints[minute] = result
    assert all(result.intent == "HOLD" for result in checkpoints.values())
    assert all(result.reason_codes == ("POSITION_HOLD",) for result in checkpoints.values())


def test_reverse_signal_first_requests_exit_and_pending_ack_does_not_fake_flat():
    teacher = build_teacher_1m("T01_EMA_9_21")
    teacher.advise(_obs(1, fast=99.0))
    teacher.advise(_obs(2, fast=101.0))
    close_request = teacher.advise(_obs(3, fast=99.0, exposure=0.3))
    assert close_request.intent == "EXIT"
    assert close_request.requested_exposure == 0.0
    pending = teacher.advise(_obs(3, fast=99.0, exposure=0.3, pending=("close",)))
    assert pending.intent == "HOLD"
    assert pending.reason_codes == ("PENDING_ACK",)


def _trace(
    minute: int, action: str, exposure: float, *, submitted: tuple[str, ...] = (),
    filled: tuple[str, ...] = (), related: tuple[str, ...] = (),
    pending: tuple[str, ...] = (),
) -> NativeActionTrace:
    return NativeActionTrace(
        decision_ts_ns=minute * MINUTE_NS,
        requested_action=0.0,
        risk_adjusted_target=0.0,
        planner_target=0.0,
        actual_exposure_after=exposure,
        native_equity_before=10_000.0,
        native_equity_after=10_000.0,
        planner_action=action,
        submitted_order_ids=submitted,
        filled_order_ids=filled,
        related_existing_order_ids=related,
        pending_order_ids_after=pending,
    )


def test_native_cancel_ack_fill_requires_confirmed_flat_and_fill_proof():
    traces = (
        _trace(1, "SUBMIT", 0.3, submitted=("close-1",), pending=("close-1",)),
        _trace(2, "WAIT", 0.3, related=("close-1",), pending=("close-1",)),
        _trace(3, "WAIT", 0.0, related=("close-1",), filled=("close-1",)),
    )
    assert verify_native_flat_lifecycle(traces, initial_abs_exposure=0.3)
    assert not verify_native_flat_lifecycle(traces[:-1], initial_abs_exposure=0.3)
    no_fill = (*traces[:-1], _trace(3, "WAIT", 0.0, related=("close-1",)))
    assert not verify_native_flat_lifecycle(no_fill, initial_abs_exposure=0.3)


def test_recovery_hard_exit_precedes_pending_guard_and_normal_teacher():
    primary = build_teacher_1m("T01_EMA_9_21")
    selector = FrozenTeacherSelector(primary=primary, recovery=RecoveryTeacher())
    result = selector.advise(
        _obs(
            1,
            fast=101.0,
            exposure=0.3,
            pending=("ordinary-entry",),
            drawdown=0.20,
        )
    )
    assert result.expert_id == "recovery_v1"
    assert result.intent == "EXIT"
    assert result.reason_codes == ("MAX_DRAWDOWN",)

