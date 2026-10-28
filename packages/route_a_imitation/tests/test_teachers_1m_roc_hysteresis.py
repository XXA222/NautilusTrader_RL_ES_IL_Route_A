from __future__ import annotations

from typing import Literal

import pytest
from rles_route_a.contracts import ExpertAdvice, NativeActionTrace
from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.indicators_1m import MINUTE_NS
from rles_route_a.teachers_1m.registry import (
    ROC_COST_HYSTERESIS_FADE_TEACHER_ID,
    ROC_COST_HYSTERESIS_TEACHER_ID,
    build_teacher_1m,
)


DELAY_NS = 1_000_000


def _obs(
    minute: int,
    *,
    roc_20: float,
    roc_60: float,
    close: float = 100.0,
    atr: float = 0.1,
    high: float | None = None,
    low: float | None = None,
    exposure: float = 0.0,
    pending: tuple[str, ...] = (),
    episode_id: str = "roc-hysteresis",
    ema_9: float | None = None,
    ema_21: float | None = None,
    source_id: str = "",
) -> PointInTimeObservation:
    close_time = minute * MINUTE_NS
    decision = close_time + DELAY_NS
    market = {
        "latest_1m_close_ts_ns": close_time,
        "roc_20": roc_20,
        "roc_60": roc_60,
        "atr_14": atr,
        "bar_high_1m": close if high is None else high,
        "bar_low_1m": close if low is None else low,
        "bar_close_1m": close,
    }
    if ema_9 is not None:
        market["ema_9"] = ema_9
    if ema_21 is not None:
        market["ema_21"] = ema_21
    return PointInTimeObservation(
        episode_id=episode_id,
        decision_ts_ns=decision,
        market=market,
        feature_available_at_ns={name: decision for name in market},
        account={
            "net_exposure": exposure,
            "equity": 10_000.0,
            "drawdown": 0.0,
            "unrealized_pnl_fraction": 0.0,
            "position_avg_px_open": 100.0 if abs(exposure) > 1e-8 else 0.0,
        },
        pending_order_ids=pending,
        source_id=source_id,
    )


def _ready_teacher(
    *, side: int = 1, cost_bps: float = 12.0, edge_bps: float = 20.0,
    exit_variant: Literal["X1", "X2", "X3", "X4", "X5", "X6", "X8"] = "X1",
    teacher_id: str = ROC_COST_HYSTERESIS_TEACHER_ID,
):
    teacher = build_teacher_1m(
        teacher_id,
        exit_variant=exit_variant,
        roc_round_trip_cost_bps=cost_bps,
        exit_round_trip_cost_bps=(
            cost_bps if exit_variant in ("X3", "X4", "X5", "X8") else None
        ),
    )
    signed_roc = side * edge_bps / 10_000
    result = None
    for minute in range(1, 22):
        result = teacher.advise(
            _obs(minute, roc_20=signed_roc, roc_60=signed_roc),
        )
    assert result is not None
    return teacher, result


def _observe_profit_lock_exit(teacher, *, side: int) -> None:
    decision_ts_ns = 22 * MINUTE_NS + DELAY_NS
    reason = (
        "COST_AWARE_ATR_PROFIT_LOCK_LONG"
        if side > 0
        else "COST_AWARE_ATR_PROFIT_LOCK_SHORT"
    )
    action = ExpertAdvice(
        teacher.expert_id, decision_ts_ns, "EXIT", 0.0, ("INDICATOR_EXIT", reason),
    )
    trace = NativeActionTrace(
        decision_ts_ns, 0.0, 0.0, 0.0, 0.0, 10_000.0, 10_000.0, "SUBMIT",
    )
    teacher.observe_native_action(action, trace, None)


def _observe_loss_guard_exit(teacher, *, side: int) -> None:
    decision_ts_ns = 22 * MINUTE_NS + DELAY_NS
    direction = "LONG" if side > 0 else "SHORT"
    action = ExpertAdvice(
        teacher.expert_id,
        decision_ts_ns,
        "EXIT",
        0.0,
        ("INDICATOR_EXIT", f"COST_AWARE_ATR_LOSS_GUARD_{direction}"),
    )
    trace = NativeActionTrace(
        decision_ts_ns, 0.0, 0.0, 0.0, 0.0, 10_000.0, 10_000.0, "SUBMIT",
    )
    teacher.observe_native_action(action, trace, None)


def test_source_change_clears_roc_volatility_and_consensus_history():
    teacher = build_teacher_1m(ROC_COST_HYSTERESIS_TEACHER_ID)
    action = None
    for minute in range(1, 22):
        action = teacher.advise(
            _obs(
                minute,
                roc_20=0.002,
                roc_60=0.002,
                close=100.0 + minute / 100.0,
                source_id="source-a",
            ),
        )
    assert action is not None
    assert len(teacher._returns) == teacher._VOLATILITY_RETURNS
    assert teacher._consensus_run == 20

    first_new_source_bar = teacher.advise(
        _obs(
            22,
            roc_20=0.002,
            roc_60=0.002,
            close=101.0,
            source_id="source-b",
        ),
    )
    assert first_new_source_bar.intent == "FLAT"
    assert first_new_source_bar.reason_codes == ("INDICATOR_PREVIOUS_NOT_READY",)
    assert len(teacher._returns) == 0
    assert teacher._consensus_run == 0

    teacher.advise(
        _obs(
            23,
            roc_20=0.002,
            roc_60=0.002,
            close=101.01,
            source_id="source-b",
        ),
    )
    assert len(teacher._returns) == 1
    assert teacher._consensus_run == 1


def test_cost_hysteresis_is_a_separate_recipe_and_base_cost_is_used():
    legacy = build_teacher_1m("T02_ROC_20_60")
    candidate = build_teacher_1m(
        ROC_COST_HYSTERESIS_TEACHER_ID,
        roc_round_trip_cost_bps=30.0,
    )
    assert legacy.recipe.entry_rule_id == "ROC_CONSENSUS_V1"
    assert candidate.expert_id == ROC_COST_HYSTERESIS_TEACHER_ID
    assert candidate.recipe.entry_rule_id == "ROC_COST_AWARE_CONSENSUS_HYSTERESIS_V1"
    assert dict(candidate.recipe.parameters)["round_trip_cost_bps"] == 30.0


def test_entry_waits_for_confirmed_consensus_and_realized_volatility():
    teacher = build_teacher_1m(ROC_COST_HYSTERESIS_TEACHER_ID)
    signed_roc = 20 / 10_000
    decisions = [
        teacher.advise(_obs(minute, roc_20=signed_roc, roc_60=signed_roc))
        for minute in range(1, 22)
    ]
    assert all(action.intent == "FLAT" for action in decisions[:-1])
    assert decisions[-1].intent == "OPEN_LONG"
    assert decisions[-1].reason_codes == ("ROC_COST_AWARE_CONSENSUS_LONG",)


def test_entry_opportunity_must_clear_cost_atr_and_realized_volatility():
    high_cost, high_cost_action = _ready_teacher(cost_bps=30.0)
    assert high_cost_action.intent == "FLAT"
    assert high_cost_action.reason_codes == ("ROC_EDGE_BELOW_COST_AND_NOISE",)

    high_atr = build_teacher_1m(ROC_COST_HYSTERESIS_TEACHER_ID)
    action = None
    for minute in range(1, 22):
        action = high_atr.advise(
            _obs(minute, roc_20=0.002, roc_60=0.002, atr=0.5),
        )
    assert action is not None and action.intent == "FLAT"
    assert action.reason_codes == ("ROC_EDGE_BELOW_COST_AND_NOISE",)

    high_volatility = build_teacher_1m(ROC_COST_HYSTERESIS_TEACHER_ID)
    action = None
    for minute in range(1, 22):
        close = 100.0 if minute % 2 else 100.1
        action = high_volatility.advise(
            _obs(minute, roc_20=0.002, roc_60=0.002, close=close),
        )
    assert action is not None and action.intent == "FLAT"
    assert action.reason_codes == ("ROC_EDGE_BELOW_COST_AND_NOISE",)


def test_long_exit_holds_single_zero_cross_and_exits_after_confirmed_cost_margin():
    teacher, opened = _ready_teacher()
    assert opened.intent == "OPEN_LONG"

    one_zero_cross = teacher.advise(
        _obs(22, roc_20=-0.0001, roc_60=0.001, exposure=0.3),
    )
    assert one_zero_cross.intent == "HOLD"

    inside_hysteresis = teacher.advise(
        _obs(23, roc_20=-0.0001, roc_60=-0.0001, exposure=0.3),
    )
    assert inside_hysteresis.intent == "HOLD"

    first_adverse_bar = teacher.advise(
        _obs(24, roc_20=-0.002, roc_60=-0.002, exposure=0.3),
    )
    assert first_adverse_bar.intent == "HOLD"

    second_adverse_bar = teacher.advise(
        _obs(25, roc_20=-0.002, roc_60=-0.002, exposure=0.3),
    )
    assert second_adverse_bar.intent == "EXIT"
    assert second_adverse_bar.reason_codes == (
        "INDICATOR_EXIT",
        "ROC_COST_HYSTERESIS_CONFIRMED_EXIT_LONG",
    )


def test_short_exit_is_symmetric():
    teacher, opened = _ready_teacher(side=-1)
    assert opened.intent == "OPEN_SHORT"
    teacher.advise(
        _obs(22, roc_20=0.0001, roc_60=-0.001, exposure=-0.3),
    )
    assert teacher.last_signal.reason_code == "POSITION_HOLD"
    teacher.advise(
        _obs(23, roc_20=0.002, roc_60=0.002, exposure=-0.3),
    )
    assert teacher.last_signal.reason_code == "POSITION_HOLD"
    exited = teacher.advise(
        _obs(24, roc_20=0.002, roc_60=0.002, exposure=-0.3),
    )
    assert exited.intent == "EXIT"
    assert exited.reason_codes == (
        "INDICATOR_EXIT",
        "ROC_COST_HYSTERESIS_CONFIRMED_EXIT_SHORT",
    )


def test_candidate_does_not_impose_a_holding_time_limit():
    teacher, opened = _ready_teacher()
    assert opened.intent == "OPEN_LONG"
    final = None
    for minute in range(22, 1463):
        final = teacher.advise(
            _obs(minute, roc_20=0.002, roc_60=0.002, exposure=0.3),
        )
    assert final is not None
    assert final.intent == "HOLD"
    assert final.reason_codes == ("POSITION_HOLD",)


def test_roc_contrarian_candidate_inverts_entries_and_exits_if_momentum_persists():
    fade, positive_consensus = _ready_teacher(
        side=1, teacher_id=ROC_COST_HYSTERESIS_FADE_TEACHER_ID,
    )
    assert positive_consensus.intent == "OPEN_SHORT"
    assert positive_consensus.reason_codes == ("ROC_COST_AWARE_CONTRARIAN_SHORT",)
    assert dict(fade.recipe.parameters)["direction_mode"] == "CONSENSUS_FADE_V1"

    negative_consensus_teacher, negative_consensus = _ready_teacher(
        side=-1, teacher_id=ROC_COST_HYSTERESIS_FADE_TEACHER_ID,
    )
    assert negative_consensus.intent == "OPEN_LONG"
    assert negative_consensus.reason_codes == ("ROC_COST_AWARE_CONTRARIAN_LONG",)
    assert negative_consensus_teacher.recipe.entry_rule_id == (
        "ROC_COST_AWARE_CONTRARIAN_CONSENSUS_HYSTERESIS_V1"
    )

    one_adverse_bar = fade.advise(
        _obs(22, roc_20=0.002, roc_60=0.002, exposure=-0.3),
    )
    assert one_adverse_bar.intent == "HOLD"
    two_adverse_bars = fade.advise(
        _obs(23, roc_20=0.002, roc_60=0.002, exposure=-0.3),
    )
    assert two_adverse_bars.intent == "EXIT"
    assert two_adverse_bars.reason_codes == (
        "INDICATOR_EXIT",
        "ROC_COST_HYSTERESIS_CONFIRMED_EXIT_SHORT",
    )


def test_profit_lock_rearm_latch_survives_gap_but_resets_for_new_episode():
    teacher, opened = _ready_teacher(exit_variant="X4")
    assert opened.intent == "OPEN_LONG"

    _observe_profit_lock_exit(teacher, side=1)
    assert teacher._cost_exit_rearm_latched is True

    gap = teacher.advise(
        _obs(23, roc_20=0.002, roc_60=0.002),
    )
    assert gap.reason_codes == ("MISSING_BAR",)
    assert teacher._cost_exit_rearm_latched is True
    assert teacher._opposite_rearm_run == 0

    teacher.reset("next-episode")
    assert teacher._cost_exit_rearm_latched is False
    assert teacher._cost_exit_side == 0
    assert teacher._opposite_rearm_run == 0

    action = None
    for minute in range(1, 22):
        action = teacher.advise(
            _obs(
                minute,
                roc_20=0.002,
                roc_60=0.002,
                episode_id="next-episode",
            ),
        )
    assert action is not None
    assert action.intent == "OPEN_LONG"
    assert action.reason_codes == ("ROC_COST_AWARE_CONSENSUS_LONG",)


def test_x4_profit_lock_requires_two_flat_opposite_consensus_bars_before_reentry():
    teacher, opened = _ready_teacher(exit_variant="X4")
    assert opened.intent == "OPEN_LONG"
    assert teacher.recipe.entry_rule_id == (
        "ROC_COST_AWARE_CONSENSUS_HYSTERESIS_OPPOSITE_REARM_V1"
    )
    assert teacher.recipe.indicator_exit_rule_id == (
        "X4_ATR_COST_PROFIT_LOCK_OPPOSITE_REARM_V1"
    )
    assert teacher._atr_exit is not None
    assert teacher._atr_exit.policy_id == "X4_ATR_COST_PROFIT_LOCK_OPPOSITE_REARM_V1"

    _observe_profit_lock_exit(teacher, side=1)
    same_regime = teacher.advise(
        _obs(22, roc_20=0.002, roc_60=0.002),
    )
    assert same_regime.intent == "FLAT"
    assert same_regime.reason_codes == (
        "ROC_PROFIT_LOCK_REARM_WAIT_OPPOSITE_CONSENSUS",
    )

    first_opposite = teacher.advise(
        _obs(23, roc_20=-0.002, roc_60=-0.002),
    )
    assert first_opposite.intent == "FLAT"
    assert first_opposite.reason_codes == (
        "ROC_PROFIT_LOCK_REARM_WAIT_OPPOSITE_CONSENSUS",
    )
    second_opposite = teacher.advise(
        _obs(24, roc_20=-0.002, roc_60=-0.002),
    )
    assert second_opposite.intent == "OPEN_SHORT"
    assert second_opposite.reason_codes == ("ROC_COST_AWARE_CONSENSUS_SHORT",)


def test_x4_short_profit_lock_rearm_is_symmetric_and_resets_on_pending_close():
    teacher, opened = _ready_teacher(side=-1, exit_variant="X4")
    assert opened.intent == "OPEN_SHORT"
    _observe_profit_lock_exit(teacher, side=-1)

    initial_opposite = teacher.advise(
        _obs(22, roc_20=0.002, roc_60=0.002),
    )
    assert initial_opposite.reason_codes == (
        "ROC_PROFIT_LOCK_REARM_WAIT_OPPOSITE_CONSENSUS",
    )
    pending_flat = teacher.advise(
        _obs(23, roc_20=0.002, roc_60=0.002, pending=("close-1",)),
    )
    assert pending_flat.intent == "FLAT"
    assert pending_flat.reason_codes == ("PENDING_ACK",)
    first_opposite = teacher.advise(
        _obs(24, roc_20=0.002, roc_60=0.002),
    )
    assert first_opposite.reason_codes == (
        "ROC_PROFIT_LOCK_REARM_WAIT_OPPOSITE_CONSENSUS",
    )
    reset_by_neutral = teacher.advise(
        _obs(25, roc_20=0.0, roc_60=0.0),
    )
    assert reset_by_neutral.reason_codes == ("ROC_HYSTERESIS_NEUTRAL",)
    first_again = teacher.advise(
        _obs(26, roc_20=0.002, roc_60=0.002),
    )
    assert first_again.reason_codes == (
        "ROC_PROFIT_LOCK_REARM_WAIT_OPPOSITE_CONSENSUS",
    )
    second_again = teacher.advise(
        _obs(27, roc_20=0.002, roc_60=0.002),
    )
    assert second_again.intent == "OPEN_LONG"


def test_x4_is_restricted_to_roc_hysteresis_and_requires_scenario_cost():
    with pytest.raises(ValueError, match="scenario round-trip cost"):
        build_teacher_1m(
            ROC_COST_HYSTERESIS_TEACHER_ID,
            exit_variant="X4",
        )
    with pytest.raises(ValueError, match="only applies to the ROC hysteresis teacher"):
        build_teacher_1m(
            "T01_EMA_9_21", exit_variant="X4", exit_round_trip_cost_bps=12.0,
        )


def test_x5_cost_atr_loss_guard_is_symmetric_and_profit_lock_stays_available():
    long_teacher = build_teacher_1m(
        ROC_COST_HYSTERESIS_TEACHER_ID,
        exit_variant="X5",
        roc_round_trip_cost_bps=12.0,
        exit_round_trip_cost_bps=12.0,
    )
    assert long_teacher._atr_exit is not None
    long_stop = long_teacher._atr_exit.evaluate(
        _obs(1, roc_20=0.0, roc_60=0.0, close=99.75, high=99.8, low=99.7, exposure=0.3),
        0.3,
    )
    assert long_stop is not None
    assert long_stop.reason_code == "COST_AWARE_ATR_LOSS_GUARD_LONG"

    short_teacher = build_teacher_1m(
        ROC_COST_HYSTERESIS_TEACHER_ID,
        exit_variant="X5",
        roc_round_trip_cost_bps=12.0,
        exit_round_trip_cost_bps=12.0,
    )
    assert short_teacher._atr_exit is not None
    short_stop = short_teacher._atr_exit.evaluate(
        _obs(1, roc_20=0.0, roc_60=0.0, close=100.25,
             high=100.3, low=100.2, exposure=-0.3),
        -0.3,
    )
    assert short_stop is not None
    assert short_stop.reason_code == "COST_AWARE_ATR_LOSS_GUARD_SHORT"

    protected = build_teacher_1m(
        ROC_COST_HYSTERESIS_TEACHER_ID,
        exit_variant="X5",
        roc_round_trip_cost_bps=12.0,
        exit_round_trip_cost_bps=12.0,
    )
    assert protected._atr_exit is not None
    assert protected._atr_exit.evaluate(
        _obs(1, roc_20=0.0, roc_60=0.0, close=100.2,
             high=100.25, low=100.0, exposure=0.3),
        0.3,
    ) is None
    assert protected._atr_exit.evaluate(
        _obs(2, roc_20=0.0, roc_60=0.0, close=100.2,
             high=100.25, low=100.0, exposure=0.3),
        0.3,
    ) is None
    assert protected._atr_exit.activated
    lock = protected._atr_exit.evaluate(
        _obs(3, roc_20=0.0, roc_60=0.0, close=100.17,
             high=100.25, low=100.1, exposure=0.3),
        0.3,
    )
    assert lock is not None
    assert lock.reason_code == "COST_AWARE_ATR_PROFIT_LOCK_LONG"


def test_x5_loss_guard_exit_requires_opposite_consensus_before_reentry():
    teacher, opened = _ready_teacher(exit_variant="X5")
    assert opened.intent == "OPEN_LONG"
    assert teacher.recipe.entry_rule_id == (
        "ROC_COST_AWARE_CONSENSUS_HYSTERESIS_COST_EXIT_REARM_V1"
    )
    _observe_loss_guard_exit(teacher, side=1)
    still_same = teacher.advise(
        _obs(22, roc_20=0.002, roc_60=0.002),
    )
    assert still_same.reason_codes == (
        "ROC_PROFIT_LOCK_REARM_WAIT_OPPOSITE_CONSENSUS",
    )
    first_opposite = teacher.advise(
        _obs(23, roc_20=-0.002, roc_60=-0.002),
    )
    assert first_opposite.intent == "FLAT"
    second_opposite = teacher.advise(
        _obs(24, roc_20=-0.002, roc_60=-0.002),
    )
    assert second_opposite.intent == "OPEN_SHORT"


def test_x5_is_roc_only_and_requires_scenario_cost():
    with pytest.raises(ValueError, match="scenario round-trip cost"):
        build_teacher_1m(ROC_COST_HYSTERESIS_TEACHER_ID, exit_variant="X5")
    with pytest.raises(ValueError, match="only applies to the ROC hysteresis teacher"):
        build_teacher_1m(
            "T01_EMA_9_21", exit_variant="X5", exit_round_trip_cost_bps=12.0,
        )



def test_x6_atr_exit_requires_opposite_consensus_before_reentry():
    teacher, opened = _ready_teacher(exit_variant="X6")
    assert opened.intent == "OPEN_LONG"
    assert teacher.recipe.entry_rule_id == (
        "ROC_COST_AWARE_CONSENSUS_HYSTERESIS_ATR_EXIT_REARM_V1"
    )
    assert teacher.recipe.indicator_exit_rule_id == (
        "X6_ATR_TRAIL_OPPOSITE_CONSENSUS_REARM_V1"
    )
    assert teacher._atr_exit is not None
    assert teacher._atr_exit.policy_id == (
        "X6_ATR_TRAIL_OPPOSITE_CONSENSUS_REARM_V1"
    )

    decision_ts_ns = 22 * MINUTE_NS + DELAY_NS
    atr_exit = ExpertAdvice(
        teacher.expert_id,
        decision_ts_ns,
        "EXIT",
        0.0,
        ("ATR_TRAILING_EXIT", "ATR_TRAIL_EXIT_LONG"),
    )
    trace = NativeActionTrace(
        decision_ts_ns, 0.0, 0.0, 0.0, 0.0, 10_000.0, 10_000.0, "SUBMIT",
    )
    teacher.observe_native_action(atr_exit, trace, None)

    same_direction = teacher.advise(
        _obs(22, roc_20=0.002, roc_60=0.002),
    )
    assert same_direction.intent == "FLAT"
    assert same_direction.reason_codes == (
        "ROC_ATR_EXIT_REARM_WAIT_OPPOSITE_CONSENSUS",
    )

    first_opposite = teacher.advise(
        _obs(23, roc_20=-0.002, roc_60=-0.002),
    )
    assert first_opposite.intent == "FLAT"
    assert first_opposite.reason_codes == (
        "ROC_ATR_EXIT_REARM_WAIT_OPPOSITE_CONSENSUS",
    )

    second_opposite = teacher.advise(
        _obs(24, roc_20=-0.002, roc_60=-0.002),
    )
    assert second_opposite.intent == "OPEN_SHORT"
    assert second_opposite.reason_codes == ("ROC_COST_AWARE_CONSENSUS_SHORT",)


def test_x6_atr_exit_rearm_is_symmetric_for_short_positions():
    teacher, opened = _ready_teacher(side=-1, exit_variant="X6")
    assert opened.intent == "OPEN_SHORT"
    decision_ts_ns = 22 * MINUTE_NS + DELAY_NS
    atr_exit = ExpertAdvice(
        teacher.expert_id,
        decision_ts_ns,
        "EXIT",
        0.0,
        ("ATR_TRAILING_EXIT", "ATR_TRAIL_EXIT_SHORT"),
    )
    trace = NativeActionTrace(
        decision_ts_ns, 0.0, 0.0, 0.0, 0.0, 10_000.0, 10_000.0, "SUBMIT",
    )
    teacher.observe_native_action(atr_exit, trace, None)

    same_direction = teacher.advise(
        _obs(22, roc_20=-0.002, roc_60=-0.002),
    )
    assert same_direction.reason_codes == (
        "ROC_ATR_EXIT_REARM_WAIT_OPPOSITE_CONSENSUS",
    )
    first_opposite = teacher.advise(
        _obs(23, roc_20=0.002, roc_60=0.002),
    )
    assert first_opposite.intent == "FLAT"
    second_opposite = teacher.advise(
        _obs(24, roc_20=0.002, roc_60=0.002),
    )
    assert second_opposite.intent == "OPEN_LONG"
    assert second_opposite.reason_codes == ("ROC_COST_AWARE_CONSENSUS_LONG",)


def test_x6_is_restricted_to_trend_roc_hysteresis():
    with pytest.raises(ValueError, match="only applies to the ROC hysteresis teacher"):
        build_teacher_1m("T01_EMA_9_21", exit_variant="X6")
    with pytest.raises(ValueError, match="only applies to the ROC hysteresis teacher"):
        build_teacher_1m(ROC_COST_HYSTERESIS_FADE_TEACHER_ID, exit_variant="X6")


def _ema_age_teacher():
    teacher = build_teacher_1m(
        ROC_COST_HYSTERESIS_TEACHER_ID,
        exit_variant="X7",
        roc_round_trip_cost_bps=12.0,
    )
    assert teacher.recipe.entry_rule_id == "ROC_COST_AWARE_EMA_CROSS_AGE_21_60_V1"
    assert teacher.recipe.indicator_exit_rule_id == "X1_ROC_COST_HYSTERESIS_EXIT_V1"
    assert dict(teacher.recipe.parameters)["entry_ema_cross_age_min_bars"] == 21
    assert dict(teacher.recipe.parameters)["entry_ema_cross_age_max_bars"] == 60
    return teacher


def test_x7_waits_until_21_minutes_after_long_ema_cross():
    teacher = _ema_age_teacher()
    action = None
    for minute in range(1, 23):
        ema_9 = 99.9 if minute == 1 else 100.1
        action = teacher.advise(
            _obs(
                minute, roc_20=0.002, roc_60=0.002,
                ema_9=ema_9, ema_21=100.0,
            ),
        )
    assert action is not None and action.intent == "FLAT"
    assert action.reason_codes == ("ROC_EMA_CROSS_AGE_OUTSIDE_21_60",)

    entered = teacher.advise(
        _obs(23, roc_20=0.002, roc_60=0.002, ema_9=100.1, ema_21=100.0),
    )
    assert entered.intent == "OPEN_LONG"
    assert entered.reason_codes == ("ROC_COST_AWARE_CONSENSUS_LONG",)


def test_x7_ema_cross_age_gate_is_symmetric_for_short_entries():
    teacher = _ema_age_teacher()
    action = None
    for minute in range(1, 24):
        ema_9 = 100.1 if minute == 1 else 99.9
        action = teacher.advise(
            _obs(
                minute, roc_20=-0.002, roc_60=-0.002,
                ema_9=ema_9, ema_21=100.0,
            ),
        )
    assert action is not None and action.intent == "OPEN_SHORT"
    assert action.reason_codes == ("ROC_COST_AWARE_CONSENSUS_SHORT",)


def test_x7_rejects_trends_older_than_60_minutes_and_is_trend_roc_only():
    teacher = _ema_age_teacher()
    action = None
    for minute in range(1, 65):
        ema_9 = 99.9 if minute == 1 else 100.1
        roc = 0.0 if minute < 63 else 0.002
        action = teacher.advise(
            _obs(minute, roc_20=roc, roc_60=roc, ema_9=ema_9, ema_21=100.0),
        )
    assert action is not None and action.intent == "FLAT"
    assert action.reason_codes == ("ROC_EMA_CROSS_AGE_OUTSIDE_21_60",)

    with pytest.raises(ValueError, match="only applies to the trend ROC hysteresis teacher"):
        build_teacher_1m("T01_EMA_9_21", exit_variant="X7")
    with pytest.raises(ValueError, match="only applies to the trend ROC hysteresis teacher"):
        build_teacher_1m(ROC_COST_HYSTERESIS_FADE_TEACHER_ID, exit_variant="X7")

def test_x8_builder_uses_cost_handoff_and_atr_opposite_rearm():
    teacher = build_teacher_1m(
        ROC_COST_HYSTERESIS_TEACHER_ID,
        exit_variant="X8",
        roc_round_trip_cost_bps=12.0,
        exit_round_trip_cost_bps=12.0,
    )
    assert teacher.recipe.indicator_exit_rule_id == "X8_ATR_TRAIL_HANDOFF_AT_4X_COST_V1"
    assert teacher.recipe.entry_rule_id == "ROC_COST_AWARE_CONSENSUS_ATR_HANDOFF_REARM_V1"
    assert dict(teacher.recipe.parameters)["cost_exit_rearm_mode"] == (
        "ATR_EXIT_OPPOSITE_CONSENSUS_V1"
    )
    with pytest.raises(ValueError, match="scenario round-trip cost"):
        build_teacher_1m(
            ROC_COST_HYSTERESIS_TEACHER_ID,
            exit_variant="X8",
            roc_round_trip_cost_bps=12.0,
        )


def test_x8_atr_exit_rearms_only_after_opposite_consensus_and_flat_ack():
    teacher, opened = _ready_teacher(exit_variant="X8")
    assert opened.intent == "OPEN_LONG"
    decision_ts_ns = 22 * MINUTE_NS + DELAY_NS
    atr_exit = ExpertAdvice(
        teacher.expert_id,
        decision_ts_ns,
        "EXIT",
        0.0,
        ("ATR_TRAILING_EXIT", "ATR_TRAIL_EXIT_LONG"),
    )
    trace = NativeActionTrace(
        decision_ts_ns, 0.0, 0.0, 0.0, 0.0, 10_000.0, 10_000.0, "SUBMIT",
    )
    teacher.observe_native_action(atr_exit, trace, None)
    same_side = teacher.advise(_obs(22, roc_20=0.002, roc_60=0.002))
    assert same_side.reason_codes == (
        "ROC_ATR_EXIT_REARM_WAIT_OPPOSITE_CONSENSUS",
    )
    first_opposite = teacher.advise(_obs(23, roc_20=-0.002, roc_60=-0.002))
    assert first_opposite.intent == "FLAT"
    second_opposite = teacher.advise(_obs(24, roc_20=-0.002, roc_60=-0.002))
    assert second_opposite.intent == "OPEN_SHORT"