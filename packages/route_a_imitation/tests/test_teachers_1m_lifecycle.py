from __future__ import annotations

import pytest

from rles_route_a.contracts import (
    ExpertAdvice, NativeActionTrace, PointInTimeObservation,
)
from rles_route_a.integration.host_gym_projection import (
    L3OrderFacts, _order_status_receipt, _validate_order_statuses,
)
from rles_route_a.indicators_1m import MINUTE_NS
from rles_route_a.integration.exit_lifecycle import verify_native_flat_lifecycle
from rles_route_a.teachers.recovery import RecoveryTeacher
from rles_route_a.teachers.selector import FrozenTeacherSelector
from rles_route_a.teachers_1m import (
    AtrTrailingExit,
    AtrTrailHandoffToIndicatorExit,
    CostAwareAtrProfitProtectExit,
    build_teacher_1m,
)


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
    avg_open_price: float = 100.0,
    missing_market_fields: tuple[str, ...] = (),
    source_id: str = "",
    fold_id: str = "",
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
    for name in missing_market_fields:
        market.pop(name, None)
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
            "position_avg_px_open": avg_open_price if abs(exposure) > 1e-8 else 0.0,
        },
        pending_order_ids=pending,
        source_id=source_id,
        fold_id=fold_id,
    )


def test_source_change_resets_ema_cross_state_within_an_episode():
    teacher = build_teacher_1m("T01_EMA_9_21")

    first_source = teacher.advise(_obs(1, fast=99.0, source_id="source-a"))
    assert first_source.intent == "FLAT"

    first_new_source_bar = teacher.advise(
        _obs(2, fast=101.0, source_id="source-b"),
    )
    assert first_new_source_bar.intent == "FLAT"
    assert first_new_source_bar.reason_codes == ("INDICATOR_PREVIOUS_NOT_READY",)

    second_new_source_bar = teacher.advise(
        _obs(3, fast=99.0, source_id="source-b"),
    )
    assert second_new_source_bar.intent == "OPEN_SHORT"
    assert second_new_source_bar.reason_codes == ("EMA_CROSS_SHORT",)


def test_fold_change_resets_ema_cross_state_within_an_episode():
    teacher = build_teacher_1m("T01_EMA_9_21")

    first_fold = teacher.advise(_obs(1, fast=99.0, fold_id="train-fold-a"))
    assert first_fold.intent == "FLAT"

    first_next_fold_bar = teacher.advise(
        _obs(2, fast=101.0, fold_id="train-fold-b"),
    )
    assert first_next_fold_bar.intent == "FLAT"
    assert first_next_fold_bar.reason_codes == ("INDICATOR_PREVIOUS_NOT_READY",)

    second_next_fold_bar = teacher.advise(
        _obs(3, fast=99.0, fold_id="train-fold-b"),
    )
    assert second_next_fold_bar.intent == "OPEN_SHORT"
    assert second_next_fold_bar.reason_codes == ("EMA_CROSS_SHORT",)


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


@pytest.mark.parametrize(
    "missing_field",
    ("bar_high_1m", "bar_low_1m", "bar_close_1m", "atr_14"),
)
def test_x2_refuses_entry_when_exit_inputs_are_missing(missing_field: str):
    teacher = build_teacher_1m("T01_EMA_9_21", exit_variant="X2")
    teacher.advise(_obs(1, fast=99.0))

    action = teacher.advise(
        _obs(2, fast=101.0, missing_market_fields=(missing_field,))
    )
    assert action.intent == "FLAT"
    assert action.reason_codes == ("ATR_EXIT_NOT_READY",)
    assert teacher.advise(_obs(3, fast=101.0)).intent == "FLAT"

    x1 = build_teacher_1m("T01_EMA_9_21", exit_variant="X1")
    x1.advise(_obs(1, fast=99.0))
    assert x1.advise(
        _obs(2, fast=101.0, missing_market_fields=(missing_field,))
    ).intent == "OPEN_LONG"


def test_x2_raises_when_exit_inputs_disappear_during_position():
    teacher = build_teacher_1m("T01_EMA_9_21", exit_variant="X2")
    teacher.advise(_obs(1, fast=99.0))
    assert teacher.advise(
        _obs(2, fast=101.0, exposure=0.3)
    ).intent == "HOLD"

    with pytest.raises(ValueError, match="ATR exit inputs missing while a position is open"):
        teacher.advise(
            _obs(3, fast=101.0, exposure=0.3,
                 missing_market_fields=("atr_14",))
        )


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
        account={"net_exposure": 0.3, "equity": 10_000.0, "position_avg_px_open": 100.0},
        exact_market_clocks_ns=(("latest_1m_close_ts_ns", close_ts),),
    )
    policy = AtrTrailingExit()
    assert policy.evaluate(observation, 0.3) is None
    assert policy.last_close_time_ns == close_ts


@pytest.mark.parametrize(
    ("exposure", "favorable_high", "favorable_low", "favorable_close",
     "gap_high", "gap_low", "gap_close", "expected_reason"),
    (
        (0.3, 111.0, 109.0, 110.0, 109.0, 106.0, 107.0, "ATR_TRAIL_EXIT_LONG"),
        (-0.3, 91.0, 89.0, 90.0, 94.0, 91.0, 93.0, "ATR_TRAIL_EXIT_SHORT"),
    ),
)
def test_x2_checks_preserved_stop_on_first_fresh_bar_after_gap(
    exposure: float,
    favorable_high: float,
    favorable_low: float,
    favorable_close: float,
    gap_high: float,
    gap_low: float,
    gap_close: float,
    expected_reason: str,
):
    teacher = build_teacher_1m("T01_EMA_9_21", exit_variant="X2")
    teacher.advise(
        _obs(1, fast=99.0, exposure=exposure, high=100.0, low=100.0,
             close=100.0, avg_open_price=100.0),
    )
    teacher.advise(
        _obs(2, fast=99.0, exposure=exposure, high=favorable_high,
             low=favorable_low, close=favorable_close, avg_open_price=100.0),
    )
    assert teacher._atr_exit is not None
    prior_trail = teacher._atr_exit.trail
    action = teacher.advise(
        _obs(4, fast=99.0, exposure=exposure, high=gap_high, low=gap_low,
             close=gap_close, avg_open_price=100.0),
    )
    assert action.intent == "EXIT"
    assert action.reason_codes == ("ATR_TRAILING_EXIT", expected_reason)
    assert teacher._atr_exit.trail == prior_trail


def test_x2_atr_trail_updates_only_on_completed_bars_and_never_loosens():
    policy = AtrTrailingExit(3.0)
    assert policy.evaluate(_obs(1, high=101, low=99, close=100, exposure=0.3), 0.3) is None
    first_trail = policy.trail
    assert first_trail == 97.0
    assert policy.evaluate(_obs(2, high=105, low=103, close=104, exposure=0.3), 0.3) is None
    second_trail = policy.trail
    assert second_trail == 102.0
    signal = policy.evaluate(_obs(3, high=103, low=100, close=101, atr=2.0, exposure=0.3), 0.3)
    assert policy.trail == second_trail
    assert signal is not None and signal.reason_code == "ATR_TRAIL_EXIT_LONG"

    policy = AtrTrailingExit(3.0)
    assert policy.evaluate(_obs(1, high=101, low=99, close=100, exposure=-0.3), -0.3) is None
    first_short = policy.trail
    assert first_short == 103.0
    assert policy.evaluate(_obs(2, high=97, low=95, close=96, exposure=-0.3), -0.3) is None
    second_short = policy.trail
    assert second_short == 98.0
    signal = policy.evaluate(_obs(3, high=100, low=97, close=99, atr=2.0, exposure=-0.3), -0.3)
    assert policy.trail == second_short
    assert signal is not None and signal.reason_code == "ATR_TRAIL_EXIT_SHORT"

    with pytest.raises(ValueError, match="new completed"):
        policy.evaluate(_obs(3, high=100, low=97, close=99, atr=2.0, exposure=-0.3), -0.3)


@pytest.mark.parametrize(
    (
        "side", "first_high", "first_low", "first_close",
        "next_high", "next_low", "next_close",
    ),
    [
        (0.3, 110.0, 99.0, 101.0, 106.0, 100.0, 105.5),
        (-0.3, 101.0, 90.0, 99.0, 100.0, 94.0, 94.5),
    ],
)
def test_x2_initial_trail_ignores_extremes_before_first_observed_fill(
    side, first_high, first_low, first_close, next_high, next_low, next_close,
):
    policy = AtrTrailingExit(3.0)
    assert policy.evaluate(
        _obs(
            1, high=first_high, low=first_low, close=first_close,
            exposure=side,
        ),
        side,
    ) is None
    assert policy.favorable_extreme == pytest.approx(first_close)

    signal = policy.evaluate(
        _obs(
            2, high=next_high, low=next_low, close=next_close,
            exposure=side,
        ),
        side,
    )
    assert signal is None


@pytest.mark.parametrize(
    ("side", "high", "low", "close", "expected_reason"),
    [
        (0.3, 100.0, 96.0, 96.5, "ATR_TRAIL_EXIT_LONG"),
        (-0.3, 104.0, 100.0, 103.5, "ATR_TRAIL_EXIT_SHORT"),
    ],
)
def test_x2_first_observed_adverse_close_checks_native_entry_anchored_stop(
    side, high, low, close, expected_reason,
):
    policy = AtrTrailingExit(3.0)
    signal = policy.evaluate(
        _obs(
            1, high=high, low=low, close=close, exposure=side,
            avg_open_price=100.0,
        ),
        side,
    )
    assert signal is not None
    assert signal.reason_code == expected_reason


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


def test_x2_exit_bar_updates_indicator_snapshot_before_returning():
    teacher = build_teacher_1m("T01_EMA_9_21", exit_variant="X2")
    teacher.advise(_obs(1, fast=99.0))
    assert teacher.advise(_obs(2, fast=101.0)).intent == "OPEN_LONG"
    assert teacher.advise(
        _obs(3, fast=101.0, exposure=0.3)
    ).intent == "HOLD"
    exited = teacher.advise(
        _obs(4, fast=99.0, high=100.0, low=96.0, close=97.0, exposure=0.3)
    )
    assert exited.intent == "EXIT"
    assert teacher._previous == {"ema_9": 99.0, "ema_21": 100.0}

    flat = teacher.advise(
        _obs(5, fast=98.0, high=99.0, low=97.0, close=98.0)
    )
    assert flat.intent == "FLAT"
    assert flat.reason_codes == ("NO_NEW_SIGNAL",)


@pytest.mark.parametrize(
    (
        "side", "initial_high", "initial_low",
        "next_high", "next_low", "next_close", "expected_reason",
    ),
    [
        (0.3, 100.7, 99.8, 100.1, 96.8, 97.0, "ATR_TRAIL_EXIT_LONG"),
        (-0.3, 100.2, 99.3, 103.1, 99.9, 103.0, "ATR_TRAIL_EXIT_SHORT"),
    ],
)
def test_x8_first_observed_position_keeps_atr_protection_before_handoff(
    side, initial_high, initial_low, next_high, next_low, next_close,
    expected_reason,
):
    policy = AtrTrailHandoffToIndicatorExit(round_trip_cost_bps=12.0)
    assert policy.evaluate(
        _obs(
            1, high=initial_high, low=initial_low, close=100.0,
            exposure=side, avg_open_price=100.0,
        ),
        side,
    ) is None
    assert not policy.handed_off

    signal = policy.evaluate(
        _obs(
            2, high=next_high, low=next_low, close=next_close,
            exposure=side, avg_open_price=100.0,
        ),
        side,
    )
    assert signal is not None
    assert signal.reason_code == expected_reason


def test_x8_atr_trail_hands_off_to_indicator_after_four_times_cost():
    policy = AtrTrailHandoffToIndicatorExit(round_trip_cost_bps=12.0)
    assert policy.evaluate(
        _obs(1, high=100.1, low=99.9, close=100.0, exposure=0.3), 0.3,
    ) is None
    assert not policy.handed_off
    assert policy.evaluate(
        _obs(2, high=100.5, low=100.0, close=100.4, exposure=0.3), 0.3,
    ) is None
    assert policy.handed_off  # 50 bps reached the 48 bps handoff level.
    signal = policy.evaluate(
        _obs(3, high=100.6, low=96.8, close=97.0, exposure=0.3), 0.3,
    )
    assert policy.trail == pytest.approx(97.6)
    assert signal is None  # The X1 indicator exit owns the trade after handoff.


@pytest.mark.parametrize(
    ("side", "high", "low", "close", "expected_reason"),
    [
        (0.3, 100.0, 96.0, 96.5, "ATR_TRAIL_EXIT_LONG"),
        (-0.3, 104.0, 100.0, 103.5, "ATR_TRAIL_EXIT_SHORT"),
    ],
)
def test_x8_first_observed_adverse_close_checks_atr_stop_before_handoff(
    side, high, low, close, expected_reason,
):
    policy = AtrTrailHandoffToIndicatorExit(round_trip_cost_bps=12.0)
    signal = policy.evaluate(
        _obs(
            1, high=high, low=low, close=close, exposure=side,
            avg_open_price=100.0,
        ),
        side,
    )
    assert signal is not None
    assert signal.reason_code == expected_reason
    assert not policy.handed_off


def test_x8_atr_risk_exit_remains_active_before_cost_handoff():
    policy = AtrTrailHandoffToIndicatorExit(round_trip_cost_bps=12.0)
    assert policy.evaluate(
        _obs(1, high=100.1, low=99.9, close=100.0, exposure=0.3), 0.3,
    ) is None
    signal = policy.evaluate(
        _obs(2, high=100.2, low=96.9, close=97.1, exposure=0.3), 0.3,
    )
    assert signal is not None and signal.reason_code == "ATR_TRAIL_EXIT_LONG"
    assert not policy.handed_off


@pytest.mark.parametrize(
    ("side", "entry_price", "high", "low", "close"),
    [
        (1, 100.5, 101.0, 100.3, 100.55),
        (-1, 99.5, 99.7, 99.0, 99.45),
    ],
)
def test_x8_handoff_rearms_from_close_after_partial_entry_fill(
    side, entry_price, high, low, close,
):
    policy = AtrTrailHandoffToIndicatorExit(round_trip_cost_bps=12.0)
    assert policy.evaluate(
        _obs(
            1, high=100.1, low=99.9, close=100.0, exposure=0.3 * side,
            avg_open_price=100.0,
        ),
        0.3 * side,
    ) is None
    assert policy.evaluate(
        _obs(
            2, high=high, low=low, close=close, exposure=0.3 * side,
            avg_open_price=entry_price,
        ),
        0.3 * side,
    ) is None
    assert not policy.handed_off
    assert policy.favorable_extreme == pytest.approx(close)


def test_x8_handoff_is_symmetric_for_short_positions():
    policy = AtrTrailHandoffToIndicatorExit(round_trip_cost_bps=12.0)
    assert policy.evaluate(
        _obs(1, high=100.1, low=99.9, close=100.0, exposure=-0.3), -0.3,
    ) is None
    assert policy.evaluate(
        _obs(2, high=100.0, low=99.5, close=99.6, exposure=-0.3), -0.3,
    ) is None
    assert policy.handed_off
    signal = policy.evaluate(
        _obs(3, high=103.2, low=99.4, close=102.8, exposure=-0.3), -0.3,
    )
    assert signal is None

@pytest.mark.parametrize(
    (
        "side", "initial_high", "initial_low",
        "next_high", "next_low", "next_close",
    ),
    [
        (0.3, 101.0, 99.8, 100.1, 99.9, 100.1),
        (-0.3, 100.2, 99.0, 100.1, 99.9, 99.9),
    ],
)
def test_x3_first_observed_position_ignores_pre_fill_extreme(
    side, initial_high, initial_low, next_high, next_low, next_close,
):
    policy = CostAwareAtrProfitProtectExit(round_trip_cost_bps=12.0)
    assert policy.evaluate(
        _obs(
            1, high=initial_high, low=initial_low, close=100.0,
            exposure=side, avg_open_price=100.0,
        ),
        side,
    ) is None
    signal = policy.evaluate(
        _obs(
            2, high=next_high, low=next_low, close=next_close,
            exposure=side, avg_open_price=100.0,
        ),
        side,
    )
    assert signal is None
    assert not policy.activated


@pytest.mark.parametrize(
    ("side", "high", "low", "close", "expected_trail"),
    [
        (0.3, 101.0, 99.8, 100.3, 100.18),
        (-0.3, 100.2, 99.0, 99.7, 99.82),
    ],
)
def test_x3_first_position_close_activates_without_pre_fill_extreme(
    side, high, low, close, expected_trail,
):
    policy = CostAwareAtrProfitProtectExit(round_trip_cost_bps=12.0)
    signal = policy.evaluate(
        _obs(
            1, high=high, low=low, close=close, exposure=side,
            avg_open_price=100.0,
        ),
        side,
    )
    assert signal is None
    assert policy.activated
    assert policy.favorable_extreme == pytest.approx(close)
    assert policy.trail == pytest.approx(expected_trail)


def test_x3_cost_profit_lock_activates_after_cost_scaled_excursion_and_ratchets():
    policy = CostAwareAtrProfitProtectExit(round_trip_cost_bps=12.0)
    assert policy.evaluate(
        _obs(1, high=100.2, low=99.9, close=100.1, exposure=0.3), 0.3,
    ) is None
    assert not policy.activated
    assert policy.evaluate(
        _obs(2, high=100.23, low=99.9, close=100.2, exposure=0.3), 0.3,
    ) is None
    assert not policy.activated  # 23 bps is below the 24 bps activation point.
    assert policy.evaluate(
        _obs(3, high=100.25, low=100.0, close=100.20, exposure=0.3), 0.3,
    ) is None
    assert policy.activated
    assert policy.trail == pytest.approx(100.18)
    signal = policy.evaluate(
        _obs(4, high=100.25, low=100.0, close=100.17, exposure=0.3), 0.3,
    )
    assert signal is not None
    assert signal.reason_code == "COST_AWARE_ATR_PROFIT_LOCK_LONG"


@pytest.mark.parametrize(
    ("side", "entry_price", "high", "low", "close"),
    [
        (1, 100.5, 101.0, 100.3, 100.55),
        (-1, 99.5, 99.7, 99.0, 99.45),
    ],
)
def test_x3_partial_entry_rearms_from_close_not_pre_fill_extreme(
    side, entry_price, high, low, close,
):
    policy = CostAwareAtrProfitProtectExit(round_trip_cost_bps=12.0)
    assert policy.evaluate(
        _obs(1, exposure=0.3 * side, avg_open_price=100.0),
        0.3 * side,
    ) is None
    assert policy.evaluate(
        _obs(
            2, exposure=0.3 * side, avg_open_price=entry_price,
            high=high, low=low, close=close,
        ),
        0.3 * side,
    ) is None
    assert not policy.activated
    assert policy.favorable_extreme == pytest.approx(close)


def test_x3_cost_profit_lock_is_symmetric_and_stress_cost_moves_activation():
    short = CostAwareAtrProfitProtectExit(round_trip_cost_bps=12.0)
    assert short.evaluate(
        _obs(1, high=100.1, low=99.8, close=99.9, exposure=-0.3), -0.3,
    ) is None
    assert short.evaluate(
        _obs(2, high=100.1, low=99.77, close=99.8, exposure=-0.3), -0.3,
    ) is None
    assert not short.activated
    assert short.evaluate(
        _obs(3, high=100.0, low=99.75, close=99.8, exposure=-0.3), -0.3,
    ) is None
    assert short.activated
    signal = short.evaluate(
        _obs(4, high=100.0, low=99.75, close=99.83, exposure=-0.3), -0.3,
    )
    assert signal is not None
    assert signal.reason_code == "COST_AWARE_ATR_PROFIT_LOCK_SHORT"

    stress = CostAwareAtrProfitProtectExit(round_trip_cost_bps=30.0)
    assert stress.evaluate(
        _obs(1, high=100.1, low=99.9, close=100.0, exposure=0.3), 0.3,
    ) is None
    assert stress.evaluate(
        _obs(2, high=100.5, low=100.0, close=100.4, exposure=0.3), 0.3,
    ) is None
    assert not stress.activated  # Stress needs 60 bps, not the Base 24 bps.


def test_x3_preserves_the_teacher_indicator_exit_as_a_fallback():
    teacher = build_teacher_1m(
        "T01_EMA_9_21", exit_variant="X3", exit_round_trip_cost_bps=12.0,
    )
    teacher.advise(_obs(1, fast=99.0))
    assert teacher.advise(_obs(2, fast=101.0)).intent == "OPEN_LONG"
    assert teacher.recipe.indicator_exit_rule_id == "X3_ATR_COST_PROFIT_PROTECT_V1"
    assert teacher.advise(
        _obs(3, fast=101.0, high=100.1, low=99.9, close=100.0, exposure=0.3)
    ).intent == "HOLD"
    exited = teacher.advise(
        _obs(4, fast=99.0, high=100.1, low=99.9, close=100.0, exposure=0.3)
    )
    assert exited.intent == "EXIT"
    assert exited.reason_codes == ("INDICATOR_EXIT", "EMA_REVERSE_CROSS")


def test_x3_requires_scenario_cost_and_x2_keeps_its_original_policy_id():
    with pytest.raises(ValueError, match="scenario round-trip cost"):
        build_teacher_1m("T01_EMA_9_21", exit_variant="X3")
    teacher = build_teacher_1m(
        "T01_EMA_9_21", exit_variant="X2",
    )
    assert teacher.recipe.indicator_exit_rule_id == "X2_ATR_TRAIL_3_V1"


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



def _order_facts(
    *,
    statuses_after: tuple[dict[str, str], ...],
    submitted: tuple[str, ...] = (),
    filled: tuple[str, ...] = (),
    related: tuple[str, ...] = (),
    pending: tuple[str, ...] = (),
    fills: tuple[dict[str, object], ...] = (),
    statuses_before: tuple[dict[str, str], ...] = (),
) -> L3OrderFacts:
    before = _validate_order_statuses(statuses_before, "before")
    after = _validate_order_statuses(statuses_after, "after")
    return L3OrderFacts(
        submitted_order_ids=submitted,
        filled_order_ids=filled,
        related_existing_order_ids=related,
        pending_order_ids_after=pending,
        native_evidence_receipt="native-evidence-receipt",
        native_order_statuses_before=before,
        native_order_statuses_after=after,
        native_order_status_receipt=_order_status_receipt(before, after),
        native_policy_fills=fills,
    )


def _order_status(
    order_id: str, status: str, side: str, quantity: str = "0.3",
    filled_qty: str = "0",
) -> dict[str, str]:
    return {
        "client_order_id": order_id, "status": status, "side": side,
        "quantity": quantity, "filled_qty": filled_qty,
    }


def test_ema_reversal_waits_for_native_close_fill_then_rechecks_next_bar():
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=99.0))
    assert teacher.advise(_obs(2, fast=101.0)).intent == "OPEN_LONG"

    close_request = teacher.advise(_obs(3, fast=99.0, exposure=0.3))
    assert close_request.intent == "EXIT"
    assert close_request.reason_codes[-1] == "EMA_REVERSE_CROSS"
    replay = teacher.advise(_obs(3, fast=99.0, exposure=0.3))
    assert replay.intent == "HOLD"
    assert replay.reason_codes == ("NO_NEW_BAR",)
    assert teacher._pending_reversal is not None
    first_status = (_order_status("close-1", "NEW", "SELL"),)
    teacher.observe_native_action(
        close_request,
        _trace(3, "SUBMIT", 0.3, submitted=("close-1",), pending=("close-1",)),
        _order_facts(
            statuses_after=first_status, submitted=("close-1",), pending=("close-1",),
        ),
    )

    pending = teacher.advise(
        _obs(4, fast=99.0, exposure=0.3, pending=("close-1",)),
    )
    assert pending.intent == "HOLD"
    assert pending.reason_codes == ("PENDING_ACK",)
    filled_status = (_order_status("close-1", "FILLED", "SELL", filled_qty="0.3"),)
    fill_facts = _order_facts(
        statuses_before=first_status, statuses_after=filled_status,
        filled=("close-1",), related=("close-1",),
        fills=({"event_id": "fill-1", "client_order_id": "close-1",
                "ts_event": 4 * MINUTE_NS},),
    )
    teacher.observe_native_action(
        pending,
        _trace(4, "WAIT", 0.0, related=("close-1",), filled=("close-1",)),
        fill_facts,
    )

    reverse = teacher.advise(_obs(5, fast=99.0, exposure=0.0))
    assert reverse.intent == "OPEN_SHORT"
    assert reverse.reason_codes == ("EMA_REVERSAL_RECHECK",)


def test_ema_exit_only_mode_preserves_the_consumed_cross_semantics():
    teacher = build_teacher_1m("T01_EMA_9_21", reversal_mode="EXIT_ONLY_V1")
    teacher.advise(_obs(1, fast=99.0))
    assert teacher.advise(_obs(2, fast=101.0)).intent == "OPEN_LONG"
    assert teacher.advise(_obs(3, fast=99.0, exposure=0.3)).intent == "EXIT"
    assert teacher.advise(_obs(4, fast=99.0, exposure=0.0)).intent == "FLAT"


def test_flat_exposure_without_matching_fill_proof_cannot_trigger_reversal():
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=99.0))
    teacher.advise(_obs(2, fast=101.0))
    exit_advice = teacher.advise(_obs(3, fast=99.0, exposure=0.3))
    status = (_order_status("close-1", "FILLED", "SELL", filled_qty="0.3"),)
    teacher.observe_native_action(
        exit_advice,
        _trace(3, "SUBMIT", 0.0, submitted=("close-1",), filled=("close-1",)),
        _order_facts(statuses_after=status, submitted=("close-1",), filled=("close-1",)),
    )
    result = teacher.advise(_obs(4, fast=99.0, exposure=0.0))
    assert result.intent == "FLAT"
    assert result.reason_codes == ("WAIT_NATIVE_CLOSE_ACK",)
    assert result.intent not in ("OPEN_LONG", "OPEN_SHORT")
    assert teacher._pending_reversal is not None


def test_ema_partial_close_reject_and_risk_override_never_reopen():
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=99.0))
    teacher.advise(_obs(2, fast=101.0))
    exit_advice = teacher.advise(_obs(3, fast=99.0, exposure=0.3))
    new_status = (_order_status("close-1", "NEW", "SELL"),)
    teacher.observe_native_action(
        exit_advice,
        _trace(3, "SUBMIT", 0.3, submitted=("close-1",), pending=("close-1",)),
        _order_facts(statuses_after=new_status, submitted=("close-1",), pending=("close-1",)),
    )
    partial_status = (_order_status("close-1", "PARTIALLY_FILLED", "SELL", filled_qty="0.1"),)
    teacher.observe_native_action(
        ExpertAdvice(teacher.expert_id, 4 * MINUTE_NS, "HOLD", 0.3, ("PENDING_ACK",)),
        _trace(4, "WAIT", 0.2, related=("close-1",), pending=("close-1",)),
        _order_facts(
            statuses_before=new_status, statuses_after=partial_status,
            related=("close-1",), pending=("close-1",),
            fills=({"event_id": "partial-1", "client_order_id": "close-1",
                    "ts_event": 4 * MINUTE_NS},),
        ),
    )
    assert teacher._pending_reversal is not None

    teacher.observe_native_action(
        ExpertAdvice("recovery_v1", 5 * MINUTE_NS, "EXIT", 0.0, ("MAX_DRAWDOWN",)),
        _trace(5, "SUBMIT", 0.0, submitted=("risk-close",), filled=("risk-close",)),
        _order_facts(
            statuses_after=(_order_status("risk-close", "FILLED", "SELL", filled_qty="0.3"),),
            submitted=("risk-close",), filled=("risk-close",),
        ),
    )
    assert teacher._pending_reversal is None
    assert teacher.advise(_obs(5, fast=99.0, exposure=0.0)).intent == "FLAT"


def test_ema_reversal_candidate_clears_on_gap_and_episode_change():
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=99.0))
    teacher.advise(_obs(2, fast=101.0))
    teacher.advise(_obs(3, fast=99.0, exposure=0.3))
    teacher.advise(_obs(5, fast=99.0, exposure=0.3))
    assert teacher._pending_reversal is None

    teacher.advise(_obs(6, fast=99.0))
    assert teacher._pending_reversal is None
    teacher.advise(_obs(7, fast=99.0))
    teacher.advise(_obs(8, fast=101.0))
    teacher.advise(_obs(9, fast=99.0, exposure=0.3))
    teacher.reset("new-episode")
    assert teacher._pending_reversal is None


def test_ema_reversal_supports_short_to_long_after_proven_close():
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=101.0))
    assert teacher.advise(_obs(2, fast=99.0)).intent == "OPEN_SHORT"
    close_request = teacher.advise(_obs(3, fast=101.0, exposure=-0.3))
    assert close_request.intent == "EXIT"
    status = (_order_status("close-short", "FILLED", "BUY", filled_qty="0.3"),)
    teacher.observe_native_action(
        close_request,
        _trace(3, "SUBMIT", 0.0, submitted=("close-short",), filled=("close-short",)),
        _order_facts(
            statuses_after=status, submitted=("close-short",), filled=("close-short",),
            fills=({"event_id": "fill-short", "client_order_id": "close-short",
                    "ts_event": 3 * MINUTE_NS},),
        ),
    )
    reverse = teacher.advise(_obs(4, fast=101.0, exposure=0.0))
    assert reverse.intent == "OPEN_LONG"
    assert reverse.reason_codes == ("EMA_REVERSAL_RECHECK",)


@pytest.mark.parametrize(
    ("initial_fast", "entry_fast", "close_fast", "exposure", "close_side", "reverse_intent"),
    (
        (99.0, 101.0, 99.0, 0.3, "SELL", "OPEN_SHORT"),
        (101.0, 99.0, 101.0, -0.3, "BUY", "OPEN_LONG"),
    ),
)
@pytest.mark.parametrize("failed_status", ("REJECTED", "CANCELED", "EXPIRED"))
def test_ema_reversal_retries_failed_close_then_reverses_once(
    initial_fast: float,
    entry_fast: float,
    close_fast: float,
    exposure: float,
    close_side: str,
    reverse_intent: str,
    failed_status: str,
):
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=initial_fast))
    opened = teacher.advise(_obs(2, fast=entry_fast))
    assert opened.intent == ("OPEN_LONG" if exposure > 0 else "OPEN_SHORT")
    exit_advice = teacher.advise(_obs(3, fast=close_fast, exposure=exposure))
    failed = (_order_status("close-failed", failed_status, close_side),)
    teacher.observe_native_action(
        exit_advice,
        _trace(3, "SUBMIT", exposure, submitted=("close-failed",)),
        _order_facts(statuses_after=failed, submitted=("close-failed",)),
    )
    assert teacher._pending_reversal is not None
    assert teacher._pending_reversal.close_order_ids == set()

    retry = teacher.advise(_obs(4, fast=close_fast, exposure=exposure))
    assert retry.intent == "EXIT"
    assert retry.reason_codes == ("EMA_REVERSE_CLOSE_RETRY",)
    filled = (_order_status("close-retry", "FILLED", close_side, filled_qty="0.3"),)
    teacher.observe_native_action(
        retry,
        _trace(
            4, "SUBMIT", 0.0, submitted=("close-retry",),
            filled=("close-retry",),
        ),
        _order_facts(
            statuses_after=(*failed, *filled),
            submitted=("close-retry",), filled=("close-retry",),
            fills=({
                "event_id": "close-retry-fill",
                "client_order_id": "close-retry",
                "ts_event": 4 * MINUTE_NS,
            },),
        ),
    )

    reverse = teacher.advise(_obs(5, fast=close_fast, exposure=0.0))
    assert reverse.intent == reverse_intent
    assert reverse.reason_codes == ("EMA_REVERSAL_RECHECK",)
    assert teacher._pending_reversal is None
    assert teacher.advise(_obs(5, fast=close_fast, exposure=0.0)).reason_codes == (
        "NO_NEW_BAR",
    )


def test_ema_reversal_waits_past_ack_delay_then_rechecks_after_native_fill():
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=99.0))
    teacher.advise(_obs(2, fast=101.0))
    close_request = teacher.advise(_obs(3, fast=99.0, exposure=0.3))
    first_status = (_order_status("close-1", "NEW", "SELL"),)
    teacher.observe_native_action(
        close_request,
        _trace(3, "SUBMIT", 0.3, submitted=("close-1",), pending=("close-1",)),
        _order_facts(
            statuses_after=first_status, submitted=("close-1",), pending=("close-1",),
        ),
    )

    for minute in range(4, 10):
        pending = teacher.advise(
            _obs(minute, fast=99.0, exposure=0.3, pending=("close-1",)),
        )
        assert pending.intent == "HOLD"
        assert pending.reason_codes == ("PENDING_ACK",)
        teacher.observe_native_action(
            pending,
            _trace(minute, "WAIT", 0.3, related=("close-1",), pending=("close-1",)),
            _order_facts(
                statuses_before=first_status, statuses_after=first_status,
                related=("close-1",), pending=("close-1",),
            ),
        )
        assert teacher._pending_reversal is not None

    waiting_fill = teacher.advise(
        _obs(10, fast=99.0, exposure=0.3, pending=("close-1",)),
    )
    filled_status = (_order_status("close-1", "FILLED", "SELL", filled_qty="0.3"),)
    teacher.observe_native_action(
        waiting_fill,
        _trace(10, "WAIT", 0.0, related=("close-1",), filled=("close-1",)),
        _order_facts(
            statuses_before=first_status, statuses_after=filled_status,
            filled=("close-1",), related=("close-1",),
            fills=({"event_id": "fill-late", "client_order_id": "close-1",
                    "ts_event": 10 * MINUTE_NS},),
        ),
    )
    reverse = teacher.advise(_obs(11, fast=99.0, exposure=0.0))
    assert reverse.intent == "OPEN_SHORT"
    assert reverse.reason_codes == ("EMA_REVERSAL_RECHECK",)


@pytest.mark.parametrize(
    ("initial_fast", "cross_fast", "exposure", "invalid_fast"),
    (
        (99.0, 101.0, 0.3, 101.0),
        (101.0, 99.0, -0.3, 99.0),
    ),
)
def test_ema_reversal_candidate_expires_when_direction_flips_before_recheck(
    initial_fast: float,
    cross_fast: float,
    exposure: float,
    invalid_fast: float,
):
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=initial_fast))
    opened = teacher.advise(_obs(2, fast=cross_fast))
    assert opened.intent == ("OPEN_LONG" if exposure > 0 else "OPEN_SHORT")

    close = teacher.advise(_obs(3, fast=initial_fast, exposure=exposure))
    assert close.intent == "EXIT"
    order_side = "SELL" if exposure > 0 else "BUY"
    status = (_order_status("close-1", "FILLED", order_side, filled_qty="0.3"),)
    teacher.observe_native_action(
        close,
        _trace(3, "SUBMIT", 0.0, submitted=("close-1",), filled=("close-1",)),
        _order_facts(
            statuses_after=status, submitted=("close-1",), filled=("close-1",),
            fills=({"event_id": "fill-1", "client_order_id": "close-1",
                    "ts_event": 3 * MINUTE_NS},),
        ),
    )

    expired = teacher.advise(_obs(4, fast=invalid_fast, exposure=0.0))
    assert expired.intent == "FLAT"
    assert expired.reason_codes == ("EMA_REVERSAL_EXPIRED",)
    assert teacher._pending_reversal is None


def test_ema_reverse_short_to_long_waits_past_ack_delay_then_rechecks_once():
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=101.0))
    assert teacher.advise(_obs(2, fast=99.0)).intent == "OPEN_SHORT"
    close_request = teacher.advise(_obs(3, fast=101.0, exposure=-0.3))
    assert close_request.intent == "EXIT"
    first_status = (_order_status("close-short", "NEW", "BUY"),)
    teacher.observe_native_action(
        close_request,
        _trace(3, "SUBMIT", -0.3, submitted=("close-short",), pending=("close-short",)),
        _order_facts(
            statuses_after=first_status, submitted=("close-short",),
            pending=("close-short",),
        ),
    )

    for minute in range(4, 10):
        pending = teacher.advise(
            _obs(minute, fast=101.0, exposure=-0.3, pending=("close-short",)),
        )
        assert pending.intent == "HOLD"
        assert pending.reason_codes == ("PENDING_ACK",)
        teacher.observe_native_action(
            pending,
            _trace(
                minute, "WAIT", -0.3, related=("close-short",),
                pending=("close-short",),
            ),
            _order_facts(
                statuses_before=first_status, statuses_after=first_status,
                related=("close-short",), pending=("close-short",),
            ),
        )
        assert teacher._pending_reversal is not None

    waiting_fill = teacher.advise(
        _obs(10, fast=101.0, exposure=-0.3, pending=("close-short",)),
    )
    filled_status = (
        _order_status("close-short", "FILLED", "BUY", filled_qty="0.3"),
    )
    teacher.observe_native_action(
        waiting_fill,
        _trace(10, "WAIT", 0.0, related=("close-short",), filled=("close-short",)),
        _order_facts(
            statuses_before=first_status, statuses_after=filled_status,
            filled=("close-short",), related=("close-short",),
            fills=({"event_id": "fill-short-late", "client_order_id": "close-short",
                    "ts_event": 10 * MINUTE_NS},),
        ),
    )

    reverse = teacher.advise(_obs(11, fast=101.0, exposure=0.0))
    assert reverse.intent == "OPEN_LONG"
    assert reverse.reason_codes == ("EMA_REVERSAL_RECHECK",)
    replay = teacher.advise(_obs(11, fast=101.0, exposure=0.0))
    assert replay.intent == "FLAT"
    assert replay.reason_codes == ("NO_NEW_BAR",)

def test_ema_reversal_is_consumed_when_native_risk_vetoes_reentry():
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=99.0))
    assert teacher.advise(_obs(2, fast=101.0)).intent == "OPEN_LONG"
    close = teacher.advise(_obs(3, fast=99.0, exposure=0.3))
    status = (_order_status("close-1", "FILLED", "SELL", filled_qty="0.3"),)
    teacher.observe_native_action(
        close,
        _trace(3, "SUBMIT", 0.0, submitted=("close-1",), filled=("close-1",)),
        _order_facts(
            statuses_after=status, submitted=("close-1",), filled=("close-1",),
            fills=({"event_id": "fill-1", "client_order_id": "close-1",
                    "ts_event": 3 * MINUTE_NS},),
        ),
    )

    reversal = teacher.advise(_obs(4, fast=99.0, exposure=0.0))
    assert reversal.intent == "OPEN_SHORT"
    risk_veto = NativeActionTrace(
        decision_ts_ns=reversal.decision_ts_ns,
        requested_action=-0.3,
        risk_adjusted_target=0.0,
        planner_target=0.0,
        actual_exposure_after=0.0,
        native_equity_before=10_000.0,
        native_equity_after=10_000.0,
        planner_action="NOOP",
        planner_reason="RISK_MAX_DRAWDOWN",
    )
    teacher.observe_native_action(reversal, risk_veto, _order_facts(statuses_after=()))

    assert teacher._pending_reversal is None
    later = teacher.advise(_obs(5, fast=99.0, exposure=0.0))
    assert later.intent == "FLAT"
    assert later.reason_codes == ("NO_NEW_SIGNAL",)

def test_ema_reversal_keeps_candidate_until_unrelated_pending_order_clears():
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=99.0))
    teacher.advise(_obs(2, fast=101.0))
    close = teacher.advise(_obs(3, fast=99.0, exposure=0.3))
    status = (_order_status("close-1", "FILLED", "SELL", filled_qty="0.3"),)
    teacher.observe_native_action(
        close,
        _trace(3, "SUBMIT", 0.0, submitted=("close-1",), filled=("close-1",)),
        _order_facts(
            statuses_after=status, submitted=("close-1",), filled=("close-1",),
            fills=({"event_id": "fill-1", "client_order_id": "close-1",
                    "ts_event": 3 * MINUTE_NS},),
        ),
    )
    waiting = teacher.advise(
        _obs(4, fast=99.0, exposure=0.0, pending=("unrelated",)),
    )
    assert waiting.intent == "FLAT"
    assert waiting.reason_codes == ("PENDING_ACK",)
    assert teacher._pending_reversal is not None
    reverse = teacher.advise(_obs(5, fast=99.0, exposure=0.0))
    assert reverse.intent == "OPEN_SHORT"
    assert teacher._pending_reversal is None


def test_confirmed_ema_close_is_discarded_if_account_is_not_flat_next_bar():
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher.advise(_obs(1, fast=99.0))
    teacher.advise(_obs(2, fast=101.0))
    close = teacher.advise(_obs(3, fast=99.0, exposure=0.3))
    status = (_order_status("close-1", "FILLED", "SELL", filled_qty="0.3"),)
    teacher.observe_native_action(
        close,
        _trace(3, "SUBMIT", 0.0, submitted=("close-1",), filled=("close-1",)),
        _order_facts(
            statuses_after=status, submitted=("close-1",), filled=("close-1",),
            fills=({"event_id": "fill-1", "client_order_id": "close-1",
                    "ts_event": 3 * MINUTE_NS},),
        ),
    )
    result = teacher.advise(
        _obs(4, fast=99.0, exposure=-0.1, pending=("manual-order",)),
    )
    assert result.intent == "HOLD"
    assert result.reason_codes == ("PENDING_ACK",)
    assert teacher._pending_reversal is None
