import pytest

from rles_route_a.teachers.forecast import RidgeForecastTeacher
from rles_route_a.teachers.recovery import RecoveryTeacher
from rles_route_a.teachers.selector import FrozenTeacherSelector
from rles_route_a.teachers.trend import TrendTeacher
from rles_route_a.teachers.wait import WaitTeacher


def _ridge(intercept: float, *, entry: float = 30.0, exit: float = 15.0):
    return RidgeForecastTeacher(
        feature_names=("trend_1h",),
        means=(0.0,),
        stds=(1.0,),
        coefficients=(intercept, 0.0),
        entry_hurdle_bps=entry,
        exit_hurdle_bps=exit,
        forecast_horizon_minutes=60,
    )


def test_high_volatility_vetoes_flat_entry_but_does_not_flatten_position(observed):
    selector = FrozenTeacherSelector(
        primary=TrendTeacher(hold_on_neutral=True),
        recovery=RecoveryTeacher(),
        max_active_vol=0.08,
        wait=WaitTeacher(),
    )

    flat = selector.advise(observed(pos=0.0, vol=0.10))
    assert flat.requested_exposure == 0.0
    assert flat.reason_codes == ("ENTRY_VETO_HIGH_VOL",)

    held = selector.advise(observed(pos=0.25, vol=0.10))
    assert held.intent == "HOLD"
    assert held.requested_exposure == 0.25
    assert held.reason_codes == ("HOLD_HIGH_VOL",)


def test_hard_risk_overrides_pending_order_wait(observed):
    selector = FrozenTeacherSelector(
        primary=TrendTeacher(), recovery=RecoveryTeacher(), wait=WaitTeacher()
    )
    advice = selector.advise(observed(pos=0.25, pnl=-0.08, pending=("entry-1",)))
    assert advice.intent == "EXIT"
    assert advice.requested_exposure == 0.0
    assert advice.reason_codes == ("POSITION_STOP_LOSS",)


def test_pending_order_blocks_direct_teacher_reentry(observed):
    advice = TrendTeacher().advise(observed(pos=0.0, pending=("open-1",)))
    assert advice.intent == "FLAT"
    assert advice.requested_exposure == 0.0
    assert advice.reason_codes == ("PENDING_ACK",)


def test_net_position_reversal_closes_before_reopening(observed):
    advice = TrendTeacher().advise(observed(pos=-0.25, hourly=0.01, daily=0.02))
    assert advice.intent == "EXIT"
    assert advice.requested_exposure == 0.0
    assert advice.reason_codes[0] == "REVERSE_CLOSE_FIRST"


def test_forecast_holds_beyond_prediction_horizon_without_a_time_stop(observed):
    teacher = _ridge(20.0)
    advice = teacher.advise(
        observed(ts=1_000_000_000 + 600 * 60_000_000_000, pos=0.25)
    )
    assert advice.intent == "HOLD"
    assert advice.requested_exposure == 0.25
    assert advice.reason_codes == ("FORECAST_HOLDS_LONG",)


def test_forecast_can_exit_early_and_reversal_waits_for_next_barrier(observed):
    early_exit = _ridge(-20.0).advise(
        observed(ts=1_000_000_000 + 15 * 60_000_000_000, pos=0.25)
    )
    assert early_exit.intent == "EXIT"
    assert early_exit.requested_exposure == 0.0
    assert early_exit.reason_codes == ("FORECAST_BELOW_EXIT_HURDLE",)

    reverse = _ridge(-35.0).advise(observed(pos=0.25))
    assert reverse.intent == "EXIT"
    assert reverse.requested_exposure == 0.0
    assert reverse.reason_codes == ("FORECAST_REVERSE_CLOSE_FIRST",)


@pytest.mark.parametrize(
    "prediction,expected",
    [(29.9, "FLAT"), (30.0, "OPEN_LONG"), (-30.0, "OPEN_SHORT")],
)
def test_flat_forecast_obeys_frozen_cost_hurdle(observed, prediction, expected):
    advice = _ridge(prediction).advise(observed(pos=0.0))
    assert advice.intent == expected
    assert advice.requested_exposure == (0.0 if expected == "FLAT" else (0.2 if prediction > 0 else -0.2))
