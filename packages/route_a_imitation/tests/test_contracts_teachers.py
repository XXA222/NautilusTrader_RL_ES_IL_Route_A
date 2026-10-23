import pytest
from rles_route_a.contracts import PointInTimeObservation, NativeActionTrace, assert_executable_exit
from rles_route_a.teachers.trend import TrendTeacher
from rles_route_a.teachers.wait import WaitTeacher
from rles_route_a.teachers.recovery import RecoveryTeacher
from rles_route_a.teachers.selector import FrozenTeacherSelector


def test_future_feature_rejected(observed):
    o = observed()
    with pytest.raises(ValueError, match="future"):
        PointInTimeObservation(
            o.episode_id,
            o.decision_ts_ns,
            o.market,
            {**o.feature_available_at_ns, "trend_1h": o.decision_ts_ns + 1},
            o.account,
        )


def test_pending_native_ids_unique(observed):
    with pytest.raises(ValueError, match="duplicate"):
        observed(pending=("x", "x"))


def test_trend_and_risk_recovery_teacher(observed):
    trend = TrendTeacher()
    selector = FrozenTeacherSelector(trend, RecoveryTeacher())
    assert trend.advise(observed(pos=0.0)).requested_exposure == 0.5
    a = selector.advise(observed(pos=-1.0))
    assert a.requested_exposure == -0.8
    assert a.intent == "REDUCE"
    a = selector.advise(observed(pos=0.5, pnl=-0.08))
    assert a.intent == "EXIT" and a.reason_codes == ("POSITION_STOP_LOSS",)
    a = selector.advise(observed(pos=0.0, dd=0.30))
    assert a.intent == "FLAT"


def test_trend_teacher_can_hold_exposure_during_neutral_alignment(observed):
    market_neutral = observed(pos=0.4, hourly=0.0, daily=0.0)
    assert TrendTeacher().advise(market_neutral).requested_exposure == 0.0

    holding = TrendTeacher(hold_on_neutral=True).advise(market_neutral)
    assert holding.requested_exposure == 0.4
    assert holding.reason_codes == ("HOLD_NEUTRAL_TREND",)

    opposite = TrendTeacher(hold_on_neutral=True).advise(
        observed(pos=0.4, hourly=-0.01, daily=-0.02)
    )
    assert opposite.requested_exposure == -0.5
    assert opposite.reason_codes == ("MTF_TREND_DOWN",)


def test_trend_teacher_supports_frozen_countertrend_direction(observed):
    countertrend = TrendTeacher(direction=-1)
    bullish_market = observed(pos=0.0, hourly=0.01, daily=0.02)
    bearish_market = observed(pos=0.0, hourly=-0.01, daily=-0.02)
    assert countertrend.advise(bullish_market).requested_exposure == -0.5
    assert countertrend.advise(bearish_market).requested_exposure == 0.5

    with pytest.raises(ValueError, match="invalid frozen trend parameters"):
        TrendTeacher(direction=0)


def test_trend_teacher_switches_direction_only_for_strong_aligned_trend(observed):
    adaptive = TrendTeacher(regime_strength_threshold=0.015)
    weak_alignment = adaptive.advise(observed(hourly=0.01, daily=0.02))
    strong_alignment = adaptive.advise(observed(hourly=0.02, daily=0.03))
    assert weak_alignment.requested_exposure == -0.5
    assert strong_alignment.requested_exposure == 0.5


def test_wait_is_not_profitable_alpha_claim(observed):
    a = WaitTeacher().advise(observed(pos=0.2))
    assert a.intent == "EXIT" and a.confidence is None


def test_native_exit_trace_verification(observed):
    a = WaitTeacher().advise(observed(pos=0.4))
    base = dict(
        decision_ts_ns=a.decision_ts_ns,
        requested_action=0.0,
        risk_adjusted_target=0.0,
        planner_target=0.0,
        native_equity_before=1000.0,
        native_equity_after=999.0,
        submitted_order_ids=("close1",),
        filled_order_ids=("close1",),
        planner_action="SUBMIT",
    )
    trace = NativeActionTrace(actual_exposure_after=0.0, **base)
    assert_executable_exit(a, trace, before_exposure=0.4)
    with pytest.raises(AssertionError, match="position remains"):
        assert_executable_exit(
            a, NativeActionTrace(actual_exposure_after=0.4, **base), before_exposure=0.4
        )
    with pytest.raises(AssertionError, match="chain"):
        assert_executable_exit(
            a,
            NativeActionTrace(actual_exposure_after=0.4, **{**base, "planner_action": "CANCEL"}),
            before_exposure=0.4,
        )
    with pytest.raises(AssertionError, match="fill proof"):
        assert_executable_exit(
            a,
            NativeActionTrace(
                actual_exposure_after=0.0,
                **{
                    **base,
                    "submitted_order_ids": ("close1",),
                    "filled_order_ids": ("old-entry",),
                    "related_existing_order_ids": ("old-entry",),
                },
            ),
            before_exposure=0.4,
        )
    with pytest.raises(AssertionError, match="planner did not preserve"):
        assert_executable_exit(
            a,
            NativeActionTrace(
                actual_exposure_after=0.0,
                **{
                    **base,
                    "planner_target": 0.2,
                },
            ),
            before_exposure=0.4,
        )
    with pytest.raises(ValueError, match="finite native exposure"):
        assert_executable_exit(a, trace, before_exposure=float("nan"))
