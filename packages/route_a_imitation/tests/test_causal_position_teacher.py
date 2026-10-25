from dataclasses import replace

import pytest

from rles_route_a.teachers.causal_position import CausalPositionTeacher
from rles_route_a.teachers.recovery import RecoveryTeacher
from rles_route_a.teachers.selector import FrozenTeacherSelector
from rles_route_a.teachers.wait import WaitTeacher


def _with_position(observation, *, duration_ns: float, avg_px: float = 100.0):
    return replace(observation, account={
        **observation.account,
        "position_duration_ns": duration_ns,
        "position_avg_px_open": avg_px,
    })


@pytest.mark.parametrize("age_minutes", (10, 60, 240, 480, 1440, 2880))
def test_p0_holds_native_position_without_age_exit_and_exits_on_invalidation(
        observed, age_minutes):
    teacher = CausalPositionTeacher()
    held = _with_position(
        observed(ts=900_000_000_000_000, pos=.25, hourly=.01, daily=.02),
        duration_ns=age_minutes * 60 * 1_000_000_000,
    )
    advice = teacher.advise(held)
    assert advice.intent == "HOLD"
    assert advice.requested_exposure == .25
    assert advice.reason_codes == ("TREND_CONTINUATION_HOLD",)

    invalidated = _with_position(
        observed(ts=held.decision_ts_ns + 15 * 60 * 1_000_000_000,
                 pos=.25, hourly=-.01, daily=-.02),
        duration_ns=held.account_value("position_duration_ns") + 15 * 60 * 1_000_000_000,
    )
    exited = teacher.advise(invalidated)
    assert exited.intent == "EXIT"
    assert exited.requested_exposure == 0.0
    assert exited.reason_codes == ("TREND_INVALIDATED_EXIT",)


def test_p0_flat_policy_does_not_always_trade_and_uses_causal_candidate(observed):
    teacher = CausalPositionTeacher()
    no_signal = teacher.advise(observed(pos=0.0, hourly=0.0, daily=0.0))
    assert no_signal.intent == "FLAT"

    candidate_obs = observed(pos=0.0, hourly=.01, daily=.02)
    candidate_obs = replace(candidate_obs, market={
        **candidate_obs.market,
        "breakout_1h": .02,
    })
    opened = teacher.advise(candidate_obs)
    assert opened.intent == "OPEN_LONG"
    assert opened.requested_exposure == .3
    assert opened.reason_codes == ("CANDIDATE_BREAKOUT",)


def test_p0_reversal_closes_first_and_opens_opposite_only_after_flat(observed):
    teacher = CausalPositionTeacher()
    held_long = _with_position(
        observed(pos=.25, hourly=-.01, daily=-.02),
        duration_ns=20 * 60 * 1_000_000_000,
    )
    held_long = replace(held_long, market={**held_long.market, "breakout_1h": -.03})
    close = teacher.advise(held_long)
    assert close.intent == "EXIT"
    assert close.requested_exposure == 0.0
    assert close.reason_codes == ("TREND_INVALIDATED_EXIT",)

    flat_next_barrier = observed(
        ts=held_long.decision_ts_ns + 60 * 1_000_000_000,
        pos=0.0, hourly=-.01, daily=-.02,
    )
    flat_next_barrier = replace(
        flat_next_barrier,
        market={**flat_next_barrier.market, "breakout_1h": -.03},
        account={**flat_next_barrier.account, "native_last_fill_age_seconds": 60.0},
    )
    cooldown = teacher.advise(flat_next_barrier)
    assert cooldown.intent == "FLAT"
    assert cooldown.reason_codes == ("ENTRY_COOLDOWN",)

    flat_after_cooldown = replace(
        flat_next_barrier,
        decision_ts_ns=flat_next_barrier.decision_ts_ns + 5 * 60 * 1_000_000_000,
        account={**flat_next_barrier.account, "native_last_fill_age_seconds": 300.0},
    )
    flat_after_cooldown = replace(
        flat_after_cooldown,
        feature_available_at_ns={
            key: min(ts, flat_after_cooldown.decision_ts_ns)
            for key, ts in flat_after_cooldown.feature_available_at_ns.items()
        },
        snapshot_id="a" * 64,
    )
    reverse = teacher.advise(flat_after_cooldown)
    assert reverse.intent == "OPEN_SHORT"
    assert reverse.requested_exposure == -.3


def test_p0_cooldown_suppresses_reentry_until_five_minutes_after_native_fill(observed):
    teacher = CausalPositionTeacher()
    observation = observed(pos=0.0)
    observation = replace(observation, market={
        **observation.market, "breakout_1h": .02,
    }, account={
        **observation.account, "native_last_fill_age_seconds": 299.0,
    })
    cooling = teacher.advise(observation)
    assert cooling.intent == "FLAT"
    assert cooling.reason_codes == ("ENTRY_COOLDOWN",)

    elapsed = replace(observation, account={
        **observation.account, "native_last_fill_age_seconds": 300.0,
    })
    opened = teacher.advise(elapsed)
    assert opened.intent == "OPEN_LONG"


def test_selector_hard_risk_preempts_p0_continuation_and_pending_order(observed):
    selector = FrozenTeacherSelector(
        primary=CausalPositionTeacher(), recovery=RecoveryTeacher(), wait=WaitTeacher(),
    )
    obs = _with_position(
        observed(pos=.25, pnl=-.08, pending=("pending-1",)),
        duration_ns=600_000_000_000,
    )
    advice = selector.advise(obs)
    assert advice.intent == "EXIT"
    assert advice.requested_exposure == 0.0
    assert advice.reason_codes == ("POSITION_STOP_LOSS",)
