import pytest
from rles_route_a.teacher_research.limit_entry import LimitBar, replay_limit_entry


def test_limit_touch_and_cancel_ordering():
    bars = (LimitBar(10, 20, 95., 105.), LimitBar(20, 30, 98., 102.),
            LimitBar(30, 40, 90., 110.))
    # First bar began before decision, so its touch is unusable.
    result = replay_limit_entry(decision_ts_ns=15, side=1, limit_price=99.,
                                bars=bars, block_end_ns=50)
    assert (result.status, result.fill_ts_ns, result.fill_price) == ("FILLED", 30, 99.)
    result = replay_limit_entry(decision_ts_ns=15, side=1, limit_price=99.,
                                bars=bars, block_end_ns=50, cancel_at_ns=25)
    assert result.status == "RIGHT_CENSORED" and result.fill_price is None
    result = replay_limit_entry(decision_ts_ns=15, side=1, limit_price=99.,
                                bars=bars, block_end_ns=50, cancel_at_ns=20)
    assert result.status == "CANCELLED" and result.fill_price is None
    result = replay_limit_entry(decision_ts_ns=15, side=-1, limit_price=120.,
                                bars=bars, block_end_ns=50)
    assert result.status == "RIGHT_CENSORED" and result.fill_ts_ns is None


def test_limit_invalid_overlapping_bars():
    with pytest.raises(ValueError, match="overlapping"):
        replay_limit_entry(decision_ts_ns=1, side=1, limit_price=1.,
            bars=(LimitBar(2, 5, 2., 3.), LimitBar(4, 6, 2., 3.)), block_end_ns=10)



def test_missing_bars_end_observation_without_claiming_cancellation():
    result = replay_limit_entry(decision_ts_ns=1, side=1, limit_price=90.,
        bars=(LimitBar(2, 3, 100., 110.), LimitBar(5, 6, 80., 110.)),
        cancel_at_ns=8, block_end_ns=10)
    assert result.status == "RIGHT_CENSORED" and result.observed_until_ns == 2


def test_complete_no_touch_and_expiry_preserve_distinct_no_entry_states():
    complete = (LimitBar(10, 20, 100., 110.), LimitBar(20, 30, 100., 110.),
                LimitBar(30, 50, 100., 110.))
    result = replay_limit_entry(decision_ts_ns=20, side=1, limit_price=90.,
                                bars=complete, block_end_ns=50)
    assert (result.status, result.observed_until_ns) == ("MISSED", 50)

    expiring = (LimitBar(10, 20, 100., 110.), LimitBar(20, 30, 100., 110.),
                LimitBar(30, 40, 100., 110.))
    result = replay_limit_entry(decision_ts_ns=20, side=1, limit_price=90.,
        bars=expiring, block_end_ns=50, cancel_at_ns=40, cancel_status="UNFILLED")
    assert (result.status, result.observed_until_ns) == ("UNFILLED", 40)

    result = replay_limit_entry(decision_ts_ns=20, side=1, limit_price=90.,
        bars=expiring, block_end_ns=50, cancel_at_ns=40)
    assert (result.status, result.observed_until_ns) == ("CANCELLED", 40)

    touched_at_boundary = (LimitBar(10, 20, 100., 110.), LimitBar(20, 30, 100., 110.),
                           LimitBar(30, 50, 80., 110.))
    result = replay_limit_entry(decision_ts_ns=20, side=1, limit_price=90.,
                                bars=touched_at_boundary, block_end_ns=50)
    assert (result.status, result.observed_until_ns, result.fill_price) == ("MISSED", 50, None)


def test_limit_entry_allows_next_bar_at_decision_and_censors_an_initial_gap():
    contiguous = replay_limit_entry(decision_ts_ns=20, side=1, limit_price=99.,
        bars=(LimitBar(10, 20, 100., 110.), LimitBar(20, 30, 98., 105.)),
        block_end_ns=40)
    assert (contiguous.status, contiguous.fill_ts_ns, contiguous.fill_price) == ("FILLED", 30, 99.)

    missing = replay_limit_entry(decision_ts_ns=20, side=1, limit_price=99.,
        bars=(LimitBar(30, 40, 80., 110.),), block_end_ns=50)
    assert (missing.status, missing.observed_until_ns, missing.fill_ts_ns) == (
        "RIGHT_CENSORED", 30, None)
