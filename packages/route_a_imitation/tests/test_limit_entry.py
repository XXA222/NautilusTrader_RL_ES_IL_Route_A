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
