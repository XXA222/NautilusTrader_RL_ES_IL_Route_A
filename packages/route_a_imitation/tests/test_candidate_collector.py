from dataclasses import replace
from hashlib import sha256

import pytest

from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.indicators_1m import MINUTE_NS
from rles_route_a.teacher_research.candidate_collector import OneMinuteCandidateCollector


def observation(minute, *, exposure=0.0, pending=(), **changes):
    ts = minute * MINUTE_NS
    market = {
        "latest_1m_close_ts_ns": float(ts),
        "ema_9": 99.0, "ema_21": 100.0,
        "roc_20": -1.0, "roc_60": -1.0,
        "bar_close_1m": 100.0, "atr_14": 1.0,
        "donchian_upper_20": 102.0, "donchian_lower_20": 98.0,
        "donchian_upper_10": 101.0, "donchian_lower_10": 99.0,
        "zscore_60": 0.0, "macd_dif": -1.0, "macd_dea": 0.0,
        "macd_hist": -1.0, "rsi_14": 72.0,
        "bollinger_mid_20": 100.0, "bollinger_upper_20_2": 102.0,
        "bollinger_lower_20_2": 98.0,
    }
    market.update(changes)
    return PointInTimeObservation(
        episode_id="e", fold_id="train", source_id="synthetic",
        snapshot_id=sha256(str(minute).encode()).hexdigest(),
        feature_spec_hash="0" * 64, decision_ts_ns=ts,
        market=market, feature_available_at_ns={k: ts for k in market},
        account={"net_exposure": exposure, "equity": 10000.0},
        pending_order_ids=pending,
    )


def test_all_eight_keep_conflicting_and_blocked_candidates_with_native_snapshot():
    collector = OneMinuteCandidateCollector()
    assert len(collector.teacher_ids) == 8
    assert collector.collect(observation(1)).groups == ()
    actual = observation(2, exposure=0.3, pending=("native-order",),
                         ema_9=101., roc_20=1., roc_60=1., rsi_14=69.)
    result = collector.collect(actual)
    assert len(result.teacher_signals) == 8
    long, = (g for g in result.groups if g.side == 1)
    short, = (g for g in result.groups if g.side == -1)
    assert {t for t, _ in long.teachers} == {"T01_EMA_9_21", "T02_ROC_20_60"}
    assert short.teachers[0][0] == "T06_RSI_14"
    assert all(group.observation is actual for group in result.groups)
    assert actual.account["net_exposure"] == 0.3
    assert actual.pending_order_ids == ("native-order",)
    assert collector.collect(actual).groups == ()


def test_discontinuity_and_scope_change_do_not_bridge_signal_history():
    collector = OneMinuteCandidateCollector()
    collector.collect(observation(1))
    assert not collector.collect(observation(3, ema_9=101.)).groups
    changed = replace(observation(4, ema_9=99.), source_id="different")
    assert not collector.collect(changed).groups
    after = replace(observation(5, ema_9=101.), source_id="different")
    assert any(group.side == 1 for group in collector.collect(after).groups)


@pytest.mark.parametrize("before,after,teacher", [
    ({"bar_close_1m": 100.}, {"bar_close_1m": 103.}, "T03_DONCHIAN_20"),
    ({"zscore_60": -2.1}, {"zscore_60": -1.9}, "T04_ZSCORE_60"),
    ({"macd_dif": -1., "macd_hist": -1.}, {"macd_dif": 1., "macd_hist": 1.}, "T05_MACD_12_26_9"),
    ({"bar_close_1m": 97.}, {"bar_close_1m": 99.}, "T07_BOLL_20_2"),
])
def test_other_teacher_signals_are_collected_independently(before, after, teacher):
    collector = OneMinuteCandidateCollector()
    collector.collect(observation(1, **before))
    result = collector.collect(observation(2, **after))
    assert any(teacher in {t for t, _ in g.teachers} for g in result.groups)


def test_pullback_history_generates_candidate_during_native_pending():
    collector = OneMinuteCandidateCollector()
    for minute, close in enumerate((100., 102., 101.8, 101.6, 101.4, 101.2, 103.), 1):
        result = collector.collect(observation(minute, bar_close_1m=close,
                                                ema_9=101., pending=("order",)))
    assert any("T08_PULLBACK_1M" in {t for t, _ in g.teachers} for g in result.groups)
