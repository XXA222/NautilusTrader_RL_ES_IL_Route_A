from __future__ import annotations

import json
from pathlib import Path

import pytest

from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.indicators_1m import MINUTE_NS
from rles_route_a.teachers_1m import FIRST_BATCH_RECIPES, build_teacher_1m


DELAY_NS = 1_000_000
ROOT = Path(__file__).resolve().parents[3]


def _obs(
    minute: int,
    values: dict[str, float],
    *,
    exposure: float = 0.0,
    pending: tuple[str, ...] = (),
    episode: str = "synthetic",
    htf: tuple[float, float, float] = (0.1, -0.2, 0.3),
) -> PointInTimeObservation:
    close_time = minute * MINUTE_NS
    decision = close_time + DELAY_NS
    market = {
        "latest_1m_close_ts_ns": close_time,
        "trend_1h": htf[0],
        "trend_4h": htf[1],
        "trend_1d": htf[2],
        **values,
    }
    return PointInTimeObservation(
        episode_id=episode,
        decision_ts_ns=decision,
        market=market,
        feature_available_at_ns={name: decision for name in market},
        account={"net_exposure": exposure, "equity": 10_000.0},
        pending_order_ids=pending,
    )


def _ema(fast: float, slow: float) -> dict[str, float]:
    return {"ema_9": fast, "ema_21": slow}


def _roc(fast: float, slow: float) -> dict[str, float]:
    return {"roc_20": fast, "roc_60": slow}


def _donchian(close: float, *, upper20: float = 101.0, lower20: float = 99.0,
              upper10: float = 101.0, lower10: float = 99.0) -> dict[str, float]:
    return {
        "bar_close_1m": close,
        "donchian_upper_20": upper20,
        "donchian_lower_20": lower20,
        "donchian_upper_10": upper10,
        "donchian_lower_10": lower10,
    }


def _zscore(value: float) -> dict[str, float]:
    return {"zscore_60": value}


def test_first_batch_registry_and_frozen_json_are_explicit():
    assert tuple(recipe.teacher_id for recipe in FIRST_BATCH_RECIPES) == (
        "T01_EMA_9_21",
        "T02_ROC_20_60",
        "T03_DONCHIAN_20",
        "T04_ZSCORE_60",
    )
    assert all(recipe.htf_mode == "FEATURE_ONLY" for recipe in FIRST_BATCH_RECIPES)
    assert all(recipe.target_exposure == 0.3 for recipe in FIRST_BATCH_RECIPES)
    assert all(recipe.scenario_cost_ids == ("base", "stress") for recipe in FIRST_BATCH_RECIPES)
    with pytest.raises(ValueError, match="unknown"):
        build_teacher_1m("UNKNOWN")

    config = json.loads((ROOT / "configs" / "route_a_1m_teacher_library_v1.json").read_text())
    ids = [row["teacher_id"] for row in config["teachers"]]
    assert len(ids) == len(set(ids)) == 8
    assert config["timeframes"]["higher_timeframe_mode"] == "FEATURE_ONLY"
    assert config["exposure"]["max_holding_minutes"] is None
    assert config["research"]["allow_test_holdout_reads"] is False
    assert config["research"]["allow_student_training"] is False


def test_teacher_uses_exact_epoch_nanosecond_bar_clock():
    start = (1_700_000_000_000_000_000 // MINUTE_NS) * MINUTE_NS

    def exact_observation(close_ts_ns: int, fast: float, slow: float) -> PointInTimeObservation:
        market = {"ema_9": fast, "ema_21": slow}
        return PointInTimeObservation(
            episode_id="epoch-ns",
            decision_ts_ns=close_ts_ns + DELAY_NS,
            market=market,
            feature_available_at_ns={name: close_ts_ns + DELAY_NS for name in market},
            account={"net_exposure": 0.0, "equity": 10_000.0},
            exact_market_clocks_ns=(("latest_1m_close_ts_ns", close_ts_ns),),
        )

    teacher = build_teacher_1m("T01_EMA_9_21")
    teacher.advise(exact_observation(start, 99.0, 100.0))
    advice = teacher.advise(exact_observation(start + MINUTE_NS, 101.0, 100.0))
    assert advice.intent == "OPEN_LONG"

    rounded_timestamp = float(start)
    market = {"latest_1m_close_ts_ns": rounded_timestamp, "ema_9": 101.0, "ema_21": 100.0}
    unsafe = PointInTimeObservation(
        episode_id="unsafe-epoch-ns",
        decision_ts_ns=int(rounded_timestamp) + DELAY_NS,
        market=market,
        feature_available_at_ns={name: int(rounded_timestamp) + DELAY_NS for name in market},
        account={"net_exposure": 0.0, "equity": 10_000.0},
    )
    with pytest.raises(ValueError, match="requires exact integer metadata"):
        build_teacher_1m("T01_EMA_9_21").advise(unsafe)


def test_t01_ema_long_short_hold_and_indicator_exit():
    teacher = build_teacher_1m("T01_EMA_9_21")
    assert teacher.advise(_obs(1, _ema(99, 100))).intent == "FLAT"
    opened = teacher.advise(_obs(2, _ema(101, 100)))
    assert (opened.intent, opened.requested_exposure, opened.reason_codes) == (
        "OPEN_LONG", 0.3, ("EMA_CROSS_LONG",),
    )
    assert teacher.advise(_obs(3, _ema(102, 100), exposure=0.3)).intent == "HOLD"
    exited = teacher.advise(_obs(4, _ema(99, 100), exposure=0.3))
    assert exited.intent == "EXIT"
    assert exited.reason_codes == ("INDICATOR_EXIT", "EMA_REVERSE_CROSS")

    teacher = build_teacher_1m("T01_EMA_9_21")
    teacher.advise(_obs(1, _ema(101, 100)))
    assert teacher.advise(_obs(2, _ema(99, 100))).intent == "OPEN_SHORT"
    assert teacher.advise(_obs(3, _ema(98, 100), exposure=-0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _ema(101, 100), exposure=-0.3)).intent == "EXIT"


def test_t02_roc_long_short_hold_and_indicator_exit():
    teacher = build_teacher_1m("T02_ROC_20_60")
    teacher.advise(_obs(1, _roc(-0.01, 0.02)))
    assert teacher.advise(_obs(2, _roc(0.01, 0.02))).intent == "OPEN_LONG"
    assert teacher.advise(_obs(3, _roc(0.02, 0.03), exposure=0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _roc(0.0, 0.03), exposure=0.3)).intent == "EXIT"

    teacher = build_teacher_1m("T02_ROC_20_60")
    teacher.advise(_obs(1, _roc(0.01, -0.02)))
    assert teacher.advise(_obs(2, _roc(-0.01, -0.02))).intent == "OPEN_SHORT"
    assert teacher.advise(_obs(3, _roc(-0.02, -0.03), exposure=-0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _roc(0.0, -0.03), exposure=-0.3)).intent == "EXIT"


def test_t03_donchian_long_short_hold_and_indicator_exit():
    teacher = build_teacher_1m("T03_DONCHIAN_20")
    teacher.advise(_obs(1, _donchian(100)))
    assert teacher.advise(_obs(2, _donchian(102))).intent == "OPEN_LONG"
    assert teacher.advise(_obs(3, _donchian(100), exposure=0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _donchian(98), exposure=0.3)).intent == "EXIT"

    teacher = build_teacher_1m("T03_DONCHIAN_20")
    teacher.advise(_obs(1, _donchian(100)))
    assert teacher.advise(_obs(2, _donchian(98))).intent == "OPEN_SHORT"
    assert teacher.advise(_obs(3, _donchian(100), exposure=-0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _donchian(102), exposure=-0.3)).intent == "EXIT"


def test_t04_zscore_long_short_hold_exit_and_fast_miss():
    teacher = build_teacher_1m("T04_ZSCORE_60")
    teacher.advise(_obs(1, _zscore(-2.2)))
    opened = teacher.advise(_obs(2, _zscore(-1.9)))
    assert opened.intent == "OPEN_LONG"
    assert opened.reason_codes == ("ZSCORE_REENTRY_LONG",)
    assert teacher.advise(_obs(3, _zscore(-1.0), exposure=0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _zscore(0.1), exposure=0.3)).intent == "EXIT"

    teacher = build_teacher_1m("T04_ZSCORE_60")
    teacher.advise(_obs(1, _zscore(2.2)))
    assert teacher.advise(_obs(2, _zscore(1.9))).intent == "OPEN_SHORT"
    assert teacher.advise(_obs(3, _zscore(1.0), exposure=-0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _zscore(-0.1), exposure=-0.3)).intent == "EXIT"

    teacher = build_teacher_1m("T04_ZSCORE_60")
    teacher.advise(_obs(1, _zscore(-2.2)))
    missed = teacher.advise(_obs(2, _zscore(-1.7)))
    assert missed.intent == "FLAT"
    assert missed.reason_codes == ("FAST_REVERSION_MISSED",)


@pytest.mark.parametrize(("teacher_id", "before", "cross"), (
    ("T01_EMA_9_21", _ema(99, 100), _ema(101, 100)),
    ("T02_ROC_20_60", _roc(-0.01, 0.01), _roc(0.01, 0.01)),
    ("T03_DONCHIAN_20", _donchian(100), _donchian(102)),
    ("T04_ZSCORE_60", _zscore(-2.2), _zscore(-1.9)),
))
def test_feature_only_htf_changes_do_not_change_raw_entry(
    teacher_id: str, before: dict[str, float], cross: dict[str, float]
):
    first = build_teacher_1m(teacher_id)
    second = build_teacher_1m(teacher_id)
    first.advise(_obs(1, before, htf=(1.0, 1.0, 1.0)))
    second.advise(_obs(1, before, htf=(-1.0, -1.0, -1.0)))
    a = first.advise(_obs(2, cross, htf=(1.0, -1.0, 1.0)))
    b = second.advise(_obs(2, cross, htf=(-1.0, 1.0, -1.0)))
    assert (a.intent, a.requested_exposure, a.reason_codes) == (
        b.intent, b.requested_exposure, b.reason_codes
    )


@pytest.mark.parametrize(("teacher_id", "before", "cross"), (
    ("T01_EMA_9_21", _ema(99, 100), _ema(101, 100)),
    ("T02_ROC_20_60", _roc(-0.01, 0.01), _roc(0.01, 0.01)),
    ("T03_DONCHIAN_20", _donchian(100), _donchian(102)),
    ("T04_ZSCORE_60", _zscore(-2.2), _zscore(-1.9)),
))
def test_pending_ack_suppresses_native_entry_but_preserves_raw_signal(
    teacher_id: str, before: dict[str, float], cross: dict[str, float]
):
    teacher = build_teacher_1m(teacher_id)
    teacher.advise(_obs(1, before))
    result = teacher.advise(_obs(2, cross, pending=("native-order",)))
    assert result.intent == "FLAT"
    assert result.reason_codes == ("PENDING_ACK",)
    assert teacher.last_signal.side == 1


def test_one_decision_per_new_bar_gap_and_warmup_fail_closed():
    teacher = build_teacher_1m("T01_EMA_9_21")
    first_obs = _obs(1, _ema(99, 100))
    assert teacher.advise(first_obs).reason_codes == ("INDICATOR_PREVIOUS_NOT_READY",)
    assert teacher.advise(first_obs).reason_codes == ("NO_NEW_BAR",)
    missing = teacher.advise(_obs(3, _ema(101, 100)))
    assert missing.reason_codes == ("MISSING_BAR",)

    teacher = build_teacher_1m("T01_EMA_9_21")
    not_ready = _obs(1, {})
    assert teacher.advise(not_ready).reason_codes == ("INDICATOR_NOT_READY",)

