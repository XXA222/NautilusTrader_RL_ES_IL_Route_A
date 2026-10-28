from __future__ import annotations

import json
from pathlib import Path

import pytest

from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.indicators_1m import MINUTE_NS
from rles_route_a.teachers_1m import (
    FIRST_BATCH_RECIPES,
    SECOND_BATCH_RECIPES,
    build_teacher_1m,
)


DELAY_NS = 1_000_000
ROOT = Path(__file__).resolve().parents[3]


def _obs(minute: int, values: dict[str, float], *, exposure: float = 0.0):
    close_time = minute * MINUTE_NS
    decision = close_time + DELAY_NS
    market = {
        "latest_1m_close_ts_ns": close_time,
        **values,
    }
    return PointInTimeObservation(
        episode_id="second-batch-synthetic",
        decision_ts_ns=decision,
        market=market,
        feature_available_at_ns={name: decision for name in market},
        account={"net_exposure": exposure, "equity": 10_000.0},
    )


def _macd(dif: float, dea: float, hist: float) -> dict[str, float]:
    return {"macd_dif": dif, "macd_dea": dea, "macd_hist": hist}


def _rsi(value: float) -> dict[str, float]:
    return {"rsi_14": value}


def _boll(close: float, mid: float, upper: float, lower: float) -> dict[str, float]:
    return {
        "bar_close_1m": close,
        "bollinger_mid_20": mid,
        "bollinger_upper_20_2": upper,
        "bollinger_lower_20_2": lower,
    }


def _pullback(
    close: float, *, fast: float = 101.0, slow: float = 100.0,
    atr: float = 0.25, upper10: float = 101.0, lower10: float = 99.0,
) -> dict[str, float]:
    return {
        "bar_close_1m": close,
        "ema_9": fast,
        "ema_21": slow,
        "atr_14": atr,
        "donchian_upper_10": upper10,
        "donchian_lower_10": lower10,
    }


def test_second_batch_registry_is_explicit_and_diagnostic_only():
    assert tuple(recipe.teacher_id for recipe in FIRST_BATCH_RECIPES) == (
        "T01_EMA_9_21", "T02_ROC_20_60", "T03_DONCHIAN_20", "T04_ZSCORE_60",
    )
    assert tuple(recipe.teacher_id for recipe in SECOND_BATCH_RECIPES) == (
        "T05_MACD_12_26_9", "T06_RSI_14", "T07_BOLL_20_2", "T08_PULLBACK_1M",
    )
    assert all(recipe.htf_mode == "FEATURE_ONLY" for recipe in SECOND_BATCH_RECIPES)
    assert all(recipe.target_exposure == 0.3 for recipe in SECOND_BATCH_RECIPES)
    assert all(recipe.scenario_cost_ids == ("base", "stress") for recipe in SECOND_BATCH_RECIPES)
    config = json.loads((ROOT / "configs" / "route_a_1m_teacher_library_v1.json").read_text())
    config_ids = {row["teacher_id"] for row in config["teachers"]}
    assert config_ids == {recipe.teacher_id for recipe in FIRST_BATCH_RECIPES + SECOND_BATCH_RECIPES}
    exits = config["x1_rule_by_teacher"]
    for recipe in SECOND_BATCH_RECIPES:
        teacher = build_teacher_1m(recipe.teacher_id)
        assert teacher.recipe.teacher_id == recipe.teacher_id
        assert teacher.recipe.indicator_exit_rule_id == exits[recipe.teacher_id]
        assert build_teacher_1m(recipe.teacher_id, exit_variant="X2").recipe.indicator_exit_rule_id == (
            "X2_ATR_TRAIL_3_V1"
        )


def test_t05_macd_cross_and_reverse_exit_are_symmetric():
    teacher = build_teacher_1m("T05_MACD_12_26_9")
    teacher.advise(_obs(1, _macd(-0.2, 0.0, -0.2)))
    assert teacher.advise(_obs(2, _macd(0.1, 0.0, 0.1))).intent == "OPEN_LONG"
    assert teacher.advise(_obs(3, _macd(0.2, 0.0, 0.2), exposure=0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _macd(-0.1, 0.0, -0.1), exposure=0.3)).intent == "EXIT"

    teacher = build_teacher_1m("T05_MACD_12_26_9")
    teacher.advise(_obs(1, _macd(0.2, 0.0, 0.2)))
    assert teacher.advise(_obs(2, _macd(-0.1, 0.0, -0.1))).intent == "OPEN_SHORT"
    assert teacher.advise(_obs(3, _macd(-0.2, 0.0, -0.2), exposure=-0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _macd(0.1, 0.0, 0.1), exposure=-0.3)).intent == "EXIT"

def test_t06_rsi_reentry_requires_extreme_then_recovery_and_exits_neutral():
    teacher = build_teacher_1m("T06_RSI_14")
    teacher.advise(_obs(1, _rsi(29.0)))
    assert teacher.advise(_obs(2, _rsi(31.0))).intent == "OPEN_LONG"
    assert teacher.advise(_obs(3, _rsi(40.0), exposure=0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _rsi(50.0), exposure=0.3)).intent == "EXIT"

    teacher = build_teacher_1m("T06_RSI_14")
    teacher.advise(_obs(1, _rsi(71.0)))
    assert teacher.advise(_obs(2, _rsi(69.0))).intent == "OPEN_SHORT"
    assert teacher.advise(_obs(3, _rsi(60.0), exposure=-0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _rsi(50.0), exposure=-0.3)).intent == "EXIT"


def test_t07_bollinger_reentry_and_midline_exit_are_symmetric():
    teacher = build_teacher_1m("T07_BOLL_20_2")
    teacher.advise(_obs(1, _boll(98.0, 105.0, 110.0, 100.0)))
    assert teacher.advise(_obs(2, _boll(100.5, 105.0, 110.0, 100.0))).intent == "OPEN_LONG"
    assert teacher.advise(_obs(3, _boll(102.0, 105.0, 110.0, 100.0), exposure=0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _boll(105.0, 105.0, 110.0, 100.0), exposure=0.3)).intent == "EXIT"

    teacher = build_teacher_1m("T07_BOLL_20_2")
    teacher.advise(_obs(1, _boll(112.0, 105.0, 110.0, 100.0)))
    assert teacher.advise(_obs(2, _boll(109.5, 105.0, 110.0, 100.0))).intent == "OPEN_SHORT"
    assert teacher.advise(_obs(3, _boll(108.0, 105.0, 110.0, 100.0), exposure=-0.3)).intent == "HOLD"
    assert teacher.advise(_obs(4, _boll(105.0, 105.0, 110.0, 100.0), exposure=-0.3)).intent == "EXIT"


@pytest.mark.parametrize(
    ("closes", "fast", "slow", "side", "exposure"),
    [
        ([100.0, 99.9, 99.8, 99.7, 99.6], 101.0, 100.0, "OPEN_LONG", 0.3),
        ([100.0, 100.1, 100.2, 100.3, 100.4], 99.0, 100.0, "OPEN_SHORT", -0.3),
    ],
)
def test_t08_pullback_then_three_bar_recovery_and_trend_exit(
    closes: list[float], fast: float, slow: float, side: str, exposure: float
):
    teacher = build_teacher_1m("T08_PULLBACK_1M")
    teacher.advise(_obs(1, _pullback(closes[0], fast=fast, slow=slow)))
    for minute, close in enumerate(closes, start=2):
        teacher.advise(_obs(minute, _pullback(close, fast=fast, slow=slow)))
    recovery = teacher.advise(
        _obs(7, _pullback(99.9 if exposure > 0 else 100.1, fast=fast, slow=slow))
    )
    assert recovery.intent == side
    assert recovery.requested_exposure == exposure
    reverse = teacher.advise(
        _obs(8, _pullback(100.0, fast=99.0 if exposure > 0 else 101.0,
                          slow=100.0, lower10=99.0, upper10=101.0), exposure=exposure)
    )
    assert reverse.intent == "EXIT"


def test_t08_missing_bar_clears_pullback_history():
    teacher = build_teacher_1m("T08_PULLBACK_1M")
    teacher.advise(_obs(1, _pullback(100.0)))
    for minute, close in enumerate((100.0, 99.9, 99.8, 99.7), start=2):
        teacher.advise(_obs(minute, _pullback(close)))
    missing = teacher.advise(_obs(7, _pullback(99.6)))
    assert missing.reason_codes == ("MISSING_BAR",)
    teacher.advise(_obs(8, _pullback(99.6)))
    result = teacher.advise(_obs(9, _pullback(99.5)))
    assert result.intent == "FLAT"
    assert result.reason_codes == ("PULLBACK_HISTORY_NOT_READY",)
