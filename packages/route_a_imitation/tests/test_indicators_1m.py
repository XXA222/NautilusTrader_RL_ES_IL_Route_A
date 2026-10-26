from __future__ import annotations

import math

import numpy as np
import pytest

from rles_data.core import Availability, CuratedStore, DataError, Event
from rles_data.features import (
    CAUSAL_1M_INDICATORS_V1_INPUT_PROFILE_ID,
    causal_1m_indicator_inputs,
)
from rles_route_a.indicators_1m import (
    CausalBar,
    MINUTE_NS,
    ROUTE_A_1M_INDICATOR_RECIPES_V1,
    StreamingIndicators1m,
    aggregate_completed_bars,
    replay_indicator_prefix,
)
from rles_route_a.integration.l2_adapter import project_indicator_snapshot


DELAY_NS = 1_000_000


def _bar(index: int, close: float, *, high: float | None = None,
         low: float | None = None, available_delay_ns: int = DELAY_NS) -> CausalBar:
    close_time = (index + 1) * MINUTE_NS
    high_value = close + 1.0 if high is None else high
    low_value = close - 1.0 if low is None else low
    return CausalBar(
        open_time_ns=close_time - MINUTE_NS,
        close_time_ns=close_time,
        available_at_ns=close_time + available_delay_ns,
        open=close,
        high=high_value,
        low=low_value,
        close=close,
        volume=10.0 + index,
    )


def _replay(closes: list[float]):
    bars = tuple(_bar(index, close) for index, close in enumerate(closes))
    return bars, replay_indicator_prefix(bars)


def _reference_ema(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    result[period - 1] = float(np.mean(values[:period]))
    alpha = 2.0 / (period + 1.0)
    for index in range(period, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def test_recipe_contract_is_explicit_and_unique():
    assert len(ROUTE_A_1M_INDICATOR_RECIPES_V1) == 8
    assert len({recipe.recipe_id for recipe in ROUTE_A_1M_INDICATOR_RECIPES_V1}) == 8
    assert {recipe.warmup_bars for recipe in ROUTE_A_1M_INDICATOR_RECIPES_V1} == {
        14, 15, 20, 21, 34, 60, 61,
    }
    assert all(recipe.input_field_names and recipe.output_field_names
               for recipe in ROUTE_A_1M_INDICATOR_RECIPES_V1)


def test_snapshot_projection_keeps_epoch_nanosecond_close_clock_exact():
    epoch_close = (1_700_000_000_000_000_000 // MINUTE_NS) * MINUTE_NS
    bar = CausalBar(
        open_time_ns=epoch_close - MINUTE_NS,
        close_time_ns=epoch_close,
        available_at_ns=epoch_close + DELAY_NS,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1.0,
    )
    snapshot = StreamingIndicators1m().update(
        bar, decision_ts_ns=bar.available_at_ns,
    )
    observation = project_indicator_snapshot(
        fold_id="synthetic",
        episode_id="epoch-ns",
        snapshot=snapshot,
        native_account_fields={"net_exposure": 0.0, "equity": 10_000.0},
    )
    assert observation.exact_market_clock("latest_1m_close_ts_ns") == epoch_close
    assert "latest_1m_close_ts_ns" not in observation.market
    assert all(at <= observation.decision_ts_ns
               for at in observation.feature_available_at_ns.values())


@pytest.mark.parametrize("closes", (
    [100.0] * 80,
    [100.0 + index * 0.25 for index in range(80)],
    [100.0 - index * 0.20 for index in range(80)],
    [100.0] * 30 + [125.0] + [110.0] * 49,
))
def test_ema_matches_independent_numpy_reference(closes: list[float]):
    _bars, snapshots = _replay(closes)
    values = np.asarray(closes, dtype=np.float64)
    ema9 = _reference_ema(values, 9)
    ema21 = _reference_ema(values, 21)
    for index, snapshot in enumerate(snapshots):
        if index < 8:
            assert "ema_9" not in snapshot.values
        else:
            assert snapshot.value("ema_9") == pytest.approx(ema9[index], abs=1e-10, rel=1e-10)
        if index < 20:
            assert "ema_21" not in snapshot.values
        else:
            assert snapshot.value("ema_21") == pytest.approx(
                ema21[index], abs=1e-10, rel=1e-10
            )


def test_macd_uses_strict_maturity_and_hist_is_not_doubled():
    closes = np.asarray([100.0 + math.sin(index / 4) + index * 0.1 for index in range(80)])
    _bars, snapshots = _replay(closes.tolist())
    ema12 = _reference_ema(closes, 12)
    ema26 = _reference_ema(closes, 26)
    dif = ema12 - ema26
    mature_dif = dif[25:]
    dea_mature = _reference_ema(mature_dif, 9)
    for index, snapshot in enumerate(snapshots):
        if index < 33:
            assert snapshot.recipe_status["route_a.macd_12_26_9.v1"] == "NOT_READY"
            continue
        expected_dif = float(dif[index])
        expected_dea = float(dea_mature[index - 25])
        assert snapshot.value("macd_dif") == pytest.approx(expected_dif, abs=1e-10)
        assert snapshot.value("macd_dea") == pytest.approx(expected_dea, abs=1e-10)
        assert snapshot.value("macd_hist") == pytest.approx(
            expected_dif - expected_dea, abs=1e-10
        )


@pytest.mark.parametrize(("closes", "expected"), (
    ([100.0 + index for index in range(15)], 100.0),
    ([100.0 - index for index in range(15)], 0.0),
    ([100.0] * 15, 50.0),
))
def test_rsi_wilder_boundaries(closes: list[float], expected: float):
    _bars, snapshots = _replay(closes)
    assert "rsi_14" not in snapshots[-2].values
    assert snapshots[-1].value("rsi_14") == expected


def test_roc_units_and_donchian_excludes_current_bar():
    closes = [100.0 + index for index in range(61)]
    bars, snapshots = _replay(closes)
    last = snapshots[-1]
    assert last.value("roc_20") == pytest.approx(closes[-1] / closes[-21] - 1.0)
    assert last.value("roc_60") == pytest.approx(closes[-1] / closes[0] - 1.0)
    assert snapshots[59].recipe_status["route_a.roc_20_60.v1"] == "NOT_READY"
    assert last.recipe_status["route_a.roc_20_60.v1"] == "READY"

    prior = tuple(_bar(index, 100.0, high=101.0, low=99.0) for index in range(20))
    current = _bar(20, 105.0, high=110.0, low=104.0)
    result = replay_indicator_prefix((*prior, current))[-1]
    assert result.value("donchian_upper_20") == 101.0
    assert result.value("donchian_lower_20") == 99.0
    assert bars[-1].high > last.value("donchian_upper_20")


def test_zscore_and_bollinger_use_population_standard_deviation():
    closes = [80.0 + index * 0.5 for index in range(60)]
    _bars, snapshots = _replay(closes)
    values = np.asarray(closes, dtype=np.float64)
    final = snapshots[-1]
    assert final.value("zscore_60") == pytest.approx(
        (values[-1] - np.mean(values)) / np.std(values, ddof=0), abs=1e-10
    )
    last20 = values[-20:]
    assert final.value("bollinger_mid_20") == pytest.approx(np.mean(last20), abs=1e-10)
    assert final.value("bollinger_upper_20_2") == pytest.approx(
        np.mean(last20) + 2 * np.std(last20, ddof=0), abs=1e-10
    )
    _bars, constant = _replay([100.0] * 60)
    assert "zscore_60" not in constant[-1].values
    assert constant[-1].recipe_status["route_a.zscore_60.v1"] == "NOT_READY"


def test_atr_wilder_uses_gap_aware_true_range():
    bars = tuple(
        _bar(index, 100.0, high=101.0, low=99.0) for index in range(13)
    ) + (_bar(13, 110.0, high=111.0, low=109.0),)
    snapshots = replay_indicator_prefix(bars)
    assert "atr_14" not in snapshots[-2].values
    assert snapshots[-1].value("atr_14") == pytest.approx((13 * 2.0 + 11.0) / 14.0)


@pytest.mark.parametrize("timeframe_minutes", (5, 15, 60, 240, 1440))
def test_completed_higher_timeframe_bucket_obeys_utc_and_availability(
    timeframe_minutes: int,
):
    bars = tuple(_bar(index, 100.0 + index) for index in range(timeframe_minutes + 1))
    decision = timeframe_minutes * MINUTE_NS + DELAY_NS
    aggregated = aggregate_completed_bars(
        bars, timeframe_minutes=timeframe_minutes, decision_ts_ns=decision
    )
    assert len(aggregated) == 1
    assert aggregated[0].open_time_ns == 0
    assert aggregated[0].close_time_ns == timeframe_minutes * MINUTE_NS
    assert aggregated[0].close == 100.0 + timeframe_minutes - 1
    assert aggregated[0].high == 100.0 + timeframe_minutes
    assert aggregated[0].volume == pytest.approx(
        sum(10.0 + index for index in range(timeframe_minutes))
    )

    delayed = tuple(
        _bar(
            index,
            100.0 + index,
            available_delay_ns=(
                2 * DELAY_NS if index == timeframe_minutes - 1 else DELAY_NS
            ),
        )
        for index in range(timeframe_minutes)
    )
    assert aggregate_completed_bars(
        delayed, timeframe_minutes=timeframe_minutes, decision_ts_ns=decision
    ) == ()


def test_invalid_clock_gap_duplicate_order_and_nonfinite_fail_closed():
    engine = StreamingIndicators1m()
    first = _bar(0, 100.0)
    snapshot = engine.update(first, decision_ts_ns=first.available_at_ns)
    assert engine.update(first, decision_ts_ns=first.available_at_ns) is snapshot
    with pytest.raises(ValueError, match="decision clock"):
        engine.update(first, decision_ts_ns=first.available_at_ns + 1)
    with pytest.raises(ValueError, match="gap"):
        engine.update(_bar(2, 102.0), decision_ts_ns=_bar(2, 102.0).available_at_ns)
    with pytest.raises(ValueError, match="not yet available"):
        engine.update(_bar(1, 101.0), decision_ts_ns=_bar(1, 101.0).close_time_ns)
    with pytest.raises(ValueError, match="finite"):
        _bar(1, math.nan)


def test_restart_replay_and_future_perturbation_preserve_every_prefix():
    prefix = tuple(_bar(index, 100.0 + math.sin(index / 7)) for index in range(70))
    uninterrupted = replay_indicator_prefix(prefix)
    rebuilt = replay_indicator_prefix(prefix)
    assert uninterrupted == rebuilt

    future_a = prefix + tuple(_bar(index, 120.0) for index in range(70, 80))
    future_b = prefix + tuple(_bar(index, 75.0 + index) for index in range(70, 80))
    assert replay_indicator_prefix(future_a)[:70] == replay_indicator_prefix(future_b)[:70]


def test_snapshot_projection_preserves_feature_availability_and_close_clock():
    snapshot = replay_indicator_prefix(tuple(_bar(index, 100.0 + index) for index in range(61)))[-1]
    observation = project_indicator_snapshot(
        fold_id="fold",
        episode_id="episode",
        snapshot=snapshot,
        native_account_fields={"net_exposure": 0.0, "equity": 10_000.0},
    )
    assert observation.decision_ts_ns == snapshot.decision_ts_ns
    assert observation.exact_market_clock("latest_1m_close_ts_ns") == snapshot.close_time_ns
    assert "latest_1m_close_ts_ns" not in observation.market
    assert set(observation.feature_available_at_ns.values()) == {snapshot.available_at_ns}
    assert max(observation.feature_available_at_ns.values()) <= observation.decision_ts_ns


def test_l2_input_profile_reuses_authorized_store_and_rejects_stale_latest_bar():
    assert CAUSAL_1M_INDICATORS_V1_INPUT_PROFILE_ID == "causal_1m_indicators_v1"
    events = []
    for minute in range(1, 5):
        close_time = minute * MINUTE_NS
        events.append(Event(
            f"bar-{minute}",
            "ohlcv_1m",
            close_time,
            close_time,
            close_time + (2 * DELAY_NS if minute == 4 else DELAY_NS),
            Availability.NATIVE,
            "a" * 64,
            {
                "instrument": "ETHUSDT-PERP.SIM",
                "open": str(100 + minute),
                "high": str(101 + minute),
                "low": "99",
                "close": str(100 + minute),
                "volume": "5",
            },
            "registered-train-source",
        ))
    store = CuratedStore(reversed(events))
    decision = 3 * MINUTE_NS + DELAY_NS
    rows = causal_1m_indicator_inputs(
        store,
        decision,
        instrument="ETHUSDT-PERP.SIM",
        start_event_ns=MINUTE_NS,
    )
    assert tuple(row.close_time_ns for row in rows) == (
        MINUTE_NS, 2 * MINUTE_NS, 3 * MINUTE_NS,
    )
    assert all(row.available_at_ns <= decision for row in rows)
    assert store._feature_cache[("ohlcv_closed", "ETHUSDT-PERP.SIM", 1)] is not None
    with pytest.raises(DataError, match="missing or stale"):
        causal_1m_indicator_inputs(
            store,
            4 * MINUTE_NS + DELAY_NS,
            instrument="ETHUSDT-PERP.SIM",
            start_event_ns=MINUTE_NS,
        )

