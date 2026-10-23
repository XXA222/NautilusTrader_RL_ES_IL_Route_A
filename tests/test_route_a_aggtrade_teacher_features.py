from __future__ import annotations

import math
from importlib import import_module
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


QA_ROOT = Path(__file__).resolve().parents[1] / "qa"
sys.path.insert(0, str(QA_ROOT))

teacher = import_module("probe_route_a_barrier_value_teacher_20261021")
runner = import_module("run_route_a_train_student_screen_20261020")


def _event(second: int, *, buy_notional: float = 2.0) -> SimpleNamespace:
    timestamp = second * 1_000_000_000
    return SimpleNamespace(
        kind="trade_flow_1s",
        ts_event_ns=timestamp,
        available_at_ns=timestamp + 10,
        source_hash="a" * 64,
        payload={
            "instrument": runner.INSTRUMENT,
            "aggressor_buy_notional": buy_notional,
            "aggressor_sell_notional": 1.0,
            "trade_count": 1,
        },
    )


def test_aggtrade_features_are_causal_and_windowed() -> None:
    events = [_event(second) for second in range(2, 902)]
    events.append(_event(902, buy_notional=1e12))  # not available at decision time
    factory = SimpleNamespace(store=SimpleNamespace(events=events))
    rows = (SimpleNamespace(decision_ts_ns=902 * 1_000_000_000),)

    features = teacher.registered_aggtrade_flow_features(factory, rows)

    assert features.shape == (1, 3)
    assert np.allclose(features[0, :2], 1 / 3)
    assert math.isclose(features[0, 2], math.log1p(299), rel_tol=1e-12)


def test_aggtrade_features_fail_closed_on_missing_flow() -> None:
    factory = SimpleNamespace(store=SimpleNamespace(events=[]))
    rows = (SimpleNamespace(decision_ts_ns=902 * 1_000_000_000),)

    with pytest.raises(RuntimeError, match="lacks timestamped aggTrades flow"):
        teacher.registered_aggtrade_flow_features(factory, rows)


def test_complete_checksum_archives_allow_zero_trade_seconds(tmp_path: Path) -> None:
    events = [_event(second) for second in range(2, 902, 3)]
    factory = SimpleNamespace(store=SimpleNamespace(events=events))
    rows = (SimpleNamespace(decision_ts_ns=902 * 1_000_000_000),)
    archive_hash = "a" * 64
    manifest = tmp_path / "import.json"
    manifest.write_text(json.dumps({
        "schema": "rles.route-a-train-snapshot.v1",
        "agg_trades_included": True,
        "agg_trades_start_day": "1970-01-01",
        "agg_trades_end_day_inclusive": "1970-01-01",
        "sources": [{
            "url": "https://data.binance.vision/data/futures/um/daily/aggTrades/"
                   "ETHUSDT/ETHUSDT-aggTrades-1970-01-01.zip",
            "sha256": archive_hash,
        }],
    }))

    features = teacher.registered_aggtrade_flow_features(
        factory, rows, archive_manifest=manifest,
    )

    assert features.shape == (1, 3)
    assert np.isfinite(features).all()
    assert features[0, 2] < math.log1p(297)


def test_complete_archive_manifest_rejects_missing_day(tmp_path: Path) -> None:
    manifest = tmp_path / "import.json"
    manifest.write_text(json.dumps({
        "schema": "rles.route-a-train-snapshot.v1",
        "agg_trades_included": True,
        "agg_trades_start_day": "1970-01-01",
        "agg_trades_end_day_inclusive": "1970-01-02",
        "sources": [{
            "url": "https://data.binance.vision/data/futures/um/daily/aggTrades/"
                   "ETHUSDT/ETHUSDT-aggTrades-1970-01-01.zip",
            "sha256": "a" * 64,
        }],
    }))
    factory = SimpleNamespace(store=SimpleNamespace(events=[_event(second) for second in range(2, 902)]))
    rows = (SimpleNamespace(decision_ts_ns=902 * 1_000_000_000),)

    with pytest.raises(RuntimeError, match="missing or duplicate"):
        teacher.registered_aggtrade_flow_features(
            factory, rows, archive_manifest=manifest,
        )


def test_value_diagnostics_report_selected_side_calibration() -> None:
    predictions = np.asarray([[2.0, -1.0], [-1.0, 2.0], [1.0, 0.0]])
    outcomes = np.asarray([[10.0, -10.0], [-8.0, 7.0], [2.0, -2.0]])

    report = teacher.evaluate_value_predictions(predictions, outcomes)

    assert report["sample_count"] == 3
    assert report["selected_side_prediction_mae_bps"] is not None
    assert report["selected_side_prediction_bias_bps"] is not None
    assert report["selected_side_prediction_realized_pearson_r"] is not None
    assert report["selected_side_positive_sign_accuracy"] == pytest.approx(1.0)


def test_fixed_horizon_labels_match_entry_to_exit_close_and_reject_gaps() -> None:
    times = np.arange(300, dtype=np.int64) * runner.MINUTE_NS
    closes = np.exp(np.arange(300, dtype=np.float64) * 0.00001)
    entries = np.asarray([5], dtype=np.int64)

    outcomes, retained = teacher.fixed_horizon_outcomes(entries, times, closes)

    assert retained.tolist() == [5]
    assert outcomes.shape == (1, 2)
    assert outcomes[0, 0] == pytest.approx(24.0 - 30.0)
    assert outcomes[0, 1] == pytest.approx(-24.0 - 30.0)

    gapped_times = times.copy()
    gapped_times[100:] += runner.MINUTE_NS
    outcomes, retained = teacher.fixed_horizon_outcomes(entries, gapped_times, closes)
    assert outcomes.shape == (0, 2)
    assert retained.size == 0


def test_registered_feature_scope_expands_to_complete_verified_row_groups(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "market.parquet.COMPLETED.json"
    marker.write_text(json.dumps({
        "schema_version": "0.4.1",
        "row_groups": [
            {
                "events": 10,
                "min_ts_event_ns": 0,
                "max_ts_event_ns": 9,
                "min_available_at_ns": 1,
                "max_available_at_ns": 10,
            },
            {
                "events": 10,
                "min_ts_event_ns": 10,
                "max_ts_event_ns": 19,
                "min_available_at_ns": 11,
                "max_available_at_ns": 20,
            },
            {
                "events": 10,
                "min_ts_event_ns": 20,
                "max_ts_event_ns": 29,
                "min_available_at_ns": 21,
                "max_available_at_ns": 30,
            },
        ],
    }))

    scope = teacher.registered_feature_read_scope(
        marker,
        requested_start_ns=15,
        requested_end_ns=18,
        source_start_ns=0,
        source_end_ns=31,
    )

    assert scope == {
        "source_mode": "row_group_scoped_v1",
        "start_ns": 0,
        "end_ns": 31,
        "row_group_count": 3,
        "row_group_indexes": [0, 1, 2],
        "event_count": 30,
    }


def test_registered_feature_scope_fails_closed_on_partial_group_boundaries(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "market.parquet.COMPLETED.json"
    marker.write_text(json.dumps({
        "schema_version": "0.4.1",
        "row_groups": [
            {
                "events": 10,
                "min_ts_event_ns": 0,
                "max_ts_event_ns": 9,
                "min_available_at_ns": 1,
                "max_available_at_ns": 10,
            },
            {
                "events": 10,
                "min_ts_event_ns": 8,
                "max_ts_event_ns": 7,
                "min_available_at_ns": 9,
                "max_available_at_ns": 8,
            },
        ],
    }))

    with pytest.raises(RuntimeError, match="bounds are inverted"):
        teacher.registered_feature_read_scope(
            marker,
            requested_start_ns=4,
            requested_end_ns=6,
            source_start_ns=0,
            source_end_ns=21,
        )
