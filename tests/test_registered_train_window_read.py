from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sqlite3

import pytest

from rles_data.core import Availability, CuratedStore, Event, MINUTE_NS
from rles_data.io import read_curated_window, write_curated
from rles_orchestrator import research_access
from rles_research.access import ProjectResearchRegistry


def _store() -> CuratedStore:
    events = []
    for minute in range(1, 5):
        price = str(100 + minute)
        timestamp = minute * MINUTE_NS
        events.append(Event(
            f"bar-{minute}", "ohlcv_1m", timestamp, timestamp, timestamp,
            Availability.NATIVE, "a" * 64,
            {"instrument": "BTC/USD", "open": price, "high": price,
             "low": price, "close": price, "volume": "1"},
            "synthetic:registered-window-test",
        ))
    return CuratedStore(events)


def test_registered_train_reader_reserves_exact_scope_before_row_group_decode(
    monkeypatch, tmp_path: Path,
) -> None:
    market_path = tmp_path / "market.parquet"
    write_curated(market_path, _store(), row_group_size=2)
    marker = market_path.with_name(market_path.name + ".COMPLETED.json")
    marker_sha = sha256(marker.read_bytes()).hexdigest()
    events = []

    def reserve(config, window, *, label):
        events.append(("reserve", config["environment"]["curated_marker_sha256"],
                       config["environment"]["curated_read_mode"], dict(window), label))

    def read(path, start_ns, end_ns, *, expected_marker_sha256,
             allow_legacy_isolated_snapshot=False):
        events.append(("decode", expected_marker_sha256, allow_legacy_isolated_snapshot,
                       start_ns, end_ns))
        return read_curated_window(path, start_ns, end_ns,
                                   expected_marker_sha256=expected_marker_sha256,
                                   allow_legacy_isolated_snapshot=allow_legacy_isolated_snapshot)

    monkeypatch.setattr(research_access, "record_development", reserve)
    monkeypatch.setattr(research_access, "read_curated_window", read)
    result = research_access.read_registered_train_window(
        {"output": str(tmp_path / "run"),
         "environment": {"market_path": str(market_path), "instrument": "BTC/USD",
                         "start_ns": 3 * MINUTE_NS, "end_ns": 4 * MINUTE_NS}},
        {"start_ns": 3 * MINUTE_NS, "end_ns": 5 * MINUTE_NS},
        marker_sha256=marker_sha,
        label="train-stage69-scoped-feature-profile-v1",
    )

    assert [event[0] for event in events] == ["reserve", "decode"]
    assert events[0][1] == events[1][1] == marker_sha
    assert events[0][2] == "row_group_scoped_v1" and events[1][2] is False
    assert [event.event_id for event in result.events] == ["bar-3", "bar-4"]


def test_registered_train_reader_rejects_evaluation_overlap_before_reservation(
    monkeypatch, tmp_path: Path,
) -> None:
    market_path = tmp_path / "unused.parquet"
    reserved = False

    def reserve(*_args, **_kwargs):
        nonlocal reserved
        reserved = True

    monkeypatch.setattr(research_access, "record_development", reserve)
    try:
        research_access.read_registered_train_window(
            {"output": str(tmp_path / "run"),
             "environment": {"market_path": str(market_path), "instrument": "BTC/USD",
                             "start_ns": MINUTE_NS, "end_ns": 4 * MINUTE_NS},
             "evaluation": {"start_ns": 4 * MINUTE_NS, "end_ns": 6 * MINUTE_NS}},
            {"start_ns": MINUTE_NS, "end_ns": 5 * MINUTE_NS},
            marker_sha256="a" * 64,
            label="train-scoped-read",
        )
    except ValueError as exc:
        assert "overlaps the evaluation interval" in str(exc)
    else:
        raise AssertionError("evaluation overlap was accepted")
    assert reserved is False


def test_registered_train_read_receipt_verifies_registry_and_source(monkeypatch, tmp_path: Path):
    registry = ProjectResearchRegistry(tmp_path / "research-access.sqlite")
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    monkeypatch.setattr(research_access, "read_curated_window",
                        lambda *_args, **_kwargs: _store())
    marker = "b" * 64
    config = {
        "output": str(tmp_path / "run"),
        "environment": {"market_path": str(tmp_path / "market.parquet"),
                        "instrument": "BTC/USD", "start_ns": MINUTE_NS,
                        "end_ns": 4 * MINUTE_NS},
    }
    _store_result, receipt = research_access.read_registered_train_window(
        config, {"start_ns": MINUTE_NS, "end_ns": 5 * MINUTE_NS},
        marker_sha256=marker, label="train-route-a-receipt-test-v1",
        return_receipt=True,
    )

    assert receipt.split == "train"
    assert receipt.registered_before_read is True
    assert receipt.source_identity == {"marker_sha256": marker}
    assert research_access.verify_registered_train_read(receipt) is True
    receipt.source_identity["marker_sha256"] = "c" * 64
    assert research_access.verify_registered_train_read(receipt) is False


def test_registered_train_rejects_imported_monthly_test_overlap_before_market_read(
        monkeypatch, tmp_path: Path):
    registry = ProjectResearchRegistry(tmp_path / "research-access.sqlite")
    test_start = 1_786_108_800_000_000_000  # 2026-08-08 UTC
    test_end = 1_786_195_200_000_000_000
    with sqlite3.connect(registry.path) as db:
        db.execute(
            "INSERT INTO access VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            ("formal-test", "legacy", "ETHUSDT", "perpetual-market-v1",
             test_start, test_end, "test", "{}", "token", "reserved", None),
        )
    market = tmp_path / "imported" / "market.parquet"
    market.parent.mkdir()
    (market.parent / "import.json").write_text(
        '{"sources":[{"url":"https://data.binance.vision/data/futures/um/'
        'monthly/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2026-08.zip"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    monkeypatch.setattr(
        research_access, "record_development",
        lambda *_args, **_kwargs: pytest.fail("reserved contaminated source"),
    )
    monkeypatch.setattr(
        research_access, "read_curated_window",
        lambda *_args, **_kwargs: pytest.fail("decoded contaminated source"),
    )
    with pytest.raises(research_access.DataError, match="physically overlaps"):
        research_access.read_registered_train_window(
            {"output": str(tmp_path / "run"),
             "environment": {"market_path": str(market),
                             "instrument": "ETHUSDT-PERP.SIM",
                             "start_ns": test_end + 1,
                             "end_ns": test_end + 3}},
            {"start_ns": test_end, "end_ns": test_end + 4},
            marker_sha256="a" * 64, label="train-monthly-physical-guard",
        )
