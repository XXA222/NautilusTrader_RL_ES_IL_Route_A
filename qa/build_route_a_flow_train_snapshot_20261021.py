"""Rebuild the frozen Train-only snapshot while preserving USD-M kline flow fields."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil

from rles_data.binance_archive import import_range
from rles_data.core import CuratedStore
from rles_data.io import read_curated, write_curated, write_fact_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "routea_train_20231230_20240614"
TARGET = ROOT / "data" / "routea_train_20231230_20240614_flow"
STAGING = ROOT / "data" / ".routea_train_20231230_20240614_flow_staging"
START_DAY = "2023-12-30"
END_DAY = "2024-06-14"
SCORE_END_NS = 1_718_409_600_000_000_000


def main() -> None:
    if TARGET.exists() or STAGING.exists():
        raise FileExistsError("immutable flow Train snapshot target or staging path exists")
    source_raw = SOURCE / "raw"
    raw_target = STAGING / "raw"
    raw_target.mkdir(parents=True)
    cache_files = sorted(
        path for path in source_raw.iterdir()
        if path.is_file() and not path.is_symlink()
        and (path.name.endswith(".zip") or path.name.endswith(".zip.source.json"))
    )
    if not cache_files:
        raise FileNotFoundError("verified source archive cache is empty")
    for path in cache_files:
        shutil.copyfile(path, raw_target / path.name)

    imported = import_range(
        symbol="ETHUSDT", start_day=START_DAY, end_day=END_DAY,
        output=STAGING, delay_ns=1_000_000, archive_interval="daily",
    )
    full = read_curated(STAGING / "market.parquet")
    clipped_events = tuple(event for event in full.events if event.ts_event_ns < SCORE_END_NS)
    clipped = CuratedStore(clipped_events)
    TARGET.mkdir(parents=False)
    write_curated(TARGET / "market.parquet", clipped)
    shutil.move(str(STAGING / "raw"), str(TARGET / "raw"))
    sources = [
        {**source, "path": str(TARGET / "raw" / Path(source["path"]).name)}
        for source in imported["sources"]
    ]
    source_record = {
        "schema": "rles.route-a-train-snapshot.v1",
        "symbol": "ETHUSDT",
        "start_day": START_DAY,
        "source_end_day_inclusive": END_DAY,
        "start_ns": imported["start_ns"],
        "end_ns": SCORE_END_NS,
        "score_end_exclusive_utc": "2024-06-15T00:00:00Z",
        "event_count": len(clipped.events),
        "unclipped_event_count": len(full.events),
        "clipped_boundary_events": len(full.events) - len(clipped.events),
        "availability": "derived",
        "availability_delay_ns": 1_000_000,
        "instrument_rules_supplied": False,
        "archive_interval": "daily",
        "sources": sources,
        "cache_entries_copied": sorted(Path(source["path"]).name for source in imported["sources"]),
        "source_import_schema": imported["schema"],
        "source_import_start_ns": imported["start_ns"],
        "source_import_end_ns": imported["end_ns"],
        "split_policy": "Train-only; all event timestamps at or after frozen Train score end were removed",
        "kline_flow_fields": [
            "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
        ],
        "source_import_sha256": sha256((TARGET / "market.parquet").read_bytes()).hexdigest(),
    }
    write_fact_json(TARGET / "import.json", source_record)
    shutil.rmtree(STAGING)
    print(json.dumps({
        "target": str(TARGET), "events": len(clipped.events),
        "start_ns": imported["start_ns"], "end_ns": SCORE_END_NS,
        "boundary_removed": len(full.events) - len(clipped.events),
        "sources": len(imported["sources"]), "cache_files": len(cache_files),
        "snapshot_hash": clipped.snapshot_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
