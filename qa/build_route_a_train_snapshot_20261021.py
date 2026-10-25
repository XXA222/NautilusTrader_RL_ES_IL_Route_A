"""Build a checksum verified Route A source clipped to the frozen Train fold."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import shutil

from rles_data.binance_archive import import_range
from rles_data.core import CuratedStore
from rles_data.io import read_curated, write_curated, write_fact_json


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data" / "routea_train_20231230_20240531"
UPSTREAM = ROOT / "data" / "binance_ethusdt_20240415_20240731"
TARGET = ROOT / "data" / "routea_train_20231230_20240614"
STAGING = ROOT / "data" / ".routea_train_20231230_20240614_staging"
START_DAY = "2023-12-30"
END_DAY = "2024-06-14"
SCORE_END_NS = 1_718_409_600_000_000_000  # 2024-06-15T00:00:00Z


def copy_verified_cache() -> list[str]:
    raw_target = STAGING / "raw"
    raw_target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    first, last = date.fromisoformat(START_DAY), date.fromisoformat(END_DAY)
    current = first
    while current <= last:
        day = current.isoformat()
        source_dir = OLD / "raw" if current <= date(2024, 5, 31) else UPSTREAM / "raw"
        for stream in ("klines", "markPriceKlines", "indexPriceKlines"):
            name = f"{stream}-{day}.zip"
            for suffix in ("", ".source.json"):
                source = source_dir / (name + suffix)
                if not source.is_file() or source.is_symlink():
                    raise FileNotFoundError(f"missing verified archive cache entry: {source}")
                shutil.copyfile(source, raw_target / (name + suffix))
            copied.append(name)
        current += timedelta(days=1)
    for month in ("2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"):
        source_dir = OLD / "raw" if month <= "2024-05" else UPSTREAM / "raw"
        name = f"fundingRate-{month}.zip"
        for suffix in ("", ".source.json"):
            source = source_dir / (name + suffix)
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(f"missing verified archive cache entry: {source}")
            shutil.copyfile(source, raw_target / (name + suffix))
        copied.append(name)
    return copied


def main() -> None:
    if TARGET.exists() or STAGING.exists():
        raise FileExistsError("immutable Train snapshot output or staging path already exists")
    copied = copy_verified_cache()
    imported = import_range(
        symbol="ETHUSDT", start_day=START_DAY, end_day=END_DAY,
        output=STAGING, delay_ns=1_000_000, archive_interval="daily",
    )
    full = read_curated(STAGING / "market.parquet")
    clipped_events = tuple(event for event in full.events if event.ts_event_ns < SCORE_END_NS)
    if not clipped_events:
        raise RuntimeError("Train boundary clipping produced an empty source")
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
        "cache_entries_copied": copied,
        "source_import_schema": imported["schema"],
        "source_import_start_ns": imported["start_ns"],
        "source_import_end_ns": imported["end_ns"],
        "split_policy": "Train-only; all event timestamps at or after frozen Train score end were removed",
    }
    write_fact_json(TARGET / "import.json", source_record)
    shutil.rmtree(STAGING)
    print(json.dumps({
        "target": str(TARGET), "events": len(clipped.events),
        "start_ns": imported["start_ns"], "end_ns": SCORE_END_NS,
        "boundary_removed": len(full.events) - len(clipped.events),
        "sources": len(imported["sources"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
