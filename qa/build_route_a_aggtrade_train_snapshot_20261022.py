"""Build an immutable Train-only snapshot with verified USD-M aggTrades flow."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import date
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any

from rles_data.binance_archive import archive_events, fetch_archive
from rles_data.core import Availability, CuratedStore, Event
from rles_data.io import read_curated, write_curated, write_fact_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "routea_train_20231230_20240614_flow"
TARGET = ROOT / "data" / "routea_train_20231230_20240614_flow_agg_20261022"
START_DAY = date(2024, 4, 15)
END_DAY = date(2024, 6, 14)
SCORE_END_NS = 1_718_409_600_000_000_000
DELAY_NS = 1_000_000
INSTRUMENT = "ETHUSDT-PERP.SIM"
SAMPLED_BOOKTICKER_404_DAYS = ("2024-04-15", "2024-06-14", "2024-07-01", "2025-01-01")


def _import_aggtrade_day(
    day: str, raw_path: str,
) -> tuple[str, dict[str, Any], tuple[dict[str, Any], ...]]:
    raw = Path(raw_path)
    relative = f"daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-{day}.zip"
    source = fetch_archive(relative, raw / f"aggTrades-{day}.zip")
    day_events = archive_events(
        Path(source["path"]), stream="aggTrades", instrument=INSTRUMENT,
        source_url=source["url"], expected_sha256=source["sha256"],
        delay_ns=DELAY_NS,
    )
    visible = tuple(
        {
            "event_id": event.event_id,
            "kind": event.kind,
            "ts_event_ns": event.ts_event_ns,
            "ts_init_ns": event.ts_init_ns,
            "available_at_ns": event.available_at_ns,
            "availability_source": event.availability_source.value,
            "source_hash": event.source_hash,
            "payload": dict(event.payload),
            "provenance": event.provenance,
        }
        for event in day_events
        if event.available_at_ns is not None
        and event.available_at_ns < SCORE_END_NS
    )
    return day, source, visible


def main() -> None:
    if not (SOURCE / "market.parquet").is_file() or not (SOURCE / "import.json").is_file():
        raise FileNotFoundError("frozen flow Train source is incomplete")
    if (TARGET / "market.parquet").exists() or (TARGET / "import.json").exists():
        raise FileExistsError("immutable aggTrades Train snapshot is already published")
    source_record = json.loads((SOURCE / "import.json").read_text(encoding="utf-8"))
    if (source_record.get("schema") != "rles.route-a-train-snapshot.v1"
            or source_record.get("end_ns") != SCORE_END_NS):
        raise PermissionError("base Train snapshot differs from the frozen score boundary")

    raw_target = TARGET / "raw"
    raw_target.mkdir(parents=True, exist_ok=True)
    cache_files = sorted(
        path for path in (SOURCE / "raw").iterdir()
        if path.is_file() and not path.is_symlink()
        and (path.name.endswith(".zip") or path.name.endswith(".zip.source.json"))
    )
    if not cache_files:
        raise FileNotFoundError("verified base Train archive cache is empty")
    for path in cache_files:
        destination = raw_target / path.name
        if destination.exists():
            if destination.stat().st_size != path.stat().st_size:
                raise FileExistsError(f"conflicting immutable archive cache: {destination}")
            continue
        shutil.copyfile(path, destination)

    base = read_curated(SOURCE / "market.parquet")
    events = list(base.events)
    sources = [
        {**source, "path": str(raw_target / Path(source["path"]).name)}
        for source in source_record["sources"]
    ]
    days = []
    current = START_DAY
    total_days = (END_DAY - START_DAY).days + 1
    while current <= END_DAY:
        days.append(current.isoformat())
        current = date.fromordinal(current.toordinal() + 1)
    worker_count = int(os.environ.get("ROUTE_A_AGGTRADE_IMPORT_WORKERS", "4"))
    if type(worker_count) is not int or not 1 <= worker_count <= min(8, os.cpu_count() or 1):
        raise ValueError("aggTrades import worker count must be between 1 and 8 CPUs")
    flow_count = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_import_aggtrade_day, day, str(raw_target))
            for day in days
        ]
        for day_index, future in enumerate(futures, start=1):
            day, source, day_events = future.result()
            sources.append(source)
            events.extend(Event(
                row["event_id"], row["kind"], row["ts_event_ns"],
                row["ts_init_ns"], row["available_at_ns"],
                Availability(row["availability_source"]), row["source_hash"],
                row["payload"], row["provenance"],
            ) for row in day_events)
            flow_count += len(day_events)
            print(json.dumps({
                "day": day, "days_done": day_index, "days_total": total_days,
                "flow_events": len(day_events), "flow_events_total": flow_count,
                "workers": worker_count,
            }, sort_keys=True), flush=True)

    store = CuratedStore(events)
    snapshot_hash = write_curated(TARGET / "market.parquet", store)
    base_sources = [Path(source["path"]).name for source in source_record["sources"]]
    agg_sources = [Path(source["path"]).name for source in sources
                   if "/aggTrades/" in source["url"]]
    metadata = {
        **source_record,
        "sources": sources,
        "event_count": len(store.events),
        "base_snapshot_hash": base.snapshot_hash,
        "agg_trades_included": True,
        "agg_trades_start_day": START_DAY.isoformat(),
        "agg_trades_end_day_inclusive": END_DAY.isoformat(),
        "agg_trades_sampling": "one_second_aggressor_flow",
        "agg_trades_event_count": flow_count,
        "bookticker_included": False,
        "bookticker_unavailable_evidence": {
            "stream": "USD-M ETHUSDT daily bookTicker",
            "sampled_dates_returning_404": list(SAMPLED_BOOKTICKER_404_DAYS),
            "policy": "No fabricated order-book proxy; absent archive means unavailable",
        },
        "availability_delay_ns": DELAY_NS,
        "cache_entries_copied": sorted(base_sources + agg_sources),
        "source_import_sha256": sha256((TARGET / "market.parquet").read_bytes()).hexdigest(),
        "snapshot_hash": snapshot_hash,
        "split_policy": (
            "Train-only; aggTrades dates are within frozen Train; all event availability "
            "at or after score end were removed"
        ),
    }
    write_fact_json(TARGET / "import.json", metadata)
    print(json.dumps({
        "target": str(TARGET), "snapshot_hash": snapshot_hash,
        "events": len(store.events), "agg_trades_flow_events": flow_count,
        "sources": len(sources), "raw_archives": len(list(raw_target.glob("*.zip"))),
        "bookticker_included": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
