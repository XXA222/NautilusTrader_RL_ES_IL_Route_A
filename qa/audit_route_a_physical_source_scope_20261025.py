"""Read import manifests and formal ledger only; never decode market events."""
from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / ".artifacts/research-access.sqlite"
OUT = ROOT / ".artifacts/route-a-physical-source-scope-audit-20261025/audit.json"
SOURCES = (
    "routea_teacher_train_20260812_20261002",
    "routea_teacher_train_20260717_20260804",
)


def _utc_ns(value: date) -> int:
    return int(datetime.combine(value, datetime.min.time(),
                                tzinfo=timezone.utc).timestamp()) * 1_000_000_000


def main() -> int:
    if OUT.exists():
        raise FileExistsError("physical-scope audit receipt is immutable")
    results = []
    with sqlite3.connect(REGISTRY.resolve().as_uri() + "?mode=ro", uri=True) as db:
        for name in SOURCES:
            source = ROOT / "data" / name / "import.json"
            encoded = source.read_bytes()
            manifest = json.loads(encoded)
            overlaps = []
            for entry in manifest["sources"]:
                path = urlparse(entry["url"]).path
                if "/monthly/" not in path:
                    continue
                match = re.search(r"-(\d{4}-\d{2})\.zip$", path)
                if match is None:
                    raise ValueError("unrecognized monthly source span")
                month = match.group(1)
                first = date.fromisoformat(month + "-01")
                last = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
                for table in ("access", "query_intent"):
                    rows = db.execute(
                        f"SELECT purpose,start_ns,end_ns FROM {table} "
                        "WHERE market=? AND family=? AND purpose IN ('test','holdout') "
                        "AND start_ns<? AND end_ns>?",
                        ("ETHUSDT", "perpetual-market-v1", _utc_ns(last), _utc_ns(first)),
                    ).fetchall()
                    overlaps.extend({
                        "archive_url": entry["url"], "archive_month": month,
                        "registry_table": table, "formal_role": purpose,
                        "formal_interval_ns": [start_ns, end_ns],
                    } for purpose, start_ns, end_ns in rows)
            results.append({
                "source": name,
                "import_manifest_sha256": sha256(encoded).hexdigest(),
                "monthly_formal_overlaps": overlaps,
                "qualification_source_eligible": not bool(overlaps),
            })
    if any(result["qualification_source_eligible"] for result in results):
        raise ValueError("expected imported-source physical overlap was not reproduced")
    result = {
        "schema": "rles.route-a.physical-source-scope-audit.v1",
        "qualification": "DQ_PHYSICAL_TEST_HOLDOUT_ARCHIVE_OVERLAP",
        "market_data_read": False,
        "formal_student_promotion": False,
        "sources": results,
        "interpretation": (
            "Curated scoring windows excluded formal dates, but monthly raw funding ZIPs "
            "were decoded before clipping. Retained P0 outcomes are engineering "
            "diagnostics only and cannot certify a teacher."
        ),
    }
    from rles_data.io import write_fact_json

    OUT.parent.mkdir(parents=True, exist_ok=True)
    digest = write_fact_json(OUT, result)
    with OUT.with_name(OUT.name + ".sha256").open("x", encoding="ascii") as marker:
        marker.write(digest + "\n")
    print(json.dumps({
        "qualification": result["qualification"],
        "overlaps_by_source": {
            row["source"]: len(row["monthly_formal_overlaps"]) for row in results
        },
        "report": str(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
