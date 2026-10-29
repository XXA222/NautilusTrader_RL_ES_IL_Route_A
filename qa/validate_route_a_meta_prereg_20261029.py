"""Read-only preflight for the independently registered Meta X1 Train study."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/route_a_meta_x1_train_prereg_20261029.json"
REGISTRY = ROOT / ".artifacts/research-access.sqlite"


def ns(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()) * 1_000_000_000


def main() -> None:
    if (ROOT.resolve() != Path("/mnt/d/Program Files/NautilusTrader_RL_ES_IL_Route_A")
            or sys.version_info[:3] != (3, 14, 7)
            or (ROOT / ".canonical-project-root").read_text(encoding="utf-8").strip()
            != r"D:\Program Files\NautilusTrader_RL_ES_IL_Route_A"):
        raise RuntimeError("canonical WSL root or Python 3.14.7 mismatch")
    raw = CONFIG.read_bytes()
    digest = sha256(raw).hexdigest()
    if CONFIG.with_suffix(CONFIG.suffix + ".sha256").read_text(encoding="ascii").strip() != digest:
        raise ValueError("independent Meta preregistration hash mismatch")
    config = json.loads(raw)
    owner = config["protocol_id"]
    start, end = (ns(config["source"][key]) for key in ("start_utc_inclusive", "end_utc_exclusive"))
    intervals = config["intervals"]
    warmup, fit, cal = (tuple(map(ns, intervals[key]))
                        for key in ("warmup", "train_fit", "train_calibration"))
    screens = tuple(tuple(map(ns, item)) for item in intervals["native_screen"])
    ordered = (warmup, fit, cal, *screens)
    if (not ordered or ordered[0][0] != start or ordered[-1][1] != end
            or any(left >= right for left, right in ordered)
            or any(a[1] != b[0] for a, b in zip(ordered, ordered[1:]))):
        raise ValueError("study intervals must partition the registered source")
    with sqlite3.connect(REGISTRY.resolve().as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT owner,market,family,start_ns,end_ns,purpose,identity,state "
                         "FROM access WHERE access_key=?", ("access:"+owner,)).fetchone()
        conflicts = []
        for table in ("access", "query_intent"):
            conflicts.extend(db.execute(
                f"SELECT owner,purpose FROM {table} WHERE market=? AND family=? "
                "AND start_ns<? AND end_ns>? AND owner<>?",
                (config["market"], config["data_family"], end, start, owner)).fetchall())
    expected = (owner, config["market"], config["data_family"], start, end, "development")
    if row is None or row[:6] != expected or row[7] != "recorded" or conflicts:
        raise PermissionError("independent Train reservation is missing or overlaps another scope")
    if json.loads(row[6]) != {"protocol_sha256": digest, "protocol_id": owner,
                              "purpose": "independent_meta_train_pre_read"}:
        raise PermissionError("independent Train reservation identity changed")
    now = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    print(json.dumps({
        "status": "REGISTERED_FUTURE_SOURCE" if now < start else "REGISTERED_SOURCE_AVAILABILITY_UNVERIFIED",
        "protocol_id": owner, "protocol_sha256": digest,
        "source_start": config["source"]["start_utc_inclusive"],
        "source_end": config["source"]["end_utc_exclusive"],
        "completed_screen_windows": sum(now >= window[1] for window in screens),
        "market_payload_read": False,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
