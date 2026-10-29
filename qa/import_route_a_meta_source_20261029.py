"""Import the preregistered Meta study source after its full window closes."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import validate_route_a_meta_prereg_20261029 as preflight
from rles_data.binance_archive import import_range

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/route_a_meta_x1_train_prereg_20261029.json"
OUTPUT = ROOT / "data/routea_meta_train_20261201_20261218"


def main():
    preflight.main()
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    start = datetime.fromisoformat(config["source"]["start_utc_inclusive"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(config["source"]["end_utc_exclusive"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) < end:
        raise RuntimeError("registered source window is incomplete; no market data was read")
    if start.time().isoformat() != "00:00:00" or end.time().isoformat() != "00:00:00":
        raise ValueError("registered daily source boundaries must be midnight UTC")
    last = end.date() - timedelta(days=1)
    months = sorted({f"{day.year:04d}-{day.month:02d}" for day in
                     (start.date() + timedelta(days=i) for i in range((end.date()-start.date()).days))})
    result = import_range(symbol=config["market"], start_day=start.date().isoformat(),
        end_day=last.isoformat(), output=OUTPUT, delay_ns=1_000_000,
        archive_interval="daily", funding_api_months=tuple(months),
        development_access_key="access:" + config["protocol_id"],
        development_protocol_sha256=sha256(raw).hexdigest())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
