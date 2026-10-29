"""Restore the independent future Train reservation before any source can exist."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

from rles_orchestrator.research_access import project_registry

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/route_a_meta_x1_train_prereg_20261029.json"


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
    start, end = (ns(config["source"][key]) for key in ("start_utc_inclusive", "end_utc_exclusive"))
    now = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    if now >= start:
        raise PermissionError("cannot create a pre-read reservation after source start")
    owner = config["protocol_id"]
    identity = {"protocol_sha256": digest, "protocol_id": owner,
                "purpose": "independent_meta_train_pre_read"}
    registry = project_registry()
    params = dict(key="access:" + owner, owner=owner, market=config["market"],
                  family=config["data_family"], start_ns=start, end_ns=end, identity=identity)
    registry.reserve(**params, purpose="development")
    if not registry.verify_development_reservation(**params):
        raise PermissionError("independent Train reservation verification failed")
    print(json.dumps({"status": "REGISTERED_FUTURE_SOURCE", "protocol_id": owner,
                      "protocol_sha256": digest, "market_payload_read": False}, sort_keys=True))


if __name__ == "__main__":
    main()
