from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))
spec = importlib.util.spec_from_file_location("meta_import", ROOT / "qa/import_route_a_meta_source_20261029.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("now,allowed", [(datetime(2026, 12, 18, tzinfo=timezone.utc), False),
                                          (datetime(2026, 12, 19, tzinfo=timezone.utc), True)])
def test_import_is_bounded_and_waits_for_complete_window(monkeypatch, now, allowed):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now
    calls = []
    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(module.preflight, "main", lambda: None)
    monkeypatch.setattr(module, "import_range", lambda **kwargs: calls.append(kwargs) or {})
    if not allowed:
        with pytest.raises(RuntimeError, match="incomplete"):
            module.main()
        assert not calls
        return
    module.main()
    assert len(calls) == 1
    args = calls[0]
    assert (args["symbol"], args["start_day"], args["end_day"]) == ("ETHUSDT", "2026-12-01", "2026-12-18")
    assert args["archive_interval"] == "daily" and args["funding_api_months"] == ("2026-12",)
    assert args["delay_ns"] == 1_000_000
    assert args["development_access_key"] == "access:route-a-meta-x1-train-20261201-20261219-v1"
    assert args["development_protocol_sha256"] == sha256(module.CONFIG.read_bytes()).hexdigest()
    assert args["output"] == module.OUTPUT
