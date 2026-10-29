from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
from urllib.parse import urlencode

import pytest

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "qa/run_route_a_meta_study_20261029.py"
spec = importlib.util.spec_from_file_location("route_a_meta_study", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_account_costs_include_first_native_sample():
    account = {"metrics": {"total_fee": 2., "total_slippage_cost": 1., "total_funding": -.1},
               "account_rows": [{"fee": .5, "slippage_cost": .2, "funding": .3},
                                {"fee": 2., "slippage_cost": 1., "funding": -.1}]}
    costs = study._account_costs_including_first_row(account)
    assert costs["total_fee"] == pytest.approx(2.5)
    assert costs["total_slippage_cost"] == pytest.approx(1.2)
    assert costs["total_funding"] == pytest.approx(.2)


def source_metadata(config):
    start = datetime.fromisoformat(config["source"]["start_utc_inclusive"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(config["source"]["end_utc_exclusive"].replace("Z", "+00:00"))
    items = []
    day = start
    while day < end:
        for stream in ("klines", "markPriceKlines", "indexPriceKlines"):
            items.append({"url":f"https://data.binance.vision/data/futures/um/daily/{stream}/"
                         f"ETHUSDT/1m/ETHUSDT-1m-{day.date().isoformat()}.zip","sha256":"a"*64})
        day += timedelta(days=1)
    query = urlencode({"symbol":"ETHUSDT","startTime":int(start.timestamp()*1000),
                       "endTime":int(end.timestamp()*1000)-1,"limit":1000})
    items.append({"url":"https://fapi.binance.com/fapi/v1/fundingRate?"+query,"sha256":"b"*64})
    return {"archive_interval":"daily", "symbol":"ETHUSDT", "instrument":study.legacy.INSTRUMENT,
            "availability_delay_ns":1_000_000, "agg_trades_included":False,
            "bookticker_included":False, "sources":items}


def test_physical_scope_rejects_monthly_and_overbroad_funding_before_payload():
    config = {"market":"ETHUSDT", "source":{"start_utc_inclusive":"2026-12-01T00:00:00Z",
                                              "end_utc_exclusive":"2026-12-03T00:00:00Z"}}
    manifest = source_metadata(config)
    study._validate_physical_sources(config, manifest)
    bad = {**manifest, "sources": list(manifest["sources"])}
    bad["sources"][0] = {**bad["sources"][0], "url":"https://data.binance.vision/data/futures/um/monthly/klines/x.zip"}
    with pytest.raises(ValueError, match="physical scope"):
        study._validate_physical_sources(config, bad)
    bad = {**manifest, "sources": list(manifest["sources"])}
    bad["sources"][-1] = {**bad["sources"][-1], "url":bad["sources"][-1]["url"].replace("limit=1000","limit=1001")}
    with pytest.raises(ValueError, match="physical scope"):
        study._validate_physical_sources(config, bad)
    with pytest.raises(ValueError, match="incomplete"):
        study._validate_physical_sources(config, {**manifest, "sources":manifest["sources"][:-1]})


def test_missing_or_invalid_manifest_never_enters_payload_verifier(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "_inside_project", lambda _: tmp_path)
    monkeypatch.setattr(study.legacy, "_verified_source", lambda *args: pytest.fail("market verifier must not run"))
    config = {"market":"ETHUSDT", "source":{"start_utc_inclusive":"2026-12-01T00:00:00Z",
                                              "end_utc_exclusive":"2026-12-03T00:00:00Z"}}
    with pytest.raises(FileNotFoundError):
        study._load_source(config, "hash", "data")
    (tmp_path/"import.json").write_text("{}")
    with pytest.raises(ValueError, match="daily market scope"):
        study._load_source(config, "hash", "data")


def test_synthetic_complete_source_passes_real_loader(tmp_path, monkeypatch):
    """Only synthetic bars and synthetic provenance; no remote data are read."""
    import json
    from rles_data.core import MINUTE_NS
    from rles_data.io import read_curated, write_curated
    from test_historical_factory import recipe
    fixture = tmp_path/"fixture"
    fixture.mkdir()
    env = recipe(fixture, bar_count=1439, start_minute=150)
    source_dir = tmp_path/"data"/"synthetic"
    source_dir.mkdir(parents=True)
    market = source_dir/"market.parquet"
    write_curated(market, read_curated(env["market_path"]))
    marker = market.with_name("market.parquet.COMPLETED.json")
    marker_doc = json.loads(marker.read_text(encoding="utf-8"))
    config = {"protocol_id":"synthetic-test", "market":"ETHUSDT",
              "source":{"start_utc_inclusive":"1970-01-01T00:00:00Z",
                        "end_utc_exclusive":"1970-01-02T00:00:00Z"}}
    manifest = source_metadata(config)
    manifest.update({"schema":"rles.binance-archive-range.v1", "start_day":"1970-01-01",
        "end_day":"1970-01-01", "start_ns":MINUTE_NS, "end_ns":1440*MINUTE_NS,
        "event_count":marker_doc["events"]})
    (source_dir/"import.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study.legacy, "ROOT", tmp_path)
    source, loaded = study._load_source(config, "a"*64, "data/synthetic")
    assert source["market_path"] == "data/synthetic/market.parquet"
    assert loaded["event_count"] == marker_doc["events"]
    assert source["read_scope_start_ns"] >= 0 and source["read_scope_end_ns"] <= 1440*MINUTE_NS


def test_candidate_collector_rejects_different_registered_teacher_set(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    config = json.loads((ROOT/"configs/route_a_meta_x1_train_prereg_20261029.json").read_text())
    config["teachers"] = config["teachers"][:-1]
    monkeypatch.setattr(study, "_load_source", lambda *args: ({"marker_sha256":"a"*64}, {}))
    monkeypatch.setattr(study, "_warmup_scaler", lambda *args: "scaler")
    monkeypatch.setattr(study, "_factory", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(study, "_provider", lambda *args: pytest.fail("candidate collection must not start"))
    with pytest.raises(ValueError, match="registered teacher set"):
        study.execute_study(config, "b"*64, source_root="unused", output=tmp_path/"study")



@pytest.mark.parametrize("reject_stress,expected", [(False, 48), (True, 46)])
def test_runner_dispatches_all_native_cells_or_preserves_missing_model(
        tmp_path, monkeypatch, reject_stress, expected):
    import json
    from types import SimpleNamespace
    from rles_route_a.teacher_research.train_meta import MetaTrainingError

    config = json.loads((ROOT/"configs/route_a_meta_x1_train_prereg_20261029.json").read_text())
    source = {"marker_sha256":"a"*64}
    monkeypatch.setattr(study, "_load_source", lambda *args: (source, {}))
    monkeypatch.setattr(study, "_warmup_scaler", lambda *args: "scaler")
    monkeypatch.setattr(study, "_factory", lambda *args, **kwargs:
                        SimpleNamespace(scenario=kwargs["scenario"]))
    grant = SimpleNamespace(source_identity="a"*64)
    monkeypatch.setattr(study, "_provider", lambda *args: (object(), grant))
    monkeypatch.setattr(study, "_prepare_context", lambda *args: None)
    monkeypatch.setattr(study, "run_native_candidate_collection", lambda **kwargs: None)
    def train(**kwargs):
        if reject_stress and kwargs["scenario"] == "stress":
            raise MetaTrainingError("insufficient", fit_outcomes=(),
                                    calibration_outcomes=(), diagnostics={})
        return SimpleNamespace(policy=object())
    monkeypatch.setattr(study, "train_and_save_from_native", train)
    calls = []
    def cell(**kwargs):
        calls.append((kwargs["strategy"], kwargs["scenario"], tuple(kwargs["window"])))
        return {"strategy_id":kwargs["strategy"], "scenario":kwargs["scenario"],
                "window":kwargs["window"], "cases":(), "equity_curve":(1000.,1000.),
                "native_order_verification":{"barriers":10,"authority":"retained_m22_session_snapshots"}}
    monkeypatch.setattr(study, "_run_screen_cell", cell)
    result = study.execute_study(config, "b"*64, source_root="unused", output=tmp_path/"study")
    assert result["expected_cells"] == 48 and result["available_cells"] == expected
    assert len(calls) == expected and (tmp_path/"study"/"comparison.json").is_file()
    assert result["qualification"] == "NOT_ESTABLISHED"
    assert result["status"] == ("INCOMPLETE" if reject_stress else "COMPLETE")
    assert ("stress" in result["training_failures"]) == reject_stress
