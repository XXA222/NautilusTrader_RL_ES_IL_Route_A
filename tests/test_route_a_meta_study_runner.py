from datetime import datetime, timedelta
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


def test_signal_coverage_requires_exact_registered_eight_teacher_set():
    from types import SimpleNamespace

    expected = tuple(sorted(study.ALL_TEACHER_RECIPE_BY_ID))
    signals = tuple((teacher_id, "NONE", "NO_SIGNAL") for teacher_id in expected)
    row = SimpleNamespace(candidates=(), candidate_error=None, teacher_signals=signals,
                          teacher_errors=())
    teacher = SimpleNamespace(decisions=(row,), policy=SimpleNamespace(margin_bps=5.))
    complete = study._score_diagnostics(teacher, None, {}, "base", 1, 2)
    assert complete["complete_eight_teacher_signal_coverage"] is True
    assert complete["quality_status"] == "COMPLETE_EIGHT_TEACHER_SIGNAL_COVERAGE"
    assert complete["candidate_quality_status"] == "NO_CANDIDATES"
    assert complete["outcomes"] == []
    assert complete["same_candidate_cost_comparison"] is None
    assert complete["missing_teacher_ids"] == []
    assert complete["unexpected_teacher_ids"] == []

    wrong = tuple(("NOT_REGISTERED" if teacher_id == expected[0] else teacher_id,
                   "NONE", "NO_SIGNAL") for teacher_id in expected)
    teacher.decisions = (SimpleNamespace(candidates=(), candidate_error=None,
        teacher_signals=wrong, teacher_errors=()),)
    incomplete = study._score_diagnostics(teacher, None, {}, "base", 1, 2)
    assert incomplete["complete_eight_teacher_signal_coverage"] is False
    assert incomplete["quality_status"] == "INCOMPLETE_EIGHT_TEACHER_SIGNAL_COVERAGE"
    assert incomplete["missing_teacher_ids"] == [expected[0]]
    assert incomplete["unexpected_teacher_ids"] == ["NOT_REGISTERED"]


def test_incomplete_teacher_coverage_keeps_scores_but_skips_candidate_quality_replay(monkeypatch):
    from types import SimpleNamespace

    expected = tuple(sorted(study.ALL_TEACHER_RECIPE_BY_ID))
    signals = tuple((teacher_id, "NONE", "NO_SIGNAL") for teacher_id in expected[:-1])
    candidate = SimpleNamespace(score=None, disposition="REJECTED_TEACHER_SIGNAL_INCOMPLETE",
                                rejection_override=None, selected=False, missing_features=())
    row = SimpleNamespace(candidates=(candidate,), candidate_error=None,
                          teacher_signals=signals, teacher_errors=((expected[-1], "failed"),))
    teacher = SimpleNamespace(decisions=(row,), policy=SimpleNamespace(margin_bps=5.))
    monkeypatch.setattr(study, "extract_model_price_path",
                        lambda *_args, **_kwargs: pytest.fail("incomplete candidate union was replayed"))

    result = study._score_diagnostics(teacher, object(), {}, "base", 1, 2)

    assert result["complete_eight_teacher_signal_coverage"] is False
    assert result["quality"] is None
    assert result["quality_status"] == "INCOMPLETE_EIGHT_TEACHER_SIGNAL_COVERAGE"
    assert result["candidate_disposition_counts"] == {
        "REJECTED_TEACHER_SIGNAL_INCOMPLETE": 1,
    }


def test_candidate_collection_error_is_retained_without_quality_replay(monkeypatch):
    from types import SimpleNamespace

    expected = tuple(sorted(study.ALL_TEACHER_RECIPE_BY_ID))
    signals = tuple((teacher_id, "NONE", "NO_SIGNAL") for teacher_id in expected)
    row = SimpleNamespace(candidates=(), candidate_error="ValueError: collection failed",
                          teacher_signals=signals, teacher_errors=())
    teacher = SimpleNamespace(decisions=(row,), policy=SimpleNamespace(margin_bps=5.))
    monkeypatch.setattr(study, "extract_model_price_path",
                        lambda *_args, **_kwargs: pytest.fail("failed collection was replayed"))

    result = study._score_diagnostics(teacher, object(), {}, "base", 1, 2)

    assert result["complete_eight_teacher_signal_coverage"] is True
    assert result["candidate_error_count"] == 1
    assert result["candidate_error_types"] == {"ValueError": 1}
    assert result["quality"] is None
    assert result["quality_status"] == "CANDIDATE_COLLECTION_ERROR"


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
    """Run the real importer and loader against generated ZIP/REST fixtures."""
    import json
    from hashlib import sha256
    from zipfile import ZipFile
    from rles_data import binance_archive
    from rles_data.core import MINUTE_NS
    source_dir = tmp_path/"data"/"synthetic"
    def fake_fetch(relative_url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for minute in range(1440):
            opened = minute * 60000
            rows.append(f"{opened},100,101,99,100,1,{opened+59999},100,1,0.5,50,0")
        with ZipFile(destination, "w") as archive:
            archive.writestr("synthetic.csv", "\n".join(rows) + "\n")
        return {"path":str(destination), "url":"https://data.binance.vision/data/futures/um/"+relative_url,
                "sha256":sha256(destination.read_bytes()).hexdigest()}
    def fake_funding(symbol, month, first, last, raw):
        rows = [{"symbol":symbol, "fundingTime":hour*3_600_000, "fundingRate":"0.0001", "rateType":"Regular"}
                for hour in (0, 8, 16)]
        content = json.dumps(rows).encode()
        query = urlencode({"symbol":symbol, "startTime":0, "endTime":86_400_000-1, "limit":1000})
        return {"path":str(raw/"synthetic-funding.json"),
                "url":"https://fapi.binance.com/fapi/v1/fundingRate?"+query,
                "sha256":sha256(content).hexdigest()}, content
    monkeypatch.setattr(binance_archive, "_guard_project_archive_scope", lambda **kwargs: None)
    monkeypatch.setattr(binance_archive, "fetch_archive", fake_fetch)
    monkeypatch.setattr(binance_archive, "fetch_funding_api", fake_funding)
    imported = binance_archive.import_range(symbol="ETHUSDT", start_day="1970-01-01", end_day="1970-01-01",
        output=source_dir, delay_ns=1_000_000, funding_api_months=("1970-01",))
    marker = source_dir/"market.parquet.COMPLETED.json"
    marker_doc = json.loads(marker.read_text(encoding="utf-8"))
    config = {"protocol_id":"synthetic-test", "market":"ETHUSDT",
              "source":{"start_utc_inclusive":"1970-01-01T00:00:00Z",
                        "end_utc_exclusive":"1970-01-02T00:00:00Z"}}
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study.legacy, "ROOT", tmp_path)
    source, loaded = study._load_source(config, "a"*64, "data/synthetic")
    assert source["market_path"] == "data/synthetic/market.parquet"
    assert loaded["event_count"] == marker_doc["events"]
    assert imported["event_count"] == loaded["event_count"]
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
                "account_metrics":{"total_fee":0.,"total_slippage_cost":0.,"total_funding":0.},
                "native_order_verification":{"barriers":10,"authority":"retained_m22_session_snapshots"}}
    monkeypatch.setattr(study, "_run_screen_cell", cell)
    result = study.execute_study(config, "b"*64, source_root="unused", output=tmp_path/"study")
    assert result["expected_cells"] == 48 and result["available_cells"] == expected
    assert len(calls) == expected and (tmp_path/"study"/"comparison.json").is_file()
    assert result["qualification"] == "NOT_ESTABLISHED"
    assert result["status"] == ("INCOMPLETE" if reject_stress else "COMPLETE")
    assert ("stress" in result["training_failures"]) == reject_stress
