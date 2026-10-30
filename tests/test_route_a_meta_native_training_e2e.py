"""Synthetic market: actual native candidates, X1 labels and calibrated model."""
from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
from math import sin
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from zipfile import ZipFile

from rles_data.core import CuratedStore, MINUTE_NS
from rles_data import binance_archive
from rles_data.io import read_curated, write_curated
from rles_orchestrator import application
from rles_route_a.integration.l2_adapter import CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES
from rles_route_a.teachers.recovery import RecoveryTeacher
from rles_route_a.teachers_1m.registry import build_teacher_1m, ALL_TEACHER_RECIPE_BY_ID
from rles_route_a.teacher_research.native_candidate_recorder import NativeCandidateRecorder, run_native_candidate_collection
from rles_route_a.teacher_research.native_training_data import extract_model_price_path
from rles_route_a.teacher_research.optional_1m_context import bind_optional_1m_context
from rles_route_a.teacher_research.risk_matched_teacher import RiskMatchedTeacher
from rles_route_a.teacher_research.x1_group_replay import make_x1_group_replay
from rles_route_a.teacher_research.train_meta import train_meta_on_registered_candidates
from rles_route_a.teacher_research.policy_io import save_frozen_entry_policy, load_frozen_entry_policy
from test_historical_factory import recipe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))
import run_route_a_1m_teacher_native as native
import run_route_a_train_student_screen_20261020 as harness
import run_route_a_meta_study_20261029 as study


def test_actual_native_candidate_collection_trains_synthetic_model(tmp_path, monkeypatch):
    environment = recipe(tmp_path, bar_count=1439, start_minute=150,
                         funding_settlement_minutes=(480,), taker_fee="0.0005")
    # Exercise the frozen Train economics exactly: 5/1 bps per leg for Base,
    # 10/5 for Stress, with the registered fee represented in the synthetic
    # instrument rule and a 2x stress fee multiplier.
    environment["execution"]["stress_fill"].update({"fee_multiplier":"2", "slippage_bps":"5"})
    preregistered_model = json.loads((ROOT/"configs/route_a_meta_x1_train_prereg_20261029.json").read_text())["model"]
    events = []
    for event in read_curated(environment["market_path"]).events:
        if event.kind in ("ohlcv_1m", "mark", "index"):
            minute = event.ts_event_ns // MINUTE_NS
            price = f"{1000 + 18*sin(minute/3) + 9*sin(minute/11):.2f}"
            payload = dict(event.payload)
            if event.kind == "ohlcv_1m":
                payload.update(open=price, high=price, low=price, close=price)
            else:
                payload["price"] = price
            event = replace(event, payload=payload)
        events.append(event)
    source_dir = tmp_path/"data"/"synthetic"
    source_dir.mkdir(parents=True)
    market = source_dir/"market.parquet"
    write_curated(market, CuratedStore(events))
    environment.update(end_ns=330*MINUTE_NS, market_path=str(market), feature_profile="causal_mtf_v4")
    environment["rules_path"] = str(environment["rules_path"])
    environment["execution"]["zero_target_policy"] = "close"
    environment["coverage_policy"]["known_schedule_times_ns"] = {environment["instrument"]: ()}
    marker = market.with_name(market.name + ".COMPLETED.json")
    marker_sha = sha256(marker.read_bytes()).hexdigest()
    config = {"output":str(tmp_path/"training-run"), "environment":environment,
              "train_data_access":{"start_ns":MINUTE_NS, "end_ns":1440*MINUTE_NS,
              "marker_sha256":marker_sha, "label":"train-synthetic-native-training"}}
    factory = application.prepare(config, fit_scaler=True, compact_report_snapshots=False,
                                  require_scoped_train_access=True)
    recorder = NativeCandidateRecorder(RiskMatchedTeacher(build_teacher_1m("T01_EMA_9_21"), RecoveryTeacher()))
    old_fold = harness.FOLD_ID
    harness.FOLD_ID = "synthetic-train-e2e"
    try:
        provider, grant = harness.make_provider(factory, recorder, marker_sha, "synthetic-native-training",
            market_feature_map=CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES)
        context = native._StreamingVisibleContext(factory, start_ns=environment["start_ns"],
            end_ns=environment["end_ns"], instrument=environment["instrument"], warmup_bars=61)
        bind_optional_1m_context(provider, context)
        _, _, records = run_native_candidate_collection(provider=provider, recorder=recorder,
            factory=factory, host_rollout=application.rollout, grant=grant,
            verify_grant=lambda g: g == grant and factory.application_registered_train_read_verifier(
                factory.application_registered_train_read), seed=7, rollout_id="synthetic-native-training",
            episode_steps=177, reset_options={"rles_episode_index":0})
    finally:
        harness.FOLD_ID = old_fold
    split = 240*MINUTE_NS
    groups = tuple(g for row in records for g in row.collection.groups)
    fit = tuple(g for g in groups if g.observation.decision_ts_ns < split)
    cal = tuple(g for g in groups if g.observation.decision_ts_ns >= split)
    opens, funding = extract_model_price_path(factory, start_ns=environment["start_ns"],
        end_ns=environment["end_ns"], funding_after_ns=records[0].observation.decision_ts_ns)
    replay = make_x1_group_replay(observations=tuple(row.observation for row in records),
        opens=opens, funding=funding, block_ends={g.event_id: split if g.observation.decision_ts_ns < split
                                                 else environment["end_ns"] for g in groups},
        fee_bps=5., slippage_bps=1., execution_recipe=factory.route_a_execution_recipe_hash, scenario="base")
    result = train_meta_on_registered_candidates(grant=grant, verify_grant=lambda g: g == grant,
        fit_groups=fit, calibration_groups=cal, fit_interval=(environment["start_ns"], split),
        calibration_interval=(split, environment["end_ns"]), replay=replay,
        execution_recipe=factory.route_a_execution_recipe_hash, scenario="base",
        feature_names=preregistered_model["features"],
        margin_bps=5., cost_bps=12., fee_bps_per_leg=5., slippage_bps_per_leg=1.)
    path = tmp_path/"synthetic-native-model.json"
    save_frozen_entry_policy(result.policy, path)
    restored = load_frozen_entry_policy(path)
    assert restored.fit_count >= 4 and restored.calibration_count >= 4
    assert restored.available_after_ns < 330*MINUTE_NS
    assert any(abs(row.observation.account_value("net_exposure")) > 0 for row in records)
    assert all(outcome.evidence_type == "MODEL_COUNTERFACTUAL" for outcome in result.fit_outcomes)
    stress_replay = make_x1_group_replay(observations=tuple(row.observation for row in records),
        opens=opens, funding=funding, block_ends={g.event_id: split if g.observation.decision_ts_ns < split
                                                 else environment["end_ns"] for g in groups},
        fee_bps=10., slippage_bps=5., execution_recipe=factory.route_a_execution_recipe_hash, scenario="stress")
    stress_result = train_meta_on_registered_candidates(grant=grant, verify_grant=lambda g: g == grant,
        fit_groups=fit, calibration_groups=cal, fit_interval=(environment["start_ns"], split),
        calibration_interval=(split, environment["end_ns"]), replay=stress_replay,
        execution_recipe=factory.route_a_execution_recipe_hash, scenario="stress",
        feature_names=preregistered_model["features"],
        margin_bps=5., cost_bps=30., fee_bps_per_leg=10., slippage_bps_per_leg=5.)
    stress_path = tmp_path/"synthetic-native-stress-model.json"
    save_frozen_entry_policy(stress_result.policy, stress_path)
    stress_model = load_frozen_entry_policy(stress_path)
    screen_environment = deepcopy(environment)
    screen_environment.update(start_ns=330*MINUTE_NS, end_ns=360*MINUTE_NS)
    screen_config = {**config, "environment": screen_environment}
    screen_factory = application.prepare(screen_config, fit_scaler=False, scaler=factory.scaler,
                                         compact_report_snapshots=False, require_scoped_train_access=True)
    stress_environment = deepcopy(screen_environment)
    stress_environment["mode"] = "stress"
    stress_factory = application.prepare({**config, "environment":stress_environment}, fit_scaler=False,
        scaler=factory.scaler, compact_report_snapshots=False, require_scoped_train_access=True)
    experiment = {"protocol_id":"synthetic-train-e2e", "teachers":sorted(ALL_TEACHER_RECIPE_BY_ID),
        "cost_scenarios":{"base":{"fee_bps_per_leg":5., "slippage_bps_per_leg":1.},
                          "stress":{"fee_bps_per_leg":10., "slippage_bps_per_leg":5.}},
        "controls":["ORIGINAL_8_X1","FLAT","FIXED_LONG_0.3","FIXED_SHORT_0.3"],
        "intervals":{"train_fit":["1970-01-01T02:30:00Z","1970-01-01T04:00:00Z"],
                     "train_calibration":["1970-01-01T04:00:00Z","1970-01-01T05:30:00Z"],
                     "native_screen":[["1970-01-01T05:30:00Z","1970-01-01T06:00:00Z"],
                                      ["1970-01-01T06:00:00Z","1970-01-01T06:30:00Z"]]}}
    source_receipt = {"marker_sha256":marker_sha,
                      "snapshot_hash":sha256(b"synthetic-snapshot").hexdigest(),
                      "import_manifest_sha256":sha256(b"synthetic-import-manifest").hexdigest()}
    cell_dir = tmp_path/"synthetic-frozen-model-screen"
    cell_dir.mkdir()
    old_fold = harness.FOLD_ID
    harness.FOLD_ID = "synthetic-train-e2e"
    try:
        cell = study._run_screen_cell(factory=screen_factory, source=source_receipt,
            config=experiment, prereg_sha=sha256(b"synthetic-study-prereg").hexdigest(),
            strategy="META_X1", scenario="base",
            window=experiment["intervals"]["native_screen"][0], model=restored, output=cell_dir)
        runs = [cell]
        for strategy in (*experiment["teachers"], "FLAT", "FIXED_LONG_0.3", "FIXED_SHORT_0.3"):
            target = tmp_path / ("synthetic-screen-" + strategy)
            target.mkdir()
            runs.append(study._run_screen_cell(factory=screen_factory, source=source_receipt,
                config=experiment, prereg_sha=sha256(b"synthetic-study-prereg").hexdigest(),
                strategy=strategy, scenario="base",
                window=experiment["intervals"]["native_screen"][0], model=restored, output=target))
        for strategy in ("META_X1", *experiment["teachers"], "FLAT", "FIXED_LONG_0.3", "FIXED_SHORT_0.3"):
            target = tmp_path / ("synthetic-stress-" + strategy)
            target.mkdir()
            runs.append(study._run_screen_cell(factory=stress_factory, source=source_receipt,
                config=experiment, prereg_sha=sha256(b"synthetic-study-prereg").hexdigest(),
                strategy=strategy, scenario="stress",
                window=experiment["intervals"]["native_screen"][0], model=stress_model, output=target))
        for scenario, model in (("base", restored), ("stress", stress_model)):
            later_environment = deepcopy(screen_environment)
            later_environment.update(start_ns=360*MINUTE_NS, end_ns=390*MINUTE_NS, mode=scenario)
            later_factory = application.prepare({**config, "environment":later_environment}, fit_scaler=False,
                scaler=factory.scaler, compact_report_snapshots=False, require_scoped_train_access=True)
            for strategy in ("META_X1", *experiment["teachers"], "FLAT", "FIXED_LONG_0.3", "FIXED_SHORT_0.3"):
                target = tmp_path / ("synthetic-later-" + scenario + "-" + strategy)
                target.mkdir()
                runs.append(study._run_screen_cell(factory=later_factory, source=source_receipt,
                    config=experiment, prereg_sha=sha256(b"synthetic-study-prereg").hexdigest(),
                    strategy=strategy, scenario=scenario,
                    window=experiment["intervals"]["native_screen"][1], model=model, output=target))
        comparison = study.compare_preregistered_native_runs(experiment, runs)
    finally:
        harness.FOLD_ID = old_fold
    assert cell["native_order_verification"]["barriers"] == 27
    assert cell["candidate_summary"]["scores"]["candidate_count"] > 0
    assert (cell_dir/"candidate-to-native.jsonl").is_file()
    assert comparison["status"] == "INCOMPLETE" and comparison["available_cells"] == 48
    assert comparison["expected_cells"] == 52 and comparison["active_status"] == "COMPLETE"
    assert comparison["preregistered_controls_not_executed"] == ["ORIGINAL_8_X1"]
    assert len(comparison["comparisons"]) == 44
    assert all(item["status"] == ("INCOMPLETE" if item["baseline"] == "ORIGINAL_8_X1" else "COMPLETE")
               for item in comparison["window_consistency"])
    assert comparison["qualification"] == "NOT_ESTABLISHED"

    # Generate offline exchange-format inputs, then exercise the production
    # importer, source loader, scaler, factories, training and coordinator.
    def timestamp(minute):
        return datetime.fromtimestamp(minute * 60, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    prereg = json.loads((ROOT/"configs/route_a_meta_x1_train_prereg_20261029.json").read_text())
    prereg["protocol_id"] = "synthetic-native-full-study"
    prereg["source"].update({"start_utc_inclusive":timestamp(0), "end_utc_exclusive":timestamp(1440)})
    prereg["intervals"] = {"warmup":[timestamp(1),timestamp(272)],
        "train_fit":[timestamp(272),timestamp(360)],
        "train_calibration":[timestamp(360),timestamp(450)],
        "native_screen":[[timestamp(450),timestamp(480)],[timestamp(480),timestamp(510)]]}
    prereg["cost_scenarios"] = experiment["cost_scenarios"]
    prereg["model"].update({"minimum_complete_fit":4, "minimum_complete_calibration":4})
    prereg_path = tmp_path/"synthetic-study.json"
    raw = json.dumps(prereg).encode()
    prereg_path.write_bytes(raw)
    prereg_path.with_suffix(".json.sha256").write_text(sha256(raw).hexdigest())
    monkeypatch.setattr(study, "PREREG", prereg_path)
    def fake_fetch(relative_url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for minute in range(1440):
            opened = minute*60000
            price = f"{1000 + 18*sin((minute+1)/3) + 9*sin((minute+1)/11):.2f}"
            rows.append(f"{opened},{price},{price},{price},{price},1000,{opened+59999},1000000,100,500,500000,0")
        with ZipFile(destination, "w") as archive:
            archive.writestr("synthetic.csv", "\n".join(rows) + "\n")
        return {"path":str(destination), "url":"https://data.binance.vision/data/futures/um/"+relative_url,
                "sha256":sha256(destination.read_bytes()).hexdigest()}
    def fake_funding(symbol, month, first, last, raw_dir):
        rows = [{"symbol":symbol, "fundingTime":hour*3_600_000+1, "fundingRate":"0.001", "rateType":"Regular"}
                for hour in (0, 8, 16)]
        content = json.dumps(rows).encode()
        query = urlencode({"symbol":symbol, "startTime":0, "endTime":86400000-1, "limit":1000})
        return {"path":str(raw_dir/"synthetic-funding.json"),
                "url":"https://fapi.binance.com/fapi/v1/fundingRate?"+query,
                "sha256":sha256(content).hexdigest()}, content
    monkeypatch.setattr(binance_archive, "_guard_project_archive_scope", lambda **kwargs: None)
    monkeypatch.setattr(binance_archive, "fetch_archive", fake_fetch)
    monkeypatch.setattr(binance_archive, "fetch_funding_api", fake_funding)
    imported = tmp_path/"data"/"imported-synthetic"
    binance_archive.import_range(symbol="ETHUSDT", start_day="1970-01-01", end_day="1970-01-01",
        output=imported, delay_ns=1_000_000, funding_api_months=("1970-01",))
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study.legacy, "ROOT", tmp_path)
    template = tmp_path/"synthetic-execution.json"
    environment["coverage_policy"]["maximum_age_ns"]["funding_update"] = 500*MINUTE_NS
    environment["coverage_policy"]["maximum_age_ns"].update(mark=61_000_000_000, index=61_000_000_000)
    template.write_text(json.dumps({"environment":environment}))
    monkeypatch.setattr(study.legacy, "EXECUTION_TEMPLATE", template)
    monkeypatch.setattr(study.legacy, "SCENARIO_RULES", Path(environment["rules_path"]))
    old_fold = harness.FOLD_ID
    harness.FOLD_ID = "synthetic-full-study"
    try:
        full = study.execute_study(prereg, sha256(raw).hexdigest(), source_root="data/imported-synthetic",
                                   output=tmp_path/"synthetic-full-study")
    finally:
        harness.FOLD_ID = old_fold
    assert full["status"] == "INCOMPLETE" and full["available_cells"] == 48
    assert full["expected_cells"] == 52 and full["active_status"] == "COMPLETE"
    assert full["preregistered_controls_not_executed"] == ["ORIGINAL_8_X1"]
    assert full["training_failures"] == {}
    assert full["qualification"] == "NOT_ESTABLISHED"
    for scenario in ("base", "stress"):
        training = tmp_path/"synthetic-full-study"/"training"/scenario
        assert (training/"COMPLETE").is_file() and (training/"policy.json").is_file()
        assert (training/"candidate-barriers.jsonl").is_file() and (training/"outcomes.jsonl").is_file()
    assert (tmp_path/"synthetic-full-study"/"comparison.json").is_file()
