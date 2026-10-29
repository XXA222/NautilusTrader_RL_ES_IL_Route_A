"""Synthetic OHLC test against the actual Nautilus/Gym engine, never historical PnL."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pytest

from rles_data.core import CuratedStore, MINUTE_NS
from rles_data.io import read_curated, write_curated
from rles_orchestrator import application
from rles_route_a.integration.l2_adapter import CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES
from rles_route_a.teachers.recovery import RecoveryTeacher
from rles_route_a.teacher_research.meta_gated_teacher import MetaGatedTeacher
from rles_route_a.teacher_research.native_meta_funnel import MetaNativeFunnelAdapter
from rles_route_a.teacher_research.native_meta_rollout import run_registered_meta_rollout
from rles_route_a.teacher_research.optional_1m_context import bind_optional_1m_context
from rles_route_a.teacher_research.policy_io import save_frozen_entry_policy, load_frozen_entry_policy
from test_historical_factory import recipe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))
sys.path.insert(0, str(ROOT / "packages/route_a_imitation/tests"))
import run_route_a_1m_teacher_native as native
import run_route_a_train_student_screen_20261020 as harness
import run_route_a_meta_study_20261029 as study
from test_meta_gated_teacher import trained_policy

pytestmark = pytest.mark.micro_golden


def test_meta_teacher_actual_native_engine_with_synthetic_market(tmp_path):
    environment = recipe(tmp_path, bar_count=245, start_minute=150)
    store = read_curated(environment["market_path"])
    events = []
    for event in store.events:
        if event.kind in ("ohlcv_1m", "mark", "index"):
            minute = event.ts_event_ns // MINUTE_NS
            price = "1000" if minute < 154 else "1005" if minute < 158 else "995"
            payload = dict(event.payload)
            if event.kind == "ohlcv_1m":
                payload.update(open=price, high=price, low=price, close=price)
            else:
                payload["price"] = price
            event = replace(event, payload=payload)
        events.append(event)
    market = tmp_path / "meta-synthetic-market.parquet"
    write_curated(market, CuratedStore(events))
    environment["end_ns"] = 164 * MINUTE_NS
    environment["market_path"] = str(market)
    environment["rules_path"] = str(environment["rules_path"])
    environment["feature_profile"] = "causal_mtf_v4"
    environment["execution"]["zero_target_policy"] = "close"
    environment["coverage_policy"]["known_schedule_times_ns"] = {environment["instrument"]: ()}
    marker = market.with_name(market.name + ".COMPLETED.json")
    marker_sha = sha256(marker.read_bytes()).hexdigest()
    config = {"output":str(tmp_path/"registered-meta"), "environment":environment,
              "train_data_access":{"start_ns":MINUTE_NS,"end_ns":246*MINUTE_NS,
              "marker_sha256":marker_sha,"label":"train-meta-synthetic-native"}}
    factory = application.prepare(config, fit_scaler=True, compact_report_snapshots=False,
                                  require_scoped_train_access=True)
    base = trained_policy()
    lower, upper, mean = base.lower.copy(), base.upper.copy(), base.mean.copy()
    lower[0], upper[0], mean[0] = 990., 1010., 1000.
    # This synthetic fixture permits teacher combinations observed in its
    # generated price jump; the frozen real-study thresholds stay unchanged.
    lower[1], upper[1] = 0., 1.
    lower[2], upper[2] = 1., 8.
    lower[3] = upper[3] = mean[3] = 22.
    policy = replace(base, lower=lower, upper=upper, mean=mean,
        source_id=marker_sha, fold_id="synthetic-meta-native",
        feature_spec_hash=factory.features.hash, available_after_ns=1,
        label_recipe=("X1", factory.route_a_execution_recipe_hash, "base"),
        fee_bps_per_leg=10., slippage_bps_per_leg=1., cost_bps=22.)
    policy_path = tmp_path / "synthetic-policy.json"
    save_frozen_entry_policy(policy, policy_path)
    policy = load_frozen_entry_policy(policy_path)
    teacher = MetaGatedTeacher(policy, recovery=RecoveryTeacher())
    old_fold = harness.FOLD_ID
    harness.FOLD_ID = "synthetic-meta-native"
    try:
        provider, grant = harness.make_provider(factory, teacher, marker_sha, "meta-native-synthetic",
                                                market_feature_map=CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES)
        context = native._StreamingVisibleContext(factory,
            start_ns=environment["start_ns"], end_ns=environment["end_ns"],
            instrument=environment["instrument"], warmup_bars=61)
        bind_optional_1m_context(provider, context)
        hook = native._TeacherResearchHook1m(factory=factory, teacher=teacher, output=tmp_path/"evidence",
            source_identity=marker_sha, scenario="base", exit_variant="X1", block_id="synthetic")
        funnel = MetaNativeFunnelAdapter(teacher=teacher, native_hook=hook)
        barriers, report = run_registered_meta_rollout(provider=provider, teacher=teacher,
            factory=factory, host_rollout=application.rollout, grant=grant,
            verify_grant=lambda g: g == grant and factory.application_registered_train_read_verifier(
                factory.application_registered_train_read),
            scenario="base", seed=7, rollout_id="meta-native-synthetic", episode_steps=10,
            trace_path=tmp_path/"trace.jsonl", reset_options={"rles_episode_index":0},
            teacher_funnel_hook=funnel)
        cell_dir = tmp_path / "study-cell"
        cell_dir.mkdir()
        experiment = {"teachers":["T01_EMA_9_21"],
            "cost_scenarios":{"base":{"fee_bps_per_leg":10., "slippage_bps_per_leg":1.}},
            "controls":["ORIGINAL_8_X1","FLAT","FIXED_LONG_0.3","FIXED_SHORT_0.3"],
            "intervals":{"native_screen":[["1970-01-01T02:30:00Z","1970-01-01T02:44:00Z"]]}}
        window = experiment["intervals"]["native_screen"][0]
        cell = study._run_screen_cell(factory=factory, source={"marker_sha256":marker_sha},
            config=experiment, strategy="META_X1", scenario="base", window=window,
            model=policy, output=cell_dir)
        comparison_rows = [cell]
        for strategy in ("T01_EMA_9_21", "FLAT", "FIXED_LONG_0.3", "FIXED_SHORT_0.3"):
            path = tmp_path / ("study-" + strategy)
            path.mkdir()
            comparison_rows.append(study._run_screen_cell(factory=factory,
                source={"marker_sha256":marker_sha}, config=experiment,
                strategy=strategy, scenario="base", window=window, model=policy, output=path))
        matrix = study.compare_preregistered_native_runs(experiment, comparison_rows)
    finally:
        harness.FOLD_ID = old_fold
    assert len(barriers) == 10
    assert report["route_a_native_order_verification"]["barriers"] == 10
    assert report["route_a_native_order_verification"]["policy_fill_events"] >= 2, repr([
        (row.observation.decision_ts_ns, row.action.intent,
         [(item.group.side, item.group.teachers, item.disposition, item.score) for item in row.candidates])
        for row in teacher.decisions if row.candidates])
    assert (tmp_path/"trace.jsonl").is_file()
    assert any(item.selected for row in teacher.decisions for item in row.candidates)
    assert any(case.status == "ORDINARY_COMPLETE" for case in hook.cases)
    assert all(barrier.trace.scenario_only for barrier in barriers)
    assert cell["native_order_verification"]["barriers"] == 11
    assert cell["equity_curve"][0] == float(factory.capital)
    assert cell["equity_curve"][1:]  # Native samples follow the initial capital.
    assert (cell_dir/"native-result.json").is_file()
    assert (cell_dir/"candidate-to-native.jsonl").is_file()
    assert matrix["status"] == "COMPLETE" and matrix["available_cells"] == 5
    assert matrix["qualification"] == "NOT_ESTABLISHED"
