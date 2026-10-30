"""Run the independent Route A Meta X1 Train study on the existing native host.

The source is not fetched by this script. It checks the frozen reservation and
complete source metadata before any market payload is decoded. Historical
development results remain scenario-only and are not a qualification claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

import run_route_a_1m_teacher_native as native
import run_route_a_teacher_student_train_screen_20261026 as legacy
import run_route_a_train_student_screen_20261020 as harness
import validate_route_a_meta_prereg_20261029 as preflight

from rles_orchestrator import application
from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_research.native_metrics import native_report
from rles_route_a.integration.l2_adapter import CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES
from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.teachers.recovery import RecoveryTeacher
from rles_route_a.teachers_1m.registry import ALL_TEACHER_RECIPE_BY_ID, build_teacher_1m
from rles_route_a.teacher_research.conditional_outcome_labels import label_grouped_candidates
from rles_route_a.teacher_research.entry_meta_policy import summarize_entry_scores
from rles_route_a.teacher_research.execution_binding import validate_native_execution
from rles_route_a.teacher_research.meta_evaluation import (compare_preregistered_native_runs,
    summarize_candidate_quality, paired_cost_candidate_quality)
from rles_route_a.teacher_research.meta_gated_teacher import MetaGatedTeacher
from rles_route_a.teacher_research.native_candidate_recorder import NativeCandidateRecorder, run_native_candidate_collection
from rles_route_a.teacher_research.native_reconciliation import reconcile_selected_native_cases
from rles_route_a.teacher_research.native_meta_funnel import MetaNativeFunnelAdapter
from rles_route_a.teacher_research.native_meta_rollout import run_registered_meta_rollout
from rles_route_a.teacher_research.optional_1m_context import bind_optional_1m_context
from rles_route_a.teacher_research.native_training_data import extract_model_price_path, train_and_save_from_native
from rles_route_a.teacher_research.risk_matched_teacher import RiskMatchedTeacher
from rles_route_a.teacher_research.train_meta import MetaTrainingError
from rles_route_a.teacher_research.x1_group_replay import make_x1_group_replay

MINUTE_NS = 60_000_000_000
PREREG = ROOT / "configs/route_a_meta_x1_train_prereg_20261029.json"
DEFAULT_SOURCE = "data/routea_meta_train_20261201_20261218"
DEFAULT_OUTPUT = ".artifacts/route-a-meta-independent-20261201-20261219"


def _inside_project(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("study paths must be relative to the canonical project")
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError("study path escapes the canonical project")
    return result


def _validate_physical_sources(config: dict, manifest: dict) -> None:
    """Reject out-of-scope physical archives before touching curated payload bytes."""
    start = datetime.fromisoformat(config["source"]["start_utc_inclusive"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(config["source"]["end_utc_exclusive"].replace("Z", "+00:00"))
    expected = set()
    day = start
    while day < end:
        for stream in ("klines", "markPriceKlines", "indexPriceKlines"):
            expected.add(f"https://data.binance.vision/data/futures/um/daily/{stream}/"
                         f"ETHUSDT/1m/ETHUSDT-1m-{day.date().isoformat()}.zip")
        day += timedelta(days=1)
    if (manifest.get("archive_interval") != "daily" or manifest.get("symbol") != config["market"]
            or manifest.get("instrument") != legacy.INSTRUMENT
            or manifest.get("availability_delay_ns") != 1_000_000
            or manifest.get("agg_trades_included") is not False
            or manifest.get("bookticker_included") is not False
            or not isinstance(manifest.get("sources"), list)):
        raise ValueError("source manifest does not match the preregistered daily market scope")
    seen, funding_count = set(), 0
    for item in manifest["sources"]:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            raise ValueError("source provenance is missing")
        url = item["url"]
        digest = item.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("source provenance has no SHA-256 digest")
        if url in expected:
            if url in seen:
                raise ValueError("duplicate daily source archive")
            seen.add(url)
            continue
        parsed = urlparse(url)
        query = parse_qs(parsed.query, strict_parsing=True)
        expected_query = {"symbol": ["ETHUSDT"], "startTime": [str(int(start.timestamp()*1000))],
                          "endTime": [str(int(end.timestamp()*1000)-1)], "limit": ["1000"]}
        if (parsed.scheme != "https" or parsed.netloc != "fapi.binance.com"
                or parsed.path != "/fapi/v1/fundingRate" or parsed.fragment
                or query != expected_query):
            raise ValueError("source archive or funding query exceeds the preregistered physical scope")
        funding_count += 1
    if seen != expected or funding_count != 1:
        raise ValueError("daily source archive or bounded funding coverage is incomplete")


def _load_source(config: dict, prereg_sha: str, source_root: str):
    """Reuse existing daily-source metadata and physical-scope verification."""
    source_dir = _inside_project(source_root)
    manifest_path = source_dir / "import.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("complete independently registered Train source is unavailable")
    # Metadata-only validation precedes legacy verification, which hashes market.parquet.
    _validate_physical_sources(config, json.loads(manifest_path.read_text(encoding="utf-8")))
    start = datetime.fromisoformat(config["source"]["start_utc_inclusive"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(config["source"]["end_utc_exclusive"].replace("Z", "+00:00"))
    spec = {"output": source_root, "start_day": start.date().isoformat(),
            "end_day_inclusive": (end - timedelta(days=1)).date().isoformat(),
            "funding_api_months": sorted({f"{start.year:04d}-{start.month:02d}",
                                          f"{(end-timedelta(days=1)).year:04d}-{(end-timedelta(days=1)).month:02d}"})}
    parent = {"protocol_id": config["protocol_id"], "scope": {
        "source_start_utc_inclusive": config["source"]["start_utc_inclusive"],
        "end_utc_exclusive": config["source"]["end_utc_exclusive"]}}
    authority = {"import_invocation": {"development_protocol_sha256": prereg_sha,
                                        "development_access_key": "access:" + config["protocol_id"]}}
    return legacy._verified_source(parent, authority, prereg_sha, spec)


def _verify_factory_grant(factory: Any, grant: HostTrainGrant) -> bool:
    read = getattr(factory, "application_registered_train_read", None)
    verifier = getattr(factory, "application_registered_train_read_verifier", None)
    return (callable(verifier) and verifier(read) is True
            and HostTrainGrant.from_registered_read(read, fold_id=grant.fold_id) == grant)


def _warmup_scaler(source, config, output):
    legacy._configure_harness(source, config, output)
    start, end = map(preflight.ns, config["intervals"]["warmup"])
    owner = ACTIVE_OWNER.set(config["protocol_id"])
    try:
        _, factory = harness.make_config(
            label=config["protocol_id"] + "-warmup-scaler",
            marker_sha=source["marker_sha256"], scope_start=source["read_scope_start_ns"],
            scope_end=source["read_scope_end_ns"], start_ns=start + 122 * MINUTE_NS,
            end_ns=end, mode="base", fit_scaler=True, source_mode=legacy.SOURCE_MODE,
            prepare_factory=True, known_funding_calendar=True)
    finally:
        ACTIVE_OWNER.reset(owner)
    if factory is None or factory.scaler.training_only is not True:
        raise RuntimeError("native warmup did not produce a Train-only scaler")
    return factory.scaler


def _factory(source, manifest, config, output, scaler, *, scenario, start_ns, end_ns):
    _, factory = legacy._prepare_factory(source, manifest, config, output,
        session_id="independent-meta-v1", start_ns=start_ns, end_ns=end_ns,
        scenario=scenario, scaler=scaler)
    return factory


def _provider(factory, teacher, source, episode):
    return harness.make_provider(factory, teacher, source["marker_sha256"], episode,
                                market_feature_map=CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES)


def _prepare_context(provider, factory, start_ns, end_ns):
    context = native._StreamingVisibleContext(
        factory, start_ns=start_ns, end_ns=end_ns, instrument=legacy.INSTRUMENT, warmup_bars=61)
    bind_optional_1m_context(provider, context)


def _steps(start_ns, end_ns):
    if (end_ns - start_ns) % MINUTE_NS:
        raise ValueError("native window must contain whole minutes")
    result = (end_ns - start_ns) // MINUTE_NS - 3
    if result <= 0:
        raise ValueError("native window is shorter than terminal settlement drain")
    return result


def _registered_screen_split_evidence(config, window, *, model_available_after_ns=None):
    """Prove temporal separation from the registered fit/calibration period.

    This is run-level evidence only. It does not claim that overlapping market
    events or candidate labels are statistically independent trials.
    """
    def interval(name):
        return tuple(map(preflight.ns, config["intervals"][name]))

    fit_start, fit_end = interval("train_fit")
    calibration_start, calibration_end = interval("train_calibration")
    screen = tuple(map(preflight.ns, window))
    registered_screens = tuple(tuple(map(preflight.ns, item))
                               for item in config["intervals"]["native_screen"])
    if screen not in registered_screens:
        raise ValueError("screen window is not present in the frozen preregistration")
    if not (fit_start < fit_end <= calibration_start < calibration_end <= screen[0] < screen[1]):
        raise ValueError("registered model and screen intervals overlap or are unordered")
    if (model_available_after_ns is not None
            and (type(model_available_after_ns) is not int
                 or model_available_after_ns > screen[0])):
        raise ValueError("frozen model was not available before its registered screen window")
    return {
        "status": "PASS_REGISTERED_TEMPORAL_SEPARATION",
        "fit_interval_ns": [fit_start, fit_end],
        "calibration_interval_ns": [calibration_start, calibration_end],
        "screen_interval_ns": list(screen),
        "model_available_after_ns": model_available_after_ns,
        "independent_trials_claimed": False,
    }


def _score_diagnostics(teacher, factory, config, scenario, start_ns, end_ns, native_cases=()):
    from collections import Counter
    decisions = tuple(candidate for row in teacher.decisions for candidate in row.candidates)
    errors = tuple(row.candidate_error for row in teacher.decisions if row.candidate_error)
    scores = tuple(item.score for item in decisions if item.score is not None)
    unscored = tuple(item for item in decisions if item.score is None)
    unscored_reasons = Counter(item.disposition for item in unscored)
    post_gate_overrides = Counter(
        item.rejection_override for item in decisions
        if item.rejection_override is not None and item.score is not None
        and item.score.action in ("LONG", "SHORT"))
    result = {"scores": summarize_entry_scores(
                  scores, margin_bps=teacher.policy.margin_bps,
                  unscored_count=len(unscored),
                  unscored_reason_counts=unscored_reasons,
                  native_selected_count=sum(item.selected for item in decisions),
                  post_gate_override_counts=post_gate_overrides),
              "unscored": len(unscored), "quality": None,
              "candidate_quality_status": "NOT_COMPUTED",
              "candidate_error_count": len(errors),
              "candidate_error_types": dict(sorted(Counter(
                  str(error).split(":", 1)[0] for error in errors).items())),
              "candidate_disposition_counts": dict(sorted(
                  Counter(item.disposition for item in decisions).items())),
              "missing_feature_counts": dict(sorted(Counter(
                  name for item in decisions for name in item.missing_features).items()))}
    reasons = {}
    for row in teacher.decisions:
        for name, raw, reason in row.teacher_signals:
            counts = reasons.setdefault(name, (Counter(), Counter()))
            counts[0][raw] += 1
            counts[1][reason] += 1
    result["teacher_signal_coverage"] = {name: {"raw_signal_counts": dict(sorted(raw.items())),
        "reason_counts": dict(sorted(reason.items()))} for name, (raw, reason) in sorted(reasons.items())}
    result["teacher_signal_error_counts"] = dict(sorted(Counter(
        teacher_id for row in teacher.decisions for teacher_id, _detail in row.teacher_errors
    ).items()))
    expected_teacher_ids = set(ALL_TEACHER_RECIPE_BY_ID)
    observed_teacher_ids = set(result["teacher_signal_coverage"])
    result["missing_teacher_ids"] = sorted(expected_teacher_ids - observed_teacher_ids)
    result["unexpected_teacher_ids"] = sorted(observed_teacher_ids - expected_teacher_ids)
    result["complete_eight_teacher_signal_coverage"] = (
        observed_teacher_ids == expected_teacher_ids
        and not result["teacher_signal_error_counts"]
        and all(sum(result["teacher_signal_coverage"][teacher_id]["raw_signal_counts"].values())
                == len(teacher.decisions) for teacher_id in expected_teacher_ids)
    )
    if not result["complete_eight_teacher_signal_coverage"]:
        # A partial teacher union changes the candidate denominator, so its
        # accepted/rejected counterfactual quality cannot be presented as a
        # complete eight-teacher comparison. Keep the per-minute signals,
        # errors and score funnel, but do not replay or summarize outcomes.
        result["quality_status"] = "INCOMPLETE_EIGHT_TEACHER_SIGNAL_COVERAGE"
        result["candidate_quality_status"] = "INCOMPLETE_TEACHER_COVERAGE"
        return result
    if errors:
        # Preserve completed native account evidence and the score funnel even
        # if the research candidate sidecar reported an error. Such a run is
        # still ineligible for candidate quality replay.
        result["quality_status"] = "CANDIDATE_COLLECTION_ERROR"
        result["candidate_quality_status"] = "COLLECTION_ERROR"
        return result
    result["quality_status"] = "COMPLETE_EIGHT_TEACHER_SIGNAL_COVERAGE"
    if not decisions:
        result["candidate_quality_status"] = "NO_CANDIDATES"
        result["outcomes"] = []
        result["same_candidate_cost_comparison"] = None
        result["native_case_reconciliation"] = None
        return result
    opens, funding = extract_model_price_path(factory, start_ns=start_ns, end_ns=end_ns,
                                               funding_after_ns=teacher.decisions[0].observation.decision_ts_ns)
    groups = tuple(item.group for item in decisions)
    costs = config["cost_scenarios"][scenario]
    replay = make_x1_group_replay(
        observations=tuple(row.observation for row in teacher.decisions),
        opens=opens, funding=funding, block_ends={g.event_id: end_ns for g in groups},
        fee_bps=costs["fee_bps_per_leg"], slippage_bps=costs["slippage_bps_per_leg"],
        execution_recipe=teacher.policy.label_recipe[1], scenario=scenario)
    labels = label_grouped_candidates(groups, replay, exit_policy_id="X1",
        execution_recipe=teacher.policy.label_recipe[1], cost_scenario_id=scenario,
        block_end_ns=end_ns, gross_return_method="simple")
    result["quality"] = summarize_candidate_quality(decisions, labels)
    result["candidate_quality_status"] = "AVAILABLE"
    result["outcomes"] = [asdict(label) for label in labels]
    result["native_case_reconciliation"] = reconcile_selected_native_cases(
        decisions, native_cases, replay, execution_recipe=teacher.policy.label_recipe[1], scenario=scenario)
    selected = tuple(d.group for d in decisions if d.selected)
    if selected and {"base", "stress"}.issubset(config["cost_scenarios"]):
        paired = {}
        for cost_scenario in ("base", "stress"):
            scenario_cost = config["cost_scenarios"][cost_scenario]
            other_replay = make_x1_group_replay(
                observations=tuple(row.observation for row in teacher.decisions), opens=opens, funding=funding,
                block_ends={g.event_id: end_ns for g in selected},
                fee_bps=scenario_cost["fee_bps_per_leg"], slippage_bps=scenario_cost["slippage_bps_per_leg"],
                execution_recipe=teacher.policy.label_recipe[1], scenario=cost_scenario)
            paired[cost_scenario] = label_grouped_candidates(selected, other_replay, exit_policy_id="X1",
                execution_recipe=teacher.policy.label_recipe[1], cost_scenario_id=cost_scenario,
                block_end_ns=end_ns, gross_return_method="simple")
        result["same_candidate_cost_comparison"] = paired_cost_candidate_quality(
            decisions, paired["base"], paired["stress"])
    else:
        result["same_candidate_cost_comparison"] = None
    return result


def _account_costs_including_first_row(account):
    """Include first-bar facts when anchoring equity to initial capital."""
    from math import fsum
    rows = account["account_rows"]
    return {**account["metrics"],
            "total_fee": fsum(row["fee"] for row in rows),
            "total_slippage_cost": fsum(row["slippage_cost"] for row in rows),
            "total_funding": fsum(row["funding"] for row in rows)}


def _run_screen_cell(*, factory, source, config, prereg_sha, strategy, scenario, window, model, output):
    start_ns, end_ns = map(preflight.ns, window)
    split_evidence = _registered_screen_split_evidence(
        config, window,
        model_available_after_ns=(model.available_after_ns if strategy == "META_X1" and model is not None
                                  else None),
    )
    seed = 20261029
    episode = f"meta-study-{strategy}-{scenario}-{start_ns}-seed-{seed}"
    recovery = RecoveryTeacher()
    if strategy == "META_X1":
        if model is None:
            return None
        teacher = MetaGatedTeacher(model, recovery=recovery)
    elif strategy in config["teachers"]:
        teacher = RiskMatchedTeacher(build_teacher_1m(strategy), recovery)
    else:
        teacher = RiskMatchedTeacher(legacy._control(strategy), recovery)
    costs = config["cost_scenarios"][scenario]
    validate_native_execution(factory, scenario=scenario,
        expected_recipe=model.label_recipe[1] if model is not None else factory.route_a_execution_recipe_hash,
        fee_bps=costs["fee_bps_per_leg"], slippage_bps=costs["slippage_bps_per_leg"])
    for attr in ("snapshots", "teacher_snapshots", "fill_references"):
        getattr(factory, attr).clear()
    provider, grant = _provider(factory, teacher, source, episode)
    _prepare_context(provider, factory, start_ns, end_ns)
    hook = native._TeacherResearchHook1m(factory=factory, teacher=teacher, output=output,
        source_identity=source["marker_sha256"], scenario=scenario, exit_variant="X1",
        block_id=window[0][:10] + "_" + window[1][:10])
    funnel = MetaNativeFunnelAdapter(teacher=teacher, native_hook=hook) if strategy == "META_X1" else hook
    if strategy == "META_X1":
        barriers, report = run_registered_meta_rollout(
            provider=provider, teacher=teacher, factory=factory, host_rollout=application.rollout,
            grant=grant, verify_grant=lambda g: _verify_factory_grant(factory, g),
            scenario=scenario, seed=seed, rollout_id=episode, episode_steps=_steps(start_ns,end_ns),
            trace_path=output/"candidate-to-native.jsonl",
            reset_options={"rles_episode_index": seed}, teacher_funnel_hook=funnel)
    else:
        barriers, report = provider.rollout_with_m22_trace(
            factory=factory, host_rollout=application.rollout, grant=grant, seed=seed,
            decision_policy=None, rollout_id=episode, episode_steps=_steps(start_ns,end_ns),
            reset_options={"rles_episode_index": seed}, teacher_funnel_hook=funnel)
        if teacher.risk_diagnostics:
            raise RuntimeError("original teacher failed while a hard-risk action was active")
        diagnostics = None
    native_cases = funnel.cases if strategy == "META_X1" else tuple(hook.cases)
    if strategy == "META_X1":
        diagnostics = _score_diagnostics(teacher, factory, config, scenario, start_ns, end_ns, native_cases)
    verify = report.get("route_a_native_order_verification")
    if (not isinstance(verify, dict) or verify.get("barriers") != len(barriers)
            or len(barriers) != _steps(start_ns, end_ns)
            or verify.get("authority") != "retained_m22_session_snapshots"
            or report.get("liquidations_observed") is not True or report.get("liquidations") != 0):
        raise RuntimeError("native run did not pass complete M2.2 and liquidation checks")
    account = native_report(factory.snapshots, fill_references=factory.fill_references,
                            sample_interval_ns=MINUTE_NS, allow_price_improvement=False,
                            include_account_rows=True)
    if account["terminal_flat"] is not True or not account["account_rows"]:
        raise RuntimeError("native episode lacks settled account evidence")
    record = {"strategy_id": strategy, "scenario": scenario, "window": tuple(window),
              "registered_screen_split": {
                  **split_evidence,
                  "protocol_id": config["protocol_id"],
                  "preregistration_sha256": prereg_sha,
                  "source_marker_sha256": source["marker_sha256"],
                  "market_snapshot_sha256": source["snapshot_hash"],
                  "import_manifest_sha256": source["import_manifest_sha256"],
                  "strategy_id": strategy,
                  "scenario": scenario,
                  "seed": seed,
              },
              "cases": native_cases,
              # Start at the configured native account capital even if the
              # first sampled barrier already contains an entry and its costs.
              "equity_curve": (float(factory.capital), *(row["equity"] for row in account["account_rows"])),
              "native_order_verification": verify, "account_metrics": _account_costs_including_first_row(account),
              "candidate_summary": ({key: value for key, value in diagnostics.items() if key != "outcomes"}
                                    if diagnostics is not None else None)}
    serializable = {**record, "cases": [asdict(case) for case in native_cases],
                    "candidate_diagnostics": diagnostics,
                    "economic_status": "SCENARIO_ONLY"}
    (output/"native-result.json").write_text(json.dumps(serializable, sort_keys=True,
        indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return record


def execute_study(config, prereg_sha, *, source_root, output):
    source, manifest = _load_source(config, prereg_sha, source_root)
    output.mkdir(parents=True, exist_ok=False)
    scaler = _warmup_scaler(source, config, output)
    fit_start = preflight.ns(config["intervals"]["train_fit"][0])
    calibration_end = preflight.ns(config["intervals"]["train_calibration"][1])
    collection_start = fit_start - 122 * MINUTE_NS
    collection_factory = _factory(source, manifest, config, output, scaler,
        scenario="base", start_ns=collection_start, end_ns=calibration_end)
    # Collect independent opportunities against a real original-teacher
    # account, including positions and drawdown, rather than an always-flat
    # account whose constant features make live account states unsupported.
    recorder = NativeCandidateRecorder(RiskMatchedTeacher(build_teacher_1m("T01_EMA_9_21"), RecoveryTeacher()))
    if tuple(sorted(config["teachers"])) != recorder.collector.teacher_ids:
        raise ValueError("registered teacher set differs from native candidate collector")
    provider, collection_grant = _provider(collection_factory, recorder, source, "meta-fit-collection")
    _prepare_context(provider, collection_factory, collection_start, calibration_end)
    run_native_candidate_collection(provider=provider, recorder=recorder, factory=collection_factory,
        host_rollout=application.rollout, grant=collection_grant,
        verify_grant=lambda g: _verify_factory_grant(collection_factory, g), seed=20261029,
        rollout_id="meta-fit-collection", episode_steps=_steps(collection_start, calibration_end),
        reset_options={"rles_episode_index": 20261029})
    if recorder.teacher.risk_diagnostics:
        raise RuntimeError("native candidate-collection teacher had a risk-clock failure")
    policies, failures = {}, {}
    for scenario in config["cost_scenarios"]:
        factory = collection_factory if scenario == "base" else _factory(
            source, manifest, config, output, scaler, scenario=scenario,
            start_ns=collection_start, end_ns=calibration_end)
        _, training_grant = _provider(factory, legacy._control("FLAT"), source, "grant-"+scenario)
        verify = lambda g: (_verify_factory_grant(collection_factory, g)
                            if g == collection_grant else _verify_factory_grant(factory, g))
        try:
            result = train_and_save_from_native(prereg_path=PREREG, recorder=recorder,
                collection_factory=collection_factory, collection_grant=collection_grant,
                training_factory=factory, training_grant=training_grant, verify_grant=verify,
                scenario=scenario, output_dir=output/"training"/scenario)
            policies[scenario] = result.policy
        except MetaTrainingError as exc:
            failures[scenario] = str(exc)
            policies[scenario] = None
    strategies = (*config["teachers"], "META_X1", "FLAT", "FIXED_LONG_0.3", "FIXED_SHORT_0.3")
    rows = []
    for window_index, window in enumerate(config["intervals"]["native_screen"], 1):
        start_ns, end_ns = map(preflight.ns, window)
        for scenario in config["cost_scenarios"]:
            factory = _factory(source, manifest, config, output, scaler,
                scenario=scenario, start_ns=start_ns, end_ns=end_ns)
            for strategy in strategies:
                cell_output = output / f"window-{window_index}" / scenario / strategy
                if strategy == "META_X1" and policies[scenario] is None:
                    continue
                cell_output.mkdir(parents=True, exist_ok=False)
                row = _run_screen_cell(factory=factory, source=source, config=config,
                    prereg_sha=prereg_sha, strategy=strategy, scenario=scenario, window=window,
                    model=policies[scenario], output=cell_output)
                if row is not None:
                    rows.append(row)
    matrix = compare_preregistered_native_runs(config, rows)
    matrix.update({"protocol_id": config["protocol_id"], "preregistration_sha256": prereg_sha,
                   "source_marker_sha256": source["marker_sha256"],
                   "training_failures": failures,
                   "registered_screen_split_design": [
                       _registered_screen_split_evidence(config, window)
                       for window in config["intervals"]["native_screen"]
                   ],
                   "scope": "Train development; no independent OOS profitability claim"})
    (output/"comparison.json").write_text(json.dumps(matrix, sort_keys=True,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="run only after the registered source window is complete")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    preflight.main()
    if not args.execute:
        return 0
    config = json.loads(PREREG.read_bytes())
    if datetime.now(timezone.utc) < datetime.fromisoformat(config["source"]["end_utc_exclusive"].replace("Z", "+00:00")):
        raise RuntimeError("registered source and native screen have not finished; no market data was read")
    output = _inside_project(args.output)
    if output.exists():
        raise FileExistsError("study output already exists; preserve it rather than overwrite")
    result = execute_study(config, sha256(PREREG.read_bytes()).hexdigest(),
                           source_root=args.source_root, output=output)
    print(json.dumps({"status": result["status"], "available_cells": result["available_cells"],
                      "expected_cells": result["expected_cells"], "output": str(output),
                      "qualification": result["qualification"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
