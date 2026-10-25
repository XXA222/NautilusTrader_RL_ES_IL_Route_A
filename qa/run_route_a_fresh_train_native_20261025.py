"""Frozen fresh Train-inner native matrix; never a formal G2 receipt."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import gc
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

import run_route_a_train_student_screen_20261020 as harness  # noqa: E402
from run_route_a_beta_attribution_fit_20261025 import controls  # noqa: E402
from run_route_a_either_scale_exit_fit_20261025 import scenario_cycles  # noqa: E402
from rles_data.core import MINUTE_NS, parse_ns  # noqa: E402
from rles_orchestrator import application  # noqa: E402
from rles_route_a.teacher_research.candidate_events import CausalOpportunityGenerator  # noqa: E402
from rles_route_a.teachers.causal_position import CausalPositionTeacher  # noqa: E402
from rles_route_a.teachers.recovery import RecoveryTeacher  # noqa: E402
from rles_route_a.teachers.selector import FrozenTeacherSelector  # noqa: E402
from rles_route_a.teachers.wait import WaitTeacher  # noqa: E402

PREREG = ROOT / "configs/route_a_teacher_july_fresh_train_warmup_errata_20261025.json"
MARKET = ROOT / "data/routea_teacher_train_20260717_20260804/market.parquet"
OUT = ROOT / ".artifacts/route-a-july-fresh-train-native-20261025-run02"
REGISTRY = ROOT / ".artifacts/research-access.sqlite"


def _frozen_input() -> tuple[dict, str, int, int, str]:
    prereg_bytes = PREREG.read_bytes()
    prereg_sha = sha256(prereg_bytes).hexdigest()
    if prereg_sha != PREREG.with_suffix(".json.sha256").read_text().strip():
        raise ValueError("fresh Train preregistration checksum mismatch")
    prereg = json.loads(prereg_bytes)
    if prereg["status"] != "frozen_after_feature_warmup_failure_before_any_native_outcome":
        raise ValueError("unfrozen research roles")
    failed = ROOT / prereg["failure_receipt"]
    if (sha256(failed.read_bytes()).hexdigest() != prereg["failure_receipt_sha256"]
            or json.loads(failed.read_text())["completed_runs"] != 0):
        raise ValueError("warmup errata lacks a verified pre-outcome failure")
    marker_path = Path(str(MARKET) + ".COMPLETED.json")
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    if marker.get("schema_version") != "0.4.1":
        raise ValueError("fresh Train source needs row-group scoped 0.4.1 marker")
    marker_sha = sha256(marker_bytes).hexdigest()
    groups = marker["row_groups"]
    scope_start = min(min(g["min_ts_event_ns"], g["min_available_at_ns"]) for g in groups)
    scope_end = 1 + max(max(g["max_ts_event_ns"], g["max_available_at_ns"]) for g in groups)
    expected_start = parse_ns(prereg["scope"]["source_start_utc_inclusive"])
    expected_end = parse_ns(prereg["scope"]["end_utc_exclusive"])
    if scope_start > expected_start + MINUTE_NS or scope_end < expected_end:
        raise ValueError("fresh source does not cover frozen Train scope")
    with sqlite3.connect(REGISTRY.resolve().as_uri() + "?mode=ro", uri=True) as db:
        conflicting = db.execute(
            "SELECT purpose,start_ns,end_ns FROM access "
            "WHERE market=? AND family=? AND purpose IN ('test','holdout') "
            "AND start_ns<? AND end_ns>?",
            ("ETHUSDT", "perpetual-market-v1", expected_end, expected_start),
        ).fetchall()
        intents = db.execute(
            "SELECT purpose,start_ns,end_ns FROM query_intent "
            "WHERE market=? AND family=? AND purpose IN ('test','holdout') "
            "AND start_ns<? AND end_ns>?",
            ("ETHUSDT", "perpetual-market-v1", expected_end, expected_start),
        ).fetchall()
    if conflicting or intents:
        raise PermissionError("fresh Train scope overlaps a registered Test/Holdout or formal intent")
    return prereg, prereg_sha, scope_start, scope_end, marker_sha


def _arms() -> dict:
    def candidate_teacher(scale_policy: str) -> FrozenTeacherSelector:
        return FrozenTeacherSelector(
            primary=CausalPositionTeacher(
                generator=CausalOpportunityGenerator(
                    rule_version="route_a_candidates_v3_recovery",
                    require_recovery_confirmation=True,
                ),
                entry_cadence_minutes=5,
                exit_scale_policy=scale_policy,
            ),
            recovery=RecoveryTeacher(), wait=WaitTeacher(),
        )
    return {
        "flat": WaitTeacher(), **controls(),
        "p0_recovery_v3_both_exit": candidate_teacher("both"),
        "p0_recovery_v3_either_exit": candidate_teacher("either"),
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"immutable fresh Train native experiment exists: {OUT}")
    prereg, prereg_sha, scope_start, scope_end, marker_sha = _frozen_input()
    arms = _arms()
    if list(arms) != prereg["arms"]:
        raise ValueError("native arm order/set differs from frozen preregistration")
    recipe = {
        "prereg_sha256": prereg_sha, "source_marker_sha256": marker_sha,
        "read_scope": [scope_start, scope_end], "windows": prereg["windows"],
        "modes": prereg["execution"]["scenario_modes"],
        "arms": {name: asdict(teacher) for name, teacher in arms.items()},
        "seed": prereg["execution"]["seed"],
        "settlement_timestamp_tolerance_ns": prereg["execution"]["settlement_timestamp_tolerance_ns"],
        "purpose": "FRESH_TRAIN_INNER_DIRECT_RULE_DIAGNOSTIC_NOT_FORMAL_G1_OR_G2",
        "economic_status": "SCENARIO_ONLY", "test_holdout_read": False,
    }
    harness.write_verified(OUT / "recipe.json", recipe)
    harness.MARKET = MARKET
    harness.MARKER = Path(str(MARKET) + ".COMPLETED.json")
    harness.IMPORT = MARKET.parent / "import.json"
    harness.OUT_BASE = OUT
    harness.FEATURE_PROFILE = "causal_mtf_v7"
    harness.FOLD_ID = "route-a-july-fresh-train-native-v1"
    rows = []
    for block, (start, end) in enumerate(recipe["windows"], 1):
        start_ns, end_ns = parse_ns(start), parse_ns(end)
        steps = (end_ns - start_ns) // MINUTE_NS - 3
        for mode in recipe["modes"]:
            label = f"block-{block:02d}-{mode}"
            print(f"PREPARE {label}", flush=True)
            config, _ = harness.make_config(
                label=label, marker_sha=marker_sha,
                scope_start=scope_start, scope_end=scope_end,
                start_ns=start_ns, end_ns=end_ns, mode=mode,
                fit_scaler=True, source_mode="row_group_scoped_v1",
                prepare_factory=False,
            )
            config["environment"]["coverage_policy"]["settlement_timestamp_tolerance_ns"] = (
                recipe["settlement_timestamp_tolerance_ns"]
            )
            config["environment"]["coverage_policy"]["provenance"] = (
                "frozen source-specific second-level funding archive/API ingress tolerance; "
                "retain raw event times and 1ms publication delay"
            )
            try:
                factory = application.prepare(
                    config, fit_scaler=True, scaler=None,
                    compact_report_snapshots=False,
                    require_scoped_train_access=True,
                )
            except Exception as exc:
                harness.write_verified(OUT / "failure.json", {
                    "phase": "native_factory_prepare", "block": block, "scenario": mode,
                    "completed_runs": len(rows), "error_type": type(exc).__name__,
                    "error": str(exc), "qualification": "BLOCKED_NOT_A_PASS",
                })
                raise
            for arm, teacher in arms.items():
                name = f"{label}-{arm}"
                print(f"RUN {name} barriers={steps}", flush=True)
                started = time.perf_counter()
                provider, grant = harness.make_provider(factory, teacher, marker_sha, name)
                barriers, native = provider.rollout_with_m22_trace(
                    factory=factory, host_rollout=application.rollout,
                    grant=grant, seed=recipe["seed"], decision_policy=None,
                    rollout_id=name, episode_steps=steps,
                    reset_options={"rles_episode_index": recipe["seed"]},
                )
                metrics = harness.metric_row(barriers, dict(native), expected_source=marker_sha)
                evidence = metrics.pop("evidence")
                encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()
                (OUT / f"{name}-barriers.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
                metrics["summary"] = asdict(metrics["summary"])
                row = {
                    "block": block, "scenario": mode, "arm": arm,
                    "elapsed_seconds": time.perf_counter() - started,
                    "metrics": metrics,
                    "host_metrics": {key: native["metrics"].get(key) for key in (
                        "net_return", "max_drawdown", "total_fee", "total_funding",
                        "total_slippage_cost", "turnover_ratio")},
                    "m22": native["route_a_native_order_verification"],
                    "trace_sha256": sha256(encoded).hexdigest(),
                    "scenario_cycle_audit": scenario_cycles(evidence),
                    "reasons": dict(Counter(reason for barrier in barriers
                                            for reason in barrier.advice.reason_codes)),
                }
                if (len(barriers) != steps or metrics["liquidations"] != 0
                        or metrics["summary"]["economic_status"] != "SCENARIO_ONLY"):
                    raise RuntimeError("native scope/risk/economic verification failed")
                harness.write_verified(OUT / f"{name}.json", row)
                rows.append(row)
                print(json.dumps({"completed": name, "net_return": metrics["net_return"],
                                  "cycles": row["scenario_cycle_audit"]}), flush=True)
                del barriers, native, provider, metrics, evidence, encoded
                gc.collect()
            del factory
            gc.collect()
    harness.write_verified(OUT / "report.json", {
        "qualification": "NOT_EVALUATED_TRAIN_INNER_SCENARIO_ONLY",
        "formal_student_promotion": False, "completed_runs": len(rows), "runs": rows,
    })
    print(f"COMPLETED_NATIVE_RUNS={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
