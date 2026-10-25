"""Matched continuous native Train-fit diagnostics; no qualification/promotion.

Four fixed 4.5-day blocks, Base/Stress, FLAT/v2/v3 recovery. No parameter search.
All arms start flat with the same source, host recipe and risk budget. Retain
barrier evidence and terminal accounting rather than independent trade sums.
"""
from __future__ import annotations

from collections import Counter
import argparse
from dataclasses import asdict
import gc
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))
import run_route_a_train_student_screen_20261020 as harness  # noqa: E402
from rles_data.core import MINUTE_NS, parse_ns  # noqa: E402
from rles_orchestrator import application  # noqa: E402
from rles_orchestrator.research_access import read_registered_train_window  # noqa: E402
from rles_route_a.teacher_research.candidate_events import CausalOpportunityGenerator  # noqa: E402
from rles_route_a.teachers.causal_position import CausalPositionTeacher  # noqa: E402
from rles_route_a.teachers.recovery import RecoveryTeacher  # noqa: E402
from rles_route_a.teachers.selector import FrozenTeacherSelector  # noqa: E402
from rles_route_a.teachers.wait import WaitTeacher  # noqa: E402

OUT = ROOT / ".artifacts/route-a-causal-teacher-fit-native-20261025"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-cadence-minutes", type=int, choices=(1, 5), default=1)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--archive-clock", action="store_true")
    args = parser.parse_args()
    output = (ROOT / ".artifacts/route-a-causal-teacher-fit-native-5m-20261025"
              if args.entry_cadence_minutes == 5 else OUT)
    if args.audit_only:
        output = ROOT / ".artifacts/route-a-causal-teacher-fit-native-dq-20261025-run02"
    if args.archive_clock:
        output = output.with_name(output.name + "-archive-clock")
    return run(output, entry_cadence_minutes=args.entry_cadence_minutes,
               audit_only=args.audit_only, archive_clock=args.archive_clock)


def run(OUT, *, entry_cadence_minutes, audit_only=False, archive_clock=False,
        arms_override=None, extra_recipe=None):
    if OUT.exists():
        raise FileExistsError(f"immutable native experiment already exists: {OUT}")
    market = ROOT / "data/routea_teacher_train_20260812_20261002/market.parquet"
    marker_path = Path(str(market) + ".COMPLETED.json")
    marker_bytes = marker_path.read_bytes()
    marker_sha = sha256(marker_bytes).hexdigest()
    groups = json.loads(marker_bytes)["row_groups"][:6]
    scope_start = min(min(g["min_ts_event_ns"], g["min_available_at_ns"]) for g in groups)
    scope_end = 1 + max(max(g["max_ts_event_ns"], g["max_available_at_ns"]) for g in groups)
    prereg_path = ROOT / "configs/route_a_teacher_e1_e2_g1_train_prereg_20261025.json"
    prereg_bytes = prereg_path.read_bytes()
    if sha256(prereg_bytes).hexdigest() != prereg_path.with_suffix(".json.sha256").read_text().strip():
        raise RuntimeError("original Train blocks checksum mismatch")
    prereg = json.loads(prereg_bytes)
    if scope_end >= parse_ns(prereg["windows"]["train_inner_screen"][0]):
        raise RuntimeError("native Train-fit probe scope reaches screen")
    def selector(generator):
        return FrozenTeacherSelector(
            primary=CausalPositionTeacher(
                generator=generator, entry_cadence_minutes=entry_cadence_minutes,
            ),
            recovery=RecoveryTeacher(), wait=WaitTeacher(),
        )
    arms = {
        "flat": WaitTeacher(),
        "p0_v2": selector(CausalOpportunityGenerator()),
        "p0_recovery_v3": selector(CausalOpportunityGenerator(
            rule_version="route_a_candidates_v3_recovery", require_recovery_confirmation=True,
        )),
    }
    if arms_override is not None:
        arms = arms_override
    recipe = {
        "source_marker_sha256": marker_sha, "read_scope": [scope_start, scope_end],
        "windows": prereg["windows"]["train_fit_blocks"], "modes": ["base"] if audit_only else ["base", "stress"],
        "audit_only": audit_only,
        "arms": {key: asdict(value) for key, value in arms.items()},
        "seed": 20261025, "episode_terminal_margin_minutes": 3,
        "entry_cadence_minutes": entry_cadence_minutes,
        "exit_cadence_minutes": 1,
        "settlement_timestamp_tolerance_ns": 1_000_000_000 if archive_clock else 15_000_000,
        "clock_policy": "source-specific 1s tolerance matches existing funding API ingress convention; no timestamp rounding or event removal" if archive_clock else "legacy 15ms source assumption",
        "purpose": "TRAIN_FIT_DIAGNOSTIC_NOT_UNSEEN_SCREEN_OR_ALPHA_QUALIFICATION",
        "economic_status": "SCENARIO_ONLY", "screen_test_holdout_read": False,
        "scaler": "host fits on same Train block; rule teachers use unscaled causal market features",
    }
    if extra_recipe is not None:
        if set(extra_recipe) & set(recipe):
            raise ValueError("supplementary recipe cannot override frozen native assumptions")
        recipe.update(extra_recipe)
    harness.write_verified(OUT / "recipe.json", recipe)
    harness.MARKET, harness.MARKER = market, marker_path
    harness.IMPORT = market.parent / "import.json"
    harness.OUT_BASE = OUT
    harness.FEATURE_PROFILE = "causal_mtf_v7"
    harness.FOLD_ID = "route-a-causal-teacher-fit-native-v1"
    rows = []
    for number, (start, end) in enumerate(recipe["windows"], 1):
        start_ns, end_ns = parse_ns(start), parse_ns(end)
        steps = (end_ns - start_ns) // MINUTE_NS - 3
        for mode in recipe["modes"]:
            label = f"block-{number:02d}-{mode}"
            print(f"PREPARE {label}", flush=True)
            config, factory = harness.make_config(
                label=label, marker_sha=marker_sha, scope_start=scope_start, scope_end=scope_end,
                start_ns=start_ns, end_ns=end_ns, mode=mode, fit_scaler=True,
                source_mode="row_group_scoped_v1",
                prepare_factory=False,
            )
            if archive_clock:
                coverage = config["environment"]["coverage_policy"]
                coverage["settlement_timestamp_tolerance_ns"] = 1_000_000_000
                coverage["provenance"] = recipe["clock_policy"]
            if audit_only:
                quality = application.audit_market(config)
                if number == 4:
                    store = read_registered_train_window(
                        config, {"start_ns": scope_start, "end_ns": scope_end},
                        marker_sha256=marker_sha, label="train-route-a-fit-funding-clock-audit",
                    )
                    expected = harness.settlement_times(start_ns, end_ns)
                    actual = sorted(e.ts_event_ns for e in store.events
                                    if e.kind == "funding_settlement"
                                    and e.payload.get("instrument") == harness.INSTRUMENT
                                    and start_ns <= e.ts_event_ns < end_ns)
                    quality["funding_clock_audit"] = {
                        "expected": expected, "actual": actual,
                        "offset_ns": [a - e for a, e in zip(actual, expected)],
                    }
                harness.write_verified(OUT / f"{label}-quality.json", quality)
                print(json.dumps(quality), flush=True)
                continue
            try:
                factory = application.prepare(
                    config, fit_scaler=True, scaler=None, compact_report_snapshots=False,
                    require_scoped_train_access=True,
                )
            except Exception as exc:
                harness.write_verified(OUT / "failure.json", {
                    "stage": "native_factory_prepare", "block": number, "scenario": mode,
                    "completed_runs": len(rows), "error_type": type(exc).__name__,
                    "error": str(exc), "qualification": "BLOCKED_NOT_A_PASS",
                })
                raise
            for arm, teacher in arms.items():
                identity = f"{label}-{arm}"
                print(f"RUN {identity} barriers={steps}", flush=True)
                started = time.perf_counter()
                provider, grant = harness.make_provider(factory, teacher, marker_sha, identity)
                barriers, native = provider.rollout_with_m22_trace(
                    factory=factory, host_rollout=application.rollout, grant=grant,
                    seed=recipe["seed"], decision_policy=None, rollout_id=identity,
                    episode_steps=steps, reset_options={"rles_episode_index": recipe["seed"]},
                )
                metrics = harness.metric_row(barriers, dict(native), expected_source=marker_sha)
                trace_bytes = json.dumps(metrics.pop("evidence"), sort_keys=True,
                                        separators=(",", ":"), allow_nan=False).encode()
                trace_path = OUT / f"{identity}-barriers.json.gz"
                trace_path.write_bytes(gzip.compress(trace_bytes, mtime=0))
                metrics["summary"] = asdict(metrics["summary"])
                row = {
                    "block": number, "scenario": mode, "arm": arm,
                    "elapsed_seconds": time.perf_counter() - started,
                    "metrics": metrics,
                    "host_metrics": {key: native["metrics"].get(key) for key in (
                        "net_return", "max_drawdown", "total_fee", "total_funding",
                        "total_slippage_cost", "turnover_ratio")},
                    "m22": native["route_a_native_order_verification"],
                    "trace_sha256": sha256(trace_bytes).hexdigest(),
                    "max_position_age_minutes": max(
                        b.observation.account_value("position_duration_ns") / MINUTE_NS for b in barriers),
                    "reasons": dict(Counter(reason for b in barriers for reason in b.advice.reason_codes)),
                }
                if len(barriers) != steps or metrics["liquidations"] != 0:
                    raise RuntimeError("native coverage/risk verification failed")
                harness.write_verified(OUT / f"{identity}.json", row)
                rows.append(row)
                print(json.dumps({"completed": identity, "net_return": metrics["net_return"],
                                  "filled_orders": metrics["filled_orders"],
                                  "seconds": row["elapsed_seconds"]}), flush=True)
                del barriers, native, provider, metrics, trace_bytes
                gc.collect()
            del factory
            gc.collect()
    report = {"qualification": "NOT_EVALUATED_TRAIN_FIT_ONLY_SCENARIO_ONLY",
              "runs": rows, "completed_runs": len(rows),
              "formal_student_promotion": False}
    harness.write_verified(OUT / "report.json", report)
    print(f"COMPLETED_NATIVE_RUNS={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
