"""Run the frozen Route A core-decision Train study on Ubuntu only.

This is a direct native comparison of the frozen entry/exit mechanisms. It is
not a G1/G2 receipt and never reads formal Test/Holdout data.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import argparse
import gc
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

import run_route_a_train_student_screen_20261020 as harness  # noqa: E402
from run_route_a_beta_attribution_fit_20261025 import FixedSideTeacher  # noqa: E402
from rles_data.core import MINUTE_NS, parse_ns  # noqa: E402
from rles_orchestrator import application  # noqa: E402
from rles_orchestrator.research_access import reject_protected_monthly_archive_source  # noqa: E402
from rles_route_a.teacher_research.candidate_events import CausalOpportunityGenerator  # noqa: E402
from rles_route_a.teachers.causal_position import CausalPositionTeacher  # noqa: E402
from rles_route_a.teachers.recovery import RecoveryTeacher  # noqa: E402
from rles_route_a.teachers.selector import FrozenTeacherSelector  # noqa: E402
from rles_route_a.teachers.wait import WaitTeacher  # noqa: E402

PREREG = ROOT / "configs/route_a_teacher_core_train_20230601_20231210_prereg.json"
ERRATA = ROOT / "configs/route_a_teacher_core_train_20230601_20231210_directional_errata.json"
MARKET = ROOT / "data/routea_teacher_train_20230529_20231209/market.parquet"
OUT = ROOT / ".artifacts/route-a-teacher-core-train-20230601-20231210-directional-v2"
REGISTRY = ROOT / ".artifacts/research-access.sqlite"
RULES = ROOT / ".artifacts/audit-development-20261016/data/scenario-rules.json"
EXPECTED_ROOT_MARKER = r"D:\Program Files\NautilusTrader_RL_ES_IL_Route_A"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _preflight() -> tuple[dict, dict, str, str, int, int, str]:
    if ROOT.resolve() != Path("/mnt/d/Program Files/NautilusTrader_RL_ES_IL_Route_A"):
        raise RuntimeError("canonical WSL project root mismatch; no run permitted")
    if (ROOT / ".canonical-project-root").read_text(encoding="utf-8").strip() != EXPECTED_ROOT_MARKER:
        raise RuntimeError("canonical project-root marker mismatch; no run permitted")
    prereg_bytes = PREREG.read_bytes()
    prereg_sha = sha256(prereg_bytes).hexdigest()
    if PREREG.with_suffix(PREREG.suffix + ".sha256").read_text(encoding="ascii").strip() != prereg_sha:
        raise RuntimeError("frozen preregistration checksum mismatch")
    prereg = json.loads(prereg_bytes)
    if prereg.get("status") != "frozen_before_source_read":
        raise RuntimeError("study recipe was not frozen before source ingestion")
    errata_bytes = ERRATA.read_bytes()
    errata_sha = sha256(errata_bytes).hexdigest()
    if ERRATA.with_suffix(ERRATA.suffix + ".sha256").read_text(encoding="ascii").strip() != errata_sha:
        raise RuntimeError("pre-outcome directional errata checksum mismatch")
    errata = json.loads(errata_bytes)
    if (errata.get("status") != "frozen_after_dq_c_preparation_failure_before_any_native_outcome"
            or errata.get("supersedes_prereg_sha256") != prereg_sha
            or errata.get("quality_purpose") != "directional"
            or errata.get("rl_training_allowed") is not False):
        raise RuntimeError("directional errata is invalid or changes the frozen teacher recipe")
    failure_path = ROOT / errata["failure_receipt"]
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if (_sha(failure_path) != errata["failure_receipt_sha256"]
            or failure.get("completed_runs") != 0
            or failure.get("error") != "DQ-C permits directional-only, low-leverage research"):
        raise RuntimeError("directional errata lacks a matching pre-outcome DQ-C receipt")
    marker_path = Path(str(MARKET) + ".COMPLETED.json")
    marker_sha = _sha(marker_path)
    if marker_path.with_suffix(marker_path.suffix + ".sha256").read_text(encoding="ascii").strip() != marker_sha:
        raise RuntimeError("curated source marker checksum mismatch")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    imported = json.loads((MARKET.parent / "import.json").read_text(encoding="utf-8"))
    if marker.get("schema_version") != "0.4.1" or imported.get("schema") != "rles.binance-archive-range.v1":
        raise RuntimeError("study requires a row-group scoped Binance archive source")
    if (imported.get("symbol") != "ETHUSDT"
            or imported.get("start_day") != "2023-05-29"
            or imported.get("end_day") != "2023-12-09"
            or imported.get("instrument_rules_supplied") is not False):
        raise RuntimeError("curated source differs from the frozen Train study")
    groups = marker.get("row_groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("curated source lacks row-group bounds")
    scope_start = min(min(row["min_ts_event_ns"], row["min_available_at_ns"]) for row in groups)
    scope_end = 1 + max(max(row["max_ts_event_ns"], row["max_available_at_ns"]) for row in groups)
    requested_start = parse_ns(prereg["scope"]["source_start_utc_inclusive"])
    requested_end = parse_ns(prereg["scope"]["end_utc_exclusive"])
    if scope_start > requested_start + MINUTE_NS or scope_end < requested_end:
        raise RuntimeError("curated row groups do not cover the frozen Train source scope")
    if imported.get("event_count") != marker.get("events"):
        raise RuntimeError("import manifest and curated marker event counts differ")
    reject_protected_monthly_archive_source(MARKET, "ETHUSDT")
    if not RULES.is_file():
        raise RuntimeError("scenario instrument rules are unavailable")
    with sqlite3.connect(REGISTRY) as db:
        rows = db.execute(
            "SELECT owner,start_ns,end_ns,purpose,identity,state FROM access "
            "WHERE market=? AND family=? AND purpose='development'",
            ("ETHUSDT", "perpetual-market-v1"),
        ).fetchall()
        reserved = False
        for owner, start, end, purpose, encoded, state in rows:
            identity = json.loads(encoded)
            if (owner == prereg["protocol_id"]
                    and identity.get("protocol_sha256") == prereg_sha
                    and start == requested_start and end == requested_end
                    and purpose == "development" and state == "recorded"):
                reserved = True
                break
        protected = db.execute(
            "SELECT purpose,start_ns,end_ns FROM access "
            "WHERE market=? AND family=? AND purpose IN ('test','holdout') "
            "AND start_ns<? AND end_ns>?",
            ("ETHUSDT", "perpetual-market-v1", requested_end, requested_start),
        ).fetchall()
        intents = db.execute(
            "SELECT purpose,start_ns,end_ns FROM query_intent "
            "WHERE market=? AND family=? AND purpose IN ('test','holdout') "
            "AND start_ns<? AND end_ns>?",
            ("ETHUSDT", "perpetual-market-v1", requested_end, requested_start),
        ).fetchall()
    if not reserved:
        raise RuntimeError("frozen Train scope is not registered before the run")
    if protected or intents:
        raise RuntimeError("Train scope overlaps a protected interval or formal query intent")
    if OUT.exists():
        raise FileExistsError(f"immutable study output already exists: {OUT}")
    return prereg, errata, prereg_sha, errata_sha, scope_start, scope_end, marker_sha


def _candidate(rule: str, *, recovery: bool, exit_policy: str,
               prereg: dict) -> FrozenTeacherSelector:
    recipe = prereg["teacher"]
    generator = CausalOpportunityGenerator(
        rule_version=rule,
        require_recovery_confirmation=recovery,
        breakout_threshold=recipe["v2_breakout_threshold"],
        pullback_z_threshold=recipe["pullback_z_threshold"],
        minimum_efficiency=recipe["minimum_efficiency"],
    )
    primary = CausalPositionTeacher(
        generator=generator,
        entry_cadence_minutes=recipe["entry_cadence_minutes"],
        size=recipe["entry_size"],
        trend_invalidation_threshold=recipe["trend_invalidation_threshold"],
        exit_scale_policy=exit_policy,
    )
    return FrozenTeacherSelector(primary=primary, recovery=RecoveryTeacher(), wait=WaitTeacher())


def _arms(prereg: dict) -> dict:
    factories = {
        "flat": lambda: WaitTeacher(),
        "fixed_long_size_0.3": lambda: FrozenTeacherSelector(
            primary=FixedSideTeacher(side=1, size=0.3, expert_id="fixed_long_v1"),
            recovery=RecoveryTeacher(), wait=WaitTeacher(),
        ),
        "fixed_short_size_0.3": lambda: FrozenTeacherSelector(
            primary=FixedSideTeacher(side=-1, size=0.3, expert_id="fixed_short_v1"),
            recovery=RecoveryTeacher(), wait=WaitTeacher(),
        ),
        "route_a_candidates_v2_both_exit": lambda: _candidate(
            "route_a_candidates_v2", recovery=False, exit_policy="both", prereg=prereg,
        ),
        "route_a_candidates_v3_recovery_both_exit": lambda: _candidate(
            "route_a_candidates_v3_recovery", recovery=True, exit_policy="both", prereg=prereg,
        ),
        "route_a_candidates_v3_recovery_either_exit": lambda: _candidate(
            "route_a_candidates_v3_recovery", recovery=True, exit_policy="either", prereg=prereg,
        ),
    }
    names = prereg["arms"]
    if len(set(names)) != len(names) or any(name not in factories for name in names):
        raise RuntimeError("frozen arm list is invalid or unsupported")
    return {name: factories[name]() for name in names}


def _run(prereg: dict, errata: dict, prereg_sha: str, errata_sha: str,
         scope_start: int, scope_end: int, marker_sha: str) -> int:
    arms = _arms(prereg)
    windows = prereg["roles"]["train_inner_screen_blocks"]
    recipe = {
        "protocol_id": prereg["protocol_id"],
        "prereg_sha256": prereg_sha,
        "pre_outcome_errata_sha256": errata_sha,
        "source_marker_sha256": marker_sha,
        "read_scope": [scope_start, scope_end],
        "screen_blocks": windows,
        "scenario_modes": prereg["execution"]["scenario_modes"],
        "arms": {name: asdict(teacher) for name, teacher in arms.items()},
        "base_costs_bps_per_leg": prereg["costs_bps_per_leg"]["base"],
        "stress_costs_bps_per_leg": prereg["costs_bps_per_leg"]["stress"],
        "seed": prereg["execution"]["seed"],
        "economic_status": "SCENARIO_ONLY",
        "quality_purpose": errata["quality_purpose"],
        "qualification": "TRAIN_INNER_DIRECT_RULE_RESEARCH_NOT_G1_OR_G2",
        "student_promotion": False,
    }
    harness.write_verified(OUT / "recipe.json", recipe)
    harness.MARKET = MARKET
    harness.MARKER = Path(str(MARKET) + ".COMPLETED.json")
    harness.IMPORT = MARKET.parent / "import.json"
    harness.RULES = RULES
    harness.OUT_BASE = OUT
    harness.FEATURE_PROFILE = "causal_mtf_v7"
    harness.FOLD_ID = prereg["protocol_id"]
    rows = []
    try:
        for block, (start, end) in enumerate(windows, 1):
            start_ns, end_ns = parse_ns(start), parse_ns(end)
            steps = (end_ns - start_ns) // MINUTE_NS - 3
            for mode in recipe["scenario_modes"]:
                label = f"block-{block:02d}-{mode}"
                print(f"PREPARE {label}", flush=True)
                config, _ = harness.make_config(
                    label=label, marker_sha=marker_sha,
                    scope_start=scope_start, scope_end=scope_end,
                    start_ns=start_ns, end_ns=end_ns, mode=mode,
                    fit_scaler=True, source_mode="row_group_scoped_v1",
                    prepare_factory=False,
                )
                config["environment"]["quality_purpose"] = errata["quality_purpose"]
                factory = application.prepare(
                    config, fit_scaler=True, scaler=None,
                    compact_report_snapshots=False,
                    require_scoped_train_access=True,
                )
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
                            "total_slippage_cost", "turnover_ratio",
                        )},
                        "m22": native["route_a_native_order_verification"],
                        "trace_sha256": sha256(encoded).hexdigest(),
                        "reasons": dict(Counter(reason for barrier in barriers
                                                for reason in barrier.advice.reason_codes)),
                    }
                    if (len(barriers) != steps or metrics["liquidations"] != 0
                            or metrics["summary"]["economic_status"] != "SCENARIO_ONLY"):
                        raise RuntimeError("native scope/risk/economic verification failed")
                    harness.write_verified(OUT / f"{name}.json", row)
                    rows.append(row)
                    print(json.dumps({"completed": name,
                                      "net_return": metrics["net_return"],
                                      "trades": metrics["summary"].get("completed_trades")},
                                     allow_nan=False), flush=True)
                    del barriers, native, provider, metrics, evidence, encoded
                    gc.collect()
                del factory
                gc.collect()
    except Exception as exc:
        harness.write_verified(OUT / "failure.json", {
            "phase": "native_direct_rule_research",
            "completed_runs": len(rows),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "qualification": "FAILED_OR_BLOCKED_NOT_A_PASS",
        })
        raise

    by_key = {(row["block"], row["scenario"], row["arm"]): row for row in rows}
    paired = []
    candidate_arms = [name for name in arms if name.startswith("route_a_candidates_")]
    for block in range(1, len(windows) + 1):
        for mode in recipe["scenario_modes"]:
            long_return = by_key[(block, mode, "fixed_long_size_0.3")]["metrics"]["net_return"]
            short_return = by_key[(block, mode, "fixed_short_size_0.3")]["metrics"]["net_return"]
            flat_return = by_key[(block, mode, "flat")]["metrics"]["net_return"]
            for arm in candidate_arms:
                value = by_key[(block, mode, arm)]["metrics"]["net_return"]
                paired.append({
                    "block": block, "scenario": mode, "arm": arm,
                    "net_return": value,
                    "excess_vs_flat": value - flat_return,
                    "excess_vs_fixed_long": value - long_return,
                    "excess_vs_fixed_short": value - short_return,
                    "exceeds_both_fixed_sides": value > max(long_return, short_return),
                })
    report = {
        "qualification": "TRAIN_INNER_DIRECT_RULE_RESEARCH_NOT_G1_OR_G2",
        "economic_status": "SCENARIO_ONLY",
        "formal_student_promotion": False,
        "completed_runs": len(rows),
        "expected_runs": len(windows) * len(recipe["scenario_modes"]) * len(arms),
        "runs": rows,
        "paired_candidate_comparisons": paired,
    }
    harness.write_verified(OUT / "report.json", report)
    print(f"COMPLETED_NATIVE_RUNS={len(rows)}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate preregistration/source/ledger metadata without reading Parquet values")
    args = parser.parse_args()
    prereg, errata, prereg_sha, errata_sha, scope_start, scope_end, marker_sha = _preflight()
    if args.dry_run:
        arms = _arms(prereg)
        print(json.dumps({
            "status": "PREFLIGHT_PASS_NO_MARKET_PAYLOAD_READ",
            "protocol_id": prereg["protocol_id"],
            "prereg_sha256": prereg_sha,
            "pre_outcome_errata_sha256": errata_sha,
            "quality_purpose": errata["quality_purpose"],
            "source_marker_sha256": marker_sha,
            "source_scope_ns": [scope_start, scope_end],
            "screen_blocks": prereg["roles"]["train_inner_screen_blocks"],
            "arms": {name: asdict(teacher) for name, teacher in arms.items()},
            "scenario_modes": prereg["execution"]["scenario_modes"],
        }, indent=2), flush=True)
        return 0
    return _run(prereg, errata, prereg_sha, errata_sha, scope_start, scope_end, marker_sha)


if __name__ == "__main__":
    raise SystemExit(main())
