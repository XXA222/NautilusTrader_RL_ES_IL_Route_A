"""Run a registered Train-only, scenario-cost Route A teacher/BC screen.

The source is the immutable 2023-12-30..2024-04-15 curated shard. It is an
old whole-snapshot format, so the reader reserves and verifies that entire
physically isolated shard before decoding it. All scored/fitted decisions are
inside Train; the pre-2024-01-01 rows are feature warm-up only. No evaluation
fold is configured or read. Costs and instrument rules remain scenario-only.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any
import torch

from rles_orchestrator import application
from rles_orchestrator.research_access import ACTIVE_OWNER, read_registered_train_window
from rles_route_a.config import BCConfig
from rles_route_a.evaluation import (
    NativePair,
    StudentPair,
    bc_action_schema_sha256,
    bc_actor_sha256,
    bc_observation_schema_sha256,
    bc_student_recipe_sha256,
    screen_student,
    screen_teacher,
    verify_student_screen,
)
from rles_route_a.integration.gym_rollout import NativeGymTeacherProvider
from rles_route_a.integration.host_gym_projection import (
    TerminalL6Receipt,
    terminal_settlement_sha256,
)
from rles_route_a.integration.wfo_train import make_route_a_wfo_train
from rles_route_a.teachers.trend import TrendTeacher
from rles_route_a.teachers.wait import WaitTeacher
from rles_route_a.training.native_bc import (
    _native_barrier_evidence,
    evaluate_bc_on_registered_train_inner_rollout,
    teacher_config_hash,
    train_bc_from_registered_native_rollout,
)
from rles_route_a.integration.trace_audit import audit_teacher_barriers
from route_a_training_runtime import configure_torch_threads


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "binance_ethusdt_20231230_20240415"
MARKET = DATA_DIR / "market.parquet"
MARKER = DATA_DIR / "market.parquet.COMPLETED.json"
IMPORT = DATA_DIR / "import.json"
RULES = ROOT / ".artifacts" / "audit-development-20261016" / "data" / "scenario-rules.json"
CONFIG = ROOT / "configs" / "ethusdt_20240101_real.json"
OUT_BASE = ROOT / ".artifacts" / "route-a-train-student-screen-20261020-run13"
FEATURE_PROFILE = os.environ.get("ROUTE_A_FEATURE_PROFILE", "causal_mtf_v5")
if FEATURE_PROFILE not in {"causal_mtf_v5", "causal_mtf_v6"}:
    raise ValueError("Route A student runner supports causal_mtf_v5 or causal_mtf_v6")
INSTRUMENT = "ETHUSDT-PERP.SIM"
FOLD_ID = "route-a-train-inner-2024q1-scenario-screen"
MINUTE_NS = 60_000_000_000
FUNDING_PERIOD_NS = 8 * 60 * MINUTE_NS
WINDOWS = (
    ("april01", "2024-04-01T00:00:00+00:00", "2024-04-01T12:00:00+00:00"),
    ("april08", "2024-04-08T00:00:00+00:00", "2024-04-08T12:00:00+00:00"),
)
SEEDS = (11, 29)
FIT = ("2024-01-02T00:00:00+00:00", "2024-01-09T00:00:00+00:00")


def ns(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("experiment timestamps must carry a timezone")
    delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def settlement_times(start_ns: int, end_ns: int) -> list[int]:
    first = ((start_ns + FUNDING_PERIOD_NS - 1) // FUNDING_PERIOD_NS) * FUNDING_PERIOD_NS
    return list(range(first, end_ns, FUNDING_PERIOD_NS))


def write_verified(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    path.write_bytes(encoded)
    digest = sha256(encoded).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest, encoding="ascii")
    return digest


def canonical_digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def load_source_identity() -> tuple[str, int, int]:
    marker_payload = json.loads(MARKER.read_text(encoding="utf-8"))
    import_payload = json.loads(IMPORT.read_text(encoding="utf-8"))
    marker_sha = sha256(MARKER.read_bytes()).hexdigest()
    if (
        not isinstance(marker_payload.get("file_sha256"), str)
        or len(marker_payload["file_sha256"]) != 64
        or type(marker_payload.get("events")) is not int
        or marker_payload["events"] <= 0
    ):
        raise RuntimeError("curated Train shard commit metadata is malformed")
    if marker_payload["schema_version"] not in {"0.4.0", "0.4.1"}:
        raise RuntimeError("runner expects an audited curated snapshot format")
    if import_payload["start_ns"] >= import_payload["end_ns"] or import_payload["end_ns"] > ns(
        "2024-07-16T00:00:00+00:00"
    ):
        raise RuntimeError("curated shard exceeds the audited Train-only boundary")
    if marker_payload["schema_version"] == "0.4.1":
        row_groups = marker_payload.get("row_groups")
        if not isinstance(row_groups, list) or not row_groups:
            raise RuntimeError("curated row-group bounds are missing from the commit marker")
        scope_start = min(
            min(row["min_ts_event_ns"], row["min_available_at_ns"])
            for row in row_groups
        )
        scope_end = 1 + max(
            max(row["max_ts_event_ns"], row["max_available_at_ns"])
            for row in row_groups
        )
        if scope_end > int(import_payload["end_ns"]):
            raise RuntimeError("curated row-group bounds exceed the import manifest")
        return marker_sha, scope_start, scope_end
    return marker_sha, int(import_payload["start_ns"]), int(import_payload["end_ns"])


def make_config(
    *,
    label: str,
    marker_sha: str,
    scope_start: int,
    scope_end: int,
    start_ns: int,
    end_ns: int,
    mode: str,
    fit_scaler: bool,
    scaler: Any = None,
    source_mode: str = "whole_snapshot_isolated_v1",
) -> tuple[dict, Any]:
    if source_mode not in {"row_group_scoped_v1", "whole_snapshot_isolated_v1"}:
        raise ValueError("unsupported registered Train source mode")
    template = json.loads(CONFIG.read_text(encoding="utf-8"))["environment"]
    environment = dict(template)
    environment.update(
        {
            "market_path": str(MARKET),
            "rules_path": str(RULES),
            "instrument": INSTRUMENT,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "mode": mode,
            "feature_profile": FEATURE_PROFILE,
            "execution": {**template["execution"], "zero_target_policy": "close"},
            "coverage_policy": {
                "maximum_age_ns": template["coverage_policy"]["maximum_age_ns"],
                "settlement_times_ns": {
                    INSTRUMENT: settlement_times(start_ns, end_ns),
                },
                "provenance": (
                    "8-hour schedule compared with every registered Train settlement event; "
                    "observed source timestamp offset is bounded at 15 ms; the import declares "
                    "1 ms availability delay; not evidence of historical account rules"
                ),
                "settlement_max_delay_ns": 1_000_000,
                "settlement_timestamp_tolerance_ns": 15_000_000,
            },
        }
    )
    config = {
        "output": str(OUT_BASE / "reservations" / label),
        "environment": environment,
        "train_data_access": {
            "start_ns": scope_start,
            "end_ns": scope_end,
            "marker_sha256": marker_sha,
            "label": "train-route-a-screen-" + label,
            "source_mode": source_mode,
        },
    }
    factory = application.prepare(
        config,
        fit_scaler=fit_scaler,
        scaler=scaler,
        compact_report_snapshots=False,
        require_scoped_train_access=True,
    )
    return config, factory


def audit_train_quality() -> int:
    """Use the same registered Train reader to explain a factory DQ rejection."""
    marker_sha, scope_start, scope_end = load_source_identity()
    template = json.loads(CONFIG.read_text(encoding="utf-8"))["environment"]
    start_ns, end_ns = map(ns, FIT)
    environment = {
        "market_path": str(MARKET),
        "rules_path": str(RULES),
        "instrument": INSTRUMENT,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "coverage_policy": {
            "maximum_age_ns": template["coverage_policy"]["maximum_age_ns"],
            "settlement_times_ns": {INSTRUMENT: settlement_times(start_ns, end_ns)},
            "provenance": (
                "8-hour schedule compared with every registered Train settlement event; "
                "observed source timestamp offset is bounded at 15 ms; the import declares "
                "1 ms availability delay; not evidence of historical account rules"
            ),
            "settlement_max_delay_ns": 1_000_000,
            "settlement_timestamp_tolerance_ns": 15_000_000,
        },
    }
    config = {
        "output": str(OUT_BASE / "reservations" / "dq-audit"),
        "environment": environment,
        "train_data_access": {
            "start_ns": scope_start,
            "end_ns": scope_end,
            "marker_sha256": marker_sha,
            "label": "train-route-a-dq-audit-20261020",
            "source_mode": "whole_snapshot_isolated_v1",
        },
    }
    owner_token = ACTIVE_OWNER.set("route-a-train-screen-20261020")
    try:
        report = application.audit_market(config)
        store = read_registered_train_window(
            config,
            {"start_ns": scope_start, "end_ns": scope_end},
            marker_sha256=marker_sha,
            label=config["train_data_access"]["label"],
            allow_legacy_isolated_snapshot=True,
        )
    finally:
        ACTIVE_OWNER.reset(owner_token)
    expected_settlements = settlement_times(start_ns, end_ns)
    actual_settlements = sorted(
        event.ts_event_ns
        for event in store.events
        if event.kind == "funding_settlement"
        and event.payload.get("instrument") == INSTRUMENT
        and start_ns <= event.ts_event_ns < end_ns
    )
    report["settlement_diagnostic"] = {
        "expected_ns": expected_settlements,
        "actual_ns": actual_settlements,
        "tolerance_ns": 15_000_000,
        "match": (
            len(actual_settlements) == len(expected_settlements)
            and all(
                abs(actual - expected) <= 15_000_000
                for actual, expected in zip(actual_settlements, expected_settlements)
            )
        ),
    }
    output_path = OUT_BASE / "quality-audit-tolerant-schedule.json"
    write_verified(output_path, report)
    quality = report["splits"]["train"]
    print(
        json.dumps(
            {
                "output": str(output_path),
                "grade": quality.get("grade"),
                "findings": quality.get("findings"),
                "missing_counts": quality.get("missing_counts"),
                "conflict_count": quality.get("conflict_count"),
                "coverage": quality.get("coverage"),
                "expected_settlement_ns": expected_settlements,
                "actual_settlement_ns": actual_settlements,
                "settlement_match": report["settlement_diagnostic"]["match"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def terminal_receipt(info: dict, source_identity: str) -> TerminalL6Receipt:
    settlement = info["terminal_settlement"]
    return TerminalL6Receipt(
        receipt_id="route-a-train-l6-" + info["execution_decision_id"],
        source_identity=source_identity,
        settlement_sha256=terminal_settlement_sha256(settlement),
        policy_barrier_ts_ns=int(info["decision_ts_ns"]),
        settlement_barrier_ts_ns=int(info["settlement_after_ns"]),
        equity_before=float(info["policy_equity_before"]),
        policy_equity_after=float(info["policy_equity_after"]),
        settled_equity_after=float(settlement["equity_after"]),
        currency=str(info["policy_equity_currency"]),
        gym_recorded_r_econ=float(info["r_econ"]),
        recorded_r_econ=float(info["r_econ"]),
        has_external_cashflow=False,
    )


def make_provider(factory: Any, teacher: Any, source_identity: str, episode_identity: str,
                  *, market_feature_map: dict[str, str] | None = None):
    def no_flow_receipt(info: dict) -> str:
        return canonical_digest(
            {
                "schema": "route-a-no-external-cashflow.v1",
                "source_identity": source_identity,
                "decision_id": info["execution_decision_id"],
                "evidence_hash": info["evidence_hash"],
                "terminal_settled": info["terminal_settled"],
            }
        )

    def verify_terminal(info: dict, source: str, receipt: TerminalL6Receipt) -> bool:
        settlement = info.get("terminal_settlement")

        def matches_equity(receipt_value: float, source_value: Any) -> bool:
            try:
                candidate = float(source_value)
            except (TypeError, ValueError, OverflowError):
                return False
            return math.isfinite(candidate) and abs(receipt_value - candidate) <= 1e-8

        checks = {
            "source_identity": source == source_identity and receipt.source_identity == source,
            "settlement_hash": isinstance(settlement, dict)
            and receipt.settlement_sha256 == terminal_settlement_sha256(settlement),
            "policy_barrier_ts": receipt.policy_barrier_ts_ns == info.get("decision_ts_ns"),
            "settlement_barrier_ts": receipt.settlement_barrier_ts_ns
            == info.get("settlement_after_ns"),
            "equity_before": matches_equity(
                receipt.equity_before, info.get("policy_equity_before")
            ),
            "policy_equity_after": matches_equity(
                receipt.policy_equity_after, info.get("policy_equity_after")
            ),
            "settled_equity_after": isinstance(settlement, dict)
            and matches_equity(receipt.settled_equity_after, settlement.get("equity_after")),
            "currency": receipt.currency == info.get("policy_equity_currency"),
            "recorded_r_econ": receipt.recorded_r_econ == info.get("r_econ"),
            "cashflow": receipt.has_external_cashflow is False,
        }
        if not all(checks.values()):
            print(
                "terminal L6 receipt mismatch fields: "
                + ", ".join(k for k, v in checks.items() if not v)
            )
        return all(checks.values())

    return NativeGymTeacherProvider.from_prepared_factory(
        factory,
        fold_id=FOLD_ID,
        teacher=teacher,
        market_feature_map=market_feature_map,
        native_order_facts_verifier=lambda info, source, facts: (
            source == source_identity
            and facts.native_order_status_receipt == info.get("native_order_status_receipt")
        ),
        no_external_cashflow_receipt=no_flow_receipt,
        no_external_cashflow_verifier=lambda info, source, receipt: (
            source == source_identity
            and info.get("terminal_settled") is False
            and receipt == no_flow_receipt(info)
        ),
        episode_id=lambda _env, _info: episode_identity,
        terminal_receipt_provider=terminal_receipt,
        terminal_receipt_verifier=verify_terminal,
    )


def metric_row(barriers: tuple, report: dict, *, expected_source: str) -> dict:
    _labels, summary = audit_teacher_barriers(barriers)
    verification = report.get("route_a_native_order_verification")
    if (
        not isinstance(verification, dict)
        or verification.get("barriers") != len(barriers)
        or verification.get("authority") != "retained_m22_session_snapshots"
    ):
        raise RuntimeError("M2.2 did not verify every native barrier")
    if summary.economic_status != "SCENARIO_ONLY":
        raise RuntimeError("the run lost its scenario-only economic label")
    if report.get("liquidations_observed") is not True:
        raise RuntimeError("native liquidation state is not observable")
    liquidations = report.get("liquidations")
    if type(liquidations) is not int or liquidations < 0:
        raise RuntimeError("native liquidation count is missing")
    if any(barrier.source_identity != expected_source for barrier in barriers):
        raise RuntimeError("native barrier source differs from the registered Train source")
    equity = [barriers[0].trace.native_equity_before]
    equity.extend(barrier.trace.native_equity_after for barrier in barriers)
    peak = equity[0]
    max_drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, (peak - value) / peak)
    net_return = summary.native_equity_end / summary.native_equity_start - 1.0
    filled = {order_id for barrier in barriers for order_id in barrier.trace.filled_order_ids}
    evidence = _native_barrier_evidence(barriers, report)
    return {
        "summary": summary,
        "net_return": net_return,
        "max_drawdown": max_drawdown,
        "filled_orders": len(filled),
        "liquidations": liquidations,
        "evidence": evidence,
        "barriers": len(barriers),
    }


def run_episode(
    factory: Any,
    teacher: TrendTeacher,
    source_identity: str,
    label: str,
    seed: int,
    episode_steps: int,
    *,
    policy: Any = None,
    market_feature_map: dict[str, str] | None = None,
) -> dict:
    provider, grant = make_provider(
        factory,
        teacher,
        source_identity,
        f"{label}-episode-{seed}",
        market_feature_map=market_feature_map,
    )
    barriers, report = provider.rollout_with_m22_trace(
        factory=factory,
        host_rollout=application.rollout,
        grant=grant,
        seed=seed,
        decision_policy=policy,
        rollout_id=f"{label}-seed-{seed}",
        episode_steps=episode_steps,
        reset_options={"rles_episode_index": seed},
    )
    result = metric_row(barriers, dict(report), expected_source=source_identity)
    return result


def main() -> int:
    training_runtime = configure_torch_threads(torch)
    marker_sha, scope_start, scope_end = load_source_identity()
    if OUT_BASE.exists():
        raise FileExistsError(f"immutable experiment output already exists: {OUT_BASE}")
    OUT_BASE.mkdir(parents=True)
    teacher = TrendTeacher(
        size=0.8,
        threshold=0.004,
        hold_on_neutral=True,
        direction=-1,
        expert_id="countertrend_hold_neutral_v1",
    )
    baseline = WaitTeacher(expert_id="wait_flat_baseline_v1")
    teacher_hash = teacher_config_hash(teacher)
    baseline_hash = teacher_config_hash(baseline)
    source_identity = marker_sha
    owner_token = ACTIVE_OWNER.set("route-a-train-screen-20261020")
    try:
        fit_start, fit_end = map(ns, FIT)
        fit_config, fit_factory = make_config(
            label="fit-jan02-jan09-base",
            marker_sha=marker_sha,
            scope_start=scope_start,
            scope_end=scope_end,
            start_ns=fit_start,
            end_ns=fit_end,
            mode="base",
            fit_scaler=True,
        )
        fit_provider, fit_grant = make_provider(
            fit_factory,
            teacher,
            source_identity,
            "fit-jan02-jan09",
        )
        feature_names = (
            "market.trend_1h",
            "market.trend_1d",
            "market.vol_1m",
            "account.net_exposure",
            "account.drawdown",
            "account.unrealized_pnl_fraction",
        )
        bc_config = BCConfig(
            feature_names=feature_names,
            context_bars=120,
            max_gap_ns=61_000_000_000,
            hidden_size=32,
            layers=1,
            batch_size=128,
            learning_rate=0.001,
            epochs=20,
            seed=31,
        )
        training_result = train_bc_from_registered_native_rollout(
            factory=fit_factory,
            provider=fit_provider,
            host_rollout=application.rollout,
            grant=fit_grant,
            config=bc_config,
            seed=31,
            rollout_options={
                "episode_steps": 2880,
                "reset_options": {"rles_episode_index": 31},
            },
        )
        if training_result.behavior_summary.collapsed_to_flat:
            raise RuntimeError("BC fit collapsed into the flat-action deadband")

        teacher_pairs = []
        student_pairs = []
        matrix_summaries = []
        for window_id, start_text, end_text in WINDOWS:
            start_ns, end_ns = ns(start_text), ns(end_text)
            if any(
                start_ns < ns(other_end) and end_ns > ns(other_start)
                for other_id, other_start, other_end in WINDOWS
                if other_id != window_id
            ):
                raise RuntimeError("preselected Train selection windows overlap")
            for scenario in ("base", "stress"):
                label = f"screen-{window_id}-{scenario}"
                _config, factory = make_config(
                    label=label,
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    mode=scenario,
                    fit_scaler=False,
                    scaler=fit_factory.scaler,
                )
                for seed in SEEDS:
                    episode_label = f"{window_id}-{scenario}"
                    teacher_run = run_episode(
                        factory,
                        teacher,
                        source_identity,
                        episode_label,
                        seed,
                        600,
                    )
                    baseline_run = run_episode(
                        factory,
                        baseline,
                        source_identity,
                        episode_label + "-baseline",
                        seed,
                        600,
                    )
                    student_run = run_episode(
                        factory,
                        teacher,
                        source_identity,
                        episode_label + "-student",
                        seed,
                        600,
                        policy=training_result.policy,
                    )
                    key = f"{window_id}-{scenario}-seed-{seed}"
                    evidence_hashes = {}
                    for role, run in (
                        ("teacher", teacher_run),
                        ("baseline", baseline_run),
                        ("student", student_run),
                    ):
                        evidence_hashes[role] = write_verified(
                            OUT_BASE / "native-evidence" / f"{key}-{role}.json",
                            run["evidence"],
                        )
                    matrix_summaries.append(
                        {
                            "window_id": window_id,
                            "scenario": scenario,
                            "seed": seed,
                            "teacher_net_return": teacher_run["net_return"],
                            "baseline_net_return": baseline_run["net_return"],
                            "student_net_return": student_run["net_return"],
                            "teacher_max_drawdown": teacher_run["max_drawdown"],
                            "student_max_drawdown": student_run["max_drawdown"],
                            "teacher_filled_orders": teacher_run["filled_orders"],
                            "student_filled_orders": student_run["filled_orders"],
                            "teacher_liquidations": teacher_run["liquidations"],
                            "student_liquidations": student_run["liquidations"],
                            "evidence_sha256": evidence_hashes,
                            "evidence_path": {
                                role: f"native-evidence/{key}-{role}.json"
                                for role in evidence_hashes
                            },
                        }
                    )
                    teacher_pairs.append(
                        NativePair(
                            window_id=window_id,
                            scenario=scenario,
                            seed=seed,
                            teacher_recipe=teacher_hash,
                            baseline_recipe=baseline_hash,
                            source_identity=source_identity,
                            teacher_net_return=teacher_run["net_return"],
                            baseline_net_return=baseline_run["net_return"],
                            teacher_max_drawdown=teacher_run["max_drawdown"],
                            baseline_max_drawdown=baseline_run["max_drawdown"],
                            teacher_filled_orders=teacher_run["filled_orders"],
                            teacher_liquidations=teacher_run["liquidations"],
                            matched_native_barriers=(
                                teacher_run["barriers"]
                                == baseline_run["barriers"]
                                == student_run["barriers"]
                                == 600
                            ),
                            scenario_only=True,
                            nonoverlap_verified=True,
                        )
                    )
                    student_pairs.append(
                        StudentPair(
                            window_id=window_id,
                            scenario=scenario,
                            seed=seed,
                            teacher_recipe=teacher_hash,
                            student_recipe=bc_student_recipe_sha256(bc_config),
                            source_identity=source_identity,
                            teacher_net_return=teacher_run["net_return"],
                            student_net_return=student_run["net_return"],
                            teacher_max_drawdown=teacher_run["max_drawdown"],
                            student_max_drawdown=student_run["max_drawdown"],
                            student_filled_orders=student_run["filled_orders"],
                            student_liquidations=student_run["liquidations"],
                            matched_native_barriers=(
                                teacher_run["barriers"] == student_run["barriers"] == 600
                            ),
                            teacher_gate_sha256="0" * 64,
                            actor_sha256=bc_actor_sha256(training_result.policy.model),
                            observation_schema_sha256=bc_observation_schema_sha256(bc_config),
                            action_schema_sha256=bc_action_schema_sha256(),
                            execution_recipe_sha256=fit_factory.route_a_execution_recipe_hash,
                            teacher_evidence_sha256=evidence_hashes["teacher"],
                            student_evidence_sha256=evidence_hashes["student"],
                            scenario_only=True,
                            nonoverlap_verified=True,
                        )
                    )
                del factory

        teacher_screen = screen_teacher(teacher_pairs)
        student_screen = None
        if teacher_screen.status == "TRAIN_INNER_SCREEN_PASS":
            student_pairs = [
                replace(
                    pair,
                    teacher_gate_sha256=teacher_screen.receipt_sha256,
                )
                for pair in student_pairs
            ]
            student_screen = screen_student(student_pairs, teacher_screen=teacher_screen)
            if not verify_student_screen(student_screen):
                raise RuntimeError("host could not recompute the complete Train promotion receipt")

        result = replace(training_result, student_screen=student_screen)
        student_status = (
            student_screen.status if student_screen is not None else "TEACHER_GATE_REJECT"
        )
        if student_screen is not None and student_screen.status == "TRAIN_INNER_SCREEN_PASS":
            eval_selection_config, eval_selection_factory = make_config(
                label="february-base-final-eval-teacher",
                marker_sha=marker_sha,
                scope_start=scope_start,
                scope_end=scope_end,
                start_ns=ns(WINDOWS[0][1]),
                end_ns=ns(WINDOWS[0][2]),
                mode="base",
                fit_scaler=False,
                scaler=fit_factory.scaler,
            )
            select_provider, select_grant = make_provider(
                eval_selection_factory,
                teacher,
                source_identity,
                "february-base-selection",
            )
            _eval_student_config, eval_student_factory = make_config(
                label="february-base-final-eval-student",
                marker_sha=marker_sha,
                scope_start=scope_start,
                scope_end=scope_end,
                start_ns=ns(WINDOWS[0][1]),
                end_ns=ns(WINDOWS[0][2]),
                mode="base",
                fit_scaler=False,
                scaler=fit_factory.scaler,
            )
            eval_student_provider, eval_student_grant = make_provider(
                eval_student_factory,
                teacher,
                source_identity,
                "february-base-final-student",
            )
            result = evaluate_bc_on_registered_train_inner_rollout(
                training_result=result,
                fit_factory=fit_factory,
                selection_factory=eval_selection_factory,
                selection_provider=select_provider,
                student_factory=eval_student_factory,
                student_provider=eval_student_provider,
                host_rollout=application.rollout,
                grant=select_grant,
                seed=11,
                student_grant=eval_student_grant,
                student_seed=29,
                rollout_options={
                    "episode_steps": 600,
                    "reset_options": {"rles_episode_index": 11},
                },
            )
            result = replace(result, student_screen=student_screen)

        experiment_report = {
            "schema": "rles.route_a.train_student_screen_run.v1",
            "source": {
                "path": str(MARKET),
                "marker_sha256": marker_sha,
                "scope_start_ns": scope_start,
                "scope_end_ns": scope_end,
                "source_mode": "whole_snapshot_isolated_v1",
                "source_schema": "0.4.0",
                "rules_path": str(RULES),
                "rules_economic_status": "SCENARIO_ONLY",
            },
            "training": {
                "window_ns": list(training_result.train_window_ns or ()),
                "seed": training_result.training_seed,
                "teacher_recipe_sha256": training_result.teacher_hash,
                "actor_state_sha256": bc_actor_sha256(training_result.policy.model),
                "label_count": training_result.label_count,
                "executable_label_count": training_result.executable_label_count,
                "behavior": asdict(training_result.behavior_summary),
                "runtime": training_runtime,
            },
            "screen_windows": [
                {"id": name, "start_ns": ns(start), "end_ns": ns(end)}
                for name, start, end in WINDOWS
            ],
            "scenarios": ["base", "stress"],
            "seeds": list(SEEDS),
            "pair_metrics": matrix_summaries,
            "teacher_screen": asdict(teacher_screen),
            "student_screen": None if student_screen is None else asdict(student_screen),
            "student_screen_reverified": (
                student_screen is not None and verify_student_screen(student_screen)
            ),
            "execution_recipe_sha256": fit_factory.route_a_execution_recipe_hash,
            "economic_status": "SCENARIO_ONLY",
        }
        report_sha = write_verified(OUT_BASE / "screen-report.json", experiment_report)
        experiment_report["screen_report_sha256"] = report_sha

        if student_screen is not None and student_screen.status == "TRAIN_INNER_SCREEN_PASS":
            wfo_factory = make_route_a_wfo_train(
                fit_candidate=lambda _run: (result, fit_factory),
                validation_backtest=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("Train-only scenario screen must not open Validation")
                ),
                write_verified=write_verified,
                digest=canonical_digest,
            )
            train_end_ns = scope_end
            wfo_run = {
                "output": str(OUT_BASE / "wfo-candidate"),
                "route": "route_a",
                "algorithm": "route_a_bc_gru",
                "environment": {
                    "start_ns": ns("2024-01-01T00:00:00+00:00"),
                    "end_ns": train_end_ns,
                    "execution": fit_config["environment"]["execution"],
                },
            }
            wfo_result = wfo_factory(wfo_run, evaluate=False)
            experiment_report["wfo_candidate"] = wfo_result
            write_verified(OUT_BASE / "screen-report-final.json", experiment_report)
        print(
            json.dumps(
                {
                    "output": str(OUT_BASE),
                    "teacher_status": teacher_screen.status,
                    "teacher_reasons": teacher_screen.reasons,
                    "student_status": student_status,
                    "student_reasons": (
                        student_screen.reasons
                        if student_screen is not None
                        else teacher_screen.reasons
                    ),
                    "student_worst_excess_return": (
                        student_screen.worst_excess_return if student_screen is not None else None
                    ),
                    "student_mean_excess_return": (
                        student_screen.mean_excess_return if student_screen is not None else None
                    ),
                    "economic_status": (
                        student_screen.economic_status
                        if student_screen is not None
                        else teacher_screen.economic_status
                    ),
                    "wfo_actor_exported": (OUT_BASE / "wfo-candidate" / "bundle.json").is_file(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if student_status == "TRAIN_INNER_SCREEN_PASS" else 2
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    import sys

    raise SystemExit(audit_train_quality() if "--audit-only" in sys.argv[1:] else main())
