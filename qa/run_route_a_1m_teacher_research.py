"""Run the pre-registered S07 Train-inner native teacher comparison serially.

All trading and accounting remain in the existing Nautilus Gym/M2.2/L6 path.
The only extra policies in this file are fixed-direction and exposure-schedule
controls passed through that same host path; this module has no order matcher
or portfolio ledger of its own.
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import gc
from hashlib import sha256
import json
import math
from pathlib import Path
import resource
import shutil
import sys
from time import monotonic
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

import run_route_a_1m_teacher_native as native  # noqa: E402
from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation  # noqa: E402
from rles_route_a.integration.l2_adapter import (  # noqa: E402
    CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES,
)
from rles_route_a.teachers.protocol import advice, classify_target  # noqa: E402
from rles_data.features import Scaler  # noqa: E402
from rles_research.native_metrics import iter_account_rows, native_report  # noqa: E402
from verify_route_a_1m_teacher_results import verify_native_run  # noqa: E402


WINDOWS = ("inner-screen-1", "inner-screen-2", "inner-screen-3")
SCENARIOS = ("base", "stress")
FIXED_CONTROLS = ("FLAT", "FIXED_LONG", "FIXED_SHORT")
PILOT_SOURCE = ROOT / ".artifacts/route-a-1m-teacher-native-s07-pilot-t01-x1-base-block1-run02"


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_write(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_frozen_scaler(output: Path) -> tuple[Scaler, dict[str, Any]]:
    path = output / "preparation" / "development-fit-scaler" / "development-scaler.json"
    payload = _read_json(path)
    raw = dict(payload["scaler"])
    raw["mean"] = tuple(raw["mean"])
    raw["std"] = tuple(raw["std"])
    scaler = Scaler(**raw)
    if (scaler.hash != payload.get("scaler_hash")
            or scaler.feature_hash != payload.get("feature_hash")
            or scaler.train_window_hash != payload.get("train_window_hash")
            or scaler.sample_count != payload.get("sample_count")
            or payload.get("inner_screen_used_for_fit") is not False):
        raise RuntimeError("persisted development-fit scaler receipt is invalid")
    return scaler, {key: value for key, value in payload.items() if key != "scaler"}


def _load_teacher_result(
    path: Path, *, doc: dict[str, Any], prereg_sha: str, scaler_hash: str,
    teacher: str, exit_id: str, scenario: str, window_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = path / "native-run.json"
    if not report_path.is_file():
        raise RuntimeError(f"incomplete native teacher artifact cannot be resumed: {path}")
    run = _read_json(report_path)
    start_ns, end_ns, block_id = native._window_bounds(doc, window_id)
    if (run.get("schema") != "rles.route-a.1m-teacher-native-run.v1"
            or run.get("preregistration_sha256") != prereg_sha
            or run.get("teacher") != teacher
            or run.get("teacher_recipe", {}).get("exit_variant") != exit_id
            or run.get("execution", {}).get("scenario") != scenario
            or run.get("execution", {}).get("scaler_hash") != scaler_hash
            or run.get("source", {}).get("window_ns") != [start_ns, end_ns]
            or run.get("block_id") != block_id):
        raise RuntimeError(f"native teacher artifact identity does not match S07 recipe: {path}")
    verification = verify_native_run(path)
    return run, verification


def _load_control_result(
    path: Path, *, doc: dict[str, Any], control_id: str, teacher: str,
    exit_id: str, scenario: str, window_id: str, schedule_sha256: str | None = None,
) -> dict[str, Any]:
    report_path = path / "control-report.json"
    if not report_path.is_file():
        raise RuntimeError(f"incomplete native control artifact cannot be resumed: {path}")
    result = _read_json(report_path)
    start_ns, end_ns, block_id = native._window_bounds(doc, window_id)
    count = (end_ns - start_ns) // native.MINUTE_NS - 3
    if (result.get("schema") != "rles.route-a.1m-teacher-native-control.v1"
            or result.get("status") != "NATIVE_CONTROL_PASS_SCENARIO_ONLY"
            or result.get("control_id") != control_id
            or result.get("teacher") != teacher or result.get("exit") != exit_id
            or result.get("scenario") != scenario or result.get("window_id") != window_id
            or result.get("block_id") != block_id
            or result.get("source_identity") != doc["source"]["marker_sha256"]
            or result.get("registered_window_ns") != [start_ns, end_ns]
            or result.get("teacher_schedule_sha256") != schedule_sha256
            or result.get("economic_status") != "SCENARIO_ONLY"):
        raise RuntimeError(f"native control identity does not match S07 recipe: {path}")
    decisions = result.get("decisions", {})
    if (decisions.get("requested") != count or decisions.get("observed") != count
            or decisions.get("m22_verified") != count
            or result.get("account", {}).get("liquidations") != 0):
        raise RuntimeError(f"native control M2.2/account verification does not reconcile: {path}")
    csv_path = path / "control-decisions.csv"
    observed = 0
    previous = 0
    with csv_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            timestamp = int(row["decision_ts_ns"])
            close = int(row["latest_1m_close_ts_ns"])
            available = int(row["latest_1m_available_at_ns"])
            if (timestamp <= previous or close > available or available > timestamp
                    or (previous and timestamp - previous != native.MINUTE_NS)):
                raise RuntimeError(f"native control per-minute clocks are invalid: {path}")
            previous = timestamp
            observed += 1
    if observed != count:
        raise RuntimeError(f"native control per-minute row count differs: {path}")
    return result


def _memory_peak_mb() -> float:
    # Linux reports KiB for ru_maxrss; this script is Ubuntu-only by contract.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _release_report_memory(factory: Any) -> None:
    """Drop already-exported per-bar reports before the next native account."""
    for name in ("snapshots", "teacher_snapshots", "fill_references"):
        value = getattr(factory, name, None)
        clear = getattr(value, "clear", None)
        if callable(clear):
            clear()
    gc.collect()
    try:
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if callable(trim):
            trim(0)
    except OSError:
        # The fixed-rule account remains correct if the C allocator lacks trim.
        pass


def _verify_resource_caps(result: dict[str, Any], doc: dict[str, Any]) -> None:
    cap_mb = float(doc["resource_limits"]["max_memory_mb"])
    cap_seconds = float(doc["resource_limits"]["max_runtime_seconds"])
    peak_mb = _memory_peak_mb()
    result["process_peak_rss_mb"] = peak_mb
    if peak_mb > cap_mb:
        raise RuntimeError(f"S07 process peak RSS {peak_mb:.1f} MiB exceeds {cap_mb:.0f} MiB")
    if result["runtime_seconds"] > cap_seconds:
        raise RuntimeError(
            f"S07 episode runtime {result['runtime_seconds']:.1f}s exceeds {cap_seconds:.0f}s"
        )


def _teacher_record(output: Path, *, teacher: str, exit_id: str,
                    scenario: str, window_id: str,
                    prepared: tuple[dict[str, Any], Any], prereg_sha: str,
                    doc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    start_ns, end_ns, block_id = native._window_bounds(doc, window_id)
    if (window_id == "inner-screen-1" and teacher == "T01_EMA_9_21"
            and exit_id == "X1" and scenario == "base"
            and PILOT_SOURCE.is_dir()):
        pilot_report = json.loads((PILOT_SOURCE / "native-run.json").read_text(encoding="utf-8"))
        if (pilot_report.get("preregistration_sha256") == prereg_sha
                and pilot_report.get("teacher") == teacher
                and pilot_report.get("teacher_recipe", {}).get("exit_variant") == exit_id
                and pilot_report.get("execution", {}).get("scenario") == scenario
                and pilot_report.get("source", {}).get("window_ns") == [start_ns, end_ns]
                and pilot_report.get("block_id") == block_id
                and pilot_report.get("execution", {}).get("scaler_hash")
                == prepared[1].scaler.hash):
            checked = verify_native_run(PILOT_SOURCE)
            shutil.copytree(PILOT_SOURCE, output)
            pilot_report["reused_for_s07_matrix"] = True
            pilot_report["reused_from"] = str(PILOT_SOURCE.relative_to(ROOT))
            _json_write(output / "native-run.json", pilot_report)
            _release_report_memory(prepared[1])
            return pilot_report, checked

    report = native._run_native_window(
        doc, prereg_sha, output, teacher_id=teacher, exit_variant=exit_id,
        scenario=scenario, window_id=window_id, prepared=prepared,
    )
    checked = verify_native_run(output)
    _verify_resource_caps(report, doc)
    _release_report_memory(prepared[1])
    return report, checked


class _NativeControl:
    """A control advice source only; execution remains the host's native account."""

    def __init__(self, expert_id: str, *, mode: str, schedule: dict[int, float] | None = None,
                 direction: int = 0, target_exposure: float = 0.3) -> None:
        if mode not in {"flat", "fixed", "exposure_matched"}:
            raise ValueError("unknown native control mode")
        if direction not in {-1, 0, 1} or not 0 < target_exposure <= 1:
            raise ValueError("invalid native control target")
        if mode == "exposure_matched" and not schedule:
            raise ValueError("exposure-matched control requires the registered teacher schedule")
        self.expert_id = expert_id
        self.mode = mode
        self.schedule = schedule or {}
        self.direction = direction
        self.target_exposure = target_exposure

    def reset(self, _episode_id: str = "") -> None:
        return None

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        if self.mode == "flat":
            target = 0.0
        elif self.mode == "fixed":
            target = self.direction * self.target_exposure
        else:
            try:
                magnitude = self.schedule[observation.decision_ts_ns]
            except KeyError as exc:
                raise ValueError("exposure schedule lacks a native decision timestamp") from exc
            target = self.direction * magnitude
        current = observation.account_value("net_exposure")
        intent = classify_target(current, target)
        reason = {
            "flat": "PRE_REGISTERED_FLAT_CONTROL",
            "fixed": "PRE_REGISTERED_FIXED_DIRECTION_CONTROL",
            "exposure_matched": "PRE_REGISTERED_DIRECTION_FIXED_EXPOSURE_SCHEDULE_CONTROL",
        }[self.mode]
        return advice(observation, self.expert_id, intent, target, reason)


def _teacher_exposure_schedule(trace_path: Path, expected_count: int) -> tuple[dict[int, float], str]:
    rows: dict[int, float] = {}
    with (trace_path / "teacher-research-1m" / "decision-rows.csv").open(
            encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            timestamp = int(row["decision_ts_ns"])
            value = float(row["requested_exposure"])
            if not math.isfinite(value) or not -1 <= value <= 1 or timestamp in rows:
                raise ValueError("teacher exposure schedule contains invalid/duplicate rows")
            rows[timestamp] = abs(value)
    if len(rows) != expected_count:
        raise ValueError("teacher exposure schedule row count differs from native decisions")
    encoded = json.dumps(sorted(rows.items()), separators=(",", ":"), allow_nan=False)
    return rows, sha256(encoded.encode("utf-8")).hexdigest()


def _run_native_control(
    output: Path,
    *,
    doc: dict[str, Any],
    teacher_id: str,
    exit_id: str,
    scenario: str,
    window_id: str,
    prepared: tuple[dict[str, Any], Any],
    control_id: str,
    mode: str,
    direction: int = 0,
    schedule: dict[int, float] | None = None,
    schedule_sha256: str | None = None,
) -> dict[str, Any]:
    started = monotonic()
    config, factory = prepared
    start_ns, end_ns, block_id = native._window_bounds(doc, window_id)
    source = doc["source"]
    target_exposure = float(doc["feature_recipe"]["target_exposure"])
    policy = _NativeControl(
        f"S07_CONTROL_{control_id}_{teacher_id}_{exit_id}_{block_id}_{scenario}",
        mode=mode, schedule=schedule, direction=direction,
        target_exposure=target_exposure,
    )
    run_label = f"s07-control-{control_id}-{teacher_id}-{exit_id}-{scenario}-{block_id}"
    provider, grant = native.harness.make_provider(
        factory,
        policy,
        source["marker_sha256"],
        run_label,
        market_feature_map=CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES,
    )
    provider.visible_context = native._StreamingVisibleContext(
        factory, start_ns=start_ns, end_ns=end_ns,
        instrument=doc["market"]["instrument"], warmup_bars=61,
    )
    episode_steps = (end_ns - start_ns) // native.MINUTE_NS - 3
    barriers, host = provider.rollout_with_m22_trace(
        factory=factory,
        host_rollout=native.application.rollout,
        grant=grant,
        seed=20261026,
        decision_policy=None,
        rollout_id=run_label,
        episode_steps=episode_steps,
        reset_options={"rles_episode_index": 20261026},
    )
    verification = host.get("route_a_native_order_verification")
    if (len(barriers) != episode_steps or not isinstance(verification, dict)
            or verification.get("barriers") != episode_steps
            or verification.get("authority") != "retained_m22_session_snapshots"):
        raise RuntimeError("native control decision count/M2.2 barrier verification failed")
    if host.get("liquidations") != 0 or host.get("liquidations_observed") is not True:
        raise RuntimeError("native control had a liquidation or omitted liquidation evidence")

    output.mkdir(parents=True, exist_ok=False)
    exposure_sum = scheduled_sum = 0.0
    exposure_time_bars = 0
    decision_fields = (
        "control_id", "teacher_id", "exit_id", "scenario_id", "block_id",
        "decision_ts_ns", "latest_1m_close_ts_ns", "latest_1m_available_at_ns",
        "intent", "requested_exposure", "scheduled_exposure", "native_exposure_after",
        "submitted_order_ids_json", "filled_order_ids_json",
    )
    with (output / "control-decisions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=decision_fields, extrasaction="raise")
        writer.writeheader()
        for barrier in barriers:
            observation = barrier.observation
            decision_ns = observation.decision_ts_ns
            close_ns = observation.market_clock_ns("latest_1m_close_ts_ns")
            available_ns = max(observation.feature_available_at_ns.values())
            if not close_ns <= available_ns <= decision_ns:
                raise RuntimeError("native control observed a non-causal 1m feature clock")
            if mode == "flat":
                scheduled = 0.0
            elif mode == "fixed":
                scheduled = direction * target_exposure
            else:
                assert schedule is not None
                scheduled = direction * schedule[decision_ns]
            actual = float(barrier.trace.actual_exposure_after)
            exposure_sum += abs(actual)
            scheduled_sum += abs(scheduled)
            exposure_time_bars += abs(actual) > 1e-8
            writer.writerow({
                "control_id": control_id,
                "teacher_id": teacher_id,
                "exit_id": exit_id,
                "scenario_id": scenario,
                "block_id": block_id,
                "decision_ts_ns": decision_ns,
                "latest_1m_close_ts_ns": close_ns,
                "latest_1m_available_at_ns": available_ns,
                "intent": barrier.advice.intent,
                "requested_exposure": barrier.advice.requested_exposure,
                "scheduled_exposure": scheduled,
                "native_exposure_after": actual,
                "submitted_order_ids_json": json.dumps(
                    barrier.trace.submitted_order_ids, separators=(",", ":"),
                ),
                "filled_order_ids_json": json.dumps(
                    barrier.native_policy_fills and tuple(
                        str(row.get("client_order_id")) for row in barrier.native_policy_fills
                    ) or (), separators=(",", ":"),
                ),
            })
            for fill in barrier.native_policy_fills:
                fill_ts = fill.get("ts_event")
                if type(fill_ts) is not int or fill_ts <= available_ns:
                    raise RuntimeError("native control fill was not after the available decision")

    account = native_report(
        factory.snapshots, fill_references=factory.fill_references,
        sample_interval_ns=native.MINUTE_NS, allow_price_improvement=False,
        include_account_rows=False,
    )
    if not factory.snapshots or not isinstance(account.get("metrics"), dict):
        raise RuntimeError("native control omitted account snapshots or metrics")
    metrics = account["metrics"]
    first, last = factory.snapshots[0], factory.snapshots[-1]
    account_row_count = 0
    account_rows_iter = iter_account_rows(
        factory.snapshots, fill_references=factory.fill_references,
        allow_price_improvement=False,
    )
    account_row_iter = iter(account_rows_iter)
    first_account_row = next(account_row_iter, None)
    if first_account_row is None:
        raise RuntimeError("native control omitted native account rows")
    with (output / "account-rows.csv").open("x", encoding="utf-8", newline="") as stream:
        account_writer = csv.DictWriter(
            stream, fieldnames=tuple(first_account_row), extrasaction="raise",
        )
        account_writer.writeheader()
        account_writer.writerow(first_account_row)
        account_row_count = 1
        for row in account_row_iter:
            account_writer.writerow(row)
            account_row_count += 1
    result = {
        "schema": "rles.route-a.1m-teacher-native-control.v1",
        "status": "NATIVE_CONTROL_PASS_SCENARIO_ONLY",
        "control_id": control_id,
        "control_mode": mode,
        "direction": direction,
        "teacher_schedule_sha256": schedule_sha256,
        "teacher": teacher_id,
        "exit": exit_id,
        "scenario": scenario,
        "window_id": window_id,
        "block_id": block_id,
        "protocol_id": doc["protocol_id"],
        "source_identity": source["marker_sha256"],
        "registered_window_ns": [start_ns, end_ns],
        "decisions": {
            "requested": episode_steps,
            "observed": len(barriers),
            "m22_verified": verification["barriers"],
            "native_fill_events": verification.get("policy_fill_events"),
            "submitted_order_count": sum(
                len(barrier.trace.submitted_order_ids) for barrier in barriers
            ),
        },
        "account": {
            "equity_currency": first.currency,
            "equity_start": float(first.equity),
            "equity_end": float(last.equity),
            "net_return": metrics.get("net_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "fees": metrics.get("total_fee"),
            "slippage_cost": metrics.get("total_slippage_cost"),
            "funding": metrics.get("total_funding"),
            "turnover_ratio": metrics.get("turnover_ratio"),
            "liquidations": host.get("liquidations"),
            "fill_count": len(account.get("fills", ())),
        },
        "exposure": {
            "mean_abs_actual_exposure": exposure_sum / len(barriers),
            "exposure_time_fraction": exposure_time_bars / len(barriers),
            "mean_abs_scheduled_exposure": scheduled_sum / len(barriers),
        },
        "runtime_seconds": monotonic() - started,
        "qualification": "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2",
        "economic_status": "SCENARIO_ONLY",
        "native_account_rows": account_row_count,
        "native_snapshot_compaction": factory.compact_report_snapshots,
        "native_snapshot_history_sharing": factory.share_snapshot_histories,
    }
    result["process_peak_rss_mb"] = _memory_peak_mb()
    _release_report_memory(factory)
    del barriers, account, account_rows_iter, account_row_iter
    _verify_resource_caps(result, doc)
    _json_write(output / "control-report.json", result)
    return result


def _matrix_row(run: dict[str, Any], *, kind: str, path: Path) -> dict[str, Any]:
    account = run["account"]
    return {
        "run_kind": kind,
        "teacher": run.get("teacher", ""),
        "exit": run.get("teacher_recipe", {}).get("exit_variant", run.get("exit", "")),
        "scenario": run.get("execution", {}).get("scenario", run.get("scenario", "")),
        "window_id": run.get("source", {}).get("window_id", run.get("window_id", "")),
        "block_id": run.get("block_id", ""),
        "control_id": run.get("control_id", ""),
        "decisions": run.get("decisions", {}).get("observed", ""),
        "native_cases": run.get("decisions", {}).get("native_trade_cases", ""),
        "ordinary_complete": run.get("verification", {}).get("ordinary_complete", ""),
        "terminal_censored": run.get("verification", {}).get("terminal_censored", ""),
        "equity_start": account.get("equity_start"),
        "equity_end": account.get("equity_end"),
        "net_return": account.get("net_return"),
        "max_drawdown": account.get("max_drawdown"),
        "fees": account.get("fees"),
        "slippage_cost": account.get("slippage_cost"),
        "funding": account.get("funding"),
        "runtime_seconds": run.get("runtime_seconds"),
        "artifact_path": str(path),
        "economic_status": run.get("economic_status", "SCENARIO_ONLY"),
    }


def _write_summary(output: Path, report: dict[str, Any]) -> None:
    rows = report.get("matrix_rows", [])
    lines = [
        "# S07 — 四教师 × 双退出 Train-inner 原生对照",
        "",
        f"状态：**{report.get('status', 'RUNNING')}**；仅 D1 development diagnostic / SCENARIO_ONLY。",
        "",
        f"预注册：`{report.get('preregistration_sha256', '')}`。时间块数：{len(report.get('source_audits', []))}。",
        f"教师账户：{sum(row['run_kind'] == 'teacher' for row in rows)}；原生控制账户：{sum(row['run_kind'] == 'control' for row in rows)}。",
        "",
        "| 类型 | 教师 | 退出 | 场景 | 时间块 | 决策 | 案例 | 终值净收益 | 最大回撤 | 费用 | 滑点 | 资金费 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_kind} | {teacher} | {exit} | {scenario} | {block_id} | {decisions} | {native_cases} | "
            "{net_return} | {max_drawdown} | {fees} | {slippage_cost} | {funding} |".format(**row)
        )
    lines.extend([
        "",
        "退出配对说明：连续组合 X1/X2 各自独立账户并允许自然再入场；结果包含再入场路径差异。",
        f"单次机会级退出隔离臂：`{report.get('exit_isolation_status', 'NOT_RUN')}`。",
        "以上均为公开历史回放 + 场景化成本/账户规则，不代表实际交易经济或 G1/G2 资格。",
    ])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _preload_completed_window(
    output: Path, *, doc: dict[str, Any], prereg_sha: str, scaler_hash: str,
    window_id: str, scenario: str, report: dict[str, Any],
) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Verify and register complete prior artifacts before constructing a large factory."""
    teachers: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    controls: dict[str, dict[str, Any]] = {}

    def add_control(path: Path, *, control_id: str, teacher: str, exit_id: str,
                    schedule_sha256: str | None = None) -> dict[str, Any]:
        relative = str(path.relative_to(output))
        result = _load_control_result(
            path, doc=doc, control_id=control_id, teacher=teacher,
            exit_id=exit_id, scenario=scenario, window_id=window_id,
            schedule_sha256=schedule_sha256,
        )
        result["artifact_path"] = relative
        controls[relative] = result
        report["runs"].append(result)
        report["matrix_rows"].append(
            _matrix_row(result, kind="control", path=Path(relative))
        )
        return result

    for teacher in native.TEACHERS:
        for exit_id in ("X1", "X2"):
            rel_path = Path("native") / "teachers" / teacher / exit_id / window_id / scenario
            run_path = output / rel_path
            if not run_path.exists():
                continue
            run, verification = _load_teacher_result(
                run_path, doc=doc, prereg_sha=prereg_sha,
                scaler_hash=scaler_hash, teacher=teacher,
                exit_id=exit_id, scenario=scenario, window_id=window_id,
            )
            run["verification"] = verification
            run["artifact_path"] = str(rel_path)
            teachers[str(rel_path)] = (run, verification)
            report["runs"].append(run)
            report["matrix_rows"].append(_matrix_row(run, kind="teacher", path=rel_path))

            start_ns, end_ns, _ = native._window_bounds(doc, window_id)
            episode_steps = (end_ns - start_ns) // native.MINUTE_NS - 3
            schedule, schedule_sha = _teacher_exposure_schedule(run_path, episode_steps)
            del schedule
            for direction_name in ("LONG", "SHORT"):
                control_id = f"EXPOSURE_MATCHED_{direction_name}"
                control_path = (
                    output / "native" / "exposure-matched" / control_id
                    / teacher / exit_id / window_id / scenario
                )
                if control_path.exists():
                    add_control(
                        control_path, control_id=control_id, teacher=teacher,
                        exit_id=exit_id, schedule_sha256=schedule_sha,
                    )

    for control_id in FIXED_CONTROLS:
        control_path = output / "native" / "controls" / control_id / window_id / scenario
        if control_path.exists():
            add_control(
                control_path, control_id=control_id, teacher="ALL", exit_id="NA",
            )
    return teachers, controls


def run_s07(output: Path, prereg_path: Path, *, resume: bool = False) -> dict[str, Any]:
    if output.exists() and not resume:
        raise FileExistsError(f"S07 output must be a new directory unless --resume is set: {output}")
    if resume and (not output.is_dir() or not (output / "progress.json").is_file()):
        raise FileNotFoundError("S07 resume requires a prior progress.json in the output directory")
    if not output.exists():
        output.mkdir(parents=True)
    doc, prereg_sha = native.verify_preregistration(prereg_path)
    prior_progress = _read_json(output / "progress.json") if resume else {}
    if resume and prior_progress.get("stage") not in {"matrix", "development_fit_scaler"}:
        raise RuntimeError("S07 resume is allowed only from a stopped matrix stage")
    if resume and (output / "report.json").exists():
        raise FileExistsError("completed S07 report is immutable and cannot be resumed")
    if resume and (output / "partial-report.json").is_file():
        prior_partial = _read_json(output / "partial-report.json")
        if (prior_partial.get("schema") != "rles.route-a.1m-teacher-research-s07.v1"
                or prior_partial.get("preregistration_sha256") != prereg_sha):
            raise RuntimeError("S07 partial report belongs to another protocol")
    expected_teacher_runs = len(native.TEACHERS) * 2 * len(SCENARIOS) * len(WINDOWS)
    report: dict[str, Any] = {
        "schema": "rles.route-a.1m-teacher-research-s07.v1",
        "status": "RUNNING",
        "protocol_id": doc["protocol_id"],
        "preregistration_sha256": prereg_sha,
        "market": doc["market"],
        "source": doc["source"],
        "roles": doc["roles"],
        "scenario_costs": doc["cost_scenarios"],
        "teacher_trial_count": 4 * 2 * 2,
        "expected_teacher_block_scenario_accounts": expected_teacher_runs,
        "resource_limits": doc["resource_limits"],
        "source_audits": [],
        "scaler_receipt": None,
        "runs": [],
        "matrix_rows": [],
        "exit_isolation_status": "EXIT_ISOLATION_BLOCKED",
        "exit_isolation_reason": (
            "The existing native provider does not expose a state-clone/prefix-stop primitive. "
            "No shadow account engine was introduced; continuous portfolio-policy X1/X2 runs "
            "are reported without causal exit attribution."
        ),
        "qualification": "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2",
        "economic_status": "SCENARIO_ONLY",
    }
    if not resume:
        _json_write(output / "progress.json", {
            "stage": "source_audits", "completed": 0, "expected": len(WINDOWS),
        })
    try:
        # Source access and physical-month guards must pass separately for every block.
        for index, window_id in enumerate(WINDOWS, start=1):
            audit_dir = output / "source-audits" / window_id
            audit_file = audit_dir / "source-audit.json"
            if resume and audit_file.is_file():
                audit = _read_json(audit_file)
            else:
                audit_dir.mkdir(parents=True, exist_ok=True)
                audit = native._source_audit(doc, prereg_sha, audit_dir, window_id=window_id)
            if (audit.get("protected_month_guard_passed_before_decode") is not True
                    or audit.get("market_rows_decoded_after_registered_train_reservation") is not True
                    or audit.get("preregistration_sha256") != prereg_sha
                    or audit.get("source", {}).get("window_id") != window_id):
                raise RuntimeError(f"registered Train source audit did not pass for {window_id}")
            report["source_audits"].append(audit)
            _json_write(output / "progress.json", {
                "stage": "source_audits", "completed": index, "expected": len(WINDOWS),
            })

        scaler_dir = output / "preparation" / "development-fit-scaler"
        if resume and (scaler_dir / "development-scaler.json").is_file():
            scaler, scaler_receipt = _load_frozen_scaler(output)
        else:
            scaler, scaler_receipt = native.fit_development_scaler(doc, scaler_dir)
        report["scaler_receipt"] = scaler_receipt
        _json_write(output / "progress.json", {
            "stage": "development_fit_scaler", "scaler_hash": scaler.hash,
        })

        for window_index, window_id in enumerate(WINDOWS, start=1):
            for scenario in SCENARIOS:
                cached_teachers, cached_controls = _preload_completed_window(
                    output, doc=doc, prereg_sha=prereg_sha, scaler_hash=scaler.hash,
                    window_id=window_id, scenario=scenario, report=report,
                )
                _write_summary(output, report)
                factory_dir = output / "preparation" / window_id / scenario
                factory_dir.mkdir(parents=True, exist_ok=True)
                prepared = native.prepare_window_factory(
                    doc, factory_dir, window_id=window_id, scenario=scenario, scaler=scaler,
                    share_snapshot_histories=True,
                )
                start_ns, end_ns, block_id = native._window_bounds(doc, window_id)
                episode_steps = (end_ns - start_ns) // native.MINUTE_NS - 3
                for teacher in native.TEACHERS:
                    for exit_id in ("X1", "X2"):
                        rel_path = Path("native") / "teachers" / teacher / exit_id / window_id / scenario
                        run_path = output / rel_path
                        if run_path.exists():
                            cached = cached_teachers.get(str(rel_path))
                            if cached is None:
                                run, verification = _load_teacher_result(
                                    run_path, doc=doc, prereg_sha=prereg_sha,
                                    scaler_hash=scaler.hash, teacher=teacher,
                                    exit_id=exit_id, scenario=scenario, window_id=window_id,
                                )
                            else:
                                run, verification = cached
                        else:
                            run, verification = _teacher_record(
                                run_path, teacher=teacher, exit_id=exit_id,
                                scenario=scenario, window_id=window_id,
                                prepared=prepared, prereg_sha=prereg_sha, doc=doc,
                            )
                        if run.get("execution", {}).get("scaler_hash") != scaler.hash:
                            raise RuntimeError("teacher run did not use the frozen development-fit scaler")
                        run["verification"] = verification
                        run["artifact_path"] = str(rel_path)
                        row = _matrix_row(run, kind="teacher", path=rel_path)
                        if cached_teachers.get(str(rel_path)) is None:
                            report["runs"].append(run)
                            report["matrix_rows"].append(row)
                            _write_summary(output, report)
                        _json_write(output / "progress.json", {
                            "stage": "matrix", "window_index": window_index,
                            "window": window_id, "scenario": scenario,
                            "current_teacher": teacher, "current_exit": exit_id,
                            "completed_teacher_accounts": sum(
                                entry.get("schema") == "rles.route-a.1m-teacher-native-run.v1"
                                for entry in report["runs"]
                            ),
                            "expected_teacher_accounts": expected_teacher_runs,
                            "max_concurrent_episodes": 1,
                        })

                        schedule, schedule_sha = _teacher_exposure_schedule(run_path, episode_steps)
                        for direction_name, direction in (("LONG", 1), ("SHORT", -1)):
                            control_id = f"EXPOSURE_MATCHED_{direction_name}"
                            control_path = (
                                output / "native" / "exposure-matched" / control_id
                                / teacher / exit_id / window_id / scenario
                            )
                            if control_path.exists():
                                cached_control = cached_controls.get(
                                    str(control_path.relative_to(output))
                                )
                                if cached_control is None:
                                    matched = _load_control_result(
                                        control_path, doc=doc, control_id=control_id,
                                        teacher=teacher, exit_id=exit_id,
                                        scenario=scenario, window_id=window_id,
                                        schedule_sha256=schedule_sha,
                                    )
                                else:
                                    matched = cached_control
                            else:
                                matched = _run_native_control(
                                    control_path, doc=doc, teacher_id=teacher, exit_id=exit_id,
                                    scenario=scenario, window_id=window_id, prepared=prepared,
                                    control_id=control_id, mode="exposure_matched",
                                    direction=direction, schedule=schedule,
                                    schedule_sha256=schedule_sha,
                                )
                            matched["artifact_path"] = str(control_path.relative_to(output))
                            if cached_controls.get(str(control_path.relative_to(output))) is None:
                                report["runs"].append(matched)
                                report["matrix_rows"].append(
                                    _matrix_row(matched, kind="control", path=control_path.relative_to(output))
                                )
                                _write_summary(output, report)
                            _json_write(output / "progress.json", {
                                "stage": "matrix", "window_index": window_index,
                                "window": window_id, "scenario": scenario,
                                "current_teacher": teacher, "current_exit": exit_id,
                                "current_control": control_id,
                                "completed_teacher_accounts": sum(
                                    entry.get("schema") == "rles.route-a.1m-teacher-native-run.v1"
                                    for entry in report["runs"]
                                ),
                                "expected_teacher_accounts": expected_teacher_runs,
                                "max_concurrent_episodes": 1,
                            })

                for control_id, mode, direction in (
                    ("FLAT", "flat", 0),
                    ("FIXED_LONG", "fixed", 1),
                    ("FIXED_SHORT", "fixed", -1),
                ):
                    control_path = (
                        output / "native" / "controls" / control_id / window_id / scenario
                    )
                    if control_path.exists():
                        cached_control = cached_controls.get(
                            str(control_path.relative_to(output))
                        )
                        if cached_control is None:
                            result = _load_control_result(
                                control_path, doc=doc, control_id=control_id,
                                teacher="ALL", exit_id="NA", scenario=scenario,
                                window_id=window_id,
                            )
                        else:
                            result = cached_control
                    else:
                        result = _run_native_control(
                            control_path, doc=doc, teacher_id="ALL", exit_id="NA",
                            scenario=scenario, window_id=window_id, prepared=prepared,
                            control_id=control_id, mode=mode, direction=direction,
                        )
                    row = _matrix_row(result, kind="control", path=control_path.relative_to(output))
                    if cached_controls.get(str(control_path.relative_to(output))) is None:
                        report["runs"].append({**result, "artifact_path": str(control_path.relative_to(output))})
                        report["matrix_rows"].append(row)
                        _write_summary(output, report)
                    _json_write(output / "progress.json", {
                        "stage": "matrix", "window_index": window_index,
                        "window": window_id, "scenario": scenario,
                        "completed_teacher_accounts": sum(
                            entry.get("schema") == "rles.route-a.1m-teacher-native-run.v1"
                            for entry in report["runs"]
                        ),
                        "expected_teacher_accounts": expected_teacher_runs,
                        "max_concurrent_episodes": 1,
                    })

        completed_teacher = sum(
            entry.get("schema") == "rles.route-a.1m-teacher-native-run.v1"
            for entry in report["runs"]
        )
        if completed_teacher != expected_teacher_runs:
            raise RuntimeError(
                f"S07 teacher matrix incomplete: {completed_teacher}/{expected_teacher_runs}"
            )
        report["status"] = "BEHAVIOR_VERIFIED_TRAIN_DIAGNOSTIC_SCENARIO_ONLY"
        report["completed_teacher_block_scenario_accounts"] = completed_teacher
        report["fixed_control_accounts"] = len(WINDOWS) * len(SCENARIOS) * len(FIXED_CONTROLS)
        report["exposure_matched_control_accounts"] = expected_teacher_runs * 2
        report["all_accounts_independent_and_reset"] = True
        report["max_concurrent_episodes"] = 1
        report["no_teacher_selected_from_results"] = True
        report["g1_status"] = "NOT_ASSESSED_DEVELOPMENT_DIAGNOSTIC_ONLY"
        report["g2_status"] = "NOT_ASSESSED_SCENARIO_ONLY"
        _json_write(output / "report.json", report)
        _write_summary(output, report)
        _csv_write(
            output / "matrix.csv", report["matrix_rows"],
            ("run_kind", "teacher", "exit", "scenario", "window_id", "block_id",
             "control_id", "decisions", "native_cases", "ordinary_complete",
             "terminal_censored", "equity_start", "equity_end", "net_return",
             "max_drawdown", "fees", "slippage_cost", "funding", "runtime_seconds",
             "artifact_path", "economic_status"),
        )
        _json_write(output / "progress.json", {
            "stage": "complete", "teacher_accounts": completed_teacher,
            "control_accounts": report["fixed_control_accounts"]
            + report["exposure_matched_control_accounts"],
        })
        return report
    except Exception as exc:
        report["status"] = "BLOCKED_NOT_A_PASS"
        report["failure"] = {
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "stage": json.loads((output / "progress.json").read_text(encoding="utf-8"))["stage"],
        }
        _json_write(output / "partial-report.json", report)
        _write_summary(output, report)
        raise


def run_control_smoke(output: Path, prereg_path: Path) -> dict[str, Any]:
    """Short S06-window smoke for the S07 control advice path only."""
    if output.exists():
        raise FileExistsError(f"S07 control smoke output must be new: {output}")
    output.mkdir(parents=True)
    doc, _prereg_sha = native.verify_preregistration(prereg_path)
    audit_dir = output / "source-audit"
    audit_dir.mkdir()
    audit = native._source_audit(doc, _prereg_sha, audit_dir, window_id="smoke")
    if audit.get("protected_month_guard_passed_before_decode") is not True:
        raise RuntimeError("native control smoke source guard failed")
    scaler, scaler_receipt = native.fit_development_scaler(
        doc, output / "preparation" / "development-fit-scaler",
    )
    prepared = native.prepare_window_factory(
        doc, output / "preparation" / "smoke" / "base",
        window_id="smoke", scenario="base", scaler=scaler,
        share_snapshot_histories=True,
    )
    fixed_long = _run_native_control(
        output / "fixed-long", doc=doc, teacher_id="ALL", exit_id="NA",
        scenario="base", window_id="smoke", prepared=prepared,
        control_id="FIXED_LONG", mode="fixed", direction=1,
    )
    trace = ROOT / ".artifacts/route-a-1m-teacher-native-s06-run06"
    steps = (native._window_bounds(doc, "smoke")[1]
             - native._window_bounds(doc, "smoke")[0]) // native.MINUTE_NS - 3
    schedule, schedule_sha = _teacher_exposure_schedule(trace, steps)
    matched: list[dict[str, Any]] = []
    for name, direction in (("EXPOSURE_MATCHED_LONG", 1), ("EXPOSURE_MATCHED_SHORT", -1)):
        matched.append(_run_native_control(
            output / name.lower(), doc=doc, teacher_id="T01_EMA_9_21", exit_id="X1",
            scenario="base", window_id="smoke", prepared=prepared,
            control_id=name, mode="exposure_matched", direction=direction,
            schedule=schedule, schedule_sha256=schedule_sha,
        ))
    result = {
        "schema": "rles.route-a.1m-teacher-control-smoke.v1",
        "status": "NATIVE_CONTROL_PASS_SCENARIO_ONLY",
        "development_scaler_receipt": scaler_receipt,
        "source_audit": audit,
        "fixed_long": fixed_long,
        "exposure_matched": matched,
        "economic_status": "SCENARIO_ONLY",
    }
    _json_write(output / "control-smoke-report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=native.ROOT_DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-plan", action="store_true")
    parser.add_argument("--smoke-control", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.dry_plan:
        teacher = len(native.TEACHERS) * 2 * len(SCENARIOS) * len(WINDOWS)
        controls = teacher * 2 + len(WINDOWS) * len(SCENARIOS) * len(FIXED_CONTROLS)
        print(json.dumps({
            "teacher_episodes": teacher,
            "fixed_control_episodes": len(WINDOWS) * len(SCENARIOS) * len(FIXED_CONTROLS),
            "exposure_matched_native_control_episodes": teacher * 2,
            "total_sequential_native_episodes": teacher + controls,
            "windows": WINDOWS,
            "scenarios": SCENARIOS,
            "max_concurrent_episodes": 1,
        }, sort_keys=True))
        return 0
    if args.smoke_control:
        if args.resume:
            parser.error("--resume applies only to a stopped full S07 matrix")
        result = run_control_smoke(output, args.prereg)
        print(json.dumps({
            "status": result["status"],
            "fixed_long_decisions": result["fixed_long"]["decisions"],
            "matched_controls": [
                {"control_id": row["control_id"], "decisions": row["decisions"],
                 "exposure": row["exposure"], "net_return": row["account"]["net_return"]}
                for row in result["exposure_matched"]
            ],
            "output": str(output),
        }, sort_keys=True))
        return 0
    result = run_s07(output, args.prereg, resume=args.resume)
    print(json.dumps({
        "status": result["status"],
        "teacher_accounts": result["completed_teacher_block_scenario_accounts"],
        "control_accounts": result["fixed_control_accounts"]
        + result["exposure_matched_control_accounts"],
        "output": str(output),
        "qualification": result["qualification"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
