"""Fail-closed integrity checks for S06 Route A 1m native smoke outputs."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

MINUTE_NS = 60_000_000_000


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def verify_native_run(output: Path) -> dict[str, Any]:
    smoke_path = output / "native-smoke.json"
    is_smoke = smoke_path.is_file()
    report = _read_json(smoke_path if is_smoke else output / "native-run.json")
    expected_schema = ("rles.route-a.1m-teacher-native-smoke.v1" if is_smoke
                       else "rles.route-a.1m-teacher-native-run.v1")
    expected_status = ("ENGINEERING_SMOKE_PASS_SCENARIO_ONLY" if is_smoke
                       else "ENGINEERING_RESEARCH_BLOCK_PASS_SCENARIO_ONLY")
    if (report.get("schema") != expected_schema
            or report.get("status") != expected_status
            or report.get("economic_status") != "SCENARIO_ONLY"
            or report.get("qualification") != "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2"):
        raise ValueError("native run is not a scenario-only development result")
    if (report["execution"].get("scenario") not in {"base", "stress"}
            or report["execution"].get("max_leverage") != "1"):
        raise ValueError("native smoke violated its pre-registered execution recipe")
    decisions = report["decisions"]
    count = decisions.get("requested")
    if (type(count) is not int or count < 1 or (is_smoke and count > 1500)
            or decisions.get("observed") != count
            or decisions.get("m22_verified") != count):
        raise ValueError("native decision count and M2.2 verified count differ")
    window = report["source"].get("window_ns")
    if (not isinstance(window, list) or len(window) != 2
            or any(type(value) is not int for value in window)
            or window[0] >= window[1]
            or count != (window[1] - window[0]) // MINUTE_NS - 3):
        raise ValueError("native decision count differs from its frozen window")
    if report["account"].get("liquidations") != 0:
        raise ValueError("native smoke had a liquidation")

    research = output / "teacher-research-1m"
    funnel = _read_json(research / "teacher-funnel.json")
    case_summary = _read_json(research / "case-summary.json")
    with (research / "decision-rows.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != count:
        raise ValueError("per-minute decision rows do not match the native barrier count")
    previous_decision = 0
    for row in rows:
        decision = int(row["decision_ts_ns"])
        close = int(row["latest_1m_close_ts_ns"])
        available = int(row["latest_1m_available_at_ns"])
        execution = int(row["execution_ts_ns"])
        if (decision <= previous_decision or close > available or available > decision
                or execution <= available):
            raise ValueError("feature/decision/execution clock ordering is not causal")
        if previous_decision and decision - previous_decision != MINUTE_NS:
            raise ValueError("native run omitted or duplicated an available 1m decision minute")
        previous_decision = decision
        if row["economic_status"] != "SCENARIO_ONLY":
            raise ValueError("decision row lost its scenario-only label")
        if row["teacher_id"] != report["teacher"] or row["scenario_id"] != report["execution"]["scenario"]:
            raise ValueError("decision row identity differs from the native run recipe")
        if row["block_id"] != report["block_id"] or not window[0] <= decision < window[1]:
            raise ValueError("decision row is outside its frozen block identity/window")
        indicators = json.loads(row["indicator_values_json"])
        if not indicators or "bar_close_1m" not in indicators:
            raise ValueError("decision row omitted the causal 1m close input")

    cases = []
    with (research / "native-trades.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                cases.append(json.loads(line))
    ids = [case["trade_case_id"] for case in cases]
    if len(ids) != len(set(ids)) or len(cases) != decisions.get("native_trade_cases"):
        raise ValueError("native trade-case identities/count differ from the smoke receipt")
    terminal_case_ids = set()
    ordinary_case_ids = set()
    rows_by_decision = {int(row["decision_ts_ns"]): row for row in rows}
    for case in cases:
        if case["economic_status"] != "SCENARIO_ONLY":
            raise ValueError("trade case lost its scenario-only label")
        entry_row = rows_by_decision.get(case["entry_decision_ts_ns"])
        if entry_row is None or case["entry_fill_ts_ns"] <= int(entry_row["latest_1m_available_at_ns"]):
            raise ValueError("entry fill did not occur after its signal was available")
        if case["status"] == "ORDINARY_COMPLETE":
            ordinary_case_ids.add(case["trade_case_id"])
            exit_row = rows_by_decision.get(next(
                int(row["decision_ts_ns"]) for row in rows
                if row["case_id"] == case["trade_case_id"]
            ))
            if (exit_row is None or case["exit_fill_ts_ns"] is None
                    or case["exit_fill_ts_ns"] <= int(exit_row["latest_1m_available_at_ns"])):
                raise ValueError("ordinary exit fill did not follow its available decision")
        elif case["status"] == "TERMINAL_CENSORED":
            terminal_case_ids.add(case["trade_case_id"])
            if any(case.get(key) is not None for key in (
                    "exit_signal_price", "exit_fill_ts_ns", "exit_fill_price")):
                raise ValueError("terminal-censored case fabricated an ordinary exit")
        else:
            raise ValueError("unknown native case lifecycle status")
    row_case_ids = {
        row["case_id"] for row in rows
        if row["case_id"] and row["case_status"] in {
            "ORDINARY_COMPLETE", "OPEN_CENSORED", "TERMINAL_CENSORED",
        }
    }
    if row_case_ids != set(ids):
        raise ValueError("decision-row lifecycle IDs differ from native trade cases")

    summary_groups = case_summary.get("groups", [])
    if sum(group.get("total_cases", 0) for group in summary_groups) != len(cases):
        raise ValueError("case summary count differs from native cases")
    for case_field, account_field in (
        ("fees", "fees"), ("slippage_cost", "slippage_cost"), ("funding", "funding"),
    ):
        case_total = sum(_finite(case[case_field], f"case.{case_field}") for case in cases)
        account_total = _finite(report["account"][account_field], f"account.{account_field}")
        if abs(case_total - account_total) > 1e-6:
            raise ValueError(f"case {case_field} does not reconcile to native account total")
    net_pnl = sum(_finite(case["realized_pnl"], "case.realized_pnl") for case in cases)
    equity_delta = (_finite(report["account"]["equity_end"], "equity_end")
                    - _finite(report["account"]["equity_start"], "equity_start"))
    if abs(net_pnl - equity_delta) > 1e-6:
        raise ValueError("native case PnL does not reconcile to account equity change")
    if funnel.get("economic_classifications") != ["SCENARIO_ONLY"]:
        raise ValueError("funnel classification is not scenario-only")
    expected_raw_long = sum(row["raw_signal"] == "LONG_ENTRY" for row in rows)
    expected_raw_short = sum(row["raw_signal"] == "SHORT_ENTRY" for row in rows)
    expected_exits = sum(row["raw_signal"] in {"EXIT_LONG", "EXIT_SHORT"} for row in rows)
    expected_eligible = sum(
        row["raw_signal"] in {"LONG_ENTRY", "SHORT_ENTRY"}
        and abs(float(row["position_before"])) <= 1e-8
        and row["bar_status"] == "READY"
        and not row["risk_veto_reason"]
        and not row["order_state_veto_reason"]
        for row in rows
    )
    funnel_groups = funnel.get("groups", [])
    if (sum(group["raw_long_signals"] for group in funnel_groups) != expected_raw_long
            or sum(group["raw_short_signals"] for group in funnel_groups) != expected_raw_short
            or sum(group["exit_signals"] for group in funnel_groups) != expected_exits
            or sum(group["entry_signals_while_flat"] for group in funnel_groups)
            != expected_eligible):
        raise ValueError("teacher funnel counts do not reconcile to per-minute raw decisions/vetoes")
    return {
        "status": ("VERIFIED_SCENARIO_ONLY_SMOKE" if is_smoke
                   else "VERIFIED_SCENARIO_ONLY_RESEARCH_BLOCK"),
        "decision_rows": len(rows),
        "m22_verified": decisions["m22_verified"],
        "native_cases": len(cases),
        "ordinary_complete": len(ordinary_case_ids),
        "terminal_censored": len(terminal_case_ids),
        "bar_status_counts": dict(sorted(Counter(row["bar_status"] for row in rows).items())),
        "ready_decision_bars": sum(row["bar_status"] == "READY" for row in rows),
        "warmup_decision_bars": sum(row["bar_status"] == "WARMUP" for row in rows),
        "raw_signals": dict(sorted(Counter(row["raw_signal"] for row in rows).items())),
        "planner_intents": dict(sorted(Counter(row["planner_intent"] for row in rows).items())),
        "planner_statuses": dict(sorted(Counter(row["planner_status"] for row in rows).items())),
        "risk_vetoes": dict(sorted(Counter(
            row["risk_veto_reason"] for row in rows if row["risk_veto_reason"]
        ).items())),
        "order_state_vetoes": dict(sorted(Counter(
            row["order_state_veto_reason"] for row in rows if row["order_state_veto_reason"]
        ).items())),
        "account_return": report["account"]["net_return"],
        "economic_status": "SCENARIO_ONLY",
        "qualification": "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2",
    }


def verify_native_smoke(output: Path) -> dict[str, Any]:
    """Keep the S06 smoke verifier API stable for its existing tests."""
    if not (output / "native-smoke.json").is_file():
        raise ValueError("bounded S06 smoke report is missing")
    return verify_native_run(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_native_run(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
