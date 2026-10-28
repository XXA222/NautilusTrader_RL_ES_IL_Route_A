"""Compare corrected X2 behavior with a hash-exact pre-fix native control."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RUN = ROOT / (
    ".artifacts/research_1m_4teachers_v1/native/teachers/"
    "T01_EMA_9_21/X2/inner-screen-1/base/native-run.json"
)
PRE_FIX_RUN = ROOT / (
    ".artifacts/route-a-x2-snapshot-fix-task3/native/"
    "ema-x2-base-inner-screen-1-pre-fix/native-run.json"
)
FIXED_RUN = ROOT / (
    ".artifacts/route-a-x2-snapshot-fix-task3/native/"
    "ema-x2-base-inner-screen-1/native-run.json"
)
HISTORICAL_CASES = HISTORICAL_RUN.parent / "teacher-research-1m/native-trades.jsonl"
PRE_FIX_CASES = PRE_FIX_RUN.parent / "teacher-research-1m/native-trades.jsonl"
FIXED_CASES = FIXED_RUN.parent / "teacher-research-1m/native-trades.jsonl"
HISTORICAL_DECISIONS = HISTORICAL_RUN.parent / "teacher-research-1m/decision-rows.csv"
PRE_FIX_DECISIONS = PRE_FIX_RUN.parent / "teacher-research-1m/decision-rows.csv"
FIXED_DECISIONS = FIXED_RUN.parent / "teacher-research-1m/decision-rows.csv"
CONTROL_RECEIPT = ROOT / (
    ".artifacts/route-a-x2-snapshot-fix-task3/controls/pre_fix/overlay-receipt.json"
)
FIXED_BASE = ROOT / "packages/route_a_imitation/src/rles_route_a/teachers_1m/base.py"
OUTPUT = ROOT / ".artifacts/route-a-x2-snapshot-fix-task3/comparison.json"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def _read_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_decisions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _case_summary(cases: list[dict[str, Any]], account_delta: float) -> dict[str, Any]:
    status_counts = Counter(str(row["status"]) for row in cases)
    side_cases = {
        side: [
            row for row in cases
            if ("LONG" if int(row["side"]) > 0 else "SHORT") == side
        ]
        for side in ("LONG", "SHORT")
    }
    side_pnl = {
        side: sum(float(row["realized_pnl"]) for row in rows)
        for side, rows in side_cases.items()
    }
    side_cashflow = {
        side: {
            "net_pnl_after_costs": side_pnl[side],
            "fees": sum(float(row["fees"]) for row in rows),
            "funding": sum(float(row["funding"]) for row in rows),
            "slippage_attribution": sum(float(row["slippage_cost"]) for row in rows),
            "price_pnl_at_actual_fill_before_fees_and_funding": (
                side_pnl[side]
                + sum(float(row["fees"]) for row in rows)
                - sum(float(row["funding"]) for row in rows)
            ),
            "estimated_price_pnl_before_slippage_fees_and_funding": (
                side_pnl[side]
                + sum(float(row["fees"]) for row in rows)
                + sum(float(row["slippage_cost"]) for row in rows)
                - sum(float(row["funding"]) for row in rows)
            ),
        }
        for side, rows in side_cases.items()
    }
    exit_counts = Counter(str(row.get("exit_reason") or "UNKNOWN") for row in cases)
    reported_pnl = sum(float(row["realized_pnl"]) for row in cases)
    fees = sum(float(row["fees"]) for row in cases)
    funding = sum(float(row["funding"]) for row in cases)
    slippage = sum(float(row["slippage_cost"]) for row in cases)
    return {
        "case_count": len(cases),
        "status_counts": dict(sorted(status_counts.items())),
        "exit_reason_counts": dict(sorted(exit_counts.items())),
        "case_reported_pnl": reported_pnl,
        "case_to_account_residual": reported_pnl - account_delta,
        "case_to_account_reconciles_within_0_02": abs(reported_pnl - account_delta) <= 0.02,
        "realized_pnl_by_side": side_pnl,
        "cashflow_attribution": {
            "net_pnl_after_costs": reported_pnl,
            "fees": fees,
            "funding": funding,
            "slippage_attribution": slippage,
            "price_pnl_at_actual_fill_before_fees_and_funding": (
                reported_pnl + fees - funding
            ),
            "estimated_price_pnl_before_slippage_fees_and_funding": (
                reported_pnl + fees + slippage - funding
            ),
            "slippage_treatment": (
                "Slippage attribution is already embedded in native fill prices; it is added back "
                "only for the estimated pre-slippage price-PnL diagnostic."
            ),
        },
        "cashflow_attribution_by_side": side_cashflow,
    }


def _entry_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["entry_decision_ts_ns"]),
        int(row["entry_fill_ts_ns"]),
        int(row["side"]),
    )


def _entry_cases_match_by_economics(
    left: list[dict[str, Any]], right: list[dict[str, Any]],
) -> dict[str, Any]:
    left_by_key = {_entry_key(row): row for row in left}
    right_by_key = {_entry_key(row): row for row in right}
    common = set(left_by_key) & set(right_by_key)
    fields = (
        "status", "entry_fill_price", "entry_notional", "exit_fill_price",
        "realized_pnl", "fees", "funding", "slippage_cost", "holding_minutes",
        "mfe_bps", "mae_bps", "exit_reason",
    )
    mismatches = {
        field: sum(left_by_key[key].get(field) != right_by_key[key].get(field) for key in common)
        for field in fields
    }
    return {
        "left_case_count": len(left),
        "right_case_count": len(right),
        "common_entry_count": len(common),
        "economic_field_mismatch_counts": mismatches,
        "same_trade_case_economics": (
            len(left_by_key) == len(left)
            and len(right_by_key) == len(right)
            and len(common) == len(left) == len(right)
            and not any(mismatches.values())
        ),
    }


def _decision_path_diagnostic(
    control: list[dict[str, str]], fixed: list[dict[str, str]],
) -> dict[str, Any]:
    if len(control) != len(fixed):
        raise ValueError("control and fixed decision counts differ")
    compare_fields = (
        "raw_signal", "raw_signal_reason", "position_before", "planner_intent",
        "requested_exposure", "risk_adjusted_target", "planner_target",
        "planner_status", "planner_reason", "native_exposure_after", "exit_reason",
    )
    differences = [
        index for index, (before, after) in enumerate(zip(control, fixed, strict=True))
        if any(before[field] != after[field] for field in compare_fields)
    ]
    indicator_differences = sum(
        before["indicator_values_json"] != after["indicator_values_json"]
        for before, after in zip(control, fixed, strict=True)
    )
    entry_signals = {"LONG_ENTRY", "SHORT_ENTRY"}
    entry_changes = [
        index for index, (before, after) in enumerate(zip(control, fixed, strict=True))
        if before["raw_signal"] != after["raw_signal"]
        and (before["raw_signal"] in entry_signals or after["raw_signal"] in entry_signals)
    ]
    followed_by_atr_exit = [
        index for index in entry_changes
        if index > 0
        and control[index - 1]["exit_reason"].startswith("ATR_TRAIL_EXIT_")
        and fixed[index - 1]["exit_reason"].startswith("ATR_TRAIL_EXIT_")
    ]
    first = differences[0] if differences else None
    first_entry = entry_changes[0] if entry_changes else None

    def describe(index: int | None) -> dict[str, Any] | None:
        if index is None:
            return None
        before, after = control[index], fixed[index]
        previous = control[index - 1] if index > 0 else None
        return {
            "decision_ts_ns": int(before["decision_ts_ns"]),
            "control_signal": before["raw_signal"],
            "control_reason": before["raw_signal_reason"],
            "fixed_signal": after["raw_signal"],
            "fixed_reason": after["raw_signal_reason"],
            "preceding_control_exit_reason": (
                None if previous is None else previous["exit_reason"]
            ),
            "preceding_fixed_exit_reason": (
                None if index == 0 else fixed[index - 1]["exit_reason"]
            ),
        }

    return {
        "decision_rows": len(control),
        "indicator_snapshot_mismatch_rows": indicator_differences,
        "behaviorally_changed_rows": len(differences),
        "entry_signal_change_count": len(entry_changes),
        "entry_signal_changes_immediately_after_same_bar_atr_exit": len(followed_by_atr_exit),
        "first_behavior_change": describe(first),
        "first_entry_signal_change": describe(first_entry),
        "compared_fields": list(compare_fields),
    }


def _overlay_package_check(receipt: dict[str, Any]) -> dict[str, Any]:
    source = ROOT / "packages/route_a_imitation/src/rles_route_a"
    overlay = ROOT / str(receipt["overlay_pythonpath"]) / "rles_route_a"
    source_files = {path.relative_to(source) for path in source.rglob("*.py")}
    overlay_files = {path.relative_to(overlay) for path in overlay.rglob("*.py")}
    if source_files != overlay_files:
        return {
            "same_file_set": False,
            "missing_from_overlay": sorted(map(str, source_files - overlay_files)),
            "extra_in_overlay": sorted(map(str, overlay_files - source_files)),
            "non_base_source_mismatches": None,
        }
    mismatches = [
        str(relative)
        for relative in sorted(source_files)
        if str(relative) != "teachers_1m/base.py"
        and hashlib.sha256((source / relative).read_bytes()).digest()
        != hashlib.sha256((overlay / relative).read_bytes()).digest()
    ]
    control_hash = hashlib.sha256(
        (overlay / "teachers_1m/base.py").read_bytes()
    ).hexdigest()
    return {
        "same_file_set": True,
        "non_base_source_mismatches": mismatches,
        "control_base_sha256": control_hash,
        "non_base_package_matches": not mismatches,
    }


def _matched_completed_entries(
    old_cases: list[dict[str, Any]], new_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    old_rows = [row for row in old_cases if row["status"] == "ORDINARY_COMPLETE"]
    new_rows = [row for row in new_cases if row["status"] == "ORDINARY_COMPLETE"]
    old_by_key = {_entry_key(row): row for row in old_rows}
    new_by_key = {_entry_key(row): row for row in new_rows}
    if len(old_by_key) != len(old_rows) or len(new_by_key) != len(new_rows):
        raise ValueError("duplicate completed trade entry identity")
    common = sorted(set(old_by_key) & set(new_by_key))
    old_unpaired = [old_by_key[key] for key in sorted(set(old_by_key) - set(new_by_key))]
    new_unpaired = [new_by_key[key] for key in sorted(set(new_by_key) - set(old_by_key))]
    shared_economics = _entry_cases_match_by_economics(
        [old_by_key[key] for key in common],
        [new_by_key[key] for key in common],
    )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            field: sum(float(row.get(field) or 0) for row in rows)
            for field in ("realized_pnl", "fees", "funding", "slippage_cost")
        }

    old_unpaired_economics = aggregate(old_unpaired)
    new_unpaired_economics = aggregate(new_unpaired)
    deltas = [
        10_000 * (
            float(new_by_key[key]["realized_pnl"]) / float(new_by_key[key]["entry_notional"])
            - float(old_by_key[key]["realized_pnl"]) / float(old_by_key[key]["entry_notional"])
        )
        for key in common
    ]
    holds = [
        float(new_by_key[key]["holding_minutes"]) - float(old_by_key[key]["holding_minutes"])
        for key in common
    ]
    max_abs_return_delta_bps = max((abs(value) for value in deltas), default=0.0)
    max_abs_holding_delta = max((abs(value) for value in holds), default=0.0)
    return {
        "old_completed_entries": len(old_rows),
        "new_completed_entries": len(new_rows),
        "exact_shared_entry_pairs": len(common),
        "old_completed_entries_without_pair": len(old_rows) - len(common),
        "new_completed_entries_without_pair": len(new_rows) - len(common),
        "shared_entry_case_fields_exactly_identical": (
            shared_economics["same_trade_case_economics"]
        ),
        "shared_entry_exact_field_mismatch_counts": (
            shared_economics["economic_field_mismatch_counts"]
        ),
        "shared_entry_returns_match_within_1e_8_bps": max_abs_return_delta_bps <= 1e-8,
        "shared_entry_holding_minutes_match_exactly": max_abs_holding_delta == 0,
        "shared_entry_max_abs_net_return_delta_bps": max_abs_return_delta_bps,
        "shared_entry_max_abs_holding_minutes_delta": max_abs_holding_delta,
        "old_unpaired_completed_economics": old_unpaired_economics,
        "new_unpaired_completed_economics": new_unpaired_economics,
        "unpaired_completed_economics_delta_new_minus_old": {
            field: new_unpaired_economics[field] - old_unpaired_economics[field]
            for field in old_unpaired_economics
        },
        "net_return_bps_delta_new_minus_old": {
            "mean": statistics.mean(deltas) if deltas else None,
            "median": _quantile(deltas, 0.5),
            "p10": _quantile(deltas, 0.1),
            "p90": _quantile(deltas, 0.9),
            "new_better_count": sum(value > 1e-9 for value in deltas),
            "old_better_count": sum(value < -1e-9 for value in deltas),
            "tie_count": sum(abs(value) <= 1e-9 for value in deltas),
        },
        "holding_minutes_delta_new_minus_old": {
            "mean": statistics.mean(holds) if holds else None,
            "median": _quantile(holds, 0.5),
        },
        "interpretation": (
            "Post-hoc common completed entries only. This isolates each matched case's "
            "realized outcome from later re-entry, but it is incomplete and selected."
        ),
    }


def run(
    pre_fix_run: Path = PRE_FIX_RUN,
    fixed_run: Path = FIXED_RUN,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    pre_fix_run = (pre_fix_run if pre_fix_run.is_absolute() else ROOT / pre_fix_run).resolve()
    fixed_run = (fixed_run if fixed_run.is_absolute() else ROOT / fixed_run).resolve()
    output = (output if output.is_absolute() else ROOT / output).resolve()
    control, fixed = _read_json(pre_fix_run), _read_json(fixed_run)
    control_cases = _read_cases(pre_fix_run.parent / "teacher-research-1m/native-trades.jsonl")
    fixed_cases = _read_cases(fixed_run.parent / "teacher-research-1m/native-trades.jsonl")
    control_decisions = _read_decisions(
        pre_fix_run.parent / "teacher-research-1m/decision-rows.csv"
    )
    fixed_decisions = _read_decisions(
        fixed_run.parent / "teacher-research-1m/decision-rows.csv"
    )
    receipt = _read_json(CONTROL_RECEIPT)
    overlay_check = _overlay_package_check(receipt)
    fixed_base_sha = hashlib.sha256(FIXED_BASE.read_bytes()).hexdigest()

    identity_checks = {
        "same_protocol_and_preregistration": (
            control["protocol_id"] == fixed["protocol_id"]
            and control["preregistration_sha256"] == fixed["preregistration_sha256"]
        ),
        "same_market_snapshot_and_registered_window": all(
            control["source"][key] == fixed["source"][key]
            for key in (
                "snapshot_hash", "marker_sha256", "events",
                "registered_read_scope_ns", "window_id", "window_ns",
            )
        ),
        "same_scenario_account_risk_scaler_and_costs": all(
            control["execution"][key] == fixed["execution"][key]
            for key in (
                "scenario", "scenario_costs", "starting_balance", "risk_budget_notional",
                "max_leverage", "scaler_hash", "scaler_fit_window_hash",
            )
        ),
        "same_teacher_entry_and_exit_recipe": all(
            control["teacher_recipe"].get(key) == fixed["teacher_recipe"].get(key)
            for key in (
                "entry_rule_id", "exit_policy_id", "exit_variant", "indicator_recipe_id",
                "higher_timeframe_mode", "target_exposure", "parameters",
            )
        ) and control["teacher_recipe"].get("ema_reversal_mode", "EXIT_ONLY_V1")
        == fixed["teacher_recipe"].get("ema_reversal_mode", "EXIT_ONLY_V1"),
        "same_complete_m22_decision_count": (
            control["decisions"]["observed"] == control["decisions"]["m22_verified"]
            == fixed["decisions"]["observed"] == fixed["decisions"]["m22_verified"]
        ),
        "zero_liquidations_both_runs": (
            float(control["account"]["liquidations"]) == 0
            and float(fixed["account"]["liquidations"]) == 0
        ),
        "control_source_hash_is_exact_pre_fix": (
            receipt["pre_fix_base_sha256"]
            == hashlib.sha256(
                (ROOT / str(receipt["control_base_path"])).read_bytes()
            ).hexdigest()
        ),
        "fixed_source_hash_matches_post_fix_receipt": (
            fixed_base_sha == receipt["post_fix_base_sha256"]
        ),
        "package_overlay_differs_only_in_teacher_base": (
            overlay_check.get("same_file_set") is True
            and overlay_check.get("non_base_package_matches") is True
        ),
    }
    if not all(identity_checks.values()):
        raise ValueError(f"runs are not sufficiently paired: {identity_checks}")

    control_delta = (
        float(control["account"]["equity_end"])
        - float(control["account"]["equity_start"])
    )
    fixed_delta = (
        float(fixed["account"]["equity_end"])
        - float(fixed["account"]["equity_start"])
    )
    control_case_summary = _case_summary(control_cases, control_delta)
    fixed_case_summary = _case_summary(fixed_cases, fixed_delta)
    if not control_case_summary["case_to_account_reconciles_within_0_02"]:
        raise ValueError("pre-fix control cases do not reconcile to the account")
    if not fixed_case_summary["case_to_account_reconciles_within_0_02"]:
        raise ValueError("corrected run cases do not reconcile to the account")

    historical = _read_json(HISTORICAL_RUN)
    if (
        historical["teacher"] == control["teacher"]
        and historical["source"]["window_id"] == control["source"]["window_id"]
        and historical["execution"]["scenario"] == control["execution"]["scenario"]
        and historical["execution"]["scenario_costs"]
        == control["execution"]["scenario_costs"]
    ):
        historical_cases = _read_cases(HISTORICAL_CASES)
        historical_decisions = _read_decisions(HISTORICAL_DECISIONS)
        hist_to_control_identity = {
            "same_protocol_and_preregistration": (
                historical["protocol_id"] == control["protocol_id"]
                and historical["preregistration_sha256"] == control["preregistration_sha256"]
            ),
            "same_account_and_decision_summary": (
                historical["account"] == control["account"]
                and historical["decisions"] == control["decisions"]
            ),
            "same_market_snapshot_and_window": all(
                historical["source"][key] == control["source"][key]
                for key in (
                    "snapshot_hash", "marker_sha256", "events",
                    "registered_read_scope_ns", "window_id", "window_ns",
                )
            ),
        }
        if not all(hist_to_control_identity.values()):
            raise ValueError(
                "historical X2 output does not match the reconstructed control: "
                f"{hist_to_control_identity}"
            )

        shared_columns = set(historical_decisions[0]) & set(control_decisions[0])
        historical_decision_mismatches = {
            column: sum(
                before[column] != after[column]
                for before, after in zip(historical_decisions, control_decisions, strict=True)
            )
            for column in sorted(shared_columns - {"case_id"})
        }
        hist_case_match = _entry_cases_match_by_economics(historical_cases, control_cases)
        historical_reproduction = {
            "validated": True,
            "run_identity_checks": hist_to_control_identity,
            "decision_row_count_equal": len(historical_decisions) == len(control_decisions),
            "decision_field_mismatch_counts_excluding_run_specific_case_id": (
                historical_decision_mismatches
            ),
            "decisions_match_except_case_ids": not any(
                historical_decision_mismatches.values()
            ),
            "trade_case_economics_match_by_exact_entry": hist_case_match,
            "interpretation": (
                "The preserved historical X2 output reproduces the hash-exact pre-fix control "
                "on the same native decisions and trade economics; only run-specific case ids differ."
            ),
        }
        if (not historical_reproduction["decisions_match_except_case_ids"]
                or not hist_case_match["same_trade_case_economics"]):
            raise ValueError("preserved historical X2 does not reproduce the pre-fix control")
    else:
        historical_reproduction = {
            "validated": False,
            "reason": "The historical output is for a different registered window or scenario.",
        }

    path_diagnostic = _decision_path_diagnostic(control_decisions, fixed_decisions)
    report = {
        "schema": "route-a.x2-snapshot-fix-native-comparison.v1",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "economic_status": "SCENARIO_ONLY",
        "identity_checks": identity_checks,
        "control_provenance": {
            "pre_fix_base_sha256": receipt["pre_fix_base_sha256"],
            "fixed_base_sha256": fixed_base_sha,
            "overlay_package_check": overlay_check,
            "historical_reproduction": historical_reproduction,
        },
        "pre_fix_control_run": str(pre_fix_run.relative_to(ROOT)),
        "fixed_run": str(fixed_run.relative_to(ROOT)),
        "comparison": {
            "pre_fix_account": control["account"],
            "fixed_account": fixed["account"],
            "net_return_percentage_point_delta": (
                float(fixed["account"]["net_return"])
                - float(control["account"]["net_return"])
            ) * 100,
            "equity_delta_usdt": fixed_delta - control_delta,
            "max_drawdown_percentage_point_delta": (
                float(fixed["account"]["max_drawdown"])
                - float(control["account"]["max_drawdown"])
            ) * 100,
            "pre_fix_decisions": control["decisions"],
            "fixed_decisions": fixed["decisions"],
            "pre_fix_trade_cases": control_case_summary,
            "fixed_trade_cases": fixed_case_summary,
            "shared_completed_entry_diagnostic": _matched_completed_entries(
                control_cases, fixed_cases,
            ),
            "decision_path_change": path_diagnostic,
        },
        "interpretation": (
            "The control and fixed runs use the same runner, feature recipe, source snapshot, "
            "registered window, scaler, account, scenario costs, and teacher recipe. The "
            "isolated package overlay differs only in teachers_1m/base.py and restores its "
            "recorded pre-fix hash. Both full-policy account runs remain negative; the fixed "
            "run is a D1 engineering diagnostic, not evidence of teacher qualification or "
            "profitable trading."
        ),
        "qualification": "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-fix-run", type=Path, default=PRE_FIX_RUN)
    parser.add_argument("--fixed-run", type=Path, default=FIXED_RUN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.pre_fix_run, args.fixed_run, args.output)
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))
