"""Pair the X5 cost/ATR loss guard against X4 using native S06 evidence."""
from __future__ import annotations

import json
from math import prod
from pathlib import Path
from typing import Any

from analyze_route_a_x1_x2_exit_economics import (
    _add_excursion_metrics,
    _economics,
    _entry_key,
    _load_cases,
    _quantile,
    _read_json,
    _round_trip_cost,
    _same_entry,
)

ROOT = Path(__file__).resolve().parents[1]
X4_ROOT = ROOT / ".artifacts/route-a-x4-opposite-rearm-task3/native"
X5_ROOT = ROOT / ".artifacts/route-a-x5-cost-atrloss-task3/native"
OUTPUT = ROOT / ".artifacts/route-a-x5-cost-atrloss-task3/paired_x4_x5_results.json"
BLOCKS = ("inner-screen-1", "inner-screen-2", "inner-screen-3")
SCENARIOS = ("base", "stress")


def _pair_identity(x4: dict[str, Any], x5: dict[str, Any]) -> dict[str, bool]:
    x4_recipe, x5_recipe = x4["teacher_recipe"], x5["teacher_recipe"]
    x4_parameters = dict(x4_recipe["parameters"])
    x5_parameters = dict(x5_recipe["parameters"])
    x4_mode = x4_parameters.pop("profit_lock_rearm_mode", None)
    x5_mode = x5_parameters.pop("cost_exit_rearm_mode", None)
    checks = {
        "same_teacher_and_indicator": (
            x4["teacher"] == x5["teacher"]
            and x4_recipe["indicator_recipe_id"] == x5_recipe["indicator_recipe_id"]
        ),
        "same_exposure_and_higher_timeframe_mode": (
            x4_recipe["target_exposure"] == x5_recipe["target_exposure"]
            and x4_recipe["higher_timeframe_mode"] == x5_recipe["higher_timeframe_mode"]
        ),
        "same_source_snapshot": (
            x4["source"]["snapshot_hash"] == x5["source"]["snapshot_hash"]
            and x4["source"]["marker_sha256"] == x5["source"]["marker_sha256"]
        ),
        "same_registered_window": x4["source"]["window_ns"] == x5["source"]["window_ns"],
        "same_scaler": (
            x4["execution"]["scaler_hash"] == x5["execution"]["scaler_hash"]
            and x4["execution"]["scaler_fit_window_hash"]
            == x5["execution"]["scaler_fit_window_hash"]
        ),
        "same_account_risk_and_costs": all(
            x4["execution"][key] == x5["execution"][key]
            for key in ("starting_balance", "risk_budget_notional", "max_leverage", "scenario_costs")
        ),
        "same_protocol_and_preregistration": (
            x4["protocol_id"] == x5["protocol_id"]
            and x4["preregistration_sha256"] == x5["preregistration_sha256"]
        ),
        "same_underlying_roc_parameters": x4_parameters == x5_parameters,
        "expected_rearm_modes": (
            x4_mode == "OPPOSITE_CONSENSUS_V1"
            and x5_mode == "OPPOSITE_CONSENSUS_V1"
        ),
        "expected_exit_policies": (
            x4_recipe["exit_policy_id"] == "X4_ATR_COST_PROFIT_LOCK_OPPOSITE_REARM_V1"
            and x5_recipe["exit_policy_id"] == "X5_ATR_COST_PROFIT_AND_LOSS_GUARD_V1"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"X4/X5 native pair identity failed: {checks}")
    return checks


def _unmatched_summary(
    old_cases: list[dict[str, Any]], new_cases: list[dict[str, Any]],
) -> dict[str, float | int]:
    old_keys = {
        _entry_key(case) for case in old_cases if case["status"] == "ORDINARY_COMPLETE"
    }
    new_keys = {
        _entry_key(case) for case in new_cases if case["status"] == "ORDINARY_COMPLETE"
    }
    new_unmatched = [
        case for case in new_cases
        if case["status"] == "ORDINARY_COMPLETE" and _entry_key(case) not in old_keys
    ]
    old_unmatched = [
        case for case in old_cases
        if case["status"] == "ORDINARY_COMPLETE" and _entry_key(case) not in new_keys
    ]
    new_pnls = [float(case["realized_pnl"]) for case in new_unmatched]
    old_pnls = [float(case["realized_pnl"]) for case in old_unmatched]
    return {
        "x5_unmatched_count": len(new_unmatched),
        "x5_unmatched_net_pnl": sum(new_pnls),
        "x5_unmatched_winners": sum(pnl > 0 for pnl in new_pnls),
        "x5_unmatched_fees": sum(float(case["fees"]) for case in new_unmatched),
        "x4_unmatched_count": len(old_unmatched),
        "x4_unmatched_net_pnl": sum(old_pnls),
        "x4_unmatched_winners": sum(pnl > 0 for pnl in old_pnls),
        "x4_unmatched_fees": sum(float(case["fees"]) for case in old_unmatched),
    }


def _delta(new: dict[str, Any], old: dict[str, Any]) -> dict[str, float]:
    return {
        "net_return_percentage_points": (
            float(new["account"]["net_return"]) - float(old["account"]["net_return"])
        ) * 100,
        "max_drawdown_percentage_points": (
            float(new["account"]["max_drawdown"])
            - float(old["account"]["max_drawdown"])
        ) * 100,
        "fees": float(new["account"]["fees"]) - float(old["account"]["fees"]),
        "slippage": (
            float(new["account"]["slippage_cost"])
            - float(old["account"]["slippage_cost"])
        ),
        "funding": float(new["account"]["funding"]) - float(old["account"]["funding"]),
    }


def run() -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "route-a.roc-hysteresis-x4-x5-native-comparison.v1",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "source_use": "previously consumed S06 Train windows; no new market read",
        "baseline_commit": "bd313c5241c31f1ccff96fa4bf76a7966be48767",
        "x5_loss_guard": (
            "before profit-lock activation, exit at a completed-bar close adverse by "
            "the wider of 2x scenario round-trip cost or 2x current ATR"
        ),
        "pairs": [],
        "scenario_summary": {},
    }
    by_scenario: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in SCENARIOS
    }
    for block in BLOCKS:
        for scenario in SCENARIOS:
            x4_dir = X4_ROOT / block / scenario
            x5_dir = X5_ROOT / block / scenario
            x4 = _read_json(x4_dir / "native-run.json")
            x5 = _read_json(x5_dir / "native-run.json")
            identity = _pair_identity(x4, x5)
            for native in (x4, x5):
                if native["decisions"]["observed"] != native["decisions"]["m22_verified"]:
                    raise RuntimeError("native decisions were not all M2.2 verified")
                if float(native["account"]["liquidations"]) != 0:
                    raise RuntimeError("liquidation found in ROC X4/X5 comparison")
            cost_bps = _round_trip_cost(x4)
            x4_cases = _add_excursion_metrics(x4_dir, _load_cases(x4_dir))
            x5_cases = _add_excursion_metrics(x5_dir, _load_cases(x5_dir))
            x4_economics = _economics(x4, x4_cases, cost_bps)
            x5_economics = _economics(x5, x5_cases, cost_bps)
            if not x4_economics["case_to_account_reconciles_within_0_02"]:
                raise RuntimeError("X4 cases do not reconcile to native account")
            if not x5_economics["case_to_account_reconciles_within_0_02"]:
                raise RuntimeError("X5 cases do not reconcile to native account")
            pair = {
                "block": block,
                "scenario": scenario,
                "round_trip_cost_bps": cost_bps,
                "pair_identity": identity,
                "x4": {
                    "account": x4["account"],
                    "decisions": x4["decisions"],
                    "trade_metrics": x4_economics,
                },
                "x5": {
                    "account": x5["account"],
                    "decisions": x5["decisions"],
                    "teacher_recipe": x5["teacher_recipe"],
                    "trade_metrics": x5_economics,
                },
                "x5_minus_x4": _delta(x5, x4),
                "same_entry_x5_vs_x4": _same_entry(x4_cases, x5_cases),
                "unmatched_x5_vs_x4": _unmatched_summary(x4_cases, x5_cases),
            }
            result["pairs"].append(pair)
            by_scenario[scenario].append(pair)
    for scenario, rows in by_scenario.items():
        x4_returns = [float(row["x4"]["account"]["net_return"]) for row in rows]
        x5_returns = [float(row["x5"]["account"]["net_return"]) for row in rows]
        result["scenario_summary"][scenario] = {
            "x4_block_returns": x4_returns,
            "x5_block_returns": x5_returns,
            "x4_chain_index": prod(1 + value for value in x4_returns) - 1,
            "x5_chain_index": prod(1 + value for value in x5_returns) - 1,
            "x5_improved_every_x4_block": all(
                row["x5_minus_x4"]["net_return_percentage_points"] > 0 for row in rows
            ),
            "x5_reduced_drawdown_every_x4_block": all(
                row["x5_minus_x4"]["max_drawdown_percentage_points"] < 0 for row in rows
            ),
            "median_same_entry_net_return_delta_bps": _quantile([
                float(row["same_entry_x5_vs_x4"][
                    "net_case_return_bps_delta_new_minus_old"]["median"]
                )
                for row in rows
                if row["same_entry_x5_vs_x4"][
                    "net_case_return_bps_delta_new_minus_old"]["median"] is not None
            ], 0.5),
            "x5_unmatched_total_net_pnl": sum(
                float(row["unmatched_x5_vs_x4"]["x5_unmatched_net_pnl"])
                for row in rows
            ),
            "x4_unmatched_total_net_pnl": sum(
                float(row["unmatched_x5_vs_x4"]["x4_unmatched_net_pnl"])
                for row in rows
            ),
        }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps(report["scenario_summary"], indent=2, sort_keys=True))
