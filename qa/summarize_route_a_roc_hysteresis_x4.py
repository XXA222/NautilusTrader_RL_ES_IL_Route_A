"""Pair ROC X4 opposite-consensus rearm against prior X1 and X3 native runs."""
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
    _pair_identity,
    _quantile,
    _read_json,
    _round_trip_cost,
    _same_entry,
)

ROOT = Path(__file__).resolve().parents[1]
X1_ROOT = ROOT / ".artifacts/route-a-roc-hysteresis-task2/native"
X3_ROOT = ROOT / ".artifacts/route-a-x3-profit-protect-task3/native"
X4_ROOT = ROOT / ".artifacts/route-a-x4-opposite-rearm-task3/native"
OUTPUT = ROOT / ".artifacts/route-a-x4-opposite-rearm-task3/paired_x1_x3_x4_results.json"
BLOCKS = ("inner-screen-1", "inner-screen-2", "inner-screen-3")
SCENARIOS = ("base", "stress")


def _x3_x4_identity(x3: dict[str, Any], x4: dict[str, Any]) -> dict[str, bool]:
    x3_recipe, x4_recipe = x3["teacher_recipe"], x4["teacher_recipe"]
    x3_parameters = dict(x3_recipe["parameters"])
    x4_parameters = dict(x4_recipe["parameters"])
    x4_mode = x4_parameters.pop("profit_lock_rearm_mode", None)
    checks = {
        "same_teacher_and_market_indicator": (
            x3["teacher"] == x4["teacher"]
            and x3_recipe["indicator_recipe_id"] == x4_recipe["indicator_recipe_id"]
        ),
        "same_exposure_and_higher_timeframe_mode": (
            x3_recipe["target_exposure"] == x4_recipe["target_exposure"]
            and x3_recipe["higher_timeframe_mode"] == x4_recipe["higher_timeframe_mode"]
        ),
        "same_source_snapshot": (
            x3["source"]["snapshot_hash"] == x4["source"]["snapshot_hash"]
            and x3["source"]["marker_sha256"] == x4["source"]["marker_sha256"]
        ),
        "same_registered_window": x3["source"]["window_ns"] == x4["source"]["window_ns"],
        "same_scaler": (
            x3["execution"]["scaler_hash"] == x4["execution"]["scaler_hash"]
            and x3["execution"]["scaler_fit_window_hash"]
            == x4["execution"]["scaler_fit_window_hash"]
        ),
        "same_account_risk_and_costs": all(
            x3["execution"][key] == x4["execution"][key]
            for key in ("starting_balance", "risk_budget_notional", "max_leverage", "scenario_costs")
        ),
        "same_protocol_and_preregistration": (
            x3["protocol_id"] == x4["protocol_id"]
            and x3["preregistration_sha256"] == x4["preregistration_sha256"]
        ),
        "same_underlying_roc_parameters": x3_parameters == x4_parameters,
        "expected_opposite_consensus_rearm": (
            x4_mode == "OPPOSITE_CONSENSUS_V1"
            and x4_recipe["entry_rule_id"]
            == "ROC_COST_AWARE_CONSENSUS_HYSTERESIS_OPPOSITE_REARM_V1"
        ),
        "expected_cost_profit_lock_policy": (
            x3_recipe["exit_policy_id"] == "X3_ATR_COST_PROFIT_PROTECT_V1"
            and x4_recipe["exit_policy_id"]
            == "X4_ATR_COST_PROFIT_LOCK_OPPOSITE_REARM_V1"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"X3/X4 native pair identity failed: {checks}")
    return checks


def _funnel(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    return {
        "groups": [
            {
                "side": row["side"],
                "raw_long_signals": row["raw_long_signals"],
                "raw_short_signals": row["raw_short_signals"],
                "filled_entries": row["filled_entries"],
                "filled_exits": row["filled_exits"],
                "terminal_censored_cases": row["terminal_censored_cases"],
                "unfilled_or_missed_cases": row["unfilled_or_missed_cases"],
                "veto_counts": row["veto_counts"],
            }
            for row in value["groups"] if row["side"] != "FLAT"
        ],
        "economic_classifications": value["economic_classifications"],
    }


def _unmatched_case_summary(
    old_cases: list[dict[str, Any]], new_cases: list[dict[str, Any]],
) -> dict[str, float | int]:
    old_keys = {
        _entry_key(case) for case in old_cases if case["status"] == "ORDINARY_COMPLETE"
    }
    unmatched = [
        case for case in new_cases
        if case["status"] == "ORDINARY_COMPLETE" and _entry_key(case) not in old_keys
    ]
    new_keys = {
        _entry_key(case) for case in new_cases if case["status"] == "ORDINARY_COMPLETE"
    }
    no_new_pair = [
        case for case in old_cases
        if case["status"] == "ORDINARY_COMPLETE" and _entry_key(case) not in new_keys
    ]
    pnls = [float(case["realized_pnl"]) for case in unmatched]
    avoided_pnls = [float(case["realized_pnl"]) for case in no_new_pair]
    return {
        "new_unmatched_complete_cases": len(unmatched),
        "new_unmatched_net_pnl": sum(pnls),
        "new_unmatched_winners": sum(value > 0 for value in pnls),
        "new_unmatched_fees": sum(float(case["fees"]) for case in unmatched),
        "old_unmatched_complete_cases": len(no_new_pair),
        "old_unmatched_net_pnl": sum(avoided_pnls),
        "old_unmatched_winners": sum(value > 0 for value in avoided_pnls),
        "old_unmatched_fees": sum(float(case["fees"]) for case in no_new_pair),
    }


def _delta(new: dict[str, Any], old: dict[str, Any]) -> dict[str, float | int]:
    return {
        "net_return_percentage_points": (
            float(new["account"]["net_return"]) - float(old["account"]["net_return"])
        ) * 100,
        "max_drawdown_percentage_points": (
            float(new["account"]["max_drawdown"])
            - float(old["account"]["max_drawdown"])
        ) * 100,
        "fee_delta": float(new["account"]["fees"]) - float(old["account"]["fees"]),
        "slippage_delta": (
            float(new["account"]["slippage_cost"])
            - float(old["account"]["slippage_cost"])
        ),
        "funding_delta": (
            float(new["account"]["funding"]) - float(old["account"]["funding"])
        ),
    }


def run() -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "route-a.roc-hysteresis-x1-x3-x4-native-comparison.v1",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "source_use": "previously consumed S06 Train windows; no new market read",
        "baseline_commit": "bd313c5241c31f1ccff96fa4bf76a7966be48767",
        "x4_rearm_rule": (
            "after a native X3-style cost-profit-lock exit, wait until flat with no pending "
            "orders, then require two consecutive opposite ROC20/ROC60 consensus bars"
        ),
        "pairs": [],
        "scenario_summary": {},
    }
    by_scenario: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in SCENARIOS
    }
    for block in BLOCKS:
        for scenario in SCENARIOS:
            x1_dir, x3_dir, x4_dir = (
                root / block / scenario for root in (X1_ROOT, X3_ROOT, X4_ROOT)
            )
            x1 = _read_json(x1_dir / "native-run.json")
            x3 = _read_json(x3_dir / "native-run.json")
            x4 = _read_json(x4_dir / "native-run.json")
            x1_x3_identity = _pair_identity(x1, x3)
            identity = _x3_x4_identity(x3, x4)
            if x1["teacher_recipe"]["parameters"] != x3["teacher_recipe"]["parameters"]:
                raise RuntimeError("X1/X3 underlying ROC entry parameters differ")
            for native in (x1, x3, x4):
                decisions = native["decisions"]
                if decisions["observed"] != decisions["m22_verified"]:
                    raise RuntimeError("native decisions were not all M2.2 verified")
                if float(native["account"]["liquidations"]) != 0:
                    raise RuntimeError("liquidation found in ROC exit comparison")
            cost_bps = _round_trip_cost(x1)
            cases_by_variant = {
                "x1": _add_excursion_metrics(x1_dir, _load_cases(x1_dir)),
                "x3": _add_excursion_metrics(x3_dir, _load_cases(x3_dir)),
                "x4": _add_excursion_metrics(x4_dir, _load_cases(x4_dir)),
            }
            economics = {
                variant: _economics(native, cases_by_variant[variant], cost_bps)
                for variant, native in (("x1", x1), ("x3", x3), ("x4", x4))
            }
            for variant, econ in economics.items():
                if not econ["case_to_account_reconciles_within_0_02"]:
                    raise RuntimeError(f"ROC {variant.upper()} cases do not reconcile to account")
            pair = {
                "block": block,
                "scenario": scenario,
                "round_trip_cost_bps": cost_bps,
                "x1_x3_pair_identity": x1_x3_identity,
                "x3_x4_pair_identity": identity,
                "x1": {
                    "account": x1["account"],
                    "decisions": x1["decisions"],
                    "trade_metrics": economics["x1"],
                    "teacher_funnel": _funnel(x1_dir / "teacher-research-1m/teacher-funnel.json"),
                },
                "x3": {
                    "account": x3["account"],
                    "trade_metrics": economics["x3"],
                    "teacher_funnel": _funnel(x3_dir / "teacher-research-1m/teacher-funnel.json"),
                },
                "x4": {
                    "account": x4["account"],
                    "decisions": x4["decisions"],
                    "teacher_recipe": x4["teacher_recipe"],
                    "trade_metrics": economics["x4"],
                    "teacher_funnel": _funnel(x4_dir / "teacher-research-1m/teacher-funnel.json"),
                },
                "x4_minus_x3": _delta(x4, x3),
                "x4_minus_x1": _delta(x4, x1),
                "same_entry_x4_vs_x3": _same_entry(cases_by_variant["x3"], cases_by_variant["x4"]),
                "x4_new_unmatched_vs_x3": _unmatched_case_summary(
                    cases_by_variant["x3"], cases_by_variant["x4"],
                ),
            }
            result["pairs"].append(pair)
            by_scenario[scenario].append(pair)
    for scenario, rows in by_scenario.items():
        x1_returns = [float(row["x1"]["account"]["net_return"]) for row in rows]
        x3_returns = [float(row["x3"]["account"]["net_return"]) for row in rows]
        x4_returns = [float(row["x4"]["account"]["net_return"]) for row in rows]
        result["scenario_summary"][scenario] = {
            "x1_block_returns": x1_returns,
            "x3_block_returns": x3_returns,
            "x4_block_returns": x4_returns,
            "x1_chain_index": prod(1 + value for value in x1_returns) - 1,
            "x3_chain_index": prod(1 + value for value in x3_returns) - 1,
            "x4_chain_index": prod(1 + value for value in x4_returns) - 1,
            "x4_improved_every_x3_block": all(
                row["x4_minus_x3"]["net_return_percentage_points"] > 0 for row in rows
            ),
            "x4_reduced_drawdown_every_x3_block": all(
                row["x4_minus_x3"]["max_drawdown_percentage_points"] < 0 for row in rows
            ),
            "x4_new_unmatched_vs_x3_total_pnl": sum(
                float(row["x4_new_unmatched_vs_x3"]["new_unmatched_net_pnl"])
                for row in rows
            ),
            "median_same_entry_return_delta_bps_x4_minus_x3": _quantile([
                float(row["same_entry_x4_vs_x3"][
                    "net_case_return_bps_delta_new_minus_old"]["median"]
                )
                for row in rows
                if row["same_entry_x4_vs_x3"][
                    "net_case_return_bps_delta_new_minus_old"]["median"] is not None
            ], 0.5),
        }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps(report["scenario_summary"], indent=2, sort_keys=True))
