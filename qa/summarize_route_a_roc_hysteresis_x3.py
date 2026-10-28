"""Compare cost-aware ROC X1 with its X3 profit-protection overlay."""
from __future__ import annotations

import json
from math import prod
from pathlib import Path
from typing import Any

from analyze_route_a_x1_x2_exit_economics import (
    _add_excursion_metrics,
    _economics,
    _load_cases,
    _pair_identity,
    _quantile,
    _round_trip_cost,
    _same_entry,
    _read_json,
)

ROOT = Path(__file__).resolve().parents[1]
X1_ROOT = ROOT / ".artifacts/route-a-roc-hysteresis-task2/native"
X3_ROOT = ROOT / ".artifacts/route-a-x3-profit-protect-task3/native"
OUTPUT = ROOT / ".artifacts/route-a-x3-profit-protect-task3/paired_x1_x3_results.json"
BLOCKS = ("inner-screen-1", "inner-screen-2", "inner-screen-3")
SCENARIOS = ("base", "stress")


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


def run() -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "route-a.roc-hysteresis-x1-x3-native-comparison.v1",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "source_use": "previously consumed S06 Train windows; no new market read",
        "baseline_commit": "bd313c5241c31f1ccff96fa4bf76a7966be48767",
        "x3_recipe": {
            "policy_id": "X3_ATR_COST_PROFIT_PROTECT_V1",
            "round_trip_cost_bps": "scenario-derived: Base 12, Stress 30",
            "activation": "favorable excursion reaches 2x round-trip cost",
            "locked_floor": "1.5x round-trip cost beyond native average open price",
            "atr_trail": "ratchet at 3x current ATR beyond favorable extreme",
            "fallback": "original cost-aware ROC hysteresis exit remains active",
        },
        "pairs": [],
        "scenario_summary": {},
    }
    by_scenario: dict[str, list[dict[str, Any]]] = {scenario: [] for scenario in SCENARIOS}
    for block in BLOCKS:
        for scenario in SCENARIOS:
            old_dir = X1_ROOT / block / scenario
            new_dir = X3_ROOT / block / scenario
            old = _read_json(old_dir / "native-run.json")
            new = _read_json(new_dir / "native-run.json")
            identity = _pair_identity(old, new)
            for native in (old, new):
                decisions = native["decisions"]
                if decisions["observed"] != decisions["m22_verified"]:
                    raise RuntimeError("X1/X3 native decisions were not all M2.2 verified")
                if float(native["account"]["liquidations"]) != 0:
                    raise RuntimeError("liquidation found in ROC X1/X3 diagnostic pair")
            cost_bps = _round_trip_cost(old)
            old_cases = _add_excursion_metrics(old_dir, _load_cases(old_dir))
            new_cases = _add_excursion_metrics(new_dir, _load_cases(new_dir))
            old_econ = _economics(old, old_cases, cost_bps)
            new_econ = _economics(new, new_cases, cost_bps)
            if not old_econ["case_to_account_reconciles_within_0_02"]:
                raise RuntimeError("ROC X1 cases do not reconcile to native account")
            if not new_econ["case_to_account_reconciles_within_0_02"]:
                raise RuntimeError("ROC X3 cases do not reconcile to native account")
            pair = {
                "block": block,
                "scenario": scenario,
                "round_trip_cost_bps": cost_bps,
                "pair_identity": identity,
                "x1": {
                    "account": old["account"],
                    "decisions": old["decisions"],
                    "trade_metrics": old_econ,
                    "teacher_funnel": _funnel(
                        old_dir / "teacher-research-1m/teacher-funnel.json",
                    ),
                },
                "x3": {
                    "account": new["account"],
                    "decisions": new["decisions"],
                    "teacher_recipe": new["teacher_recipe"],
                    "trade_metrics": new_econ,
                    "teacher_funnel": _funnel(
                        new_dir / "teacher-research-1m/teacher-funnel.json",
                    ),
                },
                "paired_delta": {
                    "net_return_percentage_points": (
                        float(new["account"]["net_return"])
                        - float(old["account"]["net_return"])
                    ) * 100,
                    "max_drawdown_percentage_points": (
                        float(new["account"]["max_drawdown"])
                        - float(old["account"]["max_drawdown"])
                    ) * 100,
                    "fee_delta": float(new["account"]["fees"])
                    - float(old["account"]["fees"]),
                    "slippage_attribution_delta": float(new["account"]["slippage_cost"])
                    - float(old["account"]["slippage_cost"]),
                    "funding_delta": float(new["account"]["funding"])
                    - float(old["account"]["funding"]),
                    "complete_trade_delta": (
                        new_econ["complete_trade_count"] - old_econ["complete_trade_count"]
                    ),
                },
                "same_entry_counterfactual": _same_entry(old_cases, new_cases),
            }
            result["pairs"].append(pair)
            by_scenario[scenario].append(pair)
    for scenario, rows in by_scenario.items():
        old_returns = [float(row["x1"]["account"]["net_return"]) for row in rows]
        new_returns = [float(row["x3"]["account"]["net_return"]) for row in rows]
        result["scenario_summary"][scenario] = {
            "x1_block_returns": old_returns,
            "x3_block_returns": new_returns,
            "x1_chain_index": prod(1 + value for value in old_returns) - 1,
            "x3_chain_index": prod(1 + value for value in new_returns) - 1,
            "all_x3_blocks_positive": all(value > 0 for value in new_returns),
            "x3_improved_every_block": all(
                row["paired_delta"]["net_return_percentage_points"] > 0 for row in rows
            ),
            "x3_improved_drawdown_every_block": all(
                row["paired_delta"]["max_drawdown_percentage_points"] < 0 for row in rows
            ),
            "median_same_entry_return_delta_bps": _quantile([
                float(
                    row["same_entry_counterfactual"]
                    ["net_case_return_bps_delta_new_minus_old"]["median"]
                )
                for row in rows
                if row["same_entry_counterfactual"][
                    "net_case_return_bps_delta_new_minus_old"]["median"] is not None
            ], 0.5),
        }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps(report["scenario_summary"], indent=2, sort_keys=True))
