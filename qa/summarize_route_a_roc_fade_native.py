"""Compare the cost-aware ROC consensus and its directional fade diagnostic."""
from __future__ import annotations

import json
from math import prod
from pathlib import Path
from typing import Any

from analyze_route_a_x1_x2_exit_economics import (
    _add_excursion_metrics,
    _economics,
    _load_cases,
    _quantile,
    _read_json,
    _round_trip_cost,
    _same_entry,
)

ROOT = Path(__file__).resolve().parents[1]
TREND_ROOT = ROOT / ".artifacts/route-a-roc-hysteresis-task2/native"
FADE_ROOT = ROOT / ".artifacts/route-a-roc-fade-task3/native"
OUTPUT = ROOT / ".artifacts/route-a-roc-fade-task3/paired_trend_vs_fade.json"
BLOCKS = ("inner-screen-1", "inner-screen-2", "inner-screen-3")
SCENARIOS = ("base", "stress")


def _pair_identity(trend: dict[str, Any], fade: dict[str, Any]) -> dict[str, bool]:
    trend_recipe, fade_recipe = trend["teacher_recipe"], fade["teacher_recipe"]
    trend_parameters = dict(trend_recipe["parameters"])
    fade_parameters = dict(fade_recipe["parameters"])
    trend_event = trend_parameters.pop("event", None)
    fade_event = fade_parameters.pop("event", None)
    fade_direction = fade_parameters.pop("direction_mode", None)
    checks = {
        "same_source_snapshot": (
            trend["source"]["snapshot_hash"] == fade["source"]["snapshot_hash"]
            and trend["source"]["marker_sha256"] == fade["source"]["marker_sha256"]
        ),
        "same_registered_window": trend["source"]["window_ns"] == fade["source"]["window_ns"],
        "same_scaler": (
            trend["execution"]["scaler_hash"] == fade["execution"]["scaler_hash"]
            and trend["execution"]["scaler_fit_window_hash"]
            == fade["execution"]["scaler_fit_window_hash"]
        ),
        "same_account_risk_and_costs": all(
            trend["execution"][key] == fade["execution"][key]
            for key in ("starting_balance", "risk_budget_notional", "max_leverage", "scenario_costs")
        ),
        "same_protocol_and_preregistration": (
            trend["protocol_id"] == fade["protocol_id"]
            and trend["preregistration_sha256"] == fade["preregistration_sha256"]
        ),
        "same_indicator_exposure_and_htf": (
            trend_recipe["indicator_recipe_id"] == fade_recipe["indicator_recipe_id"]
            and trend_recipe["target_exposure"] == fade_recipe["target_exposure"]
            and trend_recipe["higher_timeframe_mode"] == fade_recipe["higher_timeframe_mode"]
        ),
        "same_cost_and_volatility_parameters": trend_parameters == fade_parameters,
        "expected_direction_modes": (
            trend_event == "COST_AWARE_HYSTERESIS_V1"
            and fade_event == "COST_AWARE_CONTRARIAN_HYSTERESIS_V1"
            and fade_direction == "CONSENSUS_FADE_V1"
        ),
        "same_x1_exit_rule": (
            trend_recipe["exit_policy_id"] == fade_recipe["exit_policy_id"]
            == "X1_ROC_COST_HYSTERESIS_EXIT_V1"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"trend/fade native pair identity failed: {checks}")
    return checks


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
    report: dict[str, Any] = {
        "schema": "route-a.roc-cost-hysteresis-trend-vs-fade-native.v1",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "source_use": "previously consumed S06 Train windows; no new market read",
        "baseline_commit": "bd313c5241c31f1ccff96fa4bf76a7966be48767",
        "fade_hypothesis": (
            "retain the same consensus and cost/ATR/volatility filters, reverse only the position direction"
        ),
        "pairs": [],
        "scenario_summary": {},
    }
    by_scenario: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in SCENARIOS
    }
    for block in BLOCKS:
        for scenario in SCENARIOS:
            trend_dir = TREND_ROOT / block / scenario
            fade_dir = FADE_ROOT / block / scenario
            trend = _read_json(trend_dir / "native-run.json")
            fade = _read_json(fade_dir / "native-run.json")
            identity = _pair_identity(trend, fade)
            for native in (trend, fade):
                if native["decisions"]["observed"] != native["decisions"]["m22_verified"]:
                    raise RuntimeError("trend/fade decisions were not all M2.2 verified")
                if float(native["account"]["liquidations"]) != 0:
                    raise RuntimeError("liquidation found in trend/fade comparison")
            cost_bps = _round_trip_cost(trend)
            trend_cases = _add_excursion_metrics(trend_dir, _load_cases(trend_dir))
            fade_cases = _add_excursion_metrics(fade_dir, _load_cases(fade_dir))
            trend_economics = _economics(trend, trend_cases, cost_bps)
            fade_economics = _economics(fade, fade_cases, cost_bps)
            if not trend_economics["case_to_account_reconciles_within_0_02"]:
                raise RuntimeError("trend cases do not reconcile to native account")
            if not fade_economics["case_to_account_reconciles_within_0_02"]:
                raise RuntimeError("fade cases do not reconcile to native account")
            pair = {
                "block": block,
                "scenario": scenario,
                "round_trip_cost_bps": cost_bps,
                "pair_identity": identity,
                "trend": {
                    "account": trend["account"],
                    "decisions": trend["decisions"],
                    "teacher_recipe": trend["teacher_recipe"],
                    "trade_metrics": trend_economics,
                },
                "fade": {
                    "account": fade["account"],
                    "decisions": fade["decisions"],
                    "teacher_recipe": fade["teacher_recipe"],
                    "trade_metrics": fade_economics,
                },
                "fade_minus_trend": _delta(fade, trend),
                "exact_same_entry_cases": _same_entry(trend_cases, fade_cases),
            }
            report["pairs"].append(pair)
            by_scenario[scenario].append(pair)
    for scenario, pairs in by_scenario.items():
        trend_returns = [float(pair["trend"]["account"]["net_return"]) for pair in pairs]
        fade_returns = [float(pair["fade"]["account"]["net_return"]) for pair in pairs]
        report["scenario_summary"][scenario] = {
            "trend_block_returns": trend_returns,
            "fade_block_returns": fade_returns,
            "trend_chain_index": prod(1 + value for value in trend_returns) - 1,
            "fade_chain_index": prod(1 + value for value in fade_returns) - 1,
            "fade_improved_every_trend_block": all(
                pair["fade_minus_trend"]["net_return_percentage_points"] > 0
                for pair in pairs
            ),
            "fade_reduced_drawdown_every_trend_block": all(
                pair["fade_minus_trend"]["max_drawdown_percentage_points"] < 0
                for pair in pairs
            ),
            "median_exact_entry_return_delta_bps": _quantile([
                float(pair["exact_same_entry_cases"][
                    "net_case_return_bps_delta_new_minus_old"]["median"]
                )
                for pair in pairs
                if pair["exact_same_entry_cases"][
                    "net_case_return_bps_delta_new_minus_old"]["median"] is not None
            ], 0.5),
        }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report["scenario_summary"], indent=2, sort_keys=True))
