"""Pair EMA reverse-after-flat native runs with frozen X1 controls."""
from __future__ import annotations

from collections import defaultdict
import json
from math import prod
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / ".artifacts/research_1m_4teachers_v1/native/teachers/T01_EMA_9_21/X1"
OUTPUT = ROOT / ".artifacts/route-a-ema-reversal-task1/paired_base_stress_summary.json"
WINDOWS = (
    (
        "feb",
        "inner-screen-1",
        ROOT / ".artifacts/route-a-ema-reversal-task1-feb-reverse-base-20261026",
        ROOT / ".artifacts/route-a-ema-reversal-task1-feb-reverse-stress-20261027",
    ),
    (
        "mar",
        "inner-screen-2",
        ROOT / ".artifacts/route-a-ema-reversal-task1-mar-reverse-base-20261027",
        ROOT / ".artifacts/route-a-ema-reversal-task1-mar-reverse-stress-20261027",
    ),
    (
        "apr",
        "inner-screen-3",
        ROOT / ".artifacts/route-a-ema-reversal-task1-apr-reverse-base-20261027",
        ROOT / ".artifacts/route-a-ema-reversal-task1-apr-reverse-stress-20261027",
    ),
)
SCENARIOS = ("base", "stress")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _trades(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _funnel(path: Path) -> dict[str, Any]:
    data = _read_json(path)
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
            for row in data["groups"] if row["side"] != "FLAT"
        ],
        "economic_classifications": data["economic_classifications"],
    }


def _case_metrics(cases: list[dict[str, Any]], cost_bps: float) -> dict[str, Any]:
    statuses: dict[str, int] = defaultdict(int)
    pnl_by_status: dict[str, float] = defaultdict(float)
    for case in cases:
        status = str(case["status"])
        statuses[status] += 1
        pnl_by_status[status] += float(case["realized_pnl"])
    sides: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if case["status"] == "ORDINARY_COMPLETE":
            sides["LONG" if int(case["side"]) > 0 else "SHORT"].append(case)
    result: dict[str, Any] = {
        "case_status_counts": dict(statuses),
        "case_pnl_by_status": dict(pnl_by_status),
        "complete_trade_count": sum(len(rows) for rows in sides.values()),
        "sides": {},
    }
    for side, rows in sorted(sides.items()):
        pnls = [float(row["realized_pnl"]) for row in rows]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        holds = [float(row["holding_minutes"]) for row in rows]
        mfe = [float(row["mfe_bps"]) for row in rows]
        mae = [float(row["mae_bps"]) for row in rows]
        capture = [
            (float(row["realized_pnl"]) + float(row["fees"]) - float(row["funding"]))
            / (float(row["entry_notional"]) * float(row["mfe_bps"]) / 10_000)
            for row in rows
            if float(row["entry_notional"]) > 0 and float(row["mfe_bps"]) > 0
        ]
        result["sides"][side] = {
            "complete_trades": len(rows),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(rows) if rows else None,
            "net_trade_case_pnl": sum(pnls),
            "gross_cash_pnl_before_fees_and_funding": sum(
                pnl + float(row["fees"]) - float(row["funding"])
                for pnl, row in zip(pnls, rows, strict=True)
            ),
            "fees": sum(float(row["fees"]) for row in rows),
            "slippage_attribution": sum(float(row["slippage_cost"]) for row in rows),
            "funding": sum(float(row["funding"]) for row in rows),
            "average_win": statistics.mean(wins) if wins else None,
            "average_loss": statistics.mean(losses) if losses else None,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
            "holding_minutes": {
                "mean": statistics.mean(holds) if holds else None,
                "median": _quantile(holds, 0.5),
                "p90": _quantile(holds, 0.9),
            },
            "mfe_bps": {
                "mean": statistics.mean(mfe) if mfe else None,
                "median": _quantile(mfe, 0.5),
                "p90": _quantile(mfe, 0.9),
            },
            "mae_bps": {
                "mean": statistics.mean(mae) if mae else None,
                "median": _quantile(mae, 0.5),
                "p90": _quantile(mae, 0.9),
            },
            "gross_profit_capture_ratio_mean": statistics.mean(capture) if capture else None,
            "losers_with_mfe_at_least_round_trip_cost": sum(
                float(row["realized_pnl"]) < 0 and float(row["mfe_bps"]) >= cost_bps
                for row in rows
            ),
        }
    return result


def _pair_identity(old: dict[str, Any], new: dict[str, Any]) -> dict[str, bool]:
    old_recipe = old["teacher_recipe"]
    new_recipe = new["teacher_recipe"]
    old_entry_parameters = [row for row in old_recipe["parameters"] if row[0] != "reversal_mode"]
    new_entry_parameters = [row for row in new_recipe["parameters"] if row[0] != "reversal_mode"]
    checks = {
        "same_source_snapshot": (
            old["source"]["snapshot_hash"] == new["source"]["snapshot_hash"]
            and old["source"]["marker_sha256"] == new["source"]["marker_sha256"]
        ),
        "same_registered_window": old["source"]["window_ns"] == new["source"]["window_ns"],
        "same_scaler": (
            old["execution"]["scaler_hash"] == new["execution"]["scaler_hash"]
            and old["execution"]["scaler_fit_window_hash"]
            == new["execution"]["scaler_fit_window_hash"]
        ),
        "same_account_risk_and_costs": all(
            old["execution"][key] == new["execution"][key]
            for key in ("starting_balance", "risk_budget_notional", "max_leverage", "scenario_costs")
        ),
        "same_protocol_and_preregistration": (
            old["protocol_id"] == new["protocol_id"]
            and old["preregistration_sha256"] == new["preregistration_sha256"]
        ),
        "same_indicator_and_entry_parameters": (
            old_recipe["indicator_recipe_id"] == new_recipe["indicator_recipe_id"]
            and old_entry_parameters == new_entry_parameters
            and old_recipe["target_exposure"] == new_recipe["target_exposure"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"EMA old/new runs are not like-for-like: {checks}")
    return checks


def _entry_key(case: dict[str, Any]) -> tuple[int, int, int]:
    return int(case["entry_decision_ts_ns"]), int(case["entry_fill_ts_ns"]), int(case["side"])


def _same_entries(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> dict[str, Any]:
    old_map = {
        _entry_key(case): case for case in old if case["status"] == "ORDINARY_COMPLETE"
    }
    new_map = {
        _entry_key(case): case for case in new if case["status"] == "ORDINARY_COMPLETE"
    }
    common = set(old_map) & set(new_map)
    deltas = [
        float(new_map[key]["realized_pnl"]) / float(new_map[key]["entry_notional"])
        * 10_000
        - float(old_map[key]["realized_pnl"]) / float(old_map[key]["entry_notional"])
        * 10_000
        for key in common
    ]
    return {
        "same_exact_entry_complete_pairs": len(common),
        "old_complete_entries_without_new_pair": len(old_map) - len(common),
        "new_complete_entries_without_old_pair": len(new_map) - len(common),
        "net_case_return_bps_delta_new_minus_old": {
            "mean": statistics.mean(deltas) if deltas else None,
            "median": _quantile(deltas, 0.5),
            "new_better_count": sum(value > 1e-9 for value in deltas),
            "old_better_count": sum(value < -1e-9 for value in deltas),
            "tie_count": sum(abs(value) <= 1e-9 for value in deltas),
        },
    }


def run() -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "route-a.ema-reversal-native-comparison.v2",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "source_use": "previously consumed S06 Train windows; no new market read",
        "baseline_commit": "bd313c5241c31f1ccff96fa4bf76a7966be48767",
        "slippage_note": "Native realized PnL includes actual fills; slippage attribution is not subtracted twice.",
        "pairs": [],
        "scenario_summary": {},
    }
    by_scenario: dict[str, list[dict[str, Any]]] = {scenario: [] for scenario in SCENARIOS}
    for month, block, base_dir, stress_dir in WINDOWS:
        for scenario, candidate_dir in (("base", base_dir), ("stress", stress_dir)):
            old_dir = BASELINE_ROOT / block / scenario
            old = _read_json(old_dir / "native-run.json")
            new = _read_json(candidate_dir / "native-run.json")
            identity = _pair_identity(old, new)
            for native in (old, new):
                decisions = native["decisions"]
                if decisions["m22_verified"] != decisions["observed"]:
                    raise RuntimeError("EMA native decision was not fully M2.2 verified")
                if float(native["account"]["liquidations"]) != 0:
                    raise RuntimeError("liquidation in EMA reversal diagnostic")
            old_cases = _trades(old_dir / "teacher-research-1m/native-trades.jsonl")
            new_cases = _trades(candidate_dir / "teacher-research-1m/native-trades.jsonl")
            costs = old["execution"]["scenario_costs"]
            round_trip_cost_bps = 2 * (
                float(costs["taker_fee_bps_per_leg"])
                + float(costs["slippage_bps_per_leg"])
            )
            old_metrics = _case_metrics(old_cases, round_trip_cost_bps)
            new_metrics = _case_metrics(new_cases, round_trip_cost_bps)
            old_delta = float(old["account"]["equity_end"]) - float(old["account"]["equity_start"])
            new_delta = float(new["account"]["equity_end"]) - float(new["account"]["equity_start"])
            old_case_sum = sum(float(case["realized_pnl"]) for case in old_cases)
            new_case_sum = sum(float(case["realized_pnl"]) for case in new_cases)
            if abs(old_case_sum - old_delta) > 0.02 or abs(new_case_sum - new_delta) > 0.02:
                raise RuntimeError("EMA trade cases fail to reconcile to native account equity")
            pair = {
                "month": month,
                "block": block,
                "scenario": scenario,
                "round_trip_cost_bps": round_trip_cost_bps,
                "pair_identity": identity,
                "old": {
                    "account": old["account"],
                    "decisions": old["decisions"],
                    "case_metrics": old_metrics,
                    "case_pnl_reconciliation": {
                        "all_case_reported_pnl": old_case_sum,
                        "account_equity_delta": old_delta,
                        "residual": old_case_sum - old_delta,
                    },
                    "teacher_funnel": _funnel(old_dir / "teacher-research-1m/teacher-funnel.json"),
                },
                "reverse_after_flat": {
                    "account": new["account"],
                    "decisions": new["decisions"],
                    "teacher_recipe": new["teacher_recipe"],
                    "case_metrics": new_metrics,
                    "case_pnl_reconciliation": {
                        "all_case_reported_pnl": new_case_sum,
                        "account_equity_delta": new_delta,
                        "residual": new_case_sum - new_delta,
                    },
                    "teacher_funnel": _funnel(
                        candidate_dir / "teacher-research-1m/teacher-funnel.json",
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
                    "fee_delta": float(new["account"]["fees"]) - float(old["account"]["fees"]),
                    "slippage_attribution_delta": (
                        float(new["account"]["slippage_cost"])
                        - float(old["account"]["slippage_cost"])
                    ),
                    "funding_delta": (
                        float(new["account"]["funding"]) - float(old["account"]["funding"])
                    ),
                    "completed_trade_delta": (
                        new_metrics["complete_trade_count"] - old_metrics["complete_trade_count"]
                    ),
                },
                "same_entry_counterfactual": _same_entries(old_cases, new_cases),
            }
            result["pairs"].append(pair)
            by_scenario[scenario].append(pair)
    for scenario, rows in by_scenario.items():
        old_returns = [float(row["old"]["account"]["net_return"]) for row in rows]
        new_returns = [float(row["reverse_after_flat"]["account"]["net_return"]) for row in rows]
        result["scenario_summary"][scenario] = {
            "old_block_returns": old_returns,
            "reverse_block_returns": new_returns,
            "old_chain_index": prod(1 + value for value in old_returns) - 1,
            "reverse_chain_index": prod(1 + value for value in new_returns) - 1,
            "all_reverse_blocks_positive": all(value > 0 for value in new_returns),
            "reverse_improved_every_block": all(
                row["paired_delta"]["net_return_percentage_points"] > 0 for row in rows
            ),
            "reverse_improved_drawdown_every_block": all(
                row["paired_delta"]["max_drawdown_percentage_points"] < 0 for row in rows
            ),
        }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["scenario_summary"], indent=2, sort_keys=True))
