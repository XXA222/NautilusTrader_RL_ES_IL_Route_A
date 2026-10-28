"""Summarize paired native evidence for the ROC cost-hysteresis candidate."""
from __future__ import annotations

from collections import defaultdict
import json
from math import prod
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = (
    ROOT / ".artifacts/research_1m_4teachers_v1/native/teachers"
    / "T02_ROC_20_60/X1"
)
CANDIDATE_ROOT = (
    ROOT / ".artifacts/route-a-roc-hysteresis-task2/native"
)
OUTPUT = ROOT / ".artifacts/route-a-roc-hysteresis-task2/paired_native_results.json"
BLOCKS = ("inner-screen-1", "inner-screen-2", "inner-screen-3")
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


def _trade_stats(path: Path, round_trip_cost_bps: float) -> dict[str, Any]:
    sides: dict[str, list[dict[str, Any]]] = defaultdict(list)
    statuses: dict[str, int] = defaultdict(int)
    case_pnl_by_status: dict[str, float] = defaultdict(float)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            trade = json.loads(line)
            status = str(trade["status"])
            statuses[status] += 1
            case_pnl_by_status[status] += float(trade["realized_pnl"])
            if trade["status"] == "ORDINARY_COMPLETE":
                sides["LONG" if int(trade["side"]) > 0 else "SHORT"].append(trade)

    output: dict[str, Any] = {
        "case_statuses": dict(statuses),
        "case_pnl_by_status": {
            status: {
                "case_count": statuses[status],
                "case_reported_pnl": pnl,
            }
            for status, pnl in sorted(case_pnl_by_status.items())
        },
        "all_case_reported_pnl": sum(case_pnl_by_status.values()),
        "sides": {},
    }
    for side, rows in sorted(sides.items()):
        pnls = [float(row["realized_pnl"]) for row in rows]
        winners = [pnl for pnl in pnls if pnl > 0]
        losers = [pnl for pnl in pnls if pnl < 0]
        fees = sum(float(row["fees"]) for row in rows)
        funding = sum(float(row["funding"]) for row in rows)
        gross = sum(pnl + float(row["fees"]) - float(row["funding"]) for pnl, row in zip(pnls, rows, strict=True))
        holding = [float(row["holding_minutes"]) for row in rows]
        mfe = [float(row["mfe_bps"]) for row in rows if row.get("mfe_bps") is not None]
        mae = [float(row["mae_bps"]) for row in rows if row.get("mae_bps") is not None]
        capture = [
            (pnl + float(row["fees"]) - float(row["funding"]))
            / (float(row["entry_notional"]) * float(row["mfe_bps"]) / 10_000)
            for pnl, row in zip(pnls, rows, strict=True)
            if row.get("mfe_bps") not in (None, 0)
        ]
        loss_with_mfe = sum(
            float(row["realized_pnl"]) < 0
            and float(row.get("mfe_bps") or 0) >= round_trip_cost_bps
            for row in rows
        )
        output["sides"][side] = {
            "complete_trades": len(rows),
            "wins": len(winners),
            "losses": len(losers),
            "win_rate": len(winners) / len(rows) if rows else None,
            "gross_cash_pnl_before_fees_and_funding": gross,
            "fees": fees,
            "slippage_cost": sum(float(row["slippage_cost"]) for row in rows),
            "funding": funding,
            "net_trade_cash_pnl": sum(pnls),
            "average_win": statistics.mean(winners) if winners else None,
            "average_loss": statistics.mean(losers) if losers else None,
            "profit_factor": sum(winners) / abs(sum(losers)) if losers else None,
            "holding_minutes": {
                "mean": statistics.mean(holding) if holding else None,
                "median": _quantile(holding, 0.5),
                "p90": _quantile(holding, 0.9),
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
            "losers_with_mfe_at_least_round_trip_cost": int(loss_with_mfe),
        }
    output["complete_trades"] = sum(
        metrics["complete_trades"] for metrics in output["sides"].values()
    )
    return output


def _case_pnl_reconciliation(
    trade_metrics: dict[str, Any], account: dict[str, Any],
) -> dict[str, Any]:
    account_delta = float(account["equity_end"]) - float(account["equity_start"])
    all_case_pnl = float(trade_metrics["all_case_reported_pnl"])
    residual = all_case_pnl - account_delta
    complete = trade_metrics["case_pnl_by_status"].get(
        "ORDINARY_COMPLETE", {"case_count": 0, "case_reported_pnl": 0.0},
    )
    censored = trade_metrics["case_pnl_by_status"].get(
        "TERMINAL_CENSORED", {"case_count": 0, "case_reported_pnl": 0.0},
    )
    return {
        "ordinary_complete_case_pnl": complete["case_reported_pnl"],
        "terminal_censored_case_count": censored["case_count"],
        "terminal_censored_case_pnl": censored["case_reported_pnl"],
        "all_case_reported_pnl": all_case_pnl,
        "account_equity_delta": account_delta,
        "unexplained_residual": residual,
        "matches_account_delta_within_0_02": abs(residual) <= 0.02,
    }


def _funnel(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    groups = [
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
        for row in data["groups"]
        if row["side"] != "FLAT"
    ]
    return {"groups": groups, "classification": data["economic_classifications"]}


def _pair_identity(old: dict[str, Any], new: dict[str, Any]) -> dict[str, bool]:
    fields = {
        "same_source_snapshot": (
            old["source"]["snapshot_hash"] == new["source"]["snapshot_hash"]
        ),
        "same_registered_window": (
            old["source"]["window_ns"] == new["source"]["window_ns"]
        ),
        "same_scaler": (
            old["execution"]["scaler_hash"] == new["execution"]["scaler_hash"]
            and old["execution"]["scaler_fit_window_hash"]
            == new["execution"]["scaler_fit_window_hash"]
        ),
        "same_account_and_risk": all(
            old["execution"][key] == new["execution"][key]
            for key in ("starting_balance", "risk_budget_notional", "max_leverage")
        ),
        "same_scenario_costs": (
            old["execution"]["scenario_costs"] == new["execution"]["scenario_costs"]
        ),
        "same_pre_registration": (
            old["protocol_id"] == new["protocol_id"]
            and old["preregistration_sha256"] == new["preregistration_sha256"]
        ),
    }
    if not all(fields.values()):
        raise RuntimeError(f"new/old native runs are not like-for-like: {fields}")
    return fields


def run() -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "route-a.roc-cost-hysteresis-native-comparison.v2",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "economic_status": "SCENARIO_ONLY",
        "baseline_commit": "bd313c5241c31f1ccff96fa4bf76a7966be48767",
        "source_use": "previously consumed S06 Train diagnostic; no new market read",
        "slippage_note": (
            "Native equity uses observed fills; reported slippage_cost is a separate "
            "attribution and is not subtracted a second time."
        ),
        "round_trip_cost_estimate_bps": {"base": 12.0, "stress": 30.0},
        "blocks": [],
    }
    by_scenario: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in SCENARIOS
    }

    for block in BLOCKS:
        for scenario in SCENARIOS:
            old_dir = BASELINE_ROOT / block / scenario
            new_dir = CANDIDATE_ROOT / block / scenario
            old = _read_json(old_dir / "native-run.json")
            new = _read_json(new_dir / "native-run.json")
            identity = _pair_identity(old, new)
            for report in (old, new):
                decisions = report["decisions"]
                if decisions["m22_verified"] != decisions["observed"]:
                    raise RuntimeError("not every native decision was M2.2 verified")
                if report["account"]["liquidations"] != 0:
                    raise RuntimeError("liquidation found in ROC diagnostic pair")

            cost_bps = result["round_trip_cost_estimate_bps"][scenario]
            old_trades = _trade_stats(
                old_dir / "teacher-research-1m/native-trades.jsonl", cost_bps,
            )
            new_trades = _trade_stats(
                new_dir / "teacher-research-1m/native-trades.jsonl", cost_bps,
            )
            old_funnel = _funnel(old_dir / "teacher-research-1m/teacher-funnel.json")
            new_funnel = _funnel(new_dir / "teacher-research-1m/teacher-funnel.json")
            record = {
                "block": block,
                "scenario": scenario,
                "pair_identity": identity,
                "window_ns": new["source"]["window_ns"],
                "native_decisions": new["decisions"]["observed"],
                "m22_verified_decisions": new["decisions"]["m22_verified"],
                "old": {
                    "account": old["account"],
                    "decisions": old["decisions"],
                    "trade_metrics": old_trades,
                    "case_pnl_reconciliation": _case_pnl_reconciliation(
                        old_trades, old["account"],
                    ),
                    "teacher_funnel": old_funnel,
                },
                "candidate": {
                    "account": new["account"],
                    "decisions": new["decisions"],
                    "teacher_recipe": new["teacher_recipe"],
                    "trade_metrics": new_trades,
                    "case_pnl_reconciliation": _case_pnl_reconciliation(
                        new_trades, new["account"],
                    ),
                    "teacher_funnel": new_funnel,
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
                    "fee_delta": (
                        float(new["account"]["fees"]) - float(old["account"]["fees"])
                    ),
                    "slippage_attribution_delta": (
                        float(new["account"]["slippage_cost"])
                        - float(old["account"]["slippage_cost"])
                    ),
                    "funding_delta": (
                        float(new["account"]["funding"]) - float(old["account"]["funding"])
                    ),
                    "trade_case_delta": (
                        new_trades["complete_trades"] - old_trades["complete_trades"]
                    ),
                },
            }
            result["blocks"].append(record)
            by_scenario[scenario].append(record)

    result["scenario_summary"] = {}
    for scenario, rows in by_scenario.items():
        old_returns = [float(row["old"]["account"]["net_return"]) for row in rows]
        new_returns = [float(row["candidate"]["account"]["net_return"]) for row in rows]
        result["scenario_summary"][scenario] = {
            "old_trade_cases": sum(row["old"]["trade_metrics"]["complete_trades"] for row in rows),
            "candidate_trade_cases": sum(row["candidate"]["trade_metrics"]["complete_trades"] for row in rows),
            "old_block_returns": old_returns,
            "candidate_block_returns": new_returns,
            "old_block_return_chain_index": prod(1 + value for value in old_returns) - 1,
            "candidate_block_return_chain_index": prod(1 + value for value in new_returns) - 1,
            "all_candidate_blocks_positive": all(value > 0 for value in new_returns),
            "all_candidate_blocks_improve_net_return": all(
                row["paired_delta"]["net_return_percentage_points"] > 0 for row in rows
            ),
            "all_candidate_blocks_improve_drawdown": all(
                row["paired_delta"]["max_drawdown_percentage_points"] < 0 for row in rows
            ),
        }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps(report["scenario_summary"], indent=2, sort_keys=True))
