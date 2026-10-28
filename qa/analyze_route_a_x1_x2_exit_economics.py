"""Compare native X1/X2 exit economics and exact-entry trade outcomes."""
from __future__ import annotations

import csv
import json
from math import isfinite
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = ROOT / ".artifacts/research_1m_4teachers_v1/native/teachers"
OUTPUT_DIR = ROOT / ".artifacts/route-a-exit-economics-task3"
REPORT_PATH = OUTPUT_DIR / "x1_x2_exit_economics.json"
TRADES_PATH = OUTPUT_DIR / "per_trade_exit_metrics.jsonl"
TEACHERS = ("T01_EMA_9_21", "T02_ROC_20_60", "T03_DONCHIAN_20", "T04_ZSCORE_60")
BLOCKS = ("inner-screen-1", "inner-screen-2", "inner-screen-3")
SCENARIOS = ("base", "stress")
ENTRY_NOTIONAL_ROUNDING_DECIMALS = 6


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


def _pair_identity(x1: dict[str, Any], x2: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "same_source_snapshot": (
            x1["source"]["snapshot_hash"] == x2["source"]["snapshot_hash"]
            and x1["source"]["marker_sha256"] == x2["source"]["marker_sha256"]
        ),
        "same_registered_window": x1["source"]["window_ns"] == x2["source"]["window_ns"],
        "same_scaler": (
            x1["execution"]["scaler_hash"] == x2["execution"]["scaler_hash"]
            and x1["execution"]["scaler_fit_window_hash"]
            == x2["execution"]["scaler_fit_window_hash"]
        ),
        "same_account_risk_and_costs": all(
            x1["execution"][key] == x2["execution"][key]
            for key in ("starting_balance", "risk_budget_notional", "max_leverage", "scenario_costs")
        ),
        "same_protocol_and_preregistration": (
            x1["protocol_id"] == x2["protocol_id"]
            and x1["preregistration_sha256"] == x2["preregistration_sha256"]
        ),
        "same_entry_recipe": all(
            x1["teacher_recipe"][key] == x2["teacher_recipe"][key]
            for key in ("entry_rule_id", "indicator_recipe_id", "parameters",
                        "target_exposure", "higher_timeframe_mode")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"X1/X2 are not like-for-like: {checks}")
    return checks


def _load_cases(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "teacher-research-1m/native-trades.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


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
    return {
        "groups": groups,
        "economic_classifications": data["economic_classifications"],
    }


def _add_excursion_metrics(run_dir: Path, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with (run_dir / "teacher-research-1m/decision-rows.csv").open(
        encoding="utf-8", newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    end_by_case = {
        row["case_id"]: int(row["decision_ts_ns"]) for row in rows if row["case_id"]
    }
    run_end = max(int(row["decision_ts_ns"]) for row in rows)
    cases = []
    for trade in trades:
        case = dict(trade)
        if trade["status"] == "ORDINARY_COMPLETE":
            end = end_by_case.get(trade["trade_case_id"])
            if end is None:
                raise RuntimeError(f"missing exit decision row: {trade['trade_case_id']}")
            case["path_end"] = end
        elif trade["status"] == "TERMINAL_CENSORED":
            case["path_end"] = run_end
        else:
            case["path_end"] = None
        case.update({
            "path_mfe_bps": None,
            "max_mfe_to_close_giveback_bps": None,
            "last_close_return_bps": None,
            "min_close_return_bps": None,
            "path_observation_count": 0,
        })
        cases.append(case)
    active = sorted(
        (case for case in cases if case["path_end"] is not None),
        key=lambda case: int(case["entry_fill_ts_ns"]),
    )
    active_index = 0
    for row in rows:
        timestamp = int(row["decision_ts_ns"])
        while active_index < len(active) and timestamp > int(active[active_index]["path_end"]):
            active_index += 1
        if active_index >= len(active):
            break
        case = active[active_index]
        if timestamp < int(case["entry_fill_ts_ns"]):
            continue
        try:
            close = float(json.loads(row["indicator_values_json"])["bar_close_1m"])
            entry = float(case["entry_fill_price"])
            side = 1 if int(case["side"]) > 0 else -1
            current_mfe = float(row["mfe_bps"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if entry <= 0 or not all(isfinite(v) for v in (close, entry, current_mfe)):
            continue
        close_return = side * (close / entry - 1.0) * 10_000
        running_mfe = max(float(case["path_mfe_bps"] or 0.0), current_mfe)
        giveback = running_mfe - close_return
        case["path_mfe_bps"] = running_mfe
        case["max_mfe_to_close_giveback_bps"] = max(
            float(case["max_mfe_to_close_giveback_bps"] or 0.0), giveback,
        )
        case["last_close_return_bps"] = close_return
        prior_min = case["min_close_return_bps"]
        case["min_close_return_bps"] = min(
            float(prior_min) if prior_min is not None else close_return, close_return,
        )
        case["path_observation_count"] += 1
    for case in cases:
        native_mfe = float(case["mfe_bps"])
        case["path_mfe_difference_bps"] = (
            float(case["path_mfe_bps"]) - native_mfe
            if case["path_mfe_bps"] is not None else None
        )
        entry = float(case["entry_fill_price"])
        exit_price = case.get("exit_fill_price")
        case["gross_price_return_bps"] = (
            (1 if int(case["side"]) > 0 else -1)
            * (float(exit_price) / entry - 1.0) * 10_000
            if entry > 0 and exit_price is not None else None
        )
        notional = float(case["entry_notional"])
        case["net_case_return_bps"] = (
            float(case["realized_pnl"]) / notional * 10_000 if notional > 0 else None
        )
        case["gross_price_profit_capture_ratio"] = (
            float(case["gross_price_return_bps"]) / native_mfe
            if case["gross_price_return_bps"] is not None and native_mfe > 0 else None
        )
        case["mfe_to_exit_giveback_bps"] = (
            native_mfe - float(case["gross_price_return_bps"])
            if case["gross_price_return_bps"] is not None else None
        )
        case["gross_cash_pnl_before_fees_and_funding"] = (
            float(case["realized_pnl"]) + float(case["fees"]) - float(case["funding"])
        )
        case.pop("path_end", None)
    return cases


def _side_metrics(cases: list[dict[str, Any]], cost_bps: float) -> dict[str, Any]:
    result = {}
    for name, side in (("LONG", 1), ("SHORT", -1)):
        selected = [
            case for case in cases
            if case["status"] == "ORDINARY_COMPLETE"
            and (1 if int(case["side"]) > 0 else -1) == side
        ]
        pnls = [float(case["realized_pnl"]) for case in selected]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        holds = [float(case["holding_minutes"]) for case in selected]
        mfe = [float(case["mfe_bps"]) for case in selected]
        mae = [float(case["mae_bps"]) for case in selected]
        givebacks = [float(case["max_mfe_to_close_giveback_bps"])
                     for case in selected if case["max_mfe_to_close_giveback_bps"] is not None]
        captures = [float(case["gross_price_profit_capture_ratio"])
                    for case in selected if case["gross_price_profit_capture_ratio"] is not None]
        gross_returns = [float(case["gross_price_return_bps"])
                         for case in selected if case["gross_price_return_bps"] is not None]
        exit_reasons: dict[str, int] = {}
        for case in selected:
            reason = str(case.get("exit_reason") or "UNKNOWN")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        result[name] = {
            "complete_trades": len(selected),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(selected) if selected else None,
            "net_trade_case_pnl": sum(pnls),
            "gross_cash_pnl_before_fees_and_funding": sum(
                float(case["gross_cash_pnl_before_fees_and_funding"]) for case in selected
            ),
            "fees": sum(float(case["fees"]) for case in selected),
            "slippage_attribution": sum(float(case["slippage_cost"]) for case in selected),
            "funding": sum(float(case["funding"]) for case in selected),
            "mean_net_case_return_bps": statistics.mean(
                float(case["net_case_return_bps"]) for case in selected
            ) if selected else None,
            "mean_gross_price_return_bps": statistics.mean(gross_returns) if gross_returns else None,
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
            "max_mfe_to_close_giveback_bps": {
                "mean": statistics.mean(givebacks) if givebacks else None,
                "median": _quantile(givebacks, 0.5),
                "p90": _quantile(givebacks, 0.9),
            },
            "gross_price_profit_capture_ratio": {
                "mean": statistics.mean(captures) if captures else None,
                "median": _quantile(captures, 0.5),
            },
            "losers_with_mfe_at_least_round_trip_cost": sum(
                float(case["realized_pnl"]) < 0 and float(case["mfe_bps"]) >= cost_bps
                for case in selected
            ),
            "losses_with_mae_greater_than_mfe": sum(
                float(case["realized_pnl"]) < 0
                and float(case["mae_bps"]) > float(case["mfe_bps"])
                for case in selected
            ),
            "exit_reasons": dict(sorted(exit_reasons.items())),
        }
    return result


def _economics(run: dict[str, Any], cases: list[dict[str, Any]], cost_bps: float) -> dict[str, Any]:
    counts: dict[str, int] = {}
    pnl_by_status: dict[str, float] = {}
    for case in cases:
        status = str(case["status"])
        counts[status] = counts.get(status, 0) + 1
        pnl_by_status[status] = pnl_by_status.get(status, 0.0) + float(case["realized_pnl"])
    all_pnl = sum(float(case["realized_pnl"]) for case in cases)
    account = run["account"]
    equity_delta = float(account["equity_end"]) - float(account["equity_start"])
    complete = [case for case in cases if case["status"] == "ORDINARY_COMPLETE"]
    return {
        "account": account,
        "case_status_counts": counts,
        "case_pnl_by_status": pnl_by_status,
        "all_case_reported_pnl": all_pnl,
        "account_equity_delta": equity_delta,
        "case_to_account_residual": all_pnl - equity_delta,
        "case_to_account_reconciles_within_0_02": abs(all_pnl - equity_delta) <= 0.02,
        "complete_trade_count": len(complete),
        "terminal_censored_count": counts.get("TERMINAL_CENSORED", 0),
        "path_mfe_coverage_count": sum(case["path_mfe_bps"] is not None for case in complete),
        "path_mfe_mismatch_over_0_10_bps_count": sum(
            case["path_mfe_difference_bps"] is not None
            and abs(float(case["path_mfe_difference_bps"])) > 0.10 for case in complete
        ),
        "sides": _side_metrics(cases, cost_bps),
    }


def _entry_key(case: dict[str, Any]) -> tuple[int, int]:
    return (int(case["entry_fill_ts_ns"]), int(case["side"]))


def _same_entry(x1_cases: list[dict[str, Any]], x2_cases: list[dict[str, Any]]) -> dict[str, Any]:
    def index(cases: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
        result = {}
        for case in cases:
            key = _entry_key(case)
            if key in result:
                raise RuntimeError(f"duplicate entry-fill timestamp/side key: {key}")
            result[key] = case
        return result

    x1 = index(x1_cases)
    x2 = index(x2_cases)
    timestamp_side_overlap = sorted(set(x1) & set(x2))
    notional_mismatch_keys = [
        key for key in timestamp_side_overlap
        if round(float(x1[key]["entry_notional"]), ENTRY_NOTIONAL_ROUNDING_DECIMALS)
        != round(float(x2[key]["entry_notional"]), ENTRY_NOTIONAL_ROUNDING_DECIMALS)
    ]
    notional_mismatch_key_set = set(notional_mismatch_keys)
    same_notional = [key for key in timestamp_side_overlap if key not in notional_mismatch_key_set]
    status_pairs: dict[str, int] = {}
    for key in same_notional:
        status_pair = f"X1={x1[key]['status']};X2={x2[key]['status']}"
        status_pairs[status_pair] = status_pairs.get(status_pair, 0) + 1
    common = [
        key for key in same_notional
        if x1[key]["status"] == "ORDINARY_COMPLETE"
        and x2[key]["status"] == "ORDINARY_COMPLETE"
    ]
    net_delta = [float(x2[key]["net_case_return_bps"]) - float(x1[key]["net_case_return_bps"])
                 for key in common]
    price_delta = [float(x2[key]["gross_price_return_bps"]) - float(x1[key]["gross_price_return_bps"])
                   for key in common]
    hold_delta = [float(x2[key]["holding_minutes"]) - float(x1[key]["holding_minutes"])
                  for key in common]
    mfe_delta = [float(x2[key]["mfe_bps"]) - float(x1[key]["mfe_bps"]) for key in common]
    giveback_delta = [
        float(x2[key]["max_mfe_to_close_giveback_bps"])
        - float(x1[key]["max_mfe_to_close_giveback_bps"])
        for key in common
        if x1[key]["max_mfe_to_close_giveback_bps"] is not None
        and x2[key]["max_mfe_to_close_giveback_bps"] is not None
    ]
    pnl1 = sum(float(x1[key]["realized_pnl"]) for key in common)
    pnl2 = sum(float(x2[key]["realized_pnl"]) for key in common)
    old_complete_entries = sum(case["status"] == "ORDINARY_COMPLETE" for case in x1_cases)
    new_complete_entries = sum(case["status"] == "ORDINARY_COMPLETE" for case in x2_cases)
    return {
        "entry_match_rule": (
            "same entry_fill_ts_ns and side; entry_notional rounded to 6 decimal places; "
            "exit deltas include only ORDINARY_COMPLETE pairs"
        ),
        "same_fill_side_keys_before_notional_filter": len(timestamp_side_overlap),
        "entry_keys_rejected_for_rounded_notional_mismatch": len(notional_mismatch_keys),
        "same_rounded_notional_pairs_all_statuses": len(same_notional),
        "same_rounded_notional_status_pairs": dict(sorted(status_pairs.items())),
        "same_rounded_notional_pairs_excluded_from_realized_deltas": (
            len(same_notional) - len(common)
        ),
        "old_complete_entries": old_complete_entries,
        "new_complete_entries": new_complete_entries,
        "same_exact_entry_complete_pairs": len(common),
        "old_complete_entries_without_new_pair": old_complete_entries - len(common),
        "new_complete_entries_without_old_pair": new_complete_entries - len(common),
        "matched_pair_net_case_pnl_old": pnl1,
        "matched_pair_net_case_pnl_new": pnl2,
        "matched_pair_net_case_pnl_delta_new_minus_old": pnl2 - pnl1,
        "net_case_return_bps_delta_new_minus_old": {
            "mean": statistics.mean(net_delta) if net_delta else None,
            "median": _quantile(net_delta, 0.5),
            "p10": _quantile(net_delta, 0.1),
            "p90": _quantile(net_delta, 0.9),
            "x2_better_count": sum(value > 1e-9 for value in net_delta),
            "x1_better_count": sum(value < -1e-9 for value in net_delta),
            "tie_count": sum(abs(value) <= 1e-9 for value in net_delta),
        },
        "gross_price_return_bps_delta_new_minus_old": {
            "mean": statistics.mean(price_delta) if price_delta else None,
            "median": _quantile(price_delta, 0.5),
        },
        "holding_minutes_delta_new_minus_old": {
            "mean": statistics.mean(hold_delta) if hold_delta else None,
            "median": _quantile(hold_delta, 0.5),
        },
        "mfe_bps_delta_new_minus_old": {
            "mean": statistics.mean(mfe_delta) if mfe_delta else None,
            "median": _quantile(mfe_delta, 0.5),
        },
        "max_mfe_to_close_giveback_bps_delta_new_minus_old": {
            "mean": statistics.mean(giveback_delta) if giveback_delta else None,
            "median": _quantile(giveback_delta, 0.5),
        },
    }


def _round_trip_cost(run: dict[str, Any]) -> float:
    costs = run["execution"]["scenario_costs"]
    return 2.0 * (float(costs["taker_fee_bps_per_leg"]) + float(costs["slippage_bps_per_leg"]))


def _chain_index(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= 1.0 + value
    return result - 1.0


def run() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "route-a.x1-x2-exit-economics.v1",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "source_use": "previously consumed S06 Train native artifacts; no new market read",
        "mfe_mae_definition": (
            "Native fill-relative MFE/MAE. Per-minute maximum runup giveback is "
            "running native MFE minus directional close return at completed 1m marks."
        ),
        "same_entry_definition": (
            "Exact entry decision timestamp, fill timestamp, and side; both cases "
            "must complete. Return differences are normalized by each entry notional."
        ),
        "slippage_note": "Native realized PnL already reflects fills; slippage attribution is not subtracted twice.",
        "x2_rule": {
            "policy_id": "X2_ATR_TRAIL_3_V1",
            "atr_multiplier": 3.0,
            "activation": "starts on first observed position bar; no separate profit threshold",
            "long": "max(prior trail, highest high - 3 * current ATR); exit when close < trail",
            "short": "min(prior trail, lowest low + 3 * current ATR); exit when close > trail",
            "profit_lock_floor": "none; the ATR trail is not a cost-adjusted break-even lock",
            "causality": "completed current 1m high/low/close and mature ATR only",
        },
        "x1_rules": {
            "T01_EMA_9_21": "exit on a new opposite EMA9/EMA21 cross",
            "T02_ROC_20_60": "exit when either ROC20 or ROC60 reaches/crosses zero against the position",
            "T03_DONCHIAN_20": "exit when close breaches the opposite Donchian10 channel",
            "T04_ZSCORE_60": "exit when z-score returns through zero or an opposite entry condition appears",
        },
        "pairs": [],
        "summary_by_teacher_scenario": {},
        "large_mfe_losing_cases": [],
    }
    per_trade: list[dict[str, Any]] = []
    summary_inputs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for teacher in TEACHERS:
        for block in BLOCKS:
            for scenario in SCENARIOS:
                runs: dict[str, dict[str, Any]] = {}
                cases: dict[str, list[dict[str, Any]]] = {}
                econ: dict[str, dict[str, Any]] = {}
                funnels: dict[str, dict[str, Any]] = {}
                for variant in ("X1", "X2"):
                    run_dir = NATIVE_ROOT / teacher / variant / block / scenario
                    native = _read_json(run_dir / "native-run.json")
                    decisions = native["decisions"]
                    if decisions["observed"] != decisions["m22_verified"]:
                        raise RuntimeError(f"M2.2 mismatch: {run_dir}")
                    if float(native["account"]["liquidations"]) != 0:
                        raise RuntimeError(f"liquidation found: {run_dir}")
                    trade_cases = _add_excursion_metrics(run_dir, _load_cases(run_dir))
                    cost_bps = _round_trip_cost(native)
                    runs[variant] = native
                    cases[variant] = trade_cases
                    econ[variant] = _economics(native, trade_cases, cost_bps)
                    funnels[variant] = _funnel(
                        run_dir / "teacher-research-1m/teacher-funnel.json",
                    )
                    per_trade.extend({
                        "teacher_id": teacher,
                        "exit_variant": variant,
                        "block": block,
                        "scenario": scenario,
                        "round_trip_cost_bps": cost_bps,
                        **case,
                    } for case in trade_cases)
                identity = _pair_identity(runs["X1"], runs["X2"])
                if not all(econ[v]["case_to_account_reconciles_within_0_02"] for v in ("X1", "X2")):
                    raise RuntimeError(f"trade cases fail account reconciliation: {teacher}/{block}/{scenario}")
                cost_bps = _round_trip_cost(runs["X1"])
                pair = {
                    "teacher_id": teacher,
                    "block": block,
                    "scenario": scenario,
                    "round_trip_cost_bps": cost_bps,
                    "pair_identity": identity,
                    "exit_policy_ids": {
                        "X1": runs["X1"]["teacher_recipe"]["exit_policy_id"],
                        "X2": runs["X2"]["teacher_recipe"]["exit_policy_id"],
                    },
                    "x1": econ["X1"],
                    "x2": econ["X2"],
                    "x1_teacher_funnel": funnels["X1"],
                    "x2_teacher_funnel": funnels["X2"],
                    "paired_account_delta_x2_minus_x1": {
                        "net_return_percentage_points": (
                            float(runs["X2"]["account"]["net_return"])
                            - float(runs["X1"]["account"]["net_return"])
                        ) * 100,
                        "max_drawdown_percentage_points": (
                            float(runs["X2"]["account"]["max_drawdown"])
                            - float(runs["X1"]["account"]["max_drawdown"])
                        ) * 100,
                    },
                    "same_entry_counterfactual": _same_entry(cases["X1"], cases["X2"]),
                }
                report["pairs"].append(pair)
                summary_inputs.setdefault((teacher, scenario), []).append(pair)
                for variant in ("X1", "X2"):
                    for case in cases[variant]:
                        if (case["status"] == "ORDINARY_COMPLETE"
                                and float(case["realized_pnl"]) < 0
                                and float(case["mfe_bps"]) >= cost_bps):
                            report["large_mfe_losing_cases"].append({
                                "teacher_id": teacher,
                                "exit_variant": variant,
                                "block": block,
                                "scenario": scenario,
                                "trade_case_id": case["trade_case_id"],
                                "entry_decision_ts_ns": case["entry_decision_ts_ns"],
                                "side": "LONG" if int(case["side"]) > 0 else "SHORT",
                                "net_case_pnl": case["realized_pnl"],
                                "gross_price_return_bps": case["gross_price_return_bps"],
                                "mfe_bps": case["mfe_bps"],
                                "mae_bps": case["mae_bps"],
                                "max_mfe_to_close_giveback_bps": case["max_mfe_to_close_giveback_bps"],
                                "profit_capture_ratio": case["gross_price_profit_capture_ratio"],
                                "holding_minutes": case["holding_minutes"],
                                "exit_reason": case["exit_reason"],
                            })
    for (teacher, scenario), rows in sorted(summary_inputs.items()):
        x1_returns = [float(row["x1"]["account"]["net_return"]) for row in rows]
        x2_returns = [float(row["x2"]["account"]["net_return"]) for row in rows]
        report["summary_by_teacher_scenario"][f"{teacher}/{scenario}"] = {
            "x1_block_returns": x1_returns,
            "x2_block_returns": x2_returns,
            "x1_chain_index": _chain_index(x1_returns),
            "x2_chain_index": _chain_index(x2_returns),
            "x2_improved_net_return_in_every_block": all(
                row["paired_account_delta_x2_minus_x1"]["net_return_percentage_points"] > 0
                for row in rows
            ),
            "x2_improved_drawdown_in_every_block": all(
                row["paired_account_delta_x2_minus_x1"]["max_drawdown_percentage_points"] < 0
                for row in rows
            ),
        }
    report["large_mfe_losing_cases"].sort(key=lambda case: float(case["mfe_bps"]), reverse=True)
    report["large_mfe_losing_case_count"] = len(report["large_mfe_losing_cases"])
    report["large_mfe_losing_cases"] = report["large_mfe_losing_cases"][:100]
    TRADES_PATH.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_trade), encoding="utf-8",
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary_by_teacher_scenario"], indent=2, sort_keys=True))
