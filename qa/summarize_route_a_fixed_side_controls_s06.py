"""Pair native S06 fixed-side controls with the matching ROC/X1 runs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT / ".artifacts/route-a-fixed-controls-task4"
X1_ROOT = ROOT / ".artifacts/route-a-roc-hysteresis-task2/native"
OUTPUT_JSON = CONTROL_ROOT / "paired_fixed_side_vs_x1.json"
OUTPUT_MD = CONTROL_ROOT / "paired_fixed_side_vs_x1.md"
WINDOWS = ("inner-screen-1", "inner-screen-2", "inner-screen-3")
SCENARIOS = ("base", "stress")
CONTROLS = ("FIXED_LONG_0.3", "FIXED_SHORT_0.3")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _case(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"fixed-side control must have exactly one case: {path}")
    return rows[0]


def _metrics(report: dict, case: dict) -> dict:
    account = report["account"]
    decisions = report["decisions"]
    return {
        "net_return": account["net_return"],
        "net_pnl": account["equity_end"] - account["equity_start"],
        "gross_price_pnl": (
            account["equity_end"] - account["equity_start"]
            + account["fees"] + account["slippage_cost"] - account["funding"]
        ),
        "max_drawdown": account["max_drawdown"],
        "turnover_ratio": account["turnover_ratio"],
        "fees": account["fees"],
        "slippage_cost": account["slippage_cost"],
        "funding": account["funding"],
        "liquidations": account["liquidations"],
        "requested_decisions": decisions["requested"],
        "observed_decisions": decisions["observed"],
        "m22_verified_decisions": decisions["m22_verified"],
        "submitted_order_count": decisions["submitted_order_count"],
        "filled_order_count": decisions["filled_order_count"],
        "case_status": case["status"],
        "holding_minutes": case["holding_minutes"],
        "mfe_bps": case["mfe_bps"],
        "mae_bps": case["mae_bps"],
        "case_realized_pnl": case["realized_pnl"],
    }


def _identity(control: dict, x1: dict) -> dict[str, bool]:
    control_source = control["source"]
    x1_source = x1["source"]
    control_execution = control["execution"]
    x1_execution = x1["execution"]
    return {
        "same_protocol": control["protocol_id"] == x1["protocol_id"],
        "same_source_snapshot": (
            control_source["snapshot_hash"] == x1_source["snapshot_hash"]
            and control_source["marker_sha256"] == x1_source["marker_sha256"]
        ),
        "same_window": (
            control_source["window_id"] == x1_source["window_id"]
            and control_source["window_ns"] == x1_source["window_ns"]
        ),
        "same_scenario_seed_and_costs": all(
            control_execution[key] == x1_execution[key]
            for key in ("scenario", "scenario_costs")
        ) and control_execution.get("seed", 20261026) == x1_execution.get("seed", 20261026) == 20261026,
        "x1_seed_matches_native_runner_default": (
            "seed" not in x1_execution or x1_execution["seed"] == 20261026
        ),
        "same_account_and_risk": all(
            control_execution[key] == x1_execution[key]
            for key in ("starting_balance", "risk_budget_notional", "max_leverage")
        ),
        "same_scaler": all(
            control_execution[key] == x1_execution[key]
            for key in ("scaler_hash", "scaler_fit_window_hash")
        ),
        "same_feature_recipe": (
            control["verified_feature_recipe_sha256"]
            == x1["verified_feature_recipe_sha256"]
        ),
        "same_configured_target_exposure": (
            control["teacher_recipe"]["target_exposure"]
            == x1["teacher_recipe"]["target_exposure"]
            == 0.3
        ),
    }


def _account_metrics(report: dict) -> dict:
    account = report["account"]
    decisions = report["decisions"]
    return {
        "net_return": account["net_return"],
        "net_pnl": account["equity_end"] - account["equity_start"],
        "gross_price_pnl": (
            account["equity_end"] - account["equity_start"]
            + account["fees"] + account["slippage_cost"] - account["funding"]
        ),
        "max_drawdown": account["max_drawdown"],
        "turnover_ratio": account["turnover_ratio"],
        "fees": account["fees"],
        "slippage_cost": account["slippage_cost"],
        "funding": account["funding"],
        "liquidations": account["liquidations"],
        "requested_decisions": decisions["requested"],
        "m22_verified_decisions": decisions["m22_verified"],
        "trade_cases": decisions["native_trade_cases"],
    }


def _ema_reversal_reports() -> dict[tuple[str, str], tuple[Path, dict]]:
    reports = {}
    for path in ROOT.glob(".artifacts/route-a-ema-reversal-task1-*-reverse-*/native-run.json"):
        report = _json(path)
        recipe = report.get("teacher_recipe", {})
        if recipe.get("ema_reversal_mode") != "REVERSE_AFTER_FLAT_V1":
            continue
        key = (report["source"]["window_id"], report["execution"]["scenario"])
        if key in reports:
            raise ValueError(f"duplicate EMA reversal report for {key}")
        reports[key] = (path, report)
    expected = {(window, scenario) for window in WINDOWS for scenario in SCENARIOS}
    if set(reports) != expected:
        raise ValueError(f"EMA reversal reports do not cover all six S06 cells: {set(reports)}")
    return reports


def run() -> dict:
    rows = []
    for window in WINDOWS:
        for scenario in SCENARIOS:
            x1_path = X1_ROOT / window / scenario / "native-run.json"
            x1 = _json(x1_path)
            if (x1["teacher_recipe"]["exit_variant"] != "X1"
                    or x1["teacher"] != "T02_ROC_20_60_COST_HYSTERESIS_V1"):
                raise ValueError(f"unexpected X1 comparator: {x1_path}")
            for control_id in CONTROLS:
                run_dir = CONTROL_ROOT / window / scenario / control_id
                control_path = run_dir / "native-run.json"
                control = _json(control_path)
                case = _case(run_dir / "teacher-research-1m/native-trades.jsonl")
                identity = _identity(control, x1)
                if not all(identity.values()):
                    raise ValueError(f"native pair identity mismatch: {control_path}: {identity}")
                for label, report in (("control", control), ("X1", x1)):
                    decisions = report["decisions"]
                    if (decisions["requested"] != decisions["observed"]
                            or decisions["observed"] != decisions["m22_verified"]
                            or report["account"]["liquidations"] != 0):
                        raise ValueError(f"native verification failed for {label}: {control_path}")
                control_metrics = _metrics(control, case)
                x1_metrics = {
                    "net_return": x1["account"]["net_return"],
                    "net_pnl": x1["account"]["equity_end"] - x1["account"]["equity_start"],
                    "gross_price_pnl": (
                        x1["account"]["equity_end"] - x1["account"]["equity_start"]
                        + x1["account"]["fees"] + x1["account"]["slippage_cost"]
                        - x1["account"]["funding"]
                    ),
                    "max_drawdown": x1["account"]["max_drawdown"],
                    "fees": x1["account"]["fees"],
                    "slippage_cost": x1["account"]["slippage_cost"],
                    "funding": x1["account"]["funding"],
                    "turnover_ratio": x1["account"]["turnover_ratio"],
                    "trade_cases": x1["decisions"]["native_trade_cases"],
                    "requested_decisions": x1["decisions"]["requested"],
                    "m22_verified_decisions": x1["decisions"]["m22_verified"],
                    "liquidations": x1["account"]["liquidations"],
                }
                rows.append({
                    "window": window,
                    "scenario": scenario,
                    "control_id": control_id,
                    "control": control_metrics,
                    "x1": x1_metrics,
                    "net_return_delta_pp_vs_x1": (
                        control_metrics["net_return"] - x1_metrics["net_return"]
                    ) * 100,
                    "identity_checks": identity,
                    "native_report": str(control_path.relative_to(ROOT).as_posix()),
                })

    result = {
        "schema": "rles.route-a.fixed-side-vs-x1-s06.v1",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "economic_status": "SCENARIO_ONLY",
        "source_use": "already-consumed S06 Train; no Test/Holdout access",
        "risk_comparison": (
            "same configured 0.3 target exposure, account risk budget, leverage, "
            "starting balance, source, scaler, costs, and native barriers; realized "
            "exposure path and holding time intentionally differ"
        ),
        "seed_note": (
            "The historical X1 reports predate the execution-seed field. Their runner call omits a seed argument, "
            "so the native runner default 20261026 applies; the fixed controls explicitly use that same seed."
        ),
        "control_case_note": (
            "each fixed-side control retains one position through the block end; "
            "its single case is terminal-censored and account return includes native "
            "end-of-window mark-to-market, costs, and funding"
        ),
        "all_pairs_identity_verified": all(
            all(row["identity_checks"].values()) for row in rows
        ),
        "all_native_pairs_m22_complete_zero_liquidations": all(
            row["control"]["requested_decisions"]
            == row["control"]["m22_verified_decisions"]
            and row["control"]["liquidations"] == 0
            and row["x1"]["requested_decisions"]
            == row["x1"]["m22_verified_decisions"]
            and row["x1"]["liquidations"] == 0
            for row in rows
        ),
        "runs": rows,
    }
    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# S06 fixed-side controls paired with ROC/X1",
        "",
        "Classification: DEVELOPMENT_DIAGNOSTIC_ONLY; economic status: SCENARIO_ONLY.",
        "",
        "All pairs use the same 0.3 configured target, account risk budget, leverage, start balance, source snapshot, scaler, scenario costs, and full 1m M2.2 decision barriers. The realized exposure path differs by design. Fixed-side cases remain open at block end; their account return includes native mark-to-market, fees, slippage, and funding, so case PnL is terminal-censored rather than a closed-trade result.",
        "",
        "| Window | Scenario | ROC/X1 net | Fixed long net | Δ vs X1 | Fixed short net | Δ vs X1 | X1 fees + slip |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window in WINDOWS:
        for scenario in SCENARIOS:
            pair_rows = [r for r in rows if r["window"] == window and r["scenario"] == scenario]
            long_row = next(r for r in pair_rows if r["control_id"] == "FIXED_LONG_0.3")
            short_row = next(r for r in pair_rows if r["control_id"] == "FIXED_SHORT_0.3")
            x1 = long_row["x1"]
            lines.append(
                f"| {window} | {scenario} | {x1['net_return']:.3%} | "
                f"{long_row['control']['net_return']:.3%} | "
                f"{long_row['net_return_delta_pp_vs_x1']:+.3f} pp | "
                f"{short_row['control']['net_return']:.3%} | "
                f"{short_row['net_return_delta_pp_vs_x1']:+.3f} pp | "
                f"${x1['fees'] + x1['slippage_cost']:.2f} |"
            )
    lines += [
        "",
        "Control case mechanics and costs:",
        "",
        "| Window | Scenario | Side | End net PnL | DD | Fees | Slippage | Funding | Turnover | Holding min | MFE / MAE bps | Case status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        m = row["control"]
        lines.append(
            f"| {row['window']} | {row['scenario']} | {row['control_id']} | "
            f"${m['net_pnl']:.2f} | {m['max_drawdown']:.3%} | ${m['fees']:.2f} | "
            f"${m['slippage_cost']:.2f} | ${m['funding']:.2f} | "
            f"{m['turnover_ratio']:.3f} | {m['holding_minutes']:.0f} | "
            f"{m['mfe_bps']:.1f} / {m['mae_bps']:.1f} | {m['case_status']} |"
        )
    lines += [
        "",
        "This is a directional control, not an independent validation. It does not imply the better side could have been selected prospectively. Results compare trading decisions on consumed Train data only.",
        "",
        "ROC/X1 cost attribution (gross price PnL = net PnL + fees + slippage − funding):",
        "",
        "| Window | Scenario | Gross price PnL | Fees | Slippage | Funding | Net PnL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for window in WINDOWS:
        for scenario in SCENARIOS:
            x1 = next(
                row["x1"] for row in rows
                if row["window"] == window and row["scenario"] == scenario
            )
            lines.append(
                f"| {window} | {scenario} | ${x1['gross_price_pnl']:.2f} | "
                f"${x1['fees']:.2f} | ${x1['slippage_cost']:.2f} | "
                f"${x1['funding']:.2f} | ${x1['net_pnl']:.2f} |"
            )
    lines += [
        "",
        f"Per-run native reports: `{CONTROL_ROOT.relative_to(ROOT).as_posix()}/<window>/<scenario>/<control>/native-run.json`; paired source: `.artifacts/route-a-roc-hysteresis-task2/native/`; summary JSON: `{OUTPUT_JSON.relative_to(ROOT).as_posix()}`.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ema_reports = _ema_reversal_reports()
    ema_rows = []
    for row in rows:
        ema_path, ema = ema_reports[(row["window"], row["scenario"])]
        control_path = CONTROL_ROOT / row["window"] / row["scenario"] / row["control_id"] / "native-run.json"
        control = _json(control_path)
        identity = _identity(control, ema)
        identity["ema_is_reversal_mode"] = (
            ema["teacher_recipe"].get("ema_reversal_mode") == "REVERSE_AFTER_FLAT_V1"
        )
        if not all(identity.values()):
            raise ValueError(f"EMA/control identity mismatch: {control_path}: {identity}")
        control_metrics = row["control"]
        ema_metrics = _account_metrics(ema)
        if (ema_metrics["requested_decisions"] != ema_metrics["m22_verified_decisions"]
                or ema_metrics["liquidations"] != 0):
            raise ValueError(f"EMA native verification failed: {ema_path}")
        ema_rows.append({
            "window": row["window"],
            "scenario": row["scenario"],
            "control_id": row["control_id"],
            "control": control_metrics,
            "ema_reversal": ema_metrics,
            "net_return_delta_pp_vs_ema_reversal": (
                control_metrics["net_return"] - ema_metrics["net_return"]
            ) * 100,
            "identity_checks": identity,
            "control_report": str(control_path.relative_to(ROOT).as_posix()),
            "ema_reversal_report": str(ema_path.relative_to(ROOT).as_posix()),
        })
    ema_result = {
        "schema": "rles.route-a.fixed-side-vs-ema-reversal-s06.v1",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "economic_status": "SCENARIO_ONLY",
        "source_use": "already-consumed S06 Train; no Test/Holdout access",
        "seed_note": (
            "Historical EMA reversal reports omit the execution-seed field. Their runner call uses the native default "
            "20261026; fixed controls explicitly use the same seed."
        ),
        "all_pairs_identity_verified": all(
            all(item["identity_checks"].values()) for item in ema_rows
        ),
        "all_native_pairs_m22_complete_zero_liquidations": all(
            item["control"]["requested_decisions"] == item["control"]["m22_verified_decisions"]
            and item["control"]["liquidations"] == 0
            and item["ema_reversal"]["requested_decisions"]
            == item["ema_reversal"]["m22_verified_decisions"]
            and item["ema_reversal"]["liquidations"] == 0
            for item in ema_rows
        ),
        "runs": ema_rows,
    }
    ema_json = CONTROL_ROOT / "paired_fixed_side_vs_ema_reversal.json"
    ema_md = CONTROL_ROOT / "paired_fixed_side_vs_ema_reversal.md"
    ema_json.write_text(
        json.dumps(ema_result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ema_lines = [
        "# S06 fixed-side controls paired with EMA reverse-after-flat",
        "",
        "Classification: DEVELOPMENT_DIAGNOSTIC_ONLY; economic status: SCENARIO_ONLY.",
        "",
        "Pairs match the source snapshot, feature recipe, scaler, 0.3 configured target exposure, account risk budget, leverage, start balance, cost scenario, and full 1m M2.2 barriers. Historical EMA reports omit the seed field; the native runner call uses its default seed 20261026, also explicitly used for controls. Fixed-side control returns include end-of-window mark-to-market on one terminal-censored position.",
        "",
        "| Window | Scenario | EMA reverse net | Fixed LONG net | Δ vs EMA | Fixed SHORT net | Δ vs EMA | EMA fees + slip |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window in WINDOWS:
        for scenario in SCENARIOS:
            pair_rows = [r for r in ema_rows if r["window"] == window and r["scenario"] == scenario]
            long_row = next(r for r in pair_rows if r["control_id"] == "FIXED_LONG_0.3")
            short_row = next(r for r in pair_rows if r["control_id"] == "FIXED_SHORT_0.3")
            ema = long_row["ema_reversal"]
            ema_lines.append(
                f"| {window} | {scenario} | {ema['net_return']:.3%} | "
                f"{long_row['control']['net_return']:.3%} | "
                f"{long_row['net_return_delta_pp_vs_ema_reversal']:+.3f} pp | "
                f"{short_row['control']['net_return']:.3%} | "
                f"{short_row['net_return_delta_pp_vs_ema_reversal']:+.3f} pp | "
                f"${ema['fees'] + ema['slippage_cost']:.2f} |"
            )
    ema_lines += [
        "",
        "The better fixed side changes with the window, so these outcomes do not support choosing a direction after observing the block. This is a diagnostic control, not independent validation or a teacher candidate.",
        "",
        f"Paired JSON: `{ema_json.relative_to(ROOT).as_posix()}`; per-run reports: `{CONTROL_ROOT.relative_to(ROOT).as_posix()}/<window>/<scenario>/<control>/native-run.json` and `.artifacts/route-a-ema-reversal-task1-*-reverse-*/native-run.json`.",
    ]
    ema_md.write_text("\n".join(ema_lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2, allow_nan=False))
