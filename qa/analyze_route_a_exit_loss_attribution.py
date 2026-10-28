from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / ".artifacts/route-a-exit-economics-task3/per_trade_exit_metrics.jsonl"
REFERENCE_PATH = ROOT / ".artifacts/route-a-exit-economics-task3/x1_x2_exit_economics.json"
OUTPUT_DIR = ROOT / ".artifacts/route-a-exit-loss-attribution-task3-20261027"
COMPLETE = "ORDINARY_COMPLETE"
ENTRY_NOTIONAL_ROUNDING_DECIMALS = 6


def classify_loss(case: dict[str, Any]) -> str | None:
    if case["status"] != COMPLETE or float(case["realized_pnl"]) >= 0:
        return None
    if float(case["gross_price_return_bps"]) > 0:
        return "positive_price_pnl_erased_by_fees_or_funding"
    if float(case["mfe_bps"]) >= float(case["round_trip_cost_bps"]):
        return "large_mfe_giveback_or_potential_late_exit"
    if float(case["mae_bps"]) > float(case["mfe_bps"]):
        return "adverse_excursion_dominant"
    return "negative_price_or_weak_edge"


def classify_exit_timing(
    *, x1_hold_minutes: float, x2_hold_minutes: float,
    x1_net_return_bps: float, x2_net_return_bps: float,
) -> str:
    hold_delta = x2_hold_minutes - x1_hold_minutes
    return_delta = x2_net_return_bps - x1_net_return_bps
    if return_delta > 1e-9:
        if hold_delta > 0:
            return "x1_potentially_early_x2_later_better"
        if hold_delta < 0:
            return "x1_potentially_late_x2_earlier_better"
        return "x2_better_without_hold_change"
    if return_delta < -1e-9:
        if hold_delta > 0:
            return "x2_later_worse_x1_shorter_better"
        if hold_delta < 0:
            return "x2_earlier_worse_x1_longer_better"
        return "x1_better_without_hold_change"
    return "net_return_tie"


def entry_key(case: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(case["teacher_id"]), str(case["block"]), str(case["scenario"]),
        int(case["entry_fill_ts_ns"]), int(case["side"]),
    )


def loss_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if case["status"] == COMPLETE:
            groups[(
                str(case["teacher_id"]), str(case["exit_variant"]),
                str(case["block"]), str(case["scenario"]),
            )].append(case)

    result: dict[str, Any] = {}
    for (teacher, variant, block, scenario), selected in sorted(groups.items()):
        losses = [case for case in selected if float(case["realized_pnl"]) < 0]
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in losses:
            category = classify_loss(case)
            assert category is not None
            by_category[category].append(case)
        loss_amount = sum(-float(case["realized_pnl"]) for case in losses)
        result[f"{teacher}/{variant}/{block}/{scenario}"] = {
            "complete_trades": len(selected),
            "loss_trades": len(losses),
            "net_loss_amount": loss_amount,
            "loss_categories": {
                category: {
                    "trade_count": len(items),
                    "net_loss_amount": sum(-float(case["realized_pnl"]) for case in items),
                    "share_of_group_loss": (
                        sum(-float(case["realized_pnl"]) for case in items) / loss_amount
                        if loss_amount else 0.0
                    ),
                    "mean_gross_price_return_bps": mean(
                        float(case["gross_price_return_bps"]) for case in items
                    ),
                    "mean_mfe_bps": mean(float(case["mfe_bps"]) for case in items),
                    "mean_mae_bps": mean(float(case["mae_bps"]) for case in items),
                    "mean_profit_capture_ratio": mean(
                        float(case["gross_price_profit_capture_ratio"])
                        for case in items
                        if case["gross_price_profit_capture_ratio"] is not None
                    ) if any(case["gross_price_profit_capture_ratio"] is not None for case in items) else None,
                    "fees": sum(float(case["fees"]) for case in items),
                    "funding": sum(float(case["funding"]) for case in items),
                    "slippage_attribution": sum(float(case["slippage_cost"]) for case in items),
                }
                for category, items in sorted(by_category.items())
            },
        }
    return result


def side_economics_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if case["status"] != COMPLETE:
            continue
        raw_side = case["side"]
        if raw_side in (1, "1", "LONG"):
            side = "LONG"
        elif raw_side in (-1, "-1", "SHORT"):
            side = "SHORT"
        else:
            raise ValueError(f"unknown trade side: {raw_side!r}")
        groups[(
            str(case["teacher_id"]), str(case["exit_variant"]),
            str(case["block"]), str(case["scenario"]), side,
        )].append(case)

    result: dict[str, Any] = {}
    for (teacher, variant, block, scenario, side), selected in sorted(groups.items()):
        losses = [case for case in selected if float(case["realized_pnl"]) < 0]
        wins = [case for case in selected if float(case["realized_pnl"]) > 0]
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in losses:
            category = classify_loss(case)
            assert category is not None
            by_category[category].append(case)
        currencies = {str(case["equity_currency"]) for case in selected}
        if len(currencies) != 1:
            raise RuntimeError("side economics group mixes account currencies")
        result[f"{teacher}/{variant}/{block}/{scenario}/{side}"] = {
            "currency": next(iter(currencies)),
            "complete_trades": len(selected),
            "wins": len(wins),
            "loss_trades": len(losses),
            "win_rate": len(wins) / len(selected),
            "net_pnl": sum(float(case["realized_pnl"]) for case in selected),
            "gross_cash_pnl_before_fees_and_funding": sum(
                float(case["gross_cash_pnl_before_fees_and_funding"])
                for case in selected
            ),
            "fees": sum(float(case["fees"]) for case in selected),
            "funding": sum(float(case["funding"]) for case in selected),
            "slippage_attribution": sum(
                float(case["slippage_cost"]) for case in selected
            ),
            "loss_categories": {
                category: {
                    "trade_count": len(items),
                    "net_loss_amount": sum(-float(case["realized_pnl"]) for case in items),
                }
                for category, items in sorted(by_category.items())
            },
        }
    return result


def matched_timing(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, dict[tuple[str, str, str, int, int], dict[str, Any]]] = {
        "X1": {}, "X2": {},
    }
    for case in cases:
        if case["status"] != COMPLETE or case["exit_variant"] not in by_variant:
            continue
        key = entry_key(case)
        variant = str(case["exit_variant"])
        if key in by_variant[variant]:
            raise RuntimeError(f"duplicate exact-entry case: {key}")
        by_variant[variant][key] = case

    matches_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    observed_pairs: Counter[tuple[str, str, str]] = Counter()
    for key in sorted(set(by_variant["X1"]) & set(by_variant["X2"])):
        x1, x2 = by_variant["X1"][key], by_variant["X2"][key]
        if round(float(x1["entry_notional"]), ENTRY_NOTIONAL_ROUNDING_DECIMALS) != round(
            float(x2["entry_notional"]), ENTRY_NOTIONAL_ROUNDING_DECIMALS,
        ):
            continue
        category = classify_exit_timing(
            x1_hold_minutes=float(x1["holding_minutes"]),
            x2_hold_minutes=float(x2["holding_minutes"]),
            x1_net_return_bps=float(x1["net_case_return_bps"]),
            x2_net_return_bps=float(x2["net_case_return_bps"]),
        )
        matches_by_group[(key[0], key[1], key[2])].append({
            "x1_loss": float(x1["realized_pnl"]) < 0,
            "category": category,
            "return_delta_bps": (
                float(x2["net_case_return_bps"]) - float(x1["net_case_return_bps"])
            ),
        })
        observed_pairs[(key[0], key[2], key[1])] += 1

    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    expected_pairs = Counter({
        (row["teacher_id"], row["scenario"], row["block"]):
            int(row["same_entry_counterfactual"]["same_exact_entry_complete_pairs"])
        for row in reference["pairs"]
    })
    if observed_pairs != expected_pairs:
        raise RuntimeError("matched exact-entry counts differ from the existing X1/X2 report")

    result: dict[str, Any] = {}
    for (teacher, block, scenario), items in sorted(matches_by_group.items()):
        x1_losses = [item for item in items if item["x1_loss"]]
        counts = Counter(item["category"] for item in x1_losses)
        result[f"{teacher}/{block}/{scenario}"] = {
            "matched_complete_entry_pairs": len(items),
            "matched_pairs_with_x1_net_loss": len(x1_losses),
            "x1_loss_pair_timing_counts": dict(sorted(counts.items())),
            "x1_loss_pair_net_return_delta_bps": {
                category: {
                    "mean": mean(
                        item["return_delta_bps"] for item in x1_losses
                        if item["category"] == category
                    ),
                    "sum": sum(
                        item["return_delta_bps"] for item in x1_losses
                        if item["category"] == category
                    ),
                }
                for category in sorted(counts)
            },
        }
    return result



def cashflow_sign_summary(side_economics: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, row in side_economics.items():
        scenario = key.split("/")[3]
        grouped[scenario].append(row)

    result: dict[str, Any] = {}
    for scenario, rows in sorted(grouped.items()):
        result[scenario] = {
            "populated_side_cells": len(rows),
            "positive_net_pnl_cells": sum(float(row["net_pnl"]) > 0 for row in rows),
            "negative_net_pnl_cells": sum(float(row["net_pnl"]) < 0 for row in rows),
            "zero_net_pnl_cells": sum(float(row["net_pnl"]) == 0 for row in rows),
            "gross_positive_net_negative_cells": sum(
                float(row["gross_cash_pnl_before_fees_and_funding"]) > 0
                and float(row["net_pnl"]) < 0
                for row in rows
            ),
            "gross_nonpositive_net_negative_cells": sum(
                float(row["gross_cash_pnl_before_fees_and_funding"]) <= 0
                and float(row["net_pnl"]) < 0
                for row in rows
            ),
        }
    return result

def build_report() -> dict[str, Any]:
    with INPUT_PATH.open(encoding="utf-8") as stream:
        cases = [json.loads(line) for line in stream if line.strip()]
    if not cases or {case["exit_variant"] for case in cases} != {"X1", "X2"}:
        raise RuntimeError("expected existing X1/X2 per-trade output")
    side_economics = side_economics_summary(cases)
    return {
        "schema": "route-a.exit-loss-attribution.v2",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "economic_status": "SCENARIO_ONLY",
        "source_use": "existing consumed S06 Train per-trade outputs only; no market-source read or native rerun",
        "input_path": str(INPUT_PATH.relative_to(ROOT)),
        "input_rows": len(cases),
        "interpretation_limits": [
            "Cross-group count and loss-amount totals sum teacher/exit/block/scenario runs; the same entry event can appear more than once, so aggregate counts are not unique-market-trade counts.",
            "Gross fill-to-fill cash PnL includes slippage; slippage attribution is reported separately and must not be subtracted a second time.",
            "Intrabar MFE uses bar highs/lows and is an upper bound, not proof of an executable decision-time exit.",
            "Exact-entry comparisons match teacher, block, scenario, entry fill time, side, and entry notional rounded to six decimal places. Censored pairs are excluded from realized-exit deltas. They compare actual trade cashflows but do not estimate a randomized causal effect or later account re-entry paths.",
            "Loss labels are descriptive. Adverse excursion dominance does not prove that the signal, rather than the exit, caused the loss.",
        ],
        "losses_by_teacher_variant_block_scenario": loss_summary(cases),
        "side_economics_by_teacher_variant_block_scenario": side_economics,
        "cashflow_sign_by_scenario": cashflow_sign_summary(side_economics),
        "matched_x1_x2_exit_timing_by_teacher_block_scenario": matched_timing(cases),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Native X1/X2 Exit Loss Attribution",
        "",
        "DEVELOPMENT_DIAGNOSTIC_ONLY; SCENARIO_ONLY. Existing consumed-S06 per-trade output only; no market-source read or native rerun.",
        "",
        "## Completed-trade loss composition by teacher, exit, block, and scenario",
        "",
        "| Teacher | Exit | Block | Scenario | Complete | Losses | Category | Count | Share of group loss |",
        "|---|---|---|---|---:|---:|---|---:|---:|",
    ]
    for group, data in sorted(report["losses_by_teacher_variant_block_scenario"].items()):
        teacher, variant, block, scenario = group.split("/")
        for category, values in sorted(data["loss_categories"].items()):
            lines.append(
                f"| {teacher} | {variant} | {block} | {scenario} | "
                f"{data['complete_trades']} | {data['loss_trades']} | "
                f"{category} | {values['trade_count']} | "
                f"{values['share_of_group_loss']:.1%} |"
            )

    lines.extend([
        "",
        "## Per-side economics by teacher, exit, block, and scenario",
        "",
        "| Teacher | Exit | Block | Scenario | Side | Complete | Wins | Win rate | Gross before fees/funding | Fees | Funding | Net PnL | Slippage attribution |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for group, data in sorted(report["side_economics_by_teacher_variant_block_scenario"].items()):
        teacher, variant, block, scenario, side = group.split("/")
        lines.append(
            f"| {teacher} | {variant} | {block} | {scenario} | {side} | "
            f"{data['complete_trades']} | {data['wins']} | {data['win_rate']:.1%} | "
            f"{data['gross_cash_pnl_before_fees_and_funding']:.2f} {data['currency']} | "
            f"{data['fees']:.2f} {data['currency']} | {data['funding']:.2f} {data['currency']} | "
            f"{data['net_pnl']:.2f} {data['currency']} | "
            f"{data['slippage_attribution']:.2f} {data['currency']} |"
        )

    lines.extend([
        "",
        "## Loss composition by side",
        "",
        "| Teacher | Exit | Block | Scenario | Side | Complete | Losses | Category | Count | Loss amount |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|",
    ])
    for group, data in sorted(report["side_economics_by_teacher_variant_block_scenario"].items()):
        teacher, variant, block, scenario, side = group.split("/")
        for category, values in sorted(data["loss_categories"].items()):
            lines.append(
                f"| {teacher} | {variant} | {block} | {scenario} | {side} | "
                f"{data['complete_trades']} | {data['loss_trades']} | "
                f"{category} | {values['trade_count']} | "
                f"{values['net_loss_amount']:.2f} {data['currency']} |"
            )

    early = "x1_potentially_early_x2_later_better"
    late = "x1_potentially_late_x2_earlier_better"
    lines.extend([
        "",
        "## Exact-entry timing comparison on X1 losing cases by block",
        "",
        "| Teacher | Block | Scenario | Matched pairs | X1-loss pairs | X1 potentially early / X2 later better | X1 potentially late / X2 earlier better | Other or tie |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for group, data in sorted(report["matched_x1_x2_exit_timing_by_teacher_block_scenario"].items()):
        counts = data["x1_loss_pair_timing_counts"]
        other = sum(value for key, value in counts.items() if key not in (early, late))
        teacher, block, scenario = group.split("/")
        lines.append(
            f"| {teacher} | {block} | {scenario} | "
            f"{data['matched_complete_entry_pairs']} | "
            f"{data['matched_pairs_with_x1_net_loss']} | {counts.get(early, 0)} | "
            f"{counts.get(late, 0)} | {other} |"
        )


    lines.extend([
        "",
        "## Cashflow sign by scenario",
        "",
        "| Scenario | Populated side cells | Gross positive, net negative | Gross nonpositive, net negative | Net positive | Net negative | Net zero |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for scenario, data in sorted(report["cashflow_sign_by_scenario"].items()):
        lines.append(
            f"| {scenario} | {data['populated_side_cells']} | "
            f"{data['gross_positive_net_negative_cells']} | "
            f"{data['gross_nonpositive_net_negative_cells']} | "
            f"{data['positive_net_pnl_cells']} | "
            f"{data['negative_net_pnl_cells']} | "
            f"{data['zero_net_pnl_cells']} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {note}" for note in report["interpretation_limits"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "exit_loss_attribution.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (OUTPUT_DIR / "exit_loss_attribution.md").write_text(
        render_markdown(report), encoding="utf-8",
    )
    print(json.dumps({
        "classification": report["classification"],
        "input_rows": report["input_rows"],
        "loss_groups": len(report["losses_by_teacher_variant_block_scenario"]),
        "matched_groups": len(report["matched_x1_x2_exit_timing_by_teacher_block_scenario"]),
        "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
