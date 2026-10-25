"""Check every retained fresh Train native trace and publish descriptive results."""
from __future__ import annotations

import gzip
from hashlib import sha256
from itertools import product
import json
from math import log
from pathlib import Path

import numpy as np

import run_route_a_fresh_train_native_20261025 as runner

OUT = runner.OUT


def scenario_case_outcomes(evidence: list[dict]) -> tuple[list[float], bool]:
    """Account-equity flat-to-flat diagnostics, never student labels."""
    entry_equity = None
    previous = 0.0
    completed = []
    for row in evidence:
        after = row["actual_exposure_after"]
        if row["has_external_cashflow"]:
            raise ValueError("scenario path has an unmodelled external cashflow")
        if previous * after < -1e-8:
            raise ValueError("atomic native reversal prevents unambiguous case attribution")
        if abs(previous) <= 1e-8 and abs(after) > 1e-8:
            if not row["filled_order_ids"] or entry_equity is not None:
                raise ValueError("scenario entry lacks a unique native fill")
            entry_equity = row["native_equity_before"]
        elif abs(previous) > 1e-8 and abs(after) <= 1e-8:
            if not row["filled_order_ids"] or entry_equity is None:
                raise ValueError("scenario exit lacks a matched native fill")
            completed.append(log(row["native_equity_after"] / entry_equity) * 10_000)
            entry_equity = None
        previous = after
    return completed, entry_equity is not None


def checked_json(path: Path) -> dict:
    encoded = path.read_bytes()
    if path.with_suffix(path.suffix + ".sha256").read_text().strip() != sha256(encoded).hexdigest():
        raise ValueError(f"receipt checksum mismatch: {path}")
    return json.loads(encoded)


def main() -> int:
    destination = OUT / "verification.json"
    if destination.exists():
        raise FileExistsError("native verification receipt is immutable")
    prereg = checked_json(runner.PREREG)
    recipe = checked_json(OUT / "recipe.json")
    report = checked_json(OUT / "report.json")
    if (recipe["prereg_sha256"] != sha256(runner.PREREG.read_bytes()).hexdigest()
            or recipe["source_marker_sha256"] != sha256(
                Path(str(runner.MARKET) + ".COMPLETED.json").read_bytes(),
            ).hexdigest()
            or recipe["windows"] != prereg["windows"]
            or recipe["modes"] != prereg["execution"]["scenario_modes"]
            or set(recipe["arms"]) != set(prereg["arms"])
            or recipe["economic_status"] != "SCENARIO_ONLY"
            or recipe["test_holdout_read"] is not False):
        raise ValueError("native source or frozen matrix identity mismatch")
    expected = set(product(range(1, 5), ("base", "stress"), tuple(prereg["arms"])))
    rows = {(r["block"], r["scenario"], r["arm"]): r for r in report["runs"]}
    if len(report["runs"]) != len(expected) or set(rows) != expected:
        raise ValueError("native matrix has missing or duplicate arms")
    verified_barriers = 0
    case_rows = {}
    for (block, scenario, arm), row in rows.items():
        name = f"block-{block:02d}-{scenario}-{arm}"
        if checked_json(OUT / f"{name}.json") != row:
            raise ValueError("per-arm receipt differs from aggregate")
        encoded = gzip.decompress((OUT / f"{name}-barriers.json.gz").read_bytes())
        if sha256(encoded).hexdigest() != row["trace_sha256"]:
            raise ValueError("native trace digest mismatch")
        evidence = json.loads(encoded)
        expected_length = row["metrics"]["barriers"]
        if (len(evidence) != expected_length
                or row["m22"]["authority"] != "retained_m22_session_snapshots"
                or row["m22"]["barriers"] != expected_length
                or row["metrics"]["liquidations"] != 0
                or row["metrics"]["summary"]["economic_status"] != "SCENARIO_ONLY"
                or row["metrics"]["summary"]["completed_native_trade_case_count"] != 0
                or row["metrics"]["summary"]["executable_label_count"] != 0
                or any(item["scenario_only"] is not True for item in evidence)
                or runner.scenario_cycles(evidence) != row["scenario_cycle_audit"]):
            raise ValueError("native M2.2, risk, scenario-only, or case evidence mismatch")
        outcomes, censored = scenario_case_outcomes(evidence)
        if (len(outcomes) != row["scenario_cycle_audit"][
                "complete_scenario_cycles_not_education_cases"]
                or int(censored) != row["scenario_cycle_audit"][
                    "open_case_censored_at_block_end"]):
            raise ValueError("scenario flat-to-flat case audit differs from native transitions")
        case_rows[(block, scenario, arm)] = outcomes
        verified_barriers += len(evidence)
    summaries = {}
    for scenario, arm in product(("base", "stress"), prereg["arms"]):
        values = [rows[(block, scenario, arm)]["metrics"]["net_return"]
                  for block in range(1, 5)]
        bootstrap = [sum(values[index] for index in draw) / 4
                     for draw in product(range(4), repeat=4)]
        summaries[f"{scenario}:{arm}"] = {
            "block_net_returns": values,
            "positive_blocks": sum(value > 0 for value in values),
            "mean_block_return": float(np.mean(values)),
            "exploratory_block_mean_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "complete_scenario_cycles": sum(
                rows[(block, scenario, arm)]["scenario_cycle_audit"][
                    "complete_scenario_cycles_not_education_cases"
                ] for block in range(1, 5)
            ),
            "open_censored_at_block_end": sum(
                rows[(block, scenario, arm)]["scenario_cycle_audit"][
                    "open_case_censored_at_block_end"
                ] for block in range(1, 5)
            ),
            "positive_scenario_cycles_not_education_cases": sum(
                value > 0 for block in range(1, 5)
                for value in case_rows[(block, scenario, arm)]
            ),
            "negative_scenario_cycles_not_education_cases": sum(
                value < 0 for block in range(1, 5)
                for value in case_rows[(block, scenario, arm)]
            ),
        }
    comparisons = []
    for block in range(1, 5):
        for scenario in ("base", "stress"):
            baseline_long = rows[(block, scenario, "fixed_long")]["metrics"]["net_return"]
            baseline_short = rows[(block, scenario, "fixed_short")]["metrics"]["net_return"]
            for arm in ("p0_recovery_v3_both_exit", "p0_recovery_v3_either_exit"):
                value = rows[(block, scenario, arm)]["metrics"]["net_return"]
                comparisons.append({
                    "block": block, "scenario": scenario, "arm": arm,
                    "net_return": value,
                    "delta_vs_fixed_long": value - baseline_long,
                    "delta_vs_fixed_short": value - baseline_short,
                    "exceeds_both_sides": value > max(baseline_long, baseline_short),
                })
    result = {
        "qualification": "NOT_FORMAL_G1_OR_G2_SCENARIO_ONLY",
        "formal_student_promotion": False,
        "verified_runs": len(rows), "verified_native_barriers": verified_barriers,
        "recipe_sha256": sha256((OUT / "recipe.json").read_bytes()).hexdigest(),
        "summaries": summaries, "comparisons": comparisons,
        "limitations": [
            "the four new Train-inner blocks are research screen, not Test/Holdout",
            "direct rule has no fitted/calibrated opportunity model or G1 negative-control matrix",
            "fixed sides have unequal exposure time and are descriptive controls",
            "public market archives do not prove historical account fee or margin authority",
            "scenario cycles cannot become OBSERVED_NATIVE_CASE education labels",
        ],
    }
    runner.harness.write_verified(destination, result)
    print(json.dumps({
        "qualification": result["qualification"],
        "verified_runs": result["verified_runs"],
        "verified_native_barriers": verified_barriers,
        "summaries": summaries,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
