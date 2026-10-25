"""Verify retained native receipts and summarize the fixed four-block diagnostic."""
from __future__ import annotations

import gzip
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".artifacts/route-a-causal-teacher-fit-native-5m-20261025-archive-clock"
OLD = ROOT / ".artifacts/route-a-causal-teacher-fit-native-5m-20261025"


def checked_json(path):
    encoded = path.read_bytes()
    if path.with_suffix(path.suffix + ".sha256").read_text().strip() != sha256(encoded).hexdigest():
        raise ValueError(f"report checksum mismatch: {path}")
    return json.loads(encoded)


def main():
    destination = OUT / "verification.json"
    if destination.exists():
        raise FileExistsError("verification is immutable")
    report = checked_json(OUT / "report.json")
    recipe = checked_json(OUT / "recipe.json")
    expected = set(product(range(1, 5), ("base", "stress"), ("flat", "p0_v2", "p0_recovery_v3")))
    observed = {(r["block"], r["scenario"], r["arm"]) for r in report["runs"]}
    if observed != expected or len(report["runs"]) != len(expected):
        raise ValueError("matched native matrix is incomplete or duplicated")
    verified_barriers = 0
    unchanged = 0
    for row in report["runs"]:
        name = f"block-{row['block']:02d}-{row['scenario']}-{row['arm']}"
        if checked_json(OUT / f"{name}.json") != row:
            raise ValueError("aggregate report differs from per-arm receipt")
        encoded = gzip.decompress((OUT / f"{name}-barriers.json.gz").read_bytes())
        if sha256(encoded).hexdigest() != row["trace_sha256"]:
            raise ValueError("native barrier trace checksum mismatch")
        trace = json.loads(encoded)
        if (len(trace) != row["metrics"]["barriers"] or len(trace) != row["m22"]["barriers"]
                or row["m22"]["authority"] != "retained_m22_session_snapshots"
                or row["metrics"]["liquidations"] != 0
                or row["metrics"]["summary"]["economic_status"] != "SCENARIO_ONLY"
                or any(item["scenario_only"] is not True for item in trace)):
            raise ValueError("native evidence coverage/risk/authority mismatch")
        verified_barriers += len(trace)
        previous = OLD / f"{name}.json"
        if previous.exists():
            old = checked_json(previous)
            if old["metrics"] != row["metrics"] or old["host_metrics"] != row["host_metrics"]:
                raise ValueError("source clock repair changed an already-valid block's economics")
            unchanged += 1
    summaries = {}
    for scenario, arm in product(("base", "stress"), ("p0_v2", "p0_recovery_v3")):
        rows = sorted((r for r in report["runs"] if r["scenario"] == scenario and r["arm"] == arm),
                      key=lambda r: r["block"])
        values = [r["metrics"]["net_return"] for r in rows]
        # Exact empirical 4-block bootstrap distribution; exploratory only.
        bootstrap = [sum(values[i] for i in draw) / 4 for draw in product(range(4), repeat=4)]
        summaries[f"{scenario}:{arm}"] = {
            "block_net_returns": values, "positive_blocks": sum(v > 0 for v in values),
            "mean_block_return": float(np.mean(values)),
            "exploratory_block_mean_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "max_native_position_age_minutes": max(r["max_position_age_minutes"] for r in rows),
        }
    result = {
        "qualification": "NO_ELIGIBLE_ALPHA_TEACHER",
        "reason": "three losing blocks; exploratory confidence interval includes zero; Train-fit only",
        "verified_runs": len(report["runs"]), "verified_native_barriers": verified_barriers,
        "unchanged_prior_valid_runs": unchanged, "summaries": summaries,
        "recipe_sha256": sha256((OUT / "recipe.json").read_bytes()).hexdigest(),
        "entry_cadence_minutes": recipe["entry_cadence_minutes"],
        "economic_status": "SCENARIO_ONLY", "student_promotion": False,
        "limitations": ["four Train-fit blocks are not independent unseen screen evidence",
                        "seeds and cost scenarios do not increase independent market sample size",
                        "no fixed-long/short beta attribution or full G1/G2 qualification performed"],
    }
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode()
    destination.write_bytes(encoded)
    destination.with_suffix(".json.sha256").write_text(sha256(encoded).hexdigest())
    print(encoded.decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
