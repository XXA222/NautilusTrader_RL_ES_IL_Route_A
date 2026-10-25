"""Frozen either-scale trend-exit hypothesis on consumed Train-fit only.

This is a one-shot mechanism falsification, not an unseen screen or teacher
qualification. The v3 candidate, size, risk overlay, cost scenarios, windows,
native execution, and cadence remain unchanged from the prior diagnostic.
"""
from __future__ import annotations

import gzip
from hashlib import sha256
import json
from pathlib import Path

from rles_route_a.teacher_research.candidate_events import CausalOpportunityGenerator
from rles_route_a.teachers.causal_position import CausalPositionTeacher
from rles_route_a.teachers.recovery import RecoveryTeacher
from rles_route_a.teachers.selector import FrozenTeacherSelector
from rles_route_a.teachers.wait import WaitTeacher

import run_route_a_causal_teacher_fit_native_20261025 as native

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".artifacts/route-a-causal-teacher-fit-native-5m-20261025-archive-clock"
OUT = ROOT / ".artifacts/route-a-either-scale-exit-fit-native-20261025"


def checked_json(path: Path) -> dict:
    encoded = path.read_bytes()
    if path.with_suffix(path.suffix + ".sha256").read_text().strip() != sha256(encoded).hexdigest():
        raise ValueError(f"checksum mismatch: {path}")
    return json.loads(encoded)


def scenario_cycles(evidence: list[dict]) -> dict[str, int]:
    """Count native fill-backed flat-to-flat paths, never teaching labels."""
    opened = False
    complete = 0
    for row in evidence:
        before = opened
        exposure = row["actual_exposure_after"]
        now = abs(exposure) > 1e-8
        if before != now:
            if not row["filled_order_ids"] or row["pending_order_ids_after"]:
                raise ValueError("scenario position transition lacks settled native fill")
            if before and not now:
                complete += 1
        opened = now
    return {"complete_scenario_cycles_not_education_cases": complete,
            "open_case_censored_at_block_end": int(opened)}


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"immutable experiment already exists: {OUT}")
    verification = checked_json(SOURCE / "verification.json")
    old = checked_json(SOURCE / "report.json")
    if verification["verified_runs"] != 24 or verification["qualification"] != "NO_ELIGIBLE_ALPHA_TEACHER":
        raise ValueError("prior native v3 evidence is not verified")
    hypothesis = {
        "hypothesis": "Either-scale adverse trend should close stale v3 positions more often than the old both-scale exit, without a forced holding-time stop",
        "frozen_prior_recipe_sha256": verification["recipe_sha256"],
        "frozen_prior_report_sha256": sha256((SOURCE / "report.json").read_bytes()).hexdigest(),
        "frozen_exit_policy_id": "p0-trend-invalidation-either-v1:" + float(.002).hex(),
        "not_new_oos": True,
        "qualification_policy": "Train-fit mechanism only; G1/G2 unchanged; no student promotion",
    }
    teacher = FrozenTeacherSelector(
        primary=CausalPositionTeacher(
            generator=CausalOpportunityGenerator(
                rule_version="route_a_candidates_v3_recovery",
                require_recovery_confirmation=True,
            ),
            entry_cadence_minutes=5, exit_scale_policy="either",
        ),
        recovery=RecoveryTeacher(), wait=WaitTeacher(),
    )
    native.run(
        OUT, entry_cadence_minutes=5, archive_clock=True,
        arms_override={"p0_recovery_v3_either_exit": teacher},
        extra_recipe=hypothesis,
    )
    report = checked_json(OUT / "report.json")
    recipe = checked_json(OUT / "recipe.json")
    if any(recipe[key] != value for key, value in hypothesis.items()):
        raise ValueError("frozen hypothesis changed")
    expected = {(block, scenario, "p0_recovery_v3_either_exit")
                for block in range(1, 5) for scenario in ("base", "stress")}
    rows = {(r["block"], r["scenario"], r["arm"]): r for r in report["runs"]}
    if len(report["runs"]) != 8 or set(rows) != expected:
        raise ValueError("native comparison matrix incomplete")
    old_rows = {(r["block"], r["scenario"], r["arm"]): r for r in old["runs"]}
    comparison = []
    verified_barriers = 0
    for (block, scenario, arm), row in sorted(rows.items()):
        name = f"block-{block:02d}-{scenario}-{arm}"
        if checked_json(OUT / f"{name}.json") != row:
            raise ValueError("per-arm receipt differs from aggregate")
        encoded = gzip.decompress((OUT / f"{name}-barriers.json.gz").read_bytes())
        if sha256(encoded).hexdigest() != row["trace_sha256"]:
            raise ValueError("native barrier trace checksum mismatch")
        evidence = json.loads(encoded)
        if (len(evidence) != 6477 or row["m22"]["barriers"] != len(evidence)
                or row["m22"]["authority"] != "retained_m22_session_snapshots"
                or row["metrics"]["liquidations"] != 0
                or row["metrics"]["summary"]["economic_status"] != "SCENARIO_ONLY"
                or any(item["scenario_only"] is not True for item in evidence)):
            raise ValueError("native coverage, risk, or authority mismatch")
        verified_barriers += len(evidence)
        prior = old_rows[(block, scenario, "p0_recovery_v3")]
        comparison.append({
            "block": block, "scenario": scenario,
            "old_v3_net_return": prior["metrics"]["net_return"],
            "either_exit_net_return": row["metrics"]["net_return"],
            "net_delta_vs_old_v3": row["metrics"]["net_return"] - prior["metrics"]["net_return"],
            "filled_orders": row["metrics"]["filled_orders"],
            "scenario_cycle_audit": scenario_cycles(evidence),
        })
    native.harness.write_verified(OUT / "mechanism_verification.json", {
        "qualification": "NO_ELIGIBLE_ALPHA_TEACHER",
        "economic_status": "SCENARIO_ONLY", "formal_student_promotion": False,
        "verified_runs": 8, "verified_native_barriers": verified_barriers,
        "comparison": comparison,
        "limitations": ["consumed Train-fit data, not unseen screen",
                        "scenario cycles are not authorized education cases",
                        "account-specific historical economics unavailable"],
    })
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
