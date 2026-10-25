"""Predeclared fixed-side native controls on already-consumed Train-fit blocks.

This is beta attribution, not parameter selection, an unseen screen, or a G1/G2
qualification. Both sides have the v3 teacher's 0.3 entry size, selector risk
overlay, native host, clock policy, seed, windows and Base/Stress assumptions.
Exposure time is deliberately different, so the controls are not risk matched.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
from pathlib import Path

from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from rles_route_a.teachers.protocol import advice, classify_target
from rles_route_a.teachers.recovery import RecoveryTeacher
from rles_route_a.teachers.selector import FrozenTeacherSelector
from rles_route_a.teachers.wait import WaitTeacher

import run_route_a_causal_teacher_fit_native_20261025 as native

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".artifacts/route-a-causal-teacher-fit-native-5m-20261025-archive-clock"
OUT = ROOT / ".artifacts/route-a-beta-attribution-fit-native-20261025"


@dataclass(frozen=True, slots=True)
class FixedSideTeacher:
    side: int
    size: float = 0.3
    expert_id: str = "fixed_side_beta_control_v1"

    def __post_init__(self) -> None:
        if self.side not in (-1, 1) or not 0 < self.size <= 1:
            raise ValueError("fixed-side control requires a valid side and size")

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        current = observation.account_value("net_exposure")
        target = current if current * self.side > 1e-8 else self.side * self.size
        return advice(
            observation, self.expert_id, classify_target(current, target), target,
            "FIXED_SIDE_BETA_CONTROL",
        )


def checked_json(path: Path) -> dict:
    encoded = path.read_bytes()
    if path.with_suffix(path.suffix + ".sha256").read_text().strip() != sha256(encoded).hexdigest():
        raise ValueError(f"checksum mismatch: {path}")
    return json.loads(encoded)


def controls() -> dict:
    return {
        name: FrozenTeacherSelector(
            primary=FixedSideTeacher(side=side, expert_id=f"{name}_v1"),
            recovery=RecoveryTeacher(), wait=WaitTeacher(),
        )
        for name, side in (("fixed_long", 1), ("fixed_short", -1))
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"immutable beta experiment already exists: {OUT}")
    prior_verification = checked_json(SOURCE / "verification.json")
    prior_report = checked_json(SOURCE / "report.json")
    if prior_verification["verified_runs"] != 24 or prior_verification["qualification"] != "NO_ELIGIBLE_ALPHA_TEACHER":
        raise ValueError("matched v3 native source has not been verified")
    prior_recipe = checked_json(SOURCE / "recipe.json")
    if sha256((SOURCE / "recipe.json").read_bytes()).hexdigest() != prior_verification["recipe_sha256"]:
        raise ValueError("prior frozen recipe identity mismatch")
    if prior_recipe["entry_cadence_minutes"] != 5 or prior_recipe["economic_status"] != "SCENARIO_ONLY":
        raise ValueError("incompatible prior native experiment")
    hypothesis = {
        "hypothesis": "Block-02 v3 profit may be explainable by signed ETH exposure rather than event timing",
        "frozen_prior_recipe_sha256": prior_verification["recipe_sha256"],
        "frozen_prior_report_sha256": sha256((SOURCE / "report.json").read_bytes()).hexdigest(),
        "comparison_policy": "compare same block/scenario net account returns to both constant sides; never choose winning side for deployment",
        "exposure_caveat": "fixed-side controls enter earlier and can hold longer; attribution is descriptive, not exposure-risk matched",
        "qualification_policy": "fit-only diagnostic; no G1/G2 or student promotion",
    }
    native.run(
        OUT, entry_cadence_minutes=5, archive_clock=True,
        arms_override=controls(), extra_recipe=hypothesis,
    )
    report = checked_json(OUT / "report.json")
    recipe = checked_json(OUT / "recipe.json")
    if any(recipe[key] != value for key, value in hypothesis.items()):
        raise ValueError("frozen beta hypothesis changed during execution")
    expected = {(block, scenario, arm) for block in range(1, 5)
                for scenario in ("base", "stress") for arm in controls()}
    rows = {(r["block"], r["scenario"], r["arm"]): r for r in report["runs"]}
    if set(rows) != expected or len(report["runs"]) != 16:
        raise ValueError("fixed-side comparison matrix incomplete")
    verified_barriers = 0
    for (block, scenario, arm), row in rows.items():
        name = f"block-{block:02d}-{scenario}-{arm}"
        if checked_json(OUT / f"{name}.json") != row:
            raise ValueError("per-run native receipt differs from aggregate")
        trace = gzip.decompress((OUT / f"{name}-barriers.json.gz").read_bytes())
        if sha256(trace).hexdigest() != row["trace_sha256"]:
            raise ValueError("native trace checksum mismatch")
        evidence = json.loads(trace)
        if (len(evidence) != 6477 or row["m22"]["barriers"] != len(evidence)
                or row["m22"]["authority"] != "retained_m22_session_snapshots"
                or row["metrics"]["liquidations"] != 0
                or row["metrics"]["summary"]["economic_status"] != "SCENARIO_ONLY"
                or any(item["scenario_only"] is not True for item in evidence)):
            raise ValueError("native evidence, risk, or economic status mismatch")
        verified_barriers += len(evidence)
    old = {(r["block"], r["scenario"], r["arm"]): r for r in prior_report["runs"]}
    comparisons = []
    for block in range(1, 5):
        for scenario in ("base", "stress"):
            v3 = old[(block, scenario, "p0_recovery_v3")]["metrics"]["net_return"]
            long = rows[(block, scenario, "fixed_long")]["metrics"]["net_return"]
            short = rows[(block, scenario, "fixed_short")]["metrics"]["net_return"]
            comparisons.append({
                "block": block, "scenario": scenario, "v3": v3,
                "fixed_long": long, "fixed_short": short,
                "v3_minus_fixed_long": v3 - long,
                "v3_minus_fixed_short": v3 - short,
                "v3_exceeds_both_fixed_sides": v3 > max(long, short),
            })
    native.harness.write_verified(OUT / "attribution.json", {
        "qualification": "NO_ELIGIBLE_ALPHA_TEACHER",
        "economic_status": "SCENARIO_ONLY", "formal_student_promotion": False,
        "verified_runs": len(rows), "verified_native_barriers": verified_barriers,
        "comparison": comparisons,
        "limitations": ["same already-consumed Train-fit blocks; not unseen evidence",
                        "fixed-side controls are not exposure-time or risk matched",
                        "ex-post best side is not a tradable precommitted selector",
                        "historical account-specific economics remain unverified"],
    })
    print(json.dumps(comparisons, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
