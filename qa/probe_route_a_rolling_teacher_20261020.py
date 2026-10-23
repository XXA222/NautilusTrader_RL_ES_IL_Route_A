"""Compare fixed trend rules over sampled non-overlapping Train days."""

from __future__ import annotations

import json
import gc

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.teachers.trend import TrendTeacher

import run_route_a_train_student_screen_20261020 as runner


OUT = runner.ROOT / ".artifacts" / "route-a-rolling-teacher-probe-20261020-run25"
WINDOWS = (
    ("feb01", "2024-02-01T00:00:00+00:00", "2024-02-02T00:00:00+00:00"),
    ("feb08", "2024-02-08T00:00:00+00:00", "2024-02-09T00:00:00+00:00"),
    ("feb15", "2024-02-15T00:00:00+00:00", "2024-02-16T00:00:00+00:00"),
    ("feb22", "2024-02-22T00:00:00+00:00", "2024-02-23T00:00:00+00:00"),
    ("mar01", "2024-03-01T00:00:00+00:00", "2024-03-02T00:00:00+00:00"),
    ("mar08", "2024-03-08T00:00:00+00:00", "2024-03-09T00:00:00+00:00"),
    ("mar15", "2024-03-15T00:00:00+00:00", "2024-03-16T00:00:00+00:00"),
    ("mar22", "2024-03-22T00:00:00+00:00", "2024-03-23T00:00:00+00:00"),
    ("apr01", "2024-04-01T00:00:00+00:00", "2024-04-02T00:00:00+00:00"),
    ("apr08", "2024-04-08T00:00:00+00:00", "2024-04-09T00:00:00+00:00"),
    ("apr13", "2024-04-13T00:00:00+00:00", "2024-04-14T00:00:00+00:00"),
    ("apr14", "2024-04-14T00:00:00+00:00", "2024-04-15T00:00:00+00:00"),
)
EPISODE_STEPS = 1_400
SEEDS = (11, 29)


def main() -> int:
    runner.configure_torch_threads(runner.torch)
    if OUT.exists():
        raise FileExistsError(f"immutable rolling teacher output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner_token = ACTIVE_OWNER.set("route-a-rolling-teacher-probe-20261020")
    try:
        _config, fit_factory = runner.make_config(
            label="rolling-teacher-fit-feb01-feb02",
            marker_sha=marker_sha,
            scope_start=scope_start,
            scope_end=scope_end,
            start_ns=runner.ns(WINDOWS[0][1]),
            end_ns=runner.ns(WINDOWS[0][2]),
            mode="base",
            fit_scaler=True,
        )
        scaler = fit_factory.scaler
        del fit_factory
        gc.collect()

        results = []
        teachers = {
            direction: TrendTeacher(
                size=0.8,
                threshold=0.004,
                hold_on_neutral=True,
                direction=direction,
                expert_id=f"rolling_fixed_direction_{direction}_v1",
            )
            for direction in (1, -1)
        }
        for window_id, start, end in WINDOWS:
            for scenario in ("base", "stress"):
                _, factory = runner.make_config(
                    label=f"rolling-{window_id}-{scenario}",
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=runner.ns(start),
                    end_ns=runner.ns(end),
                    mode=scenario,
                    fit_scaler=False,
                    scaler=scaler,
                )
                for direction, teacher in teachers.items():
                    for seed in SEEDS:
                        label = f"rolling-{window_id}-{scenario}-d{direction}-s{seed}"
                        print(f"running {label}", flush=True)
                        metrics = runner.run_episode(
                            factory, teacher, marker_sha, label, seed, EPISODE_STEPS
                        )
                        evidence_sha = runner.write_verified(
                            OUT / "native-evidence" / f"{label}.json", metrics["evidence"]
                        )
                        results.append(
                            {
                                "window_id": window_id,
                                "scenario": scenario,
                                "seed": seed,
                                "direction": direction,
                                "teacher_recipe_sha256": runner.teacher_config_hash(teacher),
                                "net_return": metrics["net_return"],
                                "max_drawdown": metrics["max_drawdown"],
                                "filled_orders": metrics["filled_orders"],
                                "liquidations": metrics["liquidations"],
                                "native_barriers": metrics["barriers"],
                                "native_evidence_sha256": evidence_sha,
                            }
                        )
                del factory
                gc.collect()
        report = {
            "schema": "rles.route_a.rolling_teacher_probe.v1",
            "purpose": (
                "Fixed-rule Train-only rolling diagnosis; these dates are now development data "
                "and are not independent promotion evidence."
            ),
            "source_marker_sha256": marker_sha,
            "windows": [
                {"id": w, "start_ns": runner.ns(start), "end_ns": runner.ns(end)}
                for w, start, end in WINDOWS
            ],
            "scenarios": ["base", "stress"],
            "seeds": list(SEEDS),
            "episode_steps": EPISODE_STEPS,
            "teacher_grid": {"direction": [1, -1], "threshold": 0.004, "size": 0.8},
            "results": results,
            "economic_status": "SCENARIO_ONLY",
        }
        output = OUT / "rolling-teacher-probe.json"
        runner.write_verified(output, report)
        print(json.dumps({"output": str(output), "run_count": len(results)}, indent=2))
        return 0
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
