"""Compare trend directions over one-day registered Train episodes."""

from __future__ import annotations

import json

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.teachers.trend import TrendTeacher

import run_route_a_train_student_screen_20261020 as runner


OUT = runner.ROOT / ".artifacts" / "route-a-long-horizon-trend-probe-20261020-run21"
WINDOWS = (
    ("february01", "2024-02-01T00:00:00+00:00", "2024-02-02T00:00:00+00:00"),
    ("march01", "2024-03-01T00:00:00+00:00", "2024-03-02T00:00:00+00:00"),
)
EPISODE_STEPS = 1400


def main() -> int:
    runner.configure_torch_threads(runner.torch)
    if OUT.exists():
        raise FileExistsError(f"immutable long-horizon probe output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    runner.WINDOWS = WINDOWS
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner_token = ACTIVE_OWNER.set("route-a-long-trend-probe-20261020")
    try:
        _fit_config, fit_factory = runner.make_config(
            label="probe-fit-jan02-jan09-base",
            marker_sha=marker_sha,
            scope_start=scope_start,
            scope_end=scope_end,
            start_ns=runner.ns(runner.FIT[0]),
            end_ns=runner.ns(runner.FIT[1]),
            mode="base",
            fit_scaler=True,
        )
        factories = {}
        for window_id, start_text, end_text in WINDOWS:
            for scenario in ("base", "stress"):
                _config, factory = runner.make_config(
                    label=f"long-trend-{window_id}-{scenario}",
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=runner.ns(start_text),
                    end_ns=runner.ns(end_text),
                    mode=scenario,
                    fit_scaler=False,
                    scaler=fit_factory.scaler,
                )
                factories[(window_id, scenario)] = factory

        results = []
        for direction in (1, -1):
            teacher = TrendTeacher(
                size=0.8,
                threshold=0.004,
                hold_on_neutral=True,
                direction=direction,
                expert_id=f"long_horizon_trend_{direction}_v1",
            )
            for (window_id, scenario), factory in factories.items():
                for seed in (11, 29):
                    run_id = f"long-trend-{window_id}-{scenario}-d{direction}-s{seed}"
                    print(f"running {run_id}", flush=True)
                    metrics = runner.run_episode(
                        factory,
                        teacher,
                        marker_sha,
                        run_id,
                        seed,
                        EPISODE_STEPS,
                    )
                    results.append(
                        {
                            "window_id": window_id,
                            "scenario": scenario,
                            "seed": seed,
                            "direction": direction,
                            "threshold": teacher.threshold,
                            "size": teacher.size,
                            "hold_on_neutral": teacher.hold_on_neutral,
                            "teacher_recipe_sha256": runner.teacher_config_hash(teacher),
                            "net_return": metrics["net_return"],
                            "max_drawdown": metrics["max_drawdown"],
                            "filled_orders": metrics["filled_orders"],
                            "liquidations": metrics["liquidations"],
                            "native_barriers": metrics["barriers"],
                        }
                    )
        report = {
            "schema": "rles.route_a.long_horizon_trend_probe.v1",
            "purpose": "Train-inner horizon development only; not promotion evidence",
            "source_marker_sha256": marker_sha,
            "windows": [
                {"id": w, "start_ns": runner.ns(a), "end_ns": runner.ns(b)} for w, a, b in WINDOWS
            ],
            "scenarios": ["base", "stress"],
            "seeds": [11, 29],
            "episode_steps": EPISODE_STEPS,
            "teacher_grid": {"direction": [1, -1], "threshold": 0.004, "size": 0.8},
            "results": results,
            "economic_status": "SCENARIO_ONLY",
        }
        runner.write_verified(OUT / "long-horizon-probe.json", report)
        print(json.dumps({"output": str(OUT), "run_count": len(results)}, indent=2))
        return 0
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
