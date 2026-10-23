"""Train-inner-only exploratory teacher rule probe; never a promotion screen."""

from __future__ import annotations

import json

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.teachers.trend import TrendTeacher

import run_route_a_train_student_screen_20261020 as runner


OUT = runner.ROOT / ".artifacts" / "route-a-teacher-rule-probe-20261020-run11"
THRESHOLDS = (0.001, 0.004, 0.008, 0.016)
DIRECTIONS = (1, -1)


def main() -> int:
    runner.configure_torch_threads(runner.torch)
    if OUT.exists():
        raise FileExistsError(f"immutable teacher probe output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner_token = ACTIVE_OWNER.set("route-a-teacher-probe-20261020")
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
        for window_id, start_text, end_text in runner.WINDOWS:
            start_ns, end_ns = runner.ns(start_text), runner.ns(end_text)
            for scenario in ("base", "stress"):
                label = f"probe-{window_id}-{scenario}"
                _config, factory = runner.make_config(
                    label=label,
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    mode=scenario,
                    fit_scaler=False,
                    scaler=fit_factory.scaler,
                )
                factories[(window_id, scenario)] = factory

        results = []
        for direction in DIRECTIONS:
            for threshold in THRESHOLDS:
                teacher = TrendTeacher(
                    size=0.8,
                    threshold=threshold,
                    hold_on_neutral=True,
                    direction=direction,
                    expert_id=f"trend_direction_{direction}_neutral_hold_v1",
                )
                for (window_id, scenario), factory in factories.items():
                    metrics = runner.run_episode(
                        factory,
                        teacher,
                        marker_sha,
                        f"probe-{window_id}-{scenario}-d{direction}-t{threshold:g}",
                        11,
                        600,
                    )
                    results.append(
                        {
                            "window_id": window_id,
                            "scenario": scenario,
                            "seed": 11,
                            "direction": direction,
                            "threshold": threshold,
                            "size": teacher.size,
                            "teacher_recipe_sha256": runner.teacher_config_hash(teacher),
                            "net_return": metrics["net_return"],
                            "max_drawdown": metrics["max_drawdown"],
                            "filled_orders": metrics["filled_orders"],
                            "liquidations": metrics["liquidations"],
                            "native_barriers": metrics["barriers"],
                        }
                    )
        report = {
            "schema": "rles.route_a.teacher_rule_probe.v1",
            "purpose": "exploratory Train-inner tuning only; no promotion claim",
            "source_marker_sha256": marker_sha,
            "scope_ns": [scope_start, scope_end],
            "windows": [
                {"id": w, "start_ns": runner.ns(a), "end_ns": runner.ns(b)}
                for w, a, b in runner.WINDOWS
            ],
            "scenarios": ["base", "stress"],
            "seed": 11,
            "teacher_rule_grid": {
                "direction": list(DIRECTIONS),
                "threshold": list(THRESHOLDS),
                "size": 0.8,
                "hold_on_neutral": True,
            },
            "results": results,
            "economic_status": "SCENARIO_ONLY",
        }
        runner.write_verified(OUT / "probe-report.json", report)
        print(json.dumps({"output": str(OUT), "run_count": len(results)}, indent=2))
        return 0
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
