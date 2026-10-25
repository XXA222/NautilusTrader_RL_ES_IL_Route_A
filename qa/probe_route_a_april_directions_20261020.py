"""Compare frozen trend directions on Train-inner April development windows."""

from __future__ import annotations

import json

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.teachers.trend import TrendTeacher

import run_route_a_train_student_screen_20261020 as runner


OUT = runner.ROOT / ".artifacts" / "route-a-april-direction-probe-20261020-run14"


def main() -> int:
    runner.configure_torch_threads(runner.torch)
    if OUT.exists():
        raise FileExistsError(f"immutable direction probe output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner_token = ACTIVE_OWNER.set("route-a-april-direction-probe-20261020")
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
            for scenario in ("base", "stress"):
                _config, factory = runner.make_config(
                    label=f"direction-{window_id}-{scenario}",
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
                expert_id=f"trend_direction_{direction}_neutral_hold_v1",
            )
            for (window_id, scenario), factory in factories.items():
                metrics = runner.run_episode(
                    factory,
                    teacher,
                    marker_sha,
                    f"direction-{window_id}-{scenario}-d{direction}",
                    11,
                    600,
                )
                results.append(
                    {
                        "window_id": window_id,
                        "scenario": scenario,
                        "direction": direction,
                        "net_return": metrics["net_return"],
                        "max_drawdown": metrics["max_drawdown"],
                        "filled_orders": metrics["filled_orders"],
                        "liquidations": metrics["liquidations"],
                    }
                )
        report = {
            "schema": "rles.route_a.april_direction_probe.v1",
            "purpose": "Train-inner rule development only; not independent promotion evidence",
            "source_marker_sha256": marker_sha,
            "windows": [
                {"id": w, "start_ns": runner.ns(a), "end_ns": runner.ns(b)}
                for w, a, b in runner.WINDOWS
            ],
            "scenarios": ["base", "stress"],
            "seed": 11,
            "threshold": 0.004,
            "size": 0.8,
            "hold_on_neutral": True,
            "results": results,
            "economic_status": "SCENARIO_ONLY",
        }
        runner.write_verified(OUT / "direction-probe.json", report)
        print(json.dumps({"output": str(OUT), "run_count": len(results)}, indent=2))
        return 0
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
