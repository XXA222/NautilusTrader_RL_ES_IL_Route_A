"""Probe the existing causal z-score reversal teacher on registered Train."""

from __future__ import annotations

import json

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.teachers.mean_reversion import MeanReversionTeacher

import run_route_a_train_student_screen_20261020 as runner


OUT = runner.ROOT / ".artifacts" / "route-a-mean-reversion-probe-20261020-run19"
WINDOWS = (
    ("february", "2024-02-01T00:00:00+00:00", "2024-02-01T12:00:00+00:00"),
    ("march", "2024-03-01T00:00:00+00:00", "2024-03-01T12:00:00+00:00"),
    ("april01", "2024-04-01T00:00:00+00:00", "2024-04-01T12:00:00+00:00"),
    ("april08", "2024-04-08T00:00:00+00:00", "2024-04-08T12:00:00+00:00"),
)
RULES = ((1.0, 0.001), (1.0, 0.003), (2.0, 0.001), (2.0, 0.003))


def main() -> int:
    runner.configure_torch_threads(runner.torch)
    if OUT.exists():
        raise FileExistsError(f"immutable mean-reversion probe output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    runner.WINDOWS = WINDOWS
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner_token = ACTIVE_OWNER.set("route-a-mean-reversion-probe-20261020")
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
                    label=f"mean-reversion-{window_id}-{scenario}",
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
        for z_entry, range_max_abs_trend in RULES:
            teacher = MeanReversionTeacher(
                z_entry=z_entry,
                size=0.2,
                range_max_abs_trend=range_max_abs_trend,
                expert_id=f"mean_reversion_z{z_entry:g}_trend{range_max_abs_trend:g}_v1",
            )
            for (window_id, scenario), factory in factories.items():
                run_id = (
                    f"mean-reversion-{window_id}-{scenario}-z{z_entry:g}-t{range_max_abs_trend:g}"
                )
                print(f"running {run_id}", flush=True)
                try:
                    metrics = runner.run_episode(
                        factory,
                        teacher,
                        marker_sha,
                        run_id,
                        11,
                        600,
                    )
                except Exception as exc:
                    results.append(
                        {
                            "window_id": window_id,
                            "scenario": scenario,
                            "seed": 11,
                            "z_entry": z_entry,
                            "size": teacher.size,
                            "range_max_abs_trend": range_max_abs_trend,
                            "teacher_recipe_sha256": runner.teacher_config_hash(teacher),
                            "gate_status": "REJECT",
                            "failure_type": type(exc).__name__,
                            "failure": str(exc),
                        }
                    )
                    print(f"rejected {run_id}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                results.append(
                    {
                        "window_id": window_id,
                        "scenario": scenario,
                        "seed": 11,
                        "z_entry": z_entry,
                        "size": teacher.size,
                        "range_max_abs_trend": range_max_abs_trend,
                        "teacher_recipe_sha256": runner.teacher_config_hash(teacher),
                        "net_return": metrics["net_return"],
                        "max_drawdown": metrics["max_drawdown"],
                        "filled_orders": metrics["filled_orders"],
                        "liquidations": metrics["liquidations"],
                        "gate_status": "EVIDENCE_PASS",
                    }
                )
        report = {
            "schema": "rles.route_a.mean_reversion_probe.v1",
            "purpose": "Train-inner teacher development only; not independent promotion evidence",
            "source_marker_sha256": marker_sha,
            "windows": [
                {"id": w, "start_ns": runner.ns(a), "end_ns": runner.ns(b)} for w, a, b in WINDOWS
            ],
            "scenarios": ["base", "stress"],
            "seed": 11,
            "teacher_size": 0.2,
            "parameter_grid": [{"z_entry": z, "range_max_abs_trend": t} for z, t in RULES],
            "results": results,
            "economic_status": "SCENARIO_ONLY",
        }
        runner.write_verified(OUT / "mean-reversion-probe.json", report)
        print(json.dumps({"output": str(OUT), "run_count": len(results)}, indent=2))
        return 0
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
