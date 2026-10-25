"""Explore a weak-trend fade / strong-trend follow rule on registered Train."""

from __future__ import annotations

import json

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.teachers.trend import TrendTeacher

import run_route_a_train_student_screen_20261020 as runner


OUT = runner.ROOT / ".artifacts" / "route-a-regime-teacher-probe-20261020-run15"
WINDOWS = (
    ("february", "2024-02-01T00:00:00+00:00", "2024-02-01T12:00:00+00:00"),
    ("march", "2024-03-01T00:00:00+00:00", "2024-03-01T12:00:00+00:00"),
    ("april01", "2024-04-01T00:00:00+00:00", "2024-04-01T12:00:00+00:00"),
    ("april08", "2024-04-08T00:00:00+00:00", "2024-04-08T12:00:00+00:00"),
)
REGIME_THRESHOLDS = (0.004, 0.008, 0.012, 0.02)


def main() -> int:
    runner.configure_torch_threads(runner.torch)
    if OUT.exists():
        raise FileExistsError(f"immutable regime probe output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    runner.WINDOWS = WINDOWS
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner_token = ACTIVE_OWNER.set("route-a-regime-probe-20261020")
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
                    label=f"regime-{window_id}-{scenario}",
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
        for regime_threshold in REGIME_THRESHOLDS:
            teacher = TrendTeacher(
                size=0.8,
                threshold=0.004,
                hold_on_neutral=True,
                regime_strength_threshold=regime_threshold,
                expert_id=f"adaptive_trend_regime_{regime_threshold:g}_v1",
            )
            for (window_id, scenario), factory in factories.items():
                metrics = runner.run_episode(
                    factory,
                    teacher,
                    marker_sha,
                    f"regime-{window_id}-{scenario}-r{regime_threshold:g}",
                    11,
                    600,
                )
                results.append(
                    {
                        "window_id": window_id,
                        "scenario": scenario,
                        "seed": 11,
                        "regime_strength_threshold": regime_threshold,
                        "net_return": metrics["net_return"],
                        "max_drawdown": metrics["max_drawdown"],
                        "filled_orders": metrics["filled_orders"],
                        "liquidations": metrics["liquidations"],
                    }
                )
        report = {
            "schema": "rles.route_a.regime_teacher_probe.v1",
            "purpose": "Train-inner rule development only; not promotion evidence",
            "source_marker_sha256": marker_sha,
            "windows": [
                {"id": w, "start_ns": runner.ns(a), "end_ns": runner.ns(b)} for w, a, b in WINDOWS
            ],
            "scenarios": ["base", "stress"],
            "seed": 11,
            "teacher_rule": {
                "weak_aligned_trend_direction": "countertrend",
                "strong_aligned_trend_direction": "trend-following",
                "trend_threshold": 0.004,
                "regime_strength_thresholds": list(REGIME_THRESHOLDS),
                "strongness_metric": "min(abs(trend_1h), abs(trend_1d))",
                "hold_on_neutral": True,
            },
            "results": results,
            "economic_status": "SCENARIO_ONLY",
        }
        runner.write_verified(OUT / "regime-probe.json", report)
        print(json.dumps({"output": str(OUT), "run_count": len(results)}, indent=2))
        return 0
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
