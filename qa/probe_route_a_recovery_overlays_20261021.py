"""Train-only probe of position stop-loss overlays on the frozen alpha teacher."""

from __future__ import annotations

import json

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.teachers.recovery import RecoveryTeacher
from rles_route_a.teachers.selector import FrozenTeacherSelector
from rles_route_a.teachers.trend import TrendTeacher

import run_route_a_train_student_screen_20261020 as runner


OUT = runner.ROOT / ".artifacts" / "route-a-recovery-overlay-probe-20261021-run01"
WINDOWS = (
    ("april13", "2024-04-13T00:00:00+00:00", "2024-04-14T00:00:00+00:00"),
    ("april14", "2024-04-14T00:00:00+00:00", "2024-04-15T00:00:00+00:00"),
)
SCENARIOS = ("base", "stress")
SEEDS = (11, 29)
STOP_LOSSES = (None, 0.001, 0.002, 0.003, 0.005)
EPISODE_STEPS = 1_400


def main() -> int:
    runner.configure_torch_threads(runner.torch)
    if OUT.exists():
        raise FileExistsError(f"immutable recovery probe output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner_token = ACTIVE_OWNER.set("route-a-recovery-overlay-probe-20261021")
    try:
        _, fit_factory = runner.make_config(
            label="recovery-probe-scaler-jan02-jan09",
            marker_sha=marker_sha,
            scope_start=scope_start,
            scope_end=scope_end,
            start_ns=runner.ns(runner.FIT[0]),
            end_ns=runner.ns(runner.FIT[1]),
            mode="base",
            fit_scaler=True,
        )
        rows = []
        for window_id, start, end in WINDOWS:
            for scenario in SCENARIOS:
                _, factory = runner.make_config(
                    label=f"recovery-{window_id}-{scenario}",
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=runner.ns(start),
                    end_ns=runner.ns(end),
                    mode=scenario,
                    fit_scaler=False,
                    scaler=fit_factory.scaler,
                )
                for stop_loss in STOP_LOSSES:
                    primary = TrendTeacher(
                        size=0.8,
                        threshold=0.004,
                        hold_on_neutral=True,
                        direction=-1,
                        expert_id="countertrend_hold_neutral_v1",
                    )
                    teacher = (
                        primary
                        if stop_loss is None
                        else FrozenTeacherSelector(
                            primary=primary,
                            recovery=RecoveryTeacher(
                                stop_loss_fraction=stop_loss,
                                expert_id=f"recovery_stop_{stop_loss:g}_v1",
                            ),
                        )
                    )
                    for seed in SEEDS:
                        label = (
                            f"{window_id}-{scenario}-"
                            f"stop-{stop_loss if stop_loss is not None else 'none'}-seed-{seed}"
                        )
                        result = runner.run_episode(
                            factory, teacher, marker_sha, label, seed, EPISODE_STEPS
                        )
                        evidence_sha = runner.write_verified(
                            OUT / "native-evidence" / f"{label}.json", result["evidence"]
                        )
                        rows.append(
                            {
                                "window_id": window_id,
                                "scenario": scenario,
                                "seed": seed,
                                "stop_loss_fraction": stop_loss,
                                "net_return": result["net_return"],
                                "max_drawdown": result["max_drawdown"],
                                "filled_orders": result["filled_orders"],
                                "liquidations": result["liquidations"],
                                "barriers": result["barriers"],
                                "evidence_sha256": evidence_sha,
                            }
                        )
                del factory
        report = {
            "schema": "rles.route_a.recovery_overlay_probe.v1",
            "purpose": "Train-only scenario-cost development probe; not promotion evidence",
            "source_marker_sha256": marker_sha,
            "windows": [
                {"id": key, "start_ns": runner.ns(start), "end_ns": runner.ns(end)}
                for key, start, end in WINDOWS
            ],
            "scenarios": list(SCENARIOS),
            "seeds": list(SEEDS),
            "teacher_alpha": {
                "expert_id": "countertrend_hold_neutral_v1",
                "size": 0.8,
                "threshold": 0.004,
                "hold_on_neutral": True,
                "direction": -1,
            },
            "stop_loss_grid": list(STOP_LOSSES),
            "rows": rows,
            "economic_status": "SCENARIO_ONLY",
            "promotion_eligible": False,
        }
        runner.write_verified(OUT / "recovery-overlay-report.json", report)
        print(json.dumps({"output": str(OUT), "episode_count": len(rows)}, indent=2))
        return 0
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
