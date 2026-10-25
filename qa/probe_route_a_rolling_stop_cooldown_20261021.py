"""Screen stop-loss/cooldown teacher over the rolling registered Train matrix."""

from __future__ import annotations

from dataclasses import asdict
import gc
import json

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.evaluation import NativePair, screen_teacher
from rles_route_a.teachers.wait import WaitTeacher
from rles_route_a.teachers.trend import TrendTeacher

from probe_route_a_stop_cooldown_20261021 import StopCooldownTeacher
import run_route_a_train_student_screen_20261020 as runner


OUT = runner.ROOT / ".artifacts" / "route-a-rolling-stop-cooldown-probe-20261021-run01"
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
SEEDS = (11, 29)
SCENARIOS = ("base", "stress")
EPISODE_STEPS = 1_400
STOP_LOSS_FRACTION = 0.001
COOLDOWN_NS = 5 * 60_000_000_000


def make_teacher() -> StopCooldownTeacher:
    return StopCooldownTeacher(
        primary=TrendTeacher(
            size=0.8,
            threshold=0.004,
            hold_on_neutral=True,
            direction=-1,
            expert_id="countertrend_hold_neutral_v1",
        ),
        stop_loss_fraction=STOP_LOSS_FRACTION,
        cooldown_ns=COOLDOWN_NS,
        expert_id="countertrend_stop_cooldown_5m_v1",
    )


def main() -> int:
    runner.configure_torch_threads(runner.torch)
    if OUT.exists():
        raise FileExistsError(f"immutable rolling stop/cooldown output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    runner.WINDOWS = WINDOWS
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner_token = ACTIVE_OWNER.set("route-a-rolling-stop-cooldown-probe-20261021")
    try:
        _, scaler_factory = runner.make_config(
            label="rolling-stop-cooldown-scaler-feb01",
            marker_sha=marker_sha,
            scope_start=scope_start,
            scope_end=scope_end,
            start_ns=runner.ns(WINDOWS[0][1]),
            end_ns=runner.ns(WINDOWS[0][2]),
            mode="base",
            fit_scaler=True,
        )
        scaler = scaler_factory.scaler
        del scaler_factory
        gc.collect()
        teacher_recipe = runner.teacher_config_hash(make_teacher())
        baseline_recipe = runner.teacher_config_hash(WaitTeacher(expert_id="wait_flat_baseline_v1"))
        rows = []
        pairs = []
        for window_id, start, end in WINDOWS:
            for scenario in SCENARIOS:
                _, factory = runner.make_config(
                    label=f"rolling-stop-{window_id}-{scenario}",
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=runner.ns(start),
                    end_ns=runner.ns(end),
                    mode=scenario,
                    fit_scaler=False,
                    scaler=scaler,
                )
                for seed in SEEDS:
                    key = f"{window_id}-{scenario}-seed-{seed}"
                    print(f"running {key}", flush=True)
                    teacher_run = runner.run_episode(
                        factory, make_teacher(), marker_sha,
                        f"rolling-stop-{key}", seed, EPISODE_STEPS,
                    )
                    baseline_run = runner.run_episode(
                        factory, WaitTeacher(expert_id="wait_flat_baseline_v1"), marker_sha,
                        f"rolling-wait-{key}", seed, EPISODE_STEPS,
                    )
                    teacher_evidence_sha = runner.write_verified(
                        OUT / "native-evidence" / f"{key}-teacher.json",
                        teacher_run["evidence"],
                    )
                    baseline_evidence_sha = runner.write_verified(
                        OUT / "native-evidence" / f"{key}-baseline.json",
                        baseline_run["evidence"],
                    )
                    matched = (
                        teacher_run["barriers"] == baseline_run["barriers"] == EPISODE_STEPS
                    )
                    pairs.append(
                        NativePair(
                            window_id=window_id,
                            scenario=scenario,
                            seed=seed,
                            teacher_recipe=teacher_recipe,
                            baseline_recipe=baseline_recipe,
                            source_identity=marker_sha,
                            teacher_net_return=teacher_run["net_return"],
                            baseline_net_return=baseline_run["net_return"],
                            teacher_max_drawdown=teacher_run["max_drawdown"],
                            baseline_max_drawdown=baseline_run["max_drawdown"],
                            teacher_filled_orders=teacher_run["filled_orders"],
                            teacher_liquidations=teacher_run["liquidations"],
                            matched_native_barriers=matched,
                            scenario_only=True,
                            nonoverlap_verified=True,
                        )
                    )
                    rows.append(
                        {
                            "window_id": window_id,
                            "scenario": scenario,
                            "seed": seed,
                            "teacher_net_return": teacher_run["net_return"],
                            "baseline_net_return": baseline_run["net_return"],
                            "teacher_max_drawdown": teacher_run["max_drawdown"],
                            "teacher_filled_orders": teacher_run["filled_orders"],
                            "teacher_liquidations": teacher_run["liquidations"],
                            "matched_native_barriers": matched,
                            "teacher_evidence_sha256": teacher_evidence_sha,
                            "baseline_evidence_sha256": baseline_evidence_sha,
                        }
                    )
                del factory
                gc.collect()
        screen = screen_teacher(pairs)
        report = {
            "schema": "rles.route_a.rolling_stop_cooldown_probe.v1",
            "purpose": (
                "Development diagnosis of stop/cooldown teacher across previously inspected "
                "registered Train dates; not independent certification."
            ),
            "source_marker_sha256": marker_sha,
            "windows": [
                {"id": key, "start_ns": runner.ns(start), "end_ns": runner.ns(end)}
                for key, start, end in WINDOWS
            ],
            "scenarios": list(SCENARIOS),
            "seeds": list(SEEDS),
            "episode_steps": EPISODE_STEPS,
            "teacher_rule": {
                "primary": "countertrend_hold_neutral_v1",
                "stop_loss_fraction": STOP_LOSS_FRACTION,
                "cooldown_ns": COOLDOWN_NS,
                "teacher_recipe_sha256": teacher_recipe,
            },
            "baseline_recipe_sha256": baseline_recipe,
            "results": rows,
            "teacher_screen": asdict(screen),
            "economic_status": "SCENARIO_ONLY",
        }
        output = OUT / "rolling-stop-cooldown-report.json"
        runner.write_verified(output, report)
        print(json.dumps({
            "output": str(output),
            "episodes": len(rows) * 2,
            "teacher_status": screen.status,
            "teacher_reasons": screen.reasons,
            "economic_status": screen.economic_status,
        }, indent=2), flush=True)
        return 0
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
