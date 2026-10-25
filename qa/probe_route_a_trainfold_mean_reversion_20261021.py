"""Screen the frozen range-reversion rule grid on later Train-only windows."""

from __future__ import annotations

import json
from hashlib import sha256

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.evaluation import NativePair, screen_teacher
from rles_route_a.teachers.mean_reversion import MeanReversionTeacher
from rles_route_a.teachers.wait import WaitTeacher

import run_route_a_trainfold_screen_20261021 as trainfold
import run_route_a_train_student_screen_20261020 as runner


OUT = runner.ROOT / ".artifacts" / "route-a-trainfold-mean-reversion-20261021-run01"
WINDOWS = (
    ("jun05_h00", "2024-06-05T00:00:00+00:00", "2024-06-05T12:00:00+00:00"),
    ("jun13_h00", "2024-06-13T00:00:00+00:00", "2024-06-13T12:00:00+00:00"),
)
RULES = ((1.0, 0.001), (1.0, 0.003), (2.0, 0.001), (2.0, 0.003))
SEEDS = (11, 29)
SCENARIOS = ("base", "stress")
FIT_START = "2024-04-16T00:00:00+00:00"
FIT_END = "2024-05-15T00:00:00+00:00"
EPISODE_STEPS = 700


def main() -> int:
    runner.configure_torch_threads(runner.torch)
    trainfold.configure()
    if OUT.exists():
        raise FileExistsError(f"immutable mean-reversion probe output already exists: {OUT}")
    score_start = runner.ns("2024-04-16T00:00:00+00:00")
    score_end = runner.ns("2024-06-15T00:00:00+00:00")
    for window_id, start, end in WINDOWS:
        if runner.ns(start) < score_start or runner.ns(end) > score_end:
            raise PermissionError(f"probe window outside the frozen Train interval: {window_id}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    _, fit_factory = runner.make_config(
        label="range-reversion-scaler-apr16-may15",
        marker_sha=marker_sha,
        scope_start=scope_start,
        scope_end=scope_end,
        start_ns=runner.ns(FIT_START),
        end_ns=runner.ns(FIT_END),
        mode="base",
        fit_scaler=True,
    )
    scaler = fit_factory.scaler
    baseline = WaitTeacher(expert_id="wait_flat_baseline_v1")
    baseline_hash = runner.teacher_config_hash(baseline)
    owner = ACTIVE_OWNER.set("route-a-mean-reversion-probe-20261021")
    try:
        factories = {}
        for window_id, start, end in WINDOWS:
            for scenario in SCENARIOS:
                _, factory = runner.make_config(
                    label=f"mean-reversion-{window_id}-{scenario}",
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=runner.ns(start),
                    end_ns=runner.ns(end),
                    mode=scenario,
                    fit_scaler=False,
                    scaler=scaler,
                )
                factories[(window_id, scenario)] = factory

        results = []
        screens = {}
        for z_entry, max_abs_trend in RULES:
            teacher = MeanReversionTeacher(
                z_entry=z_entry,
                size=0.2,
                range_max_abs_trend=max_abs_trend,
                expert_id=f"mean_reversion_z{z_entry:g}_trend{max_abs_trend:g}_v1",
            )
            teacher_hash = runner.teacher_config_hash(teacher)
            pairs = []
            for (window_id, scenario), factory in factories.items():
                for seed in SEEDS:
                    label = f"{window_id}-{scenario}-z{z_entry:g}-t{max_abs_trend:g}-seed{seed}"
                    teacher_run = runner.run_episode(
                        factory, teacher, marker_sha, label + "-teacher", seed, EPISODE_STEPS
                    )
                    baseline_run = runner.run_episode(
                        factory, baseline, marker_sha, label + "-baseline", seed, EPISODE_STEPS
                    )
                    matched = (teacher_run["barriers"] == baseline_run["barriers"] == EPISODE_STEPS)
                    evidence_sha = {
                        role: runner.write_verified(
                            OUT / "native-evidence" / f"{label}-{role}.json", result["evidence"]
                        )
                        for role, result in (("teacher", teacher_run), ("baseline", baseline_run))
                    }
                    pairs.append(NativePair(
                        window_id=window_id,
                        scenario=scenario,
                        seed=seed,
                        teacher_recipe=teacher_hash,
                        baseline_recipe=baseline_hash,
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
                    ))
                    results.append({
                        "z_entry": z_entry,
                        "range_max_abs_trend": max_abs_trend,
                        "window_id": window_id,
                        "scenario": scenario,
                        "seed": seed,
                        "teacher_net_return": teacher_run["net_return"],
                        "baseline_net_return": baseline_run["net_return"],
                        "teacher_max_drawdown": teacher_run["max_drawdown"],
                        "teacher_filled_orders": teacher_run["filled_orders"],
                        "teacher_liquidations": teacher_run["liquidations"],
                        "matched_native_barriers": matched,
                        "evidence_sha256": evidence_sha,
                    })
                    print(f"completed {label}", flush=True)
            screen = screen_teacher(pairs)
            key = f"z{z_entry:g}_trend{max_abs_trend:g}"
            screens[key] = {
                "teacher_recipe_sha256": teacher_hash,
                "status": screen.status,
                "economic_status": screen.economic_status,
                "pair_count": screen.pair_count,
                "window_count": screen.window_count,
                "mean_excess_return": screen.mean_excess_return,
                "worst_excess_return": screen.worst_excess_return,
                "reasons": list(screen.reasons),
                "receipt_sha256": screen.receipt_sha256,
            }

        report = {
            "schema": "rles.route_a.trainfold_mean_reversion_probe.v1",
            "purpose": "Fixed Train-inner rule grid; diagnostic only, not independent promotion evidence",
            "source_marker_sha256": marker_sha,
            "wfo_report_sha256": sha256(trainfold.WFO_REPORT.read_bytes()).hexdigest(),
            "fit_scaler_window_ns": [runner.ns(FIT_START), runner.ns(FIT_END)],
            "windows": [
                {"id": window_id, "start_ns": runner.ns(start), "end_ns": runner.ns(end)}
                for window_id, start, end in WINDOWS
            ],
            "scenarios": list(SCENARIOS),
            "seeds": list(SEEDS),
            "episode_steps": EPISODE_STEPS,
            "parameter_grid": [
                {"z_entry": z, "size": 0.2, "range_max_abs_trend": trend}
                for z, trend in RULES
            ],
            "screens": screens,
            "results": results,
            "economic_status": "SCENARIO_ONLY",
        }
        runner.write_verified(OUT / "mean-reversion-report.json", report)
        print(json.dumps({"output": str(OUT), "run_count": len(results)}, indent=2))
        return 0
    finally:
        ACTIVE_OWNER.reset(owner)


if __name__ == "__main__":
    raise SystemExit(main())
