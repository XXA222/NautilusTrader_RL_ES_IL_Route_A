"""Explore causal MTF trend-pullback entries using the host's 1m z-score."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
from math import isfinite

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from rles_route_a.evaluation import NativePair, screen_teacher
from rles_route_a.teachers.protocol import advice, classify_target
from rles_route_a.teachers.wait import WaitTeacher

import run_route_a_train_student_screen_20261020 as runner


OUT = runner.ROOT / ".artifacts" / "route-a-trend-pullback-teacher-20261021-run01"
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
GRID_WINDOWS = {"feb01", "mar01", "apr01", "apr13"}
DIRECTIONS = (1, -1)
ENTRY_Z = (1.5, 2.0)
EXIT_Z = (0.0, 0.5)


@dataclass(frozen=True)
class TrendPullbackTeacher:
    direction: int
    entry_z: float
    exit_z: float
    size: float = 0.4
    trend_threshold: float = 0.004
    max_volatility: float = 0.08
    expert_id: str = "trend_pullback_v1"

    def __post_init__(self) -> None:
        if (self.direction not in {-1, 1}
                or not all(isfinite(value) for value in (
                    self.entry_z, self.exit_z, self.size,
                    self.trend_threshold, self.max_volatility,
                ))
                or self.entry_z <= self.exit_z
                or self.exit_z < 0
                or not 0 < self.size <= 1
                or self.trend_threshold <= 0
                or self.max_volatility <= 0):
            raise ValueError("invalid trend-pullback teacher parameters")

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        hourly = observation.market_value("trend_1h")
        daily = observation.market_value("trend_1d")
        zscore = observation.market_value("zscore_1m")
        volatility = observation.market_value("vol_1m")
        current = observation.account_value("net_exposure")
        target = 0.0
        reason = "NO_PULLBACK_SETUP"
        if volatility > self.max_volatility:
            reason = "VOLATILITY_LIMIT"
        elif hourly > self.trend_threshold and daily > self.trend_threshold:
            expected_side = self.direction
            if abs(current) > 1e-8:
                if current * expected_side <= 0 or zscore * expected_side >= -self.exit_z:
                    reason = "PULLBACK_EXIT"
                else:
                    target, reason = self.size * expected_side, "PULLBACK_HOLD"
            elif zscore * expected_side <= -self.entry_z:
                target, reason = self.size * expected_side, "PULLBACK_ENTRY"
        elif hourly < -self.trend_threshold and daily < -self.trend_threshold:
            expected_side = -self.direction
            if abs(current) > 1e-8:
                if current * expected_side <= 0 or zscore * expected_side >= -self.exit_z:
                    reason = "PULLBACK_EXIT"
                else:
                    target, reason = self.size * expected_side, "PULLBACK_HOLD"
            elif zscore * expected_side <= -self.entry_z:
                target, reason = self.size * expected_side, "PULLBACK_ENTRY"
        else:
            reason = "TREND_ALIGNMENT_LOST"
        return advice(
            observation,
            self.expert_id,
            classify_target(current, target),
            target,
            reason,
        )


def main() -> int:
    runner.configure_torch_threads(runner.torch)
    if OUT.exists():
        raise FileExistsError(f"immutable trend-pullback output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    runner.WINDOWS = WINDOWS
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner_token = ACTIVE_OWNER.set("route-a-trend-pullback-probe-20261021")
    try:
        _, scaler_factory = runner.make_config(
            label="pullback-scaler-feb01",
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
        grid = []
        candidates = [
            TrendPullbackTeacher(
                direction=direction,
                entry_z=entry,
                exit_z=exit_z,
                expert_id=f"trend_pullback_d{direction}_e{entry:g}_x{exit_z:g}_v1",
            )
            for direction in DIRECTIONS
            for entry in ENTRY_Z
            for exit_z in EXIT_Z
        ]
        for window_id, start, end in WINDOWS:
            if window_id not in GRID_WINDOWS:
                continue
            for scenario in SCENARIOS:
                _, factory = runner.make_config(
                    label=f"pullback-grid-{window_id}-{scenario}",
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=runner.ns(start),
                    end_ns=runner.ns(end),
                    mode=scenario,
                    fit_scaler=False,
                    scaler=scaler,
                )
                for teacher in candidates:
                    result = runner.run_episode(
                        factory, teacher, marker_sha,
                        f"grid-{window_id}-{scenario}-{teacher.expert_id}",
                        11, EPISODE_STEPS,
                    )
                    evidence_sha = runner.write_verified(
                        OUT / "grid-evidence" / (
                            f"{window_id}-{scenario}-{teacher.expert_id}.json"
                        ),
                        result["evidence"],
                    )
                    grid.append(
                        {
                            "window_id": window_id,
                            "scenario": scenario,
                            "seed": 11,
                            "teacher": asdict(teacher),
                            "teacher_recipe_sha256": runner.teacher_config_hash(teacher),
                            "net_return": result["net_return"],
                            "max_drawdown": result["max_drawdown"],
                            "filled_orders": result["filled_orders"],
                            "liquidations": result["liquidations"],
                            "evidence_sha256": evidence_sha,
                        }
                    )
                del factory
                gc.collect()
        candidate_scores = []
        for teacher in candidates:
            rows = [row for row in grid if row["teacher"]["expert_id"] == teacher.expert_id]
            candidate_scores.append(
                {
                    "teacher": asdict(teacher),
                    "mean_net_return": sum(row["net_return"] for row in rows) / len(rows),
                    "positive_pairs": sum(row["net_return"] > 0 for row in rows),
                    "worst_net_return": min(row["net_return"] for row in rows),
                    "mean_filled_orders": sum(row["filled_orders"] for row in rows) / len(rows),
                }
            )
        selected_score = max(
            candidate_scores,
            key=lambda row: (
                row["positive_pairs"], row["worst_net_return"], row["mean_net_return"]
            ),
        )
        selected_config = selected_score["teacher"]
        selected = TrendPullbackTeacher(**selected_config)
        selected_hash = runner.teacher_config_hash(selected)
        baseline = WaitTeacher(expert_id="wait_flat_baseline_v1")
        baseline_hash = runner.teacher_config_hash(baseline)
        final_rows = []
        pairs = []
        for window_id, start, end in WINDOWS:
            for scenario in SCENARIOS:
                _, factory = runner.make_config(
                    label=f"pullback-final-{window_id}-{scenario}",
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
                    teacher_run = runner.run_episode(
                        factory, selected, marker_sha,
                        f"final-pullback-{key}", seed, EPISODE_STEPS,
                    )
                    baseline_run = runner.run_episode(
                        factory, baseline, marker_sha,
                        f"final-wait-{key}", seed, EPISODE_STEPS,
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
                            teacher_recipe=selected_hash,
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
                        )
                    )
                    final_rows.append(
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
            "schema": "rles.route_a.trend_pullback_teacher_probe.v1",
            "purpose": (
                "Train-only development; parameter grid and final rolling dates are reused "
                "development windows, not independent certification."
            ),
            "source_marker_sha256": marker_sha,
            "windows": [
                {"id": key, "start_ns": runner.ns(start), "end_ns": runner.ns(end)}
                for key, start, end in WINDOWS
            ],
            "grid_windows": sorted(GRID_WINDOWS),
            "grid": {
                "direction": list(DIRECTIONS),
                "entry_z": list(ENTRY_Z),
                "exit_z": list(EXIT_Z),
                "size": 0.4,
                "trend_threshold": 0.004,
                "seed": 11,
                "scenarios": list(SCENARIOS),
            },
            "grid_results": grid,
            "candidate_scores": candidate_scores,
            "selected_teacher": asdict(selected),
            "selected_teacher_recipe_sha256": selected_hash,
            "final_scenarios": list(SCENARIOS),
            "final_seeds": list(SEEDS),
            "final_results": final_rows,
            "teacher_screen": asdict(screen),
            "economic_status": "SCENARIO_ONLY",
        }
        output = OUT / "trend-pullback-report.json"
        runner.write_verified(output, report)
        print(json.dumps({
            "output": str(output),
            "grid_episodes": len(grid),
            "final_episodes": len(final_rows) * 2,
            "selected_teacher": selected.expert_id,
            "teacher_status": screen.status,
            "teacher_reasons": screen.reasons,
            "economic_status": screen.economic_status,
        }, indent=2), flush=True)
        return 0
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
