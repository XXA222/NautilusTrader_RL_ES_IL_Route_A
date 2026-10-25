"""Probe causal re-entry cooldown after stop-loss on registered Train only."""

from __future__ import annotations

from dataclasses import dataclass, field
import json

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from rles_route_a.teachers.protocol import advice, classify_target
from rles_route_a.teachers.trend import TrendTeacher

import run_route_a_train_student_screen_20261020 as runner


OUT = runner.ROOT / ".artifacts" / "route-a-stop-cooldown-probe-20261021-run02"
WINDOWS = (
    ("april13", "2024-04-13T00:00:00+00:00", "2024-04-14T00:00:00+00:00"),
    ("april14", "2024-04-14T00:00:00+00:00", "2024-04-15T00:00:00+00:00"),
)
SCENARIOS = ("base", "stress")
SEEDS = (11, 29)
COOLDOWN_MINUTES = (5, 15, 30, 60)
STOP_LOSS_FRACTION = 0.001
EPISODE_STEPS = 1_400


@dataclass(frozen=True)
class StopCooldownTeacher:
    """Stateful within-episode rule: close at stop, then remain flat to expiry."""

    primary: TrendTeacher
    stop_loss_fraction: float
    cooldown_ns: int
    expert_id: str
    cooldown_until_ns: list[int] = field(
        default_factory=lambda: [0], init=False, repr=False, compare=False
    )

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        current = observation.account_value("net_exposure")
        unrealized = observation.account_value("unrealized_pnl_fraction")
        now_ns = observation.decision_ts_ns
        if abs(current) > 1e-8 and unrealized <= -self.stop_loss_fraction:
            self.cooldown_until_ns[0] = now_ns + self.cooldown_ns
            target, reason = 0.0, "POSITION_STOP_LOSS"
        elif now_ns < self.cooldown_until_ns[0]:
            target, reason = 0.0, "POST_STOP_COOLDOWN"
        else:
            return self.primary.advise(observation)
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
        raise FileExistsError(f"immutable stop-cooldown output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner_token = ACTIVE_OWNER.set("route-a-stop-cooldown-probe-20261021")
    try:
        _, fit_factory = runner.make_config(
            label="stop-cooldown-scaler-jan02-jan09",
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
                    label=f"cooldown-{window_id}-{scenario}",
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=runner.ns(start),
                    end_ns=runner.ns(end),
                    mode=scenario,
                    fit_scaler=False,
                    scaler=fit_factory.scaler,
                )
                for cooldown_minutes in COOLDOWN_MINUTES:
                    for seed in SEEDS:
                        teacher = StopCooldownTeacher(
                            primary=TrendTeacher(
                                size=0.8,
                                threshold=0.004,
                                hold_on_neutral=True,
                                direction=-1,
                                expert_id="countertrend_hold_neutral_v1",
                            ),
                            stop_loss_fraction=STOP_LOSS_FRACTION,
                            cooldown_ns=cooldown_minutes * 60_000_000_000,
                            expert_id=f"countertrend_stop_cooldown_{cooldown_minutes}m_v1",
                        )
                        label = f"{window_id}-{scenario}-cooldown-{cooldown_minutes}m-seed-{seed}"
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
                                "stop_loss_fraction": STOP_LOSS_FRACTION,
                                "cooldown_minutes": cooldown_minutes,
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
            "schema": "rles.route_a.stop_cooldown_probe.v1",
            "purpose": "Train-only risk-overlay development; not promotion evidence",
            "source_marker_sha256": marker_sha,
            "windows": [
                {"id": key, "start_ns": runner.ns(start), "end_ns": runner.ns(end)}
                for key, start, end in WINDOWS
            ],
            "scenarios": list(SCENARIOS),
            "seeds": list(SEEDS),
            "stop_loss_fraction": STOP_LOSS_FRACTION,
            "cooldown_minutes": list(COOLDOWN_MINUTES),
            "teacher_alpha": {
                "expert_id": "countertrend_hold_neutral_v1",
                "size": 0.8,
                "threshold": 0.004,
                "hold_on_neutral": True,
                "direction": -1,
            },
            "rows": rows,
            "economic_status": "SCENARIO_ONLY",
            "promotion_eligible": False,
        }
        runner.write_verified(OUT / "stop-cooldown-report.json", report)
        print(json.dumps({"output": str(OUT), "episode_count": len(rows)}, indent=2))
        return 0
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
