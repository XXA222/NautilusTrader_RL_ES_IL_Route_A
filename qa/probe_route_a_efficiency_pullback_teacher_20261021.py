"""Screen a pullback teacher with causal multi-timeframe efficiency filters."""

from __future__ import annotations

from dataclasses import dataclass

import probe_route_a_trend_pullback_teacher_20261021 as base


@dataclass(frozen=True)
class EfficiencyPullbackTeacher(base.TrendPullbackTeacher):
    min_efficiency_15m: float = 0.15
    min_efficiency_1h: float = 0.25
    expert_id: str = "efficiency_pullback_v1"

    def advise(self, observation):
        efficiency_15m = observation.market_value("efficiency_15m")
        efficiency_1h = observation.market_value("efficiency_1h")
        current = observation.account_value("net_exposure")
        weak_efficiency = (
            abs(efficiency_15m) < self.min_efficiency_15m
            or abs(efficiency_1h) < self.min_efficiency_1h
        )
        if abs(current) > 1e-8 and (
            weak_efficiency
        ):
            from rles_route_a.teachers.protocol import classify_target

            return base.advice(
                observation, self.expert_id, classify_target(current, 0.0), 0.0,
                "EFFICIENCY_FILTER_EXIT",
            )
        if weak_efficiency:
            from rles_route_a.teachers.protocol import classify_target

            return base.advice(
                observation, self.expert_id, classify_target(current, 0.0), 0.0,
                "LOW_TREND_EFFICIENCY",
            )
        return super().advise(observation)


base.OUT = base.runner.ROOT / ".artifacts" / "route-a-efficiency-pullback-teacher-20261021-run02"
base.TrendPullbackTeacher = EfficiencyPullbackTeacher


if __name__ == "__main__":
    raise SystemExit(base.main())
