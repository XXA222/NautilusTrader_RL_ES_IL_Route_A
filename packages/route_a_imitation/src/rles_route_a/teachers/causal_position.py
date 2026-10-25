"""P0 causal candidate and continuation baseline; not an alpha claim."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from rles_route_a.teacher_research import CausalOpportunityGenerator
from rles_route_a.teacher_research.causal_exit import TrendInvalidationExit
from .protocol import advice


@dataclass(frozen=True, slots=True)
class CausalPositionTeacher:
    """Trade frozen G-A/G-B candidates and hold until trend invalidation.

    It deliberately has no maximum holding-time rule. Costs and fills remain
    host/native facts; this simple policy is only an engineering baseline.
    Hard-risk priority is supplied by ``FrozenTeacherSelector``.
    """

    size: float = 0.3
    trend_invalidation_threshold: float = 0.002
    max_entry_volatility: float = 0.08
    entry_cooldown_seconds: float = 300.0
    entry_cadence_minutes: int = 1
    exit_scale_policy: Literal["both", "either"] = "both"
    generator: CausalOpportunityGenerator = field(
        default_factory=CausalOpportunityGenerator
    )
    expert_id: str = "causal_position_p0_v2"

    def __post_init__(self) -> None:
        if (not all(isfinite(value) for value in (
                self.size, self.trend_invalidation_threshold, self.max_entry_volatility,
                self.entry_cooldown_seconds))
                or not 0 < self.size <= 1
                or self.trend_invalidation_threshold < 0
                or self.max_entry_volatility <= 0
                or self.entry_cooldown_seconds < 0
                or type(self.entry_cadence_minutes) is not int
                or self.entry_cadence_minutes < 1
                or self.exit_scale_policy not in ("both", "either")
                or not isinstance(self.generator, CausalOpportunityGenerator)
                or not isinstance(self.expert_id, str) or not self.expert_id.strip()):
            raise ValueError("invalid frozen causal position teacher recipe")

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        current = observation.account_value("net_exposure")
        hourly = observation.market_value("trend_1h")
        daily = observation.market_value("trend_1d")

        # Position management precedes all entry logic. Native duration is
        # observed for audit, but is never a normal exit trigger.
        if abs(current) > 1e-8:
            duration = observation.account_value("position_duration_ns")
            if duration < 0:
                raise ValueError("native position duration cannot be negative")
            invalidated = TrendInvalidationExit(
                self.trend_invalidation_threshold, self.exit_scale_policy,
            ).invalidated(
                1 if current > 0 else -1, hourly, daily,
            )
            if invalidated:
                return advice(
                    observation, self.expert_id, "EXIT", 0.0,
                    "TREND_INVALIDATED_EXIT",
                )
            return advice(
                observation, self.expert_id, "HOLD", current,
                "TREND_CONTINUATION_HOLD",
            )

        # Cadence constrains entry only; an existing position is managed every
        # native barrier above. Integer UTC minute buckets tolerate the feed's
        # declared sub-minute availability delay without moving to future data.
        if (observation.decision_ts_ns // 60_000_000_000) % self.entry_cadence_minutes:
            return advice(
                observation, self.expert_id, "FLAT", 0.0, "ENTRY_CADENCE_WAIT",
            )
        if observation.market_value("vol_1m") > self.max_entry_volatility:
            return advice(
                observation, self.expert_id, "FLAT", 0.0,
                "ENTRY_VETO_HIGH_VOL",
            )
        last_fill_age_seconds = observation.account_value("native_last_fill_age_seconds")
        if last_fill_age_seconds < -1:
            raise ValueError("native last-fill age must be nonnegative or the -1 sentinel")
        if (last_fill_age_seconds >= 0
                and last_fill_age_seconds < self.entry_cooldown_seconds):
            return advice(
                observation, self.expert_id, "FLAT", 0.0,
                "ENTRY_COOLDOWN",
            )
        candidates = self.generator.candidates(observation)
        if not candidates:
            return advice(
                observation, self.expert_id, "FLAT", 0.0,
                "NO_ELIGIBLE_CANDIDATE",
            )
        sides = {candidate.side for candidate in candidates}
        if len(sides) != 1:
            return advice(
                observation, self.expert_id, "FLAT", 0.0,
                "CONFLICTING_CANDIDATE_SIDES",
            )
        candidate = candidates[0]
        target = self.size * candidate.side
        intent = "OPEN_LONG" if candidate.side > 0 else "OPEN_SHORT"
        return advice(
            observation, self.expert_id, intent, target,
            f"CANDIDATE_{candidate.family.upper()}",
        )
