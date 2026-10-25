"""Shared P0 exit decision, also used to construct research label endpoints.

This module has no market IO, account simulator, or maximum holding horizon.
Native hard-risk and pending-order handling remain owned by the host/selector.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal

from rles_route_a.contracts import PointInTimeObservation


@dataclass(frozen=True, slots=True)
class TrendInvalidationExit:
    threshold: float = 0.002
    scale_policy: Literal["both", "either"] = "both"

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("exit threshold must be finite and nonnegative")
        if self.scale_policy not in ("both", "either"):
            raise ValueError("exit scale policy must be both or either")

    @property
    def policy_id(self) -> str:
        if self.scale_policy == "both":
            return f"p0-trend-invalidation-v1:{float(self.threshold).hex()}"
        return f"p0-trend-invalidation-either-v1:{float(self.threshold).hex()}"

    def invalidated(self, side: int, hourly: float, daily: float) -> bool:
        if type(side) is not int or side not in (-1, 1):
            raise ValueError("exit side must be -1 or 1")
        if not isfinite(hourly) or not isfinite(daily):
            raise ValueError("exit features must be finite")
        hourly_invalid = side * hourly < -self.threshold
        daily_invalid = side * daily < -self.threshold
        return (hourly_invalid and daily_invalid if self.scale_policy == "both"
                else hourly_invalid or daily_invalid)


@dataclass(frozen=True, slots=True)
class CausalExitSchedule:
    """Research-only future label endpoints, NEVER an input to runtime advice.

    All barriers in the authorized interval are required. If no ordinary exit
    is observed, the outcome is censored; the interval end is not an exit.
    A caller must separately choose a strictly later fill and censor that fill
    if it falls outside the block. This schedule does not model native risk.
    """

    policy_id: str
    start_ns: int
    end_ns: int
    long_exit_decisions: tuple[int, ...]
    short_exit_decisions: tuple[int, ...]

    @classmethod
    def from_observations(
        cls, observations: Iterable[PointInTimeObservation], *,
        rule: TrendInvalidationExit, start_ns: int, end_ns: int, cadence_ns: int,
    ) -> CausalExitSchedule:
        if (any(type(value) is not int for value in (start_ns, end_ns, cadence_ns))
                or start_ns <= 0 or end_ns <= start_ns or cadence_ns <= 0):
            raise ValueError("invalid exit schedule interval/cadence")
        longs, shorts = [], []
        previous = None
        identity = None
        for obs in observations:
            ts = obs.decision_ts_ns
            if not start_ns <= ts < end_ns:
                raise ValueError("exit observation outside authorized interval")
            if ((previous is None and ts - start_ns >= cadence_ns)
                    or (previous is not None and ts - previous != cadence_ns)):
                raise ValueError("exit schedule has missing, duplicated, or reordered barriers")
            current_identity = (obs.source_id, obs.snapshot_id, obs.feature_spec_hash, obs.fold_id)
            if any(not value for value in current_identity):
                raise ValueError("exit observation provenance is required")
            if identity is not None and current_identity != identity:
                raise ValueError("exit observation provenance changed")
            identity = current_identity
            hourly = obs.market_value("trend_1h")
            daily = obs.market_value("trend_1d")
            if rule.invalidated(1, hourly, daily):
                longs.append(ts)
            if rule.invalidated(-1, hourly, daily):
                shorts.append(ts)
            previous = ts
        if previous is None or end_ns - previous > cadence_ns:
            raise ValueError("exit schedule is empty or ends before its declared coverage")
        return cls(rule.policy_id, start_ns, end_ns, tuple(longs), tuple(shorts))

    def next_exit_decision(self, *, side: int, entry_ts_ns: int, block_end_ns: int) -> int | None:
        if type(side) is not int or side not in (-1, 1):
            raise ValueError("exit side must be -1 or 1")
        if (type(entry_ts_ns) is not int or type(block_end_ns) is not int
                or not self.start_ns <= entry_ts_ns < block_end_ns <= self.end_ns):
            raise ValueError("label interval outside exit schedule coverage")
        decisions = self.long_exit_decisions if side == 1 else self.short_exit_decisions
        index = bisect_right(decisions, entry_ts_ns)
        if index == len(decisions) or decisions[index] >= block_end_ns:
            return None
        return decisions[index]
