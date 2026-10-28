"""Frozen research-side contracts for independent Route A 1m teachers."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal


RawSignal = Literal["NONE", "LONG_ENTRY", "SHORT_ENTRY", "EXIT_LONG", "EXIT_SHORT"]
Side = Literal[-1, 0, 1]


@dataclass(frozen=True, slots=True)
class TeacherSignal:
    raw_signal: RawSignal
    side: Side
    reason_code: str

    def __post_init__(self) -> None:
        if self.raw_signal not in (
            "NONE", "LONG_ENTRY", "SHORT_ENTRY", "EXIT_LONG", "EXIT_SHORT"
        ):
            raise ValueError("invalid raw teacher signal")
        if self.side not in (-1, 0, 1) or not self.reason_code:
            raise ValueError("invalid teacher signal side or reason")


@dataclass(slots=True)
class PendingReversal1m:
    """Causal EMA reverse candidate awaiting a verified native close lifecycle."""

    side: Literal[-1, 1]
    event_close_ts_ns: int
    episode_id: str
    source_id: str
    reason_code: str
    close_request_seen: bool = False
    close_order_ids: set[str] = field(default_factory=set)
    close_confirmed_bar_close_ts_ns: int | None = None

    def __post_init__(self) -> None:
        if self.side not in (-1, 1):
            raise ValueError("pending reversal side must be directional")
        if type(self.event_close_ts_ns) is not int or self.event_close_ts_ns <= 0:
            raise ValueError("pending reversal event clock must be positive")
        if not self.episode_id or not self.reason_code:
            raise ValueError("pending reversal episode and reason are required")
        if any(not isinstance(order_id, str) or not order_id.strip()
               for order_id in self.close_order_ids):
            raise ValueError("pending reversal close order ids must be nonempty")
        if (self.close_confirmed_bar_close_ts_ns is not None
                and (type(self.close_confirmed_bar_close_ts_ns) is not int
                     or self.close_confirmed_bar_close_ts_ns < self.event_close_ts_ns)):
            raise ValueError("pending reversal close confirmation clock is invalid")


@dataclass(frozen=True, slots=True)
class TeacherRecipe1m:
    teacher_id: str
    indicator_recipe_id: str
    parameters: tuple[tuple[str, float | int | str], ...]
    warmup_bars: int
    entry_rule_id: str
    indicator_exit_rule_id: str
    htf_mode: Literal["FEATURE_ONLY", "DISABLED", "OPTIONAL_FILTER"]
    target_exposure: float
    cooldown_policy: str
    order_policy: str
    risk_recipe_id: str
    scenario_cost_ids: tuple[str, ...]
    config_revision: str = "route_a_1m_multiteacher_v1"

    def __post_init__(self) -> None:
        if (not self.teacher_id or not self.indicator_recipe_id or not self.entry_rule_id
                or not self.indicator_exit_rule_id or not self.config_revision):
            raise ValueError("teacher recipe identities are required")
        if type(self.warmup_bars) is not int or self.warmup_bars <= 0:
            raise ValueError("teacher warmup must be positive")
        if self.htf_mode not in ("FEATURE_ONLY", "DISABLED", "OPTIONAL_FILTER"):
            raise ValueError("invalid higher-timeframe mode")
        if (isinstance(self.target_exposure, bool)
                or not isinstance(self.target_exposure, (int, float))
                or not isfinite(self.target_exposure)
                or not 0 < self.target_exposure <= 1):
            raise ValueError("teacher target exposure must be in (0, 1]")
        if not self.cooldown_policy or not self.order_policy or not self.risk_recipe_id:
            raise ValueError("teacher execution policies must be explicit")
        if not self.scenario_cost_ids or len(set(self.scenario_cost_ids)) != len(
            self.scenario_cost_ids
        ):
            raise ValueError("teacher cost scenarios must be explicit and unique")
        parameter_names = tuple(name for name, _value in self.parameters)
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("duplicate teacher parameter")

