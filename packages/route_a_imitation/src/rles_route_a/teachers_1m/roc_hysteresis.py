"""T02 cost and volatility aware ROC hysteresis candidate."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from math import isfinite, log, sqrt
from statistics import pstdev
from typing import Any

from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation

from .base import BaseTeacher1m
from .contracts import TeacherRecipe1m, TeacherSignal
from .exits import AtrTrailingExit


class RocCostHysteresisTeacher(BaseTeacher1m):
    """Hold through isolated zero-crossings; trade only beyond observed costs/noise."""

    required_fields: tuple[str, ...] = (
        "roc_20",
        "roc_60",
        "atr_14",
        "bar_close_1m",
    )
    _VOLATILITY_RETURNS = 20

    def __init__(
        self,
        recipe: TeacherRecipe1m,
        *,
        atr_exit: AtrTrailingExit | None = None,
    ) -> None:
        super().__init__(recipe, atr_exit=atr_exit)
        parameters = dict(recipe.parameters)
        self.round_trip_cost_bps = self._number(
            parameters, "round_trip_cost_bps", 12.0,
        )
        direction_mode = parameters.get("direction_mode", "CONSENSUS_TREND_V1")
        if direction_mode not in ("CONSENSUS_TREND_V1", "CONSENSUS_FADE_V1"):
            raise ValueError("unknown ROC consensus direction mode")
        self._direction_multiplier = 1 if direction_mode == "CONSENSUS_TREND_V1" else -1
        self.entry_confirmation_bars = self._integer(
            parameters, "entry_confirmation_bars", 2,
        )
        self.rearm_neutral_bars = self._integer(
            parameters, "rearm_neutral_bars", 2,
        )
        self.entry_atr_fraction = self._number(
            parameters, "entry_atr_fraction", 0.25,
        )
        self.entry_volatility_fraction = self._number(
            parameters, "entry_volatility_fraction", 0.5,
        )
        self.exit_cost_fraction = self._number(
            parameters, "exit_cost_fraction", 0.25,
        )
        self.exit_atr_fraction = self._number(
            parameters, "exit_atr_fraction", 0.25,
        )
        self.exit_volatility_fraction = self._number(
            parameters, "exit_volatility_fraction", 0.5,
        )
        self.exit_confirmation_bars = self._integer(
            parameters, "exit_confirmation_bars", 2,
        )
        self._returns: deque[float] = deque(maxlen=self._VOLATILITY_RETURNS)
        self._consensus_side = 0
        self._consensus_run = 0
        self._neutral_run = 0
        self._last_entry_event_side = 0
        self._decision_is_flat = True
        self._adverse_side = 0
        self._adverse_run = 0
        rearm_mode = parameters.get(
            "cost_exit_rearm_mode", parameters.get("profit_lock_rearm_mode"),
        )
        if rearm_mode not in (
            None,
            "OPPOSITE_CONSENSUS_V1",
            "ATR_EXIT_OPPOSITE_CONSENSUS_V1",
        ):
            raise ValueError("unknown profit-lock rearm mode")
        self._cost_exit_rearm_enabled = rearm_mode is not None
        self._atr_exit_rearm_enabled = (
            rearm_mode == "ATR_EXIT_OPPOSITE_CONSENSUS_V1"
        )
        self._cost_exit_rearm_latched = False
        self._cost_exit_side = 0
        self._opposite_rearm_run = 0
        self._decision_has_pending_orders = False
        age_min = parameters.get("entry_ema_cross_age_min_bars")
        age_max = parameters.get("entry_ema_cross_age_max_bars")
        self._ema_cross_age_min_bars: int | None = None
        self._ema_cross_age_max_bars: int | None = None
        self._ema_cross_side = 0
        self._ema_cross_age_bars: int | None = None
        if age_min is not None or age_max is not None:
            if (type(age_min) is not int or age_min <= 0
                    or type(age_max) is not int or age_max < age_min):
                raise ValueError("EMA cross age gate requires positive ordered min/max bars")
            self._ema_cross_age_min_bars = age_min
            self._ema_cross_age_max_bars = age_max
            self.required_fields = (*self.required_fields, "ema_9", "ema_21")

    @staticmethod
    def _number(
        parameters: Mapping[str, Any], name: str, default: float
    ) -> float:
        value = parameters.get(name, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not isfinite(float(value)) or float(value) <= 0):
            raise ValueError(f"{name} must be positive and finite")
        return float(value)

    @staticmethod
    def _integer(
        parameters: Mapping[str, Any], name: str, default: int
    ) -> int:
        value = parameters.get(name, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        self._decision_is_flat = (
            abs(observation.account_value("net_exposure")) <= 1e-8
        )
        self._decision_has_pending_orders = bool(observation.pending_order_ids)
        if (self._cost_exit_rearm_latched
                and (not self._decision_is_flat or self._decision_has_pending_orders)):
            self._opposite_rearm_run = 0
        return super().advise(observation)

    def reset(self, episode_id: str = "") -> None:
        super().reset(episode_id)
        self._cost_exit_rearm_latched = False
        self._cost_exit_side = 0
        self._opposite_rearm_run = 0

    def on_discontinuity(self) -> None:
        self._returns.clear()
        self._consensus_side = 0
        self._consensus_run = 0
        self._neutral_run = 0
        self._last_entry_event_side = 0
        self._adverse_side = 0
        self._adverse_run = 0
        # Keep an armed re-entry latch across data gaps. A discontinuity resets
        # its evidence, so the opposite consensus must be observed again.
        self._opposite_rearm_run = 0
        self._ema_cross_side = 0
        self._ema_cross_age_bars = None

    def _update_ema_cross_age(
        self, current: Mapping[str, float], previous: Mapping[str, float],
    ) -> None:
        current_spread = current["ema_9"] - current["ema_21"]
        previous_spread = previous["ema_9"] - previous["ema_21"]
        current_side = 1 if current_spread > 0 else -1 if current_spread < 0 else 0
        if current_side == 0:
            self._ema_cross_side = 0
            self._ema_cross_age_bars = None
            return
        crossed = (
            (current_side > 0 and previous_spread <= 0 < current_spread)
            or (current_side < 0 and previous_spread >= 0 > current_spread)
        )
        if crossed:
            self._ema_cross_side = current_side
            self._ema_cross_age_bars = 0
        elif self._ema_cross_side == current_side:
            if self._ema_cross_age_bars is not None:
                self._ema_cross_age_bars += 1
        else:
            # Trend age is unknown until this episode observes a causal cross.
            self._ema_cross_side = current_side
            self._ema_cross_age_bars = None

    def _ema_cross_age_allows(self, side: int) -> bool:
        min_bars = self._ema_cross_age_min_bars
        max_bars = self._ema_cross_age_max_bars
        if min_bars is None or max_bars is None:
            return True
        age_bars = self._ema_cross_age_bars
        return (
            side == self._ema_cross_side
            and age_bars is not None
            and min_bars <= age_bars <= max_bars
        )
    def observe_native_action(
        self, action: ExpertAdvice, trace: Any, native_order_facts: Any,
    ) -> None:
        super().observe_native_action(action, trace, native_order_facts)
        if not self._cost_exit_rearm_enabled or action.expert_id != self.expert_id:
            return
        if action.intent != "EXIT":
            return
        exit_sides = {
            "COST_AWARE_ATR_PROFIT_LOCK_LONG": 1,
            "COST_AWARE_ATR_PROFIT_LOCK_SHORT": -1,
            "COST_AWARE_ATR_LOSS_GUARD_LONG": 1,
            "COST_AWARE_ATR_LOSS_GUARD_SHORT": -1,
        }
        if self._atr_exit_rearm_enabled:
            exit_sides.update({
                "ATR_TRAIL_EXIT_LONG": 1,
                "ATR_TRAIL_EXIT_SHORT": -1,
            })
        exit_side = next(
            (exit_sides[reason] for reason in action.reason_codes if reason in exit_sides),
            None,
        )
        if exit_side is None:
            return
        self._cost_exit_side = exit_side
        self._cost_exit_rearm_latched = True
        self._opposite_rearm_run = 0

    def _update_volatility(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> None:
        close = current["bar_close_1m"]
        previous_close = previous["bar_close_1m"]
        if close <= 0 or previous_close <= 0:
            return
        self._returns.append(log(close / previous_close))

    def _realized_volatility_bps(self) -> float | None:
        if len(self._returns) < self._VOLATILITY_RETURNS:
            return None
        return pstdev(self._returns) * sqrt(self._VOLATILITY_RETURNS) * 10_000

    def _atr_bps(self, current: Mapping[str, float]) -> float | None:
        close = current["bar_close_1m"]
        atr = current["atr_14"]
        if close <= 0 or atr <= 0:
            return None
        return atr / close * 10_000

    def entry_signal(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> TeacherSignal:
        self._update_volatility(current, previous)
        if self._ema_cross_age_min_bars is not None:
            self._update_ema_cross_age(current, previous)
        roc_20 = current["roc_20"]
        roc_60 = current["roc_60"]
        side = 1 if roc_20 > 0 and roc_60 > 0 else (
            -1 if roc_20 < 0 and roc_60 < 0 else 0
        )
        if side == 0:
            if self._cost_exit_rearm_latched:
                self._opposite_rearm_run = 0
            self._consensus_side = 0
            self._consensus_run = 0
            self._neutral_run += 1
            if self._neutral_run >= self.rearm_neutral_bars:
                self._last_entry_event_side = 0
            return TeacherSignal("NONE", 0, "ROC_HYSTERESIS_NEUTRAL")

        self._neutral_run = 0
        if side == self._consensus_side:
            self._consensus_run += 1
        else:
            self._consensus_side = side
            self._consensus_run = 1

        if self._cost_exit_rearm_latched:
            opposite_side = -self._cost_exit_side * self._direction_multiplier
            if not self._decision_is_flat or self._decision_has_pending_orders:
                self._opposite_rearm_run = 0
                reason = (
                    "ROC_ATR_EXIT_REARM_WAIT_FLAT"
                    if self._atr_exit_rearm_enabled
                    else "ROC_PROFIT_LOCK_REARM_WAIT_FLAT"
                )
                return TeacherSignal("NONE", 0, reason)
            if side == opposite_side:
                self._opposite_rearm_run += 1
            else:
                self._opposite_rearm_run = 0
            if self._opposite_rearm_run < self.entry_confirmation_bars:
                reason = (
                    "ROC_ATR_EXIT_REARM_WAIT_OPPOSITE_CONSENSUS"
                    if self._atr_exit_rearm_enabled
                    else "ROC_PROFIT_LOCK_REARM_WAIT_OPPOSITE_CONSENSUS"
                )
                return TeacherSignal("NONE", 0, reason)
            self._cost_exit_rearm_latched = False
            self._cost_exit_side = 0
            self._opposite_rearm_run = 0

        if self._consensus_run < self.entry_confirmation_bars:
            return TeacherSignal("NONE", 0, "ROC_CONFIRMING_CONSENSUS")
        if not self._decision_is_flat:
            return TeacherSignal("NONE", 0, "ROC_POSITION_OPEN")
        entry_side = self._direction_multiplier * side
        if self._last_entry_event_side == entry_side:
            return TeacherSignal("NONE", 0, "ROC_CONSENSUS_EVENT_USED")
        if not self._ema_cross_age_allows(entry_side):
            return TeacherSignal("NONE", 0, "ROC_EMA_CROSS_AGE_OUTSIDE_21_60")

        realized_volatility_bps = self._realized_volatility_bps()
        if realized_volatility_bps is None:
            return TeacherSignal("NONE", 0, "ROC_VOLATILITY_NOT_READY")
        atr_bps = self._atr_bps(current)
        if atr_bps is None:
            return TeacherSignal("NONE", 0, "ROC_ATR_NOT_READY")
        edge_bps = min(side * roc_20, side * roc_60) * 10_000
        noise_buffer_bps = max(
            self.entry_atr_fraction * atr_bps,
            self.entry_volatility_fraction * realized_volatility_bps,
        )
        required_edge_bps = self.round_trip_cost_bps + noise_buffer_bps
        if edge_bps <= required_edge_bps:
            return TeacherSignal("NONE", 0, "ROC_EDGE_BELOW_COST_AND_NOISE")
        self._last_entry_event_side = entry_side
        direction = "CONSENSUS" if self._direction_multiplier > 0 else "CONTRARIAN"
        if entry_side > 0:
            return TeacherSignal(
                "LONG_ENTRY", 1, f"ROC_COST_AWARE_{direction}_LONG",
            )
        return TeacherSignal(
            "SHORT_ENTRY", -1, f"ROC_COST_AWARE_{direction}_SHORT",
        )

    def exit_signal(
        self,
        current_exposure: float,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> TeacherSignal | None:
        del previous
        if abs(current_exposure) <= 1e-8:
            self._adverse_side = 0
            self._adverse_run = 0
            return None

        side = 1 if current_exposure > 0 else -1
        if side != self._adverse_side:
            self._adverse_side = side
            self._adverse_run = 0

        atr_bps = self._atr_bps(current)
        realized_volatility_bps = self._realized_volatility_bps()
        if atr_bps is None:
            return None
        volatility_bps = realized_volatility_bps or 0.0
        exit_buffer_bps = max(
            self.exit_cost_fraction * self.round_trip_cost_bps,
            self.exit_atr_fraction * atr_bps,
            self.exit_volatility_fraction * volatility_bps,
        )
        threshold = exit_buffer_bps / 10_000
        adverse = (
            side * current["roc_20"] <= -threshold
            and side * current["roc_60"] <= -threshold
        )
        if adverse:
            self._adverse_run += 1
        else:
            self._adverse_run = 0
        if self._adverse_run < self.exit_confirmation_bars:
            return None
        self._adverse_run = 0
        if side > 0:
            return TeacherSignal(
                "EXIT_LONG", 0, "ROC_COST_HYSTERESIS_CONFIRMED_EXIT_LONG",
            )
        return TeacherSignal(
            "EXIT_SHORT", 0, "ROC_COST_HYSTERESIS_CONFIRMED_EXIT_SHORT",
        )
