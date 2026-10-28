"""T01 EMA9/21 crossing teacher."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from .base import BaseTeacher1m
from .contracts import TeacherRecipe1m, TeacherSignal
from .exits import AtrTrailingExit


class EmaCrossTeacher(BaseTeacher1m):
    required_fields = ("ema_9", "ema_21")

    def __init__(
        self, recipe: TeacherRecipe1m, *, atr_exit: AtrTrailingExit | None = None
    ) -> None:
        super().__init__(recipe, atr_exit=atr_exit)
        self.reversal_mode = dict(recipe.parameters).get(
            "reversal_mode", "EXIT_ONLY_V1"
        )
        if self.reversal_mode not in ("EXIT_ONLY_V1", "REVERSE_AFTER_FLAT_V1"):
            raise ValueError("unknown EMA reversal lifecycle mode")

    def reversal_side_for_exit(
        self, exit_signal: TeacherSignal
    ) -> Literal[-1, 1] | None:
        if self.reversal_mode != "REVERSE_AFTER_FLAT_V1":
            return None
        if exit_signal.raw_signal == "EXIT_LONG":
            return -1
        if exit_signal.raw_signal == "EXIT_SHORT":
            return 1
        return None

    def reversal_entry_signal(
        self, side: Literal[-1, 1], current: Mapping[str, float]
    ) -> TeacherSignal | None:
        if side < 0 and current["ema_9"] < current["ema_21"]:
            return TeacherSignal("SHORT_ENTRY", -1, "EMA_REVERSAL_RECHECK")
        if side > 0 and current["ema_9"] > current["ema_21"]:
            return TeacherSignal("LONG_ENTRY", 1, "EMA_REVERSAL_RECHECK")
        return None

    def entry_signal(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> TeacherSignal:
        if previous["ema_9"] <= previous["ema_21"] and current["ema_9"] > current["ema_21"]:
            return TeacherSignal("LONG_ENTRY", 1, "EMA_CROSS_LONG")
        if previous["ema_9"] >= previous["ema_21"] and current["ema_9"] < current["ema_21"]:
            return TeacherSignal("SHORT_ENTRY", -1, "EMA_CROSS_SHORT")
        return TeacherSignal("NONE", 0, "NO_NEW_SIGNAL")

    def exit_signal(
        self,
        current_exposure: float,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> TeacherSignal | None:
        signal = self.entry_signal(current, previous)
        if current_exposure > 0 and signal.side < 0:
            return TeacherSignal("EXIT_LONG", 0, "EMA_REVERSE_CROSS")
        if current_exposure < 0 and signal.side > 0:
            return TeacherSignal("EXIT_SHORT", 0, "EMA_REVERSE_CROSS")
        return None

