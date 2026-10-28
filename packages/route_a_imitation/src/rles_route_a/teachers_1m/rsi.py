"""T06 RSI14 oversold/overbought recovery teacher."""
from __future__ import annotations

from collections.abc import Mapping

from .base import BaseTeacher1m
from .contracts import TeacherSignal


class RsiReentryTeacher(BaseTeacher1m):
    required_fields = ("rsi_14",)
    oversold = 30.0
    overbought = 70.0
    neutral = 50.0

    def entry_signal(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> TeacherSignal:
        if previous["rsi_14"] <= self.oversold and current["rsi_14"] > self.oversold:
            return TeacherSignal("LONG_ENTRY", 1, "RSI_RECOVERED_FROM_OVERSOLD")
        if previous["rsi_14"] >= self.overbought and current["rsi_14"] < self.overbought:
            return TeacherSignal("SHORT_ENTRY", -1, "RSI_RECOVERED_FROM_OVERBOUGHT")
        return TeacherSignal("NONE", 0, "NO_NEW_SIGNAL")

    def exit_signal(
        self,
        current_exposure: float,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> TeacherSignal | None:
        del previous
        if current_exposure > 0 and current["rsi_14"] >= self.neutral:
            return TeacherSignal("EXIT_LONG", 0, "RSI_NEUTRAL_LONG")
        if current_exposure < 0 and current["rsi_14"] <= self.neutral:
            return TeacherSignal("EXIT_SHORT", 0, "RSI_NEUTRAL_SHORT")
        return None
