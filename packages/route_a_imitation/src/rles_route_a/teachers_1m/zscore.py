"""T04 Z-score60 mean-reversion confirmation teacher."""
from __future__ import annotations

from collections.abc import Mapping

from .base import BaseTeacher1m
from .contracts import TeacherSignal


class ZscoreReentryTeacher(BaseTeacher1m):
    required_fields = ("zscore_60",)

    def entry_signal(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> TeacherSignal:
        prior = previous["zscore_60"]
        value = current["zscore_60"]
        if prior <= -2.0 and value > -2.0:
            if value <= -1.8:
                return TeacherSignal("LONG_ENTRY", 1, "ZSCORE_REENTRY_LONG")
            return TeacherSignal("NONE", 0, "FAST_REVERSION_MISSED")
        if prior >= 2.0 and value < 2.0:
            if value >= 1.8:
                return TeacherSignal("SHORT_ENTRY", -1, "ZSCORE_REENTRY_SHORT")
            return TeacherSignal("NONE", 0, "FAST_REVERSION_MISSED")
        return TeacherSignal("NONE", 0, "NO_NEW_SIGNAL")

    def exit_signal(
        self,
        current_exposure: float,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> TeacherSignal | None:
        opposite = self.entry_signal(current, previous)
        value = current["zscore_60"]
        if current_exposure > 0 and (value >= 0 or opposite.side < 0):
            return TeacherSignal("EXIT_LONG", 0, "ZSCORE_MEAN_OR_OPPOSITE")
        if current_exposure < 0 and (value <= 0 or opposite.side > 0):
            return TeacherSignal("EXIT_SHORT", 0, "ZSCORE_MEAN_OR_OPPOSITE")
        return None

