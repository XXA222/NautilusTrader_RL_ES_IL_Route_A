"""T02 ROC20/60 consensus teacher."""
from __future__ import annotations

from collections.abc import Mapping

from .base import BaseTeacher1m
from .contracts import TeacherSignal


class RocConsensusTeacher(BaseTeacher1m):
    required_fields = ("roc_20", "roc_60")

    def entry_signal(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> TeacherSignal:
        long_now = current["roc_20"] > 0 and current["roc_60"] > 0
        long_before = previous["roc_20"] > 0 and previous["roc_60"] > 0
        short_now = current["roc_20"] < 0 and current["roc_60"] < 0
        short_before = previous["roc_20"] < 0 and previous["roc_60"] < 0
        if long_now and not long_before:
            return TeacherSignal("LONG_ENTRY", 1, "ROC_CONSENSUS_LONG")
        if short_now and not short_before:
            return TeacherSignal("SHORT_ENTRY", -1, "ROC_CONSENSUS_SHORT")
        return TeacherSignal("NONE", 0, "NO_NEW_SIGNAL")

    def exit_signal(
        self,
        current_exposure: float,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> TeacherSignal | None:
        del previous
        if current_exposure > 0 and (current["roc_20"] <= 0 or current["roc_60"] <= 0):
            return TeacherSignal("EXIT_LONG", 0, "ROC_CONSENSUS_BROKEN")
        if current_exposure < 0 and (current["roc_20"] >= 0 or current["roc_60"] >= 0):
            return TeacherSignal("EXIT_SHORT", 0, "ROC_CONSENSUS_BROKEN")
        return None

