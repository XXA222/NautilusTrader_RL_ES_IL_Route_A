"""T05 MACD12/26/9 directional crossover teacher."""
from __future__ import annotations

from collections.abc import Mapping

from .base import BaseTeacher1m
from .contracts import TeacherSignal


class MacdCrossTeacher(BaseTeacher1m):
    required_fields = ("macd_dif", "macd_dea", "macd_hist")

    def entry_signal(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> TeacherSignal:
        if (previous["macd_dif"] <= previous["macd_dea"]
                and current["macd_dif"] > current["macd_dea"]
                and current["macd_hist"] > previous["macd_hist"]):
            return TeacherSignal("LONG_ENTRY", 1, "MACD_CROSS_UP_HIST_IMPROVING")
        if (previous["macd_dif"] >= previous["macd_dea"]
                and current["macd_dif"] < current["macd_dea"]
                and current["macd_hist"] < previous["macd_hist"]):
            return TeacherSignal("SHORT_ENTRY", -1, "MACD_CROSS_DOWN_HIST_WEAKENING")
        return TeacherSignal("NONE", 0, "NO_NEW_SIGNAL")

    def exit_signal(
        self,
        current_exposure: float,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> TeacherSignal | None:
        del previous
        if current_exposure > 0 and current["macd_dif"] <= current["macd_dea"]:
            return TeacherSignal("EXIT_LONG", 0, "MACD_REVERSE_CROSS_LONG")
        if current_exposure < 0 and current["macd_dif"] >= current["macd_dea"]:
            return TeacherSignal("EXIT_SHORT", 0, "MACD_REVERSE_CROSS_SHORT")
        return None
