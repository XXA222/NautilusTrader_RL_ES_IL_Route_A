"""T07 Bollinger20/2 band re-entry teacher."""
from __future__ import annotations

from collections.abc import Mapping

from .base import BaseTeacher1m
from .contracts import TeacherSignal


class BollingerReentryTeacher(BaseTeacher1m):
    required_fields = (
        "bar_close_1m",
        "bollinger_mid_20",
        "bollinger_upper_20_2",
        "bollinger_lower_20_2",
    )

    def entry_signal(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> TeacherSignal:
        close = current["bar_close_1m"]
        if (previous["bar_close_1m"] < previous["bollinger_lower_20_2"]
                and close >= current["bollinger_lower_20_2"]):
            return TeacherSignal("LONG_ENTRY", 1, "BOLLINGER_REENTERED_LOWER_BAND")
        if (previous["bar_close_1m"] > previous["bollinger_upper_20_2"]
                and close <= current["bollinger_upper_20_2"]):
            return TeacherSignal("SHORT_ENTRY", -1, "BOLLINGER_REENTERED_UPPER_BAND")
        return TeacherSignal("NONE", 0, "NO_NEW_SIGNAL")

    def exit_signal(
        self,
        current_exposure: float,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> TeacherSignal | None:
        del previous
        close = current["bar_close_1m"]
        if current_exposure > 0 and (
            close >= current["bollinger_mid_20"]
            or close < current["bollinger_lower_20_2"]
        ):
            return TeacherSignal("EXIT_LONG", 0, "BOLLINGER_MID_OR_LOWER_RISK_EXIT")
        if current_exposure < 0 and (
            close <= current["bollinger_mid_20"]
            or close > current["bollinger_upper_20_2"]
        ):
            return TeacherSignal("EXIT_SHORT", 0, "BOLLINGER_MID_OR_UPPER_RISK_EXIT")
        return None
