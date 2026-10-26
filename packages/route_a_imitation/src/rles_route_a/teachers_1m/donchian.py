"""T03 Donchian20 breakout teacher with Donchian10 exit."""
from __future__ import annotations

from collections.abc import Mapping

from .base import BaseTeacher1m
from .contracts import TeacherSignal


class DonchianBreakoutTeacher(BaseTeacher1m):
    required_fields = (
        "bar_close_1m",
        "donchian_upper_20",
        "donchian_lower_20",
        "donchian_upper_10",
        "donchian_lower_10",
    )

    def entry_signal(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> TeacherSignal:
        if (previous["bar_close_1m"] <= previous["donchian_upper_20"]
                and current["bar_close_1m"] > current["donchian_upper_20"]):
            return TeacherSignal("LONG_ENTRY", 1, "DONCHIAN_BREAKOUT_UP")
        if (previous["bar_close_1m"] >= previous["donchian_lower_20"]
                and current["bar_close_1m"] < current["donchian_lower_20"]):
            return TeacherSignal("SHORT_ENTRY", -1, "DONCHIAN_BREAKOUT_DOWN")
        return TeacherSignal("NONE", 0, "NO_NEW_SIGNAL")

    def exit_signal(
        self,
        current_exposure: float,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> TeacherSignal | None:
        del previous
        if current_exposure > 0 and current["bar_close_1m"] < current["donchian_lower_10"]:
            return TeacherSignal("EXIT_LONG", 0, "DONCHIAN_EXIT_DOWN")
        if current_exposure < 0 and current["bar_close_1m"] > current["donchian_upper_10"]:
            return TeacherSignal("EXIT_SHORT", 0, "DONCHIAN_EXIT_UP")
        return None

