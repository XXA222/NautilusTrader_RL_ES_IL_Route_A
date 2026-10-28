"""T08 EMA trend with ATR-normalized pullback/recovery entries."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from .base import BaseTeacher1m
from .contracts import TeacherRecipe1m, TeacherSignal
from .exits import AtrTrailingExit


class TrendPullbackTeacher(BaseTeacher1m):
    required_fields = (
        "bar_close_1m",
        "ema_9",
        "ema_21",
        "atr_14",
        "donchian_upper_10",
        "donchian_lower_10",
    )

    def __init__(
        self, recipe: TeacherRecipe1m, *, atr_exit: AtrTrailingExit | None = None
    ) -> None:
        super().__init__(recipe, atr_exit=atr_exit)
        self._prior_closes: deque[float] = deque(maxlen=5)

    def on_discontinuity(self) -> None:
        self._prior_closes.clear()

    def entry_signal(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> TeacherSignal:
        del previous
        close = current["bar_close_1m"]
        atr = current["atr_14"]
        prior = tuple(self._prior_closes)
        self._prior_closes.append(close)
        if atr <= 0 or len(prior) < 5:
            return TeacherSignal("NONE", 0, "PULLBACK_HISTORY_NOT_READY")
        depth = (max(prior) - prior[-1]) / atr
        short_depth = (prior[-1] - min(prior)) / atr
        recover_up = close > max(prior[-3:])
        recover_down = close < min(prior[-3:])
        if (current["ema_9"] > current["ema_21"]
                and 0.3 <= depth <= 2.0 and recover_up):
            return TeacherSignal("LONG_ENTRY", 1, "UPTREND_PULLBACK_RECOVERY")
        if (current["ema_9"] < current["ema_21"]
                and 0.3 <= short_depth <= 2.0 and recover_down):
            return TeacherSignal("SHORT_ENTRY", -1, "DOWNTREND_PULLBACK_RECOVERY")
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
            current["ema_9"] <= current["ema_21"]
            or close < current["donchian_lower_10"]
        ):
            return TeacherSignal("EXIT_LONG", 0, "PULLBACK_TREND_OR_CHANNEL_EXIT_LONG")
        if current_exposure < 0 and (
            current["ema_9"] >= current["ema_21"]
            or close > current["donchian_upper_10"]
        ):
            return TeacherSignal("EXIT_SHORT", 0, "PULLBACK_TREND_OR_CHANNEL_EXIT_SHORT")
        return None
