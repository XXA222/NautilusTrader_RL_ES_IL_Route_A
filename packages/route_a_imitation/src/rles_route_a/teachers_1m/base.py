"""Common causal state machine for independent one-minute teachers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from rles_route_a.indicators_1m import MINUTE_NS
from rles_route_a.teachers.protocol import advice

from .contracts import TeacherRecipe1m, TeacherSignal
from .exits import AtrTrailingExit


class BaseTeacher1m(ABC):
    """Evaluate one completed 1m snapshot once; account execution remains native."""

    required_fields: tuple[str, ...] = ()

    def __init__(
        self, recipe: TeacherRecipe1m, *, atr_exit: AtrTrailingExit | None = None
    ) -> None:
        if recipe.htf_mode != "FEATURE_ONLY":
            raise ValueError("the frozen first-pass teacher requires htf_mode=FEATURE_ONLY")
        self.recipe = recipe
        self._episode_id = ""
        self._last_close_time_ns: int | None = None
        self._previous: dict[str, float] = {}
        self._last_signal = TeacherSignal("NONE", 0, "NOT_EVALUATED")
        self._atr_exit = atr_exit

    @property
    def expert_id(self) -> str:
        return self.recipe.teacher_id

    @property
    def last_signal(self) -> TeacherSignal:
        return self._last_signal

    def reset(self, episode_id: str = "") -> None:
        if not isinstance(episode_id, str):
            raise TypeError("episode id must be a string")
        self._episode_id = episode_id
        self._last_close_time_ns = None
        self._previous = {}
        self._last_signal = TeacherSignal("NONE", 0, "NOT_EVALUATED")
        if self._atr_exit is not None:
            self._atr_exit.reset()

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        if observation.episode_id != self._episode_id:
            self.reset(observation.episode_id)
        current_exposure = observation.account_value("net_exposure")
        try:
            close_time_ns = observation.market_clock_ns("latest_1m_close_ts_ns")
        except KeyError:
            if "latest_1m_close_ts_ns" not in observation.market:
                self._previous = {}
                self._last_signal = TeacherSignal("NONE", 0, "NO_FRESH_CAUSAL_BAR")
                return self._quiet(observation, current_exposure, "NO_FRESH_CAUSAL_BAR")
        if close_time_ns % MINUTE_NS != 0 or close_time_ns > observation.decision_ts_ns:
            raise ValueError("invalid latest completed 1m close clock")
        if observation.decision_ts_ns - close_time_ns >= MINUTE_NS:
            self._previous = {}
            self._last_signal = TeacherSignal("NONE", 0, "NO_FRESH_CAUSAL_BAR")
            return self._quiet(observation, current_exposure, "NO_FRESH_CAUSAL_BAR")
        if self._last_close_time_ns is not None:
            if close_time_ns < self._last_close_time_ns:
                raise ValueError("teacher received an out-of-order completed 1m bar")
            if close_time_ns == self._last_close_time_ns:
                self._last_signal = TeacherSignal("NONE", 0, "NO_NEW_BAR")
                return self._quiet(observation, current_exposure, "NO_NEW_BAR")
            if close_time_ns - self._last_close_time_ns != MINUTE_NS:
                self._last_close_time_ns = close_time_ns
                self._previous = {}
                self._last_signal = TeacherSignal("NONE", 0, "MISSING_BAR")
                return self._quiet(observation, current_exposure, "MISSING_BAR")
        self._last_close_time_ns = close_time_ns

        if self._atr_exit is not None:
            atr_signal = self._atr_exit.evaluate(observation, current_exposure)
            if atr_signal is not None:
                self._last_signal = atr_signal
                return advice(
                    observation,
                    self.expert_id,
                    "EXIT",
                    0.0,
                    "ATR_TRAILING_EXIT",
                    atr_signal.reason_code,
                )

        if any(name not in observation.market for name in self.required_fields):
            self._previous = {}
            self._last_signal = TeacherSignal("NONE", 0, "INDICATOR_NOT_READY")
            return self._quiet(observation, current_exposure, "INDICATOR_NOT_READY")
        current = {name: observation.market_value(name) for name in self.required_fields}
        if not self._previous:
            self._previous = current
            self._last_signal = TeacherSignal("NONE", 0, "INDICATOR_PREVIOUS_NOT_READY")
            return self._quiet(observation, current_exposure, "INDICATOR_PREVIOUS_NOT_READY")

        entry = self.entry_signal(current, self._previous)
        exit_signal = (
            None
            if self._atr_exit is not None
            else self.exit_signal(current_exposure, current, self._previous)
        )
        self._previous = current
        if abs(current_exposure) > 1e-8:
            if exit_signal is not None:
                self._last_signal = exit_signal
                return advice(
                    observation,
                    self.expert_id,
                    "EXIT",
                    0.0,
                    "INDICATOR_EXIT",
                    exit_signal.reason_code,
                )
            self._last_signal = TeacherSignal("NONE", 0, "POSITION_HOLD")
            return advice(
                observation,
                self.expert_id,
                "HOLD",
                current_exposure,
                "POSITION_HOLD",
            )

        self._last_signal = entry
        if entry.side > 0:
            return advice(
                observation,
                self.expert_id,
                "OPEN_LONG",
                self.recipe.target_exposure,
                entry.reason_code,
            )
        if entry.side < 0:
            return advice(
                observation,
                self.expert_id,
                "OPEN_SHORT",
                -self.recipe.target_exposure,
                entry.reason_code,
            )
        return advice(observation, self.expert_id, "FLAT", 0.0, entry.reason_code)

    def _quiet(
        self, observation: PointInTimeObservation, current_exposure: float, reason: str
    ) -> ExpertAdvice:
        if abs(current_exposure) > 1e-8:
            return advice(
                observation, self.expert_id, "HOLD", current_exposure, reason
            )
        return advice(observation, self.expert_id, "FLAT", 0.0, reason)

    @abstractmethod
    def entry_signal(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> TeacherSignal:
        raise NotImplementedError

    @abstractmethod
    def exit_signal(
        self,
        current_exposure: float,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> TeacherSignal | None:
        raise NotImplementedError

