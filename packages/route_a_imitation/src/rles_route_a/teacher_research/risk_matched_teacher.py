"""Apply the same position-recovery overlay to native comparison teachers."""
from __future__ import annotations

from typing import Any


class RiskMatchedTeacher:
    """Keep the original teacher's lifecycle while recovery actions take priority."""

    def __init__(self, teacher: Any, recovery: Any):
        if not callable(getattr(teacher, "advise", None)) or not callable(getattr(recovery, "intervention", None)):
            raise TypeError("native teacher and recovery overlay required")
        self.teacher, self.recovery = teacher, recovery
        self.risk_diagnostics = []

    @property
    def expert_id(self):
        return self.teacher.expert_id

    @property
    def recipe(self):
        return self.teacher.recipe

    @property
    def required_fields(self):
        return self.teacher.required_fields

    @property
    def last_signal(self):
        return self.teacher.last_signal

    def reset(self, episode_id: str = ""):
        self.teacher.reset(episode_id)
        self.risk_diagnostics.clear()

    def advise(self, observation):
        override = self.recovery.intervention(observation)
        if override is None:
            return self.teacher.advise(observation)
        # Continue the signal clock when possible, while never preventing the
        # already-computed hard-risk override from reaching native execution.
        try:
            self.teacher.advise(observation)
        except Exception as exc:
            self.risk_diagnostics.append((observation.decision_ts_ns,
                                          f"{type(exc).__name__}: {exc}"))
        return override

    def observe_native_action(self, action, trace, native_order_facts):
        callback = getattr(self.teacher, "observe_native_action", None)
        if callable(callback):
            callback(action, trace, native_order_facts)
