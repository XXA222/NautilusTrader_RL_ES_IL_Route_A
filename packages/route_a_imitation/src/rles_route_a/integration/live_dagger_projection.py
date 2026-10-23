"""Bind DAgger projection to a live native student decision barrier."""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING, Any

from rles_route_a.contracts import PointInTimeObservation

if TYPE_CHECKING:
    from rles_route_a.training.dagger_collector import ProjectedTeacherTarget


class LiveNautilusGymProjection:
    """Project teacher commands through the paused student's native L3 stack.

    Construct this adapter with the exact environment currently producing a
    student visit. The DAgger provider must yield that visit before advancing
    the environment, allowing collection to label and project it synchronously.
    """

    def __init__(self, env: Any, *, execution_recipe: str,
                 projection_recipe: str | None = None) -> None:
        if not execution_recipe or not isinstance(execution_recipe, str):
            raise ValueError("frozen native transition hash required")
        active_projection_recipe = execution_recipe if projection_recipe is None else projection_recipe
        if (not isinstance(active_projection_recipe, str) or len(active_projection_recipe) != 64
                or any(character not in "0123456789abcdef" for character in active_projection_recipe)):
            raise ValueError("window-independent frozen projection recipe required")
        if not callable(getattr(env, "rles_project_target", None)):
            raise TypeError("host Gym environment lacks native read-only target projection")
        self._env = env
        self._transition = execution_recipe
        self._recipe = active_projection_recipe

    def project_native_target(self, observation: PointInTimeObservation,
                              requested_exposure: float) -> ProjectedTeacherTarget:
        if not isinstance(observation, PointInTimeObservation):
            raise TypeError("native point-in-time student observation required")
        if not isfinite(requested_exposure) or not -1 <= requested_exposure <= 1:
            raise ValueError("finite normalized teacher command required")
        projected = self._env.rles_project_target(requested_exposure)
        if not isinstance(projected, Mapping):
            raise TypeError("host returned no native projection receipt")
        if (projected.get("decision_ts_ns") != observation.decision_ts_ns
                or projected.get("execution_recipe") != self._transition
                or projected.get("requested_exposure") != requested_exposure):
            raise ValueError("projection receipt does not bind the current student barrier")
        target = projected.get("target")
        disposition = projected.get("disposition")
        if (isinstance(target, bool) or not isinstance(target, (int, float))
                or not isfinite(target)):
            raise ValueError("host projection omitted a finite target")
        label_disposition = "PROJECTED" if disposition == "SUBMIT" else disposition
        from rles_route_a.training.dagger_collector import ProjectedTeacherTarget
        return ProjectedTeacherTarget(float(target), self._recipe, label_disposition)
