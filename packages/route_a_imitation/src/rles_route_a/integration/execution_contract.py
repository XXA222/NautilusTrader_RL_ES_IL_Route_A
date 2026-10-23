"""Route A requires native planner semantics that can execute explicit exits."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def require_route_a_flat_close(execution: Any, *, context: str = "Route A") -> None:
    """Reject omitted/legacy zero-target behavior at Route A execution boundaries."""
    policy = (execution.get("zero_target_policy") if isinstance(execution, Mapping)
              else getattr(execution, "zero_target_policy", None))
    if policy != "close":
        raise PermissionError(
            f"{context} requires explicit zero_target_policy='close' in the frozen execution recipe"
        )
