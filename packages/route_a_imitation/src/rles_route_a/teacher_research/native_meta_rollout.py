"""Run a frozen Meta teacher through the existing verified native Gym provider."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rles_route_a.integration.gym_rollout import NativeGymTeacherProvider
from rles_route_a.integration.scope import HostTrainGrant
from .meta_gated_teacher import MetaGatedTeacher, write_meta_trace


def run_registered_meta_rollout(
    *, provider: NativeGymTeacherProvider, teacher: MetaGatedTeacher,
    factory: Any, host_rollout: Callable[..., Any], grant: HostTrainGrant,
    verify_grant: Callable[[HostTrainGrant], bool], scenario: str, seed: int,
    rollout_id: str, episode_steps: int, trace_path: Path,
    reset_options: Mapping[str, Any] | None = None,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    """Use host M2.2 and L6 verification; never manufacture native receipts."""
    if not isinstance(provider, NativeGymTeacherProvider) or provider.teacher is not teacher:
        raise TypeError("native provider must own the exact Meta teacher")
    if (not isinstance(grant, HostTrainGrant) or not callable(verify_grant)
            or verify_grant(grant) is not True):
        raise PermissionError("host-verified Train grant required")
    if (provider.fold_id, provider.source_identity) != (grant.fold_id, grant.source_identity):
        raise PermissionError("native provider and registered Train grant differ")
    policy = teacher.policy
    if (policy.source_id, policy.fold_id) != (grant.source_identity, grant.fold_id):
        raise PermissionError("trained model and native Train source differ")
    if (provider.execution_recipe, scenario) != (policy.label_recipe[1], policy.label_recipe[2]):
        raise ValueError("native execution or cost scenario differs from training labels")
    if (not callable(host_rollout) or type(seed) is not int
            or not isinstance(rollout_id, str) or not rollout_id.strip()
            or type(episode_steps) is not int or episode_steps <= 0):
        raise ValueError("valid host rollout, run identity, and episode size required")
    kwargs = {"episode_steps": episode_steps}
    if reset_options is not None:
        kwargs["reset_options"] = dict(reset_options)
    barriers, native = provider.rollout_with_m22_trace(
        factory=factory, host_rollout=host_rollout, grant=grant, seed=seed,
        decision_policy=None, rollout_id=rollout_id, **kwargs)
    verification = native.get("route_a_native_order_verification")
    if (len(barriers) != episode_steps or not isinstance(verification, Mapping)
            or verification.get("barriers") != len(barriers)
            or verification.get("authority") != "retained_m22_session_snapshots"):
        raise ValueError("native M2.2 verification does not cover the complete episode")
    if len(teacher.decisions) != len(barriers) or len(teacher.native_receipts) != len(barriers):
        raise ValueError("candidate/native trace does not cover every native barrier")
    write_meta_trace(teacher, trace_path)
    return barriers, native
