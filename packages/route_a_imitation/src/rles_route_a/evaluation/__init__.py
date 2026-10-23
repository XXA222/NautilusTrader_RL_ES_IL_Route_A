"""Route A evaluation gates and evidence reports."""

from .recovery_benchmark import (
    RecoveryGateResult,
    RecoveryGateThresholds,
    compare_recovery_iterations,
)
from .teacher_gate import (
    NativePair,
    StudentPair,
    StudentScreen,
    TeacherScreen,
    screen_student,
    screen_teacher,
    verify_student_screen,
)
from .promotion_identity import (
    bc_action_schema_sha256,
    bc_actor_sha256,
    bc_observation_schema_sha256,
    bc_student_recipe_sha256,
)

__all__ = [
    "NativePair",
    "RecoveryGateResult",
    "RecoveryGateThresholds",
    "StudentPair",
    "StudentScreen",
    "TeacherScreen",
    "bc_action_schema_sha256",
    "bc_actor_sha256",
    "bc_observation_schema_sha256",
    "bc_student_recipe_sha256",
    "compare_recovery_iterations",
    "screen_student",
    "screen_teacher",
    "verify_student_screen",
]
