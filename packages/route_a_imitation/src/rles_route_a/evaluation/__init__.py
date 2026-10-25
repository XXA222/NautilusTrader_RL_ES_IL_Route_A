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
from .teacher_research_gate_v2 import (
    CandidateBlockResult,
    ScenarioCandidateCoverage,
    TeacherResearchGateV2Result,
    evaluate_teacher_research_gate_v2,
)

__all__ = [
    "NativePair",
    "CandidateBlockResult",
    "RecoveryGateResult",
    "RecoveryGateThresholds",
    "ScenarioCandidateCoverage",
    "StudentPair",
    "StudentScreen",
    "TeacherScreen",
    "TeacherResearchGateV2Result",
    "bc_action_schema_sha256",
    "bc_actor_sha256",
    "bc_observation_schema_sha256",
    "bc_student_recipe_sha256",
    "compare_recovery_iterations",
    "evaluate_teacher_research_gate_v2",
    "screen_student",
    "screen_teacher",
    "verify_student_screen",
]
