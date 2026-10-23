"""Re-use host L6 metrics, report writer and research ledger via dependency inversion."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


class NativeEconomicEvaluator(Protocol):
    """Injected from existing rles_research; not implemented in this package."""
    def from_native_facts(self, native_facts: Sequence[object], *,
                          frozen_recipe_id: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class HostEvaluationGrant:
    """Supplied by L6 AFTER query reservation; Route A cannot mint access rights."""
    fold_id: str
    split: str
    protocol_id: str
    query_id: str
    project_registry_receipt: str
    source_identity: str = ""
    # Matrix reservations bind each query to one frozen arm. Optional for the
    # single-policy helper below; mandatory in matrix_runner.
    arm_identity: tuple[str, int, str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.split, str) or self.split not in {"validation", "test", "holdout"}:
            raise PermissionError("Only L6 owns the formal evaluation boundary")
        if not all(isinstance(value, str) and value.strip() for value in
                   (self.fold_id, self.protocol_id, self.query_id,
                    self.project_registry_receipt)):
            raise PermissionError("host-reserved research grant required")
        if not isinstance(self.source_identity, str):
            raise PermissionError("invalid L6 source identity")
        if self.arm_identity is not None:
            if (not isinstance(self.arm_identity, tuple) or len(self.arm_identity) != 4
                    or not isinstance(self.arm_identity[0], str) or not self.arm_identity[0].strip()
                    or type(self.arm_identity[1]) is not int or self.arm_identity[1] < 0
                    or not isinstance(self.arm_identity[2], str)
                    or self.arm_identity[2] not in {"zero", "base", "stress"}
                    or not isinstance(self.arm_identity[3], str) or not self.arm_identity[3].strip()):
                raise PermissionError("invalid L6 frozen experiment arm identity")


class AuthorizedFormalEvaluator(Protocol):
    def evaluate_native(self, *, policy: object, frozen_recipe_id: str,
                        grant: HostEvaluationGrant) -> Mapping[str, Any]: ...


def evaluate_registered(evaluator: AuthorizedFormalEvaluator, *, policy: object,
                        recipe_id: str, grant: HostEvaluationGrant) -> Mapping[str, Any]:
    if not isinstance(grant, HostEvaluationGrant) or not recipe_id:
        raise PermissionError("Only L6-authorized evaluation may query formal data")
    return evaluator.evaluate_native(policy=policy, frozen_recipe_id=recipe_id, grant=grant)
