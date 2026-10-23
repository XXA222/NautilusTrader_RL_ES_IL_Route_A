"""Run every frozen RA5 arm through separately reserved host L6 queries."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from rles_route_a.integration.l6_research import (
    AuthorizedFormalEvaluator,
    HostEvaluationGrant,
    evaluate_registered,
)
from .experiment_matrix import (
    ExperimentArm,
    PREREGISTERED_ARMS,
    PREREGISTERED_SEEDS,
)


class L6GrantProvider(Protocol):
    def reserve(self, arm: ExperimentArm) -> HostEvaluationGrant: ...


@dataclass(frozen=True)
class MatrixArmResult:
    arm: ExperimentArm
    status: Literal["COMPLETED", "FAILED"]
    query_id: str | None
    registry_receipt: str | None
    report: Mapping[str, Any] | None = None
    failure_type: str | None = None
    failure_phase: Literal["POLICY_BUILD", "NATIVE_EVALUATION", "REPORT_VALIDATION"] | None = None
    fold_id: str | None = None
    split: str | None = None
    protocol_id: str | None = None
    source_identity: str | None = None

    def __post_init__(self) -> None:
        provenance = (self.fold_id, self.split, self.protocol_id, self.source_identity)
        if self.status == "COMPLETED":
            if (not self.query_id or not self.registry_receipt or self.report is None
                    or self.failure_type is not None or self.failure_phase is not None
                    or not all(isinstance(value, str) and value for value in provenance)):
                raise ValueError("completed matrix arm must retain its L6 receipt and report")
        elif self.status == "FAILED":
            if (not self.failure_type
                    or self.failure_phase not in {"POLICY_BUILD", "NATIVE_EVALUATION", "REPORT_VALIDATION"}
                    or self.report is not None):
                raise ValueError("failed matrix arm must retain its failure phase and type")
            if self.failure_phase == "POLICY_BUILD":
                if self.query_id is not None or self.registry_receipt is not None or any(provenance):
                    raise ValueError("policy build failure cannot claim an L6 evaluation receipt")
            elif (not self.query_id or not self.registry_receipt
                  or not all(isinstance(value, str) and value for value in provenance)):
                raise ValueError("post-reservation failure must retain its L6 provenance")
        else:
            raise ValueError("unknown matrix arm status")


@dataclass(frozen=True)
class MatrixEvaluationReport:
    arms: tuple[MatrixArmResult, ...]
    status: Literal["ALL_ARMS_EVALUATED", "CANDIDATES_WITH_FAILURES"]

    def __post_init__(self) -> None:
        expected_identities = {(name, seed, scenario)
                               for name in PREREGISTERED_ARMS for seed in PREREGISTERED_SEEDS
                               for scenario in ("zero", "base", "stress")}
        identities = {(r.arm.name, r.arm.seed, r.arm.cost_scenario) for r in self.arms}
        if not self.arms or len(identities) != len(self.arms) or identities != expected_identities:
            raise ValueError("matrix report must preserve every unique preregistered arm")
        reserved = [row for row in self.arms if row.query_id is not None]
        if len({row.query_id for row in reserved}) != len(reserved):
            raise ValueError("matrix report contains reused L6 query identities")
        scopes = {(row.fold_id, row.split, row.protocol_id, row.source_identity)
                  for row in reserved}
        if len(scopes) > 1:
            raise ValueError("matrix report does not preserve paired L6 scope")
        failed = any(row.status == "FAILED" for row in self.arms)
        expected_status = "CANDIDATES_WITH_FAILURES" if failed else "ALL_ARMS_EVALUATED"
        if self.status != expected_status:
            raise ValueError("matrix report status disagrees with preserved arm outcomes")

    @property
    def completed_count(self) -> int:
        return sum(row.status == "COMPLETED" for row in self.arms)

    @property
    def failed_count(self) -> int:
        return sum(row.status == "FAILED" for row in self.arms)


def _validate_formal_matrix(arms: Sequence[ExperimentArm]) -> tuple[ExperimentArm, ...]:
    rows = tuple(arms)
    expected = {(name, seed, scenario)
                for name in PREREGISTERED_ARMS for seed in PREREGISTERED_SEEDS
                for scenario in ("zero", "base", "stress")}
    actual = {(row.name, row.seed, row.cost_scenario) for row in rows}
    if len(actual) != len(rows) or actual != expected:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise ValueError(f"formal evaluation requires the complete frozen RA5 matrix missing={missing} extra={extra}")
    return rows


def _native_report_valid(report: Mapping[str, Any], *, query_id: str) -> bool:
    """Require the host's native economic, execution and risk facts per arm."""
    if not isinstance(query_id, str) or not query_id or report.get("query_id") != query_id:
        return False
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    required_metrics = (
        "net_return", "max_drawdown", "turnover_ratio", "total_fee",
        "total_funding", "total_slippage_cost",
    )
    if any(name not in metrics or isinstance(metrics[name], bool)
           or not isinstance(metrics[name], (int, float)) or not isfinite(metrics[name])
           for name in required_metrics):
        return False
    if (not 0 <= metrics["max_drawdown"] <= 1 or metrics["turnover_ratio"] < 0
            or metrics["total_fee"] < 0 or metrics["total_slippage_cost"] < 0):
        return False
    if (report.get("terminal_flat") is not True
            or report.get("costs_already_in_native_equity") is not True
            or report.get("liquidations_observed") is not True
            or type(report.get("liquidations")) is not int or report["liquidations"] < 0
            or type(report.get("risk_hard_interventions")) is not int
            or report["risk_hard_interventions"] < 0):
        return False
    execution = report.get("execution_decision_diagnostics")
    if not isinstance(execution, Mapping) or execution.get("components_known") is not True:
        return False
    if (not isinstance(execution.get("action_counts"), Mapping)
            or not isinstance(execution.get("native_order_status_counts"), Mapping)):
        return False
    submitted = execution.get("submit_decision_count")
    filled = execution.get("submitted_decisions_with_fills")
    return (type(submitted) is int and submitted >= 0
            and type(filled) is int and 0 <= filled <= submitted)


def evaluate_frozen_matrix(*, arms: Sequence[ExperimentArm],
                           policy_factory: Callable[[ExperimentArm], object],
                           grant_provider: L6GrantProvider,
                           evaluator: AuthorizedFormalEvaluator,
                           grant_verifier: Callable[[HostEvaluationGrant], bool]
                           ) -> MatrixEvaluationReport:
    """Evaluate all pre-registered arms; retain execution failures in place.

    Policy construction failures and evaluator errors other than authorization
    denials become explicit failed rows so a partial report cannot silently
    select only successful candidates. L6 denials propagate immediately.
    """
    rows = _validate_formal_matrix(arms)
    if (not callable(policy_factory) or not callable(getattr(grant_provider, "reserve", None))
            or not callable(grant_verifier)):
        raise TypeError("policy factory, L6 grant provider, and authority verifier are required")
    outcomes: list[MatrixArmResult] = []
    query_ids: set[str] = set()
    evaluation_scope: tuple[str, str, str, str] | None = None
    for arm in rows:
        try:
            policy = policy_factory(arm)
        except Exception as exc:
            outcomes.append(MatrixArmResult(arm, "FAILED", None, None,
                                            failure_type=type(exc).__name__,
                                            failure_phase="POLICY_BUILD"))
            continue

        grant = grant_provider.reserve(arm)
        if not isinstance(grant, HostEvaluationGrant):
            raise PermissionError("formal evaluation requires a host L6 grant")
        if grant.split != "validation":
            raise PermissionError(
                "the complete RA5 candidate matrix is Validation-only; Test/Holdout "
                "must follow host WFO frozen-selection gates"
            )
        expected_identity = (arm.name, arm.seed, arm.cost_scenario, arm.recipe_id)
        if grant.arm_identity != expected_identity:
            raise PermissionError("L6 reservation does not bind this exact frozen RA5 arm")
        if not grant.source_identity.strip() or grant_verifier(grant) is not True:
            raise PermissionError("L6 authority did not verify the reserved query/source identity")
        scope = (grant.fold_id, grant.split, grant.protocol_id, grant.source_identity)
        if evaluation_scope is None:
            evaluation_scope = scope
        elif scope != evaluation_scope:
            raise PermissionError("paired RA5 arms must share L6 Fold, split, and protocol")
        if grant.query_id in query_ids:
            raise PermissionError("each RA5 arm requires its own reserved L6 query")
        query_ids.add(grant.query_id)
        try:
            result = evaluate_registered(
                evaluator, policy=policy, recipe_id=arm.recipe_id, grant=grant,
            )
        except PermissionError:
            raise
        except Exception as exc:
            outcomes.append(MatrixArmResult(
                arm, "FAILED", grant.query_id, grant.project_registry_receipt,
                failure_type=type(exc).__name__, failure_phase="NATIVE_EVALUATION",
                fold_id=grant.fold_id, split=grant.split, protocol_id=grant.protocol_id,
                source_identity=grant.source_identity,
            ))
            continue
        if (not isinstance(result, Mapping) or not result
                or not _native_report_valid(result, query_id=grant.query_id)):
            outcomes.append(MatrixArmResult(
                arm, "FAILED", grant.query_id, grant.project_registry_receipt,
                failure_type="InvalidL6Report", failure_phase="REPORT_VALIDATION",
                fold_id=grant.fold_id, split=grant.split, protocol_id=grant.protocol_id,
                source_identity=grant.source_identity,
            ))
            continue
        outcomes.append(MatrixArmResult(
            arm, "COMPLETED", grant.query_id, grant.project_registry_receipt,
            report=dict(result), fold_id=grant.fold_id, split=grant.split,
            protocol_id=grant.protocol_id, source_identity=grant.source_identity,
        ))

    outcome_rows = tuple(outcomes)
    status: Literal["ALL_ARMS_EVALUATED", "CANDIDATES_WITH_FAILURES"] = (
        "CANDIDATES_WITH_FAILURES" if any(x.status == "FAILED" for x in outcome_rows)
        else "ALL_ARMS_EVALUATED"
    )
    return MatrixEvaluationReport(outcome_rows, status)
