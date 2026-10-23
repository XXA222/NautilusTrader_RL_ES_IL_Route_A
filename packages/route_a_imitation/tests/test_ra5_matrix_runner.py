import pytest

from rles_route_a.evaluation.experiment_matrix import (
    PREREGISTERED_ARMS, build_preregistered_matrix,
)
from rles_route_a.evaluation.matrix_runner import evaluate_frozen_matrix
from rles_route_a.integration.l6_research import HostEvaluationGrant


def frozen_matrix():
    return build_preregistered_matrix(
        recipes={name: f"frozen:{name}" for name in PREREGISTERED_ARMS},
    )


class Grants:
    def __init__(self, *, bind_arm=True, split="validation"):
        self.bind_arm = bind_arm
        self.split = split
        self.reserved = []

    def reserve(self, arm):
        self.reserved.append(arm)
        identity = (arm.name, arm.seed, arm.cost_scenario, arm.recipe_id) if self.bind_arm else None
        index = len(self.reserved)
        return HostEvaluationGrant(
            "fold-1", self.split, "wfo-v1", f"query-{index}", f"registry-receipt-{index}",
            source_identity="registered-snapshot-1", arm_identity=identity,
        )


class Evaluator:
    def __init__(self, *, fail_permission=False, wrong_query=False):
        self.calls = []
        self.fail_permission = fail_permission
        self.wrong_query = wrong_query

    def evaluate_native(self, *, policy, frozen_recipe_id, grant):
        self.calls.append((policy, frozen_recipe_id, grant))
        if self.fail_permission:
            raise PermissionError("L6 refused this evaluation")
        if (not self.wrong_query and policy.name == "random" and policy.seed == 11
                and policy.cost_scenario == "zero"):
            raise RuntimeError("native replay failed")
        if (not self.wrong_query and policy.name == "momentum" and policy.seed == 11
                and policy.cost_scenario == "zero"):
            return {"metrics": {"net_return": 0.01}}
        return {
            "metrics": {
                "net_return": 0.01, "max_drawdown": 0.05, "turnover_ratio": 0.8,
                "total_fee": 1.0, "total_funding": -0.1, "total_slippage_cost": 0.2,
            },
            "terminal_flat": True,
            "costs_already_in_native_equity": True,
            "liquidations_observed": True,
            "liquidations": 0,
            "risk_hard_interventions": 0,
            "execution_decision_diagnostics": {
                "components_known": True, "action_counts": {"NOOP": 1},
                "native_order_status_counts": {}, "submit_decision_count": 0,
                "submitted_decisions_with_fills": 0,
            },
            "query_id": f"wrong:{grant.query_id}" if self.wrong_query else grant.query_id,
        }


def test_matrix_runner_preserves_every_candidate_and_l6_failure():
    arms = frozen_matrix()
    grants = Grants()
    evaluator = Evaluator()

    def make_policy(arm):
        if arm.name == "no_trade" and arm.seed == 11 and arm.cost_scenario == "zero":
            raise ValueError("model artifact missing")
        return arm

    report = evaluate_frozen_matrix(
        arms=arms, policy_factory=make_policy, grant_provider=grants, evaluator=evaluator,
        grant_verifier=lambda grant: grant.source_identity == "registered-snapshot-1",
    )

    assert len(report.arms) == len(arms)
    assert report.failed_count == 3 and report.completed_count == len(arms) - 3
    assert report.status == "CANDIDATES_WITH_FAILURES"
    assert len(grants.reserved) == len(arms) - 1
    assert len(evaluator.calls) == len(arms) - 1  # includes replay and incomplete-report failures
    failed = {(row.arm.name, row.failure_phase, row.failure_type) for row in report.arms
              if row.status == "FAILED"}
    assert failed == {
        ("no_trade", "POLICY_BUILD", "ValueError"),
        ("random", "NATIVE_EVALUATION", "RuntimeError"),
        ("momentum", "REPORT_VALIDATION", "InvalidL6Report"),
    }
    completed = next(row for row in report.arms if row.status == "COMPLETED")
    assert completed.query_id and completed.registry_receipt
    assert (completed.fold_id, completed.split, completed.protocol_id,
            completed.source_identity) == (
                "fold-1", "validation", "wfo-v1", "registered-snapshot-1",
            )
    assert completed.report["query_id"] == completed.query_id


def test_matrix_runner_requires_l6_reservation_bound_to_exact_arm():
    with pytest.raises(PermissionError, match="does not bind"):
        evaluate_frozen_matrix(
            arms=frozen_matrix(), policy_factory=lambda arm: arm,
            grant_provider=Grants(bind_arm=False), evaluator=Evaluator(),
            grant_verifier=lambda grant: True,
        )


def test_matrix_runner_propagates_l6_authorization_denial():
    with pytest.raises(PermissionError, match="refused"):
        evaluate_frozen_matrix(
            arms=frozen_matrix(), policy_factory=lambda arm: arm,
            grant_provider=Grants(), evaluator=Evaluator(fail_permission=True),
            grant_verifier=lambda grant: True,
        )


def test_matrix_runner_rejects_unverified_l6_receipt():
    with pytest.raises(PermissionError, match="did not verify"):
        evaluate_frozen_matrix(
            arms=frozen_matrix(), policy_factory=lambda arm: arm,
            grant_provider=Grants(), evaluator=Evaluator(), grant_verifier=lambda grant: False,
        )


def test_matrix_runner_rejects_native_report_bound_to_another_query():
    report = evaluate_frozen_matrix(
        arms=frozen_matrix(), policy_factory=lambda arm: arm,
        grant_provider=Grants(), evaluator=Evaluator(wrong_query=True),
        grant_verifier=lambda grant: grant.source_identity == "registered-snapshot-1",
    )

    assert report.completed_count == 0
    assert report.failed_count == len(frozen_matrix())
    assert all(row.failure_phase == "REPORT_VALIDATION" for row in report.arms)


def test_full_candidate_matrix_cannot_consume_test_or_holdout():
    for split in ("test", "holdout"):
        evaluator = Evaluator()
        with pytest.raises(PermissionError, match="Validation-only"):
            evaluate_frozen_matrix(
                arms=frozen_matrix(), policy_factory=lambda arm: arm,
                grant_provider=Grants(split=split), evaluator=evaluator,
                grant_verifier=lambda grant: True,
            )
        assert evaluator.calls == []
