import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import rles_route_a.training.native_bc as native_bc
from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from rles_route_a.datasets.sequence_dataset import LabeledStep
from rles_route_a.config import BCConfig
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.models.bc import BCGRUPolicy
from rles_route_a.integration.trace_audit import NativeTraceSummary
from rles_route_a.training.bc_trainer import BCEpoch
from rles_route_a.training.native_bc import (
    NativeBCBehaviorSummary,
    NativeBCIndependentEvaluation,
    NativeBCTrainingResult,
    _native_barrier_evidence,
    evaluate_bc_on_registered_train_inner_rollout,
    teacher_config_hash,
)
from rles_route_a.integration.gym_rollout import NativeGymTeacherProvider
from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.integration.wfo_train import make_route_a_wfo_train
from rles_route_a.evaluation import (
    NativePair,
    StudentPair,
    bc_action_schema_sha256,
    bc_actor_sha256,
    bc_observation_schema_sha256,
    bc_student_recipe_sha256,
    screen_student,
    screen_teacher,
)


def canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def verified_write(path: Path, value) -> str:
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    path.write_bytes(raw)
    identity = hashlib.sha256(raw).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(identity, encoding="ascii")
    return identity


def barrier_evidence(count: int, prefix: str):
    return tuple(
        {
            "decision_id": f"{prefix}-{index}",
            "decision_ts_ns": index + 1,
            "requested_exposure": 0.5,
            "risk_adjusted_target": 0.4,
            "planner_target": 0.4,
            "actual_exposure_after": 0.4,
            "native_equity_before": 1000.0,
            "native_equity_after": 1001.0,
            "reward_econ": 0.001,
            "planner_action": "SUBMIT",
            "planner_reason": "scenario",
            "fills": [{"fee": "0 USD"}],
            "fee_currency": "USD",
            "scenario_only": True,
        }
        for index in range(count)
    )


class ValidationAccess:
    def __init__(self):
        self.identity = {
            "query_id": "validation-0-candidate-a",
            "candidate_id": "candidate-a",
            "fold_id": "60-15-15-0000",
            "split": "validation",
        }
        self.bundle_sha256 = None

    @property
    def consumed(self):
        return self.bundle_sha256 is not None

    def consume(self, *, bundle_sha256: str):
        self.bundle_sha256 = bundle_sha256


def training_result() -> NativeBCTrainingResult:
    config = BCConfig(feature_names=("market.trend_1h",), hidden_size=8, layers=1, seed=31)
    model = BCGRUPolicy(config.feature_names, config.hidden_size, config.layers)
    policy = BCInferencePolicy(model, config)
    labeled_steps = tuple(
        LabeledStep(
            observation=PointInTimeObservation(
                episode_id="fit-episode",
                fold_id="fold",
                decision_ts_ns=201 + index,
                market={"trend_1h": 0.01},
                feature_available_at_ns={"trend_1h": 200 + index},
                account={"net_exposure": 0.5},
            ),
            advice=ExpertAdvice("teacher", 201 + index, "OPEN_LONG", 0.5, ("native",)),
            executable_exposure=0.5,
            source_identity="source",
            trajectory_id="fit-run",
            economic_case_status="OBSERVED_NATIVE_TRADE_POSITIVE",
            case_outcome_bps=5.0,
            case_end_ts_ns=220,
            case_fact_sha256="e" * 64,
            supervision_weight=1.0 if index < 9 else 0.0,
            intent_supervision_weight=1.0 if index < 9 else 0.0,
        )
        for index in range(10)
    )
    summary = NativeTraceSummary(
        "fold", 10, 2, 2, 1, 9, 1000.0, 1010.0, "SCENARIO_ONLY",
        completed_native_trade_case_count=1,
        positive_native_trade_case_count=1,
    )
    result = NativeBCTrainingResult(
        policy,
        config,
        (BCEpoch(1, 0.5, 0.2, 0.3, None),),
        10,
        9,
        labeled_steps,
        summary,
        {},
        NativeBCBehaviorSummary(
            sample_count=10,
            executable_label_count=9,
            label_nonzero_fraction=0.8,
            prediction_nonzero_fraction=0.7,
            prediction_mean=0.1,
            prediction_std=0.3,
            prediction_min=-0.5,
            prediction_max=0.6,
            mae=0.05,
            mse=0.01,
            deadband=0.05,
            collapsed_to_flat=False,
        ),
        native_barrier_evidence=barrier_evidence(10, "fit-teacher"),
        independent_evaluation=NativeBCIndependentEvaluation(
            behavior_summary=NativeBCBehaviorSummary(
                sample_count=8,
                executable_label_count=8,
                label_nonzero_fraction=0.75,
                prediction_nonzero_fraction=0.65,
                prediction_mean=0.08,
                prediction_std=0.25,
                prediction_min=-0.4,
                prediction_max=0.5,
                mae=0.06,
                mse=0.02,
                deadband=0.05,
                collapsed_to_flat=False,
            ),
            student_behavior_summary=NativeBCBehaviorSummary(
                sample_count=8,
                executable_label_count=8,
                label_nonzero_fraction=0.75,
                prediction_nonzero_fraction=0.65,
                prediction_mean=0.08,
                prediction_std=0.25,
                prediction_min=-0.4,
                prediction_max=0.5,
                mae=0.06,
                mse=0.02,
                deadband=0.05,
                collapsed_to_flat=False,
            ),
            student_native_summary=replace(summary, decisions=8),
            student_rollout_report={"execution_mode": "base"},
            teacher_barrier_evidence=barrier_evidence(8, "selection-teacher"),
            student_barrier_evidence=barrier_evidence(8, "selection-student"),
            native_summary=replace(summary, decisions=8),
            window_ns=(300, 500),
            rollout_report={},
            fold_id="fold",
            source_identity="source",
            transition_hash="selection-transition",
            feature_hash="feature",
            teacher_hash="teacher",
            purge_ns=0,
            execution_recipe_hash="a" * 64,
        ),
        source_identity="source",
        transition_hash="transition",
        feature_hash="feature",
        execution_recipe_hash="a" * 64,
        teacher_hash="teacher",
        train_window_ns=(200, 300),
        training_seed=31,
    )
    return replace(result, student_screen=promotion_screen(result))


def promotion_screen(result: NativeBCTrainingResult):
    teacher_pairs = tuple(
        NativePair(
            window_id=window,
            scenario=scenario,
            seed=seed,
            teacher_recipe=result.teacher_hash,
            baseline_recipe="flat-baseline",
            source_identity=result.source_identity,
            teacher_net_return=0.02,
            baseline_net_return=0.005,
            teacher_max_drawdown=0.08,
            baseline_max_drawdown=0.10,
            teacher_filled_orders=2,
            teacher_liquidations=0,
            matched_native_barriers=True,
            scenario_only=True,
            nonoverlap_verified=True,
        )
        for window in ("window-1", "window-2")
        for scenario in ("base", "stress")
        for seed in (11, 29)
    )
    teacher = screen_teacher(teacher_pairs)
    actor_sha = bc_actor_sha256(result.policy.model)
    observation_sha = bc_observation_schema_sha256(result.config)
    action_sha = bc_action_schema_sha256()
    student_pairs = tuple(
        StudentPair(
            window_id=window,
            scenario=scenario,
            seed=seed,
            teacher_recipe=result.teacher_hash,
            student_recipe=bc_student_recipe_sha256(result.config),
            source_identity=result.source_identity,
            teacher_net_return=0.02,
            student_net_return=0.03,
            teacher_max_drawdown=0.08,
            student_max_drawdown=0.09,
            student_filled_orders=2,
            student_liquidations=0,
            matched_native_barriers=True,
            teacher_gate_sha256=teacher.receipt_sha256,
            actor_sha256=actor_sha,
            observation_schema_sha256=observation_sha,
            action_schema_sha256=action_sha,
            execution_recipe_sha256=result.execution_recipe_hash,
            teacher_evidence_sha256="e" * 64,
            student_evidence_sha256="f" * 64,
            scenario_only=True,
            nonoverlap_verified=True,
        )
        for window in ("window-1", "window-2")
        for scenario in ("base", "stress")
        for seed in (11, 29)
    )
    return screen_student(student_pairs, teacher_screen=teacher)


def test_native_barrier_evidence_projects_execution_fees_and_equity_without_market_rows():
    native = SimpleNamespace(
        execution_decision_id="decision-1",
        decision_ts_ns=101,
        execution_barrier_ts_ns=101,
        requested_action=0.5,
        soft_risk_target=0.45,
        risk_adjusted_target=0.4,
        planner_target=0.4,
        actual_exposure_after=0.3,
        planner_action="SUBMIT",
        planner_reason="risk-sized",
        submitted_order_ids=("order-1",),
        filled_order_ids=("order-1",),
        related_existing_order_ids=(),
        pending_order_ids_after=(),
        fee_currency="USD",
        scenario_only=True,
    )
    barrier = SimpleNamespace(
        trace=native,
        reward=SimpleNamespace(
            prior_equity=1000.0,
            next_equity=1001.0,
            recorded_r_econ=0.001,
            has_external_cashflow=False,
        ),
        trajectory_id="trajectory-1",
        advice=SimpleNamespace(expert_id="trend_v1", intent="OPEN_LONG"),
    )
    fills = [
        {"event_id": "fill-1", "quantity": "1", "price": "10", "fee": "0.1 USD", "ts_event": 102}
    ]
    projected = _native_barrier_evidence(
        (barrier,),
        {
            "execution_decision_trace": [
                {
                    "decision_id": "decision-1",
                    "decision_ts_ns": 101,
                    "actor_requested_target": 0.5,
                    "action": "SUBMIT",
                    "hard_risk_adjustment": 0.1,
                    "fills": fills,
                    "market_features": {"close": 10},
                    "account_rows": [{"equity": 1000}],
                }
            ],
        },
    )[0]

    assert projected["requested_to_soft_delta"] == pytest.approx(-0.05)
    assert projected["soft_to_risk_delta"] == pytest.approx(-0.05)
    assert projected["native_equity_before"] == 1000.0
    assert projected["native_equity_after"] == 1001.0
    assert projected["reward_econ"] == 0.001
    assert projected["fills"][0]["fee"] == "0.1 USD"
    assert projected["risk_adjustment"]["hard_risk_adjustment"] == 0.1
    assert "market_features" not in projected and "account_rows" not in projected


def test_route_a_wfo_train_writes_host_envelope_and_validation_receipts(tmp_path):
    result = training_result()
    access = ValidationAccess()
    train_factory = SimpleNamespace(
        metadata=lambda: {
            "scaler": {
                "mean": [0.0],
                "std": [1.0],
                "feature_hash": "f" * 64,
                "train_window_hash": "t" * 64,
                "sample_count": 10,
            },
            "identities": {},
        },
        execution=SimpleNamespace(zero_target_policy="close"),
        start_ns=200,
        end_ns=300,
        transition_hash="transition",
        route_a_execution_recipe_hash="a" * 64,
    )

    def fit(run):
        assert run["environment"]["end_ns"] == 600
        return result, train_factory

    run = {
        "output": str(tmp_path / "candidate-a"),
        "route": "route_a",
        "algorithm": "route_a_bc_gru",
        "seed": 31,
        "environment": {
            "instrument": "ETHUSDT-PERP",
            "start_ns": 100,
            "end_ns": 600,
            "execution": {"zero_target_policy": "close"},
        },
        "evaluation": {"split": "validation", "start_ns": 600, "end_ns": 650},
        "wfo_candidate_id": "candidate-a",
        "wfo_fold_id": "60-15-15-0000",
    }
    access = ValidationAccess()

    def validation_with_access(run_value, output, *, validation_access):
        assert validation_access is access
        bundle = json.loads((output / "bundle.json").read_text())
        assert bundle["actor"]["manifest_sha256"]
        bundle_sha = hashlib.sha256((output / "bundle.json").read_bytes()).hexdigest()
        validation_access.consume(bundle_sha256=bundle_sha)
        return {
            "execution_mode": "base",
            "evaluation_scope": {
                "split": "validation",
                "mode": "base",
                "query_id": validation_access.identity["query_id"],
                "fold_id": validation_access.identity["fold_id"],
            },
            "metrics": {"net_return": 0.01},
        }

    train = make_route_a_wfo_train(
        fit_candidate=fit,
        validation_backtest=validation_with_access,
        write_verified=verified_write,
        digest=canonical_sha,
    )
    output = Path(run["output"])
    trained = train(run, validation_access=access)
    wrapper = json.loads((output / "bundle.json").read_text())
    completion = json.loads((output / "COMPLETED.json").read_text())
    assert trained["output"] == str(output)
    assert wrapper["schema"] == "rles.historical-bundle.v1"
    assert wrapper["actor"]["kind"] == "route_a_bc"
    assert wrapper["lineage"][0]["usage"]["executable_labels"] == 9
    promotion = wrapper["lineage"][0]["usage"]["student_promotion"]
    promotion_path = output / promotion["artifact_path"]
    assert promotion["status"] == "TRAIN_INNER_SCREEN_PASS"
    assert promotion["economic_status"] == "SCENARIO_ONLY"
    assert promotion["actor_state_sha256"] == result.student_screen.actor_sha256
    assert hashlib.sha256(promotion_path.read_bytes()).hexdigest() == promotion["artifact_sha256"]
    promotion_evidence = json.loads(promotion_path.read_text())
    assert len(promotion_evidence["teacher_pairs"]) == 8
    assert len(promotion_evidence["student_pairs"]) == 8
    assert promotion_evidence["screen"]["receipt_sha256"] == result.student_screen.receipt_sha256
    evidence = wrapper["lineage"][0]["usage"]["native_barrier_evidence"]
    assert "student_native_report" not in wrapper["lineage"][0]["usage"]
    assert set(evidence) == {"fit_teacher", "selection_teacher", "selection_bc_student"}
    for artifact in evidence.values():
        path = output / artifact["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        payload = json.loads(path.read_text())
        assert len(payload["decisions"]) == artifact["decision_count"]
        if payload["stage"] == "fit_teacher":
            assert payload["summary"]["completed_native_trade_case_count"] == 1
            assert payload["summary"]["positive_native_trade_case_count"] == 1
            assert payload["summary"]["negative_entry_avoidance_label_count"] == 0
        assert payload["scope"]["transition_hash"] == (
            "transition" if payload["stage"] == "fit_teacher" else "selection-transition"
        )
        assert "account_rows" not in payload and "market_features" not in payload
    assert access.bundle_sha256 == completion["bundle_sha256"]
    assert (
        completion["report_sha256"]
        == hashlib.sha256((output / "report.json").read_bytes()).hexdigest()
    )
    assert (output / "route_a_bc" / "weights.npz").is_file()


@pytest.mark.parametrize(
    "failure", (
        "missing", "rejected", "wrong_actor", "no_teacher_cases",
        "inconsistent_case_summary", "foreign_training_label",
        "training_case_outside_fit_window",
    )
)
def test_route_a_wfo_requires_student_promotion_bound_to_frozen_actor(tmp_path, failure):
    result = training_result()
    if failure == "missing":
        result = replace(result, student_screen=None)
    elif failure == "no_teacher_cases":
        result = replace(
            result,
            label_count=0,
            executable_label_count=0,
            labeled_steps=(),
            native_summary=replace(
                result.native_summary,
                completed_native_trade_case_count=0,
                positive_native_trade_case_count=0,
            ),
        )
    elif failure == "inconsistent_case_summary":
        result = replace(
            result,
            native_summary=replace(
                result.native_summary,
                negative_entry_avoidance_label_count=1,
            ),
        )
    elif failure == "foreign_training_label":
        result = replace(
            result,
            labeled_steps=(
                replace(result.labeled_steps[0], source_identity="other-source"),
                *result.labeled_steps[1:],
            ),
        )
    elif failure == "training_case_outside_fit_window":
        result = replace(
            result,
            labeled_steps=tuple(
                replace(label, case_end_ts_ns=301) for label in result.labeled_steps
            ),
        )
    elif failure == "rejected":
        result = replace(
            result,
            student_screen=replace(
                result.student_screen,
                status="TRAIN_INNER_SCREEN_REJECT",
            ),
        )
    else:
        result = replace(
            result,
            student_screen=replace(
                result.student_screen,
                actor_sha256="0" * 64,
            ),
        )
    factory = SimpleNamespace(
        metadata=lambda: {
            "scaler": {
                "mean": [0.0],
                "std": [1.0],
                "feature_hash": "f" * 64,
                "train_window_hash": "t" * 64,
                "sample_count": 10,
            },
        },
        execution=SimpleNamespace(zero_target_policy="close"),
        start_ns=200,
        end_ns=300,
        transition_hash="transition",
        route_a_execution_recipe_hash="a" * 64,
    )
    output = tmp_path / failure
    train = make_route_a_wfo_train(
        fit_candidate=lambda _run: (result, factory),
        validation_backtest=lambda *_args, **_kwargs: pytest.fail(
            "Validation must not consume a missing or rejected student gate"
        ),
        write_verified=verified_write,
        digest=canonical_sha,
    )
    with pytest.raises(
        PermissionError,
        match=(
            "student-beats-teacher receipt|promotion receipt|positive native teacher case|"
            "case summary does not match|labels escape fitted Train"
        ),
    ):
        train(
            {
                "output": str(output),
                "environment": {
                    "start_ns": 100,
                    "end_ns": 600,
                    "execution": {"zero_target_policy": "close"},
                },
                "route": "route_a",
                "algorithm": "route_a_bc_gru",
            },
            evaluate=False,
        )
    assert not (output / "route_a_bc").exists()
    assert not (output / "bundle.json").exists()


def test_route_a_wfo_train_rejects_report_for_another_validation_query(tmp_path):
    result = training_result()
    access = ValidationAccess()
    factory = SimpleNamespace(
        metadata=lambda: {
            "scaler": {
                "mean": [0.0],
                "std": [1.0],
                "feature_hash": "f" * 64,
                "train_window_hash": "t" * 64,
                "sample_count": 10,
            },
        },
        execution=SimpleNamespace(zero_target_policy="close"),
        start_ns=200,
        end_ns=300,
        transition_hash="transition",
        route_a_execution_recipe_hash="a" * 64,
    )

    def wrong_query(_run, output, *, validation_access):
        validation_access.consume(
            bundle_sha256=hashlib.sha256((output / "bundle.json").read_bytes()).hexdigest()
        )
        return {
            "execution_mode": "base",
            "evaluation_scope": {
                "split": "validation",
                "mode": "base",
                "query_id": "different-query",
                "fold_id": "60-15-15-0000",
            },
        }

    train = make_route_a_wfo_train(
        fit_candidate=lambda _run: (result, factory),
        validation_backtest=wrong_query,
        write_verified=verified_write,
        digest=canonical_sha,
    )
    run = {
        "output": str(tmp_path / "candidate"),
        "route": "route_a",
        "algorithm": "route_a_bc_gru",
        "environment": {
            "start_ns": 100,
            "end_ns": 600,
            "execution": {"zero_target_policy": "close"},
        },
        "wfo_candidate_id": "candidate-a",
        "wfo_fold_id": "60-15-15-0000",
    }
    try:
        train(run, validation_access=access)
    except ValueError as exc:
        assert "wrong scope" in str(exc)
    else:
        raise AssertionError("WFO trainer accepted a report from a different Validation query")


@pytest.mark.parametrize("collapsed_locus", ("fit", "selection", "student"))
def test_route_a_wfo_train_blocks_flat_collapsed_bc_before_bundle_or_validation(
    tmp_path, collapsed_locus
):
    result = training_result()
    flat_behavior = replace(
        result.behavior_summary,
        prediction_nonzero_fraction=0.0,
        prediction_mean=0.0,
        prediction_std=0.0,
        prediction_min=0.0,
        prediction_max=0.0,
        collapsed_to_flat=True,
    )
    if collapsed_locus == "fit":
        result = replace(result, behavior_summary=flat_behavior)
    elif collapsed_locus == "selection":
        selection_behavior = replace(
            result.independent_evaluation.behavior_summary,
            prediction_nonzero_fraction=0.0,
            prediction_mean=0.0,
            prediction_std=0.0,
            prediction_min=0.0,
            prediction_max=0.0,
            collapsed_to_flat=True,
        )
        result = replace(
            result,
            independent_evaluation=replace(
                result.independent_evaluation,
                behavior_summary=selection_behavior,
            ),
        )
    else:
        student_behavior = replace(
            result.independent_evaluation.student_behavior_summary,
            prediction_nonzero_fraction=0.0,
            prediction_mean=0.0,
            prediction_std=0.0,
            prediction_min=0.0,
            prediction_max=0.0,
            collapsed_to_flat=True,
        )
        result = replace(
            result,
            independent_evaluation=replace(
                result.independent_evaluation,
                student_behavior_summary=student_behavior,
            ),
        )
    output = tmp_path / "flat-candidate"
    validation_calls = []
    factory = SimpleNamespace(
        metadata=lambda: {
            "scaler": {
                "mean": [0.0],
                "std": [1.0],
                "feature_hash": "f" * 64,
                "train_window_hash": "t" * 64,
                "sample_count": 10,
            },
        },
        execution=SimpleNamespace(zero_target_policy="close"),
        start_ns=200,
        end_ns=300,
        transition_hash="transition",
        route_a_execution_recipe_hash="a" * 64,
    )
    train = make_route_a_wfo_train(
        fit_candidate=lambda _run: (result, factory),
        validation_backtest=lambda *_args, **_kwargs: validation_calls.append(True),
        write_verified=verified_write,
        digest=canonical_sha,
    )

    try:
        train(
            {
                "output": str(output),
                "environment": {
                    "start_ns": 100,
                    "end_ns": 600,
                    "execution": {"zero_target_policy": "close"},
                },
                "route": "route_a",
                "algorithm": "route_a_bc_gru",
            },
            evaluate=False,
        )
    except ValueError as exc:
        assert "collapsed" in str(exc)
    else:
        raise AssertionError("WFO published a flat-collapsed BC actor")

    assert validation_calls == []
    assert not (output / "bundle.json").exists()


def test_route_a_independent_bc_evaluation_requires_feature_lookback_purge():
    result = training_result()
    source_identity = "a" * 64
    result = replace(
        result,
        source_identity=source_identity,
        transition_hash="transition-1",
        feature_hash="feature",
        train_window_ns=(1, 10),
        training_seed=31,
    )
    read = SimpleNamespace(
        split="train",
        registered_before_read=True,
        source_identity={"marker_sha256": source_identity},
        owner="owner",
        reservation_id="reservation",
    )

    class Factory:
        split = "train_inner"
        mode = "base"
        compact_report_snapshots = False
        route_a_execution_recipe_hash = "a" * 64
        features = SimpleNamespace(hash="feature", max_lookback_ns=5)
        execution = SimpleNamespace(zero_target_policy="close")
        application_registered_train_read = read
        application_registered_train_read_verifier = staticmethod(lambda _read: True)

        def __init__(self, start_ns, end_ns):
            self.start_ns, self.end_ns = start_ns, end_ns
            self.transition_hash = f"transition-{start_ns}"

        def make_env(self):
            raise AssertionError("overlapping evaluation must fail before environment creation")

    fit_factory = Factory(1, 10)
    selection_factory = Factory(14, 30)  # one ns short of the frozen lookback purge
    student_factory = Factory(14, 30)
    grant = HostTrainGrant.from_registered_read(read, fold_id="fold")
    provider = NativeGymTeacherProvider(
        env_factory=selection_factory.make_env,
        teacher=SimpleNamespace(advise=lambda _observation: None),
        fold_id="fold",
        execution_recipe=selection_factory.transition_hash,
        source_identity=source_identity,
        native_order_facts_verifier=lambda *_args: True,
        no_external_cashflow_receipt=lambda _info: "receipt",
        no_external_cashflow_verifier=lambda *_args: True,
        episode_id=lambda *_args: "episode",
    )
    student_provider = NativeGymTeacherProvider(
        env_factory=student_factory.make_env,
        teacher=SimpleNamespace(advise=lambda _observation: None),
        fold_id="fold",
        execution_recipe=student_factory.transition_hash,
        source_identity=source_identity,
        native_order_facts_verifier=lambda *_args: True,
        no_external_cashflow_receipt=lambda _info: "receipt",
        no_external_cashflow_verifier=lambda *_args: True,
        episode_id=lambda *_args: "episode",
    )

    try:
        evaluate_bc_on_registered_train_inner_rollout(
            training_result=result,
            fit_factory=fit_factory,
            selection_factory=selection_factory,
            selection_provider=provider,
            student_factory=student_factory,
            student_provider=student_provider,
            host_rollout=lambda *_args, **_kwargs: pytest.fail("must not start a rollout"),
            grant=grant,
            seed=32,
            student_grant=grant,
            student_seed=33,
        )
    except ValueError as exc:
        assert "feature lookback" in str(exc)
    else:
        raise AssertionError("BC evaluation accepted a selection window inside the feature purge")


def test_route_a_independent_bc_evaluation_replays_separate_registered_train_window(
    monkeypatch, observed
):
    source_identity = "a" * 64
    read = SimpleNamespace(
        split="train",
        registered_before_read=True,
        source_identity={"marker_sha256": source_identity},
        owner="owner",
        reservation_id="reservation",
    )

    class Factory:
        split = "train_inner"
        mode = "base"
        compact_report_snapshots = False
        route_a_execution_recipe_hash = "a" * 64
        features = SimpleNamespace(hash="feature", max_lookback_ns=5)
        execution = SimpleNamespace(zero_target_policy="close")
        application_registered_train_read = read
        application_registered_train_read_verifier = staticmethod(lambda _read: True)

        def __init__(self, start_ns, end_ns):
            self.start_ns, self.end_ns = start_ns, end_ns
            self.transition_hash = f"transition-{start_ns}"

        def make_env(self):
            raise AssertionError("the verified host rollout callback owns env creation")

    fit_factory, selection_factory = Factory(1, 100), Factory(106, 200)
    selection_read = SimpleNamespace(
        split="train",
        registered_before_read=True,
        source_identity={"marker_sha256": source_identity},
        owner="owner",
        reservation_id="selection-reservation",
    )
    selection_factory.application_registered_train_read = selection_read
    grant = HostTrainGrant.from_registered_read(selection_read, fold_id="fold")
    student_factory = Factory(106, 200)
    student_read = SimpleNamespace(
        split="train",
        registered_before_read=True,
        source_identity={"marker_sha256": source_identity},
        owner="owner",
        reservation_id="student-reservation",
    )
    student_factory.application_registered_train_read = student_read
    student_grant = HostTrainGrant.from_registered_read(student_read, fold_id="fold")
    from rles_route_a.teachers.trend import TrendTeacher

    teacher = TrendTeacher()
    provider = NativeGymTeacherProvider(
        env_factory=selection_factory.make_env,
        teacher=teacher,
        fold_id="fold",
        execution_recipe=selection_factory.transition_hash,
        source_identity=source_identity,
        native_order_facts_verifier=lambda *_args: True,
        no_external_cashflow_receipt=lambda _info: "receipt",
        no_external_cashflow_verifier=lambda *_args: True,
        episode_id=lambda *_args: "episode",
    )
    student_provider = NativeGymTeacherProvider(
        env_factory=student_factory.make_env,
        teacher=TrendTeacher(),
        fold_id="fold",
        execution_recipe=student_factory.transition_hash,
        source_identity=source_identity,
        native_order_facts_verifier=lambda *_args: True,
        no_external_cashflow_receipt=lambda _info: "receipt",
        no_external_cashflow_verifier=lambda *_args: True,
        episode_id=lambda *_args: "episode",
    )
    fit_observation = replace(observed(ts=50, episode="fit"), fold_id="fold")
    selection_observation = replace(observed(ts=110, episode="selection"), fold_id="fold")
    fit_label = LabeledStep(
        fit_observation,
        ExpertAdvice("trend_v1", 50, "OPEN_LONG", 0.5, ("trend",)),
        0.5,
        source_identity=source_identity,
    )
    selection_label = LabeledStep(
        selection_observation,
        ExpertAdvice("trend_v1", 110, "OPEN_LONG", 0.5, ("trend",)),
        0.5,
        source_identity=source_identity,
    )
    student_label = LabeledStep(
        selection_observation,
        ExpertAdvice("bc_student_native_eval", 110, "OPEN_LONG", 0.5, ("BC_STUDENT_NATIVE_EVAL",)),
        0.5,
        source_identity=source_identity,
    )
    result = replace(
        training_result(),
        labeled_steps=(fit_label,),
        source_identity=source_identity,
        transition_hash=fit_factory.transition_hash,
        feature_hash="feature",
        execution_recipe_hash="a" * 64,
        teacher_hash=teacher_config_hash(teacher),
        market_feature_map=tuple(sorted(provider.market_feature_map.items())),
        train_window_ns=(1, 100),
        training_seed=31,
    )
    native = SimpleNamespace(fold_id="fold", decisions=1)
    behavior = replace(result.behavior_summary, sample_count=1, executable_label_count=1)
    calls = []
    provider.rollout_with_m22_trace = lambda **kwargs: (
        calls.append(kwargs)
        or (
            ("teacher-barrier",),
            {
                "execution_decision_trace": ("decision",),
                "execution_mode": "base",
            },
        )
    )
    student_provider.rollout_with_m22_trace = lambda **kwargs: (
        calls.append(kwargs)
        or (
            ("student-barrier",),
            {
                "execution_decision_trace": ("decision",),
                "execution_mode": "base",
            },
        )
    )
    monkeypatch.setattr(native_bc, "audit_teacher_barriers", lambda barriers: ((), native))
    monkeypatch.setattr(
        native_bc,
        "teacher_native_rollout",
        lambda provider_fn, **_kwargs: (
            selection_label if next(iter(provider_fn())) == "teacher-barrier" else student_label,
        ),
    )
    monkeypatch.setattr(native_bc, "_native_behavior_summary", lambda *_args, **_kwargs: behavior)
    monkeypatch.setattr(
        native_bc,
        "_native_barrier_evidence",
        lambda barriers, _report: barrier_evidence(1, str(next(iter(barriers)))),
    )

    evaluated = evaluate_bc_on_registered_train_inner_rollout(
        training_result=result,
        fit_factory=fit_factory,
        selection_factory=selection_factory,
        selection_provider=provider,
        student_factory=student_factory,
        student_provider=student_provider,
        host_rollout=lambda *_args, **_kwargs: None,
        grant=grant,
        seed=32,
        student_grant=student_grant,
        student_seed=33,
    )

    assert len(calls) == 2 and calls[0]["factory"] is selection_factory
    assert calls[0]["seed"] == 32
    assert grant.reservation_id == "selection-reservation"
    assert calls[1]["factory"] is student_factory
    assert calls[1]["decision_policy"] is result.policy
    assert calls[1]["seed"] == 33
    assert student_grant.reservation_id == "student-reservation"
    assert evaluated.independent_evaluation.window_ns == (106, 200)
    assert evaluated.independent_evaluation.purge_ns == 5
    assert evaluated.independent_evaluation.behavior_summary.sample_count == 1
    assert evaluated.independent_evaluation.student_native_summary.decisions == 1


def test_route_a_wfo_final_retrain_skips_validation_report(tmp_path):
    result = training_result()
    factory = SimpleNamespace(
        metadata=lambda: {
            "scaler": {
                "mean": [0.0],
                "std": [1.0],
                "feature_hash": "f" * 64,
                "train_window_hash": "t" * 64,
                "sample_count": 10,
            },
        },
        execution=SimpleNamespace(zero_target_policy="close"),
        start_ns=200,
        end_ns=300,
        transition_hash="transition",
        route_a_execution_recipe_hash="a" * 64,
    )
    train = make_route_a_wfo_train(
        fit_candidate=lambda _run: (result, factory),
        validation_backtest=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("final retrain must not evaluate Validation")
        ),
        write_verified=verified_write,
        digest=canonical_sha,
    )
    output = tmp_path / "final"
    trained = train(
        {
            "output": str(output),
            "environment": {
                "start_ns": 100,
                "end_ns": 600,
                "execution": {"zero_target_policy": "close"},
            },
            "route": "route_a",
            "algorithm": "route_a_bc_gru",
        },
        evaluate=False,
    )
    assert trained["output"] == str(output)
    assert not (output / "report.json").exists()
    assert json.loads((output / "COMPLETED.json").read_text())["report_sha256"] is None


@pytest.mark.parametrize(
    "failure",
    ("fit_outside_train", "selection_outside_train", "wrong_factory", "wrong_selection_recipe"),
)
def test_route_a_wfo_rejects_unbound_or_out_of_scope_train_result(tmp_path, failure):
    result = training_result()
    factory = SimpleNamespace(
        start_ns=200,
        end_ns=300,
        transition_hash="transition",
        route_a_execution_recipe_hash="a" * 64,
        execution=SimpleNamespace(zero_target_policy="close"),
    )
    train_end = 600
    if failure == "fit_outside_train":
        train_end = 200
    elif failure == "selection_outside_train":
        train_end = 400
    elif failure == "wrong_factory":
        factory.transition_hash = "other-transition"
    else:
        result = replace(
            result,
            independent_evaluation=replace(
                result.independent_evaluation, execution_recipe_hash="b" * 64
            ),
        )
    output = tmp_path / failure
    train = make_route_a_wfo_train(
        fit_candidate=lambda _run: (result, factory),
        validation_backtest=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Validation must not run")
        ),
        write_verified=verified_write,
        digest=canonical_sha,
    )
    with pytest.raises(PermissionError):
        train(
            {
                "output": str(output),
                "environment": {
                    "start_ns": 100,
                    "end_ns": train_end,
                    "execution": {"zero_target_policy": "close"},
                },
            },
            evaluate=False,
        )
    assert not (output / "bundle.json").exists()


def test_route_a_wfo_train_requires_close_policy_before_fitting_or_output(tmp_path):
    output = tmp_path / "blocked-candidate"
    calls = []
    train = make_route_a_wfo_train(
        fit_candidate=lambda _run: calls.append("fit"),
        validation_backtest=lambda *_args, **_kwargs: {},
        write_verified=verified_write,
        digest=canonical_sha,
    )
    run = {
        "output": str(output),
        "environment": {"start_ns": 100, "end_ns": 200},
        "route": "route_a",
        "algorithm": "route_a_bc_gru",
    }

    try:
        train(run, evaluate=False)
    except PermissionError as exc:
        assert "zero_target_policy='close'" in str(exc)
    else:
        raise AssertionError("WFO accepted the legacy deadband execution profile")

    assert calls == []
    assert not output.exists()
