"""Train Route A BC from a verified registered-Train host Gym episode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from hashlib import sha256
import json
from typing import Any, Callable, Mapping, Protocol, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from rles_route_a.config import BCConfig
from rles_route_a.contracts import INTENTS
from rles_route_a.datasets.sequence_dataset import EpisodeSequenceDataset, LabeledStep
from rles_route_a.export.inference import BCInferencePolicy, intent_aware_exposure
from rles_route_a.evaluation.teacher_gate import StudentScreen
from rles_route_a.integration.gym_rollout import NativeGymTeacherProvider
from rles_route_a.integration.native_rollout import teacher_native_rollout
from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.integration.trace_audit import NativeTraceSummary, audit_teacher_barriers
from rles_route_a.integration.execution_contract import require_route_a_flat_close
from rles_route_a.models.bc import BCGRUPolicy
from .bc_trainer import BCEpoch, set_seed, train_bc


class _FeatureRecipe(Protocol):
    hash: str
    max_lookback_ns: int


class _RouteATrainFactory(Protocol):
    """Validated host factory fields consumed by Route A training/evaluation."""

    split: str
    mode: str
    transition_hash: str
    start_ns: int
    end_ns: int
    features: _FeatureRecipe
    execution: Any
    compact_report_snapshots: bool
    application_registered_train_read: Any
    application_registered_train_read_verifier: Callable[[Any], bool]

    def make_env(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class NativeBCBehaviorSummary:
    """Action-domain diagnostics over the reconciled native training episode."""

    sample_count: int
    executable_label_count: int
    label_nonzero_fraction: float
    prediction_nonzero_fraction: float
    prediction_mean: float
    prediction_std: float
    prediction_min: float
    prediction_max: float
    mae: float | None
    mse: float | None
    deadband: float
    collapsed_to_flat: bool


@dataclass(frozen=True)
class NativeBCIndependentEvaluation:
    """Reconciled teacher-label metrics from a purged Train-inner window."""

    behavior_summary: NativeBCBehaviorSummary
    native_summary: NativeTraceSummary
    window_ns: tuple[int, int]
    fold_id: str
    source_identity: str
    transition_hash: str
    feature_hash: str
    teacher_hash: str
    purge_ns: int
    teacher_barrier_evidence: tuple[Mapping[str, Any], ...]
    student_barrier_evidence: tuple[Mapping[str, Any], ...]
    student_behavior_summary: NativeBCBehaviorSummary
    student_native_summary: NativeTraceSummary
    student_rollout_report: Mapping[str, Any]
    rollout_report: Mapping[str, Any]
    execution_recipe_hash: str = ""


@dataclass(frozen=True)
class NativeBCTrainingResult:
    policy: BCInferencePolicy
    config: BCConfig
    history: tuple[BCEpoch, ...]
    label_count: int
    executable_label_count: int
    labeled_steps: tuple[LabeledStep, ...]
    native_summary: NativeTraceSummary
    native_rollout_report: Mapping[str, Any]
    behavior_summary: NativeBCBehaviorSummary
    native_barrier_evidence: tuple[Mapping[str, Any], ...] = ()
    source_identity: str = ""
    transition_hash: str = ""
    execution_recipe_hash: str = ""
    feature_hash: str = ""
    teacher_hash: str = ""
    market_feature_map: tuple[tuple[str, str], ...] = ()
    train_window_ns: tuple[int, int] | None = None
    training_seed: int | None = None
    independent_evaluation: NativeBCIndependentEvaluation | None = None
    student_screen: StudentScreen | None = None


def teacher_config_hash(teacher: Any) -> str:
    """Hash a frozen dataclass teacher's type and declared configuration."""
    params = getattr(type(teacher), "__dataclass_params__", None)
    if not is_dataclass(teacher) or params is None or params.frozen is not True:
        raise TypeError("registered Route A BC requires a frozen dataclass teacher")
    payload = {
        "type": f"{type(teacher).__module__}.{type(teacher).__qualname__}",
        "config": asdict(cast(Any, teacher)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _route_a_execution_recipe_hash(factory: Any) -> str:
    """Read the window-independent frozen recipe identity from a Train factory."""
    value = getattr(factory, "route_a_execution_recipe_hash", None)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "Train factory lacks a frozen window-independent Route A execution recipe hash"
        )
    return value


def _native_behavior_summary(
    model: BCGRUPolicy,
    dataset: EpisodeSequenceDataset,
    *,
    batch_size: int,
    device: str,
    inference_mode: str = "continuous_exposure_v1",
    deadband: float = 0.05,
) -> NativeBCBehaviorSummary:
    model.eval()
    predictions: list[np.ndarray] = []
    intent_predictions: list[np.ndarray] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        with torch.no_grad():
            logits, exposure = model(batch["x"].to(device), batch["length"].to(device))
        values = exposure.detach().cpu().numpy().astype(np.float64, copy=False)
        classes = logits.argmax(dim=-1).detach().cpu().numpy()
        if not np.isfinite(values).all() or not np.isfinite(logits.detach().cpu().numpy()).all():
            raise FloatingPointError("native BC behavior evaluation produced non-finite actions")
        predictions.append(values)
        intent_predictions.append(classes)
    raw_predicted = np.concatenate(predictions)
    predicted = raw_predicted.copy()
    if inference_mode == "intent_aware_v1":
        predicted = np.asarray(
            [
                intent_aware_exposure(
                    INTENTS[int(intent_index)],
                    float(raw_target),
                    step.observation.account_value("net_exposure"),
                )
                for step, intent_index, raw_target in zip(
                    dataset.steps, np.concatenate(intent_predictions), raw_predicted,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
    elif inference_mode != "continuous_exposure_v1":
        raise ValueError("unsupported BC behavior-summary inference mode")
    weights = np.asarray([step.supervision_weight for step in dataset.steps], dtype=np.float64)
    labels = np.asarray(
        [step.advice.requested_exposure for step in dataset.steps], dtype=np.float64
    )
    executable = weights > 0
    errors = predicted[executable] - labels[executable] if executable.any() else None
    return NativeBCBehaviorSummary(
        sample_count=len(dataset),
        executable_label_count=int(executable.sum()),
        label_nonzero_fraction=(float(np.mean(np.abs(labels[executable]) > deadband))
                                if executable.any() else 0.0),
        prediction_nonzero_fraction=float(np.mean(np.abs(predicted) > deadband)),
        prediction_mean=float(np.mean(predicted)),
        prediction_std=float(np.std(predicted)),
        prediction_min=float(np.min(predicted)),
        prediction_max=float(np.max(predicted)),
        mae=float(np.mean(np.abs(errors))) if errors is not None else None,
        mse=float(np.mean(np.square(errors))) if errors is not None else None,
        deadband=deadband,
        collapsed_to_flat=bool(np.all(np.abs(predicted) <= deadband)),
    )


def _native_barrier_evidence(
    barriers: tuple[Any, ...], rollout_report: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    """Project native Train barriers into a market-feature-free audit trace."""
    raw_trace = rollout_report.get("execution_decision_trace")
    if not isinstance(raw_trace, (tuple, list)):
        raise ValueError("native rollout report omitted its M2.2 action trace")
    trace_by_id = {
        row.get("decision_id"): row
        for row in raw_trace
        if isinstance(row, Mapping) and isinstance(row.get("decision_id"), str)
    }
    evidence = []
    for barrier in barriers:
        native = barrier.trace
        reward = barrier.reward
        row = trace_by_id.get(native.execution_decision_id)
        actor_target = None if row is None else row.get("actor_requested_target")
        if (
            row is None
            or row.get("decision_ts_ns") != native.execution_barrier_ts_ns
            or isinstance(actor_target, bool)
            or not isinstance(actor_target, (int, float, np.number))
            or not np.isfinite(float(actor_target))
            or not np.isclose(float(actor_target), native.requested_action, rtol=0.0, atol=1e-6)
        ):
            raise ValueError("M2.2 action trace does not bind the native Route A barrier")
        fills = row.get("fills", ())
        if not isinstance(fills, (tuple, list)):
            raise ValueError("native M2.2 fill evidence is malformed")
        evidence.append(
            {
                "decision_id": native.execution_decision_id,
                "decision_ts_ns": native.decision_ts_ns,
                "execution_barrier_ts_ns": native.execution_barrier_ts_ns,
                "trajectory_id": barrier.trajectory_id,
                "actor_id": barrier.advice.expert_id,
                "intent": barrier.advice.intent,
                "requested_exposure": native.requested_action,
                "soft_risk_target": native.soft_risk_target,
                "risk_adjusted_target": native.risk_adjusted_target,
                "planner_target": native.planner_target,
                "requested_to_soft_delta": (
                    None
                    if native.soft_risk_target is None
                    else native.soft_risk_target - native.requested_action
                ),
                "soft_to_risk_delta": (
                    None
                    if native.soft_risk_target is None
                    else native.risk_adjusted_target - native.soft_risk_target
                ),
                "actual_exposure_after": native.actual_exposure_after,
                "native_equity_before": reward.prior_equity,
                "native_equity_after": reward.next_equity,
                "reward_econ": reward.recorded_r_econ,
                "has_external_cashflow": reward.has_external_cashflow,
                "planner_action": native.planner_action,
                "planner_reason": native.planner_reason,
                "submitted_order_ids": list(native.submitted_order_ids),
                "filled_order_ids": list(native.filled_order_ids),
                "related_existing_order_ids": list(native.related_existing_order_ids),
                "pending_order_ids_after": list(native.pending_order_ids_after),
                "fee_currency": native.fee_currency,
                "fills": [dict(fill) for fill in fills],
                "risk_adjustment": {
                    "hard_risk_adjustment": row.get("hard_risk_adjustment"),
                    "execution_adjusted": row.get("action") != "NOOP",
                },
                "scenario_only": native.scenario_only,
            }
        )
    if len(evidence) != len(raw_trace):
        raise ValueError("M2.2 trace and native barrier evidence have different coverage")
    return tuple(evidence)


def train_bc_from_registered_native_rollout(
    *,
    factory: Any,
    provider: NativeGymTeacherProvider,
    host_rollout: Callable[..., tuple[dict, np.ndarray]],
    grant: HostTrainGrant,
    config: BCConfig,
    seed: int,
    device: str = "cpu",
    patience: int = 8,
    rollout_options: Mapping[str, Any] | None = None,
) -> NativeBCTrainingResult:
    """Collect, reconcile, label, and train from one registered native episode.

    This entry point accepts only the provider assembled from a prepared host
    factory. Its retained Train receipt is reverified before rollout; the
    provider then reconciles each policy barrier against the same-run M2.2
    trace and requires L6 reward/terminal receipts before labels are created.
    """
    if not isinstance(provider, NativeGymTeacherProvider):
        raise TypeError("registered native Gym teacher provider required")
    if not isinstance(grant, HostTrainGrant):
        raise PermissionError("host registered Train grant required")
    if not isinstance(config, BCConfig):
        raise TypeError("strict Route A BC config required")
    if type(seed) is not int or not 0 <= seed < 2**32 or config.seed != seed:
        raise ValueError("rollout and BC training must share one frozen non-negative seed")
    if not callable(host_rollout):
        raise TypeError("host orchestrator rollout callable required")
    factory = cast(_RouteATrainFactory, factory)
    factory_end_ns = getattr(factory, "end_ns", None)
    if type(factory_end_ns) is not int:
        raise ValueError("registered BC Train factory has no integer end boundary")
    factory_mode = getattr(factory, "mode", None)
    if (
        getattr(factory, "split", None) != "train_inner"
        or not isinstance(factory_mode, str)
        or not callable(getattr(factory, "make_env", None))
    ):
        raise PermissionError("BC training requires the host Train-inner Gym factory")
    require_route_a_flat_close(getattr(factory, "execution", None), context="Route A BC training")
    if getattr(factory, "compact_report_snapshots", None) is not False:
        raise PermissionError(
            "native BC requires full retained M2.2 snapshots for order/fill reconciliation"
        )
    execution_recipe_hash = _route_a_execution_recipe_hash(factory)
    read = getattr(factory, "application_registered_train_read", None)
    verifier = getattr(factory, "application_registered_train_read_verifier", None)
    if not callable(verifier) or verifier(read) is not True:
        raise PermissionError("host project registry did not verify the retained Train read")
    expected_grant = HostTrainGrant.from_registered_read(read, fold_id=grant.fold_id)
    if grant != expected_grant:
        raise PermissionError("Route A Train grant differs from the prepared host read receipt")
    if (
        provider.fold_id != grant.fold_id
        or provider.source_identity != grant.source_identity
        or provider.execution_recipe != getattr(factory, "transition_hash", None)
        or getattr(provider.env_factory, "__self__", None) is not factory
    ):
        raise PermissionError(
            "Route A provider is not bound to this Train factory, Fold, and source"
        )

    options = dict(rollout_options or {})
    if "seed" in options or "include_execution_trace" in options:
        raise ValueError("frozen seed and M2.2 trace options are owned by the Route A trainer")
    barriers, rollout_report = provider.rollout_with_m22_trace(
        factory=factory,
        host_rollout=host_rollout,
        grant=grant,
        seed=seed,
        **options,
    )
    labels, summary = audit_teacher_barriers(barriers)
    if len(labels) != len(barriers):
        raise ValueError("native teacher labels do not cover every reconciled decision")
    if (summary.positive_native_trade_case_count == 0
            or summary.executable_label_count == 0):
        case_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}
        for row in labels:
            case_counts[row.economic_case_status] = (
                case_counts.get(row.economic_case_status, 0) + 1
            )
            action_counts[row.action_status] = action_counts.get(row.action_status, 0) + 1
        raise ValueError(
            "native teacher rollout contains no reconciled positive flat-to-flat trade cases; "
            f"economic_status={summary.economic_status}, decisions={summary.decisions}, "
            f"case_status_counts={dict(sorted(case_counts.items()))}, "
            f"action_status_counts={dict(sorted(action_counts.items()))}, "
            f"unique_positive_cases={summary.positive_native_trade_case_count}, "
            f"unique_completed_cases={summary.completed_native_trade_case_count}, "
            f"native_fills={summary.native_fills}, "
            f"verified_lifecycle_barriers="
            f"{sum(bool(row.native_order_lifecycle_verified) for row in barriers)}, "
            f"native_economic_barriers="
            f"{sum(bool(row.native_economic_evidence_authority) for row in barriers)}"
        )
    if any(
        row.economic_case_status.startswith("OBSERVED_NATIVE_TRADE_")
        and type(row.case_end_ts_ns) is int
        and cast(int, row.case_end_ts_ns) > factory_end_ns
        for row in labels
    ):
        raise PermissionError("teacher trade-case outcome extends beyond the registered fit window")
    if any(
        row.economic_case_status == "OPEN_CASE_CENSORED"
        and type(row.case_censor_ts_ns) is int
        and cast(int, row.case_censor_ts_ns) > factory_end_ns
        for row in labels
    ):
        raise PermissionError("censored trade prefix extends beyond the registered fit window")
    if len(rollout_report.get("execution_decision_trace", ())) != len(labels) or rollout_report.get(
        "execution_mode"
    ) not in (None, factory_mode):
        raise ValueError("M2.2 native rollout report differs from the Train factory episode")
    dataset = EpisodeSequenceDataset(
        labels,
        feature_names=config.feature_names,
        context_bars=config.context_bars,
        max_gap_ns=config.max_gap_ns,
    )
    if dataset.fold_id != grant.fold_id or dataset.source_identity != grant.source_identity:
        raise PermissionError("Route A BC sequence dataset lost registered Train scope")
    set_seed(config.seed)
    model = BCGRUPolicy(config.feature_names, config.hidden_size, config.layers)
    history = train_bc(model, dataset, config, device=device, patience=patience)
    behavior = _native_behavior_summary(
        model,
        dataset,
        batch_size=config.batch_size,
        device=device,
        inference_mode=config.inference_mode,
    )
    barrier_evidence = _native_barrier_evidence(barriers, rollout_report)
    policy = BCInferencePolicy(model, config, device=device)
    return NativeBCTrainingResult(
        policy=policy,
        config=config,
        history=tuple(history),
        label_count=len(labels),
        executable_label_count=sum(
            row.supervision_weight is not None and row.supervision_weight > 0 for row in labels
        ),
        labeled_steps=labels,
        native_summary=summary,
        native_rollout_report=dict(rollout_report),
        behavior_summary=behavior,
        native_barrier_evidence=barrier_evidence,
        source_identity=grant.source_identity,
        transition_hash=getattr(factory, "transition_hash"),
        execution_recipe_hash=execution_recipe_hash,
        feature_hash=getattr(getattr(factory, "features"), "hash"),
        teacher_hash=teacher_config_hash(provider.teacher),
        market_feature_map=tuple(sorted(provider.market_feature_map.items())),
        train_window_ns=(getattr(factory, "start_ns"), getattr(factory, "end_ns")),
        training_seed=seed,
    )


def evaluate_bc_on_registered_train_inner_rollout(
    *,
    training_result: NativeBCTrainingResult,
    fit_factory: Any,
    selection_factory: Any,
    selection_provider: NativeGymTeacherProvider,
    student_factory: Any,
    student_provider: NativeGymTeacherProvider,
    host_rollout: Callable[..., tuple[dict, np.ndarray]],
    grant: HostTrainGrant,
    seed: int,
    student_grant: HostTrainGrant,
    student_seed: int,
    rollout_options: Mapping[str, Any] | None = None,
) -> NativeBCTrainingResult:
    """Attach an independent, purged Train-inner native teacher evaluation.

    Selection uses a separately prepared factory from the same registered
    Train source. Its scoring window starts after the fit window plus the
    frozen feature lookback. No model weights are changed by this call.
    """
    if not isinstance(training_result, NativeBCTrainingResult):
        raise TypeError("completed Route A native BC training result required")
    if (
        not isinstance(selection_provider, NativeGymTeacherProvider)
        or not isinstance(grant, HostTrainGrant)
        or not callable(host_rollout)
    ):
        raise TypeError("registered selection provider, Train grant, and host rollout required")
    if type(seed) is not int or not 0 <= seed < 2**32 or seed == training_result.training_seed:
        raise ValueError("independent Train-inner rollout requires a distinct nonnegative seed")
    if not isinstance(student_provider, NativeGymTeacherProvider) or not isinstance(
        student_grant, HostTrainGrant
    ):
        raise TypeError("a separately registered native BC student replay is required")
    fit_factory = cast(_RouteATrainFactory, fit_factory)
    selection_factory = cast(_RouteATrainFactory, selection_factory)
    student_factory = cast(_RouteATrainFactory, student_factory)
    registered_grants = []
    for candidate in (fit_factory, selection_factory, student_factory):
        if (
            getattr(candidate, "split", None) != "train_inner"
            or not isinstance(getattr(candidate, "mode", None), str)
            or not callable(getattr(candidate, "make_env", None))
            or getattr(candidate, "compact_report_snapshots", None) is not False
        ):
            raise PermissionError(
                "independent BC evaluation requires full-snapshot Train-inner factories"
            )
        require_route_a_flat_close(
            getattr(candidate, "execution", None), context="Route A Train-inner BC evaluation"
        )
        read = getattr(candidate, "application_registered_train_read", None)
        verifier = getattr(candidate, "application_registered_train_read_verifier", None)
        if not callable(verifier) or verifier(read) is not True:
            raise PermissionError("host registry did not verify a Train-inner evaluation read")
        registered_grants.append(
            HostTrainGrant.from_registered_read(
                read,
                fold_id=grant.fold_id,
            )
        )
    fit_grant, selection_grant, student_registered_grant = registered_grants
    execution_recipe_hash = _route_a_execution_recipe_hash(fit_factory)
    if execution_recipe_hash != training_result.execution_recipe_hash:
        raise ValueError("BC result differs from the fit factory's frozen execution recipe")
    if any(
        _route_a_execution_recipe_hash(candidate) != execution_recipe_hash
        for candidate in (selection_factory, student_factory)
    ):
        raise ValueError("fit, selection, and student factories differ in frozen execution recipe")
    if selection_grant != grant:
        raise PermissionError("selection grant differs from its registered Train read")
    if student_registered_grant != student_grant:
        raise PermissionError("student replay grant differs from its registered Train read")
    if (
        fit_grant.fold_id != grant.fold_id
        or fit_grant.source_identity != training_result.source_identity
        or fit_grant.owner != grant.owner
        or student_grant.fold_id != grant.fold_id
        or student_grant.source_identity != grant.source_identity
        or student_grant.owner != grant.owner
    ):
        raise PermissionError(
            "fit, selection, and student Train reads differ in Fold, source, or owner"
        )

    fit_window = training_result.train_window_ns
    if (
        fit_window is None
        or (fit_factory.start_ns, fit_factory.end_ns) != fit_window
        or fit_factory.transition_hash != training_result.transition_hash
        or type(fit_window[0]) is not int
        or type(fit_window[1]) is not int
        or fit_window[0] >= fit_window[1]
    ):
        raise ValueError("BC result is not bound to the supplied fit window")
    if (
        selection_factory is fit_factory
        or student_factory is fit_factory
        or student_factory is selection_factory
    ):
        raise ValueError("fit, selection, and student replay require separate prepared factories")
    lookback_ns = selection_factory.features.max_lookback_ns
    if (
        type(lookback_ns) is not int
        or lookback_ns < 0
        or selection_factory.end_ns <= selection_factory.start_ns
        or selection_factory.start_ns < fit_window[1] + lookback_ns
    ):
        raise ValueError("Train-inner selection window overlaps fit data or its feature lookback")
    if (
        selection_factory.mode != fit_factory.mode
        or selection_factory.features.hash != training_result.feature_hash
    ):
        raise ValueError("fit and selection factories differ in mode or feature recipe")
    if (
        student_factory.start_ns != selection_factory.start_ns
        or student_factory.end_ns != selection_factory.end_ns
        or student_factory.mode != selection_factory.mode
        or student_factory.features.hash != training_result.feature_hash
    ):
        raise ValueError(
            "BC student replay must match the independent teacher selection window and recipe"
        )
    if (
        grant.fold_id != training_result.native_summary.fold_id
        or grant.source_identity != training_result.source_identity
        or selection_provider.fold_id != grant.fold_id
        or selection_provider.source_identity != grant.source_identity
        or selection_provider.execution_recipe != selection_factory.transition_hash
        or getattr(selection_provider.env_factory, "__self__", None) is not selection_factory
    ):
        raise PermissionError(
            "independent BC evaluation differs from fitted Train source or recipe"
        )
    if (
        tuple(sorted(selection_provider.market_feature_map.items()))
        != training_result.market_feature_map
    ):
        raise ValueError("fit and selection teacher feature projections differ")
    selection_teacher_hash = teacher_config_hash(selection_provider.teacher)
    if selection_teacher_hash != training_result.teacher_hash:
        raise ValueError("fit and selection teacher configurations differ")
    if (
        student_provider.fold_id != student_grant.fold_id
        or student_provider.source_identity != student_grant.source_identity
        or student_provider.execution_recipe != student_factory.transition_hash
        or getattr(student_provider.env_factory, "__self__", None) is not student_factory
        or tuple(sorted(student_provider.market_feature_map.items()))
        != training_result.market_feature_map
        or teacher_config_hash(student_provider.teacher) != training_result.teacher_hash
    ):
        raise PermissionError("BC student replay provider differs from the fitted Train recipe")
    if (
        type(student_seed) is not int
        or not 0 <= student_seed < 2**32
        or student_seed in {seed, training_result.training_seed}
    ):
        raise ValueError("student native replay requires its own deterministic Train-inner seed")

    options = dict(rollout_options or {})
    if "seed" in options or "include_execution_trace" in options:
        raise ValueError("selection seed and M2.2 trace options are owned by the evaluator")
    barriers, report = selection_provider.rollout_with_m22_trace(
        factory=selection_factory,
        host_rollout=host_rollout,
        grant=grant,
        seed=seed,
        **options,
    )
    _audited, native_summary = audit_teacher_barriers(barriers)
    labels = teacher_native_rollout(
        lambda **_kwargs: iter(barriers), fold_id=grant.fold_id, grant=grant,
    )
    if not labels or any(
        not getattr(selection_factory, "start_ns")
        <= row.observation.decision_ts_ns
        < getattr(selection_factory, "end_ns")
        for row in labels
    ):
        raise ValueError("independent BC evaluation labels escaped the scoring window")
    fit_times = [row.observation.decision_ts_ns for row in training_result.labeled_steps]
    selection_times = [row.observation.decision_ts_ns for row in labels]
    fit_teachers = {row.advice.expert_id for row in training_result.labeled_steps}
    selection_teachers = {row.advice.expert_id for row in labels}
    if (
        not fit_times
        or max(fit_times) >= min(selection_times)
        or fit_teachers != selection_teachers
    ):
        raise ValueError(
            "selection labels are not temporally independent or use a different teacher"
        )
    dataset = EpisodeSequenceDataset(
        labels,
        feature_names=training_result.config.feature_names,
        context_bars=training_result.config.context_bars,
        max_gap_ns=training_result.config.max_gap_ns,
    )
    if dataset.fold_id != grant.fold_id or dataset.source_identity != grant.source_identity:
        raise PermissionError("independent BC evaluation lost its registered Train scope")
    behavior = _native_behavior_summary(
        training_result.policy.model,
        dataset,
        batch_size=training_result.config.batch_size,
        device=training_result.policy.device,
        inference_mode=training_result.config.inference_mode,
    )
    if len(report.get("execution_decision_trace", ())) != len(labels) or report.get(
        "execution_mode"
    ) not in (None, getattr(selection_factory, "mode")):
        raise ValueError("M2.2 report differs from the independent BC selection episode")

    student_barriers, student_report = student_provider.rollout_with_m22_trace(
        factory=student_factory,
        host_rollout=host_rollout,
        grant=student_grant,
        seed=student_seed,
        decision_policy=training_result.policy,
        rollout_id="route-a-bc-student-train-inner-evaluation",
        **options,
    )
    _student_audited, student_native_summary = audit_teacher_barriers(student_barriers)
    student_labels = teacher_native_rollout(
        lambda **_kwargs: iter(student_barriers),
        fold_id=student_grant.fold_id, grant=student_grant,
    )
    if (
        not student_labels
        or any(
            not getattr(selection_factory, "start_ns")
            <= row.observation.decision_ts_ns
            < getattr(selection_factory, "end_ns")
            for row in student_labels
        )
        or {row.advice.expert_id for row in student_labels} != {"bc_student_native_eval"}
        or len(student_report.get("execution_decision_trace", ())) != len(student_labels)
        or student_report.get("execution_mode") not in (None, getattr(student_factory, "mode"))
    ):
        raise ValueError("native BC student replay differs from its frozen Train-inner scope")
    student_dataset = EpisodeSequenceDataset(
        student_labels,
        feature_names=training_result.config.feature_names,
        context_bars=training_result.config.context_bars,
        max_gap_ns=training_result.config.max_gap_ns,
    )
    if (
        student_dataset.fold_id != student_grant.fold_id
        or student_dataset.source_identity != student_grant.source_identity
    ):
        raise PermissionError("native BC student replay lost its registered Train scope")
    student_behavior = _native_behavior_summary(
        training_result.policy.model,
        student_dataset,
        batch_size=training_result.config.batch_size,
        device=training_result.policy.device,
        inference_mode=training_result.config.inference_mode,
    )
    teacher_evidence = _native_barrier_evidence(barriers, report)
    student_evidence = _native_barrier_evidence(student_barriers, student_report)
    selection = NativeBCIndependentEvaluation(
        behavior_summary=behavior,
        native_summary=native_summary,
        window_ns=(getattr(selection_factory, "start_ns"), getattr(selection_factory, "end_ns")),
        fold_id=grant.fold_id,
        source_identity=grant.source_identity,
        transition_hash=getattr(selection_factory, "transition_hash"),
        feature_hash=getattr(getattr(selection_factory, "features"), "hash"),
        teacher_hash=selection_teacher_hash,
        purge_ns=lookback_ns,
        teacher_barrier_evidence=teacher_evidence,
        student_barrier_evidence=student_evidence,
        student_behavior_summary=student_behavior,
        student_native_summary=student_native_summary,
        student_rollout_report=dict(student_report),
        rollout_report=dict(report),
        execution_recipe_hash=execution_recipe_hash,
    )
    return replace(training_result, independent_evaluation=selection)
