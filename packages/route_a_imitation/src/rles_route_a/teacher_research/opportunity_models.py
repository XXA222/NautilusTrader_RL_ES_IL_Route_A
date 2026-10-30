"""Train-only frozen-exit opportunity models for Route A research.

This module consumes outcomes produced by an explicitly identified research
simulator. It performs no market-data IO and never converts a counterfactual
label into native execution or economic authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Literal, Mapping, Sequence

import numpy as np
from .candidate_events import OpportunityEvent


Split = Literal["train_fit", "train_calibration", "train_inner_screen"]
OutcomeEvidence = Literal["MODEL_COUNTERFACTUAL"]
ReturnBasisId = Literal[
    "LOG_REFERENCE_ENTRY_NOTIONAL_BPS",
    "LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS",
]
RETURN_BASIS_IDS = frozenset({
    "LOG_REFERENCE_ENTRY_NOTIONAL_BPS",
    "LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS",
})


@dataclass(frozen=True, slots=True)
class OpportunityOutcome:
    """One complete candidate outcome under a frozen continuation policy.

    ``net_return_bps`` is reconciled as gross reference-price return less fees
    and modeled slippage plus signed funding. A censored candidate carries no
    complete-trade economics and is never turned into a zero-return example.
    """

    event_id: str
    candidate_family: Literal["breakout", "pullback"]
    candidate_side: Literal[-1, 1]
    decision_ts_ns: int
    feature_available_at_ns: int
    rule_version: str
    snapshot_id: str
    feature_spec_hash: str
    feature_vector_sha256: str
    label_end_ts_ns: int | None
    net_return_bps: float | None
    fold_id: str
    source_identity: str
    split: Split
    independent_block_id: str
    independent_block_start_ts_ns: int
    independent_block_end_ts_ns: int
    exit_policy_id: str
    execution_recipe: str
    cost_scenario_id: str
    return_basis_id: ReturnBasisId
    simulation_receipt_id: str
    outcome_method: str
    evidence_type: OutcomeEvidence = "MODEL_COUNTERFACTUAL"
    economic_status: Literal["SCENARIO_ONLY"] = "SCENARIO_ONLY"
    censor_ts_ns: int | None = None
    gross_return_bps: float | None = None
    fees_bps: float | None = None
    slippage_bps: float | None = None
    funding_cashflow_bps: float | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.event_id, str) or len(self.event_id) != 64
                or any(c not in "0123456789abcdef" for c in self.event_id)):
            raise ValueError("outcome event_id must be lowercase SHA-256")
        if self.candidate_family not in {"breakout", "pullback"}:
            raise ValueError("outcome candidate family is invalid")
        if type(self.candidate_side) is not int or self.candidate_side not in {-1, 1}:
            raise ValueError("outcome candidate side must be -1 or +1")
        if not isinstance(self.rule_version, str) or not self.rule_version.strip():
            raise ValueError("outcome requires its frozen candidate rule version")
        for name in ("snapshot_id", "feature_spec_hash", "feature_vector_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"outcome requires its {name} SHA-256")
        if (type(self.decision_ts_ns) is not int or self.decision_ts_ns <= 0
                or type(self.feature_available_at_ns) is not int
                or not 0 <= self.feature_available_at_ns <= self.decision_ts_ns):
            raise ValueError("outcome timestamps must be valid and causal")
        for name in ("fold_id", "source_identity", "independent_block_id",
                     "exit_policy_id", "execution_recipe", "cost_scenario_id",
                     "return_basis_id", "simulation_receipt_id", "outcome_method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"outcome {name} is required")
        if self.return_basis_id not in RETURN_BASIS_IDS:
            raise ValueError("outcome return basis is not an explicit supported reference basis")
        if (type(self.independent_block_start_ts_ns) is not int
                or type(self.independent_block_end_ts_ns) is not int
                or self.independent_block_start_ts_ns < 0
                or self.independent_block_end_ts_ns <= self.independent_block_start_ts_ns):
            raise ValueError("outcome independent block needs a valid [start, end) interval")
        if self.split not in {"train_fit", "train_calibration", "train_inner_screen"}:
            raise PermissionError("opportunity research cannot ingest Validation/Test/Holdout")
        if self.evidence_type != "MODEL_COUNTERFACTUAL":
            raise ValueError("only full-universe model counterfactual labels can train this model")
        if self.economic_status != "SCENARIO_ONLY":
            raise ValueError("opportunity-model outputs remain SCENARIO_ONLY")
        if self.net_return_bps is None:
            if (self.label_end_ts_ns is not None or type(self.censor_ts_ns) is not int
                    or self.censor_ts_ns <= self.decision_ts_ns):
                raise ValueError("censored outcomes require only a future censor clock")
            if any(value is not None for value in (
                    self.gross_return_bps, self.fees_bps, self.slippage_bps,
                    self.funding_cashflow_bps)):
                raise ValueError("censored outcomes cannot supply complete-trade economics")
        elif (isinstance(self.net_return_bps, bool)
              or not isinstance(self.net_return_bps, (int, float))
              or not math.isfinite(self.net_return_bps)
              or type(self.label_end_ts_ns) is not int
              or self.label_end_ts_ns <= self.decision_ts_ns
              or self.censor_ts_ns is not None):
            raise ValueError("complete outcomes require finite net return and future end clock")
        else:
            components = (self.gross_return_bps, self.fees_bps,
                          self.slippage_bps, self.funding_cashflow_bps)
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in components):
                raise ValueError("complete outcomes require finite gross/cost components")
            if self.fees_bps < 0 or self.slippage_bps < 0:
                raise ValueError("fees and modeled slippage must be nonnegative")
            expected_net = (self.gross_return_bps - self.fees_bps
                            - self.slippage_bps + self.funding_cashflow_bps)
            tolerance = 1e-8 * max(1.0, abs(self.net_return_bps), abs(expected_net))
            if abs(self.net_return_bps - expected_net) > tolerance:
                raise ValueError("gross, fees, slippage and funding do not reconcile to net return")


@dataclass(frozen=True, slots=True)
class NetValueEstimate:
    event_id: str
    feature_vector_sha256: str
    fold_id: str
    source_identity: str
    exit_policy_id: str
    execution_recipe: str
    cost_scenario_id: str
    return_basis_id: str
    rule_version: str
    outcome_method: str
    feature_spec_hash: str
    expected_net_increment_bps: float
    global_residual_floor_proxy_bps: float
    epistemic_uncertainty_bps: float
    probability_positive: float
    fold_calibration_id: str
    in_support: bool
    economic_status: str
    evidence_type: str


@dataclass(frozen=True, slots=True)
class FrozenOpportunityModel:
    """Deterministic Logistic + ridge baseline with later-block calibration."""

    feature_names: tuple[str, ...]
    feature_spec_hash: str
    mean: np.ndarray
    scale: np.ndarray
    fit_min: np.ndarray
    fit_max: np.ndarray
    logistic_weights: np.ndarray
    calibrator: tuple[float, float]
    value_weights: np.ndarray
    calibration_residual_p10_offset_bps: float
    residual_variance: float
    value_covariance: np.ndarray
    fit_sample_count: int
    calibration_sample_count: int
    fit_independent_blocks: int
    calibration_independent_blocks: int
    calibration_id: str
    fold_id: str
    source_identity: str
    exit_policy_id: str
    execution_recipe: str
    rule_version: str
    outcome_method: str
    cost_scenario_id: str
    return_basis_id: str
    evidence_type: str
    economic_status: str
    fit_effective_label_count: float
    calibration_effective_label_count: float
    fit_weighting_id: str

    def predict(
        self, event: OpportunityEvent, features: Sequence[float] | np.ndarray,
    ) -> NetValueEstimate:
        if not isinstance(event, OpportunityEvent):
            raise TypeError("prediction requires an auditable causal OpportunityEvent")
        if (event.feature_spec_hash != self.feature_spec_hash
                or event.fold_id != self.fold_id
                or event.source_identity != self.source_identity
                or event.rule_version != self.rule_version):
            raise ValueError("prediction event differs from the frozen model source/fold/schema")
        row = _feature_row(features, len(self.feature_names))
        vector_sha256 = feature_vector_sha256(
            event.event_id, event.feature_spec_hash, self.feature_names, row,
        )
        design_row = _design_features(row, event.side, event.family)
        normalized = (design_row - self.mean) / self.scale
        design = np.concatenate((np.array([1.0]), normalized))
        raw_logit = float(design @ self.logistic_weights)
        calibrated_probability = _sigmoid(
            self.calibrator[0] * raw_logit + self.calibrator[1]
        )
        expected = float(design @ self.value_weights)
        uncertainty = math.sqrt(max(
            0.0, self.residual_variance * float(design @ self.value_covariance @ design)
        ))
        in_support = bool(
            np.all(design_row >= self.fit_min) and np.all(design_row <= self.fit_max)
        )
        return NetValueEstimate(
            event_id=event.event_id,
            feature_vector_sha256=vector_sha256,
            fold_id=self.fold_id,
            source_identity=self.source_identity,
            exit_policy_id=self.exit_policy_id,
            execution_recipe=self.execution_recipe,
            cost_scenario_id=self.cost_scenario_id,
            return_basis_id=self.return_basis_id,
            rule_version=self.rule_version,
            outcome_method=self.outcome_method,
            feature_spec_hash=self.feature_spec_hash,
            expected_net_increment_bps=expected,
            # This is a mean plus one global residual offset, not a conditional q10.
            global_residual_floor_proxy_bps=(
                expected + self.calibration_residual_p10_offset_bps
            ),
            epistemic_uncertainty_bps=uncertainty,
            probability_positive=calibrated_probability,
            fold_calibration_id=self.calibration_id,
            in_support=in_support,
            economic_status=self.economic_status,
            evidence_type=self.evidence_type,
        )


def fit_opportunity_model(
    fit_outcomes: Sequence[OpportunityOutcome],
    calibration_outcomes: Sequence[OpportunityOutcome],
    feature_rows: Mapping[str, Sequence[float]],
    *,
    candidate_events: Mapping[str, OpportunityEvent],
    feature_names: Sequence[str],
    l2: float = 1.0,
    learning_rate: float = 0.05,
    iterations: int = 2000,
) -> FrozenOpportunityModel:
    """Fit on Train-fit and Platt-calibrate on later Train-calibration only.

    Any fit label reaching the first calibration decision is purged. Fixed
    hyperparameters are intentionally small and explicit; this helper does no
    hyperparameter search and does not read a screen/Validation/Test split.
    """
    names = tuple(feature_names)
    if (not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("unique frozen feature names are required")
    if (not math.isfinite(l2) or l2 <= 0 or not math.isfinite(learning_rate)
            or learning_rate <= 0 or type(iterations) is not int or iterations < 1):
        raise ValueError("invalid fixed Logistic/ridge recipe")
    if not fit_outcomes or not calibration_outcomes:
        raise ValueError("fit and later calibration samples are both required")
    all_rows = (*fit_outcomes, *calibration_outcomes)
    if any(not isinstance(row, OpportunityOutcome) for row in all_rows):
        raise TypeError("opportunity outcomes must use the audited outcome contract")
    if any(row.split != "train_fit" for row in fit_outcomes):
        raise PermissionError("model fitting accepts only Train-fit outcomes")
    if any(row.split != "train_calibration" for row in calibration_outcomes):
        raise PermissionError("calibration accepts only later Train-calibration outcomes")
    if len({row.event_id for row in all_rows}) != len(all_rows):
        raise ValueError("fit/calibration event ids must be disjoint")
    if set(feature_rows) != {row.event_id for row in all_rows}:
        raise ValueError("feature rows must exactly match fit/calibration candidate ids")
    if set(candidate_events) != {row.event_id for row in all_rows}:
        raise ValueError("candidate events must exactly match fit/calibration outcome ids")
    for outcome in all_rows:
        event = candidate_events[outcome.event_id]
        if not isinstance(event, OpportunityEvent):
            raise TypeError("candidate event mapping must contain OpportunityEvent records")
        if (
            event.event_id != outcome.event_id
            or event.family != outcome.candidate_family
            or event.side != outcome.candidate_side
            or event.decision_ts_ns != outcome.decision_ts_ns
            or event.feature_available_at_ns != outcome.feature_available_at_ns
            or event.rule_version != outcome.rule_version
            or event.snapshot_id != outcome.snapshot_id
            or event.feature_spec_hash != outcome.feature_spec_hash
            or event.fold_id != outcome.fold_id
            or event.source_identity != outcome.source_identity
        ):
            raise ValueError("candidate event provenance differs from its outcome label")
        feature_values = _feature_row(feature_rows[outcome.event_id], len(names))
        feature_sha256 = feature_vector_sha256(
            outcome.event_id, outcome.feature_spec_hash, names, feature_values,
        )
        if feature_sha256 != outcome.feature_vector_sha256:
            raise ValueError("candidate feature values differ from the label-bound snapshot")
    first_calibration = min(row.decision_ts_ns for row in calibration_outcomes)
    purged = tuple(
        row for row in fit_outcomes
        if row.net_return_bps is not None
        and row.label_end_ts_ns is not None
        and row.label_end_ts_ns < first_calibration
    )
    calibration = tuple(row for row in calibration_outcomes if row.net_return_bps is not None)
    if len(purged) < 4 or len(calibration) < 4:
        raise ValueError("insufficient complete, purged Train-fit/calibration outcomes")
    if len({row.independent_block_id for row in purged}) < 2:
        raise ValueError("Train-fit outcomes need at least two independent time blocks")
    if len({row.independent_block_id for row in calibration}) < 2:
        raise ValueError("Train-calibration outcomes need at least two later time blocks")
    fit_intervals = _validate_independent_block_intervals(purged, "Train-fit")
    calibration_intervals = _validate_independent_block_intervals(
        calibration, "Train-calibration",
    )
    if max(end for _, end in fit_intervals) > min(start for start, _ in calibration_intervals):
        raise ValueError("Train-calibration time blocks must begin after Train-fit blocks end")
    identity_fields = (
        "fold_id", "source_identity", "exit_policy_id", "execution_recipe",
        "cost_scenario_id", "return_basis_id", "evidence_type", "economic_status",
        "feature_spec_hash", "rule_version", "outcome_method",
    )
    for field_name in identity_fields:
        if len({getattr(row, field_name) for row in (*purged, *calibration)}) != 1:
            raise ValueError(f"fit/calibration mismatch in {field_name}")
    fit_x = _matrix(purged, feature_rows, names)
    cal_x = _matrix(calibration, feature_rows, names)
    fit_y = np.asarray([_complete_return(row) for row in purged], dtype=np.float64)
    cal_y = np.asarray([_complete_return(row) for row in calibration], dtype=np.float64)
    fit_weights = overlap_uniqueness_weights(purged)
    calibration_weights = overlap_uniqueness_weights(calibration)
    fit_positive = (fit_y > 0).astype(np.float64)
    cal_positive = (cal_y > 0).astype(np.float64)
    if len(np.unique(fit_positive)) != 2 or len(np.unique(cal_positive)) != 2:
        raise ValueError("Logistic fit and later calibration each require wins and losses")

    mean = np.average(fit_x, axis=0, weights=fit_weights)
    scale = np.sqrt(np.average((fit_x - mean) ** 2, axis=0, weights=fit_weights))
    scale[scale < 1e-12] = 1.0
    fit_z = (fit_x - mean) / scale
    cal_z = (cal_x - mean) / scale
    fit_design = np.column_stack((np.ones(len(fit_z)), fit_z))
    cal_design = np.column_stack((np.ones(len(cal_z)), cal_z))

    logistic = _fit_logistic(fit_design, fit_positive, l2, learning_rate, iterations,
                             sample_weights=fit_weights)
    cal_logits = cal_design @ logistic
    calibrator = _fit_platt(cal_logits, cal_positive, learning_rate, iterations,
                            sample_weights=calibration_weights)

    penalty = np.eye(fit_design.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    ridge_weights = fit_weights
    weighted_design = fit_design * ridge_weights[:, None]
    gram = fit_design.T @ weighted_design + penalty
    value_weights = np.linalg.solve(gram, fit_design.T @ (ridge_weights * fit_y))
    residual = fit_y - fit_design @ value_weights
    calibration_residual = cal_y - cal_design @ value_weights
    effective_n = uniqueness_effective_label_count(fit_weights)
    degrees = max(1.0, effective_n - fit_design.shape[1])
    residual_mse = float(np.dot(ridge_weights, residual ** 2)) / float(ridge_weights.sum())
    residual_variance = max(residual_mse * effective_n / degrees, 1e-12)
    value_covariance = np.linalg.solve(gram, np.eye(gram.shape[0]))
    calibration_material = json.dumps({
        "records": sorted((
            row.event_id, row.decision_ts_ns, row.label_end_ts_ns,
            _complete_return(row).hex(), row.feature_vector_sha256,
            row.simulation_receipt_id,
        ) for row in calibration),
        "fold_id": purged[0].fold_id,
        "source_identity": purged[0].source_identity,
        "rule_version": purged[0].rule_version,
        "outcome_method": purged[0].outcome_method,
        "feature_spec_hash": purged[0].feature_spec_hash,
        "exit_policy_id": purged[0].exit_policy_id,
        "execution_recipe": purged[0].execution_recipe,
        "cost_scenario_id": purged[0].cost_scenario_id,
        "return_basis_id": purged[0].return_basis_id,
        "feature_names": names,
        "l2": l2,
        "learning_rate": learning_rate,
        "iterations": iterations,
        "overlap_weighting": "average_inverse_concurrency_v1",
    }, sort_keys=True, separators=(",", ":"))
    calibration_id = sha256(calibration_material.encode("utf-8")).hexdigest()
    return FrozenOpportunityModel(
        feature_names=names,
        feature_spec_hash=purged[0].feature_spec_hash,
        mean=_readonly(mean),
        scale=_readonly(scale),
        fit_min=_readonly(fit_x.min(axis=0)),
        fit_max=_readonly(fit_x.max(axis=0)),
        logistic_weights=_readonly(logistic),
        calibrator=calibrator,
        value_weights=_readonly(value_weights),
        calibration_residual_p10_offset_bps=_weighted_quantile(
            calibration_residual, calibration_weights, 0.10,
        ),
        residual_variance=residual_variance,
        value_covariance=_readonly(value_covariance),
        fit_sample_count=len(purged),
        calibration_sample_count=len(calibration),
        fit_independent_blocks=len({row.independent_block_id for row in purged}),
        calibration_independent_blocks=len({
            row.independent_block_id for row in calibration
        }),
        calibration_id=calibration_id,
        fold_id=purged[0].fold_id,
        source_identity=purged[0].source_identity,
        exit_policy_id=purged[0].exit_policy_id,
        execution_recipe=purged[0].execution_recipe,
        rule_version=purged[0].rule_version,
        outcome_method=purged[0].outcome_method,
        cost_scenario_id=purged[0].cost_scenario_id,
        return_basis_id=purged[0].return_basis_id,
        evidence_type=purged[0].evidence_type,
        economic_status=purged[0].economic_status,
        fit_effective_label_count=uniqueness_effective_label_count(fit_weights),
        calibration_effective_label_count=uniqueness_effective_label_count(calibration_weights),
        fit_weighting_id="AVERAGE_INVERSE_LABEL_CONCURRENCY_V1",
    )


def _feature_row(values: Sequence[float] | np.ndarray, width: int) -> np.ndarray:
    row = np.asarray(values, dtype=np.float64)
    if row.shape != (width,) or not np.all(np.isfinite(row)):
        raise ValueError("feature row has wrong shape or non-finite values")
    return row


def overlap_uniqueness_weights(outcomes: Sequence[object]) -> np.ndarray:
    """Share model influence across complete labels covering the same time."""
    rows = tuple(outcomes)
    if not rows:
        return np.zeros(0, dtype=np.float64)
    if len({getattr(row, "event_id", None) for row in rows}) != len(rows):
        raise ValueError("uniqueness weights require distinct complete future labels")
    events: list[tuple[int, int, int]] = []
    durations = []
    for i, row in enumerate(rows):
        start = getattr(row, "decision_ts_ns", None)
        end = getattr(row, "label_end_ts_ns", None)
        value = getattr(row, "net_return_bps", None)
        status = getattr(row, "outcome_status", "COMPLETE")
        if (status != "COMPLETE" or value is None or type(start) is not int
                or type(end) is not int or end <= start):
            raise ValueError("uniqueness weights require distinct complete future labels")
        durations.append(end - start)
        events.extend(((start, 1, i), (end, 0, i)))
    events.sort()
    active: set[int] = set()
    covered = np.zeros(len(rows), dtype=np.float64)
    cursor = events[0][0]
    position = 0
    while position < len(events):
        timestamp = events[position][0]
        if timestamp > cursor and active:
            duration_per_label = (timestamp - cursor) / len(active)
            for i in active:
                covered[i] += duration_per_label
        while position < len(events) and events[position][0] == timestamp:
            _, kind, i = events[position]
            if kind == 0:
                active.remove(i)
            else:
                active.add(i)
            position += 1
        cursor = timestamp
    weights = covered / np.asarray(durations, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0) or np.any(weights > 1 + 1e-12):
        raise ValueError("overlap uniqueness weights are invalid")
    return weights


def uniqueness_effective_label_count(weights: Sequence[float] | np.ndarray) -> float:
    """Sum average-uniqueness weights; this is a time-coverage count, not Kish ESS."""
    values = np.asarray(weights, dtype=np.float64)
    if not len(values):
        return 0.0
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("effective unique-label count requires positive finite weights")
    return float(values.sum())


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if (values.shape != weights.shape or not np.isfinite(values).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not 0 <= quantile <= 1):
        raise ValueError("weighted quantile requires finite aligned values and positive weights")
    order = np.argsort(values, kind="stable")
    sorted_values, sorted_weights = values[order], weights[order]
    threshold = quantile * sorted_weights.sum()
    index = min(int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left")),
                len(values) - 1)
    return float(sorted_values[index])


def _validate_independent_block_intervals(
    outcomes: Sequence[OpportunityOutcome], split_name: str,
) -> tuple[tuple[int, int], ...]:
    intervals_by_id: dict[str, tuple[int, int]] = {}
    for outcome in outcomes:
        if (outcome.decision_ts_ns < outcome.independent_block_start_ts_ns
                or outcome.decision_ts_ns >= outcome.independent_block_end_ts_ns):
            raise ValueError(f"{split_name} decision must lie inside its independent block")
        if (outcome.label_end_ts_ns is not None
                and outcome.label_end_ts_ns >= outcome.independent_block_end_ts_ns):
            raise ValueError(f"{split_name} outcome label crosses its independent block boundary")
        interval = (
            outcome.independent_block_start_ts_ns,
            outcome.independent_block_end_ts_ns,
        )
        prior = intervals_by_id.setdefault(outcome.independent_block_id, interval)
        if prior != interval:
            raise ValueError(f"{split_name} block ID has inconsistent time boundaries")
    intervals = tuple(sorted(intervals_by_id.values()))
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError(f"{split_name} independent time blocks overlap")
    return intervals


def _matrix(
    outcomes: Sequence[OpportunityOutcome],
    feature_rows: Mapping[str, Sequence[float] | np.ndarray],
    feature_names: Sequence[str],
) -> np.ndarray:
    missing = [row.event_id for row in outcomes if row.event_id not in feature_rows]
    if missing:
        raise ValueError("feature matrix is missing candidate event rows")
    features = []
    for outcome in outcomes:
        row = _feature_row(feature_rows[outcome.event_id], len(feature_names))
        actual_sha256 = feature_vector_sha256(
            outcome.event_id, outcome.feature_spec_hash, feature_names, row,
        )
        if actual_sha256 != outcome.feature_vector_sha256:
            raise ValueError("candidate feature values differ from the label-bound snapshot")
        features.append(_design_features(
            row, outcome.candidate_side, outcome.candidate_family,
        ))
    return np.vstack(features)


def _complete_return(outcome: OpportunityOutcome) -> float:
    value = outcome.net_return_bps
    if value is None or not math.isfinite(value):
        raise ValueError("complete opportunity return is required and must be finite")
    return float(value)


def feature_vector_sha256(
    event_id: str,
    feature_spec_hash: str,
    feature_names: Sequence[str],
    values: Sequence[float] | np.ndarray,
) -> str:
    """Bind ordered causal feature values to one candidate and feature schema."""
    if (not isinstance(event_id, str) or not re.fullmatch(r"[0-9a-f]{64}", event_id)
            or not isinstance(feature_spec_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", feature_spec_hash)):
        raise ValueError("feature vector identity needs event and feature-spec SHA-256")
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("feature vector identity needs ordered unique feature names")
    row = _feature_row(values, len(names))
    payload = json.dumps({
        "event_id": event_id,
        "feature_spec_hash": feature_spec_hash,
        "feature_names": names,
        "values_hex": tuple(float(value).hex() for value in row),
    }, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _design_features(row: np.ndarray, side: int, family: str) -> np.ndarray:
    family_code = 0.0 if family == "breakout" else 1.0
    return np.concatenate((row, np.asarray((float(side), family_code))))


def _fit_logistic(
    design: np.ndarray, target: np.ndarray, l2: float,
    learning_rate: float, iterations: int,
    sample_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    observation_weights = _validated_sample_weights(sample_weights, len(target))
    observation_weights /= observation_weights.sum()
    weights = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.ones_like(weights)
    penalty[0] = 0.0
    for _ in range(iterations):
        probability = np.asarray([_sigmoid(float(x)) for x in design @ weights])
        gradient = design.T @ (observation_weights * (probability - target)) + l2 * penalty * weights
        weights -= learning_rate * gradient
        if not np.all(np.isfinite(weights)):
            raise ValueError("Logistic fit diverged; reject this frozen recipe")
    return weights


def _fit_platt(
    logits: np.ndarray, target: np.ndarray, learning_rate: float, iterations: int,
    sample_weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[float, float]:
    design = np.column_stack((logits, np.ones(len(logits))))
    params = np.array([1.0, 0.0], dtype=np.float64)
    observation_weights = _validated_sample_weights(sample_weights, len(target))
    observation_weights /= observation_weights.sum()
    for _ in range(iterations):
        probability = np.asarray([_sigmoid(float(x)) for x in design @ params])
        gradient = design.T @ (observation_weights * (probability - target))
        params -= learning_rate * gradient
        if not np.all(np.isfinite(params)):
            raise ValueError("Platt calibration diverged; reject this calibration block")
    return float(params[0]), float(params[1])


def _validated_sample_weights(
    sample_weights: Sequence[float] | np.ndarray | None, count: int,
) -> np.ndarray:
    if sample_weights is None:
        return np.ones(count, dtype=np.float64)
    result = np.asarray(sample_weights, dtype=np.float64)
    if (result.shape != (count,) or not np.isfinite(result).all()
            or np.any(result <= 0)):
        raise ValueError("sample weights must be finite, positive, and row-aligned")
    return result.copy()


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 700.0)))
    exp_value = math.exp(max(value, -700.0))
    return exp_value / (1.0 + exp_value)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result
