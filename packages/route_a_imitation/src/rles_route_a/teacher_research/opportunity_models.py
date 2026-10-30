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


def summarize_prediction_calibration(
    predicted_net_bps: Sequence[float],
    observed_net_bps: Sequence[float],
    predicted_probability_positive: Sequence[float],
    observed_positive: Sequence[int | bool],
    *,
    sample_weights: Sequence[float] | np.ndarray | None = None,
    bin_count: int = 5,
) -> dict[str, object]:
    """Compare frozen-model predictions with later calibration outcomes.

    Bins are equal-count, stable-sorted groups. This reports calibration only;
    it never adjusts model parameters or candidate acceptance thresholds.
    """
    predicted = np.asarray(predicted_net_bps, dtype=np.float64)
    observed = np.asarray(observed_net_bps, dtype=np.float64)
    probability = np.asarray(predicted_probability_positive, dtype=np.float64)
    positive = np.asarray(observed_positive, dtype=np.float64)
    if (predicted.ndim != 1 or not len(predicted)
            or observed.shape != predicted.shape or probability.shape != predicted.shape
            or positive.shape != predicted.shape
            or not all(np.isfinite(x).all() for x in (predicted, observed, probability, positive))
            or np.any((probability < 0) | (probability > 1))
            or np.any((positive != 0) & (positive != 1))
            or type(bin_count) is not int or bin_count < 1):
        raise ValueError("calibration predictions and outcomes must be finite aligned vectors")
    weights = _validated_sample_weights(sample_weights, len(predicted))
    if len(predicted) != len(observed):
        raise ValueError("calibration prediction and outcome counts differ")

    def weighted_mean(values: np.ndarray, indices: np.ndarray) -> float:
        selected_weights = weights[indices]
        return float(np.average(values[indices], weights=selected_weights))

    def bins_by(values: np.ndarray, order_values: np.ndarray) -> list[dict[str, float | int]]:
        order = np.argsort(order_values, kind="stable")
        bins = np.array_split(order, min(bin_count, len(order)))
        result = []
        for indices in bins:
            result.append({
                "samples": int(len(indices)),
                "weight_sum": float(weights[indices].sum()),
                "mean_predicted_net_bps": weighted_mean(predicted, indices),
                "mean_observed_net_bps": weighted_mean(observed, indices),
                "mean_net_prediction_error_bps": weighted_mean(predicted - observed, indices),
                "mean_predicted_probability_positive": weighted_mean(probability, indices),
                "observed_positive_rate": weighted_mean(positive, indices),
            })
        return result

    error = predicted - observed
    clipped = np.clip(probability, 1e-15, 1 - 1e-15)
    effective_count = float(weights.sum() ** 2 / np.square(weights).sum())
    return {
        "samples": int(len(predicted)),
        "effective_weighted_samples": effective_count,
        "mean_predicted_net_bps": weighted_mean(predicted, np.arange(len(predicted))),
        "mean_observed_net_bps": weighted_mean(observed, np.arange(len(observed))),
        "mean_net_prediction_error_bps": weighted_mean(error, np.arange(len(error))),
        "net_prediction_mae_bps": float(np.average(np.abs(error), weights=weights)),
        "net_prediction_rmse_bps": float(np.sqrt(np.average(np.square(error), weights=weights))),
        "mean_predicted_probability_positive": weighted_mean(probability, np.arange(len(probability))),
        "observed_positive_rate": weighted_mean(positive, np.arange(len(positive))),
        "probability_brier_score": float(np.average(np.square(probability - positive), weights=weights)),
        "probability_log_loss": float(np.average(
            -(positive * np.log(clipped) + (1 - positive) * np.log(1 - clipped)),
            weights=weights,
        )),
        "net_return_bins": bins_by(predicted, predicted),
        "probability_bins": bins_by(probability, probability),
        "binning": "stable_equal_count_weighted_summary_v1",
        "thresholds_adjusted": False,
    }


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
       #n½òÚ$z{-®éÜj×¸¡½ÕÑ½µ”è=ÁÁ½ÉÑÕ¹¥Ñå=ÕÑ½µ”¤€´ø™±½…Ðè(€€€Ù…±Õ”€ô½ÕÑ½µ”¹¹•Ñ}É•ÑÕÉ¹}‰ÁÌ(€€€¥˜Ù…±Õ”¥Ì9½¹”½È¹½Ðµ…Ñ ¹¥Í™¥¹¥Ñ”¡Ù…±Õ”¤è(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰½µÁ±•Ñ”½ÁÁ½ÉÑÕ¹¥ÑäÉ•ÑÕÉ¸¥ÌÉ•ÅÕ¥É•…¹µÕÍÐ‰”™¥¹¥Ñ”ˆ¤(€€€É•ÑÕÉ¸™±½…Ð¡Ù…±Õ”¤(()‘•˜™•…ÑÕÉ•}Ù•Ñ½É}Í¡„ÈÔØ (€€€•Ù•¹Ñ}¥èÍÑÈ°(€€€™•…ÑÕÉ•}ÍÁ•}¡…Í èÍÑÈ°(€€€™•…ÑÕÉ•}¹…µ•ÌèM•ÅÕ•¹•mÍÑÉt°(€€€Ù…±Õ•ÌèM•ÅÕ•¹•m™±½…Ñtð¹À¹¹‘…ÉÉ…ä°(¤€´øÍÑÈè(€€€€ˆˆ‰	¥¹½É‘•É•…ÕÍ…°™•…ÑÕÉ”Ù…±Õ•ÌÑ¼½¹”…¹‘¥‘…Ñ”…¹™•…ÑÕÉ”Í¡•µ„¸ˆˆˆ(€€€¥˜€¡¹½Ð¥Í¥¹ÍÑ…¹”¡•Ù•¹Ñ}¥°ÍÑÈ¤½È¹½ÐÉ”¹™Õ±±µ…Ñ ¡È‰lÀ´å„µ™uìØÑôˆ°•Ù•¹Ñ}¥¤(€€€€€€€€€€€½È¹½Ð¥Í¥¹ÍÑ…¹”¡™•…ÑÕÉ•}ÍÁ•}¡…Í °ÍÑÈ¤(€€€€€€€€€€€½È¹½ÐÉ”¹™Õ±±µ…Ñ ¡È‰lÀ´å„µ™uìØÑôˆ°™•…ÑÕÉ•}ÍÁ•}¡…Í ¤¤è(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰™•…ÑÕÉ”Ù•Ñ½È¥‘•¹Ñ¥Ñä¹••‘Ì•Ù•¹Ð…¹™•…ÑÕÉ”µÍÁ•ŒM!´ÈÔØˆ¤(€€€¹…µ•Ì€ôÑÕÁ±”¡™•…ÑÕÉ•}¹…µ•Ì¤(€€€¥˜¹½Ð¹…µ•Ì½È±•¸¡Í•Ð¡¹…µ•Ì¤¤€„ô±•¸¡¹…µ•Ì¤½È…¹ä¡¹½Ð¹…µ”™½È¹…µ”¥¸¹…µ•Ì¤è(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰™•…ÑÕÉ”Ù•Ñ½È¥‘•¹Ñ¥Ñä¹••‘Ì½É‘•É•Õ¹¥ÅÕ”™•…ÑÕÉ”¹…µ•Ìˆ¤(€€€É½Ü€ô}™•…ÑÕÉ•}É½Ü¡Ù…±Õ•Ì°±•¸¡¹…µ•Ì¤¤(€€€Á…å±½…€ô©Í½¸¹‘ÕµÁÌ¡ì(€€€€€€€€‰•Ù•¹Ñ}¥ˆè•Ù•¹Ñ}¥°(€€€€€€€€‰™•…ÑÕÉ•}ÍÁ•}¡…Í ˆè™•…ÑÕÉ•}ÍÁ•}¡…Í °(€€€€€€€€‰™•…ÑÕÉ•}¹…µ•Ìˆè¹…µ•Ì°(€€€€€€€€‰Ù…±Õ•Í}¡•àˆèÑÕÁ±”¡™±½…Ð¡Ù…±Õ”¤¹¡•à ¤™½ÈÙ…±Õ”¥¸É½Ü¤°(€€€ô°Í½ÉÑ}­•åÌõQÉÕ”°Í•Á…É…Ñ½ÉÌô ˆ°ˆ°€ˆèˆ¤¤(€€€É•ÑÕÉ¸Í¡„ÈÔØ¡Á…å±½…¹•¹½‘” ‰ÕÑ˜´àˆ¤¤¹¡•á‘¥•ÍÐ ¤(()‘•˜}‘•Í¥¹}™•…ÑÕÉ•Ì¡É½Üè¹À¹¹‘…ÉÉ…ä°Í¥‘”è¥¹Ð°™…µ¥±äèÍÑÈ¤€´ø¹À¹¹‘…ÉÉ…äè(€€€™…µ¥±å}½‘”€ô€À¸À¥˜™…µ¥±ä€ôô€‰‰É•…­½ÕÐˆ•±Í”€Ä¸À(€€€É•ÑÕÉ¸¹À¹½¹…Ñ•¹…Ñ” ¡É½Ü°¹À¹…Í…ÉÉ…ä ¡™±½…Ð¡Í¥‘”¤°™…µ¥±å}½‘”¤¤¤¤(()‘•˜}™¥Ñ}±½¥ÍÑ¥Œ (€€€‘•Í¥¸è¹À¹¹‘…ÉÉ…ä°Ñ…É•Ðè¹À¹¹‘…ÉÉ…ä°°Èè™±½…Ð°(€€€±•…É¹¥¹}É…Ñ”è™±½…Ð°¥Ñ•É…Ñ¥½¹Ìè¥¹Ð°(€€€Í…µÁ±•}Ý•¥¡ÑÌèM•ÅÕ•¹•m™±½…Ñtð¹À¹¹‘…ÉÉ…äð9½¹”€ô9½¹”°(¤€´ø¹À¹¹‘…ÉÉ…äè(€€€½‰Í•ÉÙ…Ñ¥½¹}Ý•¥¡ÑÌ€ô}Ù…±¥‘…Ñ•‘}Í…µÁ±•}Ý•¥¡ÑÌ¡Í…µÁ±•}Ý•¥¡ÑÌ°±•¸¡Ñ…É•Ð¤¤(€€€½‰Í•ÉÙ…Ñ¥½¹}Ý•¥¡ÑÌ€¼ô½‰Í•ÉÙ…Ñ¥½¹}Ý•¥¡ÑÌ¹ÍÕ´ ¤(€€€Ý•¥¡ÑÌ€ô¹À¹é•É½Ì¡‘•Í¥¸¹Í¡…Á•lÅt°‘ÑåÁ”õ¹À¹™±½…ÐØÐ¤(€€€Á•¹…±Ñä€ô¹À¹½¹•Í}±¥­”¡Ý•¥¡ÑÌ¤(€€€Á•¹…±ÑålÁt€ô€À¸À(€€€™½È|¥¸É…¹”¡¥Ñ•É…Ñ¥½¹Ì¤è(€€€€€€€ÁÉ½‰…‰¥±¥Ñä€ô¹À¹…Í…ÉÉ…ä¡m}Í¥µ½¥¡™±½…Ð¡à¤¤™½Èà¥¸‘•Í¥¸ Ý•¥¡ÑÍt¤(€€€€€€€É…‘¥•¹Ð€ô‘•Í¥¸¹P €¡½‰Í•ÉÙ…Ñ¥½¹}Ý•¥¡ÑÌ€¨€¡ÁÉ½‰…‰¥±¥Ñä€´Ñ…É•Ð¤¤€¬°È€¨Á•¹…±Ñä€¨Ý•¥¡ÑÌ(€€€€€€€Ý•¥¡ÑÌ€´ô±•…É¹¥¹}É…Ñ”€¨É…‘¥•¹Ð(€€€€€€€¥˜¹½Ð¹À¹…±°¡¹À¹¥Í™¥¹¥Ñ”¡Ý•¥¡ÑÌ¤¤è(€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰1½¥ÍÑ¥Œ™¥Ð‘¥Ù•É•ìÉ•©•ÐÑ¡¥Ì™É½é•¸É•¥Á”ˆ¤(€€€É•ÑÕÉ¸Ý•¥¡ÑÌ(()‘•˜}™¥Ñ}Á±…ÑÐ (€€€±½¥ÑÌè¹À¹¹‘…ÉÉ…ä°Ñ…É•Ðè¹À¹¹‘…ÉÉ…ä°±•…É¹¥¹}É…Ñ”è™±½…Ð°¥Ñ•É…Ñ¥½¹Ìè¥¹Ð°(€€€Í…µÁ±•}Ý•¥¡ÑÌèM•ÅÕ•¹•m™±½…Ñtð¹À¹¹‘…ÉÉ…äð9½¹”€ô9½¹”°(¤€´øÑÕÁ±•m™±½…Ð°™±½…Ñtè(€€€‘•Í¥¸€ô¹À¹½±Õµ¹}ÍÑ…¬ ¡±½¥ÑÌ°¹À¹½¹•Ì¡±•¸¡±½¥ÑÌ¤¤¤¤(€€€Á…É…µÌ€ô¹À¹…ÉÉ…ä¡lÄ¸À°€À¸Át°‘ÑåÁ”õ¹À¹™±½…ÐØÐ¤(€€€½‰Í•ÉÙ…Ñ¥½¹}Ý•¥¡ÑÌ€ô}Ù…±¥‘…Ñ•‘}Í…µÁ±•}Ý•¥¡ÑÌ¡Í…µÁ±•}Ý•¥¡ÑÌ°±•¸¡Ñ…É•Ð¤¤(€€€½‰Í•ÉÙ…Ñ¥½¹}Ý•¥¡ÑÌ€¼ô½‰Í•ÉÙ…Ñ¥½¹}Ý•¥¡ÑÌ¹ÍÕ´ ¤(€€€™½È|¥¸É…¹”¡¥Ñ•É…Ñ¥½¹Ì¤è(€€€€€€€ÁÉ½‰…‰¥±¥Ñä€ô¹À¹…Í…ÉÉ…ä¡m}Í¥µ½¥¡™±½…Ð¡à¤¤™½Èà¥¸‘•Í¥¸ Á…É…µÍt¤(€€€€€€€É…‘¥•¹Ð€ô‘•Í¥¸¹P €¡½‰Í•ÉÙ…Ñ¥½¹}Ý•¥¡ÑÌ€¨€¡ÁÉ½‰…‰¥±¥Ñä€´Ñ…É•Ð¤¤(€€€€€€€Á…É…µÌ€´ô±•…É¹¥¹}É…Ñ”€¨É…‘¥•¹Ð(€€€€€€€¥˜¹½Ð¹À¹…±°¡¹À¹¥Í™¥¹¥Ñ”¡Á…É…µÌ¤¤è(€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰A±…ÑÐ…±¥‰É…Ñ¥½¸‘¥Ù•É•ìÉ•©•ÐÑ¡¥Ì…±¥‰É…Ñ¥½¸‰±½¬ˆ¤(€€€É•ÑÕÉ¸™±½…Ð¡Á…É…µÍlÁt¤°™±½…Ð¡Á…É…µÍlÅt¤(()‘•˜}Ù…±¥‘…Ñ•‘}Í…µÁ±•}Ý•¥¡ÑÌ (€€€Í…µÁ±•}Ý•¥¡ÑÌèM•ÅÕ•¹•m™±½…Ñtð¹À¹¹‘…ÉÉ…äð9½¹”°½Õ¹Ðè¥¹Ð°(¤€´ø¹À¹¹‘…ÉÉ…äè(€€€¥˜Í…µÁ±•}Ý•¥¡ÑÌ¥Ì9½¹”è(€€€€€€€É•ÑÕÉ¸¹À¹½¹•Ì¡½Õ¹Ð°‘ÑåÁ”õ¹À¹™±½…ÐØÐ¤(€€€É•ÍÕ±Ð€ô¹À¹…Í…ÉÉ…ä¡Í…µÁ±•}Ý•¥¡ÑÌ°‘ÑåÁ”õ¹À¹™±½…ÐØÐ¤(€€€¥˜€¡É•ÍÕ±Ð¹Í¡…Á”€„ô€¡½Õ¹Ð°¤½È¹½Ð¹À¹¥Í™¥¹¥Ñ”¡É•ÍÕ±Ð¤¹…±° ¤(€€€€€€€€€€€½È¹À¹…¹ä¡É•ÍÕ±Ð€ðô€À¤¤è(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰Í…µÁ±”Ý•¥¡ÑÌµÕÍÐ‰”™¥¹¥Ñ”°Á½Í¥Ñ¥Ù”°…¹É½Üµ…±¥¹•ˆ¤(€€€É•ÑÕÉ¸É•ÍÕ±Ð¹½Áä ¤(()‘•˜}Í¥µ½¥¡Ù…±Õ”è™±½…Ð¤€´ø™±½…Ðè(€€€¥˜Ù…±Õ”€øô€Àè(€€€€€€€É•ÑÕÉ¸€Ä¸À€¼€ Ä¸À€¬µ…Ñ ¹•áÀ µµ¥¸¡Ù…±Õ”°€ÜÀÀ¸À¤¤¤(€€€•áÁ}Ù…±Õ”€ôµ…Ñ ¹•áÀ¡µ…à¡Ù…±Õ”°€´ÜÀÀ¸À¤¤(€€€É•ÑÕÉ¸•áÁ}Ù…±Õ”€¼€ Ä¸À€¬•áÁ}Ù…±Õ”¤(()‘•˜}É•…‘½¹±ä¡…ÉÉ…äè¹À¹¹‘…ÉÉ…ä¤€´ø¹À¹¹‘…ÉÉ…äè(€€€É•ÍÕ±Ð€ô¹À¹…ÉÉ…ä¡…ÉÉ…ä°‘ÑåÁ”õ¹À¹™±½…ÐØÐ°½ÁäõQÉÕ”¤(€€€É•ÍÕ±Ð¹Í•Ñ™±…Ì¡ÝÉ¥Ñ”õ…±Í”¤(€€€É•ÑÕÉ¸É•ÍÕ±Ð