"""Prediction-time diagnostics kept separate from realized screen outcomes.

Nothing in this module fits a model or selects candidates. It preserves the
full event denominator, frozen gate results, support-range violations, and
descriptive score distributions for an audit report.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal, Mapping, Sequence

from .candidate_events import CandidateFamily, OpportunityEvent
from .opportunity_models import NetValueEstimate


SplitName = Literal["train_fit", "train_calibration", "train_inner_screen"]
MetricName = Literal[
    "probability_positive",
    "expected_net_increment_bps",
    "global_residual_floor_proxy_bps",
    "epistemic_uncertainty_bps",
]


@dataclass(frozen=True, slots=True)
class CandidatePredictionDiagnostic:
    event_id: str
    block_id: str
    split: SplitName
    decision_ts_ns: int
    family: CandidateFamily
    side: Literal[-1, 1]
    cost_scenario_id: str | None
    in_fit_support: bool | None
    probability_positive: float | None
    expected_net_increment_bps: float | None
    global_residual_floor_proxy_bps: float | None
    epistemic_uncertainty_bps: float | None
    pass_probability: bool | None
    pass_expected_net: bool | None
    pass_global_residual_floor_proxy: bool | None
    pass_all_four: bool | None
    first_rejection_code: str


@dataclass(frozen=True, slots=True)
class ScoreDistribution:
    group_id: str
    split: SplitName
    family: CandidateFamily | None
    side: Literal[-1, 1] | None
    cost_scenario_id: str | None
    metric: MetricName
    count: int
    minimum: float
    p05: float
    p10: float
    p25: float
    median: float
    p75: float
    p90: float
    p95: float
    maximum: float


@dataclass(frozen=True, slots=True)
class SupportFeatureDiagnostic:
    feature_name: str
    fit_minimum: float
    fit_maximum: float
    candidate_count: int
    out_of_support_count: int


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    bin_index: int
    count: int
    mean_predicted_probability: float | None
    observed_positive_rate: float | None


@dataclass(frozen=True, slots=True)
class OpportunityPredictionReport:
    candidate_count: int
    prediction_count: int
    missing_prediction_ids: tuple[str, ...]
    rows: tuple[CandidatePredictionDiagnostic, ...]
    score_distributions: tuple[ScoreDistribution, ...]
    support_features: tuple[SupportFeatureDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class ScreenOutcomeAudit:
    """Outcome-side audit; never an input to prediction or gate selection."""

    candidate_count: int
    complete_count: int
    censored_count: int
    missing_outcome_count: int
    selected_complete_count: int
    rejected_complete_count: int
    all_complete_net_bps: tuple[float, ...]
    selected_complete_net_bps: tuple[float, ...]
    rejected_complete_net_bps: tuple[float, ...]
    brier_score: float | None
    log_loss: float | None
    calibration_bins: tuple[CalibrationBin, ...]
    calibration_residual_p10_offset_bps: float | None
    global_residual_floor_proxy_identity_max_abs_error_bps: float | None
    actual_below_global_residual_floor_proxy_rate: float | None


@dataclass(frozen=True, slots=True)
class DiagnosticAblationSet:
    ablation_id: Literal["F0", "F1", "F2", "F3", "F4", "F5"]
    selected_ids: tuple[str, ...]
    evidence_status: Literal["DIAGNOSTIC_REUSE_NOT_OOS"] = "DIAGNOSTIC_REUSE_NOT_OOS"


@dataclass(frozen=True, slots=True)
class DiagnosticAblationOutcomeAudit:
    """Post-prediction outcomes for a frozen diagnostic-only candidate set."""

    ablation_id: Literal["F0", "F1", "F2", "F3", "F4", "F5"]
    evidence_status: Literal["DIAGNOSTIC_REUSE_NOT_OOS"]
    prediction_cost_scenario_id: str | None
    outcome_cost_scenario_id: str | None
    candidate_count: int
    selected_count: int
    complete_count: int
    censored_count: int
    missing_outcome_count: int
    selected_complete_count: int
    selected_censored_count: int
    selected_missing_outcome_count: int
    rejected_complete_count: int
    all_complete_net_bps: tuple[float, ...]
    selected_complete_net_bps: tuple[float, ...]
    rejected_complete_net_bps: tuple[float, ...]


_METRICS: tuple[MetricName, ...] = (
    "probability_positive",
    "expected_net_increment_bps",
    "global_residual_floor_proxy_bps",
    "epistemic_uncertainty_bps",
)


def _finite(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(
    values: Sequence[float], *, group_id: str, split: SplitName,
    family: CandidateFamily | None, side: Literal[-1, 1] | None,
    cost_scenario_id: str | None, metric: MetricName,
) -> ScoreDistribution:
    return ScoreDistribution(
        group_id=group_id,
        split=split,
        family=family,
        side=side,
        cost_scenario_id=cost_scenario_id,
        metric=metric,
        count=len(values),
        minimum=min(values),
        p05=_quantile(values, 0.05),
        p10=_quantile(values, 0.10),
        p25=_quantile(values, 0.25),
        median=_quantile(values, 0.50),
        p75=_quantile(values, 0.75),
        p90=_quantile(values, 0.90),
        p95=_quantile(values, 0.95),
        maximum=max(values),
    )


def build_opportunity_prediction_report(
    events: Sequence[OpportunityEvent],
    estimates: Mapping[str, NetValueEstimate],
    block_by_event_id: Mapping[str, str],
    split_by_event_id: Mapping[str, SplitName],
    *,
    feature_rows: Mapping[str, Mapping[str, float]] | None = None,
    fit_extrema: Mapping[str, tuple[float, float]] | None = None,
) -> OpportunityPredictionReport:
    """Build gate/score diagnostics without any realized outcome input.

    Events are the authoritative denominator. Missing or invalid predictions
    remain visible with null scores and an explicit rejection code.
    """
    rows = tuple(events)
    event_by_id = {event.event_id: event for event in rows}
    ids = set(event_by_id)
    if len(event_by_id) != len(rows):
        raise ValueError("candidate diagnostic event ids must be unique")
    if set(block_by_event_id) != ids or set(split_by_event_id) != ids:
        raise ValueError("candidate diagnostic metadata must cover the full event denominator")
    if set(estimates) - ids:
        raise ValueError("prediction report contains an unknown candidate id")
    if any(not isinstance(block_by_event_id[event_id], str)
           or not block_by_event_id[event_id].strip() for event_id in ids):
        raise ValueError("every candidate requires a block id")
    if any(split_by_event_id[event_id] not in {
            "train_fit", "train_calibration", "train_inner_screen"} for event_id in ids):
        raise ValueError("candidate diagnostic split is invalid")

    for event_id, prediction in estimates.items():
        event = event_by_id[event_id]
        if (prediction.event_id != event_id or prediction.fold_id != event.fold_id
                or prediction.source_identity != event.source_identity
                or prediction.rule_version != event.rule_version
                or prediction.feature_spec_hash != event.feature_spec_hash):
            raise ValueError("prediction provenance differs from its candidate event")
        if type(prediction.in_support) is not bool:
            raise ValueError("prediction in-support status must be an explicit boolean")
        if (prediction.economic_status != "SCENARIO_ONLY"
                or prediction.evidence_type != "MODEL_COUNTERFACTUAL"):
            raise ValueError("opportunity prediction diagnostics require scenario counterfactual estimates")
        probability = _finite(prediction.probability_positive, "probability_positive")
        if not 0 <= probability <= 1:
            raise ValueError("probability_positive must lie in [0, 1]")
        for metric in _METRICS[1:]:
            _finite(getattr(prediction, metric), metric)

    feature_diagnostics: list[SupportFeatureDiagnostic] = []
    if (feature_rows is None) != (fit_extrema is None):
        raise ValueError("feature rows and fit extrema must be supplied together")
    if feature_rows is not None and fit_extrema is not None:
        if set(feature_rows) != ids or not fit_extrema:
            raise ValueError("support audit requires every candidate and nonempty fit extrema")
        feature_names = set(fit_extrema)
        if any(set(feature_rows[event_id]) != feature_names for event_id in ids):
            raise ValueError("candidate feature rows must match the fit feature schema")
        outside_by_id = {event_id: False for event_id in ids}
        for name, bounds in sorted(fit_extrema.items()):
            if len(bounds) != 2:
                raise ValueError("fit feature extrema must be min/max pairs")
            low = _finite(bounds[0], f"fit.{name}.min")
            high = _finite(bounds[1], f"fit.{name}.max")
            if low > high:
                raise ValueError("fit feature minimum exceeds maximum")
            out_count = 0
            for event_id in ids:
                value = _finite(feature_rows[event_id][name], f"feature.{name}")
                outside = value < low or value > high
                out_count += int(outside)
                outside_by_id[event_id] = outside_by_id[event_id] or outside
            feature_diagnostics.append(SupportFeatureDiagnostic(
                feature_name=name,
                fit_minimum=low,
                fit_maximum=high,
                candidate_count=len(ids),
                out_of_support_count=out_count,
            ))
        for event_id, support_estimate in estimates.items():
            if support_estimate.in_support != (not outside_by_id[event_id]):
                raise ValueError("model in-support flag differs from fit-extrema audit")

    diagnostic_rows: list[CandidatePredictionDiagnostic] = []
    score_groups: dict[tuple[str, SplitName, CandidateFamily, Literal[-1, 1], str],
                       dict[MetricName, list[float]]] = {}
    for event_id in sorted(ids):
        event = event_by_id[event_id]
        split = split_by_event_id[event_id]
        estimate = estimates.get(event_id)
        if estimate is None:
            diagnostic_rows.append(CandidatePredictionDiagnostic(
                event_id=event_id,
                block_id=block_by_event_id[event_id],
                split=split,
                decision_ts_ns=event.decision_ts_ns,
                family=event.family,
                side=event.side,
                cost_scenario_id=None,
                in_fit_support=None,
                probability_positive=None,
                expected_net_increment_bps=None,
                global_residual_floor_proxy_bps=None,
                epistemic_uncertainty_bps=None,
                pass_probability=None,
                pass_expected_net=None,
                pass_global_residual_floor_proxy=None,
                pass_all_four=None,
                first_rejection_code="PREDICTION_MISSING",
            ))
            continue
        values = {
            "probability_positive": float(estimate.probability_positive),
            "expected_net_increment_bps": float(estimate.expected_net_increment_bps),
            "global_residual_floor_proxy_bps": float(
                estimate.global_residual_floor_proxy_bps
            ),
            "epistemic_uncertainty_bps": float(estimate.epistemic_uncertainty_bps),
        }
        pass_probability = values["probability_positive"] > 0.5
        pass_mean = values["expected_net_increment_bps"] > 0.0
        pass_floor_proxy = values["global_residual_floor_proxy_bps"] > 0.0
        passed = (estimate.in_support, pass_probability, pass_mean, pass_floor_proxy)
        first_rejection = next((code for code, passed_gate in zip((
            "OUT_OF_FIT_SUPPORT",
            "CALIBRATED_PROBABILITY_NOT_ABOVE_0_5",
            "EXPECTED_NET_INCREMENT_NOT_POSITIVE",
            "GLOBAL_RESIDUAL_FLOOR_PROXY_NOT_POSITIVE",
        ), passed, strict=True) if not passed_gate), "SELECTED")
        scenario = estimate.cost_scenario_id
        diagnostic_rows.append(CandidatePredictionDiagnostic(
            event_id=event_id,
            block_id=block_by_event_id[event_id],
            split=split,
            decision_ts_ns=event.decision_ts_ns,
            family=event.family,
            side=event.side,
            cost_scenario_id=scenario,
            in_fit_support=estimate.in_support,
            probability_positive=values["probability_positive"],
            expected_net_increment_bps=values["expected_net_increment_bps"],
            global_residual_floor_proxy_bps=values["global_residual_floor_proxy_bps"],
            epistemic_uncertainty_bps=values["epistemic_uncertainty_bps"],
            pass_probability=pass_probability,
            pass_expected_net=pass_mean,
            pass_global_residual_floor_proxy=pass_floor_proxy,
            pass_all_four=all(passed),
            first_rejection_code=first_rejection,
        ))
        key = (block_by_event_id[event_id], split, event.family, event.side, scenario)
        grouped_values = score_groups.setdefault(key, {metric: [] for metric in _METRICS})
        for metric in _METRICS:
            grouped_values[metric].append(values[metric])

    distributions = tuple(
        _distribution(values[metric], group_id=key[0], split=key[1],
                      family=key[2], side=key[3], cost_scenario_id=key[4], metric=metric)
        for key, metrics in sorted(score_groups.items())
        for metric in _METRICS
        for values in (metrics,)
    )
    return OpportunityPredictionReport(
        candidate_count=len(ids),
        prediction_count=len(estimates),
        missing_prediction_ids=tuple(sorted(ids - set(estimates))),
        rows=tuple(diagnostic_rows),
        score_distributions=distributions,
        support_features=tuple(feature_diagnostics),
    )


def audit_screen_outcomes_after_prediction(
    report: OpportunityPredictionReport,
    outcomes_by_id: Mapping[str, object],
    *,
    calibration_residual_p10_offset_bps: float,
) -> ScreenOutcomeAudit:
    """Audit the global residual floor proxy after prediction; never feeds gates.

    The proxy is expected value plus the calibration residual's marginal P10
    offset. Its below-rate is descriptive on this reused Train-inner screen;
    it is not conditional-quantile coverage or independent qualification evidence.
    """
    from .opportunity_models import OpportunityOutcome

    residual = _finite(
        calibration_residual_p10_offset_bps,
        "calibration_residual_p10_offset_bps",
    )
    row_by_id = {row.event_id: row for row in report.rows}
    if any(row.split != "train_inner_screen" for row in report.rows):
        raise PermissionError("outcome calibration audit requires an exclusively Train-inner screen report")
    if set(outcomes_by_id) - set(row_by_id):
        raise ValueError("screen outcome audit contains an unknown candidate")
    complete: list[tuple[CandidatePredictionDiagnostic, float]] = []
    censored_count = 0
    for event_id in sorted(outcomes_by_id):
        outcome = outcomes_by_id[event_id]
        if not isinstance(outcome, OpportunityOutcome):
            raise TypeError("screen outcome audit requires OpportunityOutcome records")
        row = row_by_id[event_id]
        if outcome.event_id != event_id or outcome.decision_ts_ns != row.decision_ts_ns:
            raise ValueError("screen outcome identity differs from the candidate report")
        if outcome.candidate_family != row.family or outcome.candidate_side != row.side:
            raise ValueError("screen outcome direction/family differs from its candidate")
        if row.split != "train_inner_screen" or outcome.split != "train_inner_screen":
            raise PermissionError("post-hoc screen diagnostics accept only Train-inner screen outcomes")
        if (row.cost_scenario_id is not None
                and outcome.cost_scenario_id != row.cost_scenario_id):
            raise ValueError("screen outcome and prediction cost scenarios differ")
        if outcome.net_return_bps is None:
            if outcome.censor_ts_ns is None:
                raise ValueError("null screen outcome must explicitly carry a censor clock")
            censored_count += 1
        else:
            complete.append((row, _finite(outcome.net_return_bps, "outcome.net_return_bps")))

    scored = [(row, actual, row.probability_positive) for row, actual in complete
              if row.probability_positive is not None]
    brier: float | None
    logloss: float | None
    below_proxy_rate: float | None
    max_identity_error: float | None
    calibration_bins: tuple[CalibrationBin, ...]
    if scored:
        brier = math.fsum((float(probability) - float(actual > 0)) ** 2
                          for _row, actual, probability in scored) / len(scored)
        epsilon = 1e-15
        logloss = -math.fsum(
            (int(actual > 0) * math.log(min(1 - epsilon, max(epsilon, float(probability))))
             + int(actual <= 0) * math.log1p(-min(1 - epsilon,
                                                  max(epsilon, float(probability)))))
            for _row, actual, probability in scored
        ) / len(scored)
        bins: list[CalibrationBin] = []
        for bin_index in range(10):
            members = [
                (float(probability), int(actual > 0))
                for _row, actual, probability in scored
                if min(9, int(float(probability) * 10)) == bin_index
            ]
            bins.append(CalibrationBin(
                bin_index=bin_index,
                count=len(members),
                mean_predicted_probability=(
                    math.fsum(probability for probability, _ in members) / len(members)
                    if members else None
                ),
                observed_positive_rate=(
                    math.fsum(float(positive) for _, positive in members) / len(members)
                    if members else None
                ),
            ))
        calibration_bins = tuple(bins)
        proxy_errors = [
            actual - (float(row.expected_net_increment_bps) + residual)
            for row, actual, _probability in scored
            if row.expected_net_increment_bps is not None
        ]
        proxy_identity_errors = [
            abs(float(row.global_residual_floor_proxy_bps) -
                (float(row.expected_net_increment_bps) + residual))
            for row, _actual, _probability in scored
            if (row.global_residual_floor_proxy_bps is not None
                and row.expected_net_increment_bps is not None)
        ]
        below_proxy_rate = (sum(error < 0 for error in proxy_errors) / len(proxy_errors)
                            if proxy_errors else None)
        max_identity_error = max(proxy_identity_errors) if proxy_identity_errors else None
    else:
        brier = logloss = below_proxy_rate = max_identity_error = None
        calibration_bins = tuple(CalibrationBin(index, 0, None, None) for index in range(10))

    selected_values = tuple(actual for row, actual in complete if row.pass_all_four is True)
    rejected_values = tuple(actual for row, actual in complete if row.pass_all_four is False)
    return ScreenOutcomeAudit(
        candidate_count=report.candidate_count,
        complete_count=len(complete),
        censored_count=censored_count,
        missing_outcome_count=report.candidate_count - len(outcomes_by_id),
        selected_complete_count=len(selected_values),
        rejected_complete_count=len(rejected_values),
        all_complete_net_bps=tuple(actual for _, actual in complete),
        selected_complete_net_bps=selected_values,
        rejected_complete_net_bps=rejected_values,
        brier_score=brier,
        log_loss=logloss,
        calibration_bins=calibration_bins,
        calibration_residual_p10_offset_bps=residual if scored else None,
        global_residual_floor_proxy_identity_max_abs_error_bps=max_identity_error,
        actual_below_global_residual_floor_proxy_rate=below_proxy_rate,
    )


def audit_screen_ablation_outcomes_after_prediction(
    report: OpportunityPredictionReport,
    outcomes_by_id: Mapping[str, object],
    ablations: Sequence[DiagnosticAblationSet],
    *,
    expected_outcome_cost_scenario_id: str | None = None,
) -> tuple[DiagnosticAblationOutcomeAudit, ...]:
    """Audit F0-F5 outcomes after selection, optionally for a paired cost scenario.

    The prediction report and frozen ablation IDs determine membership before
    outcomes are inspected. Supplying a different outcome cost scenario is an
    explicit paired Base/Stress audit; without it, prediction and outcome costs
    must match exactly.
    """
    from .opportunity_models import OpportunityOutcome

    rows = {row.event_id: row for row in report.rows}
    if len(rows) != len(report.rows) or report.candidate_count != len(rows):
        raise ValueError("screen ablation report event ids must be unique")
    if any(row.split != "train_inner_screen" for row in report.rows):
        raise PermissionError("ablation outcome audit requires an exclusively Train-inner screen")
    if set(outcomes_by_id) - set(rows):
        raise ValueError("screen ablation audit contains an unknown candidate")
    if (expected_outcome_cost_scenario_id is not None
            and (not isinstance(expected_outcome_cost_scenario_id, str)
                 or not expected_outcome_cost_scenario_id.strip())):
        raise ValueError("paired outcome cost scenario must be a nonempty identity")
    expected_ids = ("F0", "F1", "F2", "F3", "F4", "F5")
    if tuple(ablation.ablation_id for ablation in ablations) != expected_ids:
        raise ValueError("diagnostic outcome audit requires the ordered frozen F0-F5 sets")
    if any(ablation.evidence_status != "DIAGNOSTIC_REUSE_NOT_OOS" for ablation in ablations):
        raise ValueError("screen ablation evidence must remain marked DIAGNOSTIC_REUSE_NOT_OOS")
    prediction_costs = {row.cost_scenario_id for row in report.rows
                        if row.cost_scenario_id is not None}
    if len(prediction_costs) > 1:
        raise ValueError("one screen prediction report cannot mix cost scenarios")
    prediction_cost = next(iter(prediction_costs), None)

    complete: dict[str, float] = {}
    censored: set[str] = set()
    outcome_costs: set[str] = set()
    for event_id in sorted(outcomes_by_id):
        outcome = outcomes_by_id[event_id]
        row = rows[event_id]
        if not isinstance(outcome, OpportunityOutcome):
            raise TypeError("screen ablation audit requires OpportunityOutcome records")
        if (outcome.event_id != event_id or outcome.decision_ts_ns != row.decision_ts_ns
                or outcome.candidate_family != row.family or outcome.candidate_side != row.side):
            raise ValueError("screen ablation outcome identity differs from its prediction row")
        if outcome.split != "train_inner_screen":
            raise PermissionError("screen ablation outcomes accept only Train-inner screen rows")
        if expected_outcome_cost_scenario_id is None:
            if (row.cost_scenario_id is not None
                    and outcome.cost_scenario_id != row.cost_scenario_id):
                raise ValueError("paired cost scenario must be explicitly identified")
        elif outcome.cost_scenario_id != expected_outcome_cost_scenario_id:
            raise ValueError("screen outcome differs from the requested paired cost scenario")
        outcome_costs.add(outcome.cost_scenario_id)
        if outcome.net_return_bps is None:
            if outcome.censor_ts_ns is None:
                raise ValueError("censored screen outcome must retain its censor clock")
            censored.add(event_id)
        else:
            complete[event_id] = _finite(outcome.net_return_bps, "outcome.net_return_bps")
    if len(outcome_costs) > 1:
        raise ValueError("one screen ablation audit cannot mix outcome cost scenarios")
    outcome_cost = next(iter(outcome_costs), expected_outcome_cost_scenario_id)

    audits: list[DiagnosticAblationOutcomeAudit] = []
    for ablation in ablations:
        selected_ids = set(ablation.selected_ids)
        if len(selected_ids) != len(ablation.selected_ids) or not selected_ids <= set(rows):
            raise ValueError("diagnostic ablation contains duplicate or unknown candidates")
        selected_complete = tuple(
            complete[event_id] for event_id in sorted(selected_ids & complete.keys())
        )
        rejected_complete = tuple(
            complete[event_id] for event_id in sorted(set(complete) - selected_ids)
        )
        audits.append(DiagnosticAblationOutcomeAudit(
            ablation_id=ablation.ablation_id,
            evidence_status=ablation.evidence_status,
            prediction_cost_scenario_id=prediction_cost,
            outcome_cost_scenario_id=outcome_cost,
            candidate_count=len(rows),
            selected_count=len(selected_ids),
            complete_count=len(complete),
            censored_count=len(censored),
            missing_outcome_count=len(rows) - len(outcomes_by_id),
            selected_complete_count=len(selected_complete),
            selected_censored_count=len(selected_ids & censored),
            selected_missing_outcome_count=len(selected_ids - set(outcomes_by_id)),
            rejected_complete_count=len(rejected_complete),
            all_complete_net_bps=tuple(complete[event_id] for event_id in sorted(complete)),
            selected_complete_net_bps=selected_complete,
            rejected_complete_net_bps=rejected_complete,
        ))
    return tuple(audits)


def build_diagnostic_ablation_sets(
    report: OpportunityPredictionReport,
    *,
    negative_control_count: int,
    frozen_seed: int,
) -> tuple[DiagnosticAblationSet, ...]:
    """Build F0-F5 diagnostic-only sets without reading any outcome field."""
    if type(negative_control_count) is not int or not 0 <= negative_control_count <= report.candidate_count:
        raise ValueError("negative-control count must be within the frozen candidate denominator")
    if type(frozen_seed) is not int:
        raise ValueError("negative-control seed must be an explicit frozen integer")
    rows = report.rows
    ids = tuple(sorted(row.event_id for row in rows))
    by_id = {row.event_id: row for row in rows}
    f0 = tuple(event_id for event_id in ids if by_id[event_id].pass_all_four is True)
    f1_ids: list[str] = []
    for event_id in ids:
        row = by_id[event_id]
        expected = row.expected_net_increment_bps
        if row.in_fit_support is True and expected is not None and expected > 0.0:
            f1_ids.append(event_id)
    f1 = tuple(f1_ids)
    f2 = tuple(event_id for event_id in f1
               if by_id[event_id].pass_probability is True)
    f3 = tuple(event_id for event_id in f1
               if by_id[event_id].pass_global_residual_floor_proxy is True)
    randomized = list(ids)
    random.Random(frozen_seed).shuffle(randomized)
    f5 = tuple(randomized[:negative_control_count])
    return (
        DiagnosticAblationSet("F0", f0),
        DiagnosticAblationSet("F1", f1),
        DiagnosticAblationSet("F2", f2),
        DiagnosticAblationSet("F3", f3),
        DiagnosticAblationSet("F4", ids),
        DiagnosticAblationSet("F5", f5),
    )
