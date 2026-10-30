"""Small Train-fit/calibration opportunity baseline; never a native fill."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from .candidate_event_union import CandidateGroup
from .conditional_outcome_labels import GroupOutcome
from .opportunity_models import (_fit_logistic, _fit_platt, _sigmoid,
                                overlap_uniqueness_weights,
                                uniqueness_effective_label_count)


def missing_candidate_features(observation, names: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for name in names
                 if (name.startswith("market.") and name[7:] not in observation.market)
                 or (name.startswith("account.") and name[8:] not in observation.account))


def _features(group: CandidateGroup, names: Sequence[str], cost_bps: float | None = None) -> np.ndarray:
    obs = group.observation
    values = []
    for name in names:
        if name.startswith("market."):
            values.append(obs.market_value(name[7:]))
        elif name.startswith("account."):
            values.append(obs.account_value(name[8:]))
        elif name == "consensus.count":
            values.append(float(len(group.teachers)))
        elif name.startswith("teacher.") and len(name) > len("teacher."):
            values.append(float(name[8:] in {teacher for teacher, _ in group.teachers}))
        elif name == "cost.round_trip_bps":
            if cost_bps is None:
                raise ValueError("the configured cost feature requires a frozen cost")
            values.append(cost_bps)
        else:
            raise ValueError("only configured causal market/account/teacher/consensus/cost features are permitted")
    return np.asarray([*values, float(group.side)], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class EntryScore:
    action: str
    probability_positive: float
    expected_net_bps: float
    in_support: bool
    reason: str
    unsupported_features: tuple[str, ...] = ()
    calibration_mae_bps: float = 0.0


@dataclass(frozen=True, slots=True)
class FrozenEntryMetaPolicy:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    logistic: np.ndarray
    calibrator: tuple[float, float]
    value_weights: np.ndarray
    source_id: str
    fold_id: str
    feature_spec_hash: str
    label_recipe: tuple[str, str, str]
    available_after_ns: int
    margin_bps: float
    cost_bps: float | None = None
    calibration_mae_bps: float = 0.0
    return_basis_id: str = "LOG_REFERENCE_ENTRY_NOTIONAL_BPS"
    calibration_brier: float = 0.0
    fit_count: int = 0
    calibration_count: int = 0
    fee_bps_per_leg: float | None = None
    slippage_bps_per_leg: float | None = None
    fit_effective_label_count: float = 0.0
    calibration_effective_label_count: float = 0.0
    fit_weighting_id: str = "AVERAGE_INVERSE_LABEL_CONCURRENCY_V1"

    def score(self, group: CandidateGroup) -> EntryScore:
        obs = group.observation
        if (obs.source_id, obs.fold_id, obs.feature_spec_hash) != (
                self.source_id, self.fold_id, self.feature_spec_hash):
            raise ValueError("entry model source/fold/schema mismatch")
        if obs.decision_ts_ns <= self.available_after_ns:
            raise ValueError("entry model cannot predict before calibration labels are available")
        x = _features(group, self.feature_names, self.cost_bps)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite entry feature")
        z = np.concatenate(([1.], (x - self.mean) / self.scale))
        probability = _sigmoid(self.calibrator[0] * float(z @ self.logistic) + self.calibrator[1])
        expected = float(z @ self.value_weights)
        if not isfinite(probability) or not isfinite(expected):
            raise ValueError("nonfinite opportunity prediction")
        support = bool(np.all(x >= self.lower) and np.all(x <= self.upper))
        unsupported = tuple(name for name, value, lo, hi in
                            zip((*self.feature_names, "side"), x, self.lower, self.upper)
                            if value < lo or value > hi)
        if obs.pending_order_ids:
            return EntryScore("NO_TRADE", probability, expected, support, "PENDING_ORDER",
                              unsupported, self.calibration_mae_bps)
        if abs(obs.account_value("net_exposure")) > 1e-8:
            return EntryScore("NO_TRADE", probability, expected, support, "POSITION_OCCUPIED",
                              unsupported, self.calibration_mae_bps)
        if not support:
            return EntryScore("NO_TRADE", probability, expected, support, "OUT_OF_SUPPORT",
                              unsupported, self.calibration_mae_bps)
        if expected <= self.margin_bps:
            return EntryScore("NO_TRADE", probability, expected, support, "NET_EDGE_TOO_SMALL",
                              unsupported, self.calibration_mae_bps)
        if expected - self.calibration_mae_bps <= self.margin_bps:
            return EntryScore("NO_TRADE", probability, expected, support,
                              "CALIBRATION_ERROR_EXCEEDS_EDGE", unsupported,
                              self.calibration_mae_bps)
        return EntryScore("LONG" if group.side > 0 else "SHORT",
                          probability, expected, support, "POSITIVE_NET_EDGE",
                          calibration_mae_bps=self.calibration_mae_bps)


def fit_entry_meta_policy(
    fit: Sequence[GroupOutcome], calibration: Sequence[GroupOutcome],
    groups: Mapping[str, CandidateGroup], *, feature_names: Sequence[str],
    margin_bps: float, l2: float = 1.0, cost_bps: float | None = None,
    fee_bps_per_leg: float | None = None, slippage_bps_per_leg: float | None = None,
) -> FrozenEntryMetaPolicy:
    """Fit complete labels for executable flat candidates, then calibrate later."""
    names = tuple(feature_names)
    if (not names or len(set(names)) != len(names)
            or not isfinite(margin_bps) or margin_bps < 0 or not isfinite(l2) or l2 <= 0):
        raise ValueError("fixed features, nonnegative margin and positive regularization required")
    if cost_bps is not None and (isinstance(cost_bps, bool) or not isinstance(cost_bps, (int, float))
                                 or not isfinite(cost_bps) or cost_bps < 0):
        raise ValueError("cost must be finite and nonnegative")
    if (fee_bps_per_leg is None) != (slippage_bps_per_leg is None):
        raise ValueError("fee and slippage must be specified together")
    if fee_bps_per_leg is not None:
        assert slippage_bps_per_leg is not None
        if (any(isinstance(x, bool) or not isinstance(x, (int, float))
                or not isfinite(x) or x < 0 for x in (fee_bps_per_leg, slippage_bps_per_leg))
                or cost_bps is None
                or abs(cost_bps - 2 * (fee_bps_per_leg + slippage_bps_per_leg)) > 1e-9):
            raise ValueError("round-trip cost must match the frozen per-leg costs")
    all_labels = tuple(fit) + tuple(calibration)
    if len(fit) < 4 or len(calibration) < 4:
        raise ValueError("fit and later calibration need at least four complete samples")
    if (len({x.event_id for x in all_labels}) != len(all_labels)
            or set(groups) != {x.event_id for x in all_labels}):
        raise ValueError("fit/calibration groups must cover unique events exactly")
    if any(x.evidence_type != "MODEL_COUNTERFACTUAL" or x.economic_status != "SCENARIO_ONLY"
           or x.net_return_bps is None or x.label_end_ts_ns is None
           or x.outcome_status != "COMPLETE" or x.censor_ts_ns is not None for x in all_labels):
        raise ValueError("complete scenario-only research labels required")
    if any(groups[x.event_id].observation.pending_order_ids
           or abs(groups[x.event_id].observation.account_value("net_exposure")) > 1e-8
           for x in all_labels):
        raise ValueError("entry model fit/calibration accepts only executable flat candidates")
    first_cal = min(x.decision_ts_ns for x in calibration)
    if any(x.label_end_ts_ns is None or x.label_end_ts_ns >= first_cal for x in fit):
        raise ValueError("fit labels overlap calibration")
    if any(x.label_end_ts_ns is None or x.label_end_ts_ns <= x.decision_ts_ns for x in all_labels):
        raise ValueError("labels must end after the decision")
    bases = {x.return_basis_id for x in all_labels}
    if len(bases) != 1:
        raise ValueError("return bases cannot be mixed")
    recipes = {(x.exit_policy_id, x.execution_recipe, x.cost_scenario_id) for x in all_labels}
    if len(recipes) != 1:
        raise ValueError("frozen exit/execution/cost recipes cannot be mixed")
    identities = {(x.source_identity, x.fold_id) for x in all_labels}
    if len(identities) != 1:
        raise ValueError("source/fold identities cannot be mixed")
    if any((groups[x.event_id].observation.source_id, groups[x.event_id].observation.fold_id,
            groups[x.event_id].observation.decision_ts_ns) !=
           (x.source_identity, x.fold_id, x.decision_ts_ns) for x in all_labels):
        raise ValueError("label does not match its causal candidate")
    specs = {groups[x.event_id].observation.feature_spec_hash for x in all_labels}
    if len(specs) != 1:
        raise ValueError("candidate feature schema changed")
    train_x = np.stack([_features(groups[x.event_id], names, cost_bps) for x in fit])
    cal_x = np.stack([_features(groups[x.event_id], names, cost_bps) for x in calibration])
    train_y = np.asarray([x.net_return_bps for x in fit], dtype=np.float64)
    cal_y = np.asarray([x.net_return_bps for x in calibration], dtype=np.float64)
    if (not all(np.isfinite(x).all() for x in (train_x, cal_x, train_y, cal_y))
            or len(np.unique(train_y > 0)) != 2 or len(np.unique(cal_y > 0)) != 2):
        raise ValueError("finite fit/calibration examples with wins and losses required")
    fit_weights = overlap_uniqueness_weights(fit)
    calibration_weights = overlap_uniqueness_weights(calibration)
    mean = np.average(train_x, axis=0, weights=fit_weights)
    scale = np.sqrt(np.average((train_x - mean) ** 2, axis=0, weights=fit_weights))
    scale[scale < 1e-12] = 1.
    train_z = np.column_stack((np.ones(len(train_x)), (train_x - mean) / scale))
    cal_z = np.column_stack((np.ones(len(cal_x)), (cal_x - mean) / scale))
    logistic = _fit_logistic(train_z, (train_y > 0).astype(float), l2, 0.05, 2000,
                             sample_weights=fit_weights)
    calibrator = _fit_platt(cal_z @ logistic, (cal_y > 0).astype(float), 0.05, 2000,
                            sample_weights=calibration_weights)
    penalty = np.eye(train_z.shape[1]) * l2
    penalty[0, 0] = 0.
    weighted_train_z = train_z * fit_weights[:, None]
    value = np.linalg.solve(train_z.T @ weighted_train_z + penalty,
                            train_z.T @ (fit_weights * train_y))
    for array in (mean, scale, logistic, value):
        array.setflags(write=False)
    lower, upper = train_x.min(axis=0), train_x.max(axis=0)
    lower.setflags(write=False)
    upper.setflags(write=False)
    return FrozenEntryMetaPolicy(
        names, mean, scale, lower, upper, logistic, calibrator, value,
        all_labels[0].source_identity, all_labels[0].fold_id, next(iter(specs)),
        next(iter(recipes)), max(x.label_end_ts_ns or 0 for x in calibration), margin_bps,
        cost_bps,
        float(np.average(np.abs(cal_z @ value - cal_y), weights=calibration_weights)),
        next(iter(bases)),
        float(np.average((np.asarray([_sigmoid(calibrator[0] * float(x) + calibrator[1])
                                     for x in cal_z @ logistic]) - (cal_y > 0)) ** 2,
                         weights=calibration_weights)),
        len(fit), len(calibration), fee_bps_per_leg, slippage_bps_per_leg,
        uniqueness_effective_label_count(fit_weights),
        uniqueness_effective_label_count(calibration_weights),
        "AVERAGE_INVERSE_LABEL_CONCURRENCY_V1",
    )


def summarize_entry_scores(
    scores: Sequence[EntryScore], *, margin_bps: float | None = None,
    unscored_count: int = 0, unscored_reason_counts: Mapping[str, int] | None = None,
    native_selected_count: int | None = None,
    post_gate_override_counts: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Report the whole candidate denominator and actual sequential gate funnel."""
    from collections import Counter
    rows = tuple(scores)
    if type(unscored_count) is not int or unscored_count < 0:
        raise ValueError("unscored candidate count must be a nonnegative integer")
    if (native_selected_count is not None
            and (type(native_selected_count) is not int or native_selected_count < 0
                 or native_selected_count > len(rows) + unscored_count)):
        raise ValueError("native selected count must fit the complete candidate denominator")
    unscored_reasons = dict(unscored_reason_counts or {})
    if (any(not isinstance(name, str) or not name.strip()
            or type(count) is not int or count < 0
            for name, count in unscored_reasons.items())
            or (unscored_reasons and sum(unscored_reasons.values()) != unscored_count)):
        raise ValueError("unscored rejection reasons must cover the unscored candidate count")
    override_reasons = dict(post_gate_override_counts or {})
    accepted = sum(row.action in ("LONG", "SHORT") for row in rows)
    if (any(not isinstance(name, str) or not name.strip()
            or type(count) is not int or count < 0
            for name, count in override_reasons.items())
            or sum(override_reasons.values()) > accepted):
        raise ValueError("post-gate override counts must fit model-accepted candidates")
    post_gate_overrides = sum(override_reasons.values())
    native_eligible = accepted - post_gate_overrides
    if native_selected_count is not None and native_selected_count > native_eligible:
        raise ValueError("native selection cannot exceed post-override model accepts")
    reasons = Counter(row.reason for row in rows)
    unsupported = Counter(name for row in rows for name in row.unsupported_features)
    total = len(rows) + unscored_count
    def distribution(values):
        if not values:
            return None
        return {key: float(value) for key, value in zip(
            ("min", "p25", "median", "p75", "max"),
            np.quantile(values, (0, .25, .5, .75, 1)))}
    if margin_bps is not None and (not isfinite(margin_bps) or margin_bps < 0):
        raise ValueError("margin must be finite and nonnegative")
    gates = None if margin_bps is None else {
        "fixed_margin_bps": margin_bps,
        "support_pass": sum(row.in_support for row in rows),
        "edge_pass": sum(row.expected_net_bps > margin_bps for row in rows),
        "calibration_adjusted_edge_pass": sum(
            row.expected_net_bps - row.calibration_mae_bps > margin_bps for row in rows
        ),
        "support_and_edge_pass": sum(row.in_support and row.expected_net_bps > margin_bps for row in rows),
        "account_ineligible": sum(row.reason in ("PENDING_ORDER", "POSITION_OCCUPIED") for row in rows),
        "interpretation": "overlapping gate counts among scored candidates; not native fills",
    }
    funnel = None
    if margin_bps is not None:
        account_eligible = tuple(row for row in rows
                                 if row.reason not in ("PENDING_ORDER", "POSITION_OCCUPIED"))
        support_eligible = tuple(row for row in account_eligible if row.in_support)
        edge_eligible = tuple(row for row in support_eligible
                              if row.expected_net_bps > margin_bps)
        calibration_eligible = tuple(row for row in edge_eligible
                                     if row.expected_net_bps - row.calibration_mae_bps > margin_bps)
        stages = {
            "total_candidates": total,
            "scored_candidates": len(rows),
            "account_eligible": len(account_eligible),
            "in_support": len(support_eligible),
            "above_frozen_net_edge_margin": len(edge_eligible),
            "above_margin_after_calibration_error": len(calibration_eligible),
            "model_gate_accepted": accepted,
            "native_eligible_after_overrides": native_eligible,
        }
        if native_selected_count is not None:
            stages["native_selected_after_arbitration"] = native_selected_count
        funnel = {
            "gate_order": ("scored", "account", "support", "net_edge",
                           "calibration_error", "model_gate", "native_constraints",
                           "native_arbitration"),
            "stage_counts": stages,
            "rejected_at_stage": {
                "unscored": unscored_count,
                "account": len(rows) - len(account_eligible),
                "support": len(account_eligible) - len(support_eligible),
                "net_edge": len(support_eligible) - len(edge_eligible),
                "calibration_error": len(edge_eligible) - len(calibration_eligible),
                "post_gate_native_constraints": post_gate_overrides,
                "native_arbitration_or_owner_readiness": (
                    native_eligible - native_selected_count
                    if native_selected_count is not None else None
                ),
            },
            "stage_coverage_of_all_candidates": {
                name: count / total if total else None
                for name, count in stages.items() if name != "total_candidates"
            },
            "interpretation": "sequential policy gates; rejects are assigned to the first failed stage",
        }
    return {
        "candidate_count": total, "scored_count": len(rows),
        "unscored_count": unscored_count, "accepted_count": accepted,
        "model_gate_accepted_count": accepted,
        "native_selected_count": native_selected_count,
        "post_gate_override_count": post_gate_overrides,
        "post_gate_override_reason_counts": dict(sorted(override_reasons.items())),
        "coverage": accepted / total if total else None,
        "native_selected_coverage": (native_selected_count / total
                                     if total and native_selected_count is not None else None),
        "reason_counts": dict(sorted(reasons.items())),
        "unscored_reason_counts": dict(sorted(unscored_reasons.items())),
        "unsupported_feature_counts": dict(sorted(unsupported.items())),
        "expected_net_bps": distribution([row.expected_net_bps for row in rows]),
        "calibration_mae_bps": distribution([row.calibration_mae_bps for row in rows]),
        "expected_net_after_calibration_mae_bps": distribution([
            row.expected_net_bps - row.calibration_mae_bps for row in rows
        ]),
        "probability_positive": distribution([row.probability_positive for row in rows]),
        "gate_counts": gates, "sequential_gate_funnel": funnel,
        "coverage_basis": "model gate acceptance before single-position native arbitration",
    }
