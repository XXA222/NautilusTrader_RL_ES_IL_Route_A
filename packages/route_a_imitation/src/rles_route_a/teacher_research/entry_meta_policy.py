"""Small Train-fit/calibration opportunity baseline; never a native fill."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from .candidate_event_union import CandidateGroup
from .conditional_outcome_labels import GroupOutcome
from .opportunity_models import _fit_logistic, _fit_platt, _sigmoid


def _features(group: CandidateGroup, names: Sequence[str]) -> np.ndarray:
    obs = group.observation
    values = []
    for name in names:
        if name.startswith("market."):
            values.append(obs.market_value(name[7:]))
        elif name.startswith("account."):
            values.append(obs.account_value(name[8:]))
        else:
            raise ValueError("only causal market/account observation features are permitted")
    return np.asarray([*values, float(group.side)], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class EntryScore:
    action: str
    probability_positive: float
    expected_net_bps: float
    in_support: bool
    reason: str


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

    def score(self, group: CandidateGroup) -> EntryScore:
        obs = group.observation
        if (obs.source_id, obs.fold_id, obs.feature_spec_hash) != (
                self.source_id, self.fold_id, self.feature_spec_hash):
            raise ValueError("entry model source/fold/schema mismatch")
        if obs.decision_ts_ns <= self.available_after_ns:
            raise ValueError("entry model cannot predict before calibration labels are available")
        x = _features(group, self.feature_names)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite entry feature")
        z = np.concatenate(([1.], (x - self.mean) / self.scale))
        probability = _sigmoid(self.calibrator[0] * float(z @ self.logistic) + self.calibrator[1])
        expected = float(z @ self.value_weights)
        if not isfinite(probability) or not isfinite(expected):
            raise ValueError("nonfinite opportunity prediction")
        support = bool(np.all(x >= self.lower) and np.all(x <= self.upper))
        if obs.pending_order_ids:
            return EntryScore("NO_TRADE", probability, expected, support, "PENDING_ORDER")
        if abs(obs.account_value("net_exposure")) > 1e-8:
            return EntryScore("NO_TRADE", probability, expected, support, "POSITION_OCCUPIED")
        if not support:
            return EntryScore("NO_TRADE", probability, expected, support, "OUT_OF_SUPPORT")
        if expected <= self.margin_bps:
            return EntryScore("NO_TRADE", probability, expected, support, "NET_EDGE_TOO_SMALL")
        return EntryScore("LONG" if group.side > 0 else "SHORT",
                          probability, expected, support, "POSITIVE_NET_EDGE")


def fit_entry_meta_policy(
    fit: Sequence[GroupOutcome], calibration: Sequence[GroupOutcome],
    groups: Mapping[str, CandidateGroup], *, feature_names: Sequence[str],
    margin_bps: float, l2: float = 1.0,
) -> FrozenEntryMetaPolicy:
    """Fit only earlier complete research labels and calibrate on later labels."""
    names = tuple(feature_names)
    if (not names or len(set(names)) != len(names)
            or not isfinite(margin_bps) or margin_bps < 0 or not isfinite(l2) or l2 <= 0):
        raise ValueError("fixed features, nonnegative margin and positive regularization required")
    all_labels = tuple(fit) + tuple(calibration)
    if len(fit) < 4 or len(calibration) < 4:
        raise ValueError("fit and later calibration need at least four complete samples")
    if (len({x.event_id for x in all_labels}) != len(all_labels)
            or set(groups) != {x.event_id for x in all_labels}):
        raise ValueError("fit/calibration groups must cover unique events exactly")
    if any(x.evidence_type != "MODEL_COUNTERFACTUAL" or x.economic_status != "SCENARIO_ONLY"
           or x.net_return_bps is None or x.label_end_ts_ns is None for x in all_labels):
        raise ValueError("complete scenario-only research labels required")
    first_cal = min(x.decision_ts_ns for x in calibration)
    if any(x.label_end_ts_ns is None or x.label_end_ts_ns >= first_cal for x in fit):
        raise ValueError("fit labels overlap calibration")
    if any(x.label_end_ts_ns is None or x.label_end_ts_ns <= x.decision_ts_ns for x in all_labels):
        raise ValueError("labels must end after the decision")
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
    train_x = np.stack([_features(groups[x.event_id], names) for x in fit])
    cal_x = np.stack([_features(groups[x.event_id], names) for x in calibration])
    train_y = np.asarray([x.net_return_bps for x in fit], dtype=np.float64)
    cal_y = np.asarray([x.net_return_bps for x in calibration], dtype=np.float64)
    if (not all(np.isfinite(x).all() for x in (train_x, cal_x, train_y, cal_y))
            or len(np.unique(train_y > 0)) != 2 or len(np.unique(cal_y > 0)) != 2):
        raise ValueError("finite fit/calibration examples with wins and losses required")
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale < 1e-12] = 1.
    train_z = np.column_stack((np.ones(len(train_x)), (train_x - mean) / scale))
    cal_z = np.column_stack((np.ones(len(cal_x)), (cal_x - mean) / scale))
    logistic = _fit_logistic(train_z, (train_y > 0).astype(float), l2, 0.05, 2000)
    calibrator = _fit_platt(cal_z @ logistic, (cal_y > 0).astype(float), 0.05, 2000)
    penalty = np.eye(train_z.shape[1]) * l2
    penalty[0, 0] = 0.
    value = np.linalg.solve(train_z.T @ train_z + penalty, train_z.T @ train_y)
    for array in (mean, scale, logistic, value):
        array.setflags(write=False)
    lower, upper = train_x.min(axis=0), train_x.max(axis=0)
    lower.setflags(write=False)
    upper.setflags(write=False)
    return FrozenEntryMetaPolicy(
        names, mean, scale, lower, upper, logistic, calibrator, value,
        all_labels[0].source_identity, all_labels[0].fold_id, next(iter(specs)),
        next(iter(recipes)), max(x.label_end_ts_ns or 0 for x in calibration), margin_bps,
    )
