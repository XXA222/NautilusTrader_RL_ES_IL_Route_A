"""Train the simple meta model on host-authorized, disjoint Train intervals."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from rles_route_a.integration.scope import HostTrainGrant
from .candidate_event_union import CandidateGroup
from .conditional_outcome_labels import FrozenReplay, GroupOutcome, label_grouped_candidates, purge_before
from .entry_meta_policy import FrozenEntryMetaPolicy, fit_entry_meta_policy


class MetaTrainingError(ValueError):
    """Retain the full candidate-label denominator when fitting cannot proceed."""

    def __init__(self, message, *, fit_outcomes, calibration_outcomes, diagnostics):
        super().__init__(message)
        self.fit_outcomes = fit_outcomes
        self.calibration_outcomes = calibration_outcomes
        self.diagnostics = diagnostics


def _overlap_diagnostics(labels):
    """Count overlapping label windows; correlated events are not independent trials."""
    from heapq import heappop, heappush

    active = []
    overlapping = 0
    maximum = 0
    for label in sorted(labels, key=lambda item: (item.decision_ts_ns, item.event_id)):
        while active and active[0] <= label.decision_ts_ns:
            heappop(active)
        overlapping += bool(active)
        heappush(active, label.label_end_ts_ns)
        maximum = max(maximum, len(active))
    return {"labels": len(labels), "labels_starting_during_another": overlapping,
            "max_concurrent_labels": maximum,
            "interpretation": "overlapping labels are dependent; counts are not independent trials"}


def _account_candidate_coverage(groups):
    """Report account-state coverage for the entire candidate denominator."""
    occupied = sum(abs(group.observation.account_value("net_exposure")) > 1e-8 for group in groups)
    pending = sum(bool(group.observation.pending_order_ids) for group in groups)
    return {"candidates": len(groups), "position_occupied": occupied,
            "pending_order": pending,
            "position_or_pending": sum(abs(group.observation.account_value("net_exposure")) > 1e-8
                                       or bool(group.observation.pending_order_ids) for group in groups)}


@dataclass(frozen=True, slots=True)
class MetaTrainingResult:
    policy: FrozenEntryMetaPolicy
    fit_outcomes: tuple[GroupOutcome, ...]
    calibration_outcomes: tuple[GroupOutcome, ...]
    diagnostics: dict[str, object]


def train_meta_on_registered_candidates(
    *, grant: HostTrainGrant, verify_grant: Callable[[HostTrainGrant], bool],
    fit_groups: Sequence[CandidateGroup], calibration_groups: Sequence[CandidateGroup],
    fit_interval: tuple[int, int], calibration_interval: tuple[int, int],
    replay: Callable[[CandidateGroup, str, str, str], FrozenReplay],
    execution_recipe: str, scenario: str, feature_names: Sequence[str],
    margin_bps: float, cost_bps: float, embargo_ns: int = 0,
    minimum_complete_fit: int = 4, minimum_complete_calibration: int = 4,
    fee_bps_per_leg: float | None = None, slippage_bps_per_leg: float | None = None,
    l2: float = 1.0,
) -> MetaTrainingResult:
    """No market IO: host must verify its registered read before passing snapshots."""
    if not isinstance(grant, HostTrainGrant) or not callable(verify_grant) or verify_grant(grant) is not True:
        raise PermissionError("host-verified Train grant is required")
    if (not isinstance(fit_interval, tuple) or not isinstance(calibration_interval, tuple)
            or len(fit_interval) != 2 or len(calibration_interval) != 2
            or any(type(t) is not int for t in (*fit_interval, *calibration_interval))
            or not 0 < fit_interval[0] < fit_interval[1] <= calibration_interval[0] < calibration_interval[1]):
        raise ValueError("fit and calibration need ordered non-overlapping intervals")
    if (type(minimum_complete_fit) is not int or minimum_complete_fit < 4
            or type(minimum_complete_calibration) is not int or minimum_complete_calibration < 4):
        raise ValueError("minimum complete sample counts must be integers of at least four")
    fit_groups, calibration_groups = tuple(fit_groups), tuple(calibration_groups)
    for group in (*fit_groups, *calibration_groups):
        obs = group.observation
        if (obs.source_id, obs.fold_id) != (grant.source_identity, grant.fold_id):
            raise PermissionError("candidate does not match the verified Train grant")
    if any(not fit_interval[0] <= g.observation.decision_ts_ns < fit_interval[1] for g in fit_groups):
        raise ValueError("fit candidate outside declared interval")
    if any(not calibration_interval[0] <= g.observation.decision_ts_ns < calibration_interval[1]
           for g in calibration_groups):
        raise ValueError("calibration candidate outside declared interval")
    kwargs = dict(exit_policy_id="X1", execution_recipe=execution_recipe,
                  cost_scenario_id=scenario, gross_return_method="simple")
    fit_outcomes = (label_grouped_candidates(fit_groups, replay, block_end_ns=fit_interval[1], **kwargs)
                    if fit_groups else ())
    calibration_outcomes = (label_grouped_candidates(calibration_groups, replay,
                            block_end_ns=calibration_interval[1], **kwargs) if calibration_groups else ())
    interval_fit = purge_before(fit_outcomes, next_start_ns=calibration_interval[0],
                                embargo_ns=embargo_ns)
    interval_cal = tuple(x for x in calibration_outcomes if x.outcome_status == "COMPLETE"
                         and x.label_end_ts_ns is not None and x.net_return_bps is not None)
    group_by_id = {g.event_id: g for g in (*fit_groups, *calibration_groups)}
    def ready(label):
        market = group_by_id[label.event_id].observation.market
        return all(not name.startswith("market.") or name[7:] in market for name in feature_names)
    eligible_fit = tuple(x for x in interval_fit if ready(x))
    eligible_cal = tuple(x for x in interval_cal if ready(x))
    diagnostics = {
        "evidence_type": "MODEL_COUNTERFACTUAL",
        "fit_candidates": len(fit_groups), "fit_complete_used": len(eligible_fit),
        "fit_account_coverage": _account_candidate_coverage(fit_groups),
        "fit_status": dict(sorted(Counter(x.outcome_status for x in fit_outcomes).items())),
        "fit_purged_or_incomplete": len(fit_outcomes) - len(interval_fit),
        "fit_feature_not_ready": len(interval_fit) - len(eligible_fit),
        "calibration_candidates": len(calibration_groups), "calibration_complete_used": len(eligible_cal),
        "calibration_account_coverage": _account_candidate_coverage(calibration_groups),
        "calibration_status": dict(sorted(Counter(x.outcome_status for x in calibration_outcomes).items())),
        "calibration_feature_not_ready": len(interval_cal) - len(eligible_cal),
        "minimum_complete_fit": minimum_complete_fit,
        "minimum_complete_calibration": minimum_complete_calibration,
        "fit_overlap": _overlap_diagnostics(eligible_fit),
        "calibration_overlap": _overlap_diagnostics(eligible_cal),
    }
    if len(eligible_fit) < minimum_complete_fit or len(eligible_cal) < minimum_complete_calibration:
        raise MetaTrainingError("insufficient complete fit or calibration candidates",
            fit_outcomes=fit_outcomes, calibration_outcomes=calibration_outcomes, diagnostics=diagnostics)
    selected = {x.event_id for x in (*eligible_fit, *eligible_cal)}
    all_groups = {group.event_id: group for group in (*fit_groups, *calibration_groups)
                  if group.event_id in selected}
    try:
        model = fit_entry_meta_policy(eligible_fit, eligible_cal, all_groups, feature_names=feature_names,
                                      margin_bps=margin_bps, cost_bps=cost_bps, l2=l2,
                                      fee_bps_per_leg=fee_bps_per_leg,
                                      slippage_bps_per_leg=slippage_bps_per_leg)
    except ValueError as exc:
        raise MetaTrainingError(f"model fitting rejected: {exc}", fit_outcomes=fit_outcomes,
            calibration_outcomes=calibration_outcomes, diagnostics=diagnostics) from exc
    diagnostics.update({"calibration_mae_bps": model.calibration_mae_bps,
                        "calibration_brier": model.calibration_brier,
                        "model_available_after_ns": model.available_after_ns})
    return MetaTrainingResult(model, fit_outcomes, calibration_outcomes, diagnostics)



def train_meta_from_preregistered_x1(
    *, prereg_path, grant: HostTrainGrant, verify_grant, fit_groups, calibration_groups,
    observations, opens, funding, scenario: str, native_factory,
) -> MetaTrainingResult:
    """Apply exact frozen X1 features, costs and sample minima to host Train facts."""
    from datetime import datetime
    from hashlib import sha256
    import json
    from pathlib import Path
    from .x1_group_replay import make_x1_group_replay

    path = Path(prereg_path)
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip() != digest:
        raise ValueError("Meta preregistration hash mismatch")
    config = json.loads(raw)
    if (config.get("schema") != "rles.route-a.meta-x1-train-prereg.v1"
            or not isinstance(grant, HostTrainGrant)
            or grant.owner != config["protocol_id"]
            or not callable(verify_grant) or verify_grant(grant) is not True):
        raise PermissionError("verified grant must bind the independently preregistered protocol")
    if scenario not in config["cost_scenarios"]:
        raise ValueError("unregistered cost scenario")

    def ns(text):
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()) * 1_000_000_000

    fit_interval = tuple(map(ns, config["intervals"]["train_fit"]))
    cal_interval = tuple(map(ns, config["intervals"]["train_calibration"]))
    warmup_start = ns(config["intervals"]["warmup"][0])
    observations, opens, funding = tuple(observations), tuple(opens), tuple(funding)
    if (not observations or any(not warmup_start <= o.decision_ts_ns < cal_interval[1]
                                for o in observations)
            or any(not warmup_start <= timestamp <= cal_interval[1] for timestamp, _ in opens)
            or any(not warmup_start <= timestamp < cal_interval[1] for timestamp, _, _ in funding)):
        raise ValueError("model fitting inputs cannot include native-screen data")
    fit_groups, calibration_groups = tuple(fit_groups), tuple(calibration_groups)
    if len({g.event_id for g in (*fit_groups, *calibration_groups)}) != len(fit_groups) + len(calibration_groups):
        raise ValueError("duplicate fit or calibration candidate")
    block_ends = {group.event_id: fit_interval[1] for group in fit_groups}
    block_ends.update({group.event_id: cal_interval[1] for group in calibration_groups})
    read = getattr(native_factory, "application_registered_train_read", None)
    authority = getattr(native_factory, "application_registered_train_read_verifier", None)
    if (not callable(authority) or authority(read) is not True
            or HostTrainGrant.from_registered_read(read, fold_id=grant.fold_id) != grant):
        raise PermissionError("model training requires the factory's verified native Train receipt")
    feature_hash = getattr(getattr(native_factory, "features", None), "hash", None)
    if not feature_hash or any(o.feature_spec_hash != feature_hash for o in observations):
        raise ValueError("native factory and training observation feature schemas differ")
    costs = config["cost_scenarios"][scenario]
    from .execution_binding import validate_native_execution
    execution_recipe = getattr(native_factory, "route_a_execution_recipe_hash", None)
    if config["execution"]["entry"] != "next_open_market":
        raise ValueError("unsupported preregistered entry model")
    validate_native_execution(native_factory, scenario=scenario, expected_recipe=execution_recipe,
        fee_bps=costs["fee_bps_per_leg"], slippage_bps=costs["slippage_bps_per_leg"])
    replay = make_x1_group_replay(observations=observations, opens=opens, funding=funding,
        block_ends=block_ends, fee_bps=costs["fee_bps_per_leg"],
        slippage_bps=costs["slippage_bps_per_leg"], execution_recipe=execution_recipe, scenario=scenario)
    model = config["model"]
    return train_meta_on_registered_candidates(
        grant=grant, verify_grant=verify_grant, fit_groups=fit_groups,
        calibration_groups=calibration_groups, fit_interval=fit_interval,
        calibration_interval=cal_interval, replay=replay, execution_recipe=execution_recipe,
        scenario=scenario, feature_names=model["features"], margin_bps=model["margin_bps"],
        cost_bps=2 * (costs["fee_bps_per_leg"] + costs["slippage_bps_per_leg"]),
        minimum_complete_fit=model["minimum_complete_fit"],
        minimum_complete_calibration=model["minimum_complete_calibration"], l2=model["l2"],
        fee_bps_per_leg=costs["fee_bps_per_leg"],
        slippage_bps_per_leg=costs["slippage_bps_per_leg"])
