"""Train the simple meta model on host-authorized, disjoint Train intervals."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from rles_route_a.integration.scope import HostTrainGrant
from .candidate_event_union import CandidateGroup
from .conditional_outcome_labels import FrozenReplay, GroupOutcome, label_grouped_candidates, purge_before
from .entry_meta_policy import FrozenEntryMetaPolicy, fit_entry_meta_policy


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
    if not fit_groups or not calibration_groups:
        raise ValueError("both Train intervals require candidates")
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
    fit_outcomes = label_grouped_candidates(fit_groups, replay, block_end_ns=fit_interval[1], **kwargs)
    calibration_outcomes = label_grouped_candidates(calibration_groups, replay,
                                                     block_end_ns=calibration_interval[1], **kwargs)
    eligible_fit = purge_before(fit_outcomes, next_start_ns=calibration_interval[0],
                                embargo_ns=embargo_ns)
    eligible_cal = tuple(x for x in calibration_outcomes if x.outcome_status == "COMPLETE"
                         and x.label_end_ts_ns is not None and x.net_return_bps is not None)
    if len(eligible_fit) < minimum_complete_fit or len(eligible_cal) < minimum_complete_calibration:
        raise ValueError("insufficient complete fit or calibration candidates")
    selected = {x.event_id for x in (*eligible_fit, *eligible_cal)}
    all_groups = {group.event_id: group for group in (*fit_groups, *calibration_groups)
                  if group.event_id in selected}
    model = fit_entry_meta_policy(eligible_fit, eligible_cal, all_groups, feature_names=feature_names,
                                  margin_bps=margin_bps, cost_bps=cost_bps)
    return MetaTrainingResult(model, fit_outcomes, calibration_outcomes, {
        "evidence_type": "MODEL_COUNTERFACTUAL",
        "fit_candidates": len(fit_groups), "fit_complete_used": len(eligible_fit),
        "fit_status": dict(sorted(Counter(x.outcome_status for x in fit_outcomes).items())),
        "fit_purged_or_incomplete": len(fit_outcomes) - len(eligible_fit),
        "calibration_candidates": len(calibration_groups), "calibration_complete_used": len(eligible_cal),
        "calibration_status": dict(sorted(Counter(x.outcome_status for x in calibration_outcomes).items())),
        "calibration_mae_bps": model.calibration_mae_bps,
        "calibration_brier": model.calibration_brier,
        "model_available_after_ns": model.available_after_ns,
    })



def train_meta_from_preregistered_x1(
    *, prereg_path, grant: HostTrainGrant, verify_grant, fit_groups, calibration_groups,
    observations, opens, funding, scenario: str,
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
    costs = config["cost_scenarios"][scenario]
    execution_recipe = config["execution"]["entry"]
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
        minimum_complete_calibration=model["minimum_complete_calibration"])
