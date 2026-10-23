"""Execution reachability diagnostics from native trace only."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from statistics import mean, pstdev
from typing import Sequence
from rles_route_a.contracts import NativeActionTrace
from rles_route_a.datasets.transitions import OfflineTransition


@dataclass(frozen=True)
class OfflinePolicyActionQuery:
    """Out-of-sample policy action at a frozen-encoder state, for diagnostics only."""

    state: tuple[float, ...]
    action: float
    fold_id: str
    episode_id: str
    decision_ts_ns: int
    source_identity: str
    frozen_encoder_id: str
    execution_recipe: str
    stratum: str

    def __post_init__(self) -> None:
        if (not isinstance(self.state, tuple) or not self.state
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not isfinite(value) for value in self.state)):
            raise ValueError("policy support query needs a finite frozen state vector")
        if (not isinstance(self.action, (int, float)) or isinstance(self.action, bool)
                or not isfinite(self.action) or not -1 <= self.action <= 1):
            raise ValueError("policy support query action must be finite and in [-1, 1]")
        if (type(self.decision_ts_ns) is not int or self.decision_ts_ns <= 0
                or any(not isinstance(value, str) or not value.strip() for value in
                       (self.fold_id, self.episode_id, self.source_identity,
                        self.frozen_encoder_id, self.execution_recipe, self.stratum))):
            raise ValueError("policy support query provenance and stratum are required")


def audit_offline_policy_action_support(
        training_rows: Sequence[OfflineTransition],
        policy_queries: Sequence[OfflinePolicyActionQuery], *,
        training_strata: Sequence[str],
        max_state_distance: float, action_tolerance: float,
        neighbors: int = 5,
) -> dict[str, object]:
    """Measure held-out action support conditional on state and frozen strata.

    Query strata must be preregistered buckets, for example
    ``market_regime|position_direction_and_size|holding_duration``. State
    distances are standardized using training rows only. This report is a
    diagnostic, not a value estimate, confidence interval, or promotion gate.
    """
    if not training_rows or not policy_queries:
        raise ValueError("offline action support needs training rows and policy queries")
    if (len(training_strata) != len(training_rows)
            or any(not isinstance(value, str) or not value.strip()
                   for value in training_strata)):
        raise ValueError("one frozen market/position/duration stratum is required per Train row")
    if (isinstance(max_state_distance, bool)
            or not isinstance(max_state_distance, (int, float))
            or not isfinite(max_state_distance) or max_state_distance <= 0
            or isinstance(action_tolerance, bool)
            or not isinstance(action_tolerance, (int, float))
            or not isfinite(action_tolerance) or action_tolerance <= 0
            or type(neighbors) is not int or neighbors < 1):
        raise ValueError("frozen positive state/action thresholds and neighbor count required")
    if any(not isinstance(row, OfflineTransition) for row in training_rows):
        raise TypeError("training action support must come from native offline transitions")
    if any(not isinstance(query, OfflinePolicyActionQuery) for query in policy_queries):
        raise TypeError("policy action support queries must retain evaluation provenance")

    references = tuple(training_rows)
    queries = tuple(policy_queries)
    if any(not all((row.fold_id, row.source_identity, row.frozen_encoder_id,
                    row.native_execution_recipe)) for row in references):
        raise ValueError("native support rows must retain registered identities")
    reference_ids = {(row.fold_id, row.source_identity, row.frozen_encoder_id,
                      row.native_execution_recipe) for row in references}
    query_ids = {(row.fold_id, row.source_identity, row.frozen_encoder_id,
                  row.execution_recipe) for row in queries}
    if (len(reference_ids) != 1 or len(query_ids) != 1
            or len({(row.source_identity, row.frozen_encoder_id,
                     row.native_execution_recipe) for row in references}
                   | {(row.source_identity, row.frozen_encoder_id,
                      row.execution_recipe) for row in queries}) != 1):
        raise PermissionError("support comparison mixes Fold, source, encoder, or execution recipe")
    train_episodes = {row.episode_id for row in references}
    query_episodes = {row.episode_id for row in queries}
    if (not all(train_episodes) or not all(query_episodes)
            or train_episodes.intersection(query_episodes)):
        raise PermissionError("action support queries must use episodes disjoint from training")
    if any(type(row.decision_ts_ns) is not int or row.decision_ts_ns <= 0
           for row in references):
        raise ValueError("native action support rows must retain decision timestamps")
    if min(query.decision_ts_ns for query in queries) <= max(
            row.decision_ts_ns for row in references):
        raise PermissionError("policy support queries must follow the entire training time range")
    dimension = len(references[0].state)
    if (any(len(row.state) != dimension for row in references)
            or any(len(query.state) != dimension for query in queries)):
        raise ValueError("support state dimensions differ from the frozen encoder")

    # Use only training-state moments. Constant dimensions are omitted from the
    # distance by assigning unit scale; they contribute no separation anyway.
    scales = [pstdev(row.state[index] for row in references) or 1.0
              for index in range(dimension)]
    grouped: dict[str, list[OfflineTransition]] = {}
    for row, stratum in zip(references, training_strata, strict=True):
        grouped.setdefault(stratum, []).append(row)
    per_stratum: dict[str, dict[str, float | int]] = {}
    state_supported = 0
    action_supported = 0
    nearest_action_gaps: list[float] = []
    for query in queries:
        references_in_stratum = grouped.get(query.stratum, [])
        if not references_in_stratum:
            metrics = per_stratum.setdefault(query.stratum, {
                "query_count": 0, "train_count": 0, "state_supported_count": 0,
                "action_supported_count": 0, "mean_nearest_state_distance": 0.0,
                "mean_nearest_action_gap": 0.0,
            })
            metrics["query_count"] += 1
            continue
        distances = [
            sum(((a - b) / scale) ** 2
                for a, b, scale in zip(query.state, row.state, scales, strict=True)) ** 0.5
            for row in references_in_stratum
        ]
        ordered = sorted(zip(distances, references_in_stratum, strict=True),
                         key=lambda item: item[0])
        nearest_distance = ordered[0][0]
        nearest_rows = [row for _, row in ordered[:neighbors]]
        nearest_action_gap = min(abs(query.action - row.requested_action)
                                 for row in nearest_rows)
        supported_state = nearest_distance <= max_state_distance
        supported_action = supported_state and nearest_action_gap <= action_tolerance
        state_supported += int(supported_state)
        action_supported += int(supported_action)
        nearest_action_gaps.append(nearest_action_gap)
        metrics = per_stratum.setdefault(query.stratum, {
            "query_count": 0, "train_count": len(references_in_stratum),
            "state_supported_count": 0, "action_supported_count": 0,
            "mean_nearest_state_distance": 0.0,
            "mean_nearest_action_gap": 0.0,
        })
        count = int(metrics["query_count"])
        metrics["mean_nearest_state_distance"] = (
            (float(metrics["mean_nearest_state_distance"]) * count + nearest_distance)
            / (count + 1)
        )
        metrics["mean_nearest_action_gap"] = (
            (float(metrics["mean_nearest_action_gap"]) * count + nearest_action_gap)
            / (count + 1)
        )
        metrics["query_count"] = count + 1
        metrics["state_supported_count"] += int(supported_state)
        metrics["action_supported_count"] += int(supported_action)
    for metrics in per_stratum.values():
        count = int(metrics["query_count"])
        metrics["state_supported_fraction"] = (
            int(metrics["state_supported_count"]) / count if count else 0.0
        )
        metrics["action_supported_fraction"] = (
            int(metrics["action_supported_count"]) / count if count else 0.0
        )
    return {
        "status": "DIAGNOSTIC_ONLY",
        "training_fold_id": next(iter(reference_ids))[0],
        "query_fold_ids": sorted({query.fold_id for query in queries}),
        "source_identity": next(iter(reference_ids))[1],
        "frozen_encoder_id": next(iter(reference_ids))[2],
        "execution_recipe": next(iter(reference_ids))[3],
        "training_count": len(references),
        "query_count": len(queries),
        "state_supported_fraction": state_supported / len(queries),
        "action_supported_fraction": action_supported / len(queries),
        "mean_nearest_action_gap": (mean(nearest_action_gaps)
                                    if nearest_action_gaps else 0.0),
        "max_state_distance": float(max_state_distance),
        "action_tolerance": float(action_tolerance),
        "neighbors": neighbors,
        "strata": per_stratum,
    }


def audit_action_identifiability(rows: Sequence[OfflineTransition], *,
                                 action_tolerance: float = 1e-6
                                 ) -> dict[str, float | int | str]:
    """Measure exact-state command aliases in replay; this is diagnostic only.

    A pair is comparable only when its exact encoded state is repeated with
    distinct commands. Alias outcomes require the same native risk/planner
    mapping and same observed reward/next-state/episode boundary.
    """
    if not rows:
        raise ValueError("no offline transitions")
    if (isinstance(action_tolerance, bool) or not isinstance(action_tolerance, (int, float))
            or not isfinite(action_tolerance) or action_tolerance <= 0):
        raise ValueError("action tolerance must be finite and positive")
    groups: dict[tuple[float, ...], list[OfflineTransition]] = {}
    for row in rows:
        if row.action_trace is None:
            raise ValueError("native action mapping required for identifiability audit")
        groups.setdefault(row.state, []).append(row)
    pair_count = alias_count = eligible_states = 0
    for group in groups.values():
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                if abs(left.requested_action - right.requested_action) <= action_tolerance:
                    continue
                pair_count += 1
                lt, rt = left.action_trace, right.action_trace
                if lt is None or rt is None:
                    raise ValueError("native action mapping required for identifiability audit")
                left_outcome = (lt.soft_risk_target, lt.risk_adjusted_target,
                    lt.planner_target, lt.planner_action, left.native_log_reward,
                    left.next_state, left.terminated, left.truncated, left.bootstrap_allowed)
                right_outcome = (rt.soft_risk_target, rt.risk_adjusted_target,
                    rt.planner_target, rt.planner_action, right.native_log_reward,
                    right.next_state, right.terminated, right.truncated, right.bootstrap_allowed)
                if left_outcome == right_outcome:
                    alias_count += 1
        if any(abs(left.requested_action - right.requested_action) > action_tolerance
               for i, left in enumerate(group) for right in group[i + 1:]):
            eligible_states += 1
    return {
        "repeated_state_groups_with_distinct_commands": eligible_states,
        "distinct_command_pairs_compared": pair_count,
        "aliased_pair_count": alias_count,
        "aliased_pair_fraction": alias_count / pair_count if pair_count else 0.0,
        "status": "DIAGNOSTIC_ONLY" if pair_count else "INSUFFICIENT_EXACT_STATE_REPEATS",
    }


def summarize_actions(rows: Sequence[NativeActionTrace], *,
                      meaningful_delta: float = 1e-4) -> dict[str, float | int | str]:
    if not rows:
        raise ValueError("no native trace")
    if (isinstance(meaningful_delta, bool) or not isinstance(meaningful_delta, (int, float))
            or not isfinite(meaningful_delta) or meaningful_delta <= 0):
        raise ValueError("meaningful action threshold must be finite and positive")
    n = len(rows)
    requests = [row.requested_action for row in rows]
    exposures = [row.actual_exposure_after for row in rows]
    direct_pairs = list(zip(requests, requests[1:]))
    submitted = [order_id for row in rows for order_id in row.submitted_order_ids]
    submitted_filled = {
        order_id for row in rows
        for order_id in set(row.submitted_order_ids).intersection(row.filled_order_ids)
    }
    submit_barriers = [row for row in rows if row.planner_action == "SUBMIT"]
    submit_barriers_with_fill = sum(
        bool(set(row.submitted_order_ids).intersection(row.filled_order_ids))
        for row in submit_barriers
    )
    soft_targets = [(row, row.soft_risk_target) for row in rows
                    if row.soft_risk_target is not None]
    return {
        "n": n,
        "nonzero_request_fraction": sum(abs(x.requested_action) > meaningful_delta for x in rows) / n,
        "soft_risk_target_known_fraction": len(soft_targets) / n,
        "nonzero_soft_risk_target_fraction": (
            sum(abs(target) > meaningful_delta for _, target in soft_targets)
            / len(soft_targets) if soft_targets else 0.0
        ),
        "nonzero_risk_adjusted_fraction": sum(abs(x.risk_adjusted_target) > meaningful_delta for x in rows) / n,
        "nonzero_planner_target_fraction": sum(abs(x.planner_target) > meaningful_delta for x in rows) / n,
        "submit_fraction": sum(x.planner_action == "SUBMIT" for x in rows) / n,
        "noop_fraction": sum(x.planner_action == "NOOP" for x in rows) / n,
        "cancel_fraction": sum(x.planner_action == "CANCEL" for x in rows) / n,
        "wait_fraction": sum(x.planner_action == "WAIT" for x in rows) / n,
        "submit_barrier_count": len(submit_barriers),
        "submit_barriers_with_current_fill_fraction": (
            submit_barriers_with_fill / len(submit_barriers) if submit_barriers else 0.0
        ),
        "submitted_order_count": len(submitted),
        "current_submitted_order_fill_fraction": (
            len(submitted_filled) / len(submitted) if submitted else 0.0
        ),
        "mean_request_abs": mean(abs(x.requested_action) for x in rows),
        "request_std": pstdev(requests),
        "request_flat_fraction": sum(abs(value) <= meaningful_delta for value in requests) / n,
        "request_direct_reversal_fraction": (
            sum(abs(left) > meaningful_delta and abs(right) > meaningful_delta
                and left * right < 0 for left, right in direct_pairs) / len(direct_pairs)
            if direct_pairs else 0.0
        ),
        "actual_exposure_std": pstdev(exposures),
        "actual_flat_fraction": sum(abs(value) <= meaningful_delta for value in exposures) / n,
        "mean_request_to_soft_risk_gap": (
            mean(abs(row.requested_action - target) for row, target in soft_targets)
            if soft_targets else 0.0
        ),
        "mean_soft_to_hard_risk_gap": (
            mean(abs(target - row.risk_adjusted_target) for row, target in soft_targets)
            if soft_targets else 0.0
        ),
        "mean_requested_to_planner_gap": mean(abs(x.requested_action - x.planner_target) for x in rows),
        "meaningful_risk_rewrite_fraction": sum(abs(x.requested_action - x.risk_adjusted_target) > meaningful_delta for x in rows) / n,
        "filled_event_count": sum(len(x.filled_order_ids) for x in rows),
        "terminal_equity": rows[-1].native_equity_after,
        "economic_status": "SCENARIO_ONLY" if any(x.scenario_only for x in rows) else "HOST_NATIVE_FACTS",
    }
