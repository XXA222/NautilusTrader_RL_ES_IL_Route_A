"""Execution reachability diagnostics from native trace only."""
from __future__ import annotations
from math import isfinite
from statistics import mean, pstdev
from typing import Sequence
from rles_route_a.contracts import NativeActionTrace
from rles_route_a.datasets.transitions import OfflineTransition


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
