"""RA1 teacher eligibility from native action evidence, not train imitation loss.

A teacher can be engineering-eligible while economic alpha is UNVERIFIED.
Only host L6 can authorize formal economic claims on approved windows.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Literal, Sequence
from rles_route_a.contracts import NativeActionTrace


@dataclass(frozen=True)
class TeacherEligibility:
    expert_id: str
    engine_trace_status: Literal["ENGINEERING_PASS", "FAIL"]
    economic_status: Literal["ECONOMIC_UNVERIFIED", "SCENARIO_ONLY"]
    decision_count: int
    nonzero_command_count: int
    native_submit_count: int
    native_fill_count: int
    reason: str


def teacher_quality(expert_id: str, traces: Sequence[NativeActionTrace], *,
                    alpha_teacher: bool, min_nonzero_commands: int = 1,
                    min_native_fills: int = 1, tolerance: float = 1e-6) -> TeacherEligibility:
    if not isinstance(expert_id, str) or not expert_id.strip() or not traces:
        raise ValueError("trace and teacher id required")
    if type(alpha_teacher) is not bool:
        raise TypeError("alpha_teacher must be explicit Boolean")
    if (type(min_nonzero_commands) is not int or min_nonzero_commands < 1
            or type(min_native_fills) is not int or min_native_fills < 1):
        raise ValueError("positive integer teacher reachability thresholds required")
    if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
            or not isfinite(tolerance) or tolerance < 0):
        raise ValueError("finite nonnegative reachability tolerance required")
    if any(not isinstance(row, NativeActionTrace) for row in traces):
        raise TypeError("native action traces required")
    times = [r.decision_ts_ns for r in traces]
    if times != sorted(set(times)):
        raise ValueError("unsorted or duplicate native decision barriers")
    nonzero = sum(abs(r.requested_action) > tolerance for r in traces)
    submit = sum(r.planner_action == "SUBMIT" for r in traces)
    # A fill on an existing order during WAIT/CANCEL does not prove the
    # teacher's current command executed. Count only this barrier's submitted
    # order after its target is reached and no residual order remains.
    executable_filled_orders: set[str] = set()
    for row in traces:
        current_fills = set(row.submitted_order_ids).intersection(row.filled_order_ids)
        target_reached = (abs(row.actual_exposure_after - row.planner_target) <= tolerance
                          and not row.pending_order_ids_after)
        if row.planner_action == "SUBMIT" and target_reached:
            executable_filled_orders.update(current_fills)
    fills = len(executable_filled_orders)
    # A flat/risk teacher legitimately makes zero trades on safe markets.
    eligible = (not alpha_teacher) or (nonzero >= min_nonzero_commands
                                      and submit > 0 and fills >= min_native_fills)
    return TeacherEligibility(
        expert_id, "ENGINEERING_PASS" if eligible else "FAIL",
        "SCENARIO_ONLY" if any(r.scenario_only for r in traces) else "ECONOMIC_UNVERIFIED",
        len(traces), nonzero, submit, fills,
        "REACHABLE_NATIVE_ACTIONS" if eligible else "NO_EXECUTABLE_EXPERT_TRADES",
    )
