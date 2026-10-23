"""Check a multi-bar cancel → ACK → native fill path for explicit flat intent."""
from __future__ import annotations
from math import isfinite
from typing import Sequence
from rles_route_a.contracts import NativeActionTrace


def verify_native_flat_lifecycle(traces: Sequence[NativeActionTrace], *,
                                 initial_abs_exposure: float, tolerance: float = 1e-6) -> bool:
    """Returns True ONLY with a proven native close and terminal flat exposure.

    On cancel/ack pending, target=0 alone is not enough. The caller is required
    to wait for L3 barrier resolution and provide every relevant native trace.
    """
    if (not traces or not isinstance(initial_abs_exposure, (int, float))
            or isinstance(initial_abs_exposure, bool) or not isfinite(initial_abs_exposure)):
        raise ValueError("nonflat initial position and native barriers required")
    if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
            or not isfinite(tolerance) or tolerance <= 0):
        raise ValueError("finite positive exposure tolerance required")
    if initial_abs_exposure <= tolerance:
        raise ValueError("nonflat initial position and native barriers required")
    if any(not isinstance(trace, NativeActionTrace) for trace in traces):
        raise TypeError("native action traces required")
    if any(traces[i].decision_ts_ns >= traces[i+1].decision_ts_ns for i in range(len(traces)-1)):
        raise ValueError("native close barriers must have increasing timestamps")
    if any(abs(trace.requested_action) > tolerance for trace in traces):
        raise ValueError("native close lifecycle must consistently request flat")
    if any(abs(trace.risk_adjusted_target) > tolerance or abs(trace.planner_target) > tolerance
           for trace in traces):
        raise ValueError("native close lifecycle changed the flat target in risk or planner")
    if traces[-1].pending_order_ids_after:
        return False
    if abs(traces[-1].actual_exposure_after) > tolerance:
        return False
    # A submitted close can fill on a later barrier (for example after WAIT).
    # Track only orders submitted during this flat-intent lifecycle so a fill
    # from a pre-existing entry order cannot certify the close.
    submitted: set[str] = set()
    close_fill_seen = False
    for trace in traces:
        submitted.update(trace.submitted_order_ids)
        if submitted.intersection(trace.filled_order_ids):
            close_fill_seen = True
    if not close_fill_seen:
        return False
    return True
