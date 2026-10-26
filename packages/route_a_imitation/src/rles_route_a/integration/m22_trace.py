"""Cross-check Route A barriers against the host's existing M2.2 trace.

This is a join/consistency adapter, not a second execution ledger. M2.2 remains
authoritative for native order status, fills, fees, and unfilled outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isclose, isfinite
from typing import Any

from .trace_audit import AuditedTeacherBarrier


@dataclass(frozen=True, slots=True)
class M22DecisionTraceSummary:
    decisions: int
    submits: int
    related_order_references: int


@dataclass(frozen=True, slots=True)
class M22OrderFactsSummary:
    barriers: int
    order_status_facts: int
    policy_fill_events: int


def _snapshot_status_rows(
    snapshot: Any, order_ids: set[str], *, label: str
) -> tuple[dict[str, str], ...]:
    all_orders = getattr(snapshot, "all_orders", None)
    by_requested_ids = getattr(all_orders, "for_ids", None)
    if callable(by_requested_ids):
        rows = by_requested_ids(order_ids)
    elif isinstance(all_orders, (tuple, list)):
        rows = all_orders
    else:
        raise ValueError(f"M2.2 {label} snapshot lacks complete native order history")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"M2.2 {label} order row is malformed")
        identity = row.get("client_order_id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"M2.2 {label} order row lacks client identity")
        if identity not in order_ids:
            continue
        status, side = row.get("status"), row.get("side")
        if (
            not isinstance(status, str)
            or not status.strip()
            or status != status.upper()
            or not isinstance(side, str)
            or side not in {"BUY", "SELL"}
        ):
            raise ValueError(f"M2.2 {label} order status or side is invalid")
        try:
            quantity = Decimal(str(row.get("quantity")))
            filled_qty = Decimal(str(row.get("filled_qty")))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"M2.2 {label} order quantities are invalid") from exc
        if (
            not quantity.is_finite()
            or not filled_qty.is_finite()
            or quantity <= 0
            or filled_qty < 0
            or filled_qty > quantity
        ):
            raise ValueError(f"M2.2 {label} order quantities are outside valid bounds")
        values = {
            "client_order_id": identity,
            "status": status,
            "side": side,
            "quantity": str(row["quantity"]),
            "filled_qty": str(row["filled_qty"]),
        }
        previous = by_id.get(identity)
        if previous is not None and previous != values:
            raise ValueError(f"M2.2 {label} order snapshots conflict for {identity}")
        by_id[identity] = values
    missing_ids = sorted(order_ids - by_id.keys())
    if missing_ids:
        raise ValueError(f"M2.2 {label} snapshot omits Route A order identities: {missing_ids}")
    return tuple(by_id[key] for key in sorted(by_id))


def _snapshot_fills(snapshot: Any, *, start_ns: int, end_ns: int) -> tuple[Mapping[str, Any], ...]:
    fills = getattr(snapshot, "fills", None)
    in_interval = getattr(fills, "for_event_interval", None)
    if callable(in_interval):
        records = in_interval(start_ns, end_ns)
    elif isinstance(fills, (tuple, list)):
        records = fills
    else:
        raise ValueError("M2.2 snapshot lacks native fill history")
    result = []
    for fill in records:
        if not isinstance(fill, Mapping):
            raise ValueError("M2.2 native fill row is malformed")
        timestamp = fill.get("ts_event", fill.get("ts_event_ns"))
        if type(timestamp) is not int or timestamp < 0:
            raise ValueError("M2.2 native fill clock is invalid")
        if start_ns < timestamp <= end_ns:
            result.append(fill)
    result.sort(key=lambda row: str(row.get("event_id", "")))
    return tuple(result)


def _fill_fingerprint(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str, str, int]:
    event_id = row.get("event_id")
    client_order_id = row.get("client_order_id")
    timestamp = row.get("ts_event", row.get("ts_event_ns"))
    if (
        not isinstance(event_id, str)
        or not event_id.strip()
        or not isinstance(client_order_id, str)
        or not client_order_id.strip()
        or type(timestamp) is not int
    ):
        raise ValueError("native fill lacks event, order, or clock identity")
    quantity = row.get("last_qty", row.get("quantity"))
    price = row.get("last_px", row.get("price"))
    fee = row.get("commission", row.get("fee"))
    if quantity is None or price is None or fee is None:
        raise ValueError("native fill lacks quantity, price, or fee facts")
    side = row.get("order_side")
    fee_parts = str(fee).split()
    if not isinstance(side, str) or side not in {"BUY", "SELL"} or len(fee_parts) != 2:
        raise ValueError("native fill side or commission currency is invalid")
    try:
        quantity_value = Decimal(str(quantity))
        price_value = Decimal(str(price))
        fee_value = Decimal(fee_parts[0])
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("native fill economics are invalid") from exc
    if (
        not quantity_value.is_finite()
        or not price_value.is_finite()
        or not fee_value.is_finite()
        or quantity_value <= 0
        or price_value <= 0
        or not fee_parts[1].strip()
    ):
        raise ValueError("native fill economics are outside valid bounds")

    def canonical(value: Decimal) -> str:
        return str(value.normalize())

    return (
        event_id,
        client_order_id,
        side,
        canonical(quantity_value),
        canonical(price_value),
        canonical(fee_value),
        fee_parts[1],
        timestamp,
    )


def verify_m22_order_facts(
    barriers: Sequence[AuditedTeacherBarrier],
    order_facts: Sequence[Any],
    snapshots: Sequence[Any],
) -> M22OrderFactsSummary:
    """Reconcile Gym order snapshots/fills against the independent M2.2 session snapshots.

    Historical factories retain M2.2 account snapshots at each stable barrier.
    This join checks the exact before/after order status and policy-fill window
    from the Gym projection against those retained native snapshots; it does
    not use the Gym receipt hash as proof by itself.
    """
    if (
        not barriers
        or len(barriers) != len(order_facts)
        or not isinstance(snapshots, (tuple, list))
        or not snapshots
    ):
        raise ValueError("Route A barriers, order facts, and retained M2.2 snapshots are required")
    by_clock: dict[int, Any] = {}
    last_clock = 0
    for snapshot in snapshots:
        timestamp = getattr(snapshot, "decision_ts_ns", None)
        if type(timestamp) is not int or timestamp <= 0 or timestamp < last_clock:
            raise ValueError("M2.2 snapshots have missing or regressing stable barrier clocks")
        # The native session may materialize several snapshots at the same
        # clock while it drains event callbacks. The final retained snapshot
        # for that clock is the stable account state consumed by Gym.step().
        by_clock[timestamp] = snapshot
        last_clock = timestamp

    compared_orders: set[str] = set()
    compared_fills: set[str] = set()
    for barrier, facts in zip(barriers, order_facts, strict=True):
        before_ns = barrier.observation.decision_ts_ns
        after_ns = barrier.trace.execution_barrier_ts_ns
        if after_ns <= before_ns:
            raise ValueError("Route A order fact has an invalid M2.2 barrier interval")
        before_snapshot, after_snapshot = by_clock.get(before_ns), by_clock.get(after_ns)
        if before_snapshot is None or after_snapshot is None:
            raise ValueError("M2.2 session snapshots do not cover a Route A policy barrier")
        expected_before = tuple(dict(row) for row in facts.native_order_statuses_before)
        expected_after = tuple(dict(row) for row in facts.native_order_statuses_after)
        before_ids = {row["client_order_id"] for row in expected_before}
        after_ids = {row["client_order_id"] for row in expected_after}
        if (
            _snapshot_status_rows(before_snapshot, before_ids, label="pre-barrier")
            != expected_before
        ):
            raise ValueError("Gym pre-barrier order states differ from independent M2.2 snapshot")
        if _snapshot_status_rows(after_snapshot, after_ids, label="post-barrier") != expected_after:
            raise ValueError("Gym post-barrier order states differ from independent M2.2 snapshot")
        pending_ids = tuple(
            sorted(
                str(row["client_order_id"])
                for row in getattr(after_snapshot, "orders", ())
                if isinstance(row, Mapping) and isinstance(row.get("client_order_id"), str)
            )
        )
        if pending_ids != facts.pending_order_ids_after:
            raise ValueError("Gym pending-order identities differ from independent M2.2 snapshot")
        expected_fills = tuple(
            sorted(
                (_fill_fingerprint(row) for row in facts.native_policy_fills),
                key=lambda value: value[0],
            )
        )
        actual_fills = tuple(
            sorted(
                (
                    _fill_fingerprint(row)
                    for row in _snapshot_fills(
                        after_snapshot,
                        start_ns=before_ns,
                        end_ns=after_ns,
                    )
                ),
                key=lambda value: value[0],
            )
        )
        if expected_fills != actual_fills:
            raise ValueError("Gym policy fills differ from independent M2.2 snapshot events")
        if compared_fills & {item[0] for item in actual_fills}:
            raise ValueError("M2.2 policy fill event was assigned to multiple barriers")
        compared_fills.update(item[0] for item in actual_fills)
        compared_orders.update(before_ids | after_ids)

    return M22OrderFactsSummary(
        barriers=len(barriers),
        order_status_facts=len(compared_orders),
        policy_fill_events=len(compared_fills),
    )


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"M2.2 decision trace is missing numeric {key}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"M2.2 decision trace has invalid numeric {key}") from exc
    if not isfinite(result):
        raise ValueError(f"M2.2 decision trace has nonfinite {key}")
    return result


def _same_number(actual: float, row: Mapping[str, Any], key: str, tolerance: float) -> bool:
    return isclose(actual, _number(row, key), rel_tol=tolerance, abs_tol=tolerance)


def verify_m22_decision_trace(
    barriers: Sequence[AuditedTeacherBarrier],
    execution_decision_trace: Sequence[Mapping[str, Any]],
    *,
    policy_barrier_equity_by_id: Mapping[str, Mapping[str, Any]],
    tolerance: float = 1e-8,
) -> M22DecisionTraceSummary:
    """Require one-to-one decision identity and action/equity agreement.

    The host must call this with ``report["execution_decision_trace"]`` from
    ``rollout(include_execution_trace=True)`` for the same deterministic run.
    Native order lifecycle details are checked by the separate M2.2 authority
    verifier; this function checks that both reports refer to the same action
    decision and policy-barrier facts.
    """
    if not barriers or not execution_decision_trace:
        raise ValueError("Route A barriers and M2.2 execution decision trace are required")
    if not isinstance(policy_barrier_equity_by_id, Mapping):
        raise TypeError("same-run Gym policy-barrier equity facts are required")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("invalid M2.2 comparison tolerance")

    route_by_id: dict[str, AuditedTeacherBarrier] = {}
    for barrier in barriers:
        identity = barrier.trace.execution_decision_id
        if not identity or identity in route_by_id:
            raise ValueError("Route A barrier has missing or duplicate execution decision identity")
        route_by_id[identity] = barrier

    m22_by_id: dict[str, Mapping[str, Any]] = {}
    for row in execution_decision_trace:
        if not isinstance(row, Mapping):
            raise TypeError("M2.2 decision trace rows must be mappings")
        row_identity = row.get("decision_id")
        if (
            not isinstance(row_identity, str)
            or not row_identity.strip()
            or row_identity in m22_by_id
        ):
            raise ValueError("M2.2 trace has missing or duplicate decision identity")
        m22_by_id[row_identity] = row
    if (
        route_by_id.keys() != m22_by_id.keys()
        or route_by_id.keys() != policy_barrier_equity_by_id.keys()
    ):
        raise ValueError("Route A and M2.2 decision identities do not form an exact match")

    submits = related_refs = 0
    for identity, barrier in route_by_id.items():
        native = barrier.trace
        row = m22_by_id[identity]
        policy_equity = policy_barrier_equity_by_id[identity]
        if not isinstance(policy_equity, Mapping):
            raise TypeError("same-run Gym policy-barrier equity row must be a mapping")
        required_text = {
            "plan_id": native.execution_plan_id,
            "idempotency_key": native.execution_idempotency_key,
            "action": native.planner_action,
            "reason": native.planner_reason,
            "equity_currency": native.fee_currency,
        }
        for key, expected in required_text.items():
            if not isinstance(expected, str) or not expected or row.get(key) != expected:
                raise ValueError(f"M2.2 decision {identity} disagrees on {key}")
        if (
            not native.execution_barrier_ts_ns
            or row.get("decision_ts_ns") != native.execution_barrier_ts_ns
        ):
            raise ValueError(f"M2.2 decision {identity} disagrees on execution barrier clock")

        client_id = (
            native.submitted_order_ids[0]
            if native.planner_action == "SUBMIT" and len(native.submitted_order_ids) == 1
            else None
        )
        if row.get("client_order_id") != client_id:
            raise ValueError(f"M2.2 decision {identity} disagrees on submitted order identity")
        related = row.get("related_order_ids")
        if (
            not isinstance(related, (tuple, list))
            or tuple(related) != native.related_existing_order_ids
        ):
            raise ValueError(f"M2.2 decision {identity} disagrees on related order identities")
        related_refs += len(related)
        submits += native.planner_action == "SUBMIT"

        numeric_pairs = (
            (native.requested_action, "actor_requested_target"),
            (native.risk_adjusted_target, "hard_risk_target"),
            (native.planner_target, "quantity_rounded_target"),
            (native.actual_exposure_after, "executed_exposure"),
            (native.native_equity_before, "policy_equity_before"),
        )
        for actual, key in numeric_pairs:
            if not _same_number(actual, row, key, tolerance):
                raise ValueError(f"M2.2 decision {identity} disagrees on {key}")

        # Compare M2.2 against the exact same-run Gym policy barrier. A terminal
        # Route A trace carries settled equity instead, which is independently
        # bound by L6 and therefore cannot supply this policy-barrier value.
        for key in ("policy_equity_before", "policy_equity_after", "policy_equity_change"):
            if not _same_number(_number(policy_equity, key), row, key, tolerance):
                raise ValueError(f"M2.2 decision {identity} disagrees on {key}")
        if not _same_number(
            native.native_equity_before, policy_equity, "policy_equity_before", tolerance
        ):
            raise ValueError(f"Route A decision {identity} disagrees on policy_equity_before")
        if barrier.terminal_l6_receipt is None and not _same_number(
            native.native_equity_after, policy_equity, "policy_equity_after", tolerance
        ):
            raise ValueError(f"Route A decision {identity} disagrees on policy_equity_after")

        soft = row.get("soft_target")
        if native.soft_risk_target is None:
            if soft is not None:
                raise ValueError(f"M2.2 decision {identity} has an unprojected SoftRisk target")
        elif soft is None or not isclose(
            native.soft_risk_target,
            _number(row, "soft_target"),
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ValueError(f"M2.2 decision {identity} disagrees on SoftRisk target")

        expected_change = _number(policy_equity, "policy_equity_after") - _number(
            policy_equity, "policy_equity_before"
        )
        if not _same_number(expected_change, row, "policy_equity_change", tolerance):
            raise ValueError(f"M2.2 decision {identity} disagrees on policy equity change")

    return M22DecisionTraceSummary(
        decisions=len(route_by_id),
        submits=int(submits),
        related_order_references=related_refs,
    )
