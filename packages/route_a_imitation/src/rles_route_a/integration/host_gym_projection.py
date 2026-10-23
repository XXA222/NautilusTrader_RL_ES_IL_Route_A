"""Fail-closed projection from snapshot 80194a Gym.step() information.

`Gym.step` reports the POST-ADVANCE decision_ts_ns and a policy-barrier equity
pair. A teacher observation is the PRE-ADVANCE decision state; conflating those
clocks leaks the next bar into supervision. L3 order facts are supplied
separately because Gym info does not prove fills or pending order closure.

This module neither reads data nor imports/modifies NautilusTrader or RLES.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping, Any

from rles_route_a.contracts import ExpertAdvice, NativeActionTrace, PointInTimeObservation
from .native_economics import NativeRewardReceipt
from .trace_audit import AuditedTeacherBarrier


def _finite(raw: object, field: str) -> float:
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"missing or invalid native {field}")
    try:
        # Round-trip via Decimal so large numeric strings fail explicitly.
        dec = Decimal(str(raw))
        val = float(dec)
    except (ValueError, TypeError, InvalidOperation, OverflowError) as exc:
        raise ValueError(f"invalid native {field}") from exc
    if not dec.is_finite() or not isfinite(val):
        raise ValueError(f"nonfinite native {field}")
    return val


def _ids(ids: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(ids, tuple) or len(ids) != len(set(ids)) or any(
        type(x) is not str or not x or x.strip() != x for x in ids
    ):
        raise ValueError(f"invalid L3 {name} identities")
    return ids


def _info_ids(raw: object, name: str) -> tuple[str, ...]:
    if not isinstance(raw, (tuple, list)):
        raise ValueError(f"native Gym {name} identities are missing")
    ids = tuple(raw)
    if any(type(value) is not str or not value or value.strip() != value
           for value in ids):
        raise ValueError(f"invalid native Gym {name} identities")
    if len(ids) != len(set(ids)):
        raise ValueError(f"invalid native Gym {name} identities")
    return ids


@dataclass(frozen=True, slots=True)
class L3OrderFacts:
    """Signed-off L3/M2.2 native order evidence, never derived from SUBMIT.

    `filled_order_ids` may include historical orders only when listed as
    `related_existing_order_ids`.  It does not represent unique FillEvents or
    amount filled; partials must be blocked by the residual/pending label gate.
    """
    submitted_order_ids: tuple[str, ...] = ()
    filled_order_ids: tuple[str, ...] = ()
    related_existing_order_ids: tuple[str, ...] = ()
    pending_order_ids_after: tuple[str, ...] = ()
    native_evidence_receipt: str = ""
    native_order_statuses_before: tuple[Mapping[str, str], ...] = ()
    native_order_statuses_after: tuple[Mapping[str, str], ...] = ()
    native_order_status_receipt: str = ""
    native_policy_fills: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in ("submitted_order_ids", "filled_order_ids",
                     "related_existing_order_ids", "pending_order_ids_after"):
            _ids(getattr(self, name), name)
        if not isinstance(self.native_evidence_receipt, str) or not self.native_evidence_receipt.strip():
            raise ValueError("L3 native evidence receipt required")
        if set(self.filled_order_ids) - (
                set(self.submitted_order_ids) | set(self.related_existing_order_ids)):
            raise ValueError("filled order missing L3 provenance")
        before = _validate_order_statuses(self.native_order_statuses_before, "before")
        after = _validate_order_statuses(self.native_order_statuses_after, "after")
        if self.native_order_status_receipt:
            expected = _order_status_receipt(before, after)
            if self.native_order_status_receipt != expected:
                raise ValueError("native order status receipt mismatch")
        elif before or after:
            raise ValueError("native order status identity receipt required")
        if (not isinstance(self.native_policy_fills, tuple)
                or any(not isinstance(row, Mapping) for row in self.native_policy_fills)):
            raise ValueError("native policy-barrier fills must be a tuple of native mappings")
        fill_ids: set[str] = set()
        known_order_ids = set(self.submitted_order_ids) | set(self.related_existing_order_ids)
        for fill in self.native_policy_fills:
            event_id = fill.get("event_id")
            client_order_id = fill.get("client_order_id")
            ts_event = fill.get("ts_event", fill.get("ts_event_ns"))
            if (not isinstance(event_id, str) or not event_id.strip()
                    or event_id in fill_ids
                    or not isinstance(client_order_id, str) or not client_order_id.strip()
                    or client_order_id not in known_order_ids
                    or type(ts_event) is not int or ts_event < 0):
                raise ValueError("native policy-barrier fill identity, provenance, or clock is invalid")
            fill_ids.add(event_id)


def _validate_order_statuses(rows: tuple[Mapping[str, str], ...], label: str
                             ) -> tuple[dict[str, str], ...]:
    if not isinstance(rows, tuple):
        raise ValueError(f"native order status {label} must be a tuple")
    normalized: list[dict[str, str]] = []
    identities: set[str] = set()
    required = {"client_order_id", "status", "side", "quantity", "filled_qty"}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError(f"native order status {label} has an invalid schema")
        item = dict(row)
        if any(type(value) is not str or not value.strip() for value in item.values()):
            raise ValueError(f"native order status {label} has missing native fields")
        identity = item["client_order_id"]
        if identity in identities:
            raise ValueError(f"duplicate native order status {label} identity")
        identities.add(identity)
        normalized.append(item)
    if tuple(row["client_order_id"] for row in normalized) != tuple(sorted(identities)):
        raise ValueError(f"native order status {label} rows are not deterministically ordered")
    return tuple(normalized)


def _order_status_receipt(before: tuple[dict[str, str], ...],
                           after: tuple[dict[str, str], ...]) -> str:
    encoded = json.dumps({"before": before, "after": after}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def l3_order_facts_from_gym_info(gym_info: Mapping[str, Any]) -> L3OrderFacts:
    """Build order facts only from the Gym bridge's native per-barrier trace."""
    if not isinstance(gym_info, Mapping):
        raise TypeError("native Gym info mapping required")
    action = gym_info.get("execution_plan_action")
    if action not in {"SUBMIT", "NOOP", "WAIT", "CANCEL"}:
        raise ValueError("unrecognized native planner action")
    submitted = gym_info.get("execution_client_order_id")
    if action == "SUBMIT":
        if type(submitted) is not str or not submitted.strip():
            raise ValueError("native SUBMIT has no client order identity")
        submitted_ids: tuple[str, ...] = (submitted,)
    else:
        if submitted is not None:
            raise ValueError("non-SUBMIT barrier cannot claim a new order")
        submitted_ids = ()
    related = gym_info.get("execution_related_order_ids", ())
    before_pending = gym_info.get("native_pending_order_ids_before")
    pending_after = gym_info.get("native_pending_order_ids_after")
    if not isinstance(related, (tuple, list)):
        raise ValueError("native pre-barrier order lifecycle is missing")
    related_ids = _info_ids(related, "related order")
    before_pending_ids = _info_ids(before_pending, "pre-barrier pending order")
    pending_after_ids = _info_ids(pending_after, "post-barrier pending order")
    if action in {"WAIT", "CANCEL"} and not set(related_ids).issubset(set(before_pending_ids)):
        raise ValueError("native related order is absent from the pre-barrier snapshot")
    if action not in {"WAIT", "CANCEL"} and related_ids:
        raise ValueError("planner action has unexpected related order identities")
    raw_fills = gym_info.get("native_policy_fills")
    if not isinstance(raw_fills, tuple):
        raise ValueError("native policy-barrier fill events are missing")
    filled_ids = []
    expected_sources = set(submitted_ids) | set(related_ids)
    for fill in raw_fills:
        if not isinstance(fill, Mapping):
            raise ValueError("native policy fill must be a mapping")
        identity = fill.get("client_order_id")
        if not isinstance(identity, str) or not identity.strip() or identity not in expected_sources:
            raise ValueError("native policy fill lacks current or related order provenance")
        filled_ids.append(identity)
    statuses_before = _validate_order_statuses(
        gym_info.get("native_order_statuses_before", ()), "before",
    )
    statuses_after = _validate_order_statuses(
        gym_info.get("native_order_statuses_after", ()), "after",
    )
    status_ids = {row["client_order_id"] for row in statuses_after}
    required_after = expected_sources | set(pending_after_ids)
    if not required_after.issubset(status_ids):
        raise ValueError("native order status trace omits a referenced policy order")
    if not set(before_pending_ids).issubset(
            {row["client_order_id"] for row in statuses_before}):
        raise ValueError("native pre-barrier status trace omits a pending order")
    receipt = gym_info.get("native_order_status_receipt")
    if type(receipt) is not str or not receipt:
        raise ValueError("native M2.2 order status receipt is missing")
    return L3OrderFacts(
        submitted_order_ids=submitted_ids,
        filled_order_ids=tuple(sorted(set(filled_ids))),
        related_existing_order_ids=related_ids,
        pending_order_ids_after=pending_after_ids,
        native_evidence_receipt=receipt,
        native_order_statuses_before=statuses_before,
        native_order_statuses_after=statuses_after,
        native_order_status_receipt=receipt,
        native_policy_fills=tuple(dict(row) for row in raw_fills),
    )


@dataclass(frozen=True, slots=True)
class HostGymProjection:
    """Pre-decision trace and policy-barrier reward with separate post time."""
    teacher_barrier: AuditedTeacherBarrier
    advance_ts_ns: int
    policy_currency: str
    native_evidence_receipt: str


def terminal_settlement_sha256(settlement: Mapping[str, Any]) -> str:
    """Hash the exact JSON-native Gym terminal settlement facts for L6 binding."""
    if not isinstance(settlement, Mapping):
        raise TypeError("native terminal settlement mapping required")
    try:
        payload = json.dumps(dict(settlement), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("terminal settlement facts must be finite JSON data") from exc
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class TerminalL6Receipt:
    """Host-issued L6 certification binding settled equity to Gym terminal facts.

    The caller must obtain this receipt from its project accounting/access
    authority; Route A does not mint receipts or infer external cashflows.
    """
    receipt_id: str
    source_identity: str
    settlement_sha256: str
    policy_barrier_ts_ns: int
    settlement_barrier_ts_ns: int
    equity_before: float
    policy_equity_after: float
    settled_equity_after: float
    currency: str
    gym_recorded_r_econ: float
    recorded_r_econ: float
    has_external_cashflow: bool = False
    cashflow_adjustment_receipt: str | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in
                   (self.receipt_id, self.source_identity, self.currency)):
            raise ValueError("terminal L6 receipt identity, source, and currency are required")
        for name in ("settlement_sha256",):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(
                    ch not in "0123456789abcdef" for ch in value):
                raise ValueError("terminal L6 settlement digest must be SHA-256")
        if (type(self.policy_barrier_ts_ns) is not int or self.policy_barrier_ts_ns <= 0
                or type(self.settlement_barrier_ts_ns) is not int
                or self.settlement_barrier_ts_ns < self.policy_barrier_ts_ns):
            raise ValueError("terminal L6 clocks are invalid")
        for name in ("equity_before", "policy_equity_after", "settled_equity_after"):
            if _finite(getattr(self, name), name) <= 0:
                raise ValueError(f"terminal L6 {name} must be positive")
        _finite(self.gym_recorded_r_econ, "gym_recorded_r_econ")
        _finite(self.recorded_r_econ, "recorded_r_econ")
        if type(self.has_external_cashflow) is not bool:
            raise TypeError("terminal L6 cashflow flag must be explicit")
        if self.has_external_cashflow:
            if (not isinstance(self.cashflow_adjustment_receipt, str)
                    or not self.cashflow_adjustment_receipt.strip()):
                raise ValueError("terminal external cashflow requires an L6 adjustment receipt")
        elif self.cashflow_adjustment_receipt is not None:
            raise ValueError("no-flow terminal receipt cannot carry a cashflow adjustment")


def project_terminal_teacher_barrier(*, observation: PointInTimeObservation,
                                     advice: ExpertAdvice,
                                     gym_info: Mapping[str, Any],
                                     l3_orders: L3OrderFacts,
                                     terminal_receipt: TerminalL6Receipt,
                                     execution_recipe: str,
                                     source_identity: str,
                                     scenario_only: bool = True) -> HostGymProjection:
    """Project a terminal decision only when host L6 binds the final settlement.

    Policy fills/orders remain scoped to the policy barrier; force-flat terminal
    fills stay in the hashed terminal settlement record and are not relabelled
    as teacher-action fills.
    """
    if not isinstance(terminal_receipt, TerminalL6Receipt):
        raise TypeError("host-issued terminal L6 receipt required")
    if not isinstance(gym_info, Mapping) or gym_info.get("terminal_settled") is not True:
        raise ValueError("settled native terminal Gym decision required")
    settlement = gym_info.get("terminal_settlement")
    if not isinstance(settlement, Mapping):
        raise ValueError("native terminal settlement facts are missing")
    if terminal_receipt.source_identity != source_identity:
        raise ValueError("terminal L6 receipt source identity mismatch")
    if terminal_receipt.settlement_sha256 != terminal_settlement_sha256(settlement):
        raise ValueError("terminal L6 receipt does not bind Gym settlement facts")
    policy_ts = gym_info.get("decision_ts_ns")
    settlement_ts = gym_info.get("settlement_after_ns")
    if (type(policy_ts) is not int or type(settlement_ts) is not int
            or terminal_receipt.policy_barrier_ts_ns != policy_ts
            or terminal_receipt.settlement_barrier_ts_ns != settlement_ts
            or settlement.get("policy_barrier_ts_ns") != policy_ts
            or settlement.get("settlement_barrier_ts_ns") != settlement_ts):
        raise ValueError("terminal L6/Gym clock binding mismatch")
    if settlement.get("post_terminal_open_position_count") != 0:
        raise ValueError("terminal Gym settlement has an open position")
    if settlement.get("equity_currency") != gym_info.get("policy_equity_currency"):
        raise ValueError("terminal Gym equity currency changed")
    before = _finite(gym_info.get("policy_equity_before"), "policy_equity_before")
    policy_after = _finite(gym_info.get("policy_equity_after"), "policy_equity_after")
    settled_after = _finite(settlement.get("equity_after"), "terminal equity_after")
    settlement_before = _finite(settlement.get("equity_before"), "terminal equity_before")
    delta = _finite(settlement.get("equity_delta"), "terminal equity_delta")
    if (abs(before - terminal_receipt.equity_before) > 1e-8
            or abs(policy_after - terminal_receipt.policy_equity_after) > 1e-8
            or abs(settled_after - terminal_receipt.settled_equity_after) > 1e-8
            or abs(settlement_before - policy_after) > max(1e-8, abs(policy_after) * 1e-10)
            or abs(settled_after - settlement_before - delta) > max(1e-8, abs(settlement_before) * 1e-10)):
        raise ValueError("terminal L6/Gym equity conservation mismatch")
    if terminal_receipt.currency != gym_info.get("policy_equity_currency"):
        raise ValueError("terminal L6 receipt currency mismatch")
    if abs(terminal_receipt.gym_recorded_r_econ
           - _finite(gym_info.get("r_econ"), "r_econ")) > 1e-10:
        raise ValueError("terminal L6 receipt does not bind native Gym reward")
    if abs(_finite(settlement.get("pre_terminal_net_exposure"), "pre_terminal_net_exposure")
           - _finite(gym_info.get("policy_executed_action_pre_terminal"),
                     "policy_executed_action_pre_terminal")) > 1e-8:
        raise ValueError("terminal settlement does not bind policy exposure")

    # Reuse the nonterminal action/order contract with the L6-settled equity
    # pair. For cashflow-adjusted rewards, pass a conservation-valid temporary
    # value to the shared policy projection, then replace it with L6's bound
    # recorded reward and validate its cashflow receipt.
    normalized_info = dict(gym_info)
    normalized_info.pop("terminal_settlement", None)
    normalized_info["terminal_settled"] = False
    normalized_info["policy_equity_after"] = str(settled_after)
    normalized_info["policy_equity_change"] = str(settled_after - before)
    if terminal_receipt.has_external_cashflow:
        from math import log
        normalized_info["r_econ"] = log(settled_after) - log(before)
    projection = project_nonterminal_teacher_barrier(
        observation=observation, advice=advice, gym_info=normalized_info,
        l3_orders=l3_orders, execution_recipe=execution_recipe,
        source_identity=source_identity,
        no_external_cashflow_receipt=terminal_receipt.receipt_id,
        scenario_only=scenario_only,
    )
    reward = NativeRewardReceipt(
        before, settled_after, terminal_receipt.recorded_r_econ,
        terminal_receipt.has_external_cashflow,
        terminal_receipt.cashflow_adjustment_receipt,
    )
    reward.validate()
    barrier = replace(projection.teacher_barrier, reward=reward,
                      terminal_l6_receipt=terminal_receipt.receipt_id,
                      terminal_settlement_ts_ns=terminal_receipt.settlement_barrier_ts_ns)
    return replace(projection, teacher_barrier=barrier)


def project_nonterminal_teacher_barrier(*, observation: PointInTimeObservation,
                                        advice: ExpertAdvice,
                                        gym_info: Mapping[str, Any],
                                        l3_orders: L3OrderFacts,
                                        execution_recipe: str,
                                        source_identity: str,
                                        no_external_cashflow_receipt: str,
                                        scenario_only: bool = True) -> HostGymProjection:
    """Project a real nonterminal Gym step, never a synthetic account event.

    The host must pass the L2/L3 *pre-step* observation, the existing Gym info,
    and independent M2.2 order evidence. A terminal Gym step is deliberately
    rejected: its r_econ may include extra force-flat settlement beyond the
    `policy_equity_after` barrier. A separate L6 terminal receipt is required.
    """
    if not isinstance(observation, PointInTimeObservation) or not isinstance(advice, ExpertAdvice):
        raise TypeError("point-in-time observation and teacher advice required")
    if not isinstance(gym_info, Mapping) or not isinstance(l3_orders, L3OrderFacts):
        raise TypeError("native Gym mapping and L3 order facts required")
    if not all(isinstance(v, str) and v.strip() for v in
               (execution_recipe, source_identity, no_external_cashflow_receipt)):
        raise ValueError("frozen source, execution recipe, and L6 no-cashflow receipt required")
    if gym_info.get("terminal_settled") is not False or "terminal_settlement" in gym_info:
        raise ValueError("terminal settlement needs a distinct native L6 equity receipt")
    if type(scenario_only) is not bool:
        raise TypeError("scenario_only must be explicit boolean")
    if advice.decision_ts_ns != observation.decision_ts_ns:
        raise ValueError("expert advice/pre-step clock mismatch")
    pending_before = _info_ids(
        gym_info.get("native_pending_order_ids_before"), "pre-barrier pending order",
    )
    pending_after = _info_ids(
        gym_info.get("native_pending_order_ids_after"), "post-barrier pending order",
    )
    if set(pending_before) != set(observation.pending_order_ids):
        raise ValueError("Gym pre-barrier pending orders differ from teacher observation")
    if set(pending_after) != set(l3_orders.pending_order_ids_after):
        raise ValueError("Gym post-barrier pending orders differ from independent L3 facts")
    if type(gym_info.get("decision_ts_ns")) is not int:
        raise ValueError("post-advance Gym clock missing")
    advance = gym_info["decision_ts_ns"]
    if advance <= observation.decision_ts_ns:
        raise ValueError("Gym advance clock did not exceed pre-step decision clock")
    action = gym_info.get("execution_plan_action")
    if action not in {"SUBMIT", "NOOP", "CANCEL", "WAIT"}:
        raise ValueError("unrecognized native planner action")
    for key in ("execution_decision_id", "execution_plan_id",
                "execution_idempotency_key", "execution_plan_reason",
                "evidence_hash", "policy_equity_currency"):
        if not isinstance(gym_info.get(key), str) or not gym_info[key].strip():
            raise ValueError(f"missing native {key}")
    req = _finite(gym_info.get("requested_action"), "requested_action")
    if abs(req - advice.requested_exposure) > 1e-6:
        raise ValueError("Gym policy action differs from teacher request")
    expected_submit = gym_info.get("execution_client_order_id")
    if action == "SUBMIT":
        if type(expected_submit) is not str or expected_submit not in l3_orders.submitted_order_ids:
            raise ValueError("native SUBMIT has no matching L3 order receipt")
    elif expected_submit is not None:
        raise ValueError("non-SUBMIT barrier cannot claim submitted client order")
    declared_cancel = gym_info.get("execution_cancel_order_ids", ())
    if (not isinstance(declared_cancel, (tuple, list)) or
            any(type(x) is not str or not x or x.strip() != x for x in declared_cancel) or
            len(set(declared_cancel)) != len(declared_cancel)):
        raise ValueError("invalid native cancellation facts")
    if action == "CANCEL" and not set(declared_cancel).issubset(
            set(l3_orders.related_existing_order_ids)):
        raise ValueError("cancel IDs missing L3 existing-order provenance")
    related = gym_info.get("execution_related_order_ids", ())
    if not isinstance(related, (tuple, list)) or any(
            type(x) is not str or not x or x.strip() != x for x in related):
        raise ValueError("invalid native related-order IDs")
    if action in {"CANCEL", "WAIT"} and not set(related).issubset(
            set(l3_orders.related_existing_order_ids)):
        raise ValueError("WAIT/CANCEL related orders missing original L3 evidence")
    if action != "SUBMIT" and l3_orders.submitted_order_ids:
        raise ValueError("L3 submit evidence inconsistent with native planner status")
    before = _finite(gym_info.get("policy_equity_before"), "policy_equity_before")
    after = _finite(gym_info.get("policy_equity_after"), "policy_equity_after")
    change = _finite(gym_info.get("policy_equity_change"), "policy_equity_change")
    if abs((after - before) - change) > max(1e-7, abs(before) * 1e-10):
        raise ValueError("Gym policy equity change not conserved")
    raw_soft_target = gym_info.get("risk_soft_target")
    soft_target = (None if raw_soft_target is None
                   else _finite(raw_soft_target, "risk_soft_target"))
    # Do not silently use the post-terminal net_exposure field: current active
    # exposure at policy barrier is the explicit pre-terminal executed action.
    trace = NativeActionTrace(
        decision_ts_ns=observation.decision_ts_ns,
        requested_action=req,
        risk_adjusted_target=_finite(gym_info.get("risk_adjusted_target"), "risk_adjusted_target"),
        planner_target=_finite(gym_info.get("quantity_rounded_target_exposure"),
                               "quantity_rounded_target_exposure"),
        actual_exposure_after=_finite(gym_info.get("policy_executed_action_pre_terminal"),
                                      "policy_executed_action_pre_terminal"),
        native_equity_before=before, native_equity_after=after,
        planner_action=action,
        submitted_order_ids=l3_orders.submitted_order_ids,
        filled_order_ids=l3_orders.filled_order_ids,
        related_existing_order_ids=l3_orders.related_existing_order_ids,
        pending_order_ids_after=l3_orders.pending_order_ids_after,
        fee_currency=gym_info["policy_equity_currency"],
        scenario_only=scenario_only,
        soft_risk_target=soft_target,
        execution_decision_id=gym_info["execution_decision_id"],
        execution_plan_id=gym_info["execution_plan_id"],
        execution_idempotency_key=gym_info["execution_idempotency_key"],
        planner_reason=gym_info["execution_plan_reason"],
        execution_barrier_ts_ns=advance,
    )
    # Host L6 must explicitly assert that this policy barrier has no external
    # transfers. A lone Gym equity ratio cannot distinguish returns/deposits.
    # `r_econ` is log(policy equity change) for NONTERMINAL no-flow steps only.
    receipt = NativeRewardReceipt(before, after,
                                  _finite(gym_info.get("r_econ"), "r_econ"), False)
    receipt.validate()
    return HostGymProjection(
        AuditedTeacherBarrier(observation, advice, trace, receipt, execution_recipe,
                              source_identity,
                              native_policy_fills=l3_orders.native_policy_fills),
        advance, gym_info["policy_equity_currency"],
        l3_orders.native_evidence_receipt,
    )
