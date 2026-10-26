"""Staged, causal decision records and opportunity funnels for 1m teachers."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Protocol

from rles_route_a.contracts import ExpertAdvice, Intent, PointInTimeObservation
from rles_route_a.indicators_1m import MINUTE_NS
from rles_route_a.teachers_1m.contracts import RawSignal


BarStatus = Literal["READY", "WARMUP", "GAP", "STALE"]
PositionSide = Literal["LONG", "SHORT", "FLAT"]
CaseStatus = Literal[
    "NONE", "ORDINARY_COMPLETE", "OPEN_CENSORED", "TERMINAL_CENSORED", "LIQUIDATION"
]


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _row_side(row: TeacherDecisionRecord) -> PositionSide:
    raw_signal = row.pre.raw_signal
    signal_side: PositionSide | None = (
        "LONG" if raw_signal in ("LONG_ENTRY", "EXIT_LONG")
        else "SHORT" if raw_signal in ("SHORT_ENTRY", "EXIT_SHORT")
        else None
    )
    position_side: PositionSide = (
        "LONG" if row.pre.position_before > 1e-8
        else "SHORT" if row.pre.position_before < -1e-8
        else "FLAT"
    )
    if (signal_side is not None and position_side != "FLAT"
            and signal_side != position_side):
        raise ValueError("teacher raw signal side conflicts with native position side")
    return signal_side or position_side


@dataclass(frozen=True, slots=True)
class TeacherPreDecision:
    teacher_id: str
    recipe_id: str
    fold_id: str
    source_identity: str
    decision_ts_ns: int
    latest_1m_close_ts_ns: int
    latest_1m_available_at_ns: int
    bar_status: BarStatus
    indicator_values: Mapping[str, float]
    raw_signal: RawSignal
    raw_signal_reason: str
    position_before: float
    native_pending_orders: int
    risk_veto_reason: str | None
    order_state_veto_reason: str | None
    exit_policy_id: str
    scenario_id: str
    economic_status: str
    block_id: str

    def __post_init__(self) -> None:
        if any(not value for value in (
            self.teacher_id, self.recipe_id, self.fold_id, self.source_identity,
            self.raw_signal_reason, self.exit_policy_id, self.scenario_id,
            self.economic_status, self.block_id,
        )):
            raise ValueError("decision record identities and reasons are required")
        if (type(self.decision_ts_ns) is not int or type(self.latest_1m_close_ts_ns) is not int
                or type(self.latest_1m_available_at_ns) is not int
                or self.latest_1m_close_ts_ns % MINUTE_NS != 0
                or not 0 < self.latest_1m_close_ts_ns <= self.latest_1m_available_at_ns
                <= self.decision_ts_ns):
            raise ValueError("decision record violates causal clock ordering")
        if self.bar_status not in ("READY", "WARMUP", "GAP", "STALE"):
            raise ValueError("invalid 1m bar status")
        if self.raw_signal not in (
            "NONE", "LONG_ENTRY", "SHORT_ENTRY", "EXIT_LONG", "EXIT_SHORT"
        ):
            raise ValueError("invalid raw signal")
        if type(self.native_pending_orders) is not int or self.native_pending_orders < 0:
            raise ValueError("pending order count must be non-negative")
        _finite(self.position_before, "position_before")
        normalized = {
            name: _finite(value, f"indicator.{name}")
            for name, value in self.indicator_values.items()
        }
        object.__setattr__(self, "indicator_values", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class TeacherPlannerRecord:
    advice_intent: Intent
    requested_exposure: float
    risk_adjusted_target: float
    planner_target: float
    planner_status: str
    planner_reason: str
    submitted_order_ids: tuple[str, ...] = ()
    related_existing_order_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.advice_intent not in (
            "FLAT", "OPEN_LONG", "OPEN_SHORT", "HOLD", "ADD", "REDUCE", "EXIT"
        ) or not self.planner_status or not self.planner_reason:
            raise ValueError("invalid planner decision")
        for name in ("requested_exposure", "risk_adjusted_target", "planner_target"):
            value = _finite(getattr(self, name), name)
            if not -1 <= value <= 1:
                raise ValueError(f"{name} outside NETTING exposure contract")
        for name in ("submitted_order_ids", "related_existing_order_ids"):
            order_ids = getattr(self, name)
            if (len(order_ids) != len(set(order_ids))
                    or any(not order_id for order_id in order_ids)):
                raise ValueError(f"invalid {name}")
        if self.planner_status not in {"SUBMIT", "NOOP", "WAIT", "CANCEL"}:
            raise ValueError("unknown native planner status")
        if ((self.planner_status == "SUBMIT") != bool(self.submitted_order_ids)):
            raise ValueError("native SUBMIT status and order identities disagree")
        if set(self.submitted_order_ids) & set(self.related_existing_order_ids):
            raise ValueError("new and existing native order identities overlap")


@dataclass(frozen=True, slots=True)
class TeacherExecutionRecord:
    execution_ts_ns: int
    native_filled_order_ids: tuple[str, ...]
    native_exposure_after: float
    fee: float
    fee_currency: str
    equity_currency: str
    slippage_cost: float
    funding: float
    account_equity: float
    mfe_bps: float
    mae_bps: float
    position_age_minutes: float
    case_status: CaseStatus = "NONE"
    exit_reason: str | None = None
    case_id: str | None = None
    pending_order_ids_after: tuple[str, ...] = ()
    unfilled_order_reasons: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.execution_ts_ns) is not int or self.execution_ts_ns <= 0:
            raise ValueError("execution clock must be positive")
        if (len(self.native_filled_order_ids) != len(set(self.native_filled_order_ids))
                or any(not order_id for order_id in self.native_filled_order_ids)):
            raise ValueError("invalid filled native order ids")
        for name in (
            "native_exposure_after", "fee", "slippage_cost", "funding", "account_equity",
            "mfe_bps", "mae_bps", "position_age_minutes",
        ):
            _finite(getattr(self, name), name)
        if self.fee < 0 or self.slippage_cost < 0 or self.account_equity <= 0:
            raise ValueError("costs must be non-negative and equity positive")
        if (not self.fee_currency or not self.equity_currency
                or self.fee_currency != self.equity_currency):
            raise ValueError("fee and account equity must share one explicit currency")
        if self.position_age_minutes < 0:
            raise ValueError("position age cannot be negative")
        if self.case_status not in (
            "NONE", "ORDINARY_COMPLETE", "OPEN_CENSORED", "TERMINAL_CENSORED", "LIQUIDATION"
        ):
            raise ValueError("invalid trade-case status")
        if self.case_status == "NONE":
            if self.case_id is not None:
                raise ValueError("case id requires an explicit lifecycle status")
        elif not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("lifecycle status requires a native trade case identity")
        if (not isinstance(self.pending_order_ids_after, tuple)
                or len(self.pending_order_ids_after) != len(set(self.pending_order_ids_after))
                or any(not order_id for order_id in self.pending_order_ids_after)):
            raise ValueError("invalid native pending order identities after execution")
        if not isinstance(self.unfilled_order_reasons, tuple):
            raise ValueError("unfilled native order reasons must be immutable pairs")
        reason_ids: set[str] = set()
        for pair in self.unfilled_order_reasons:
            if (not isinstance(pair, tuple) or len(pair) != 2
                    or not all(isinstance(value, str) and value.strip() for value in pair)
                    or pair[0] in reason_ids):
                raise ValueError("unfilled native orders require unique nonempty reason pairs")
            reason_ids.add(pair[0])


@dataclass(frozen=True, slots=True)
class TeacherDecisionRecord:
    pre: TeacherPreDecision
    planner: TeacherPlannerRecord | None = None
    execution: TeacherExecutionRecord | None = None

    def __post_init__(self) -> None:
        if self.execution is not None and self.planner is None:
            raise ValueError("post-execution facts require a planner record")
        if self.execution is not None and self.execution.execution_ts_ns <= self.pre.decision_ts_ns:
            raise ValueError("native execution fact cannot precede or equal its decision")
        if (self.planner is not None and self.execution is not None
                and set(self.execution.native_filled_order_ids)
                - (set(self.planner.submitted_order_ids)
                   | set(self.planner.related_existing_order_ids))):
            raise ValueError("fill ids require submitted order provenance or a related existing order")


@dataclass(frozen=True, slots=True)
class TeacherFunnel1m:
    teacher_id: str
    exit_policy_id: str
    scenario_id: str
    block_id: str
    side: PositionSide
    decisions: int
    eligible_decision_bars: int
    raw_long_signals: int
    raw_short_signals: int
    exit_signals: int
    entry_signals_while_flat: int
    submitted_entries: int
    filled_entries: int
    submitted_exits: int
    filled_exits: int
    ordinary_flat_to_flat_cases: int
    terminal_censored_cases: int
    open_censored_cases: int
    unfilled_or_missed_cases: int
    liquidations: int
    veto_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if any(not value for value in (
            self.teacher_id, self.exit_policy_id, self.scenario_id, self.block_id,
        )):
            raise ValueError("funnel group identities are required")
        if self.side not in ("LONG", "SHORT", "FLAT"):
            raise ValueError("funnel direction must be explicit")
        values = (
            self.decisions, self.eligible_decision_bars, self.raw_long_signals,
            self.raw_short_signals, self.exit_signals, self.entry_signals_while_flat,
            self.submitted_entries, self.filled_entries, self.submitted_exits,
            self.filled_exits, self.ordinary_flat_to_flat_cases,
            self.terminal_censored_cases, self.open_censored_cases,
            self.unfilled_or_missed_cases, self.liquidations,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("funnel counts must be non-negative integers")
        object.__setattr__(self, "veto_counts", MappingProxyType(dict(self.veto_counts)))


def build_teacher_funnel_1m(
    records: Iterable[TeacherDecisionRecord], *,
    side: PositionSide | None = None,
    associated_fill_order_ids: Iterable[str] = (),
) -> TeacherFunnel1m:
    rows = tuple(records)
    if not rows:
        raise ValueError("teacher funnel requires decision rows")

    keys = {
        (
            row.pre.teacher_id, row.pre.exit_policy_id, row.pre.scenario_id,
            row.pre.block_id,
        )
        for row in rows
    }
    if len(keys) != 1:
        raise ValueError("one funnel cannot mix teacher, exit, or scenario identities")
    teacher_id, exit_policy_id, scenario_id, block_id = next(iter(keys))
    observed_sides = {_row_side(row) for row in rows}
    if side is None:
        if len(observed_sides) != 1:
            raise ValueError("one funnel cannot mix LONG, SHORT, and FLAT decisions")
        side = next(iter(observed_sides))
    elif side not in ("LONG", "SHORT", "FLAT") or not observed_sides.issubset({side}):
        raise ValueError("funnel side does not match its decision rows")
    if side is None:
        raise AssertionError("funnel side refinement failed")
    vetoes: Counter[str] = Counter()
    raw_long = raw_short = exits = eligible_entries = 0
    submitted_entry_ids: set[str] = set()
    submitted_exit_ids: set[str] = set()
    filled_ids = set(associated_fill_order_ids) | {
        order_id for row in rows if row.execution is not None
        for order_id in row.execution.native_filled_order_ids
    }
    complete_ids: set[str] = set()
    terminal_ids: set[str] = set()
    open_censored_ids: set[str] = set()
    liquidation_ids: set[str] = set()
    missed = 0
    for row in rows:
        pre, planner, execution = row.pre, row.planner, row.execution
        raw_long += pre.raw_signal == "LONG_ENTRY" and side == "LONG"
        raw_short += pre.raw_signal == "SHORT_ENTRY" and side == "SHORT"
        exits += pre.raw_signal in ("EXIT_LONG", "EXIT_SHORT")
        is_entry = pre.raw_signal in ("LONG_ENTRY", "SHORT_ENTRY")
        is_flat = abs(pre.position_before) <= 1e-8
        veto_reasons = tuple(
            reason for reason in (pre.risk_veto_reason, pre.order_state_veto_reason)
            if reason is not None
        )
        vetoes.update(veto_reasons)
        entry_eligible = (
            is_entry and is_flat and not veto_reasons and pre.bar_status == "READY"
        )
        eligible_entries += entry_eligible and _row_side(row) == side
        submitted = bool(planner and planner.submitted_order_ids)
        if (planner and planner.advice_intent in ("OPEN_LONG", "OPEN_SHORT")
                and submitted and _row_side(row) == side):
            submitted_entry_ids.update(planner.submitted_order_ids)
        if planner and planner.advice_intent == "EXIT" and submitted and _row_side(row) == side:
            submitted_exit_ids.update(planner.submitted_order_ids)
        if entry_eligible and _row_side(row) == side:
            entry_orders = (set(planner.submitted_order_ids)
                            if planner and planner.advice_intent in ("OPEN_LONG", "OPEN_SHORT")
                            else set())
            if not entry_orders.intersection(filled_ids):
                missed += 1
        if execution and execution.case_id and _row_side(row) == side:
            if execution.case_status == "ORDINARY_COMPLETE":
                complete_ids.add(execution.case_id)
            elif execution.case_status == "TERMINAL_CENSORED":
                terminal_ids.add(execution.case_id)
            elif execution.case_status == "OPEN_CENSORED":
                open_censored_ids.add(execution.case_id)
            elif execution.case_status == "LIQUIDATION":
                liquidation_ids.add(execution.case_id)
    return TeacherFunnel1m(
        teacher_id=teacher_id,
        exit_policy_id=exit_policy_id,
        scenario_id=scenario_id,
        block_id=block_id,
        side=side,
        decisions=len(rows),
        eligible_decision_bars=sum(row.pre.bar_status == "READY" for row in rows),
        raw_long_signals=raw_long,
        raw_short_signals=raw_short,
        exit_signals=exits,
        entry_signals_while_flat=eligible_entries,
        submitted_entries=len(submitted_entry_ids),
        filled_entries=len(submitted_entry_ids & filled_ids),
        submitted_exits=len(submitted_exit_ids),
        filled_exits=len(submitted_exit_ids & filled_ids),
        ordinary_flat_to_flat_cases=len(complete_ids),
        terminal_censored_cases=len(terminal_ids),
        open_censored_cases=len(open_censored_ids),
        unfilled_or_missed_cases=missed,
        liquidations=len(liquidation_ids),
        veto_counts=vetoes,
    )


def group_teacher_funnels_1m(
    records: Iterable[TeacherDecisionRecord],
) -> tuple[TeacherFunnel1m, ...]:
    groups: dict[tuple[str, str, str, str], list[TeacherDecisionRecord]] = {}
    for row in records:
        key = (
            row.pre.teacher_id, row.pre.exit_policy_id,
            row.pre.scenario_id, row.pre.block_id,
        )
        groups.setdefault(key, []).append(row)
    funnels = []
    for key in sorted(groups):
        account_rows = groups[key]
        all_fill_ids = {
            order_id for row in account_rows if row.execution is not None
            for order_id in row.execution.native_filled_order_ids
        }
        for side in ("LONG", "SHORT", "FLAT"):
            side_rows = tuple(row for row in account_rows if _row_side(row) == side)
            if side_rows:
                funnels.append(build_teacher_funnel_1m(
                    side_rows,
                    side=side,
                    associated_fill_order_ids=all_fill_ids,
                ))
    return tuple(funnels)


class TeacherFunnelResearchHook1m(Protocol):
    """Optional staged sidecar: capture pre-decision, then native post-step facts.

    ``commit`` is called only after the existing host M2.2 decision and retained
    order snapshots reconcile. The hook owns no market reader or account ledger.
    """

    def capture_pre_decision(
        self, observation: PointInTimeObservation, advice: ExpertAdvice,
        visible_context: Mapping[str, Any], teacher: object,
    ) -> TeacherPreDecision: ...

    def capture_post_decision(
        self, pre: TeacherPreDecision, advice: ExpertAdvice,
        gym_info: Mapping[str, Any], barrier: Any, order_facts: Any,
    ) -> TeacherDecisionRecord: ...

    def build_native_cases(
        self, records: tuple[TeacherDecisionRecord, ...],
    ) -> tuple[Any, ...]: ...

    def commit(
        self, records: tuple[TeacherDecisionRecord, ...], cases: tuple[Any, ...],
    ) -> None: ...


def validate_staged_teacher_decision(
    record: TeacherDecisionRecord, *, observation: PointInTimeObservation,
    advice: ExpertAdvice, gym_info: Mapping[str, Any], barrier: Any, order_facts: Any,
) -> None:
    """Bind staged rows to exact same-step native L3/L6 facts."""
    if not isinstance(record, TeacherDecisionRecord) or record.pre.decision_ts_ns != observation.decision_ts_ns:
        raise ValueError("teacher research hook returned a mismatched pre-decision record")
    if (record.pre.teacher_id != advice.expert_id
            or record.pre.decision_ts_ns != advice.decision_ts_ns
            or abs(record.pre.position_before - observation.account_value("net_exposure")) > 1e-8):
        raise ValueError("teacher research pre-decision differs from the host observation/advice")
    try:
        exact_close_ts_ns = observation.exact_market_clock("latest_1m_close_ts_ns")
    except KeyError as exc:
        raise ValueError("native 1m research requires an exact integer completed-bar clock") from exc
    if exact_close_ts_ns != record.pre.latest_1m_close_ts_ns:
        raise ValueError("teacher research close clock differs from exact host market metadata")
    trace = getattr(barrier, "trace", None)
    if trace is None:
        raise TypeError("teacher research hook requires an audited native barrier")
    planner, execution = record.planner, record.execution
    if planner is None or execution is None:
        raise ValueError("native research row must be completed only after its host step")
    if (planner.advice_intent != advice.intent
            or abs(planner.requested_exposure - advice.requested_exposure) > 1e-8
            or abs(planner.risk_adjusted_target - trace.risk_adjusted_target) > 1e-8
            or abs(planner.planner_target - trace.planner_target) > 1e-8
            or planner.planner_status != trace.planner_action
            or planner.planner_reason != trace.planner_reason
            or planner.submitted_order_ids != trace.submitted_order_ids
            or planner.related_existing_order_ids != trace.related_existing_order_ids):
        raise ValueError("teacher planner row differs from the existing native trace")
    native_fills = tuple(sorted(order_facts.filled_order_ids))
    if (execution.native_filled_order_ids != native_fills
            or execution.execution_ts_ns != trace.execution_barrier_ts_ns
            or abs(execution.account_equity - trace.native_equity_after) > 1e-8
            or execution.equity_currency != trace.fee_currency
            or execution.fee_currency != trace.fee_currency
            or execution.pending_order_ids_after != trace.pending_order_ids_after):
        raise ValueError("teacher execution row differs from same-step native L3/L6 facts")
    unfilled_submitted = set(planner.submitted_order_ids) - set(execution.native_filled_order_ids)
    reason_by_order = dict(execution.unfilled_order_reasons)
    for order_id in unfilled_submitted:
        if (order_id not in execution.pending_order_ids_after
                and order_id not in reason_by_order):
            raise ValueError(
                "INCONSISTENT_NATIVE_ACCOUNTING: unfilled submitted order lacks a native "
                "pending state or explicit rejection/cancellation reason"
            )
    if set(reason_by_order) - unfilled_submitted:
        raise ValueError("unfilled-order reasons reference a filled or unsubmitted order")
    raw_policy_fills = getattr(order_facts, "native_policy_fills", None)
    if not isinstance(raw_policy_fills, tuple):
        raise ValueError("native L3 fill-event facts are required for fee reconciliation")
    fees_by_currency: dict[str, Decimal] = {}
    for fill in raw_policy_fills:
        if not isinstance(fill, Mapping):
            raise ValueError("malformed native L3 fill event for fee reconciliation")
        commission = fill.get("commission", fill.get("fee"))
        if not isinstance(commission, str):
            raise ValueError("native fill commission is missing for fee reconciliation")
        parts = commission.split()
        if len(parts) != 2 or not parts[1]:
            raise ValueError("native fill commission requires amount and currency")
        try:
            amount = Decimal(parts[0])
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("native fill commission amount is invalid") from exc
        if not amount.is_finite() or amount < 0:
            raise ValueError("native fill commission must be finite and non-negative")
        fees_by_currency[parts[1]] = fees_by_currency.get(parts[1], Decimal(0)) + amount
    if len(fees_by_currency) > 1:
        raise ValueError("native fill commissions span currencies and cannot be summed")
    fee_currency = next(iter(fees_by_currency), trace.fee_currency)
    native_fee = next(iter(fees_by_currency.values()), Decimal(0))
    try:
        reported_fee = Decimal(str(execution.fee))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("reported native fee is invalid") from exc
    if (fee_currency != execution.fee_currency
            or abs(native_fee - reported_fee) > Decimal("0.00000001")):
        raise ValueError("teacher fee differs from same-barrier native fill commissions")
    raw_funding = gym_info.get("native_funding")
    if not isinstance(raw_funding, (tuple, list)):
        raise ValueError("native funding events are required for funding reconciliation")
    funding_by_currency: dict[str, Decimal] = {}
    for event in raw_funding:
        if not isinstance(event, Mapping):
            raise ValueError("malformed native funding event")
        change = event.get("pnl_change")
        if not isinstance(change, str):
            raise ValueError("native funding PnL change is missing")
        parts = change.split()
        if len(parts) != 2 or not parts[1]:
            raise ValueError("native funding PnL change requires amount and currency")
        try:
            amount = Decimal(parts[0])
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("native funding amount is invalid") from exc
        if not amount.is_finite():
            raise ValueError("native funding amount must be finite")
        funding_by_currency[parts[1]] = funding_by_currency.get(parts[1], Decimal(0)) + amount
    if len(funding_by_currency) > 1:
        raise ValueError("native funding spans currencies and cannot be summed")
    funding_currency = next(iter(funding_by_currency), trace.fee_currency)
    native_funding = next(iter(funding_by_currency.values()), Decimal(0))
    try:
        reported_funding = Decimal(str(execution.funding))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("reported native funding is invalid") from exc
    if (funding_currency != execution.equity_currency
            or abs(native_funding - reported_funding) > Decimal("0.00000001")):
        raise ValueError("teacher funding differs from same-barrier native funding")

