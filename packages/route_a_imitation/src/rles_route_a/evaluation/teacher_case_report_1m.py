"""Native-fact trade cases and account-level summaries for 1m teachers."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
import json
from math import fsum, isfinite
from pathlib import Path
import re
from typing import Literal

from rles_route_a.evaluation.teacher_funnel_1m import (
    TeacherDecisionRecord,
    group_teacher_funnels_1m,
)


TradeCaseStatus = Literal["ORDINARY_COMPLETE", "OPEN_CENSORED", "TERMINAL_CENSORED"]


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class NativeTradeCase1m:
    trade_case_id: str
    signal_id: str
    teacher_id: str
    exit_policy_id: str
    scenario_id: str
    block_id: str
    side: Literal[-1, 1]
    status: TradeCaseStatus
    entry_decision_ts_ns: int
    entry_signal_price: float
    entry_fill_ts_ns: int
    entry_fill_price: float
    entry_notional: float
    exit_signal_price: float | None
    exit_fill_ts_ns: int | None
    exit_fill_price: float | None
    fees: float
    fee_currency: str
    equity_currency: str
    slippage_cost: float
    funding: float
    realized_pnl: float
    equity_start: float
    equity_end: float
    mfe_bps: float
    mae_bps: float
    holding_minutes: float
    exit_reason: str | None
    economic_status: str
    candidate_event_id: str | None = None
    candidate_teacher_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not value for value in (
            self.trade_case_id, self.signal_id, self.teacher_id, self.exit_policy_id,
            self.scenario_id, self.block_id, self.economic_status,
            self.fee_currency, self.equity_currency,
        )):
            raise ValueError("trade case identities are required")
        if self.side not in (-1, 1) or self.status not in (
            "ORDINARY_COMPLETE", "OPEN_CENSORED", "TERMINAL_CENSORED"
        ):
            raise ValueError("invalid trade case side or status")
        if (type(self.entry_decision_ts_ns) is not int or type(self.entry_fill_ts_ns) is not int
                or self.entry_fill_ts_ns <= self.entry_decision_ts_ns):
            raise ValueError("entry fill must follow its causal decision")
        for name in (
            "entry_signal_price", "entry_fill_price", "entry_notional", "fees",
            "slippage_cost", "funding", "realized_pnl", "equity_start", "equity_end",
            "mfe_bps", "mae_bps", "holding_minutes",
        ):
            _finite(getattr(self, name), name)
        if (min(self.entry_signal_price, self.entry_fill_price, self.entry_notional,
                self.equity_start, self.equity_end) <= 0
                or self.fees < 0 or self.slippage_cost < 0 or self.holding_minutes < 0):
            raise ValueError("trade prices/notional/equity must be positive and costs non-negative")
        if self.fee_currency != self.equity_currency:
            raise ValueError("trade fees and account equity must use the same currency")
        if (self.candidate_event_id is None) != (not self.candidate_teacher_ids):
            raise ValueError("candidate provenance requires both event and teacher identities")
        if (self.candidate_event_id is not None
                and (not isinstance(self.candidate_teacher_ids, tuple)
                     or not re.fullmatch(r"[0-9a-f]{64}", self.candidate_event_id)
                     or len(self.candidate_teacher_ids) != len(set(self.candidate_teacher_ids))
                     or any(not isinstance(name, str) or not name.strip()
                            for name in self.candidate_teacher_ids))):
            raise ValueError("native candidate provenance is invalid")
        if self.status == "ORDINARY_COMPLETE":
            if (self.exit_signal_price is None or self.exit_fill_ts_ns is None
                    or self.exit_fill_price is None or not self.exit_reason
                    or self.exit_fill_ts_ns <= self.entry_fill_ts_ns):
                raise ValueError("complete native case requires a later exit signal and fill")
            if self.exit_reason == "EPISODE_TERMINAL_FLATTEN":
                raise ValueError("terminal flatten cannot be an ordinary complete case")
            if self.exit_signal_price <= 0 or self.exit_fill_price <= 0:
                raise ValueError("exit prices must be positive")
        elif any(value is not None for value in (
            self.exit_signal_price, self.exit_fill_ts_ns, self.exit_fill_price,
        )):
            raise ValueError("censored case cannot fabricate an exit signal or fill")

    @property
    def price_return(self) -> float | None:
        if self.exit_fill_price is None:
            return None
        return self.side * (self.exit_fill_price / self.entry_fill_price - 1.0)

    @property
    def notional_net_return(self) -> float:
        return self.realized_pnl / self.entry_notional

    @property
    def account_return(self) -> float:
        return self.equity_end / self.equity_start - 1.0


@dataclass(frozen=True, slots=True)
class TeacherCaseSummary1m:
    teacher_id: str
    exit_policy_id: str
    scenario_id: str
    block_id: str
    equity_currency: str
    total_cases: int
    ordinary_complete: int
    open_censored: int
    terminal_censored: int
    winners: int
    losers: int
    neutral: int
    gross_profit: float
    gross_loss: float
    profit_factor: float | None
    fees: float
    slippage_cost: float
    funding: float
    net_pnl: float
    account_return: float
    max_mfe_bps: float | None
    min_mae_bps: float | None
    holding_minutes: tuple[float, ...]


def summarize_native_trade_cases_1m(
    cases: tuple[NativeTradeCase1m, ...],
) -> TeacherCaseSummary1m:
    if not cases:
        raise ValueError("native case summary requires at least one case")
    keys = {
        (case.teacher_id, case.exit_policy_id, case.scenario_id, case.block_id)
        for case in cases
    }
    if len(keys) != 1:
        raise ValueError("one case summary cannot mix teacher, exit, or scenario identities")
    teacher_id, exit_policy_id, scenario_id, block_id = next(iter(keys))
    currencies = {case.equity_currency for case in cases}
    if len(currencies) != 1:
        raise ValueError("one case summary cannot mix account currencies")
    completed = tuple(case for case in cases if case.status == "ORDINARY_COMPLETE")
    profits = tuple(case.realized_pnl for case in completed if case.realized_pnl > 0)
    losses = tuple(case.realized_pnl for case in completed if case.realized_pnl < 0)
    gross_profit = fsum(profits)
    gross_loss = fsum(losses)
    profit_factor = None if gross_loss == 0 else gross_profit / abs(gross_loss)
    equity_start = cases[0].equity_start
    equity_end = cases[-1].equity_end
    return TeacherCaseSummary1m(
        teacher_id=teacher_id,
        exit_policy_id=exit_policy_id,
        scenario_id=scenario_id,
        block_id=block_id,
        equity_currency=next(iter(currencies)),
        total_cases=len(cases),
        ordinary_complete=len(completed),
        open_censored=sum(case.status == "OPEN_CENSORED" for case in cases),
        terminal_censored=sum(case.status == "TERMINAL_CENSORED" for case in cases),
        winners=len(profits),
        losers=len(losses),
        neutral=sum(case.realized_pnl == 0 for case in completed),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        fees=fsum(case.fees for case in cases),
        slippage_cost=fsum(case.slippage_cost for case in cases),
        funding=fsum(case.funding for case in cases),
        net_pnl=fsum(case.realized_pnl for case in cases),
        account_return=equity_end / equity_start - 1.0,
        max_mfe_bps=max((case.mfe_bps for case in cases), default=None),
        min_mae_bps=min((case.mae_bps for case in cases), default=None),
        holding_minutes=tuple(case.holding_minutes for case in cases),
    )


def group_native_trade_case_summaries_1m(
    cases: tuple[NativeTradeCase1m, ...],
) -> tuple[TeacherCaseSummary1m, ...]:
    groups: dict[tuple[str, str, str, str], list[NativeTradeCase1m]] = {}
    for case in cases:
        key = (case.teacher_id, case.exit_policy_id, case.scenario_id, case.block_id)
        groups.setdefault(key, []).append(case)
    return tuple(
        summarize_native_trade_cases_1m(tuple(groups[key])) for key in sorted(groups)
    )


def verify_teacher_native_accounting_1m(
    records: tuple[TeacherDecisionRecord, ...],
    cases: tuple[NativeTradeCase1m, ...],
    *,
    expected_submitted_order_ids: tuple[str, ...],
    expected_filled_order_ids: tuple[str, ...],
    expected_case_ids: tuple[str, ...],
) -> None:
    """Fail closed unless sidecar identities exactly match host-native evidence."""
    if not records or any(row.planner is None or row.execution is None for row in records):
        raise ValueError("native account report requires one completed post-step row per decision")
    submitted = {
        order_id for row in records if row.planner is not None
        for order_id in row.planner.submitted_order_ids
    }
    filled = {
        order_id for row in records if row.execution is not None
        for order_id in row.execution.native_filled_order_ids
    }
    case_ids = tuple(case.trade_case_id for case in cases)
    duplicate_cases = len(case_ids) != len(set(case_ids))
    expected_submitted = set(expected_submitted_order_ids)
    expected_filled = set(expected_filled_order_ids)
    expected_cases = set(expected_case_ids)
    if (duplicate_cases
            or len(expected_submitted) != len(expected_submitted_order_ids)
            or len(expected_filled) != len(expected_filled_order_ids)
            or len(expected_cases) != len(expected_case_ids)
            or submitted != expected_submitted or filled != expected_filled
            or set(case_ids) != expected_cases):
        raise ValueError(
            "INCONSISTENT_NATIVE_ACCOUNTING: submitted/fill/case identities differ "
            "from retained trace and L6 case evidence"
        )
_DECISION_COLUMNS = (
    "teacher_id", "recipe_id", "fold_id", "source_identity", "block_id",
    "scenario_id", "exit_policy_id", "decision_ts_ns", "latest_1m_close_ts_ns",
    "latest_1m_available_at_ns", "bar_status", "raw_signal", "raw_signal_reason",
    "position_before", "native_pending_orders", "risk_veto_reason",
    "order_state_veto_reason", "indicator_values_json", "planner_intent",
    "requested_exposure", "risk_adjusted_target", "planner_target", "planner_status",
    "planner_reason",
    "submitted_order_ids_json", "related_existing_order_ids_json", "execution_ts_ns",
    "filled_order_ids_json", "native_exposure_after", "fee", "fee_currency",
    "equity_currency", "slippage_cost", "funding", "account_equity", "mfe_bps",
    "mae_bps", "position_age_minutes",
    "pending_order_ids_after_json", "unfilled_order_reasons_json", "case_id",
    "case_status", "exit_reason", "economic_status",
)


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_teacher_research_artifacts_1m(
    output_dir: str | Path,
    records: tuple[TeacherDecisionRecord, ...],
    cases: tuple[NativeTradeCase1m, ...],
    *,
    expected_submitted_order_ids: tuple[str, ...],
    expected_filled_order_ids: tuple[str, ...],
    expected_case_ids: tuple[str, ...],
) -> Path:
    """Write controlled JSON/CSV equivalents of the four frozen S05 outputs.

    CSV is used as the portable columnar equivalent where PyArrow/Parquet is
    not part of the Ubuntu environment; its exact schema is pinned above.
    The destination must not exist, so artifacts are never overwritten.
    """
    destination = Path(output_dir)
    verify_teacher_native_accounting_1m(
        records, cases,
        expected_submitted_order_ids=expected_submitted_order_ids,
        expected_filled_order_ids=expected_filled_order_ids,
        expected_case_ids=expected_case_ids,
    )
    if not records:
        raise ValueError("S05 artifacts require at least one native decision row")
    destination.mkdir(parents=True, exist_ok=False)

    funnel_payload = {
        "schema": "rles.route-a.teacher-funnel-1m.v1",
        "execution_classification": "M22_VERIFIED_NATIVE_FACTS",
        "economic_classifications": sorted({row.pre.economic_status for row in records}),
        "groups": [
            {
                field.name: (
                    dict(funnel.veto_counts)
                    if field.name == "veto_counts" else getattr(funnel, field.name)
                )
                for field in dataclass_fields(funnel)
            }
            for funnel in group_teacher_funnels_1m(records)
        ],
    }
    (destination / "teacher-funnel.json").write_text(
        _compact_json(funnel_payload) + "\n", encoding="utf-8",
    )

    with (destination / "decision-rows.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_DECISION_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in records:
            pre, planner, execution = row.pre, row.planner, row.execution
            values = {
                "teacher_id": pre.teacher_id,
                "recipe_id": pre.recipe_id,
                "fold_id": pre.fold_id,
                "source_identity": pre.source_identity,
                "block_id": pre.block_id,
                "scenario_id": pre.scenario_id,
                "exit_policy_id": pre.exit_policy_id,
                "decision_ts_ns": pre.decision_ts_ns,
                "latest_1m_close_ts_ns": pre.latest_1m_close_ts_ns,
                "latest_1m_available_at_ns": pre.latest_1m_available_at_ns,
                "bar_status": pre.bar_status,
                "raw_signal": pre.raw_signal,
                "raw_signal_reason": pre.raw_signal_reason,
                "position_before": pre.position_before,
                "native_pending_orders": pre.native_pending_orders,
                "risk_veto_reason": pre.risk_veto_reason,
                "order_state_veto_reason": pre.order_state_veto_reason,
                "indicator_values_json": _compact_json(dict(pre.indicator_values)),
                "planner_intent": None if planner is None else planner.advice_intent,
                "requested_exposure": None if planner is None else planner.requested_exposure,
                "risk_adjusted_target": None if planner is None else planner.risk_adjusted_target,
                "planner_target": None if planner is None else planner.planner_target,
                "planner_status": None if planner is None else planner.planner_status,
                "planner_reason": None if planner is None else planner.planner_reason,
                "submitted_order_ids_json": _compact_json(
                    [] if planner is None else planner.submitted_order_ids
                ),
                "related_existing_order_ids_json": _compact_json(
                    [] if planner is None else planner.related_existing_order_ids
                ),
                "execution_ts_ns": None if execution is None else execution.execution_ts_ns,
                "filled_order_ids_json": _compact_json(
                    [] if execution is None else execution.native_filled_order_ids
                ),
                "native_exposure_after": None if execution is None else execution.native_exposure_after,
                "fee": None if execution is None else execution.fee,
                "fee_currency": None if execution is None else execution.fee_currency,
                "equity_currency": None if execution is None else execution.equity_currency,
                "slippage_cost": None if execution is None else execution.slippage_cost,
                "funding": None if execution is None else execution.funding,
                "account_equity": None if execution is None else execution.account_equity,
                "mfe_bps": None if execution is None else execution.mfe_bps,
                "mae_bps": None if execution is None else execution.mae_bps,
                "position_age_minutes": None if execution is None else execution.position_age_minutes,
                "pending_order_ids_after_json": _compact_json(
                    [] if execution is None else execution.pending_order_ids_after
                ),
                "unfilled_order_reasons_json": _compact_json(
                    [] if execution is None else execution.unfilled_order_reasons
                ),
                "case_id": None if execution is None else execution.case_id,
                "case_status": None if execution is None else execution.case_status,
                "exit_reason": None if execution is None else execution.exit_reason,
                "economic_status": pre.economic_status,
            }
            writer.writerow(values)

    with (destination / "native-trades.jsonl").open("x", encoding="utf-8", newline="") as stream:
        for case in cases:
            case_row = {
                field.name: getattr(case, field.name)
                for field in dataclass_fields(case)
            }
            stream.write(_compact_json(case_row) + "\n")

    summary_payload = {
        "schema": "rles.route-a.teacher-case-summary-1m.v1",
        "economic_classifications": sorted({case.economic_status for case in cases}),
        "groups": [
            {field.name: getattr(summary, field.name) for field in dataclass_fields(summary)}
            for summary in group_native_trade_case_summaries_1m(cases)
        ],
    }
    (destination / "case-summary.json").write_text(
        _compact_json(summary_payload) + "\n", encoding="utf-8",
    )
    return destination

