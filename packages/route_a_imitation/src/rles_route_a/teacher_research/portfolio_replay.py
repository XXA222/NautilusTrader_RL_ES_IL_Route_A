"""Causal single-position admission for counterfactual portfolio replays.

The selector deliberately has no outcome/return argument: future realized
labels cannot decide whether an earlier candidate was admitted.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Literal, Mapping, Sequence

from .candidate_events import OpportunityEvent
from .causal_exit import TrendInvalidationExit
from .opportunity_models import OpportunityOutcome
from .opportunity_labels import CounterfactualTradeReplay
from rles_route_a.contracts import PointInTimeObservation


MINUTE_NS = 60_000_000_000


@dataclass(frozen=True, slots=True)
class ResearchPortfolioRecipe:
    """Frozen scenario account settings; never an exchange/native authority."""

    scenario_id: str
    target_fraction: float
    starting_equity: float
    fee_bps_per_leg: float
    slippage_bps_per_leg: float
    entry_cadence_minutes: int
    terminal_policy: Literal["mark_only", "scenario_force_flat"] = "mark_only"

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("research portfolio scenario identity is required")
        numeric = (self.target_fraction, self.starting_equity,
                   self.fee_bps_per_leg, self.slippage_bps_per_leg)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in numeric):
            raise ValueError("research portfolio recipe values must be finite")
        if not 0 < self.target_fraction <= 1 or self.starting_equity <= 0:
            raise ValueError("research portfolio target/equity is outside its valid range")
        if self.fee_bps_per_leg < 0 or self.slippage_bps_per_leg < 0:
            raise ValueError("research portfolio costs cannot be negative")
        if type(self.entry_cadence_minutes) is not int or self.entry_cadence_minutes <= 0:
            raise ValueError("research portfolio entry cadence must be positive minutes")
        if self.terminal_policy not in ("mark_only", "scenario_force_flat"):
            raise ValueError("terminal policy must distinguish mark-only from scenario force-flat")


@dataclass(frozen=True, slots=True)
class ResearchMinuteBar:
    open_ts_ns: int
    close_available_at_ns: int
    open_price: float
    close_price: float

    def __post_init__(self) -> None:
        if (type(self.open_ts_ns) is not int or type(self.close_available_at_ns) is not int
                or self.open_ts_ns <= 0 or self.close_available_at_ns <= self.open_ts_ns):
            raise ValueError("research bar clocks must be positive and causal")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0
               for value in (self.open_price, self.close_price)):
            raise ValueError("research bar prices must be finite and positive")


@dataclass(frozen=True, slots=True)
class ResearchFundingSettlement:
    ts_ns: int
    rate: float
    mark_ts_ns: int
    mark_price: float

    def __post_init__(self) -> None:
        if (type(self.ts_ns) is not int or type(self.mark_ts_ns) is not int
                or self.ts_ns <= 0 or not 0 < self.mark_ts_ns < self.ts_ns):
            raise ValueError("funding clocks must be strictly causal and positive")
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in (self.rate, self.mark_price))
                or self.mark_price <= 0):
            raise ValueError("funding rate/mark must be finite and the mark positive")


@dataclass(frozen=True, slots=True)
class ResearchEquitySnapshot:
    ts_ns: int
    cash_usdt: float
    position_qty: float
    position_entry_ts_ns: int | None
    mark_price: float
    cumulative_fees_usdt: float
    cumulative_slippage_usdt: float
    cumulative_funding_cashflow_usdt: float
    realized_pnl_usdt: float
    unrealized_pnl_usdt: float
    equity_usdt: float
    peak_equity_usdt: float
    drawdown_usdt: float
    pending_order: str | None


@dataclass(frozen=True, slots=True)
class CandidateDecisionRecord:
    event_id: str
    decision_ts_ns: int
    selected_by_frozen_model: bool
    status: str
    execution_ts_ns: int | None


@dataclass(frozen=True, slots=True)
class ResearchPortfolioResult:
    start_equity_usdt: float
    end_equity_usdt: float
    equity_return: float
    realized_trade_count: int
    open_censored_count: int
    terminal_force_flat_count: int
    filled_order_count: int
    total_turnover_usdt: float
    fees_paid_usdt: float
    funding_cashflow_signed_usdt: float
    slippage_cost_usdt: float
    exposure_minutes: float
    max_drawdown_usdt: float
    candidate_decision_trace: tuple[CandidateDecisionRecord, ...]
    equity_timeline: tuple[ResearchEquitySnapshot, ...]
    economic_status: Literal["SCENARIO_ONLY"] = "SCENARIO_ONLY"
    simulator_classification: Literal["RESEARCH_SCENARIO_NOT_NATIVE_FILLS"] = (
        "RESEARCH_SCENARIO_NOT_NATIVE_FILLS"
    )


@dataclass(slots=True)
class _ResearchPosition:
    qty: float
    entry_price: float
    entry_ts_ns: int
    exit_requested: bool = False


@dataclass(frozen=True, slots=True)
class _PendingResearchOrder:
    side: Literal[-1, 1]
    action: Literal["ENTRY", "EXIT"]
    decision_ts_ns: int
    execution_ts_ns: int
    event_ids: tuple[str, ...] = ()


def simulate_causal_research_portfolio(
    recipe: ResearchPortfolioRecipe,
    candidates: Sequence[OpportunityEvent],
    selected_ids: set[str],
    bars: Sequence[ResearchMinuteBar],
    observations: Sequence[PointInTimeObservation],
    funding: Sequence[ResearchFundingSettlement],
    exit_rule: TrendInvalidationExit,
    *,
    start_ns: int,
    end_ns: int,
    max_funding_mark_age_ns: int = MINUTE_NS,
) -> ResearchPortfolioResult:
    """Run an event-time single-position USDT scenario account.

    Only frozen candidate-selection ids are read before an entry. The function
    has no labels/outcomes argument, advances through funding, legal next-open
    fills, closed-bar marks and point-in-time exit decisions, and never applies
    a fixed holding-time stop. This is explicitly not a native fill simulator.
    """
    if (type(start_ns) is not int or type(end_ns) is not int
            or start_ns <= 0 or end_ns <= start_ns):
        raise ValueError("portfolio replay requires an authorized [start, end) interval")
    if type(max_funding_mark_age_ns) is not int or max_funding_mark_age_ns <= 0:
        raise ValueError("funding mark maximum age must be a positive integer")
    if not bars or not observations:
        raise ValueError("portfolio replay requires closed bars and exit observations")
    if not isinstance(exit_rule, TrendInvalidationExit):
        raise TypeError("portfolio replay requires the shared causal TrendInvalidationExit")

    bar_rows = tuple(bars)
    observation_rows = tuple(observations)
    funding_rows = tuple(funding)
    candidate_rows = tuple(candidates)
    if any(not isinstance(row, ResearchMinuteBar) for row in bar_rows):
        raise TypeError("portfolio bars must be ResearchMinuteBar records")
    if any(not isinstance(row, PointInTimeObservation) for row in observation_rows):
        raise TypeError("portfolio management path must use point-in-time observations")
    bar_times = tuple(row.open_ts_ns for row in bar_rows)
    if any(not start_ns <= row.open_ts_ns < end_ns
           or row.close_available_at_ns >= end_ns for row in bar_rows):
        raise ValueError("bar path crosses the authorized portfolio interval")
    if any(right - left != MINUTE_NS for left, right in zip(bar_times, bar_times[1:])):
        raise ValueError("portfolio execution path has a missing or duplicate one-minute bar")
    close_times = tuple(row.close_available_at_ns for row in bar_rows)
    if any(right - left != MINUTE_NS for left, right in zip(close_times, close_times[1:])):
        raise ValueError("portfolio close-availability path has a gap or duplicate clock")
    if close_times != tuple(row.decision_ts_ns for row in observation_rows):
        raise ValueError("every minute needs exactly one matching causal management observation")
    if any(not start_ns <= row.decision_ts_ns < end_ns for row in observation_rows):
        raise ValueError("management observation lies outside the authorized interval")
    if any(not start_ns <= row.ts_ns < end_ns for row in funding_rows):
        raise ValueError("funding path includes an out-of-window settlement")
    if any(row.mark_ts_ns < start_ns or row.ts_ns - row.mark_ts_ns >= max_funding_mark_age_ns
           for row in funding_rows):
        raise ValueError("funding settlement lacks a strictly fresh authorized prior mark")
    if any(right.ts_ns <= left.ts_ns for left, right in zip(funding_rows, funding_rows[1:])):
        raise ValueError("funding settlements must be strictly time ordered")

    event_by_id = {event.event_id: event for event in candidate_rows}
    if len(event_by_id) != len(candidate_rows):
        raise ValueError("candidate event identities must be unique")
    if selected_ids - event_by_id.keys():
        raise ValueError("frozen model selection references an unknown candidate")
    if any(not start_ns <= event.decision_ts_ns < end_ns
           or event.feature_available_at_ns > event.decision_ts_ns for event in candidate_rows):
        raise ValueError("candidate decision/feature clock is noncausal or out of window")
    if any(event.decision_ts_ns not in set(close_times) for event in candidate_rows):
        raise ValueError("candidate decisions must align to a closed-bar observation")
    if candidate_rows:
        cadence_ns = recipe.entry_cadence_minutes * MINUTE_NS
        phase = candidate_rows[0].decision_ts_ns % cadence_ns
        if any(event.decision_ts_ns % cadence_ns != phase for event in candidate_rows):
            raise ValueError("candidate events violate the frozen entry cadence")

    candidate_at: dict[int, list[OpportunityEvent]] = {}
    for event in candidate_rows:
        candidate_at.setdefault(event.decision_ts_ns, []).append(event)
    open_price_at = {row.open_ts_ns: row.open_price for row in bar_rows}
    open_times = tuple(open_price_at)

    def next_open_after(decision_ts_ns: int) -> int | None:
        index = bisect_right(open_times, decision_ts_ns)
        if index >= len(open_times) or open_times[index] >= end_ns:
            return None
        return open_times[index]

    # Sort by timestamp and explicit causal priority: settlement, execution,
    # closed-bar mark, then strategy decision.
    timeline_events: list[tuple[int, int, str, object]] = []
    timeline_events.extend((row.ts_ns, 0, "funding", row) for row in funding_rows)
    timeline_events.extend((row.open_ts_ns, 1, "open", row) for row in bar_rows)
    timeline_events.extend((row.close_available_at_ns, 2, "mark", row) for row in bar_rows)
    timeline_events.extend((row.decision_ts_ns, 3, "decision", row)
                           for row in observation_rows)
    timeline_events.sort(key=lambda row: (row[0], row[1]))

    cash = recipe.starting_equity
    position: _ResearchPosition | None = None
    pending: _PendingResearchOrder | None = None
    mark_price = 0.0
    mark_ts_ns: int | None = None
    realized_pnl = 0.0
    fees_paid = 0.0
    slippage_cost = 0.0
    funding_cashflow = 0.0
    turnover = 0.0
    exposure_minutes = 0.0
    realized_trade_count = 0
    terminal_force_flat_count = 0
    filled_order_count = 0
    peak_equity = recipe.starting_equity
    max_drawdown = 0.0
    snapshots: list[ResearchEquitySnapshot] = []
    decision_trace: list[CandidateDecisionRecord] = []

    def current_equity() -> tuple[float, float]:
        unrealized = (
            position.qty * (mark_price - position.entry_price)
            if position is not None and mark_price > 0 else 0.0
        )
        return cash + unrealized, unrealized

    def record(ts_ns: int) -> None:
        nonlocal peak_equity, max_drawdown
        equity, unrealized = current_equity()
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        max_drawdown = max(max_drawdown, drawdown)
        snapshots.append(ResearchEquitySnapshot(
            ts_ns=ts_ns,
            cash_usdt=cash,
            position_qty=position.qty if position is not None else 0.0,
            position_entry_ts_ns=position.entry_ts_ns if position is not None else None,
            mark_price=mark_price,
            cumulative_fees_usdt=fees_paid,
            cumulative_slippage_usdt=slippage_cost,
            cumulative_funding_cashflow_usdt=funding_cashflow,
            realized_pnl_usdt=realized_pnl,
            unrealized_pnl_usdt=unrealized,
            equity_usdt=equity,
            peak_equity_usdt=peak_equity,
            drawdown_usdt=drawdown,
            pending_order=(pending.action if pending is not None else None),
        ))

    def apply_leg_cost(notional: float) -> None:
        nonlocal cash, fees_paid, slippage_cost, turnover
        fee = notional * recipe.fee_bps_per_leg / 10_000.0
        slip = notional * recipe.slippage_bps_per_leg / 10_000.0
        cash -= fee + slip
        fees_paid += fee
        slippage_cost += slip
        turnover += abs(notional)

    def execute_pending(bar: ResearchMinuteBar) -> None:
        nonlocal position, pending, cash, realized_pnl, filled_order_count
        nonlocal realized_trade_count, exposure_minutes
        nonlocal mark_price, mark_ts_ns
        if pending is None or pending.execution_ts_ns != bar.open_ts_ns:
            return
        if pending.action == "ENTRY":
            if position is not None:
                raise RuntimeError("single-position ledger received an entry while exposed")
            equity, _ = current_equity()
            notional = equity * recipe.target_fraction
            if notional <= 0:
                raise ValueError("nonpositive equity cannot fund a new scenario position")
            qty = pending.side * notional / bar.open_price
            apply_leg_cost(notional)
            position = _ResearchPosition(qty, bar.open_price, bar.open_ts_ns)
            filled_order_count += 1
        else:
            if position is None or pending.side != (1 if position.qty > 0 else -1):
                raise RuntimeError("single-position ledger exit does not match the open side")
            notional = abs(position.qty * bar.open_price)
            pnl = position.qty * (bar.open_price - position.entry_price)
            cash += pnl
            realized_pnl += pnl
            apply_leg_cost(notional)
            exposure_minutes += (bar.open_ts_ns - position.entry_ts_ns) / MINUTE_NS
            position = None
            filled_order_count += 1
            realized_trade_count += 1
        pending = None
        mark_price = bar.open_price

    for ts_ns, _priority, kind, payload in timeline_events:
        if kind == "funding":
            settlement = payload
            if not isinstance(settlement, ResearchFundingSettlement):
                raise TypeError("funding timeline entry has the wrong DTO")
            mark_price = settlement.mark_price
            mark_ts_ns = settlement.mark_ts_ns
            if position is not None:
                cashflow = -position.qty * settlement.mark_price * settlement.rate
                cash += cashflow
                funding_cashflow += cashflow
            record(ts_ns)
        elif kind == "open":
            bar = payload
            if not isinstance(bar, ResearchMinuteBar):
                raise TypeError("open timeline entry has the wrong DTO")
            execute_pending(bar)
            mark_price = bar.open_price
            mark_ts_ns = bar.open_ts_ns
        elif kind == "mark":
            bar = payload
            if not isinstance(bar, ResearchMinuteBar):
                raise TypeError("mark timeline entry has the wrong DTO")
            mark_price = bar.close_price
            mark_ts_ns = bar.close_available_at_ns
            record(ts_ns)
        else:
            observation = payload
            if not isinstance(observation, PointInTimeObservation):
                raise TypeError("decision timeline entry has the wrong DTO")
            if position is not None and not position.exit_requested:
                if exit_rule.invalidated(
                    1 if position.qty > 0 else -1,
                    observation.market_value("trend_1h"),
                    observation.market_value("trend_1d"),
                ):
                    position.exit_requested = True
                    execution_ts = next_open_after(ts_ns)
                    if execution_ts is not None:
                        pending = _PendingResearchOrder(
                            side=1 if position.qty > 0 else -1,
                            action="EXIT",
                            decision_ts_ns=ts_ns,
                            execution_ts_ns=execution_ts,
                        )
            at_time = sorted(candidate_at.get(ts_ns, ()), key=lambda event: event.event_id)
            if at_time:
                if position is not None or pending is not None:
                    status = "POSITION_BUSY" if position is not None else "ORDER_PENDING"
                    decision_trace.extend(CandidateDecisionRecord(
                        event_id=event.event_id,
                        decision_ts_ns=ts_ns,
                        selected_by_frozen_model=event.event_id in selected_ids,
                        status=(status if event.event_id in selected_ids
                                else "FROZEN_MODEL_VETO"),
                        execution_ts_ns=None,
                    ) for event in at_time)
                else:
                    chosen = [event for event in at_time if event.event_id in selected_ids]
                    decision_trace.extend(CandidateDecisionRecord(
                        event_id=event.event_id,
                        decision_ts_ns=ts_ns,
                        selected_by_frozen_model=False,
                        status="FROZEN_MODEL_VETO",
                        execution_ts_ns=None,
                    ) for event in at_time if event.event_id not in selected_ids)
                    sides = {event.side for event in chosen}
                    if len(sides) > 1:
                        decision_trace.extend(CandidateDecisionRecord(
                            event_id=event.event_id,
                            decision_ts_ns=ts_ns,
                            selected_by_frozen_model=True,
                            status="CONFLICTING_CANDIDATE_SIDES",
                            execution_ts_ns=None,
                        ) for event in chosen)
                    elif chosen:
                        side = chosen[0].side
                        execution_ts = next_open_after(ts_ns)
                        status = "ENTRY_ORDER_SUBMITTED" if execution_ts is not None else "NO_IN_BLOCK_ENTRY"
                        if execution_ts is not None:
                            pending = _PendingResearchOrder(
                                side=side,
                                action="ENTRY",
                                decision_ts_ns=ts_ns,
                                execution_ts_ns=execution_ts,
                                event_ids=tuple(event.event_id for event in chosen),
                            )
                        decision_trace.extend(CandidateDecisionRecord(
                            event_id=event.event_id,
                            decision_ts_ns=ts_ns,
                            selected_by_frozen_model=True,
                            status=(status if index == 0 else "SAME_SIDE_DEDUPLICATED"),
                            execution_ts_ns=execution_ts,
                        ) for index, event in enumerate(chosen))
            record(ts_ns)

    if mark_ts_ns is None or mark_ts_ns < start_ns:
        raise ValueError("portfolio replay ended without an authorized mark")
    if position is not None and mark_ts_ns <= position.entry_ts_ns:
        raise ValueError("open position has no post-entry terminal mark")
    if position is not None and recipe.terminal_policy == "scenario_force_flat":
        notional = abs(position.qty * mark_price)
        pnl = position.qty * (mark_price - position.entry_price)
        cash += pnl
        realized_pnl += pnl
        apply_leg_cost(notional)
        exposure_minutes += (mark_ts_ns - position.entry_ts_ns) / MINUTE_NS
        position = None
        filled_order_count += 1
        terminal_force_flat_count = 1
        record(mark_ts_ns)
    elif position is not None:
        exposure_minutes += (mark_ts_ns - position.entry_ts_ns) / MINUTE_NS

    end_equity, _ = current_equity()
    return ResearchPortfolioResult(
        start_equity_usdt=recipe.starting_equity,
        end_equity_usdt=end_equity,
        equity_return=(end_equity - recipe.starting_equity) / recipe.starting_equity,
        realized_trade_count=realized_trade_count,
        open_censored_count=int(position is not None),
        terminal_force_flat_count=terminal_force_flat_count,
        filled_order_count=filled_order_count,
        total_turnover_usdt=turnover,
        fees_paid_usdt=fees_paid,
        funding_cashflow_signed_usdt=funding_cashflow,
        slippage_cost_usdt=slippage_cost,
        exposure_minutes=exposure_minutes,
        max_drawdown_usdt=max_drawdown,
        candidate_decision_trace=tuple(sorted(
            decision_trace, key=lambda row: (row.decision_ts_ns, row.event_id),
        )),
        equity_timeline=tuple(snapshots),
    )


@dataclass(frozen=True, slots=True)
class SequentialAdmission:
    accepted_ids: tuple[str, ...]
    busy_ids: tuple[str, ...]
    no_in_block_entry_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PortfolioTradeAccounting:
    event_id: str
    entry_ts_ns: int
    terminal_ts_ns: int
    net_return_bps: float
    status: Literal["REALIZED", "TERMINAL_MARK_ONLY"]
    fees_bps: float
    slippage_bps: float
    funding_bps: float
    exposure_minutes: float


@dataclass(frozen=True, slots=True)
class SequentialPortfolioResult:
    admission: SequentialAdmission
    trades: tuple[PortfolioTradeAccounting, ...]
    realized_net_bps: float
    terminal_mark_only_bps: float
    total_net_bps: float
    fees_bps: float
    slippage_bps: float
    funding_bps: float
    filled_orders: int
    open_censored_count: int
    exposure_minutes: float


def replay_single_position_portfolio(
    chosen_ids: set[str],
    block_events: Sequence[OpportunityEvent],
    replay_by_id: Mapping[str, CounterfactualTradeReplay],
    outcomes: Mapping[str, OpportunityOutcome],
    terminal_marks: Mapping[str, tuple[int, float]],
    *,
    block_end_ns: int,
    gross_return_method: Literal["log", "simple"] = "log",
) -> SequentialPortfolioResult:
    """Account a one-position ledger with realized/censored values separated.

    The future label is checked only after admission and never decides entry.
    Censored open positions use a last in-block mark, entry costs and accrued
    funding; they do not incur a fictitious exit fee or fill.
    """
    if gross_return_method not in ("log", "simple"):
        raise ValueError("portfolio return method must be log or simple")
    admission = admit_single_position_candidates(
        chosen_ids, block_events, replay_by_id, block_end_ns=block_end_ns,
    )
    by_id = {event.event_id: event for event in block_events}
    trades: list[PortfolioTradeAccounting] = []
    realized = 0.0
    terminal_mark = 0.0
    total_fees = 0.0
    total_slippage = 0.0
    total_funding = 0.0
    filled_orders = 0
    exposure = 0.0
    open_count = 0

    for event_id in admission.accepted_ids:
        event = by_id[event_id]
        replay = replay_by_id[event_id]
        outcome = outcomes.get(event_id)
        if outcome is None or outcome.event_id != event_id:
            raise ValueError("admitted candidate is missing its bound outcome record")
        if (
            outcome.candidate_family != event.family
            or outcome.candidate_side != event.side
            or outcome.decision_ts_ns != event.decision_ts_ns
            or outcome.feature_available_at_ns != event.feature_available_at_ns
            or outcome.rule_version != event.rule_version
            or outcome.snapshot_id != event.snapshot_id
            or outcome.feature_spec_hash != event.feature_spec_hash
            or outcome.fold_id != event.fold_id
            or outcome.source_identity != event.source_identity
            or outcome.independent_block_end_ts_ns != block_end_ns
            or outcome.simulation_receipt_id != replay.simulation_receipt_id
        ):
            raise ValueError("portfolio event and outcome provenance do not match")
        total_funding += replay.funding_cashflow_bps
        total_fees += replay.entry_fee_bps
        total_slippage += replay.entry_slippage_bps

        if replay.exit_ts_ns is not None:
            if (outcome.net_return_bps is None or replay.exit_price is None
                    or outcome.label_end_ts_ns != replay.exit_ts_ns):
                raise ValueError("closed portfolio position requires a complete outcome")
            price_ratio = replay.exit_price / replay.entry_price
            price_return = math.log(price_ratio) if gross_return_method == "log" else price_ratio - 1.0
            gross_bps = event.side * price_return * 10_000.0
            expected_net = (
                gross_bps - replay.entry_fee_bps - replay.exit_fee_bps
                - replay.entry_slippage_bps - replay.exit_slippage_bps
                + replay.funding_cashflow_bps
            )
            if not math.isclose(expected_net, outcome.net_return_bps,
                                rel_tol=0.0, abs_tol=1e-7):
                raise ValueError("portfolio cashflows differ from the bound counterfactual label")
            total_fees += replay.exit_fee_bps
            total_slippage += replay.exit_slippage_bps
            realized += expected_net
            filled_orders += 2
            terminal_ts = replay.exit_ts_ns
            status: Literal["REALIZED", "TERMINAL_MARK_ONLY"] = "REALIZED"
            net = expected_net
        else:
            outcome_censor_ts = outcome.censor_ts_ns
            if (outcome.net_return_bps is not None or outcome.label_end_ts_ns is not None
                    or replay.censor_ts_ns is None or type(outcome_censor_ts) is not int):
                raise ValueError("censored portfolio position cannot carry realized PnL")
            mark = terminal_marks.get(event_id)
            if mark is None:
                raise ValueError("censored portfolio position requires a terminal mark")
            terminal_ts, terminal_price = mark
            if (type(terminal_ts) is not int or terminal_ts <= replay.entry_ts_ns
                    or terminal_ts >= block_end_ns or terminal_price <= 0
                    or not math.isfinite(terminal_price)):
                raise ValueError("terminal mark must be finite, causal, and inside its block")
            if (outcome_censor_ts != min(replay.censor_ts_ns, block_end_ns)
                    or terminal_ts > outcome_censor_ts):
                raise ValueError("terminal mark may not follow the censor boundary")
            price_ratio = terminal_price / replay.entry_price
            price_return = math.log(price_ratio) if gross_return_method == "log" else price_ratio - 1.0
            net = (
                event.side * price_return * 10_000.0
                - replay.entry_fee_bps - replay.entry_slippage_bps
                + replay.funding_cashflow_bps
            )
            terminal_mark += net
            filled_orders += 1  # entry only; terminal mark is not a forced close
            open_count += 1
            status = "TERMINAL_MARK_ONLY"

        minutes = (terminal_ts - replay.entry_ts_ns) / 60_000_000_000
        exposure += minutes
        trades.append(PortfolioTradeAccounting(
            event_id=event_id,
            entry_ts_ns=replay.entry_ts_ns,
            terminal_ts_ns=terminal_ts,
            net_return_bps=net,
            status=status,
            fees_bps=(replay.entry_fee_bps + (replay.exit_fee_bps if status == "REALIZED" else 0.0)),
            slippage_bps=(replay.entry_slippage_bps +
                          (replay.exit_slippage_bps if status == "REALIZED" else 0.0)),
            funding_bps=replay.funding_cashflow_bps,
            exposure_minutes=minutes,
        ))

    return SequentialPortfolioResult(
        admission=admission,
        trades=tuple(trades),
        realized_net_bps=realized,
        terminal_mark_only_bps=terminal_mark,
        total_net_bps=realized + terminal_mark,
        fees_bps=total_fees,
        slippage_bps=total_slippage,
        funding_bps=total_funding,
        filled_orders=filled_orders,
        open_censored_count=open_count,
        exposure_minutes=exposure,
    )


def admit_single_position_candidates(
    chosen_ids: set[str],
    block_events: Sequence[OpportunityEvent],
    replay_by_id: Mapping[str, CounterfactualTradeReplay],
    *,
    block_end_ns: int,
) -> SequentialAdmission:
    """Admit frozen selections sequentially; do not inspect realized outcomes.

    A closed replay frees the one-position ledger only after its exit. An
    open/censored replay occupies it through the half-open block boundary.
    Entries at or after the block end are not fills and are reported
    separately. This is a research replay contract, not a native fill claim.
    """
    if type(block_end_ns) is not int or block_end_ns <= 0:
        raise ValueError("portfolio replay requires a positive block boundary")
    by_id = {event.event_id: event for event in block_events}
    if len(by_id) != len(block_events):
        raise ValueError("portfolio candidate event ids must be unique")
    if chosen_ids - by_id.keys():
        raise ValueError("portfolio selection references an event outside its block")

    candidates = sorted(
        (by_id[event_id] for event_id in chosen_ids),
        key=lambda event: (event.decision_ts_ns, event.event_id),
    )
    accepted: list[str] = []
    busy: list[str] = []
    no_entry: list[str] = []
    last_exit_ns = -1
    open_to_boundary = False
    for event in candidates:
        replay = replay_by_id.get(event.event_id)
        if not isinstance(replay, CounterfactualTradeReplay):
            raise ValueError("selected candidate is missing its replay record")
        if replay.event_id != event.event_id:
            raise ValueError("candidate and replay identities do not match")
        if replay.entry_ts_ns <= event.decision_ts_ns:
            raise ValueError("portfolio entry must be strictly after its decision")
        if replay.entry_ts_ns >= block_end_ns:
            no_entry.append(event.event_id)
            continue
        if open_to_boundary or replay.entry_ts_ns <= last_exit_ns:
            busy.append(event.event_id)
            continue
        if replay.exit_ts_ns is not None:
            if replay.exit_ts_ns <= replay.entry_ts_ns or replay.exit_ts_ns >= block_end_ns:
                raise ValueError("portfolio exit must remain inside its independent block")
            if replay.exit_price is None or replay.censor_ts_ns is not None:
                raise ValueError("closed replay needs an exit price and cannot also be censored")
            last_exit_ns = replay.exit_ts_ns
        else:
            if replay.censor_ts_ns is None or replay.censor_ts_ns > block_end_ns:
                raise ValueError("open replay must be censored no later than its block end")
            open_to_boundary = True
        accepted.append(event.event_id)
    return SequentialAdmission(tuple(accepted), tuple(busy), tuple(no_entry))
