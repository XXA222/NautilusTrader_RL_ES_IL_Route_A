"""Next-open, constant-quantity research replay of the shared ordinary exit.

This is MODEL_COUNTERFACTUAL/SCENARIO_ONLY. Independent candidates may overlap;
neither portfolio risk nor native liquidation/margin/ACK are simulated here.
"""
from __future__ import annotations

from bisect import bisect_right
from hashlib import sha256
import json
from math import fsum, isfinite

from .candidate_events import OpportunityEvent
from .causal_exit import CausalExitSchedule
from .opportunity_labels import CounterfactualTradeReplay


def attach_funding_mark_proxies(
    rates: tuple[tuple[int, float], ...], marks: tuple[tuple[int, float], ...], *,
    max_age_ns: int = 60_000_000_000,
) -> tuple[tuple[int, float, float], ...]:
    """Use the last closed mark, allowing archive settlement clock jitter.

    Never round a settlement forward into a future mark. This is explicitly
    a price proxy, not the exchange's authoritative funding notional.
    """
    times = tuple(ts for ts, _ in marks)
    if (not times or type(max_age_ns) is not int or max_age_ns <= 0
            or any(right <= left for left, right in zip(times, times[1:]))):
        raise ValueError("invalid funding mark proxy coverage")
    result = []
    for ts, rate in rates:
        index = bisect_right(times, ts) - 1
        if index < 0 or ts - times[index] >= max_age_ns:
            raise ValueError("funding settlement lacks a recent closed mark")
        price = marks[index][1]
        if not isfinite(rate) or not isfinite(price) or price <= 0:
            raise ValueError("invalid funding rate/mark")
        result.append((ts, rate, price))
    return tuple(result)


def make_causal_exit_replay(
    *, schedule: CausalExitSchedule, opens: tuple[tuple[int, float], ...],
    funding: tuple[tuple[int, float, float], ...], block_ends: dict[str, int],
    fee_bps: float, slippage_bps: float, execution_recipe: str, scenario: str,
):
    """Funding tuples are (settlement timestamp, rate, settlement mark).

    Funding at entry is excluded and funding at exit included (frozen ordering
    assumption). Both execution legs and funding use entry-notional bps.
    No post-boundary execution price or funding is accessed for a censored case.
    """
    if (not opens or not execution_recipe or not scenario
            or any(not isfinite(v) or v < 0 for v in (fee_bps, slippage_bps))):
        raise ValueError("invalid causal replay inputs")
    open_times = tuple(row[0] for row in opens)
    funding_times = tuple(row[0] for row in funding)
    for times in (open_times, funding_times):
        if any(type(ts) is not int or ts <= 0 for ts in times) or any(
            right <= left for left, right in zip(times, times[1:])
        ):
            raise ValueError("replay clocks must be strictly increasing")
    if any(not isfinite(price) or price <= 0 for _, price in opens):
        raise ValueError("execution open must be finite and positive")
    if any(right - left != 60_000_000_000 for left, right in zip(open_times, open_times[1:])):
        raise ValueError("one-minute execution path has a gap")
    if any(not isfinite(rate) or not isfinite(mark) or mark <= 0 for _, rate, mark in funding):
        raise ValueError("funding needs a finite rate and positive settlement mark")

    def replay(event: OpportunityEvent, exit_policy: str, recipe: str,
               cost_scenario: str) -> CounterfactualTradeReplay:
        if (exit_policy, recipe, cost_scenario) != (schedule.policy_id, execution_recipe, scenario):
            raise ValueError("causal label/runtime exit or execution recipe mismatch")
        block_end = block_ends[event.event_id]
        if not schedule.start_ns <= event.decision_ts_ns < block_end <= schedule.end_ns:
            raise ValueError("candidate outside causal exit coverage")
        entry_index = bisect_right(open_times, event.decision_ts_ns)
        # No synthetic entry price: caller must provide an in-block next open.
        # Candidates without one need explicit no-entry censoring upstream.
        if entry_index >= len(opens) or open_times[entry_index] >= block_end:
            raise ValueError("candidate has no authorized in-block entry open")
        entry_ts, entry_price = opens[entry_index]
        decision = schedule.next_exit_decision(
            side=event.side, entry_ts_ns=entry_ts, block_end_ns=block_end,
        )
        exit_index = bisect_right(open_times, decision) if decision is not None else len(opens)
        closed = exit_index < len(opens) and open_times[exit_index] < block_end
        exit_ts, exit_price = opens[exit_index] if closed else (None, None)
        end = exit_ts if closed else block_end - 1
        payments = funding[bisect_right(funding_times, entry_ts):bisect_right(funding_times, end)]
        funding_cashflow = -event.side * fsum(rate * mark / entry_price for _, rate, mark in payments) * 10_000
        exit_ratio = exit_price / entry_price if closed else 0.0
        facts = [event.event_id, exit_policy, recipe, scenario, entry_ts, entry_price,
                 decision, exit_ts, exit_price, payments, fee_bps, slippage_bps, block_end]
        receipt = sha256(json.dumps(facts, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        return CounterfactualTradeReplay(
            event_id=event.event_id, entry_ts_ns=entry_ts, entry_price=entry_price,
            exit_ts_ns=exit_ts, exit_price=exit_price,
            entry_fee_bps=fee_bps, exit_fee_bps=fee_bps * exit_ratio,
            entry_slippage_bps=slippage_bps, exit_slippage_bps=slippage_bps * exit_ratio,
            funding_cashflow_bps=funding_cashflow,
            censor_ts_ns=None if closed else block_end, simulation_receipt_id=receipt,
        )

    return replay
