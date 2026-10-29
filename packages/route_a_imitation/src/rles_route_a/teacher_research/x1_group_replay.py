"""Causal per-candidate X1 research replay using the selected original teacher."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import replace
from hashlib import sha256
import json
from math import fsum, isfinite

from rles_route_a.indicators_1m import MINUTE_NS
from rles_route_a.teachers_1m.registry import ALL_TEACHER_RECIPE_BY_ID, build_teacher_1m
from .conditional_outcome_labels import FrozenReplay, NoEntryReplay
from .opportunity_labels import CounterfactualTradeReplay


def make_x1_group_replay(*, observations, opens, funding, block_ends,
                         fee_bps, slippage_bps, execution_recipe, scenario):
    """Replay a grouped event with the same deterministic X1 owner as runtime.

    This is a counterfactual signal-and-price-path calculation, not a second
    account, risk engine, matching engine, or evidence of a native fill.
    """
    observations, opens, funding = tuple(observations), tuple(opens), tuple(funding)
    times = tuple(o.decision_ts_ns for o in observations)
    open_times = tuple(t for t, _ in opens)
    funding_times = tuple(t for t, _, _ in funding)
    if (not observations or not opens or not execution_recipe or not scenario
            or any(isinstance(v, bool) or not isfinite(v) or v < 0
                   for v in (fee_bps, slippage_bps))):
        raise ValueError("invalid X1 replay inputs")
    if any(b - a != MINUTE_NS for a, b in zip(times, times[1:])):
        raise ValueError("observations must have complete one-minute coverage")
    if any(b - a != MINUTE_NS for a, b in zip(open_times, open_times[1:])):
        raise ValueError("execution opens must have complete one-minute coverage")
    if any(not isfinite(p) or p <= 0 for _, p in opens):
        raise ValueError("invalid open prices")
    if any(b <= a for a, b in zip(funding_times, funding_times[1:])):
        raise ValueError("funding clocks must be strictly increasing")
    if any(not isfinite(rate) or not isfinite(mark) or mark <= 0 for _, rate, mark in funding):
        raise ValueError("invalid funding facts")
    identity = (observations[0].episode_id, observations[0].source_id,
                observations[0].fold_id, observations[0].feature_spec_hash)
    if any((o.episode_id, o.source_id, o.fold_id, o.feature_spec_hash) != identity
           for o in observations):
        raise ValueError("X1 replay cannot mix episode/source/fold/schema")
    by_time = {o.decision_ts_ns: i for i, o in enumerate(observations)}

    def replay(group, exit_policy, recipe, cost):
        if (exit_policy, recipe, cost) != ("X1", execution_recipe, scenario):
            raise ValueError("X1 label/runtime recipe mismatch")
        obs = group.observation
        if obs.decision_ts_ns not in by_time or observations[by_time[obs.decision_ts_ns]] != obs:
            raise ValueError("candidate must match its observed causal snapshot")
        block_end = block_ends[group.event_id]
        if type(block_end) is not int or block_end <= obs.decision_ts_ns:
            raise ValueError("invalid X1 block end")
        final_index = bisect_left(times, block_end) - 1
        if final_index < 0 or block_end - times[final_index] > MINUTE_NS:
            raise ValueError("X1 replay observation coverage ends before the block")
        owner = group.teachers[0][0]
        if owner not in ALL_TEACHER_RECIPE_BY_ID:
            raise ValueError("X1 candidate has unknown originating teacher")
        entry_index = bisect_right(open_times, obs.decision_ts_ns)
        if entry_index >= len(opens) or open_times[entry_index] >= block_end:
            return FrozenReplay(NoEntryReplay(group.event_id, "RIGHT_CENSORED", block_end,
                                              "no-in-block-execution-open"),
                                exit_policy, recipe, cost)
        entry_ts, entry_price = opens[entry_index]
        teacher = build_teacher_1m(owner)
        decision_index = by_time[obs.decision_ts_ns]
        for past in observations[:decision_index + 1]:
            flat = replace(past, account={**dict(past.account), "net_exposure": 0.0},
                           pending_order_ids=())
            teacher.advise(flat)
        signal = teacher.last_signal
        if signal.side != group.side or signal.raw_signal not in ("LONG_ENTRY", "SHORT_ENTRY"):
            raise ValueError("X1 owner signal does not reproduce the candidate")
        exit_decision = None
        for future in observations[decision_index + 1:]:
            if future.decision_ts_ns >= block_end:
                break
            if future.decision_ts_ns <= entry_ts:
                # The new fill cannot influence an earlier/simultaneous decision.
                flat = replace(future, account={**dict(future.account), "net_exposure": 0.0},
                               pending_order_ids=())
                teacher.advise(flat)
                continue
            position = replace(future, account={**dict(future.account),
                "net_exposure": group.side * teacher.recipe.target_exposure,
                "position_avg_px_open": entry_price}, pending_order_ids=())
            action = teacher.advise(position)
            if action.intent == "EXIT":
                exit_decision = future.decision_ts_ns
                break
        exit_index = bisect_right(open_times, exit_decision) if exit_decision is not None else len(opens)
        closed = exit_index < len(opens) and open_times[exit_index] < block_end
        exit_ts, exit_price = opens[exit_index] if closed else (None, None)
        end = exit_ts if exit_ts is not None else block_end - 1
        payments = funding[bisect_right(funding_times, entry_ts):bisect_right(funding_times, end)]
        funding_bps = -group.side * fsum(rate * mark / entry_price for _, rate, mark in payments) * 10000
        exit_ratio = exit_price / entry_price if exit_price is not None else 0.0
        receipt = sha256(json.dumps([group.event_id, owner, entry_ts, entry_price, exit_decision,
            exit_ts, exit_price, payments, fee_bps, slippage_bps, block_end],
            allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        fact = CounterfactualTradeReplay(
            group.event_id, entry_ts, entry_price, exit_ts, exit_price, fee_bps,
            fee_bps * exit_ratio, slippage_bps, slippage_bps * exit_ratio,
            funding_bps, None if closed else block_end, receipt)
        return FrozenReplay(fact, exit_policy, recipe, cost)
    return replay
