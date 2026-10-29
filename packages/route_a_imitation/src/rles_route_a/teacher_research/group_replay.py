"""Bridge deduplicated candidates to the existing causal fixed-exit replay."""
from __future__ import annotations

from bisect import bisect_right
from types import SimpleNamespace

from .causal_exit_replay import make_causal_exit_replay
from .conditional_outcome_labels import FrozenReplay, NoEntryReplay


def make_group_market_replay(*, schedule, opens, funding, block_ends,
                             fee_bps, slippage_bps, execution_recipe, scenario):
    """Replay each candidate from its next eligible open, with explicit no-entry censor."""
    underlying = make_causal_exit_replay(
        schedule=schedule, opens=opens, funding=funding, block_ends=block_ends,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        execution_recipe=execution_recipe, scenario=scenario)
    times = tuple(timestamp for timestamp, _ in opens)

    def replay(group, exit_policy, recipe, cost_scenario):
        if (exit_policy, recipe, cost_scenario) != (schedule.policy_id, execution_recipe, scenario):
            raise ValueError("causal label/runtime exit or execution recipe mismatch")
        block_end = block_ends[group.event_id]
        decision = group.observation.decision_ts_ns
        if not schedule.start_ns <= decision < block_end <= schedule.end_ns:
            raise ValueError("candidate outside causal exit coverage")
        entry_index = bisect_right(times, decision)
        if entry_index >= len(times) or times[entry_index] >= block_end:
            result = NoEntryReplay(group.event_id, "RIGHT_CENSORED", block_end,
                                   "no-in-block-execution-open")
        else:
            event = SimpleNamespace(event_id=group.event_id, decision_ts_ns=decision, side=group.side)
            result = underlying(event, exit_policy, recipe, cost_scenario)
        return FrozenReplay(result, exit_policy, recipe, cost_scenario)
    return replay
