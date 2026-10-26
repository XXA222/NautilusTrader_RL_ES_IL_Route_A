"""Full-denominator fixed-exit counterfactual labels for Train research.

The injected replay must use the frozen research execution/exit recipe and
authorized source. This adapter accounts for the supplied price path and
scenario costs; it does not perform market IO or claim native fills.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Literal, Sequence

from .candidate_events import OpportunityEvent
from .opportunity_models import OpportunityOutcome, feature_vector_sha256


@dataclass(frozen=True, slots=True)
class CounterfactualTradeReplay:
    """Frozen-price-path result with transaction costs itemized exactly once.

    Entry/exit prices are the pre-slippage reference prices. Fee and slippage
    fields are per-leg costs in the same entry-notional-bps basis; funding is
    a signed net cashflow in that basis. This DTO is research-only, not a
    native fill receipt.
    """

    event_id: str
    entry_ts_ns: int
    entry_price: float
    exit_ts_ns: int | None
    exit_price: float | None
    entry_fee_bps: float
    exit_fee_bps: float
    entry_slippage_bps: float
    exit_slippage_bps: float
    funding_cashflow_bps: float
    censor_ts_ns: int | None
    simulation_receipt_id: str

    def __post_init__(self) -> None:
        if (not isinstance(self.event_id, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.event_id)):
            raise ValueError("counterfactual replay must bind a candidate event SHA-256")
        if type(self.entry_ts_ns) is not int or self.entry_ts_ns <= 0:
            raise ValueError("counterfactual replay entry clock is invalid")
        if (isinstance(self.entry_price, bool) or not isinstance(self.entry_price, (int, float))
                or not math.isfinite(self.entry_price) or self.entry_price <= 0):
            raise ValueError("counterfactual entry price must be finite and positive")
        for name in ("entry_fee_bps", "exit_fee_bps", "entry_slippage_bps",
                     "exit_slippage_bps"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"counterfactual {name} must be finite and nonnegative")
        if (isinstance(self.funding_cashflow_bps, bool)
                or not isinstance(self.funding_cashflow_bps, (int, float))
                or not math.isfinite(self.funding_cashflow_bps)):
            raise ValueError("funding cashflow must be finite signed net bps")
        if not isinstance(self.simulation_receipt_id, str) or not self.simulation_receipt_id.strip():
            raise ValueError("counterfactual simulator receipt identity is required")
        if self.exit_ts_ns is None:
            if (self.exit_price is not None or type(self.censor_ts_ns) is not int
                    or self.censor_ts_ns <= self.entry_ts_ns):
                raise ValueError("open replay must be censored after entry without an exit outcome")
        elif (type(self.exit_ts_ns) is not int or self.exit_ts_ns <= self.entry_ts_ns
              or isinstance(self.exit_price, bool)
              or not isinstance(self.exit_price, (int, float))
              or not math.isfinite(self.exit_price) or self.exit_price <= 0
              or self.censor_ts_ns is not None):
            raise ValueError("closed replay needs a valid exit and cannot also be censored")


@dataclass(frozen=True, slots=True)
class CounterfactualOutcomeBatch:
    outcomes: tuple[OpportunityOutcome, ...]
    candidate_events: tuple[OpportunityEvent, ...]
    feature_rows: tuple[tuple[str, tuple[float, ...]], ...]

    @property
    def censored_count(self) -> int:
        return sum(outcome.net_return_bps is None for outcome in self.outcomes)

    @property
    def complete_count(self) -> int:
        return len(self.outcomes) - self.censored_count

    def event_map(self) -> dict[str, OpportunityEvent]:
        return {event.event_id: event for event in self.candidate_events}

    def feature_map(self) -> dict[str, tuple[float, ...]]:
        return dict(self.feature_rows)


ReplayCandidate = Callable[
    [OpportunityEvent, str, str, str], CounterfactualTradeReplay
]


def label_counterfactual_candidates(
    events: Sequence[OpportunityEvent],
    replay_candidate: ReplayCandidate,
    feature_rows: dict[str, Sequence[float]],
    block_ids: dict[str, str],
    block_intervals: dict[str, tuple[int, int]],
    *,
    feature_names: Sequence[str],
    split: Literal["train_fit", "train_calibration", "train_inner_screen"],
    exit_policy_id: str,
    execution_recipe: str,
    cost_scenario_id: str,
    return_basis_id: str,
    outcome_method: str = "fixed-exit-model-counterfactual-v1",
    gross_return_method: Literal["log", "simple"] = "log",
) -> CounterfactualOutcomeBatch:
    """Replay every candidate once, preserving winners, losers and censoring.

    The callback receives the same frozen exit policy, execution recipe, and
    cost scenario for every event. Missing, duplicate, or mismatched candidate
    outputs fail the whole batch rather than silently shrinking its denominator.
    """
    candidates = tuple(events)
    if gross_return_method not in ("log", "simple"):
        raise ValueError("gross return method must be log or simple")
    if (not candidates or any(not isinstance(event, OpportunityEvent) for event in candidates)
            or not callable(replay_candidate)):
        raise ValueError("causal candidate events and frozen replay callback are required")
    event_ids = tuple(event.event_id for event in candidates)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("counterfactual candidate universe contains duplicate events")
    if (set(feature_rows) != set(event_ids) or set(block_ids) != set(event_ids)
            or set(block_intervals) != set(event_ids)):
        raise ValueError("features and independent time-block metadata must cover every candidate")
    names = tuple(feature_names)
    if (not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("fixed unique feature names are required")
    for name, value in (("exit_policy_id", exit_policy_id),
                        ("execution_recipe", execution_recipe),
                        ("cost_scenario_id", cost_scenario_id),
                        ("return_basis_id", return_basis_id),
                        ("outcome_method", outcome_method)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"counterfactual label recipe requires {name}")

    intervals_by_block: dict[str, tuple[int, int]] = {}
    for event in candidates:
        block_id = block_ids[event.event_id]
        interval = block_intervals[event.event_id]
        if (not isinstance(block_id, str) or not block_id.strip()
                or not isinstance(interval, tuple) or len(interval) != 2
                or type(interval[0]) is not int or type(interval[1]) is not int
                or interval[0] < 0 or interval[1] <= interval[0]):
            raise ValueError("candidate block needs a valid identity and [start, end) interval")
        if not interval[0] <= event.decision_ts_ns < interval[1]:
            raise ValueError("candidate decision must lie inside its declared independent block")
        prior = intervals_by_block.setdefault(block_id, interval)
        if prior != interval:
            raise ValueError("candidates in the same independent block need identical boundaries")
    sorted_intervals = tuple(sorted(intervals_by_block.values()))
    if any(left[1] > right[0] for left, right in zip(sorted_intervals, sorted_intervals[1:])):
        raise ValueError("candidate independent time blocks must not overlap")

    outcomes: list[OpportunityOutcome] = []
    frozen_features: list[tuple[str, tuple[float, ...]]] = []
    for event in candidates:
        block_id = block_ids[event.event_id]
        if not isinstance(block_id, str) or not block_id.strip():
            raise ValueError("candidate independent block identity is missing")
        values = tuple(float(value) for value in feature_rows[event.event_id])
        vector_sha = feature_vector_sha256(
            event.event_id, event.feature_spec_hash, names, values,
        )
        replay = replay_candidate(
            event, exit_policy_id, execution_recipe, cost_scenario_id,
        )
        if not isinstance(replay, CounterfactualTradeReplay) or replay.event_id != event.event_id:
            raise ValueError("fixed-exit replay output does not match its candidate event")
        if replay.entry_ts_ns <= event.decision_ts_ns:
            raise ValueError("counterfactual fill must occur after the causal decision")
        block_end_ts_ns = block_intervals[event.event_id][1]
        if replay.entry_ts_ns >= block_end_ts_ns:
            label_end_ts_ns = None
            net_return_bps = None
            censor_ts_ns = block_end_ts_ns
        elif replay.exit_ts_ns is None:
            label_end_ts_ns = None
            net_return_bps = None
            if replay.censor_ts_ns is None:
                raise ValueError("open counterfactual replay must declare its censor time")
            censor_ts_ns = min(replay.censor_ts_ns, block_end_ts_ns)
        elif replay.exit_ts_ns >= block_end_ts_ns:
            # An outcome beyond the independent block boundary must not leak
            # back into this block's fit, calibration, or evaluation labels.
            label_end_ts_ns = None
            net_return_bps = None
            censor_ts_ns = block_end_ts_ns
        else:
            exit_ts_ns = replay.exit_ts_ns
            exit_price = replay.exit_price
            if exit_price is None:
                raise ValueError("closed counterfactual replay must declare its exit price")
            if exit_ts_ns <= event.decision_ts_ns:
                raise ValueError("counterfactual exit precedes its candidate decision")
            price_ratio = exit_price / replay.entry_price
            price_return = math.log(price_ratio) if gross_return_method == "log" else price_ratio - 1
            gross_bps = event.side * price_return * 10_000.0
            modeled_cost_bps = (
                replay.entry_fee_bps + replay.exit_fee_bps
                + replay.entry_slippage_bps + replay.exit_slippage_bps
            )
            net_return_bps = gross_bps - modeled_cost_bps + replay.funding_cashflow_bps
            label_end_ts_ns = exit_ts_ns
            censor_ts_ns = None
        outcomes.append(OpportunityOutcome(
            event_id=event.event_id,
            candidate_family=event.family,
            candidate_side=event.side,
            decision_ts_ns=event.decision_ts_ns,
            feature_available_at_ns=event.feature_available_at_ns,
            rule_version=event.rule_version,
            snapshot_id=event.snapshot_id,
            feature_spec_hash=event.feature_spec_hash,
            feature_vector_sha256=vector_sha,
            label_end_ts_ns=label_end_ts_ns,
            net_return_bps=net_return_bps,
            fold_id=event.fold_id,
            source_identity=event.source_identity,
            split=split,
            independent_block_id=block_id,
            independent_block_start_ts_ns=block_intervals[event.event_id][0],
            independent_block_end_ts_ns=block_intervals[event.event_id][1],
            exit_policy_id=exit_policy_id,
            execution_recipe=execution_recipe,
            cost_scenario_id=cost_scenario_id,
            return_basis_id=return_basis_id,
            simulation_receipt_id=replay.simulation_receipt_id,
            outcome_method=outcome_method,
            evidence_type="MODEL_COUNTERFACTUAL",
            economic_status="SCENARIO_ONLY",
            censor_ts_ns=censor_ts_ns,
        ))
        frozen_features.append((event.event_id, values))
    return CounterfactualOutcomeBatch(
        outcomes=tuple(outcomes),
        candidate_events=candidates,
        feature_rows=tuple(frozen_features),
    )
