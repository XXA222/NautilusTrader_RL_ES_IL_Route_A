"""Frozen-exit counterfactual labels for deduplicated teacher opportunities."""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Callable, Literal, Sequence

from .candidate_event_union import CandidateGroup
from .opportunity_labels import CounterfactualTradeReplay


@dataclass(frozen=True, slots=True)
class NoEntryReplay:
    """No native fill is implied; terminal and boundary states remain distinct."""
    event_id: str
    status: Literal["UNFILLED", "CANCELLED", "MISSED", "RIGHT_CENSORED"]
    observed_until_ns: int
    simulation_receipt_id: str

    def __post_init__(self) -> None:
        if (not isinstance(self.event_id, str) or len(self.event_id) != 64
                or any(c not in "0123456789abcdef" for c in self.event_id)
                or self.status not in ("UNFILLED", "CANCELLED", "MISSED", "RIGHT_CENSORED")
                or type(self.observed_until_ns) is not int or self.observed_until_ns <= 0
                or not isinstance(self.simulation_receipt_id, str)
                or not self.simulation_receipt_id.strip()):
            raise ValueError("invalid no-entry research replay")


@dataclass(frozen=True, slots=True)
class FrozenReplay:
    """The research replay declares the execution and exit it actually used."""
    trade: CounterfactualTradeReplay | NoEntryReplay
    exit_policy_id: str
    execution_recipe: str
    cost_scenario_id: str


@dataclass(frozen=True, slots=True)
class GroupOutcome:
    event_id: str
    decision_ts_ns: int
    source_identity: str
    fold_id: str
    exit_policy_id: str
    execution_recipe: str
    cost_scenario_id: str
    label_end_ts_ns: int | None
    censor_ts_ns: int | None
    net_return_bps: float | None
    evidence_type: str = "MODEL_COUNTERFACTUAL"
    economic_status: str = "SCENARIO_ONLY"
    outcome_status: str = "COMPLETE"
    return_basis_id: str = "LOG_REFERENCE_ENTRY_NOTIONAL_BPS"


def label_grouped_candidates(
    groups: Sequence[CandidateGroup],
    replay: Callable[[CandidateGroup, str, str, str], FrozenReplay],
    *, exit_policy_id: str, execution_recipe: str, cost_scenario_id: str,
    block_end_ns: int, gross_return_method: Literal["log", "simple"] = "log",
) -> tuple[GroupOutcome, ...]:
    """One research outcome per deduplicated event, with end-of-window censoring."""
    if gross_return_method not in ("log", "simple"):
        raise ValueError("gross return method must be log or simple")
    if (not groups or not callable(replay) or type(block_end_ns) is not int
            or any(not isinstance(v, str) or not v.strip()
                   for v in (exit_policy_id, execution_recipe, cost_scenario_id))):
        raise ValueError("groups, frozen recipes, and block end are required")
    if len({g.event_id for g in groups}) != len(groups):
        raise ValueError("duplicate grouped candidate")
    if len({(g.observation.fold_id, g.observation.source_id,
             g.observation.feature_spec_hash) for g in groups}) != 1:
        raise ValueError("mixed candidate source/fold/schema")
    result = []
    for group in groups:
        obs = group.observation
        if obs.decision_ts_ns >= block_end_ns:
            raise ValueError("candidate outside registered block")
        receipt = replay(group, exit_policy_id, execution_recipe, cost_scenario_id)
        if (not isinstance(receipt, FrozenReplay)
                or (receipt.exit_policy_id, receipt.execution_recipe, receipt.cost_scenario_id)
                != (exit_policy_id, execution_recipe, cost_scenario_id)
                or not isinstance(receipt.trade, (CounterfactualTradeReplay, NoEntryReplay))
                or receipt.trade.event_id != group.event_id):
            raise ValueError("replay event or frozen exit/execution/cost recipe mismatch")
        fact = receipt.trade
        basis = ("LOG_REFERENCE_ENTRY_NOTIONAL_BPS" if gross_return_method == "log"
                 else "LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS")
        if isinstance(fact, NoEntryReplay):
            if fact.observed_until_ns <= obs.decision_ts_ns:
                raise ValueError("no-entry observation must be after decision")
            if fact.observed_until_ns > block_end_ns:
                raise ValueError("no-entry observation exceeds registered block")
            result.append(GroupOutcome(
                group.event_id, obs.decision_ts_ns, obs.source_id, obs.fold_id,
                exit_policy_id, execution_recipe, cost_scenario_id, None,
                fact.observed_until_ns if fact.status == "RIGHT_CENSORED" else None, None,
                outcome_status=fact.status, return_basis_id=basis,
            ))
            continue
        if fact.entry_ts_ns <= obs.decision_ts_ns:
            raise ValueError("entry must occur after the decision")
        if fact.entry_ts_ns >= block_end_ns or fact.exit_ts_ns is None or fact.exit_ts_ns >= block_end_ns:
            end, net = None, None
            censor = min(fact.censor_ts_ns or block_end_ns, block_end_ns)
        else:
            end, censor = fact.exit_ts_ns, None
            assert fact.exit_price is not None
            ratio = fact.exit_price / fact.entry_price
            price_return = log(ratio) if gross_return_method == "log" else ratio - 1
            gross = group.side * price_return * 10000
            net = (gross - fact.entry_fee_bps - fact.exit_fee_bps
                   - fact.entry_slippage_bps - fact.exit_slippage_bps
                   + fact.funding_cashflow_bps)
        result.append(GroupOutcome(
            group.event_id, obs.decision_ts_ns, obs.source_id, obs.fold_id,
            exit_policy_id, execution_recipe, cost_scenario_id, end, censor, net,
            outcome_status="RIGHT_CENSORED" if net is None else "COMPLETE",
            return_basis_id=basis,
        ))
    return tuple(result)


def purge_before(labels: Sequence[GroupOutcome], *, next_start_ns: int,
                 embargo_ns: int = 0) -> tuple[GroupOutcome, ...]:
    """Only completed labels strictly before the next data interval may fit."""
    if (type(next_start_ns) is not int or next_start_ns <= 0
            or type(embargo_ns) is not int or embargo_ns < 0):
        raise ValueError("valid next start and embargo required")
    return tuple(label for label in labels if label.label_end_ts_ns is not None
                 and label.label_end_ts_ns + embargo_ns < next_start_ns)
