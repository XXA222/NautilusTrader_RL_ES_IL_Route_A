"""Frozen-exit counterfactual labels for deduplicated teacher opportunities."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
import re
from typing import Callable, Literal, Sequence

from .candidate_event_union import CandidateGroup
from .opportunity_labels import CounterfactualTradeReplay
from .opportunity_models import RETURN_BASIS_IDS


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
    simulation_receipt_id: str
    evidence_type: str = "MODEL_COUNTERFACTUAL"
    economic_status: str = "SCENARIO_ONLY"
    outcome_status: str = "COMPLETE"
    return_basis_id: str = "LOG_REFERENCE_ENTRY_NOTIONAL_BPS"
    observed_until_ns: int | None = None
    gross_return_bps: float | None = None
    fees_bps: float | None = None
    slippage_bps: float | None = None
    funding_cashflow_bps: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not re.fullmatch(r"[0-9a-f]{64}", self.event_id):
            raise ValueError("group outcome event_id must bind a lowercase SHA-256 candidate")
        if type(self.decision_ts_ns) is not int or self.decision_ts_ns <= 0:
            raise ValueError("group outcome decision clock must be a positive integer timestamp")
        if any(not isinstance(value, str) or not value.strip() for value in (
                self.source_identity, self.fold_id, self.exit_policy_id,
                self.execution_recipe, self.cost_scenario_id, self.simulation_receipt_id)):
            raise ValueError("group outcome source, recipe, and simulation receipt identities are required")
        if self.evidence_type != "MODEL_COUNTERFACTUAL" or self.economic_status != "SCENARIO_ONLY":
            raise ValueError("group outcomes must remain scenario-only model counterfactuals")
        if self.return_basis_id not in RETURN_BASIS_IDS:
            raise ValueError("group outcome return basis must use an explicit reference basis")
        if self.outcome_status not in ("COMPLETE", "RIGHT_CENSORED", "UNFILLED", "CANCELLED", "MISSED"):
            raise ValueError("group outcome status is invalid")
        if self.observed_until_ns is not None and (
                type(self.observed_until_ns) is not int or self.observed_until_ns <= self.decision_ts_ns):
            raise ValueError("group outcome observation end must follow its decision")
        if self.outcome_status == "COMPLETE":
            if (isinstance(self.net_return_bps, bool)
                    or not isinstance(self.net_return_bps, (int, float))
                    or not isfinite(self.net_return_bps)
                    or type(self.label_end_ts_ns) is not int
                    or self.label_end_ts_ns <= self.decision_ts_ns
                    or self.censor_ts_ns is not None or self.observed_until_ns is not None):
                raise ValueError("complete group outcome requires a finite return and later exit")
            components = (self.gross_return_bps, self.fees_bps,
                          self.slippage_bps, self.funding_cashflow_bps)
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not isfinite(value) for value in components):
                raise ValueError("complete outcome requires all finite gross/cost components")
            expected_net = (self.gross_return_bps - self.fees_bps - self.slippage_bps
                            + self.funding_cashflow_bps)
            tolerance = 1e-8 * max(1.0, abs(self.net_return_bps), abs(expected_net))
            if abs(self.net_return_bps - expected_net) > tolerance:
                raise ValueError("gross, fees, slippage and funding do not reconcile to net return")
        elif (self.net_return_bps is not None or self.label_end_ts_ns is not None):
            raise ValueError("incomplete group outcome cannot supply a return or exit clock")
        elif any(value is not None for value in (self.gross_return_bps, self.fees_bps,
                                                 self.slippage_bps, self.funding_cashflow_bps)):
            raise ValueError("incomplete outcome cannot supply complete-trade economics")
        elif self.outcome_status == "RIGHT_CENSORED":
            if (type(self.censor_ts_ns) is not int or self.censor_ts_ns <= self.decision_ts_ns
                    or (self.observed_until_ns is not None
                        and self.observed_until_ns != self.censor_ts_ns)):
                raise ValueError("right-censored group outcome requires a matching future observation clock")
        elif (self.censor_ts_ns is not None or type(self.observed_until_ns) is not int):
            raise ValueError("unfilled group outcome requires an observed-until clock, not a censor clock")


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
                simulation_receipt_id=fact.simulation_receipt_id,
                outcome_status=fact.status, return_basis_id=basis,
                observed_until_ns=fact.observed_until_ns,
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
            fees = fact.entry_fee_bps + fact.exit_fee_bps
            slippage = fact.entry_slippage_bps + fact.exit_slippage_bps
            funding = fact.funding_cashflow_bps
            net = gross - fees - slippage + funding
        result.append(GroupOutcome(
            group.event_id, obs.decision_ts_ns, obs.source_id, obs.fold_id,
            exit_policy_id, execution_recipe, cost_scenario_id, end, censor, net,
            outcome_status="RIGHT_CENSORED" if net is None else "COMPLETE",
            return_basis_id=basis,
            simulation_receipt_id=fact.simulation_receipt_id,
            observed_until_ns=censor,
            gross_return_bps=gross if net is not None else None,
            fees_bps=fees if net is not None else None,
            slippage_bps=slippage if net is not None else None,
            funding_cashflow_bps=funding if net is not None else None,
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
