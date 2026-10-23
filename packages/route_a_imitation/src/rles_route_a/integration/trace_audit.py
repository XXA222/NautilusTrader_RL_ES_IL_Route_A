"""Fail-closed A0/A1 native teacher trace gate (no market IO or shadow ledger)."""
from __future__ import annotations
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from typing import Mapping, Sequence, Any
from rles_route_a.contracts import PointInTimeObservation, ExpertAdvice, NativeActionTrace
from rles_route_a.datasets.sequence_dataset import LabeledStep
from rles_route_a.integration.native_rollout import TeacherFact, native_teacher_step
from rles_route_a.integration.native_economics import NativeRewardReceipt


@dataclass(frozen=True)
class AuditedTeacherBarrier:
    observation: PointInTimeObservation
    advice: ExpertAdvice
    trace: NativeActionTrace
    reward: NativeRewardReceipt
    execution_recipe: str
    source_identity: str
    terminal_l6_receipt: str | None = None
    trajectory_id: str = ""
    terminal_settlement_ts_ns: int | None = None
    native_order_lifecycle_verified: bool = False
    native_policy_fills: tuple[Mapping[str, Any], ...] = ()
    native_economic_evidence_authority: str | None = None

    @property
    def native(self) -> NativeActionTrace:
        """Allow the BC label path to consume the same audited native trace."""
        return self.trace


@dataclass(frozen=True)
class NativeTraceSummary:
    """Native counts; step PnL is diagnostic until a trade case is attributed."""
    fold_id: str
    decisions: int
    native_submits: int
    native_fills: int
    blocked_label_count: int
    executable_label_count: int
    native_equity_start: float
    native_equity_end: float
    economic_status: str
    # Per-episode balances cannot be compounded across account resets.
    episode_equity_segments: tuple[tuple[str, float, float], ...] = ()
    unique_filled_order_refs: int = 0
    intent_supervised_count: int = 0
    completed_native_trade_case_count: int = 0
    positive_native_trade_case_count: int = 0
    negative_native_trade_case_count: int = 0
    neutral_native_trade_case_count: int = 0
    negative_entry_avoidance_label_count: int = 0


def audit_teacher_barriers(barriers: Sequence[AuditedTeacherBarrier], *,
                          native_reward_tolerance: float = 1e-8) -> tuple[tuple[LabeledStep, ...], NativeTraceSummary]:
    if not barriers:
        raise ValueError("native teacher trace cannot be empty")
    folds = {b.observation.fold_id for b in barriers}
    recipes = {b.execution_recipe for b in barriers}
    sources = {b.source_identity for b in barriers}
    if len(folds) != 1 or not next(iter(folds)) or len(recipes) != 1 or not next(iter(recipes)) or len(sources) != 1 or not next(iter(sources)):
        raise ValueError("native teacher trace must bind exactly one Fold, execution recipe, and source")
    last_ts: dict[str,int] = {}
    samples: list[LabeledStep] = []
    submits = fills = 0
    previous_equity: float | None = None
    previous_episode: str | None = None
    closed_episodes: set[str] = set()
    terminal_episodes: set[str] = set()
    episode_equity_segments: list[tuple[str, float, float]] = []
    unique_filled: set[tuple[str, str]] = set()
    for b in barriers:
        obs, advice, trace = b.observation, b.advice, b.trace
        if (not isinstance(b.trajectory_id, str)
                or (b.trajectory_id and not b.trajectory_id.strip())):
            raise ValueError("invalid native teacher trajectory identity")
        if obs.episode_id in terminal_episodes:
            raise ValueError("native teacher trace continues after an L6 terminal barrier")
        if (b.terminal_l6_receipt is not None
                and (not isinstance(b.terminal_l6_receipt, str)
                     or not b.terminal_l6_receipt.strip())):
            raise ValueError("invalid L6 terminal receipt identity")
        if b.terminal_l6_receipt is None:
            if b.terminal_settlement_ts_ns is not None:
                raise ValueError("terminal settlement clock has no L6 receipt")
        elif (type(b.terminal_settlement_ts_ns) is not int
              or b.terminal_settlement_ts_ns < trace.execution_barrier_ts_ns):
            raise ValueError("terminal L6 settlement clock is missing or precedes policy barrier")
        if not (obs.decision_ts_ns == trace.decision_ts_ns == advice.decision_ts_ns):
            raise ValueError("inconsistent decision timestamp in native teacher trace")
        if obs.episode_id in last_ts and obs.decision_ts_ns <= last_ts[obs.episode_id]:
            raise ValueError("nonmonotonic native episode clock")
        if obs.episode_id != previous_episode:
            if obs.episode_id in closed_episodes:
                raise ValueError("interleaved native episodes cannot be a single trajectory")
            if previous_episode is not None:
                closed_episodes.add(previous_episode)
            episode_equity_segments.append((obs.episode_id, trace.native_equity_before,
                                            trace.native_equity_after))
        last_ts[obs.episode_id] = obs.decision_ts_ns
        if previous_episode == obs.episode_id and previous_equity is not None:
            if abs(trace.native_equity_before - previous_equity) > max(1e-6, abs(previous_equity)*1e-8):
                raise ValueError("native equity discontinuity within episode")
        b.reward.validate(tolerance=native_reward_tolerance)
        if (abs(b.reward.prior_equity - trace.native_equity_before) > 1e-8
                or abs(b.reward.next_equity - trace.native_equity_after) > 1e-8):
            raise ValueError("reward receipt and native trace equity disagree")
        # Filled order IDs may legitimately recur across barriers when a host
        # reports cumulative/partial order facts; do NOT deduplicate without
        # unique native FillEvent IDs. L3 remains the accounting authority.
        # Pass the audited barrier itself so L6's settled reward receipt reaches
        # the labeler. Reconstructing a bare TeacherFact here silently dropped
        # economics and let scenario-only/unassessed actions look BC-eligible.
        samples.append(replace(native_teacher_step(b),
                               source_identity=b.source_identity,
                               trajectory_id=b.trajectory_id))
        submits += trace.planner_action == "SUBMIT"
        fills += len(trace.filled_order_ids)  # historical alias: order mentions, NOT FillEvents
        unique_filled.update((obs.episode_id, order) for order in trace.filled_order_ids)
        ep, first, _ = episode_equity_segments[-1]
        episode_equity_segments[-1] = (ep, first, trace.native_equity_after)
        previous_equity = trace.native_equity_after
        previous_episode = obs.episode_id
        if b.terminal_l6_receipt is not None:
            terminal_episodes.add(obs.episode_id)
    samples = list(_attribute_completed_trade_cases(barriers, samples))
    blocked = sum(sample.supervision_weight == 0 for sample in samples)
    intent_supervised = sum(
        sample.intent_supervision_weight is not None
        and sample.intent_supervision_weight > 0
        for sample in samples
    )
    trade_case_facts: dict[str, tuple[str, float, int]] = {}
    for sample in samples:
        if not sample.economic_case_status.startswith("OBSERVED_NATIVE_TRADE_"):
            continue
        case_outcome_bps = sample.case_outcome_bps
        case_end_ts_ns = sample.case_end_ts_ns
        if (not sample.case_fact_sha256 or case_outcome_bps is None
                or type(case_end_ts_ns) is not int):
            raise ValueError("completed native trade label is missing its case receipt")
        case_fact = (
            sample.economic_case_status,
            float(case_outcome_bps),
            case_end_ts_ns,
        )
        previous = trade_case_facts.setdefault(sample.case_fact_sha256, case_fact)
        if previous != case_fact:
            raise ValueError("one native trade case digest has conflicting outcomes")
    case_status_counts = {
        status: sum(fact[0] == status for fact in trade_case_facts.values())
        for status in (
            "OBSERVED_NATIVE_TRADE_POSITIVE",
            "OBSERVED_NATIVE_TRADE_NEGATIVE",
            "OBSERVED_NATIVE_TRADE_NEUTRAL",
        )
    }
    results = NativeTraceSummary(
        fold_id=next(iter(folds)), decisions=len(barriers), native_submits=submits,
        native_fills=fills, blocked_label_count=blocked,
        executable_label_count=len(barriers)-blocked,
        native_equity_start=barriers[0].trace.native_equity_before,
        native_equity_end=barriers[-1].trace.native_equity_after,
        economic_status=(
            "SCENARIO_ONLY" if any(b.trace.scenario_only for b in barriers)
            else "OBSERVED_NATIVE_TRADE_CASES"
            if any(label.economic_case_status.startswith("OBSERVED_NATIVE_TRADE_")
                   for label in samples)
            else "ECONOMIC_UNVERIFIED"
        ),
        episode_equity_segments=tuple(episode_equity_segments),
        unique_filled_order_refs=len(unique_filled),
        intent_supervised_count=intent_supervised,
        completed_native_trade_case_count=len(trade_case_facts),
        positive_native_trade_case_count=case_status_counts[
            "OBSERVED_NATIVE_TRADE_POSITIVE"
        ],
        negative_native_trade_case_count=case_status_counts[
            "OBSERVED_NATIVE_TRADE_NEGATIVE"
        ],
        neutral_native_trade_case_count=case_status_counts[
            "OBSERVED_NATIVE_TRADE_NEUTRAL"
        ],
        negative_entry_avoidance_label_count=sum(
            sample.label_origin == "INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE"
            and sample.intent_supervision_weight is not None
            and sample.intent_supervision_weight > 0
            for sample in samples
        ))
    return tuple(samples), results


def _attribute_completed_trade_cases(
        barriers: Sequence[AuditedTeacherBarrier],
        step_labels: Sequence[LabeledStep]) -> tuple[LabeledStep, ...]:
    """Attach full flat-to-flat, fee-inclusive native outcomes to decisions.

    A case is eligible only when the M2.2 order lifecycle was reconciled for
    every barrier and the position starts flat, is opened by a proven fill, and
    returns flat (or has an L6-bound terminal liquidation). Step PnL alone is
    never a positive teaching case. Supervision weights within each positive
    case are normalized independently, so a long holding period does not count
    as more independent evidence than a short round trip.
    """
    labels = list(step_labels)
    episode_start = 0
    while episode_start < len(barriers):
        episode = barriers[episode_start].observation.episode_id
        episode_end = episode_start + 1
        while (episode_end < len(barriers)
               and barriers[episode_end].observation.episode_id == episode):
            episode_end += 1
        case_start: int | None = None
        case_valid = True
        for index in range(episode_start, episode_end):
            barrier = barriers[index]
            observation, trace = barrier.observation, barrier.trace
            try:
                before = observation.account_value("net_exposure")
            except (KeyError, TypeError, ValueError):
                case_valid = False
                case_start = None
                continue
            after = trace.actual_exposure_after
            before_flat = abs(before) <= 1e-8
            after_flat = abs(after) <= 1e-8
            terminal_close = bool(barrier.terminal_l6_receipt)
            if case_start is None:
                if before_flat and not after_flat:
                    case_start = index
                    case_valid = (
                        bool(barrier.native_order_lifecycle_verified)
                        and bool(barrier.native_economic_evidence_authority)
                        and labels[index].action_status == "FILLED"
                        and not trace.scenario_only
                        and not barrier.reward.has_external_cashflow
                    )
                # A rollout that starts with inventory has no attributable
                # entry decision; wait until it returns flat before teaching.
                continue
            if (not barrier.native_order_lifecycle_verified
                    or not barrier.native_economic_evidence_authority
                    or trace.scenario_only or barrier.reward.has_external_cashflow):
                case_valid = False
            if before_flat and not after_flat:
                # A second entry before the previous position was closed would
                # mean the preceding case was not a single flat-to-flat trade.
                case_valid = False
            if before * after < 0:
                # Atomic flips have ambiguous close/open attribution.
                case_valid = False
            completed = (not after_flat and terminal_close) or (not before_flat and after_flat)
            if not completed:
                continue
            case_end_ts = barrier.terminal_settlement_ts_ns if terminal_close else trace.execution_barrier_ts_ns
            if (case_end_ts is None and terminal_close) or type(case_end_ts) is not int:
                case_valid = False
            elif case_end_ts <= barriers[case_start].observation.decision_ts_ns:
                case_valid = False
            if not terminal_close and labels[index].action_status != "FILLED":
                case_valid = False
            if case_valid:
                digest_payload = {
                    "source_identity": barrier.source_identity,
                    "execution_recipe": barrier.execution_recipe,
                    "trajectory_id": barrier.trajectory_id,
                    "native_economic_evidence_authority": (
                        barriers[case_start].native_economic_evidence_authority
                    ),
                    "episode_id": episode,
                    "entry_decision_ts_ns": barriers[case_start].observation.decision_ts_ns,
                    "end_ts_ns": case_end_ts,
                    "barriers": [{
                        "decision_id": item.trace.execution_decision_id,
                        "plan_id": item.trace.execution_plan_id,
                        "requested_action": item.trace.requested_action,
                        "exposure_before": item.observation.account_value("net_exposure"),
                        "exposure_after": item.trace.actual_exposure_after,
                        "equity_before": item.reward.prior_equity,
                        "equity_after": item.reward.next_equity,
                        "filled_orders": item.trace.filled_order_ids,
                        "native_policy_fills": [dict(fill) for fill in item.native_policy_fills],
                        "native_order_lifecycle_verified": item.native_order_lifecycle_verified,
                        "l6_terminal_receipt": item.terminal_l6_receipt,
                    } for item in barriers[case_start:index + 1]],
                }
                case_digest = sha256(json.dumps(
                    digest_payload, sort_keys=True, separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")).hexdigest()
                trade_outcome_bps = math.log(
                    barrier.reward.next_equity
                    / barriers[case_start].reward.prior_equity
                ) * 10_000.0
                if trade_outcome_bps > 0:
                    status = "OBSERVED_NATIVE_TRADE_POSITIVE"
                elif trade_outcome_bps < 0:
                    status = "OBSERVED_NATIVE_TRADE_NEGATIVE"
                else:
                    status = "OBSERVED_NATIVE_TRADE_NEUTRAL"
                case_rows: list[tuple[int, LabeledStep | None]] = []
                for row_index in range(case_start, index + 1):
                    row_barrier = barriers[row_index]
                    if status == "OBSERVED_NATIVE_TRADE_POSITIVE":
                        action_label = native_teacher_step(TeacherFact(
                            row_barrier.observation,
                            row_barrier.advice,
                            row_barrier.trace,
                        ))
                    elif row_index == case_start:
                        # A completed losing round trip is a defensible
                        # counterexample only at its flat-to-position entry:
                        # with hindsight, the safe action from that exact flat
                        # state was to remain flat. Do not relabel later HOLD,
                        # REDUCE, or EXIT decisions from the losing trajectory.
                        original = labels[row_index]
                        action_label = replace(
                            original,
                            advice=replace(
                                row_barrier.advice,
                                intent="FLAT",
                                requested_exposure=0.0,
                                reason_codes=tuple(dict.fromkeys((
                                    *row_barrier.advice.reason_codes,
                                    "negative_native_trade_entry_avoidance",
                                ))),
                            ),
                            supervision_weight=0.0,
                            intent_supervision_weight=1.0,
                            label_origin="INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE",
                            economic_case_status=status,
                            case_outcome_bps=trade_outcome_bps,
                            case_end_ts_ns=case_end_ts,
                            case_fact_sha256=case_digest,
                        )
                    else:
                        action_label = None
                    case_rows.append((row_index, action_label))

                for row_index, action_label in case_rows:
                    labels[row_index] = replace(
                        labels[row_index],
                        advice=(
                            action_label.advice
                            if action_label is not None else labels[row_index].advice
                        ),
                        label_origin=(
                            action_label.label_origin
                            if action_label is not None else labels[row_index].label_origin
                        ),
                        economic_case_status=status,
                        case_outcome_bps=trade_outcome_bps,
                        case_end_ts_ns=case_end_ts,
                        case_fact_sha256=case_digest,
                        supervision_weight=(
                            float(action_label.supervision_weight or 0.0)
                            if action_label is not None else 0.0
                        ),
                        intent_supervision_weight=(
                            float(action_label.intent_supervision_weight or 0.0)
                            if action_label is not None else 0.0
                        ),
                    )
            case_start = None
            case_valid = True
        episode_start = episode_end
    return tuple(labels)
