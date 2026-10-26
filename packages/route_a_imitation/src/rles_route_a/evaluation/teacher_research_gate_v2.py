"""Exploratory G1 opportunity gate; never replaces the production teacher gate."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class CandidateBlockResult:
    """One paired candidate-universe result for one block and cost scenario."""

    preregistration_id: str
    trial_id: str
    independent_block_id: str
    block_start_ts_ns: int
    block_end_ts_ns: int
    split: Literal["train_inner_screen"]
    cost_scenario_id: str
    candidate_universe_ids: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    candidate_outcomes_bps: tuple[tuple[str, float | None], ...]
    candidate_rejection_reasons: tuple[tuple[str, str | None], ...]
    baseline_net_return_bps: float
    candidate_net_return_bps: float
    negative_control_net_return_bps: float
    baseline_risk_adjusted_excess_bps: float
    candidate_risk_adjusted_excess_bps: float
    baseline_tail_loss_bps: float
    candidate_tail_loss_bps: float
    baseline_max_drawdown_bps: float
    candidate_max_drawdown_bps: float
    baseline_calibration_log_loss: float
    candidate_calibration_log_loss: float
    baseline_turnover_bps: float
    candidate_turnover_bps: float
    candidate_fees_bps: float
    candidate_funding_bps: float
    candidate_filled_orders: int
    candidate_exposure_minutes: float
    risk_budget_id: str
    exit_policy_id: str
    execution_recipe: str
    evidence_type: Literal["MODEL_COUNTERFACTUAL"] = "MODEL_COUNTERFACTUAL"
    economic_status: Literal["SCENARIO_ONLY"] = "SCENARIO_ONLY"

    def __post_init__(self) -> None:
        for name in (
            "preregistration_id", "trial_id", "independent_block_id",
            "cost_scenario_id", "risk_budget_id", "exit_policy_id",
            "execution_recipe",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"G1 result requires {name}")
        if self.split != "train_inner_screen":
            raise PermissionError("G1 research gate accepts only Train-inner screen rows")
        if (type(self.block_start_ts_ns) is not int or type(self.block_end_ts_ns) is not int
                or self.block_start_ts_ns < 0 or self.block_end_ts_ns <= self.block_start_ts_ns):
            raise ValueError("G1 block must declare a valid half-open [start, end) time interval")
        if (self.evidence_type != "MODEL_COUNTERFACTUAL"
                or self.economic_status != "SCENARIO_ONLY"):
            raise ValueError("G1 outputs remain model-counterfactual SCENARIO_ONLY")
        universe = tuple(self.candidate_universe_ids)
        selected = tuple(self.selected_candidate_ids)
        if (not universe or len(universe) != len(set(universe))
                or any(not isinstance(value, str)
                       or re.fullmatch(r"[0-9a-f]{64}", value) is None
                       for value in universe)):
            raise ValueError("G1 must preserve a unique full candidate denominator of SHA-256 event ids")
        if (len(selected) != len(set(selected))
                or any(value not in set(universe) for value in selected)):
            raise ValueError("selected events must be a unique subset of all candidates")
        outcomes = tuple(self.candidate_outcomes_bps)
        reasons = tuple(self.candidate_rejection_reasons)
        if (len(outcomes) != len(universe)
                or len({event_id for event_id, _ in outcomes}) != len(outcomes)
                or {event_id for event_id, _ in outcomes} != set(universe)):
            raise ValueError("candidate outcome map must preserve the complete candidate denominator")
        if (len(reasons) != len(universe)
                or len({event_id for event_id, _ in reasons}) != len(reasons)
                or {event_id for event_id, _ in reasons} != set(universe)):
            raise ValueError("candidate rejection-reason map must preserve the complete denominator")
        selected_set = set(selected)
        for event_id, outcome_bps in outcomes:
            if outcome_bps is not None and (
                isinstance(outcome_bps, bool)
                or not isinstance(outcome_bps, (int, float))
                or not math.isfinite(outcome_bps)
            ):
                raise ValueError("candidate outcomes must be finite net bps or explicitly censored")
        for event_id, reason in reasons:
            if event_id in selected_set and reason is not None:
                raise ValueError("selected candidates cannot carry a rejection reason")
            if event_id not in selected_set and (
                not isinstance(reason, str) or not reason.strip()
            ):
                raise ValueError("every rejected candidate requires a frozen reason code")
        object.__setattr__(self, "candidate_universe_ids", universe)
        object.__setattr__(self, "selected_candidate_ids", selected)
        object.__setattr__(self, "candidate_outcomes_bps", outcomes)
        object.__setattr__(self, "candidate_rejection_reasons", reasons)
        for name in (
            "baseline_net_return_bps", "candidate_net_return_bps",
            "negative_control_net_return_bps", "baseline_risk_adjusted_excess_bps",
            "candidate_risk_adjusted_excess_bps", "baseline_tail_loss_bps",
            "candidate_tail_loss_bps", "baseline_max_drawdown_bps",
            "candidate_max_drawdown_bps", "baseline_calibration_log_loss",
            "candidate_calibration_log_loss", "baseline_turnover_bps",
            "candidate_turnover_bps", "candidate_fees_bps", "candidate_funding_bps",
            "candidate_exposure_minutes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"G1 {name} must be a finite net return after scenario costs")
        for name in (
            "baseline_tail_loss_bps", "candidate_tail_loss_bps",
            "baseline_max_drawdown_bps", "candidate_max_drawdown_bps",
            "baseline_calibration_log_loss", "candidate_calibration_log_loss",
            "baseline_turnover_bps", "candidate_turnover_bps",
            "candidate_fees_bps", "candidate_exposure_minutes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"G1 {name} uses a nonnegative magnitude convention")
        if type(self.candidate_filled_orders) is not int or self.candidate_filled_orders < 0:
            raise ValueError("G1 filled-order count must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class TeacherResearchGateV2Result:
    status: str
    reason_codes: tuple[str, ...]
    independent_blocks: int
    candidate_events: int
    selected_events: int
    selection_rate: float
    mean_worst_scenario_increment_bps: float | None
    increment_ci95_bps: tuple[float, float] | None
    negative_control_delta_ci95_bps: tuple[float, float] | None
    candidate_vs_negative_control_ci95_bps: tuple[float, float] | None
    scenario_increment_ci95_bps: tuple[tuple[str, float, float], ...]
    risk_adjusted_excess_ci95_bps: tuple[float, float] | None
    tail_loss_reduction_ci95_bps: tuple[float, float] | None
    max_drawdown_reduction_ci95_bps: tuple[float, float] | None
    calibration_log_loss_reduction_ci95: tuple[float, float] | None
    turnover_change_ci95_bps: tuple[float, float] | None
    scenario_execution_diagnostics: tuple[tuple[str, float, float, int, float], ...]
    scenario_candidate_coverage: tuple["ScenarioCandidateCoverage", ...]
    preregistration_id: str
    trial_id: str


@dataclass(frozen=True, slots=True)
class ScenarioCandidateCoverage:
    """Full-denominator candidate and rejected-opportunity diagnostics."""

    scenario_id: str
    candidate_count: int
    selected_count: int
    complete_count: int
    censored_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    selected_complete_count: int
    selected_censored_count: int
    selected_positive_count: int
    selected_negative_count: int
    selected_neutral_count: int
    rejected_complete_count: int
    rejected_censored_count: int
    rejected_positive_count: int
    rejected_negative_count: int
    rejected_neutral_count: int
    missed_positive_opportunity_bps: float
    mean_complete_outcome_bps: float | None
    q10_complete_outcome_bps: float | None
    mean_selected_outcome_bps: float | None
    q10_selected_outcome_bps: float | None
    mean_rejected_outcome_bps: float | None
    q10_rejected_outcome_bps: float | None
    rejection_reason_counts: tuple[tuple[str, int], ...]


def evaluate_teacher_research_gate_v2(
    rows: Sequence[CandidateBlockResult],
    *,
    required_cost_scenarios: Sequence[str] = ("base", "stress"),
    minimum_independent_blocks: int = 4,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> TeacherResearchGateV2Result:
    """Evaluate paired block deltas without counting scenarios/seeds as samples.

    Returns an exploratory Train-inner signal only. Even a passing result is
    never an Alpha Teacher qualification or an economic-authority receipt.
    """
    scenarios = tuple(required_cost_scenarios)
    if (len(scenarios) < 2 or len(set(scenarios)) != len(scenarios)
            or any(not isinstance(value, str) or not value for value in scenarios)):
        raise ValueError("G1 requires unique, explicit Base/Stress-like cost scenarios")
    if (type(minimum_independent_blocks) is not int or minimum_independent_blocks < 4
            or type(bootstrap_replicates) is not int or bootstrap_replicates < 1000
            or type(bootstrap_seed) is not int):
        raise ValueError("G1 requires at least four blocks and a fixed bootstrap recipe")
    if not rows or any(not isinstance(row, CandidateBlockResult) for row in rows):
        raise ValueError("G1 paired block results are required")
    identities = {
        (row.preregistration_id, row.trial_id, row.risk_budget_id,
         row.exit_policy_id, row.execution_recipe)
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError("G1 rows must share frozen preregistration, trial, risk, exit, and execution")
    block_ids = {row.independent_block_id for row in rows}
    by_block_scenario = {(row.independent_block_id, row.cost_scenario_id): row for row in rows}
    if len(by_block_scenario) != len(rows):
        raise ValueError("duplicate block/scenario rows do not create independent samples")
    if any({row.cost_scenario_id for row in rows
            if row.independent_block_id == block} != set(scenarios) for block in block_ids):
        raise ValueError("every independent block must have every frozen cost scenario")
    block_intervals: dict[str, tuple[int, int]] = {}
    for block in block_ids:
        block_rows = [row for row in rows if row.independent_block_id == block]
        intervals = {(row.block_start_ts_ns, row.block_end_ts_ns) for row in block_rows}
        if len(intervals) != 1:
            raise ValueError("all cost scenarios for a block must share identical time boundaries")
        block_intervals[block] = next(iter(intervals))
    ordered_blocks = tuple(sorted(block_ids, key=lambda block: block_intervals[block][0]))
    all_candidate_ids = [
        event_id
        for block in ordered_blocks
        for event_id in by_block_scenario[(block, scenarios[0])].candidate_universe_ids
    ]
    if len(all_candidate_ids) != len(set(all_candidate_ids)):
        raise ValueError("candidate event identities must not be reused across independent blocks")
    for left_block, right_block in zip(ordered_blocks, ordered_blocks[1:]):
        left_end = block_intervals[left_block][1]
        right_start = block_intervals[right_block][0]
        if left_end > right_start:
            raise ValueError("G1 independent time blocks must not overlap")
    for block in ordered_blocks:
        block_rows = [by_block_scenario[(block, scenario)] for scenario in scenarios]
        reference = block_rows[0]
        for row in block_rows[1:]:
            if (row.candidate_universe_ids != reference.candidate_universe_ids
                    or row.selected_candidate_ids != reference.selected_candidate_ids):
                raise ValueError("Base/Stress must reuse the same candidate universe and selections")

    base = rows[0]
    candidate_events = sum(len(by_block_scenario[(block, scenarios[0])].candidate_universe_ids)
                           for block in ordered_blocks)
    selected_events = sum(len(by_block_scenario[(block, scenarios[0])].selected_candidate_ids)
                          for block in ordered_blocks)
    selection_rate = selected_events / candidate_events if candidate_events else 0.0
    candidate_deltas: dict[str, list[float]] = {scenario: [] for scenario in scenarios}
    control_deltas: dict[str, list[float]] = {scenario: [] for scenario in scenarios}
    for scenario in scenarios:
        for block in ordered_blocks:
            row = by_block_scenario[(block, scenario)]
            candidate_deltas[scenario].append(
                float(row.candidate_net_return_bps - row.baseline_net_return_bps)
            )
            control_deltas[scenario].append(
                float(row.negative_control_net_return_bps - row.baseline_net_return_bps)
            )
    # Cost modes are sensitivity views of the same market block, not additional
    # samples. Use the least favorable scenario within each block, then resample
    # only those independent block values.
    conservative_candidate = np.min(
        np.asarray([candidate_deltas[s] for s in scenarios], dtype=np.float64), axis=0
    )
    conservative_control = np.min(
        np.asarray([control_deltas[s] for s in scenarios], dtype=np.float64), axis=0
    )
    conservative_control_gap = np.min(
        np.asarray([
            np.asarray(candidate_deltas[s], dtype=np.float64)
            - np.asarray(control_deltas[s], dtype=np.float64)
            for s in scenarios
        ]),
        axis=0,
    )
    risk_adjusted_deltas = _scenario_deltas(
        by_block_scenario, ordered_blocks, scenarios,
        "candidate_risk_adjusted_excess_bps", "baseline_risk_adjusted_excess_bps",
    )
    tail_reduction_deltas = _scenario_deltas(
        by_block_scenario, ordered_blocks, scenarios,
        "baseline_tail_loss_bps", "candidate_tail_loss_bps",
    )
    max_drawdown_reduction_deltas = _scenario_deltas(
        by_block_scenario, ordered_blocks, scenarios,
        "baseline_max_drawdown_bps", "candidate_max_drawdown_bps",
    )
    calibration_reduction_deltas = _scenario_deltas(
        by_block_scenario, ordered_blocks, scenarios,
        "baseline_calibration_log_loss", "candidate_calibration_log_loss",
    )
    turnover_change_deltas = _scenario_deltas(
        by_block_scenario, ordered_blocks, scenarios,
        "candidate_turnover_bps", "baseline_turnover_bps",
    )
    # For the first three diagnostics, smaller is worse. For turnover, larger
    # is worse. Cost scenarios remain sensitivity views of each same block.
    conservative_risk_adjusted = np.min(
        np.asarray([risk_adjusted_deltas[s] for s in scenarios], dtype=np.float64), axis=0
    )
    conservative_tail_reduction = np.min(
        np.asarray([tail_reduction_deltas[s] for s in scenarios], dtype=np.float64), axis=0
    )
    conservative_max_drawdown_reduction = np.min(
        np.asarray([max_drawdown_reduction_deltas[s] for s in scenarios], dtype=np.float64), axis=0
    )
    conservative_calibration_reduction = np.min(
        np.asarray([calibration_reduction_deltas[s] for s in scenarios], dtype=np.float64), axis=0
    )
    conservative_turnover_change = np.max(
        np.asarray([turnover_change_deltas[s] for s in scenarios], dtype=np.float64), axis=0
    )
    enough_blocks = len(ordered_blocks) >= minimum_independent_blocks
    candidate_ci = _block_bootstrap_ci(
        conservative_candidate, bootstrap_replicates, bootstrap_seed,
    ) if enough_blocks else None
    control_ci = _block_bootstrap_ci(
        conservative_control, bootstrap_replicates, bootstrap_seed + 1,
    ) if enough_blocks else None
    control_gap_ci = _block_bootstrap_ci(
        conservative_control_gap, bootstrap_replicates, bootstrap_seed + 3,
    ) if enough_blocks else None
    risk_adjusted_ci = _block_bootstrap_ci(
        conservative_risk_adjusted, bootstrap_replicates, bootstrap_seed + 4,
    ) if enough_blocks else None
    tail_reduction_ci = _block_bootstrap_ci(
        conservative_tail_reduction, bootstrap_replicates, bootstrap_seed + 5,
    ) if enough_blocks else None
    max_drawdown_reduction_ci = _block_bootstrap_ci(
        conservative_max_drawdown_reduction, bootstrap_replicates, bootstrap_seed + 8,
    ) if enough_blocks else None
    calibration_reduction_ci = _block_bootstrap_ci(
        conservative_calibration_reduction, bootstrap_replicates, bootstrap_seed + 6,
    ) if enough_blocks else None
    turnover_change_ci = _block_bootstrap_ci(
        conservative_turnover_change, bootstrap_replicates, bootstrap_seed + 7,
    ) if enough_blocks else None
    per_scenario = tuple(
        (scenario, *_block_bootstrap_ci(
            np.asarray(candidate_deltas[scenario], dtype=np.float64),
            bootstrap_replicates, bootstrap_seed + 2 + index,
        ))
        for index, scenario in enumerate(scenarios)
    ) if enough_blocks else ()
    execution_diagnostics = tuple(
        (
            scenario,
            float(np.mean([
                by_block_scenario[(block, scenario)].candidate_fees_bps
                for block in ordered_blocks
            ])),
            float(np.mean([
                by_block_scenario[(block, scenario)].candidate_funding_bps
                for block in ordered_blocks
            ])),
            sum(
                by_block_scenario[(block, scenario)].candidate_filled_orders
                for block in ordered_blocks
            ),
            float(np.mean([
                by_block_scenario[(block, scenario)].candidate_exposure_minutes
                for block in ordered_blocks
            ])),
        )
        for scenario in scenarios
    )
    scenario_candidate_coverage: list[ScenarioCandidateCoverage] = []
    for scenario in scenarios:
        all_outcomes: list[float] = []
        selected_outcomes: list[float] = []
        rejected_outcomes: list[float] = []
        selected_positive = selected_negative = selected_neutral = 0
        rejected_positive = rejected_negative = rejected_neutral = 0
        missed_positive_bps = 0.0
        reason_counts: dict[str, int] = {}
        selected_count = candidate_count = censored_count = 0
        selected_censored_count = rejected_censored_count = 0
        positive_count = negative_count = neutral_count = 0
        for block in ordered_blocks:
            row = by_block_scenario[(block, scenario)]
            selected_ids = set(row.selected_candidate_ids)
            outcomes = dict(row.candidate_outcomes_bps)
            reasons = dict(row.candidate_rejection_reasons)
            candidate_count += len(row.candidate_universe_ids)
            selected_count += len(selected_ids)
            for event_id in row.candidate_universe_ids:
                outcome = outcomes[event_id]
                is_selected = event_id in selected_ids
                if outcome is None:
                    censored_count += 1
                    if is_selected:
                        selected_censored_count += 1
                    else:
                        rejected_censored_count += 1
                else:
                    all_outcomes.append(float(outcome))
                    (selected_outcomes if is_selected else rejected_outcomes).append(
                        float(outcome)
                    )
                    if outcome > 0:
                        positive_count += 1
                        if is_selected:
                            selected_positive += 1
                        else:
                            rejected_positive += 1
                            missed_positive_bps += float(outcome)
                    elif outcome < 0:
                        negative_count += 1
                        if is_selected:
                            selected_negative += 1
                        else:
                            rejected_negative += 1
                    else:
                        neutral_count += 1
                        if is_selected:
                            selected_neutral += 1
                        else:
                            rejected_neutral += 1
                reason = reasons[event_id]
                if reason is not None:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        scenario_candidate_coverage.append(ScenarioCandidateCoverage(
            scenario_id=scenario,
            candidate_count=candidate_count,
            selected_count=selected_count,
            complete_count=len(all_outcomes),
            censored_count=censored_count,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            selected_complete_count=len(selected_outcomes),
            selected_censored_count=selected_censored_count,
            selected_positive_count=selected_positive,
            selected_negative_count=selected_negative,
            selected_neutral_count=selected_neutral,
            rejected_complete_count=len(rejected_outcomes),
            rejected_censored_count=rejected_censored_count,
            rejected_positive_count=rejected_positive,
            rejected_negative_count=rejected_negative,
            rejected_neutral_count=rejected_neutral,
            missed_positive_opportunity_bps=missed_positive_bps,
            mean_complete_outcome_bps=(
                float(np.mean(all_outcomes)) if all_outcomes else None
            ),
            q10_complete_outcome_bps=(
                float(np.quantile(all_outcomes, 0.10)) if all_outcomes else None
            ),
            mean_selected_outcome_bps=(
                float(np.mean(selected_outcomes)) if selected_outcomes else None
            ),
            q10_selected_outcome_bps=(
                float(np.quantile(selected_outcomes, 0.10)) if selected_outcomes else None
            ),
            mean_rejected_outcome_bps=(
                float(np.mean(rejected_outcomes)) if rejected_outcomes else None
            ),
            q10_rejected_outcome_bps=(
                float(np.quantile(rejected_outcomes, 0.10)) if rejected_outcomes else None
            ),
            rejection_reason_counts=tuple(sorted(reason_counts.items())),
        ))
    mean_increment = float(np.mean(conservative_candidate)) if len(ordered_blocks) else None

    status_reasons: tuple[str, ...]
    if len(ordered_blocks) < minimum_independent_blocks:
        status, status_reasons = "INSUFFICIENT_EVIDENCE", ("TOO_FEW_INDEPENDENT_BLOCKS",)
    elif selected_events == 0:
        status, status_reasons = "NO_ELIGIBLE_ALPHA_TEACHER", ("ALL_FLAT_NO_TRADE",)
    elif candidate_ci is None or candidate_ci[0] <= 0:
        status, status_reasons = "NO_PREDICTABLE_INCREMENT", ("NONPOSITIVE_BLOCK_BOOTSTRAP_LOWER_BOUND",)
    elif control_gap_ci is None or control_gap_ci[0] <= 0:
        status, status_reasons = "NO_PREDICTABLE_INCREMENT", ("NEGATIVE_CONTROL_NOT_BEATEN",)
    else:
        status, status_reasons = "G1_TRAIN_INNER_SIGNAL_SCENARIO_ONLY", ()
    return TeacherResearchGateV2Result(
        status=status,
        reason_codes=status_reasons,
        independent_blocks=len(ordered_blocks),
        candidate_events=candidate_events,
        selected_events=selected_events,
        selection_rate=selection_rate,
        mean_worst_scenario_increment_bps=mean_increment,
        increment_ci95_bps=candidate_ci,
        negative_control_delta_ci95_bps=control_ci,
        candidate_vs_negative_control_ci95_bps=control_gap_ci,
        scenario_increment_ci95_bps=per_scenario,
        risk_adjusted_excess_ci95_bps=risk_adjusted_ci,
        tail_loss_reduction_ci95_bps=tail_reduction_ci,
        max_drawdown_reduction_ci95_bps=max_drawdown_reduction_ci,
        calibration_log_loss_reduction_ci95=calibration_reduction_ci,
        turnover_change_ci95_bps=turnover_change_ci,
        scenario_execution_diagnostics=execution_diagnostics,
        scenario_candidate_coverage=tuple(scenario_candidate_coverage),
        preregistration_id=base.preregistration_id,
        trial_id=base.trial_id,
    )


def _block_bootstrap_ci(
    values: np.ndarray, replicates: int, seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[sampled].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _scenario_deltas(
    rows: dict[tuple[str, str], CandidateBlockResult],
    blocks: tuple[str, ...],
    scenarios: tuple[str, ...],
    positive_field: str,
    reference_field: str,
) -> dict[str, list[float]]:
    return {
        scenario: [
            float(
                getattr(rows[(block, scenario)], positive_field)
                - getattr(rows[(block, scenario)], reference_field)
            )
            for block in blocks
        ]
        for scenario in scenarios
    }
