"""Diagnostic accounting for the frozen four-gate opportunity funnel.

This module reports what each unchanged screen gate did. It does not tune
thresholds, inspect outcomes, or decide whether a teacher is qualified.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import re
from typing import Literal, Mapping

from .candidate_events import CandidateFamily
from .opportunity_models import NetValueEstimate


GateName = Literal[
    "in_fit_support",
    "calibrated_probability_gt_0_5",
    "expected_net_increment_gt_0",
    "global_residual_floor_proxy_gt_0",
]


@dataclass(frozen=True, slots=True)
class GateCount:
    gate: GateName
    passed: int
    failed: int


@dataclass(frozen=True, slots=True)
class GateCombinationCount:
    bits: str
    count: int


@dataclass(frozen=True, slots=True)
class GateFunnelGroup:
    block_id: str
    family: CandidateFamily
    side: Literal[-1, 1]
    candidate_count: int
    selected_count: int
    rejection_reason_counts: tuple[tuple[str, int], ...]
    combinations: tuple[GateCombinationCount, ...]


@dataclass(frozen=True, slots=True)
class FrozenGateFunnel:
    candidate_count: int
    selected_count: int
    gate_counts: tuple[GateCount, ...]
    sequential_funnel: tuple[tuple[str, int], ...]
    combinations: tuple[GateCombinationCount, ...]
    by_block_family_side: tuple[GateFunnelGroup, ...]
    selected_ids: tuple[str, ...]
    rejection_reasons: tuple[tuple[str, str], ...]


_GATE_NAMES: tuple[GateName, ...] = (
    "in_fit_support",
    "calibrated_probability_gt_0_5",
    "expected_net_increment_gt_0",
    "global_residual_floor_proxy_gt_0",
)


def summarize_frozen_gate_funnel(
    estimates: Mapping[str, NetValueEstimate],
    group_by_id: Mapping[str, tuple[str, CandidateFamily, Literal[-1, 1]]],
) -> FrozenGateFunnel:
    """Count all four-gate combinations and descriptive groups.

    Gate order and comparisons are frozen to the Route A E1/E2 recipe:
    support, calibrated probability > 0.5, expected increment > 0, and the
    global residual floor proxy > 0. The last value is not a conditional q10.
    The reason code is the first failed gate, preserving runner diagnostics.
    """
    event_ids = set(estimates)
    if event_ids != set(group_by_id):
        raise ValueError("gate estimates and candidate groups must have identical event ids")
    if any(not re.fullmatch(r"[0-9a-f]{64}", event_id) for event_id in event_ids):
        raise ValueError("gate funnel requires auditable SHA-256 event ids")

    combinations = {"".join(bits): 0 for bits in product("01", repeat=4)}
    counts = {name: 0 for name in _GATE_NAMES}
    sequential = [len(event_ids), 0, 0, 0, 0]
    reasons: dict[str, str] = {}
    selected: list[str] = []
    grouped: dict[tuple[str, CandidateFamily, Literal[-1, 1]], list[tuple[str, str]]] = {}

    for event_id in sorted(event_ids):
        estimate = estimates[event_id]
        if estimate.event_id != event_id or type(estimate.in_support) is not bool:
            raise ValueError("gate estimate identity/support is malformed")
        values = (estimate.probability_positive, estimate.expected_net_increment_bps,
                  estimate.global_residual_floor_proxy_bps)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("gate estimate contains a nonfinite score")
        passed = (
            estimate.in_support,
            estimate.probability_positive > 0.5,
            estimate.expected_net_increment_bps > 0.0,
            estimate.global_residual_floor_proxy_bps > 0.0,
        )
        bits = "".join("1" if value else "0" for value in passed)
        combinations[bits] += 1
        for name, value in zip(_GATE_NAMES, passed, strict=True):
            counts[name] += int(value)
        for stage in range(1, len(passed) + 1):
            if all(passed[:stage]):
                sequential[stage] += 1

        first_failed = next((index for index, value in enumerate(passed) if not value), None)
        if first_failed is None:
            selected.append(event_id)
            reason = "SELECTED"
        else:
            reason = (
                "OUT_OF_FIT_SUPPORT",
                "CALIBRATED_PROBABILITY_NOT_ABOVE_0_5",
                "EXPECTED_NET_INCREMENT_NOT_POSITIVE",
                "GLOBAL_RESIDUAL_FLOOR_PROXY_NOT_POSITIVE",
            )[first_failed]
            reasons[event_id] = reason
        grouped.setdefault(group_by_id[event_id], []).append((bits, reason))

    group_rows = []
    for (block_id, family, side), rows in sorted(grouped.items()):
        group_combinations = {bits: 0 for bits in combinations}
        reason_counts: dict[str, int] = {}
        selected_count = 0
        for bits, reason in rows:
            group_combinations[bits] += 1
            if reason == "SELECTED":
                selected_count += 1
            else:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        group_rows.append(GateFunnelGroup(
            block_id=block_id,
            family=family,
            side=side,
            candidate_count=len(rows),
            selected_count=selected_count,
            rejection_reason_counts=tuple(sorted(reason_counts.items())),
            combinations=tuple(GateCombinationCount(bits, count)
                               for bits, count in sorted(group_combinations.items())),
        ))

    return FrozenGateFunnel(
        candidate_count=len(event_ids),
        selected_count=len(selected),
        gate_counts=tuple(
            GateCount(name, counts[name], len(event_ids) - counts[name])
            for name in _GATE_NAMES
        ),
        sequential_funnel=tuple(zip(
            ("all_candidates", *_GATE_NAMES), sequential, strict=True,
        )),
        combinations=tuple(
            GateCombinationCount(bits, count)
            for bits, count in sorted(combinations.items())
        ),
        by_block_family_side=tuple(group_rows),
        selected_ids=tuple(selected),
        rejection_reasons=tuple(sorted(reasons.items())),
    )
