"""Controlled outcome-conditioned BC ablation datasets from native trade cases."""
from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass, replace
from typing import Literal, Sequence, cast

import torch
from torch.utils.data import Dataset, RandomSampler

from rles_route_a.datasets.sequence_dataset import LabeledStep


OutcomeSupervisionArmName = Literal[
    "full_teacher_behavior",
    "positive_native_case_weighted",
    "positive_plus_negative_entry_avoidance",
]


@dataclass(frozen=True)
class OutcomeSupervisionArm:
    """One controlled training arm; row order and causal contexts are preserved."""

    name: OutcomeSupervisionArmName
    steps: tuple[LabeledStep, ...]
    fold_id: str
    source_identity: str
    positive_case_count: int
    negative_case_count: int
    exposure_supervised_rows: int
    intent_supervised_rows: int


def build_outcome_supervision_arms(
        steps: Sequence[LabeledStep],
) -> tuple[OutcomeSupervisionArm, OutcomeSupervisionArm,
           OutcomeSupervisionArm]:
    """Build the three preregistered outcome-label controls without reordering.

    Positive trade cases are weighted to equal total pre-minibatch
    exposure/intent weight per completed trade, rather than per decision row.
    Negative-entry avoidance contributes only its explicit intent label; it
    never fabricates a realized flat exposure. All cases must be fee-inclusive
    native receipts in one Train Fold/source; scenario-only or unassessed data
    fail closed.
    """
    if not steps or any(not isinstance(step, LabeledStep) for step in steps):
        raise ValueError("native labeled Train steps are required")
    rows = tuple(steps)
    folds = {step.observation.fold_id for step in rows}
    sources = {step.source_identity for step in rows}
    if len(folds) != 1 or not next(iter(folds)) or len(sources) != 1 or not next(iter(sources)):
        raise PermissionError("outcome ablation must use one registered Train Fold and source")
    fold_id = next(iter(folds))
    source_identity = next(iter(sources))
    if any(step.split != "train" or step.observation.fold_id != fold_id
           or step.source_identity != source_identity for step in rows):
        raise PermissionError("outcome ablation cannot mix folds, sources, or splits")

    positive_cases: dict[str, list[int]] = {}
    negative_avoidance_cases: dict[str, list[int]] = {}
    case_facts: dict[str, tuple[str, float, int]] = {}
    for index, step in enumerate(rows):
        if step.economic_case_status.startswith("SCENARIO_ONLY"):
            raise PermissionError("scenario-only outcomes cannot enter native outcome ablation")
        if step.economic_case_status.startswith("OBSERVED_NATIVE_TRADE_"):
            if (not step.case_fact_sha256 or step.case_outcome_bps is None
                    or step.case_end_ts_ns is None):
                raise ValueError("native trade case is missing bound outcome facts")
            fact = (step.economic_case_status, float(step.case_outcome_bps),
                    step.case_end_ts_ns)
            previous = case_facts.setdefault(step.case_fact_sha256, fact)
            if previous != fact:
                raise ValueError("one trade receipt digest has conflicting case outcomes")
        if step.economic_case_status == "OBSERVED_NATIVE_TRADE_POSITIVE":
            if not step.case_fact_sha256:
                raise ValueError("positive native case is missing its receipt identity")
            positive_cases.setdefault(step.case_fact_sha256, []).append(index)
        if step.label_origin == "INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE":
            if step.economic_case_status != "OBSERVED_NATIVE_TRADE_NEGATIVE" or not step.case_fact_sha256:
                raise ValueError("negative-entry label lacks a native losing-trade receipt")
            negative_avoidance_cases.setdefault(step.case_fact_sha256, []).append(index)
    if not positive_cases:
        raise ValueError("outcome ablation requires at least one completed positive native trade")
    if not negative_avoidance_cases:
        raise ValueError("three-arm ablation requires a native negative-entry avoidance label")
    if any(len(indices) != 1 for indices in negative_avoidance_cases.values()):
        raise ValueError("a losing native trade may contribute only one entry-avoidance label")

    observed = tuple(
        step if step.label_origin == "OBSERVED_TEACHER_ACTION" else _exclude(step)
        for step in rows
    )
    weighted = [
        _exclude(step)
        if step.label_origin == "INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE"
        else replace(step, supervision_weight=0.0,
                     intent_supervision_weight=0.0)
        for step in rows
    ]
    for case_rows in positive_cases.values():
        exposure_total = sum(_weight(rows[i].supervision_weight) for i in case_rows)
        intent_total = sum(_weight(rows[i].intent_supervision_weight) for i in case_rows)
        if exposure_total <= 0 or intent_total <= 0:
            raise ValueError("positive native trade case has no executable BC labels")
        for i in case_rows:
            step = rows[i]
            exposure_weight = (_weight(step.supervision_weight) / exposure_total
                               if exposure_total > 0 else 0.0)
            intent_weight = (_weight(step.intent_supervision_weight) / intent_total
                             if intent_total > 0 else 0.0)
            weighted[i] = replace(
                step, supervision_weight=exposure_weight,
                intent_supervision_weight=intent_weight,
            )

    avoid = list(weighted)
    for case_rows in negative_avoidance_cases.values():
        intent_total = sum(_weight(rows[i].intent_supervision_weight)
                           for i in case_rows)
        for i in case_rows:
            step = rows[i]
            avoid[i] = replace(
                step, supervision_weight=0.0,
                intent_supervision_weight=(
                    _weight(step.intent_supervision_weight) / intent_total
                    if intent_total > 0 else 0.0
                ),
            )

    def arm(name: OutcomeSupervisionArmName,
            values: tuple[LabeledStep, ...]) -> OutcomeSupervisionArm:
        return OutcomeSupervisionArm(
            name=name, steps=values, fold_id=fold_id,
            source_identity=source_identity,
            positive_case_count=len(positive_cases),
            negative_case_count=len(negative_avoidance_cases),
            exposure_supervised_rows=sum(_weight(step.supervision_weight) > 0
                                         for step in values),
            intent_supervised_rows=sum(_weight(step.intent_supervision_weight) > 0
                                       for step in values),
        )

    return (
        arm("full_teacher_behavior", observed),
        arm("positive_native_case_weighted", tuple(weighted)),
        arm("positive_plus_negative_entry_avoidance", tuple(avoid)),
    )


def _exclude(step: LabeledStep) -> LabeledStep:
    """Retain row position/receipt while making an omitted counterfactual explicit."""
    advice = replace(
        step.advice,
        reason_codes=("excluded_from_outcome_ablation",),
    )
    return replace(
        step, advice=advice, supervision_weight=0.0,
        intent_supervision_weight=0.0,
        label_origin="EXCLUDED_FROM_OUTCOME_ABLATION",
    )


def _weight(value: float | None) -> float:
    if value is None:
        raise ValueError("validated native label is missing its supervision weight")
    return float(value)


def outcome_ablation_sampler(dataset: Dataset, *, seed: int) -> RandomSampler:
    """Use an identical-seed row sampler to bypass BC's automatic case sampler.

    Construct one sampler per arm with the same-sized EpisodeSequenceDataset;
    pass it explicitly to ``train_bc``. Keep all other optimizer, class-weight,
    batching, epoch, and early-stop settings identical across arms.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("outcome ablation requires a frozen non-negative sampler seed")
    return RandomSampler(cast(Sized, dataset),
                         generator=torch.Generator().manual_seed(seed))
