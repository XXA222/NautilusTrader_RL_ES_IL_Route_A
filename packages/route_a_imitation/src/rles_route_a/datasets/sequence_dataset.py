"""Strictly causal, fold-isolated and episode-isolated supervised sequences.

The exposure head learns the teacher's pre-risk requested action. Native realized
exposure remains available as outcome evidence, with unexecuted decisions masked.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Sequence
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from rles_route_a.contracts import PointInTimeObservation, ExpertAdvice, INTENTS


@dataclass(frozen=True)
class LabeledStep:
    observation: PointInTimeObservation
    advice: ExpertAdvice
    executable_exposure: float
    action_status: str = "FILLED"
    split: str = "train"
    supervision_weight: float | None = None
    source_identity: str = ""
    trajectory_id: str = ""
    intent_supervision_weight: float | None = None
    economic_case_status: str = "UNASSESSED"
    case_outcome_bps: float | None = None
    case_end_ts_ns: int | None = None
    case_fact_sha256: str = ""
    label_origin: str = "OBSERVED_TEACHER_ACTION"

    def __post_init__(self) -> None:
        if self.split != "train":
            raise PermissionError("Route A supervised training cannot ingest Validation/Test/Holdout")
        if self.advice.decision_ts_ns != self.observation.decision_ts_ns:
            raise ValueError("teacher timestamp does not match observation")
        if not isfinite(self.executable_exposure) or not -1 <= self.executable_exposure <= 1:
            raise ValueError("invalid executable label")
        if not isfinite(self.advice.requested_exposure) or not -1 <= self.advice.requested_exposure <= 1:
            raise ValueError("invalid requested exposure label")
        if self.action_status not in {"FILLED", "NOOP", "WAIT", "CANCEL", "UNFILLED", "PROJECTED"}:
            raise ValueError("unknown native action status")
        default_weight = 1. if self.action_status in {"FILLED", "NOOP", "PROJECTED"} else 0.
        weight = default_weight if self.supervision_weight is None else self.supervision_weight
        if not isinstance(weight, (float, int)) or isinstance(weight, bool) or not isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("supervision weight must be finite in [0,1]")
        if self.action_status in {"WAIT", "CANCEL", "UNFILLED"} and weight != 0:
            raise ValueError("unexecuted/pending target cannot be trained as a realized fill")
        object.__setattr__(self, "supervision_weight", float(weight))
        intent_weight = weight if self.intent_supervision_weight is None else self.intent_supervision_weight
        if (not isinstance(intent_weight, (float, int)) or isinstance(intent_weight, bool)
                or not isfinite(intent_weight) or not 0 <= intent_weight <= 1):
            raise ValueError("intent supervision weight must be finite in [0,1]")
        object.__setattr__(self, "intent_supervision_weight", float(intent_weight))
        valid_case_statuses = {
            "UNASSESSED", "SCENARIO_ONLY", "OBSERVED_NATIVE_STEP_POSITIVE",
            "OBSERVED_NATIVE_STEP_NEGATIVE", "OBSERVED_NATIVE_STEP_NEUTRAL",
            "OBSERVED_NATIVE_TRADE_POSITIVE", "OBSERVED_NATIVE_TRADE_NEGATIVE",
            "OBSERVED_NATIVE_TRADE_NEUTRAL",
        }
        if self.economic_case_status not in valid_case_statuses:
            raise ValueError("unknown Route A economic case status")
        if self.case_outcome_bps is not None and (
            not isinstance(self.case_outcome_bps, (float, int))
            or isinstance(self.case_outcome_bps, bool)
            or not isfinite(self.case_outcome_bps)
        ):
            raise ValueError("case outcome must be finite basis points")
        assessed = self.economic_case_status != "UNASSESSED"
        if assessed:
            if (self.case_outcome_bps is None
                    or type(self.case_end_ts_ns) is not int
                    or self.case_end_ts_ns <= self.observation.decision_ts_ns
                    or not re.fullmatch(r"[0-9a-f]{64}", self.case_fact_sha256)):
                raise ValueError("assessed economic cases require outcome, future end clock, and fact digest")
        elif (self.case_outcome_bps is not None or self.case_end_ts_ns is not None
              or self.case_fact_sha256):
            raise ValueError("unassessed cases cannot carry outcome facts")
        if (self.economic_case_status == "OBSERVED_NATIVE_STEP_POSITIVE"
                and self.case_outcome_bps is not None
                and self.case_outcome_bps <= 0):
            raise ValueError("native positive step must have positive fee-inclusive return")
        if (self.economic_case_status == "OBSERVED_NATIVE_STEP_NEGATIVE"
                and self.case_outcome_bps is not None
                and self.case_outcome_bps >= 0):
            raise ValueError("native negative step must have negative fee-inclusive return")
        if (self.economic_case_status == "OBSERVED_NATIVE_STEP_NEUTRAL"
                and self.case_outcome_bps is not None
                and self.case_outcome_bps != 0):
            raise ValueError("native neutral step must have zero fee-inclusive return")
        if (self.economic_case_status == "OBSERVED_NATIVE_TRADE_POSITIVE"
                and self.case_outcome_bps is not None
                and self.case_outcome_bps <= 0):
            raise ValueError("native positive trade case must have positive fee-inclusive return")
        if (self.economic_case_status == "OBSERVED_NATIVE_TRADE_NEGATIVE"
                and self.case_outcome_bps is not None
                and self.case_outcome_bps >= 0):
            raise ValueError("native negative trade case must have negative fee-inclusive return")
        if (self.economic_case_status == "OBSERVED_NATIVE_TRADE_NEUTRAL"
                and self.case_outcome_bps is not None
                and self.case_outcome_bps != 0):
            raise ValueError("native neutral trade case must have zero fee-inclusive return")
        if self.label_origin not in {
            "OBSERVED_TEACHER_ACTION",
            "INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE",
            "EXCLUDED_FROM_OUTCOME_ABLATION",
        }:
            raise ValueError("unknown Route A teacher-label origin")
        inferred_avoidance = (
            self.label_origin == "INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE"
        )
        avoidance_reason = "negative_native_trade_entry_avoidance"
        if inferred_avoidance:
            if (
                self.economic_case_status != "OBSERVED_NATIVE_TRADE_NEGATIVE"
                or self.advice.intent != "FLAT"
                or self.advice.requested_exposure != 0.0
                or avoidance_reason not in self.advice.reason_codes
                or self.supervision_weight != 0.0
                or self.intent_supervision_weight is None
                or self.intent_supervision_weight <= 0.0
            ):
                raise ValueError("inferred negative-entry label violates its counterfactual contract")
        elif avoidance_reason in self.advice.reason_codes:
            raise ValueError("inferred negative-entry reason requires its explicit label origin")
        if (self.label_origin == "EXCLUDED_FROM_OUTCOME_ABLATION"
                and (self.supervision_weight != 0.0
                     or self.intent_supervision_weight != 0.0
                     or "excluded_from_outcome_ablation" not in self.advice.reason_codes)):
            raise ValueError("excluded ablation row must be explicitly marked and unsupervised")
        if self.source_identity and (not isinstance(self.source_identity, str) or not self.source_identity.strip()):
            raise ValueError("invalid source identity")


class EpisodeSequenceDataset(Dataset):
    """Chronological samples with independent contiguous episode contexts.

    max_gap_ns may be set to one decision-bar interval (or a documented tolerance)
    to reset sequence history after market-data outages. Default preserves v0.1
    behavior for synthetic/nonminute experiments.
    """

    def __init__(self, steps: Sequence[LabeledStep], *, feature_names: Sequence[str],
                 context_bars: int, max_gap_ns: int | None = None):
        if not steps or not feature_names or len(set(feature_names)) != len(feature_names) or type(context_bars) is not int or context_bars < 1:
            raise ValueError("steps, unique features and positive context required")
        if any(not isinstance(x, str) or not x for x in feature_names):
            raise ValueError("invalid frozen feature names")
        if max_gap_ns is not None and (type(max_gap_ns) is not int or max_gap_ns <= 0):
            raise ValueError("max_gap_ns must be positive integer nanoseconds")
        self.feature_names = tuple(feature_names)
        self.context_bars = context_bars
        self.steps = tuple(steps)
        self.max_gap_ns = max_gap_ns
        sources = {item.source_identity for item in self.steps if item.source_identity}
        if len(sources) > 1:
            raise ValueError("supervised dataset cannot mix registered Train sources")
        self.source_identity = next(iter(sources)) if sources else ""
        folds = {item.observation.fold_id for item in self.steps}
        if len(folds) != 1 or not next(iter(folds)):
            raise ValueError("all sequence samples must belong to ONE registered Fold")
        self.fold_id = next(iter(folds))
        self.start_index: list[int] = []
        self._features: list[np.ndarray] = []
        seen_groups: set[tuple[str,str,str]] = set()
        current_group: tuple[str,str,str] | None = None
        last_clock: int | None = None
        start = 0
        for idx, item in enumerate(self.steps):
            o = item.observation
            # Distinct DAgger rollouts may revisit the same market timestamp with
            # different native portfolio states; they must not be concatenated.
            group = (o.fold_id, o.episode_id, item.trajectory_id)
            if group != current_group:
                if group in seen_groups:
                    raise ValueError("discontiguous episode can leak hidden state")
                seen_groups.add(group)
                current_group, start, last_clock = group, idx, None
            if last_clock is not None:
                if o.decision_ts_ns <= last_clock:
                    raise ValueError("non-monotonic decision timestamps")
                if max_gap_ns is not None and o.decision_ts_ns - last_clock > max_gap_ns:
                    start = idx
            last_clock = o.decision_ts_ns
            self.start_index.append(start)
            values = []
            for key in self.feature_names:
                if key.startswith("market."):
                    values.append(o.market_value(key[7:]))
                elif key.startswith("account."):
                    values.append(o.account_value(key[8:]))
                else:
                    raise ValueError(f"unknown feature prefix: {key}")
            row = np.asarray(values, dtype=np.float32)
            if not np.isfinite(row).all():
                raise ValueError("feature overflow after float32 conversion")
            self._features.append(row)
        # Contiguous read-only feature matrix avoids per-row Python array objects
        # and repeated conversion of a 120-step window at each DataLoader fetch.
        self._matrix = np.ascontiguousarray(np.stack(self._features))
        self._matrix.setflags(write=False)
        self._features.clear()

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self.steps):
            raise IndexError(index)
        first = max(self.start_index[index], index - self.context_bars + 1)
        length = index + 1 - first
        x = np.zeros((self.context_bars, len(self.feature_names)), dtype=np.float32)
        x[-length:] = self._matrix[first:index+1]
        mask = np.zeros(self.context_bars, dtype=bool)
        mask[-length:] = True
        item = self.steps[index]
        return {"x": torch.from_numpy(x), "valid_mask": torch.from_numpy(mask),
                "length": torch.tensor(length, dtype=torch.int64),
                "intent": torch.tensor(INTENTS.index(item.advice.intent), dtype=torch.int64),
                "exposure": torch.tensor(item.advice.requested_exposure, dtype=torch.float32),
                "sample_weight": torch.tensor(item.supervision_weight, dtype=torch.float32),
                "command_sample_weight": torch.tensor(
                    item.intent_supervision_weight, dtype=torch.float32
                ),
                "intent_sample_weight": torch.tensor(
                    item.intent_supervision_weight, dtype=torch.float32
                ),
                "case_outcome_bps": torch.tensor(
                    item.case_outcome_bps or 0.0, dtype=torch.float32
                ),
                "case_assessed": torch.tensor(
                    item.economic_case_status != "UNASSESSED", dtype=torch.bool
                )}
