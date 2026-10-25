"""BC training with Train-only inner split and deterministic seed control."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sized
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from typing import cast
from rles_route_a.config import BCConfig
from rles_route_a.contracts import INTENTS
from rles_route_a.models.bc import BCGRUPolicy, bc_losses
from rles_route_a.models.mlp_baseline import BCMLPBaseline


@dataclass(frozen=True)
class BCEpoch:
    epoch: int
    train_loss: float
    train_exposure_huber: float
    train_intent_ce: float
    inner_loss: float | None


def _exposure_sample_weight(batch: dict[str, torch.Tensor], mode: str,
                            device: str) -> torch.Tensor:
    key = "command_sample_weight" if mode == "teacher_request_v2" else "sample_weight"
    if key not in batch:
        raise ValueError(f"BC dataset is missing required {key} for {mode}")
    return batch[key].to(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch_eval(model: torch.nn.Module, dataset: Dataset, batch_size: int, device: str,
                exposure_weight: float, intent_weight: float,
                intent_class_weights: torch.Tensor | None = None,
                exposure_target_mode: str = "executed_only_v1") -> float:
    model.eval()
    loss_sum = 0.0
    count = 0.0
    with torch.no_grad():
        for b in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits, a = model(b["x"].to(device), b["length"].to(device))
            loss = bc_losses(logits, a, target_intent=b["intent"].to(device),
                             target_exposure=b["exposure"].to(device),
                             exposure_weight=exposure_weight, intent_weight=intent_weight,
                             sample_weight=_exposure_sample_weight(
                                 b, exposure_target_mode, device
                             ),
                             intent_sample_weight=(
                                 b["intent_sample_weight"].to(device)
                                 if "intent_sample_weight" in b else None
                             ),
                             intent_class_weights=intent_class_weights)
            effective = float(loss["effective_samples"].item())
            loss_sum += float(loss["loss"].item()) * effective
            count += effective
    if count == 0:
        raise ValueError("Train-inner validation has no supervised teacher action labels")
    return loss_sum / count


def _intent_class_weights(train: Dataset, mode: str, device: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode not in {"inverse_sqrt_v1", "inverse_frequency_v1"}:
        raise ValueError("unsupported BC intent class weighting")
    steps = getattr(train, "steps", None)
    if not isinstance(steps, tuple) or not steps:
        raise ValueError("class-balanced BC requires audited sequence labels")
    counts = np.zeros(len(INTENTS), dtype=np.float64)
    for step in steps:
        weight = float(step.intent_supervision_weight)
        if weight > 0:
            counts[INTENTS.index(step.advice.intent)] += weight
    active = counts > 0
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("class-balanced BC has no executable intent labels")
    class_weights = np.zeros_like(counts)
    if mode == "inverse_sqrt_v1":
        class_weights[active] = np.sqrt(total / counts[active])
    else:
        class_weights[active] = total / counts[active]
    class_weights /= float(np.dot(counts, class_weights) / total)
    return torch.as_tensor(class_weights, dtype=torch.float32, device=device)


def _balanced_intent_sampler(
    train: Dataset, *, seed: int, inverse_frequency_power: float = 1.0
) -> WeightedRandomSampler:
    steps = getattr(train, "steps", None)
    if not isinstance(steps, tuple) or not steps:
        raise ValueError("balanced intent sampling requires audited sequence labels")
    counts = np.zeros(len(INTENTS), dtype=np.float64)
    for step in steps:
        counts[INTENTS.index(step.advice.intent)] += float(step.intent_supervision_weight)
    if not np.any(counts > 0):
        raise ValueError("balanced intent sampling has no teacher action labels")
    if not 0 < inverse_frequency_power <= 1:
        raise ValueError("inverse intent frequency power must be in (0, 1]")
    weights = [
        (float(step.intent_supervision_weight)
         / counts[INTENTS.index(step.advice.intent)] ** inverse_frequency_power)
        if step.intent_supervision_weight > 0 else 0.0
        for step in steps
    ]
    return WeightedRandomSampler(
        weights,
        num_samples=len(steps),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _native_case_balanced_sampler(
    train: Dataset, *, seed: int, exposure_target_mode: str
) -> WeightedRandomSampler | None:
    """Balance positive native cases and explicit losing-entry counterexamples.

    Sampling is case-aware because per-case row weights alone are normalized again
    inside each BC minibatch and therefore cannot balance cases spanning different
    numbers of batches. Negative cases contribute only their explicit flat-at-entry
    counterexample; all other negative, neutral, and unassessed rows stay excluded.
    """
    steps = getattr(train, "steps", None)
    if not isinstance(steps, tuple) or not steps:
        return None
    eligible: dict[str, list[tuple[int, float]]] = {}
    for index, step in enumerate(steps):
        supervised_weight = (
            float(step.intent_supervision_weight)
            if exposure_target_mode == "teacher_request_v2"
            else float(step.supervision_weight)
        )
        positive_case = step.economic_case_status == "OBSERVED_NATIVE_TRADE_POSITIVE"
        negative_entry_counterexample = (
            step.economic_case_status == "OBSERVED_NATIVE_TRADE_NEGATIVE"
            and step.label_origin == "INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE"
            and "negative_native_trade_entry_avoidance" in step.advice.reason_codes
            and step.advice.intent == "FLAT"
            and step.advice.requested_exposure == 0.0
        )
        if (positive_case or negative_entry_counterexample) and step.case_fact_sha256 and supervised_weight > 0:
            eligible.setdefault(step.case_fact_sha256, []).append(
                (index, supervised_weight)
            )
    if not eligible:
        return None
    weights = torch.zeros(len(steps), dtype=torch.double)
    for case_rows in eligible.values():
        case_supervision = sum(weight for _, weight in case_rows)
        if case_supervision <= 0:
            continue
        for index, _ in case_rows:
            weights[index] = 1.0 / case_supervision
    sample_count = sum(len(case_rows) for case_rows in eligible.values())
    return WeightedRandomSampler(
        weights.tolist(), num_samples=sample_count, replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def train_bc(model: torch.nn.Module, train: Dataset, config: BCConfig, *,
             device: str = "cpu", inner_validation: Dataset | None = None,
             patience: int = 8, sampler: Sampler[int] | None = None) -> list[BCEpoch]:
    if type(patience) is not int or patience < 1:
        raise ValueError("positive BC early-stop patience required")
    if len(cast(Sized, train)) == 0 or (
            inner_validation is not None and len(cast(Sized, inner_validation)) == 0):
        raise ValueError("empty Train/Train-inner dataset")
    if isinstance(model, BCGRUPolicy):
        actual_features = model.encoder.feature_names
    elif isinstance(model, BCMLPBaseline):
        actual_features = model.feature_names
    else:
        raise TypeError("BC trainer supports only the frozen GRU and MLP policies")
    if tuple(actual_features) != tuple(config.feature_names):
        raise ValueError("model feature schema differs from frozen BC config")
    if hasattr(train, "fold_id") and inner_validation is not None and hasattr(inner_validation, "fold_id") and train.fold_id != inner_validation.fold_id:
        raise ValueError("BC Train-inner validation fold identity mismatch")
    if hasattr(train, "steps"):
        supervised = tuple(
            item.intent_supervision_weight is not None
            and item.intent_supervision_weight > 0
            if config.exposure_target_mode == "teacher_request_v2"
            else item.supervision_weight is not None and item.supervision_weight > 0
            for item in train.steps
        )
    else:
        supervised = (True,)
    if hasattr(train, "steps") and not any(supervised):
        statuses: dict[str, int] = {}
        for item in train.steps:
            statuses[item.action_status] = statuses.get(item.action_status, 0) + 1
        label_facts = [
            (item.action_status, item.advice.requested_exposure,
             item.executable_exposure, item.supervision_weight)
            for item in train.steps
        ]
        raise ValueError(
            "no teacher actions allowed by the selected BC target recipe; "
            f"native action statuses={dict(sorted(statuses.items()))}; "
            f"(status, requested, actual, weight)={label_facts}"
        )
    if hasattr(train, "max_gap_ns") and train.max_gap_ns != config.max_gap_ns:
        raise ValueError("BC Train history gap contract differs from inference config")
    if (inner_validation is not None and hasattr(inner_validation, "max_gap_ns")
            and inner_validation.max_gap_ns != config.max_gap_ns):
        raise ValueError("BC Train-inner history gap contract differs from inference config")
    set_seed(config.seed)
    model.to(device)
    intent_class_weights = _intent_class_weights(
        train, config.intent_class_weighting, device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    active_sampler = sampler
    if active_sampler is None:
        active_sampler = _native_case_balanced_sampler(
            train, seed=config.seed,
            exposure_target_mode=config.exposure_target_mode,
        )
    if active_sampler is None and config.intent_sampling in {
        "balanced_intent_v1", "sqrt_balanced_intent_v1",
        "moderately_balanced_intent_v1"
    }:
        power = {
            "balanced_intent_v1": 1.0,
            "sqrt_balanced_intent_v1": 0.5,
            "moderately_balanced_intent_v1": 0.75,
        }[config.intent_sampling]
        active_sampler = _balanced_intent_sampler(
            train, seed=config.seed, inverse_frequency_power=power
        )
    loader = DataLoader(
        train, batch_size=config.batch_size, shuffle=active_sampler is None,
        generator=generator, sampler=active_sampler,
    )
    best = float("inf")
    best_weights: dict | None = None
    no_gain = 0
    history: list[BCEpoch] = []
    for ep in range(config.epochs):
        model.train()
        total = np.zeros(3, dtype=np.float64)
        n = 0.0
        for b in loader:
            opt.zero_grad(set_to_none=True)
            logits, a = model(b["x"].to(device), b["length"].to(device))
            d = bc_losses(logits, a, target_intent=b["intent"].to(device),
                          target_exposure=b["exposure"].to(device),
                          exposure_weight=config.exposure_weight,
                          intent_weight=config.intent_weight,
                          sample_weight=_exposure_sample_weight(
                              b, config.exposure_target_mode, device
                          ),
                          intent_sample_weight=(
                              b["intent_sample_weight"].to(device)
                              if "intent_sample_weight" in b else None
                          ),
                          intent_class_weights=intent_class_weights)
            if not bool(torch.isfinite(d["loss"]).item()):
                raise FloatingPointError("non-finite BC training loss")
            if (
                float(_exposure_sample_weight(
                    b, config.exposure_target_mode, device
                ).sum().item()) == 0
                and float(b.get("intent_sample_weight", torch.zeros(1)).sum().item()) == 0
            ):
                continue
            d["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                                   error_if_nonfinite=True)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("BC nonfinite gradient")
            opt.step()
            weight = float(d["effective_samples"].item())
            total += weight * np.array([float(d[k].item()) for k in ("loss", "exposure_huber", "intent_ce")])
            n += weight
        inner = (_epoch_eval(model, inner_validation, config.batch_size, device,
                              config.exposure_weight, config.intent_weight,
                              intent_class_weights, config.exposure_target_mode)
                 if inner_validation is not None else None)
        if n == 0:
            raise ValueError("no effective BC training minibatches")
        mean_metrics = (total / n).tolist()
        history.append(BCEpoch(ep + 1, float(mean_metrics[0]),
                               float(mean_metrics[1]), float(mean_metrics[2]), inner))
        if inner is not None:
            if inner < best - 1e-7:
                best, no_gain = inner, 0
                best_weights = copy.deepcopy(model.state_dict())
            else:
                no_gain += 1
                if no_gain >= patience:
                    break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return history
