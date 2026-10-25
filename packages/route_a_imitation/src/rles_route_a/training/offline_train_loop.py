"""Actual in-memory Train-only update loop for IQL and TD3+BC.

No dataset file read, no native Engine simulation and no Validation/Test lookup.
The calling host must prove RA3 and transaction economics before invoking.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Protocol
import torch
from rles_route_a.datasets.transitions import OfflineBatch, OfflineTransition


@dataclass(frozen=True)
class OfflineReadiness:
    """External RA3/L6 evidence, not a means of self-granting market access."""
    ra3_recovery_accepted: bool
    native_economic_reward_verified: bool
    selected_action_trace_verified: bool
    train_source_registered: bool
    native_execution_recipe: str
    research_receipt: str
    registered_fold_id: str = ""
    reward_scale: float = 1.0
    allow_synthetic_unit_only: bool = False
    # Optional fixed decision clock. Required for a strict temporal discount
    # audit in host integration; 1m=60_000_000_000 ns, 5m=300_000_000_000.
    decision_period_ns: int | None = None
    registered_source_identity: str = ""
    frozen_encoder_identity: str = ""

    def require(self) -> None:
        for name in ("ra3_recovery_accepted", "native_economic_reward_verified",
                     "selected_action_trace_verified", "train_source_registered",
                     "allow_synthetic_unit_only"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"offline readiness {name} must be a strict boolean")
        if (self.decision_period_ns is not None and
                (type(self.decision_period_ns) is not int or self.decision_period_ns <= 0)):
            raise ValueError("positive decision period nanoseconds required")
        if (type(self.reward_scale) not in (int, float) or not isfinite(self.reward_scale)
                or self.reward_scale <= 0):
            raise ValueError("offline reward scale must be finite positive")
        for name in ("registered_fold_id", "registered_source_identity",
                     "frozen_encoder_identity", "native_execution_recipe", "research_receipt"):
            value = getattr(self, name)
            if not isinstance(value, str) or (value and not value.strip()):
                raise ValueError(f"offline readiness {name} must be a nonempty identity string")
        if self.allow_synthetic_unit_only:
            if self.registered_fold_id or self.registered_source_identity or self.frozen_encoder_identity:
                raise PermissionError("synthetic unit readiness cannot claim registered production identities")
        elif not (self.registered_fold_id and self.registered_source_identity
                  and self.frozen_encoder_identity):
            raise PermissionError(
                "production offline RL requires registered Fold, source, and frozen encoder identities"
            )
        if not (self.ra3_recovery_accepted and self.native_economic_reward_verified
                and self.selected_action_trace_verified and self.train_source_registered
                and self.native_execution_recipe and self.research_receipt):
            raise PermissionError("offline RL requires RA3 native, action and Train-only gates")


class ReplaySampler:
    """Checkpointable CPU random index sampling, independent of CUDA RNG."""
    def __init__(self, seed: int):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("invalid replay sampler seed")
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def draw(self, population: int, batch_size: int) -> torch.Tensor:
        if population < 1 or batch_size < 1:
            raise ValueError("invalid replay sample size")
        return torch.randint(0, population, (batch_size,), generator=self.rng)

    def state(self) -> torch.Tensor:
        return self.rng.get_state().clone()

    def restore(self, state: torch.Tensor) -> None:
        if not isinstance(state,torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError("invalid saved replay sampler")
        self.rng.set_state(state)


class OfflineLearner(Protocol):
    device: torch.device
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float | None]: ...


def fit_offline(learner: OfflineLearner, transitions: tuple[OfflineTransition, ...], *,
                readiness: OfflineReadiness, updates: int, batch_size: int,
                seed: int, sampler: ReplaySampler | None = None,
                readiness_verifier: Callable[
                    [OfflineReadiness, tuple[OfflineTransition, ...]], bool
                ] | None = None) -> tuple[dict[str, float | None], ...]:
    readiness.require()
    if type(updates) is not int or updates <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("positive update and batch budgets required")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid seed")
    if not transitions:
        raise ValueError("empty offline training replay")
    if len({x.provenance for x in transitions}) != len(transitions):
        raise ValueError("duplicate native transition provenance; ambiguous replay")
    folds = {t.fold_id for t in transitions}
    if len(folds) != 1:
        raise PermissionError("offline replay may not mix Fold identities")
    if readiness.registered_fold_id:
        if folds != {readiness.registered_fold_id}:
            raise PermissionError("offline replay Fold must match host research receipt")
        if any(not t.episode_id or not t.native_execution_recipe
               or t.native_execution_recipe != readiness.native_execution_recipe
               or t.decision_ts_ns <= 0 or t.next_decision_ts_ns <= t.decision_ts_ns
               or t.action_trace is None
               or t.action_trace.decision_ts_ns != t.decision_ts_ns
               for t in transitions):
            raise PermissionError("native episode/clock/execution provenance or action mapping missing or inconsistent")
        if ({t.source_identity for t in transitions} != {readiness.registered_source_identity}
                or {t.frozen_encoder_id for t in transitions} != {readiness.frozen_encoder_identity}):
            raise PermissionError("offline replay may not mix registered sources or frozen encoders")
    elif not readiness.allow_synthetic_unit_only:
        raise PermissionError("unregistered replay may only be used in explicit synthetic unit tests")
    # Strict native temporal audit: do not bootstrap across missing decisions,
    # nor allow an arbitrary discount_steps to change the algorithm horizon.
    if readiness.registered_fold_id:
        timeline: dict[str, list[OfflineTransition]] = {}
        for t in transitions:
            dt = t.next_decision_ts_ns - t.decision_ts_ns
            if readiness.decision_period_ns is not None:
                if dt != t.discount_steps * readiness.decision_period_ns:
                    raise ValueError("offline gamma discount does not match native decision clock")
            timeline.setdefault(t.episode_id, []).append(t)
        for episode, rows in timeline.items():
            rows.sort(key=lambda t: t.decision_ts_ns)
            for current, following in zip(rows, rows[1:]):
                if current.decision_ts_ns == following.decision_ts_ns:
                    raise ValueError("repeated decision in native offline episode")
                if current.terminated or current.truncated:
                    raise ValueError("offline episode has samples after native terminal")
                if (current.next_decision_ts_ns != following.decision_ts_ns and
                        current.bootstrap_allowed):
                    raise ValueError("offline bootstrap crosses unobserved episode gap")
        if (not callable(readiness_verifier)
                or readiness_verifier(readiness, transitions) is not True):
            raise PermissionError("host authority did not verify RA3/L6 readiness for this exact replay")
    elif readiness_verifier is not None:
        raise PermissionError("synthetic unit replay cannot use a production readiness verifier")
    # Large multi-year 1m replay remains on CPU. Move ONLY sampled minibatches
    # to a GPU learner; storing s/ns for millions of minutes on 8 GiB VRAM can
    # otherwise exhaust device memory before even one optimizer update.
    data = OfflineBatch(transitions, device="cpu", reward_scale=readiness.reward_scale)
    rng = sampler if sampler is not None else ReplaySampler(seed)
    records: list[dict[str, float | None]] = []
    for _ in range(updates):
        draw = rng.draw(data.size, batch_size)
        batch = data.indices(draw)
        if learner.device.type != "cpu":
            batch = {key: tensor.to(learner.device, non_blocking=False)
                     for key, tensor in batch.items()}
        stats = learner.train_step(batch)
        if any(value is not None and not isfinite(value) for value in stats.values()):
            raise FloatingPointError("nonfinite offline trainer metrics")
        records.append(dict(stats))
    return tuple(records)
