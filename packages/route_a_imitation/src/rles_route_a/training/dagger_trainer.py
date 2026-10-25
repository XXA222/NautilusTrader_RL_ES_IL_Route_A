"""Warm-start BC from a provenance-bound DAgger aggregate."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Callable

from rles_route_a.datasets.daggerset import DAggerSet
from rles_route_a.datasets.sequence_dataset import EpisodeSequenceDataset, LabeledStep
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.models.bc import BCGRUPolicy
from torch.utils.data import Sampler
from .bc_trainer import BCEpoch, train_bc


@dataclass(frozen=True)
class DAggerFineTuneResult:
    """Updated student plus the exact Train aggregate used for the update."""

    policy: BCInferencePolicy
    history: tuple[BCEpoch, ...]
    sample_count: int
    dagger_sample_count: int
    fold_id: str
    source_identity: str
    requested_expert_fraction: float
    sampled_expert_fraction: float
    samples_per_epoch: int


class DAggerMixtureSampler(Sampler[int]):
    """Deterministically sample an exact expert/DAgger mix each epoch."""

    def __init__(self, steps: tuple, *, expert_count: int, expert_fraction: float,
                 samples_per_epoch: int | None, seed: int):
        if (not isinstance(expert_fraction, (int, float)) or isinstance(expert_fraction, bool)
                or not 0 < expert_fraction < 1):
            raise ValueError("expert_fraction must be finite and between zero and one")
        if any(not isinstance(step, LabeledStep) for step in steps):
            raise TypeError("DAgger mixture steps are invalid")
        if type(expert_count) is not int or not 0 < expert_count < len(steps):
            raise ValueError("DAgger mixture requires expert and student samples")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("DAgger mixture seed must be non-negative")
        size = max(len(steps), 100) if samples_per_epoch is None else samples_per_epoch
        if type(size) is not int or size < 2:
            raise ValueError("mixture samples_per_epoch must be an integer >= 2")
        expert = [i for i, step in enumerate(steps[:expert_count])
                  if step.supervision_weight > 0 or step.intent_supervision_weight > 0]
        dagger = [i for i, step in enumerate(steps[expert_count:], start=expert_count)
                  if step.supervision_weight > 0 or step.intent_supervision_weight > 0]
        if not expert or not dagger:
            raise ValueError("both expert and DAgger pools need executable labels")
        expert_draws = round(size * float(expert_fraction))
        dagger_draws = size - expert_draws
        if expert_draws < 1 or dagger_draws < 1:
            raise ValueError("mixture sample size cannot represent both requested pools")
        self.expert_indices = tuple(expert)
        self.dagger_indices = tuple(dagger)
        self.expert_draws = expert_draws
        self.dagger_draws = dagger_draws
        self.num_samples = size
        self.sampled_expert_fraction = expert_draws / size
        self._seed = seed
        self._epoch = 0

    @staticmethod
    def _draw(pool: tuple[int, ...], count: int, rng: random.Random) -> list[int]:
        if count <= len(pool):
            return rng.sample(pool, count)
        return rng.choices(pool, k=count)

    def __iter__(self):
        rng = random.Random(self._seed + self._epoch)
        self._epoch += 1
        indices = self._draw(self.expert_indices, self.expert_draws, rng)
        indices.extend(self._draw(self.dagger_indices, self.dagger_draws, rng))
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


def fine_tune_bc_on_daggerset(
        *, policy: BCInferencePolicy, aggregate: DAggerSet, grant: HostTrainGrant,
        train_grant_verifier: Callable[[HostTrainGrant], bool],
        device: str = "cpu", patience: int = 8, expert_fraction: float = 0.7,
        samples_per_epoch: int | None = None,
) -> DAggerFineTuneResult:
    """Continue an existing GRU student on the full expert+DAgger Train set.

    The model is copied before fitting so a failed update cannot partially
    mutate the caller's live policy. Replace that policy with the returned one
    only after this function succeeds; the new wrapper starts with empty
    episode history.
    """
    if not isinstance(policy, BCInferencePolicy) or not isinstance(policy.model, BCGRUPolicy):
        raise TypeError("an existing Route A GRU BC student policy is required")
    if not isinstance(aggregate, DAggerSet):
        raise TypeError("provenance-bound DAggerSet required")
    if not isinstance(grant, HostTrainGrant) or grant.split != "train":
        raise PermissionError("registered Train grant required for DAgger fine-tuning")
    if not callable(train_grant_verifier) or train_grant_verifier(grant) is not True:
        raise PermissionError("host authority did not verify the Train grant for fine-tuning")
    if (aggregate.fold_id != grant.fold_id
            or aggregate.source_identity != grant.source_identity):
        raise PermissionError("DAgger aggregate differs from the registered Train Fold or source")

    steps = aggregate.training_steps()
    dataset = EpisodeSequenceDataset(
        steps, feature_names=policy.cfg.feature_names,
        context_bars=policy.cfg.context_bars, max_gap_ns=policy.cfg.max_gap_ns,
    )
    if dataset.fold_id != grant.fold_id or dataset.source_identity != grant.source_identity:
        raise PermissionError("DAgger sequence dataset lost its registered Train provenance")
    sampler = DAggerMixtureSampler(
        dataset.steps, expert_count=len(aggregate.original),
        expert_fraction=expert_fraction, samples_per_epoch=samples_per_epoch,
        seed=policy.cfg.seed,
    )

    # Warm-start from the current actor weights while keeping the current live
    # policy unchanged until the complete update has succeeded.
    model = deepcopy(policy.model)
    history = train_bc(
        model, dataset, policy.cfg, device=device, patience=patience, sampler=sampler,
    )
    updated_policy = BCInferencePolicy(model, policy.cfg, device=device)
    return DAggerFineTuneResult(
        policy=updated_policy, history=tuple(history), sample_count=len(dataset),
        dagger_sample_count=aggregate.collected, fold_id=grant.fold_id,
        source_identity=grant.source_identity,
        requested_expert_fraction=expert_fraction,
        sampled_expert_fraction=sampler.sampled_expert_fraction,
        samples_per_epoch=sampler.num_samples,
    )
