"""Train-only append-only student-state aggregation, transactionally per batch."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Sequence
from rles_route_a.datasets.sequence_dataset import LabeledStep


@dataclass(frozen=True)
class AggregatedLabel:
    iteration: int
    student_policy_id: str
    sample: LabeledStep
    execution_recipe: str = ""


class DAggerSet:
    def __init__(self, original: tuple[LabeledStep, ...]):
        if not original:
            raise ValueError("initial expert dataset required")
        if any(not isinstance(sample, LabeledStep) for sample in original):
            raise TypeError("initial DAgger rows must be audited LabeledStep records")
        folds = {s.observation.fold_id for s in original}
        if len(folds) != 1 or not next(iter(folds)):
            raise ValueError("DAgger original data must belong to one registered Fold")
        self.fold_id = next(iter(folds))
        source_ids = {x.source_identity for x in original}
        source_identity = next(iter(source_ids)) if len(source_ids) == 1 else None
        if not isinstance(source_identity, str) or not source_identity.strip():
            raise ValueError("original DAgger data must bind one registered Train source")
        self.source_identity = source_identity
        self.original = original
        self._added: list[AggregatedLabel] = []
        self._ids: set[tuple[str,int,str]] = set()
        self._projection_recipe: str | None = None

    def append(self, label: AggregatedLabel) -> None:
        self.extend((label,))

    def extend(self, labels: Sequence[AggregatedLabel]) -> None:
        """Atomic logical insertion: validate all rows before mutating collection."""
        batch: set[tuple[str,int,str]] = set()
        recipe = self._projection_recipe
        for label in labels:
            if type(label.iteration) is not int or label.iteration < 1 or not label.student_policy_id:
                raise ValueError("iteration/student identity required")
            obs = label.sample.observation
            if obs.fold_id != self.fold_id:
                raise PermissionError("DAgger may not aggregate labels across different Folds")
            if label.sample.source_identity != self.source_identity:
                raise PermissionError("DAgger sample provenance must match original source")
            if label.execution_recipe:
                if recipe is not None and label.execution_recipe != recipe:
                    raise ValueError("DAgger collection cannot mix projection execution recipes")
                recipe = label.execution_recipe
            elif recipe is not None:
                raise ValueError("DAgger sample missing frozen execution recipe")
            key = (obs.episode_id, obs.decision_ts_ns, label.student_policy_id)
            if key in self._ids or key in batch:
                raise ValueError("duplicate learner-state label")
            batch.add(key)
        self._ids.update(batch)
        self._added.extend(labels)
        self._projection_recipe = recipe

    def training_steps(self) -> tuple[LabeledStep, ...]:
        """Stable chronological trajectories; NEVER splice into teacher's episode.

        Original labels are preserved. Each student rollout is a new trajectory;
        same original episode ID and clock across different policies is legal.
        """
        grouped: dict[tuple[int,str,str], list[LabeledStep]] = {}
        for row in self._added:
            obs = row.sample.observation
            key = (row.iteration, row.student_policy_id, obs.episode_id)
            trajectory_id = f"dagger:{row.iteration}:{row.student_policy_id}:{obs.episode_id}"
            grouped.setdefault(key, []).append(replace(row.sample, trajectory_id=trajectory_id))
        results = list(self.original)
        for rows in grouped.values():
            rows.sort(key=lambda x:x.observation.decision_ts_ns)
            results.extend(rows)
        return tuple(results)

    def fork(self) -> DAggerSet:
        """Return an independent append-only copy for transactional training."""
        duplicate = DAggerSet(self.original)
        duplicate.extend(tuple(self._added))
        return duplicate

    def added_since(self, count: int) -> tuple[AggregatedLabel, ...]:
        if type(count) is not int or not 0 <= count <= len(self._added):
            raise ValueError("invalid DAgger aggregate checkpoint")
        return tuple(self._added[count:])

    @property
    def collected(self) -> int:
        return len(self._added)
