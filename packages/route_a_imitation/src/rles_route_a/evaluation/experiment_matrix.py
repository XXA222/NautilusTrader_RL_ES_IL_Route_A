"""Freeze experiment identity; the L6 host alone runs real economic comparison."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

CostScenario = Literal["zero", "base", "stress"]
ExperimentStage = Literal["baseline", "teacher", "bc", "dagger", "iql", "td3bc"]

ARM_STAGES: dict[str, ExperimentStage] = {
    "no_trade": "baseline",
    "always_long": "baseline",
    "momentum": "baseline",
    "random": "baseline",
    "teacher_flat_risk": "teacher",
    "teacher_trend": "teacher",
    "teacher_selector": "teacher",
    "bc_mlp": "bc",
    "bc_gru_1m": "bc",
    "bc_gru_multiscale": "bc",
    "bc_perturbed_multiscale": "bc",
    "bc_dagger_multiscale": "dagger",
    "iql_dagger_frozen_encoder": "iql",
    "td3bc_dagger": "td3bc",
}

PREREGISTERED_ARMS = tuple(ARM_STAGES)
PREREGISTERED_SEEDS = (11, 23, 31, 47, 59)


@dataclass(frozen=True)
class ExperimentArm:
    name: str
    stage: ExperimentStage
    recipe_id: str
    seed: int
    cost_scenario: CostScenario

    def __post_init__(self) -> None:
        if (not isinstance(self.name, str) or not self.name.strip()
                or not isinstance(self.recipe_id, str) or not self.recipe_id.strip()
                or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or ARM_STAGES.get(self.name) != self.stage
                or self.stage not in {"baseline", "teacher", "bc", "dagger", "iql", "td3bc"}
                or self.cost_scenario not in {"zero", "base", "stress"}):
            raise ValueError("frozen experiment arm requires identities")


def build_matrix(*, recipes: dict[str, str], seeds: tuple[int, ...] = (11, 23, 31, 47, 59),
                 scenarios: tuple[CostScenario, ...] = ("zero", "base", "stress")) -> tuple[ExperimentArm, ...]:
    """Build an explicit matrix; formal RA5 runs should use the complete builder below."""
    if not isinstance(recipes, dict) or not recipes or any(k not in ARM_STAGES for k in recipes):
        raise ValueError("only known pre-registered experiment arms allowed")
    if (not isinstance(seeds, tuple) or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("experiment seeds must be unique unsigned integers")
    if (not isinstance(scenarios, tuple) or not scenarios
            or any(scenario not in {"zero", "base", "stress"} for scenario in scenarios)
            or len(set(scenarios)) != len(scenarios)):
        raise ValueError("cost scenarios must be unique members of zero/base/stress")
    if any(not isinstance(recipe, str) or not recipe.strip() for recipe in recipes.values()):
        raise ValueError("every experiment arm must bind a frozen recipe identity")
    arms = [ExperimentArm(name, ARM_STAGES[name], recipe, seed, scenario)
            for name, recipe in sorted(recipes.items()) for seed in seeds for scenario in scenarios]
    if len({(a.name, a.seed, a.cost_scenario) for a in arms}) != len(arms):
        raise ValueError("duplicate experiment identity")
    return tuple(arms)


def build_preregistered_matrix(*, recipes: dict[str, str],
                               seeds: tuple[int, ...] = PREREGISTERED_SEEDS,
                               scenarios: tuple[CostScenario, ...] = ("zero", "base", "stress")) -> tuple[ExperimentArm, ...]:
    """Build the complete B0/E1-E8 comparison; missing arms cannot disappear."""
    if not isinstance(recipes, dict):
        raise TypeError("frozen experiment recipes must be a mapping")
    if set(recipes) != set(PREREGISTERED_ARMS):
        missing = sorted(set(PREREGISTERED_ARMS) - set(recipes))
        extra = sorted(set(recipes) - set(PREREGISTERED_ARMS))
        raise ValueError(f"incomplete preregistered arm set missing={missing} extra={extra}")
    if (not isinstance(scenarios, tuple)
            or set(scenarios) != {"zero", "base", "stress"}):
        raise ValueError("formal RA5 comparison requires zero/base/stress cost scenarios")
    if seeds != PREREGISTERED_SEEDS:
        raise ValueError("formal RA5 comparison requires the frozen five-seed set")
    return build_matrix(recipes=recipes, seeds=seeds, scenarios=scenarios)
