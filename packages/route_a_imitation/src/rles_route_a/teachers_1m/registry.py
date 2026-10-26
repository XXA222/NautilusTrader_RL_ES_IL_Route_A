"""Explicit frozen registry for independent Route A one-minute teachers."""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .base import BaseTeacher1m
from .contracts import TeacherRecipe1m
from .donchian import DonchianBreakoutTeacher
from .ema import EmaCrossTeacher
from .exits import AtrTrailingExit
from .momentum import RocConsensusTeacher
from .zscore import ZscoreReentryTeacher


def _recipe(
    *, teacher_id: str, indicator_recipe_id: str,
    parameters: tuple[tuple[str, float | int | str], ...], warmup_bars: int,
    entry_rule_id: str, indicator_exit_rule_id: str,
) -> TeacherRecipe1m:
    return TeacherRecipe1m(
        teacher_id=teacher_id,
        indicator_recipe_id=indicator_recipe_id,
        parameters=parameters,
        warmup_bars=warmup_bars,
        entry_rule_id=entry_rule_id,
        indicator_exit_rule_id=indicator_exit_rule_id,
        htf_mode="FEATURE_ONLY",
        target_exposure=0.3,
        cooldown_policy="NEW_EVENT_ONLY_NO_TIME_COOLDOWN",
        order_policy="MARKET_NATIVE_L3_NETTING",
        risk_recipe_id="NATIVE_ROUTE_A_HARD_SOFT_RISK",
        scenario_cost_ids=("base", "stress"),
    )

FIRST_BATCH_RECIPES = (
    _recipe(
        teacher_id="T01_EMA_9_21",
        indicator_recipe_id="route_a.ema_9_21.v1",
        parameters=(("fast", 9), ("slow", 21), ("reentry_mode", "NEW_CROSS_ONLY")),
        warmup_bars=21,
        entry_rule_id="EMA_CROSS_V1",
        indicator_exit_rule_id="X1_EMA_REVERSE_CROSS_V1",
    ),
    _recipe(
        teacher_id="T02_ROC_20_60",
        indicator_recipe_id="route_a.roc_20_60.v1",
        parameters=(("fast", 20), ("slow", 60), ("event", "FIRST_CONSENSUS_ONLY")),
        warmup_bars=61,
        entry_rule_id="ROC_CONSENSUS_V1",
        indicator_exit_rule_id="X1_ROC_CONSENSUS_BROKEN_V1",
    ),
    _recipe(
        teacher_id="T03_DONCHIAN_20",
        indicator_recipe_id="route_a.donchian_20_10.v1",
        parameters=(("entry_window", 20), ("exit_window", 10), ("breakout_margin", 0.0)),
        warmup_bars=21,
        entry_rule_id="DONCHIAN_PRIOR_WINDOW_BREAKOUT_V1",
        indicator_exit_rule_id="X1_DONCHIAN_10_OPPOSITE_V1",
    ),
    _recipe(
        teacher_id="T04_ZSCORE_60",
        indicator_recipe_id="route_a.zscore_60.v1",
        parameters=(("lookback", 60), ("entry_abs", 2.0), ("reentry_abs", 1.8)),
        warmup_bars=60,
        entry_rule_id="ZSCORE_REENTRY_CONFIRMATION_V1",
        indicator_exit_rule_id="X1_ZSCORE_MEAN_OR_OPPOSITE_V1",
    ),
)

FIRST_BATCH_RECIPE_BY_ID = {recipe.teacher_id: recipe for recipe in FIRST_BATCH_RECIPES}

_FACTORIES: dict[str, type[BaseTeacher1m]] = {
    "T01_EMA_9_21": EmaCrossTeacher,
    "T02_ROC_20_60": RocConsensusTeacher,
    "T03_DONCHIAN_20": DonchianBreakoutTeacher,
    "T04_ZSCORE_60": ZscoreReentryTeacher,
}

if set(FIRST_BATCH_RECIPE_BY_ID) != set(_FACTORIES):
    raise RuntimeError("first-batch teacher recipes and factories differ")


def build_teacher_1m(
    teacher_id: str, *, exit_variant: Literal["X1", "X2"] = "X1"
) -> BaseTeacher1m:
    if exit_variant not in ("X1", "X2"):
        raise ValueError(f"unknown Route A 1m exit variant: {exit_variant}")
    try:
        recipe = FIRST_BATCH_RECIPE_BY_ID[teacher_id]
        factory = _FACTORIES[teacher_id]
    except KeyError as exc:
        raise ValueError(f"unknown Route A 1m teacher: {teacher_id}") from exc
    if exit_variant == "X2":
        recipe = replace(recipe, indicator_exit_rule_id="X2_ATR_TRAIL_3_V1")
        return factory(recipe, atr_exit=AtrTrailingExit(3.0))
    return factory(recipe)

