"""Explicit frozen registry for independent Route A one-minute teachers."""
from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Literal

from .base import BaseTeacher1m
from .contracts import TeacherRecipe1m
from .donchian import DonchianBreakoutTeacher
from .ema import EmaCrossTeacher
from .exits import (
    AtrTrailingExit,
    AtrTrailHandoffToIndicatorExit,
    CostAwareAtrProfitAndLossGuardExit,
    CostAwareAtrProfitProtectExit,
)
from .bollinger import BollingerReentryTeacher
from .macd import MacdCrossTeacher
from .momentum import RocConsensusTeacher
from .pullback import TrendPullbackTeacher
from .rsi import RsiReentryTeacher
from .roc_hysteresis import RocCostHysteresisTeacher
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

ROC_COST_HYSTERESIS_TEACHER_ID = "T02_ROC_20_60_COST_HYSTERESIS_V1"
ROC_COST_HYSTERESIS_FADE_TEACHER_ID = "T02_ROC_20_60_COST_HYSTERESIS_FADE_V1"


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

SECOND_BATCH_RECIPES = (
    _recipe(
        teacher_id="T05_MACD_12_26_9",
        indicator_recipe_id="route_a.macd_12_26_9.v1",
        parameters=(
            ("fast", 12), ("slow", 26), ("signal", 9),
            ("entry_rule", "CROSS_AND_HIST_IMPROVEMENT"),
        ),
        warmup_bars=34,
        entry_rule_id="MACD_CROSS_HIST_IMPROVING_V1",
        indicator_exit_rule_id="X1_MACD_REVERSE_CROSS_V1",
    ),
    _recipe(
        teacher_id="T06_RSI_14",
        indicator_recipe_id="route_a.rsi_14.v1",
        parameters=(
            ("lookback", 14), ("oversold", 30),
            ("overbought", 70), ("neutral_exit", 50),
        ),
        warmup_bars=15,
        entry_rule_id="RSI_EXTREME_REENTRY_V1",
        indicator_exit_rule_id="X1_RSI_NEUTRAL_V1",
    ),
    _recipe(
        teacher_id="T07_BOLL_20_2",
        indicator_recipe_id="route_a.bollinger_20_2.v1",
        parameters=(
            ("lookback", 20), ("std_multiplier", 2.0),
            ("exit", "MID_OR_OUTER_RISK"),
        ),
        warmup_bars=20,
        entry_rule_id="BOLLINGER_OUTSIDE_REENTRY_V1",
        indicator_exit_rule_id="X1_BOLLINGER_MID_OR_RISK_V1",
    ),
    _recipe(
        teacher_id="T08_PULLBACK_1M",
        indicator_recipe_id="route_a.ema_pullback_1m.v1",
        parameters=(
            ("ema_fast", 9), ("ema_slow", 21), ("atr", 14),
            ("pullback_lookback", 5), ("recovery_lookback", 3),
            ("pullback_atr_min", 0.3), ("pullback_atr_max", 2.0),
        ),
        warmup_bars=21,
        entry_rule_id="EMA_TREND_ATR_PULLBACK_RECOVERY_V1",
        indicator_exit_rule_id="X1_PULLBACK_TREND_OR_CHANNEL_V1",
    ),
)
SECOND_BATCH_RECIPE_BY_ID = {
    recipe.teacher_id: recipe for recipe in SECOND_BATCH_RECIPES
}
ALL_TEACHER_RECIPE_BY_ID = {
    **FIRST_BATCH_RECIPE_BY_ID,
    **SECOND_BATCH_RECIPE_BY_ID,
}

_FACTORIES: dict[str, type[BaseTeacher1m]] = {
    "T01_EMA_9_21": EmaCrossTeacher,
    "T02_ROC_20_60": RocConsensusTeacher,
    "T03_DONCHIAN_20": DonchianBreakoutTeacher,
    "T04_ZSCORE_60": ZscoreReentryTeacher,
    "T05_MACD_12_26_9": MacdCrossTeacher,
    "T06_RSI_14": RsiReentryTeacher,
    "T07_BOLL_20_2": BollingerReentryTeacher,
    "T08_PULLBACK_1M": TrendPullbackTeacher,
}

if set(ALL_TEACHER_RECIPE_BY_ID) != set(_FACTORIES):
    raise RuntimeError("teacher recipes and factories differ")


def build_teacher_1m(
    teacher_id: str, *,
    exit_variant: Literal["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"] = "X1",
    reversal_mode: Literal["EXIT_ONLY_V1", "REVERSE_AFTER_FLAT_V1"] = "EXIT_ONLY_V1",
    roc_round_trip_cost_bps: float | None = None,
    exit_round_trip_cost_bps: float | None = None,
) -> BaseTeacher1m:
    if exit_variant not in ("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"):
        raise ValueError(f"unknown Route A 1m exit variant: {exit_variant}")
    if reversal_mode not in ("EXIT_ONLY_V1", "REVERSE_AFTER_FLAT_V1"):
        raise ValueError(f"unknown EMA reversal mode: {reversal_mode}")
    if roc_round_trip_cost_bps is not None and (
        isinstance(roc_round_trip_cost_bps, bool)
        or not isinstance(roc_round_trip_cost_bps, (int, float))
        or not isfinite(roc_round_trip_cost_bps)
        or roc_round_trip_cost_bps <= 0
    ):
        raise ValueError("ROC round-trip cost must be positive and finite")
    if exit_round_trip_cost_bps is not None and (
        isinstance(exit_round_trip_cost_bps, bool)
        or not isinstance(exit_round_trip_cost_bps, (int, float))
        or not isfinite(exit_round_trip_cost_bps)
        or exit_round_trip_cost_bps <= 0
    ):
        raise ValueError("exit round-trip cost must be positive and finite")
    if exit_variant in ("X3", "X4", "X5", "X8") and exit_round_trip_cost_bps is None:
        raise ValueError(f"{exit_variant} profit protection requires the scenario round-trip cost")
    if exit_variant not in ("X3", "X4", "X5", "X8") and exit_round_trip_cost_bps is not None:
        raise ValueError("exit round-trip cost only applies to X3/X4/X5/X8 exit policies")
    if exit_variant in ("X4", "X5", "X6", "X8") and teacher_id != ROC_COST_HYSTERESIS_TEACHER_ID:
        raise ValueError("X4/X5/X6 cost-exit rearm only applies to the ROC hysteresis teacher")
    if exit_variant == "X7" and teacher_id != ROC_COST_HYSTERESIS_TEACHER_ID:
        raise ValueError("X7 EMA cross-age gate only applies to the trend ROC hysteresis teacher")
    factory: type[BaseTeacher1m]
    if teacher_id in (
        ROC_COST_HYSTERESIS_TEACHER_ID,
        ROC_COST_HYSTERESIS_FADE_TEACHER_ID,
    ):
        fade_mode = teacher_id == ROC_COST_HYSTERESIS_FADE_TEACHER_ID
        entry_rule_id = (
            "ROC_COST_AWARE_CONTRARIAN_CONSENSUS_HYSTERESIS_V1"
            if fade_mode
            else "ROC_COST_AWARE_CONSENSUS_HYSTERESIS_V1"
        )
        recipe = _recipe(
            teacher_id=teacher_id,
            indicator_recipe_id="route_a.roc_20_60.v1",
            parameters=(
                ("fast", 20), ("slow", 60),
                ("event", "COST_AWARE_CONTRARIAN_HYSTERESIS_V1" if fade_mode
                 else "COST_AWARE_HYSTERESIS_V1"),
                ("round_trip_cost_bps", 12.0),
                ("entry_confirmation_bars", 2),
                ("rearm_neutral_bars", 2),
                ("entry_atr_fraction", 0.25),
                ("entry_volatility_fraction", 0.5),
                ("exit_cost_fraction", 0.25),
                ("exit_atr_fraction", 0.25),
                ("exit_volatility_fraction", 0.5),
                ("exit_confirmation_bars", 2),
            ),
            warmup_bars=61,
            entry_rule_id=entry_rule_id,
            indicator_exit_rule_id="X1_ROC_COST_HYSTERESIS_EXIT_V1",
        )
        factory = RocCostHysteresisTeacher
        if roc_round_trip_cost_bps is not None:
            parameters = tuple(
                (name, value) for name, value in recipe.parameters
                if name != "round_trip_cost_bps"
            )
            recipe = replace(
                recipe,
                parameters=(*parameters, ("round_trip_cost_bps", float(
                    roc_round_trip_cost_bps,
                ))),
            )
        if fade_mode:
            recipe = replace(
                recipe,
                parameters=(*recipe.parameters, ("direction_mode", "CONSENSUS_FADE_V1")),
            )
        if exit_variant == "X4":
            recipe = replace(
                recipe,
                parameters=(*recipe.parameters, (
                    "profit_lock_rearm_mode", "OPPOSITE_CONSENSUS_V1",
                )),
                entry_rule_id=(
                    "ROC_COST_AWARE_CONSENSUS_HYSTERESIS_OPPOSITE_REARM_V1"
                ),
            )
        if exit_variant == "X5":
            recipe = replace(
                recipe,
                parameters=(*recipe.parameters, (
                    "cost_exit_rearm_mode", "OPPOSITE_CONSENSUS_V1",
                )),
                entry_rule_id=(
                    "ROC_COST_AWARE_CONSENSUS_HYSTERESIS_COST_EXIT_REARM_V1"
                ),
            )
        if exit_variant == "X6":
            recipe = replace(
                recipe,
                parameters=(*recipe.parameters, (
                    "cost_exit_rearm_mode", "ATR_EXIT_OPPOSITE_CONSENSUS_V1",
                )),
                entry_rule_id=(
                    "ROC_COST_AWARE_CONSENSUS_HYSTERESIS_ATR_EXIT_REARM_V1"
                ),
                indicator_exit_rule_id=(
                    "X6_ATR_TRAIL_OPPOSITE_CONSENSUS_REARM_V1"
                ),
            )
        if exit_variant == "X8":
            policy_id = "X8_ATR_TRAIL_HANDOFF_AT_4X_COST_V1"
            recipe = replace(
                recipe,
                parameters=(*recipe.parameters, (
                    "cost_exit_rearm_mode", "ATR_EXIT_OPPOSITE_CONSENSUS_V1",
                )),
                entry_rule_id=(
                    "ROC_COST_AWARE_CONSENSUS_ATR_HANDOFF_REARM_V1"
                ),
                indicator_exit_rule_id=policy_id,
            )
        if exit_variant == "X7":
            recipe = replace(
                recipe,
                parameters=(
                    *recipe.parameters,
                    ("entry_ema_cross_age_min_bars", 21),
                    ("entry_ema_cross_age_max_bars", 60),
                ),
                entry_rule_id="ROC_COST_AWARE_EMA_CROSS_AGE_21_60_V1",
                indicator_exit_rule_id="X1_ROC_COST_HYSTERESIS_EXIT_V1",
            )
    else:
        if roc_round_trip_cost_bps is not None:
            raise ValueError("ROC round-trip cost only applies to the hysteresis candidate")
        try:
            recipe = ALL_TEACHER_RECIPE_BY_ID[teacher_id]
            factory = _FACTORIES[teacher_id]
        except KeyError as exc:
            raise ValueError(f"unknown Route A 1m teacher: {teacher_id}") from exc
    if teacher_id != "T01_EMA_9_21" and reversal_mode != "EXIT_ONLY_V1":
        raise ValueError("EMA reversal mode only applies to T01_EMA_9_21")
    if reversal_mode == "REVERSE_AFTER_FLAT_V1" and exit_variant != "X1":
        raise ValueError("EMA reversal mode is only defined for X1 exits")
    if (teacher_id == "T01_EMA_9_21"
            and reversal_mode == "REVERSE_AFTER_FLAT_V1"):
        recipe = replace(
            recipe,
            parameters=(*recipe.parameters, ("reversal_mode", reversal_mode)),
            entry_rule_id="EMA_CROSS_REVERSE_AFTER_FLAT_V1",
        )
    if exit_variant == "X2":
        recipe = replace(recipe, indicator_exit_rule_id="X2_ATR_TRAIL_3_V1")
        return factory(recipe, atr_exit=AtrTrailingExit(3.0))
    if exit_variant == "X6":
        policy_id = "X6_ATR_TRAIL_OPPOSITE_CONSENSUS_REARM_V1"
        return factory(
            recipe,
            atr_exit=AtrTrailingExit(3.0, policy_id=policy_id),
        )
    if exit_variant == "X8":
        assert exit_round_trip_cost_bps is not None
        policy_id = "X8_ATR_TRAIL_HANDOFF_AT_4X_COST_V1"
        return factory(
            recipe,
            atr_exit=AtrTrailHandoffToIndicatorExit(
                atr_multiplier=3.0,
                round_trip_cost_bps=float(exit_round_trip_cost_bps),
                handoff_cost_multiple=4.0,
                policy_id=policy_id,
            ),
        )
    if exit_variant in ("X3", "X4", "X5"):
        assert exit_round_trip_cost_bps is not None
        policy_id_by_variant = {
            "X3": "X3_ATR_COST_PROFIT_PROTECT_V1",
            "X4": "X4_ATR_COST_PROFIT_LOCK_OPPOSITE_REARM_V1",
            "X5": "X5_ATR_COST_PROFIT_AND_LOSS_GUARD_V1",
        }
        policy_id = policy_id_by_variant[exit_variant]
        recipe = replace(recipe, indicator_exit_rule_id=policy_id)
        if exit_variant == "X5":
            return factory(
                recipe,
                atr_exit=CostAwareAtrProfitAndLossGuardExit(
                    atr_multiplier=3.0,
                    round_trip_cost_bps=float(exit_round_trip_cost_bps),
                    policy_id=policy_id,
                ),
            )
        return factory(
            recipe,
            atr_exit=CostAwareAtrProfitProtectExit(
                atr_multiplier=3.0,
                round_trip_cost_bps=float(exit_round_trip_cost_bps),
                policy_id=policy_id,
            ),
        )
    return factory(recipe)

