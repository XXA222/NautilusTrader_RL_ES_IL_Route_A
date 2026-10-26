"""Frozen Route A one-minute indicator recipes."""
from __future__ import annotations

from .contracts import IndicatorRecipe


AVAILABILITY = (
    "advance exactly once per closed 1m bar; expose only when "
    "bar.available_at_ns <= decision_ts_ns"
)


EMA_9_21 = IndicatorRecipe(
    "route_a.ema_9_21.v1",
    "EMA9/EMA21",
    (("fast", 9), ("slow", 21), ("alpha", "2/(n+1)")),
    21,
    ("close",),
    ("ema_9", "ema_21"),
    "first EMA value is the SMA of the first n closes; subsequent values are recursive",
    AVAILABILITY,
)

ROC_20_60 = IndicatorRecipe(
    "route_a.roc_20_60.v1",
    "ROC20/ROC60",
    (("fast", 20), ("slow", 60)),
    61,
    ("close",),
    ("roc_20", "roc_60"),
    "close_t/close_(t-n)-1 stored as a decimal ratio, never percent",
    AVAILABILITY,
)

DONCHIAN_20_10 = IndicatorRecipe(
    "route_a.donchian_20_10.v1",
    "Donchian20/10",
    (("entry_window", 20), ("exit_window", 10)),
    21,
    ("high", "low"),
    (
        "donchian_upper_20",
        "donchian_lower_20",
        "donchian_upper_10",
        "donchian_lower_10",
    ),
    "thresholds use prior bars only; the current bar is excluded",
    AVAILABILITY,
)

ZSCORE_60 = IndicatorRecipe(
    "route_a.zscore_60.v1",
    "Zscore60",
    (("lookback", 60), ("ddof", 0)),
    60,
    ("close",),
    ("zscore_60",),
    "population standard deviation (ddof=0); zero variance remains NOT_READY",
    AVAILABILITY,
)

ATR_14 = IndicatorRecipe(
    "route_a.atr_14.v1",
    "ATR14",
    (("lookback", 14), ("smoothing", "Wilder")),
    14,
    ("high", "low", "close"),
    ("atr_14",),
    "TR=max(high-low,abs(high-prev_close),abs(low-prev_close)); first TR uses high-low",
    AVAILABILITY,
)

MACD_12_26_9 = IndicatorRecipe(
    "route_a.macd_12_26_9.v1",
    "MACD12/26/9",
    (("fast", 12), ("slow", 26), ("signal", 9)),
    34,
    ("close",),
    ("macd_dif", "macd_dea", "macd_hist"),
    "EMA SMA initialization; DEA is EMA9 of mature DIF; HIST=DIF-DEA without x2",
    AVAILABILITY,
)

RSI_14 = IndicatorRecipe(
    "route_a.rsi_14.v1",
    "RSI14",
    (("lookback", 14), ("smoothing", "Wilder")),
    15,
    ("close",),
    ("rsi_14",),
    "first 14 close changes averaged, then Wilder recursion; flat=50 and one-sided=0/100",
    AVAILABILITY,
)

BOLLINGER_20_2 = IndicatorRecipe(
    "route_a.bollinger_20_2.v1",
    "Bollinger20/2",
    (("lookback", 20), ("std_multiplier", 2.0), ("ddof", 0)),
    20,
    ("close",),
    ("bollinger_mid_20", "bollinger_upper_20_2", "bollinger_lower_20_2"),
    "SMA20 plus/minus two population standard deviations (ddof=0)",
    AVAILABILITY,
)

ROUTE_A_1M_INDICATOR_RECIPES_V1 = (
    EMA_9_21,
    ROC_20_60,
    DONCHIAN_20_10,
    ZSCORE_60,
    ATR_14,
    MACD_12_26_9,
    RSI_14,
    BOLLINGER_20_2,
)

ROUTE_A_1M_INDICATOR_RECIPE_BY_ID = {
    recipe.recipe_id: recipe for recipe in ROUTE_A_1M_INDICATOR_RECIPES_V1
}

if len(ROUTE_A_1M_INDICATOR_RECIPE_BY_ID) != len(ROUTE_A_1M_INDICATOR_RECIPES_V1):
    raise RuntimeError("duplicate Route A 1m indicator recipe id")

