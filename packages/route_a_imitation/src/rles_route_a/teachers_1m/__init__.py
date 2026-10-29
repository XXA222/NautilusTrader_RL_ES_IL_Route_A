"""Independent Route A one-minute teachers."""

from .base import BaseTeacher1m
from .contracts import TeacherRecipe1m, TeacherSignal
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
from .registry import (
    ALL_TEACHER_RECIPE_BY_ID,
    FIRST_BATCH_RECIPES,
    FIRST_BATCH_RECIPE_BY_ID,
    SECOND_BATCH_RECIPES,
    SECOND_BATCH_RECIPE_BY_ID,
    build_teacher_1m,
)
from .zscore import ZscoreReentryTeacher

__all__ = (
    "BaseTeacher1m",
    "AtrTrailingExit",
    "AtrTrailHandoffToIndicatorExit",
    "CostAwareAtrProfitAndLossGuardExit",
    "CostAwareAtrProfitProtectExit",
    "BollingerReentryTeacher",
    "DonchianBreakoutTeacher",
    "EmaCrossTeacher",
    "MacdCrossTeacher",
    "FIRST_BATCH_RECIPES",
    "FIRST_BATCH_RECIPE_BY_ID",
    "SECOND_BATCH_RECIPES",
    "SECOND_BATCH_RECIPE_BY_ID",
    "ALL_TEACHER_RECIPE_BY_ID",
    "RocConsensusTeacher",
    "RsiReentryTeacher",
    "TrendPullbackTeacher",
    "TeacherRecipe1m",
    "TeacherSignal",
    "ZscoreReentryTeacher",
    "build_teacher_1m",
)

