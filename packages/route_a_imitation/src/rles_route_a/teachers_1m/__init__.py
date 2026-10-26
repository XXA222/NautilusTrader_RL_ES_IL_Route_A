"""Independent Route A one-minute teachers."""

from .base import BaseTeacher1m
from .contracts import TeacherRecipe1m, TeacherSignal
from .donchian import DonchianBreakoutTeacher
from .ema import EmaCrossTeacher
from .exits import AtrTrailingExit
from .momentum import RocConsensusTeacher
from .registry import FIRST_BATCH_RECIPES, FIRST_BATCH_RECIPE_BY_ID, build_teacher_1m
from .zscore import ZscoreReentryTeacher

__all__ = (
    "BaseTeacher1m",
    "AtrTrailingExit",
    "DonchianBreakoutTeacher",
    "EmaCrossTeacher",
    "FIRST_BATCH_RECIPES",
    "FIRST_BATCH_RECIPE_BY_ID",
    "RocConsensusTeacher",
    "TeacherRecipe1m",
    "TeacherSignal",
    "ZscoreReentryTeacher",
    "build_teacher_1m",
)

