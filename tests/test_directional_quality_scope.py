import pytest

from rles_data.core import DataError
from rles_orchestrator import application


def test_directional_only_quality_cannot_enter_rl_training() -> None:
    with pytest.raises(DataError, match="directional-only data quality cannot be used for RL training"):
        application.train({"environment": {"quality_purpose": "directional"}})
