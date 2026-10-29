import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rles_execution.spec import ResolvedExecutionConfig
from rles_route_a.teacher_research.execution_binding import validate_native_execution


def test_native_resolved_costs_and_window_independent_recipe():
    root = Path(__file__).resolve().parents[3]
    doc = json.loads((root / "configs/ethusdt_20240101_real.json").read_text())
    execution = ResolvedExecutionConfig.from_dict(doc["environment"]["execution"])
    factory = SimpleNamespace(mode="base", execution=execution,
        route_a_execution_recipe_hash="recipe-identity",
        rule=SimpleNamespace(payload={"taker_fee": "0.0005"}))
    validate_native_execution(factory, scenario="base", expected_recipe="recipe-identity",
                              fee_bps=5., slippage_bps=1.)
    factory.mode = "stress"
    validate_native_execution(factory, scenario="stress", expected_recipe="recipe-identity",
                              fee_bps=10., slippage_bps=5.)
    with pytest.raises(ValueError, match="per-leg"):
        validate_native_execution(factory, scenario="stress", expected_recipe="recipe-identity",
                                  fee_bps=9., slippage_bps=6.)
    with pytest.raises(ValueError, match="recipe"):
        validate_native_execution(factory, scenario="stress", expected_recipe="other",
                                  fee_bps=10., slippage_bps=5.)
