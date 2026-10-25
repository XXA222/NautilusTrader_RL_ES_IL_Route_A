import json
from pathlib import Path

import pytest

from rles_research.baselines import BaselinePolicy, BaselineSpec
from qa.train_signal_baselines import _available_memory_from_meminfo, _project_path


@pytest.mark.parametrize("closes,trend", [([100.0] * 21, 0.0),
                                         ([100.0 + i for i in range(21)], 0.4),
                                         ([100.0 - i for i in range(21)], -0.4)])
def test_causal_reversal_is_opposite_trend_with_same_bounded_risk_target(closes, trend):
    momentum = BaselinePolicy(BaselineSpec("B2_simple_momentum", 31, target_long=0.4))
    reversal = BaselinePolicy(BaselineSpec("B5_simple_reversal", 31, target_long=0.4))
    assert momentum.action(closes) == trend
    assert reversal.action(closes) == -trend
    assert momentum.action(closes[:2]) == reversal.action(closes[:2]) == 0


def test_fixed_short_is_a_direction_baseline_and_validates_market_inputs():
    policy = BaselinePolicy(BaselineSpec("B4_always_short", 31, target_long=0.4))
    assert policy.action([100.0]) == -0.4
    with pytest.raises(ValueError, match="historic close"):
        policy.action([float("nan")])


def test_ubuntu_baseline_memory_reader_uses_memavailable(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 8000000 kB\nMemAvailable: 123456 kB\n", encoding="ascii")
    assert _available_memory_from_meminfo(meminfo) == 123456 * 1024


@pytest.mark.parametrize("content", ["MemTotal: 1000 kB\n", "MemAvailable: nope kB\n"])
def test_ubuntu_baseline_memory_reader_fails_closed_on_missing_or_invalid_memavailable(
        tmp_path, content):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(content, encoding="ascii")
    with pytest.raises(OSError, match="MemAvailable"):
        _available_memory_from_meminfo(meminfo)


def test_train_baseline_paths_resolve_within_canonical_project():
    assert _project_path(".artifacts/audit-development-20261016/data/market.parquet").is_absolute()
    with pytest.raises(ValueError, match="Windows absolute paths"):
        _project_path(r"D:\Program Files\NautilusTrader_RL_ES\market.parquet")
    with pytest.raises(ValueError, match="canonical project"):
        _project_path("../../outside.parquet")


def test_frozen_train_baseline_plan_uses_project_relative_paths():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/train_signal_baselines_20261016.json").read_text())
    paths = (plan["bundle"], plan["source"]["market_path"], plan["source"]["rules_path"])
    assert all(not Path(value).is_absolute() for value in paths)
    assert all(not value.startswith("D:\\") for value in paths)
    assert all(_project_path(value).is_relative_to(root) for value in paths)
