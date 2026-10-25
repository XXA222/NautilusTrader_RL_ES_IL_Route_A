from __future__ import annotations

from hashlib import sha256
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from rles_data.core import MINUTE_NS, parse_ns
from rles_route_a.teacher_research import OpportunityEvent

_RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "qa/run_route_a_teacher_e1_e2_g1_20261025.py"
)
_SPEC = importlib.util.spec_from_file_location("route_a_teacher_research_runner", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)
_replay_factory = _RUNNER._replay_factory
_fit_block_coverage = _RUNNER._fit_block_coverage


class _GuardedValues:
    def __init__(self, values: dict[int, float]):
        self.values = values
        self.read_indices: list[int] = []

    def __getitem__(self, index: int) -> float:
        self.read_indices.append(index)
        if index not in self.values:
            raise AssertionError(f"market value outside the in-block allowlist: {index}")
        return self.values[index]


def _event() -> OpportunityEvent:
    digest = lambda value: sha256(value.encode()).hexdigest()
    return OpportunityEvent(
        event_id=digest("event"),
        decision_ts_ns=11 * MINUTE_NS,
        feature_available_at_ns=11 * MINUTE_NS,
        side=1,
        family="breakout",
        fold_id="train-inner",
        source_identity="registered-train",
        rule_version="route_a_candidates_v2",
        snapshot_id=digest("snapshot"),
        feature_spec_hash=digest("feature-spec"),
    )


def _costs() -> dict:
    return {
        "base": {"taker_fee_bps_per_leg": 5.0, "slippage_bps_per_leg": 1.0},
    }


def test_cross_block_exit_is_censored_before_exit_price_or_funding_read():
    event = _event()
    # Bars are available at 11m, 12m, 13m, 14m. The first post-decision open
    # is at 12m, but the 240m label ends well beyond the 13m block boundary.
    bar_times = [11 * MINUTE_NS, 12 * MINUTE_NS, 13 * MINUTE_NS, 14 * MINUTE_NS]
    opens = _GuardedValues({2: 101.0})
    closes = _GuardedValues({0: 100.0})
    funding = _GuardedValues({})
    replay = _replay_factory(
        bars=[object()] * len(bar_times),
        bar_times=bar_times,
        opens=opens,
        closes=closes,
        funding=funding,
        funding_times=[13 * MINUTE_NS],
        scenario="base",
        costs=_costs(),
        block_ends={event.event_id: 13 * MINUTE_NS},
    )(event, "fixed-240m-label-only-v1", "market-like-v1", "base")

    assert replay.entry_ts_ns == 12 * MINUTE_NS
    assert replay.exit_ts_ns is None
    assert replay.censor_ts_ns == 13 * MINUTE_NS
    assert opens.read_indices == [2]
    assert closes.read_indices == []
    assert funding.read_indices == []


def test_post_boundary_entry_is_censored_without_reading_its_open():
    event = _event()
    bar_times = [11 * MINUTE_NS, 12 * MINUTE_NS, 13 * MINUTE_NS]
    opens = _GuardedValues({})
    closes = _GuardedValues({0: 100.0})
    funding = _GuardedValues({})
    replay = _replay_factory(
        bars=[object()] * len(bar_times),
        bar_times=bar_times,
        opens=opens,
        closes=closes,
        funding=funding,
        funding_times=[12 * MINUTE_NS],
        scenario="base",
        costs=_costs(),
        block_ends={event.event_id: 12 * MINUTE_NS},
    )(event, "fixed-240m-label-only-v1", "market-like-v1", "base")

    assert replay.entry_ts_ns == 12 * MINUTE_NS
    assert replay.exit_ts_ns is None
    assert replay.censor_ts_ns == 12 * MINUTE_NS + 1
    assert opens.read_indices == []
    assert closes.read_indices == [0]
    assert funding.read_indices == []


def test_fit_block_coverage_exposes_configured_block_without_purged_sample():
    prereg = {
        "windows": {
            "train_fit": ["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"],
            "train_fit_blocks": [
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
                ["2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
            ],
            "train_calibration": ["2026-01-03T00:00:00Z", "2026-01-04T00:00:00Z"],
            "train_calibration_blocks": [[
                "2026-01-03T00:00:00Z", "2026-01-04T00:00:00Z",
            ]],
            "train_inner_screen": ["2026-01-04T00:00:00Z", "2026-01-05T00:00:00Z"],
            "screen_blocks": [["2026-01-04T00:00:00Z", "2026-01-05T00:00:00Z"]],
        },
    }
    first_block = "train_fit-block-01"
    second_block = "train_fit-block-02"
    outcomes = (
        SimpleNamespace(
            independent_block_id=first_block,
            net_return_bps=3.0,
            label_end_ts_ns=parse_ns("2026-01-01T12:00:00Z"),
        ),
        SimpleNamespace(
            independent_block_id=second_block,
            net_return_bps=None,
            label_end_ts_ns=None,
        ),
    )
    result = _fit_block_coverage(
        prereg, outcomes,
        calibration_start_ns=parse_ns("2026-01-03T00:00:00Z"),
    )

    assert result["configured_block_count"] == 2
    assert result["effective_block_count"] == 1
    assert result["all_preregistered_blocks_effective"] is False
    assert result["by_block"][first_block]["purged_complete"] == 1
    assert result["by_block"][second_block]["censored"] == 1
