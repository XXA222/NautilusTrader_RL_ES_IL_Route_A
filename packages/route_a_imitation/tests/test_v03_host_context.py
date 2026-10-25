from __future__ import annotations

from types import SimpleNamespace

import pytest

from rles_route_a.integration.l2_adapter import (
    CAUSAL_MTF_V7_ROUTE_A_FEATURES,
    project_host_visible_context,
)


def _feature(value: float, available_at_ns: int):
    return SimpleNamespace(value=value, available_at_ns=available_at_ns, unit="ratio")


def _context(decision_ts_ns: int = 120_000_000_000):
    return {
        "decision_ts_ns": decision_ts_ns,
        "source_id": "registered-train-source",
        "snapshot_id": "a" * 64,
        "feature_spec_hash": "b" * 64,
        "market": {
            "closed_return_1h": _feature(0.01, decision_ts_ns - 1),
            "closed_return_1d": _feature(0.03, decision_ts_ns - 2),
            "volatility_1m": _feature(0.002, decision_ts_ns - 3),
            "zscore_1m": _feature(1.5, decision_ts_ns - 4),
            "trend_efficiency_15m": _feature(0.4, decision_ts_ns - 5),
            "trend_efficiency_60m": _feature(0.7, decision_ts_ns - 6),
        },
        "account": {
            "net_exposure": 0.2,
            "drawdown": 0.01,
            "unrealized_pnl_pct": 0.004,
            "native_last_fill_age_seconds": -1.0,
        },
        "risk": {"headroom": 0.5},
        "execution": {"pending_order_count": 1.0},
        "time": {"funding_settlement_within_24h": 1.0},
        "pending_order_ids": ("open-order-1",),
    }


def test_host_context_maps_closed_mtf_and_native_account_facts():
    observation = project_host_visible_context(
        context=_context(), fold_id="fold-train", episode_id="episode-1",
    )

    assert observation.market == {
        "trend_1h": 0.01,
        "trend_1d": 0.03,
        "vol_1m": 0.002,
        "zscore_1m": 1.5,
        "efficiency_15m": 0.4,
        "efficiency_1h": 0.7,
    }
    assert observation.feature_available_at_ns["trend_1d"] == observation.decision_ts_ns - 2
    assert observation.account_value("unrealized_pnl_fraction") == 0.004
    assert observation.account_value("execution.pending_order_count") == 1.0
    assert observation.pending_order_ids == ("open-order-1",)
    assert observation.source_id == "registered-train-source"
    assert observation.snapshot_id == "a" * 64
    assert observation.feature_spec_hash == "b" * 64


def test_host_context_rejects_future_or_missing_daily_features():
    context = _context()
    context["market"]["closed_return_1d"] = _feature(0.03, context["decision_ts_ns"] + 1)
    with pytest.raises(ValueError, match="future or invalid market feature"):
        project_host_visible_context(context=context, fold_id="fold", episode_id="episode")

    context = _context()
    del context["market"]["closed_return_1d"]
    with pytest.raises(ValueError, match="host causal feature is missing"):
        project_host_visible_context(context=context, fold_id="fold", episode_id="episode")


def test_host_context_requires_native_pending_order_identities():
    context = _context()
    del context["pending_order_ids"]
    with pytest.raises(ValueError, match="pending order identities"):
        project_host_visible_context(context=context, fold_id="fold", episode_id="episode")


def test_v7_host_projection_feeds_the_causal_candidate_generator():
    from rles_route_a.teacher_research import CausalOpportunityGenerator

    context = _context()
    context["market"]["breakout_1h"] = _feature(.02, context["decision_ts_ns"] - 7)
    for host_name in CAUSAL_MTF_V7_ROUTE_A_FEATURES.values():
        context["market"].setdefault(
            host_name, _feature(.001, context["decision_ts_ns"] - 8)
        )
    observation = project_host_visible_context(
        context=context,
        fold_id="fold-train",
        episode_id="episode-v7",
        market_feature_map=CAUSAL_MTF_V7_ROUTE_A_FEATURES,
    )
    candidates = CausalOpportunityGenerator().candidates(observation)
    breakout, = candidates
    assert breakout.family == "breakout"
    assert breakout.source_identity == "registered-train-source"
    assert breakout.snapshot_id == "a" * 64
    assert breakout.feature_spec_hash == "b" * 64
    assert breakout.feature_available_at_ns == max(
        context["market"]["breakout_1h"].available_at_ns,
        context["market"]["closed_return_1d"].available_at_ns,
    )


def test_candidate_snapshot_fingerprint_binds_market_clock_and_native_state():
    from rles_route_a.integration.gym_rollout import _visible_snapshot_id

    context = _context()
    original = _visible_snapshot_id(context)
    assert original == _visible_snapshot_id(context)

    changed_market = _context()
    changed_market["market"]["closed_return_1h"] = _feature(.02, 119_999_999_999)
    assert _visible_snapshot_id(changed_market) != original

    changed_account = _context()
    changed_account["account"]["net_exposure"] = -.2
    assert _visible_snapshot_id(changed_account) != original
