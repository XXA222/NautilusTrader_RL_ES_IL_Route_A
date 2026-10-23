from types import SimpleNamespace

import numpy as np
import pytest

from rles_route_a.config import BCConfig
from rles_route_a.export.bundle import save_bc_bundle
from rles_route_a.models.bc import BCGRUPolicy
from rles_route_a.integration.native_inference import load_route_a_predictor_factory


def context(ts: int, *, daily_available: int | None = None):
    return {
        "decision_ts_ns": ts,
        "market": {
            "closed_return_1h": SimpleNamespace(value=.01, available_at_ns=ts-1),
            "closed_return_1d": SimpleNamespace(
                value=.02, available_at_ns=ts-2 if daily_available is None else daily_available,
            ),
            "volatility_1m": SimpleNamespace(value=.003, available_at_ns=ts-3),
            "zscore_1m": SimpleNamespace(value=0.2, available_at_ns=ts-4),
            "trend_efficiency_15m": SimpleNamespace(value=.3, available_at_ns=ts-5),
            "trend_efficiency_60m": SimpleNamespace(value=.6, available_at_ns=ts-6),
        },
        "account": {"net_exposure": .2, "unrealized_pnl_pct": .01},
        "risk": {"headroom": .5}, "execution": {"pending_order_count": 0.},
        "time": {"funding_settlement_within_24h": 1.}, "pending_order_ids": (),
    }


class Env:
    def __init__(self):
        self.ts = 60_000_000_000
        self.future_daily = False

    def rles_visible_context_inputs(self):
        return context(self.ts, daily_available=self.ts+1 if self.future_daily else None)


def test_route_a_bundle_predicts_from_each_live_pre_action_host_context(tmp_path):
    config = BCConfig(
        feature_names=("market.trend_1h", "account.net_exposure"),
        context_bars=3, hidden_size=8, layers=1, seed=5,
    )
    model = BCGRUPolicy(config.feature_names, config.hidden_size, config.layers)
    bundle = save_bc_bundle(model, config, tmp_path / "ra-bc")
    _policy, loaded_config, predictor_factory = load_route_a_predictor_factory(
        bundle, fold_id="fold-001",
        episode_id=lambda _env, info: info["episode_id"],
    )
    assert loaded_config == config

    env = Env()
    predict = predictor_factory(env, {"episode_id": "episode-1"})
    first = predict(np.zeros((1, 5), dtype=np.float32))
    env.ts += 60_000_000_000
    second = predict(np.zeros((1, 5), dtype=np.float32))
    assert first.shape == second.shape == (1, 1)
    assert np.isfinite(first).all() and np.isfinite(second).all()

    next_episode_predict = predictor_factory(env, {"episode_id": "episode-2"})
    assert next_episode_predict(np.zeros((1, 5), dtype=np.float32)).shape == (1, 1)


def test_route_a_native_predictor_rejects_noncausal_live_context(tmp_path):
    config = BCConfig(feature_names=("market.trend_1h", "market.trend_1d"), hidden_size=8, layers=1)
    bundle = save_bc_bundle(
        BCGRUPolicy(config.feature_names, config.hidden_size, config.layers),
        config, tmp_path / "ra-bc",
    )
    _, _, predictor_factory = load_route_a_predictor_factory(
        bundle, fold_id="fold-001", episode_id=lambda _env, info: info["episode_id"],
    )
    env = Env()
    env.future_daily = True
    predict = predictor_factory(env, {"episode_id": "episode-1"})
    with pytest.raises(ValueError, match="future or invalid market feature"):
        predict(np.zeros((1, 5), dtype=np.float32))
