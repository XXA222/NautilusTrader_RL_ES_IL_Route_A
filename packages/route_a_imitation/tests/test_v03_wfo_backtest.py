from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rles_route_a.config import BCConfig
from rles_route_a.export.bundle import save_bc_bundle
from rles_route_a.integration.wfo_backtest import (
    make_route_a_wfo_backtest,
    make_route_a_wfo_stress_validation,
)
from rles_route_a.models.bc import BCGRUPolicy


@dataclass(frozen=True)
class Quality:
    status: str = "USABLE"


class Access:
    def __init__(self, bundle_sha: str):
        self.key = "access-key"
        self.identity = {
            "query_id": "test-fold-candidate", "bundle_sha": bundle_sha,
            "split": "test", "start_ns": 120, "end_ns": 180,
            "market": "ETHUSDT-PERP", "mode": "base",
        }
        self.consumed = False

    def consume(self, **scope):
        assert not self.consumed
        assert scope == {key: self.identity[key] for key in (
            "bundle_sha", "split", "start_ns", "end_ns", "market", "mode",
        )}
        self.consumed = True


class ValidationAccess:
    def __init__(self, *, mode="base"):
        self.identity = {
            "protocol_id": "wfo", "query_id": "validation-0-candidate-a",
            "candidate_id": "candidate-a", "fold_id": "60-15-15-0000",
            "source_sha256": "b" * 64, "market": "ETHUSDT-PERP",
            "split": "validation", "start_ns": 120, "end_ns": 180,
            "mode": mode,
        }
        self.bundle_sha256 = None

    @property
    def consumed(self):
        return self.bundle_sha256 is not None

    def consume(self, *, bundle_sha256):
        assert not self.consumed
        assert len(bundle_sha256) == 64
        self.bundle_sha256 = bundle_sha256


class Env:
    def rles_visible_context_inputs(self):
        ts = 120
        feature = lambda value, available: SimpleNamespace(value=value, available_at_ns=available)
        return {
            "decision_ts_ns": ts,
            "market": {"closed_return_1h": feature(.01, ts-1),
                       "closed_return_1d": feature(.02, ts-2),
                       "volatility_1m": feature(.03, ts-3),
                       "zscore_1m": feature(.1, ts-4),
                       "trend_efficiency_15m": feature(.4, ts-5),
                       "trend_efficiency_60m": feature(.6, ts-6)},
            "account": {"net_exposure": 0.}, "risk": {}, "execution": {},
            "time": {}, "pending_order_ids": (),
        }


def artifact(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "candidate"
    root.mkdir()
    cfg = BCConfig(feature_names=("market.trend_1h",), hidden_size=8, layers=1)
    save_bc_bundle(BCGRUPolicy(cfg.feature_names, cfg.hidden_size, cfg.layers), cfg,
                   root / "route_a_bc")
    manifest_sha = hashlib.sha256((root / "route_a_bc" / "manifest.json").read_bytes()).hexdigest()
    payload = {
        "schema": "rles.historical-bundle.v1",
        "actor": {"kind": "route_a_bc", "path": "route_a_bc",
                  "manifest_sha256": manifest_sha},
        "soft_multipliers": {},
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    (root / "bundle.json").write_bytes(raw)
    return root, hashlib.sha256(raw).hexdigest()


def test_wfo_backtest_consumes_scope_before_factory_and_runs_host_predictor(tmp_path):
    root, bundle_sha = artifact(tmp_path)
    access = Access(bundle_sha)
    factory_order = []

    def build_factory(run, _bundle, capability):
        assert capability is access and access.consumed
        factory_order.append("factory")
        return SimpleNamespace(quality=Quality(), mode="base", split="test",
                               start_ns=120, end_ns=180, instrument="ETHUSDT-PERP",
                               execution=SimpleNamespace(zero_target_policy="close"))

    def rollout(_factory, _predict, **kwargs):
        assert access.consumed
        assert kwargs["include_execution_trace"] is True
        predict = kwargs["predictor_factory"](Env(), {"episode_id": "episode-1"})
        action = predict(np.zeros((1, 4), dtype=np.float32))
        assert action.shape == (1, 1)
        return {"metrics": {"net_return": 0.1}, "terminal_flat": True}, np.zeros((1, 1))

    callback = make_route_a_wfo_backtest(
        factory_builder=build_factory, host_rollout=rollout,
        episode_id=lambda _env, info: info["episode_id"],
        market_key=lambda instrument: instrument,
    )
    report = callback({
        "seed": 4, "environment": {"instrument": "ETHUSDT-PERP",
                                     "execution": {"zero_target_policy": "close"}},
        "evaluation": {"split": "test", "start_ns": 120, "end_ns": 180},
    }, root, evaluation_access=access)
    assert factory_order == ["factory"]
    assert report["evaluation_scope"]["query_id"] == "test-fold-candidate"
    assert report["dataset_quality"] == {"status": "USABLE"}


def test_wfo_backtest_rejects_bad_scope_before_factory(tmp_path):
    root, bundle_sha = artifact(tmp_path)
    access = Access(bundle_sha)
    access.identity["end_ns"] = 181
    called = False

    def build_factory(*_args):
        nonlocal called
        called = True
        raise AssertionError("factory must not open before the WFO capability is consumed")

    callback = make_route_a_wfo_backtest(
        factory_builder=build_factory, host_rollout=lambda *_args, **_kwargs: None,
        episode_id=lambda _env, info: info["episode_id"],
        market_key=lambda instrument: instrument,
    )
    with pytest.raises(AssertionError):
        callback({"environment": {"instrument": "ETHUSDT-PERP",
                                   "execution": {"zero_target_policy": "close"}},
                  "evaluation": {"split": "test", "start_ns": 120, "end_ns": 180}},
                 root, evaluation_access=access)
    assert not called


def test_wfo_backtest_requires_close_policy_before_consuming_access(tmp_path):
    root, bundle_sha = artifact(tmp_path)
    access = Access(bundle_sha)
    called = False

    def build_factory(*_args):
        nonlocal called
        called = True
        raise AssertionError("deadband execution must fail before factory construction")

    callback = make_route_a_wfo_backtest(
        factory_builder=build_factory, host_rollout=lambda *_args, **_kwargs: None,
        episode_id=lambda _env, info: info["episode_id"],
        market_key=lambda instrument: instrument,
    )
    with pytest.raises(PermissionError, match="zero_target_policy='close'"):
        callback({
            "environment": {"instrument": "ETHUSDT-PERP", "execution": {
                "zero_target_policy": "deadband",
            }},
            "evaluation": {"split": "test", "start_ns": 120, "end_ns": 180},
        }, root, evaluation_access=access)
    assert not access.consumed
    assert not called


def test_wfo_validation_rollout_consumes_candidate_capability_and_bundle_hash(tmp_path):
    root, _bundle_sha = artifact(tmp_path)
    access = ValidationAccess()

    def build_factory(run, _bundle, capability):
        assert capability is access and access.consumed
        return SimpleNamespace(quality=Quality(), mode="base", split="validation",
                               start_ns=120, end_ns=180, instrument="ETHUSDT-PERP",
                               execution=SimpleNamespace(zero_target_policy="close"))

    def rollout(_factory, _predict, **kwargs):
        predict = kwargs["predictor_factory"](Env(), {"episode_id": "validation-ep"})
        assert predict(np.zeros((1, 4), dtype=np.float32)).shape == (1, 1)
        return {"metrics": {"net_return": .01}}, np.zeros((1, 1))

    callback = make_route_a_wfo_backtest(
        factory_builder=build_factory, host_rollout=rollout,
        episode_id=lambda _env, info: info["episode_id"],
        market_key=lambda instrument: instrument,
    )
    report = callback({
        "seed": 9, "environment": {"instrument": "ETHUSDT-PERP",
                                     "execution": {"zero_target_policy": "close"}},
        "evaluation": {"split": "validation", "start_ns": 120, "end_ns": 180},
        "wfo_candidate_id": "candidate-a", "wfo_fold_id": "60-15-15-0000",
    }, root, validation_access=access)
    assert access.consumed
    assert report["evaluation_scope"]["formal"] is False
    assert report["evaluation_scope"]["query_id"] == "validation-0-candidate-a"


def test_wfo_backtest_validates_actor_before_consuming_capability(tmp_path):
    root, _bundle_sha = artifact(tmp_path)
    bundle_path = root / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["actor"]["manifest_sha256"] = "0" * 64
    bundle_path.write_text(json.dumps(bundle, sort_keys=True))
    access = ValidationAccess()
    callback = make_route_a_wfo_backtest(
        factory_builder=lambda *_args: (_ for _ in ()).throw(
            AssertionError("invalid actor must not open an evaluation factory")),
        host_rollout=lambda *_args, **_kwargs: None,
        episode_id=lambda _env, info: info["episode_id"],
        market_key=lambda instrument: instrument,
    )
    with pytest.raises(ValueError, match="does not bind the immutable"):
        callback({
            "environment": {"instrument": "ETHUSDT-PERP",
                             "execution": {"zero_target_policy": "close"}},
            "evaluation": {"split": "validation", "start_ns": 120, "end_ns": 180},
            "wfo_candidate_id": "candidate-a", "wfo_fold_id": "60-15-15-0000",
        }, root, validation_access=access)
    assert not access.consumed


def test_wfo_stress_validation_consumes_stress_capability_and_exact_window(tmp_path):
    root, _bundle_sha = artifact(tmp_path)
    access = ValidationAccess(mode="stress")

    def build_factory(run, _bundle, capability):
        assert capability is access and access.consumed
        assert run["evaluation"]["mode"] == "stress"
        return SimpleNamespace(quality=Quality(), mode="stress", split="validation",
                               start_ns=120, end_ns=180, instrument="ETHUSDT-PERP",
                               execution=SimpleNamespace(zero_target_policy="close"))

    callback = make_route_a_wfo_stress_validation(
        factory_builder=build_factory,
        host_rollout=lambda *_args, **_kwargs: (
            {"metrics": {"net_return": -0.01}, "terminal_flat": True},
            np.zeros((1, 1)),
        ),
        episode_id=lambda _env, info: info["episode_id"],
        market_key=lambda instrument: instrument,
    )
    coverage = {"maximum_age_ns": {"mark": 1}}
    run = {
        "seed": 9, "environment": {"instrument": "ETHUSDT-PERP",
                                     "execution": {"zero_target_policy": "close"}},
        "evaluation": {
            "split": "validation", "mode": "stress", "start_ns": 120,
            "end_ns": 180, "coverage_policy": coverage,
        },
        "wfo_candidate_id": "candidate-a", "wfo_fold_id": "60-15-15-0000",
    }
    report = callback(
        run, root, start_ns=120, end_ns=180, coverage_policy=coverage,
        validation_access=access,
    )
    assert report["execution_mode"] == "stress"
    assert report["evaluation_scope"]["mode"] == "stress"
