"""Run the Route A recurrent actor on the host Gym's visible native context.

The returned predictor factory is designed for the host orchestrator's
``rollout(..., predictor_factory=...)`` hook. It does not inspect the flattened
observation or open market data; each action is built from the current
pre-action L2/L3 context on the active Gym environment.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from rles_route_a.config import BCConfig
from rles_route_a.export.bundle import load_bc_bundle
from rles_route_a.export.inference import BCInferencePolicy
from .l2_adapter import CAUSAL_MTF_V6_ROUTE_A_FEATURES, project_host_visible_context


def route_a_predictor_factory(
        policy: BCInferencePolicy, *, fold_id: str,
        episode_id: Callable[[Any, Mapping[str, Any]], str],
        visible_context: Callable[[Any], Mapping[str, Any]] | None = None,
        market_feature_map: Mapping[str, str] | None = None,
) -> Callable[[Any, Mapping[str, Any]], Callable[[np.ndarray], np.ndarray]]:
    """Bind one recurrent policy instance to a host rollout episode.

    ``episode_id`` must resolve a stable identity from the host environment and
    its reset receipt. The predictor then resets hidden history for each call
    to this factory and reads the live pre-action context on every step.
    """
    if not isinstance(policy, BCInferencePolicy):
        raise TypeError("Route A BC inference policy required")
    if not isinstance(fold_id, str) or not fold_id.strip():
        raise ValueError("registered WFO Fold identity required")
    if not callable(episode_id):
        raise TypeError("host episode identity resolver required")
    context_reader = visible_context or (lambda env: env.rles_visible_context_inputs())
    if not callable(context_reader):
        raise TypeError("host visible-context reader required")
    required_market = {
        name.removeprefix("market.")
        for name in policy.cfg.feature_names
        if name.startswith("market.")
    }
    if not required_market:
        raise ValueError("Route A BC inference requires at least one declared market feature")
    if market_feature_map is None:
        feature_map = {
            name: host_name
            for name, host_name in CAUSAL_MTF_V6_ROUTE_A_FEATURES.items()
            if name in required_market
        }
        missing = required_market - set(feature_map)
        if missing:
            raise ValueError(f"BC model requires unmapped causal market features: {sorted(missing)}")
    else:
        feature_map = dict(market_feature_map)
        missing = required_market - set(feature_map)
        if missing:
            raise ValueError(f"BC model requires unmapped causal market features: {sorted(missing)}")

    def create(env: Any, reset_info: Mapping[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
        if not isinstance(reset_info, Mapping):
            raise TypeError("host Gym reset receipt must be a mapping")
        current_episode = episode_id(env, reset_info)
        if not isinstance(current_episode, str) or not current_episode.strip():
            raise ValueError("host reset receipt has no stable episode identity")
        policy.reset(current_episode)

        def predict(batch: np.ndarray) -> np.ndarray:
            values = np.asarray(batch)
            if values.ndim != 2 or values.shape[0] != 1:
                raise ValueError("Route A recurrent inference requires one host decision at a time")
            observation = project_host_visible_context(
                context=context_reader(env), fold_id=fold_id,
                episode_id=current_episode, market_feature_map=feature_map,
            )
            action = policy.predict_exposure(observation)
            return np.asarray([[action]], dtype=np.float32)

        return predict

    return create


def load_route_a_predictor_factory(
        bundle_path: str | Path, *, fold_id: str,
        episode_id: Callable[[Any, Mapping[str, Any]], str],
        device: str = "cpu",
        visible_context: Callable[[Any], Mapping[str, Any]] | None = None,
        market_feature_map: Mapping[str, str] | None = None,
) -> tuple[BCInferencePolicy, BCConfig,
           Callable[[Any, Mapping[str, Any]], Callable[[np.ndarray], np.ndarray]]]:
    """Load an immutable Route A BC bundle and bind it to host rollout hooks."""
    model, config = load_bc_bundle(bundle_path, device=device)
    policy = BCInferencePolicy(model, config, device=device)
    predictor = route_a_predictor_factory(
        policy, fold_id=fold_id, episode_id=episode_id,
        visible_context=visible_context, market_feature_map=market_feature_map,
    )
    return policy, config, predictor
