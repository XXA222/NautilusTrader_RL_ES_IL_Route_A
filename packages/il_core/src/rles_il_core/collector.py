"""Run a frozen teacher inside the existing native Gym environment."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from math import isclose, log
from typing import Any

import numpy as np

from .dataset import EpisodeTrajectory, Transition
from .experts import ExpertPolicy, FrozenVisibleContext


def _visible_teacher_context(env: Any, *, decision_ts_ns: int,
                             fallback_market: Mapping[str, float]) -> FrozenVisibleContext:
    reader = getattr(env, "rles_visible_context_inputs", None)
    if not callable(reader):
        return FrozenVisibleContext(decision_ts_ns, fallback_market)
    raw = reader()
    if (not isinstance(raw, Mapping) or raw.get("decision_ts_ns") != decision_ts_ns
            or not isinstance(raw.get("market"), Mapping)
            or not isinstance(raw.get("account"), Mapping)):
        raise ValueError("native visible context does not match the current teacher barrier")
    market: dict[str, float] = {}
    for name, feature in raw["market"].items():
        value = getattr(feature, "value", feature)
        available_at = getattr(feature, "available_at_ns", decision_ts_ns)
        if type(available_at) is not int or available_at > decision_ts_ns:
            raise ValueError("teacher feature was not available at the current native barrier")
        market[name] = float(value)
    account = {str(name): float(value) for name, value in raw["account"].items()}
    for group in ("risk", "execution", "time"):
        facts = raw.get(group, {})
        if not isinstance(facts, Mapping):
            raise ValueError("native visible context group is malformed")
        account.update({f"{group}.{name}": float(value) for name, value in facts.items()})
    pending = raw.get("pending_order_ids")
    if not isinstance(pending, tuple):
        raise ValueError("native visible context lacks pending order identities")
    return FrozenVisibleContext(decision_ts_ns, market or fallback_market, account, pending)


def collect_episode(env: Any, expert: ExpertPolicy, *, seed: int,
                    options: dict[str, Any] | None = None,
                    execution_recipe_hash: str | None = None) -> EpisodeTrajectory:
    """Collect one complete native account episode without relabeling or filtering."""
    if type(seed) is not int or seed < 0:
        raise ValueError("non-negative deterministic episode seed required")
    observation, reset_info = env.reset(seed=seed, options=options)
    if not isinstance(reset_info, dict):
        raise ValueError("environment reset metadata missing")
    if reset_info.get("split") != "train":
        raise ValueError("offline collector accepts Train-inner episodes only")
    expert.reset(dict(reset_info))
    expert_id = expert.expert_id
    observations = [tuple(float(x) for x in observation)]
    transitions: list[Transition] = []
    current_ts = int(reset_info["decision_ts_ns"])
    identity = dict(reset_info["native_identity"])
    identity.update({
        "transition_hash": str(reset_info["transition_hash"]),
        "schema_hash": str(reset_info["observation_schema_hash"]),
    })
    window = dict(reset_info["episode_window"])
    market = dict(reset_info["market_features"])
    metadata = {
        "seed": seed, "split": "train_inner",
        "initial_equity": str(reset_info["initial_equity"]),
        "expert_id": expert_id,
        "expert_parameters": (asdict(expert.config)  # type: ignore[arg-type]
                              if hasattr(expert, "config")
                              and is_dataclass(expert.config)
                              else {}),
        "native_identity": identity,
        "episode_window": window,
    }
    if execution_recipe_hash is not None:
        if (len(execution_recipe_hash) != 64
                or any(c not in "0123456789abcdef" for c in execution_recipe_hash)):
            raise ValueError("invalid frozen execution recipe hash")
        metadata["execution_recipe_hash"] = execution_recipe_hash
    while True:
        context = _visible_teacher_context(
            env, decision_ts_ns=current_ts, fallback_market=market,
        )
        action = float(expert.act(observation=np.asarray(observation), context=context))
        if not np.isfinite(action) or not -1 <= action <= 1:
            raise ValueError("expert action must be finite target exposure in [-1,1]")
        next_observation, reward, terminated, truncated, info = env.step(
            np.asarray([action], dtype=np.float32),
        )
        if not isinstance(info, dict):
            raise ValueError("Gym step facts missing")
        if ("native_equity_before" not in info or "native_equity" not in info
                or "native_fills" not in info or "native_funding" not in info):
            raise ValueError("native account and ledger transition facts are required")
        end_ts = int(info["decision_ts_ns"])
        if (int(info["native_decision_ts_before_ns"]) != current_ts
                or int(info["native_decision_ts_after_ns"]) != end_ts or end_ts <= current_ts):
            raise ValueError("native account clocks do not align with the policy decision barrier")
        if not isclose(float(reward), float(info["r_train"]), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Gym return reward differs from recorded r_train")
        equity_before, equity_after = float(info["native_equity_before"]), float(info["native_equity"])
        if equity_before > 0 and equity_after > 0:
            if not isclose(float(info["r_econ"]), log(equity_after / equity_before),
                           rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("economic reward differs from native equity change")
        terminal = bool(terminated or truncated)
        settlement = info.get("terminal_settlement") if terminal else None
        next_ts = (int(settlement["settlement_barrier_ts_ns"])
                   if settlement is not None else end_ts)
        execution = {
            key: info.get(key) for key in (
                "requested_action", "risk_soft_adjustment", "risk_hard_adjustment",
                "risk_guard_intervention", "risk_intervention_components_known",
                "risk_soft_target", "risk_adjusted_target", "quantity_rounded_target_exposure",
                "execution_plan_action", "execution_plan_reason", "execution_decision_id",
                "execution_plan_id", "execution_idempotency_key", "execution_delta_qty",
                "execution_cancel_order_ids", "execution_related_order_ids",
                "execution_client_order_id", "executed_action", "policy_executed_action_pre_terminal",
                "native_net_exposure", "native_exposure_ts_ns", "drawdown", "margin_facts_known",
                "maintenance_margin_breach", "native_liquidation_detected", "insolvency",
            )
        }
        transitions.append(Transition(
            observation=observations[-1], action=action,
            next_observation=tuple(float(x) for x in next_observation),
            r_econ=float(info["r_econ"]), r_shape=float(info["r_shape"]),
            r_train=float(info["r_train"]), terminated=bool(terminated),
            truncated=bool(truncated), decision_ts_ns=current_ts,
            next_decision_ts_ns=next_ts,
            native_equity_before=str(info["native_equity_before"]),
            native_equity_after=str(info["native_equity"]), expert_id=expert_id,
            native_identity=identity, episode_window=window, execution=execution,
            native_fills=tuple(info["native_fills"]), native_funding=tuple(info["native_funding"]),
            terminal_settlement=settlement,
        ))
        observations.append(tuple(float(x) for x in next_observation))
        if terminal:
            break
        observation = next_observation
        current_ts = end_ts
        market = dict(info["market_features"])
    return EpisodeTrajectory(expert_id, tuple(observations), tuple(transitions), metadata)


def collect_episodes(env_factory: Any, expert_factory: Any, *, seeds: tuple[int, ...],
                     episode_options: tuple[dict[str, Any] | None, ...] | None = None
                     ) -> tuple[EpisodeTrajectory, ...]:
    """Serial native collection keeps account/session ownership explicit."""
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("one or more non-negative episode seeds required")
    if getattr(env_factory, "split", None) != "train_inner":
        raise ValueError("native collection is restricted to the Train-inner source factory")
    options = episode_options or tuple(None for _ in seeds)
    if len(options) != len(seeds):
        raise ValueError("episode options and seeds differ in length")
    output = []
    for index, seed in enumerate(seeds):
        env = env_factory.make_env(env_factory.transition_hash)
        expert = expert_factory()
        try:
            output.append(collect_episode(
                env, expert, seed=seed, options=options[index],
                execution_recipe_hash=getattr(env_factory, "route_a_execution_recipe_hash", None),
            ))
        finally:
            env.close()
    return tuple(output)
