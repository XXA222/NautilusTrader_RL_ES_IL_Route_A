"""Gymnasium single-asset decision-barrier environment; no second account ledger."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from typing import Any, Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import re

from .observation import ObservationBuilder, ObservationSchema
from .reward import RewardCalculator, RewardConfig
from .episode import EpisodeReplay, EpisodeResumeState


class ContractNotReady(RuntimeError):
    """Native, L1, or L2 barrier contract cannot be proven; block evaluation."""


def _native_constraint_facts(snapshot: Any) -> tuple[bool, bool, bool]:
    """Return known margin facts, a maintenance breach, and native liquidation."""
    margins = getattr(snapshot, "margins", None)
    liquidations = getattr(snapshot, "liquidation_orders", None)
    if (getattr(snapshot, "liquidation_capability", False) is not True
            or not isinstance(margins, tuple) or liquidations is None):
        return False, False, False
    try:
        maintenance = Decimal(0)
        for row in margins:
            if not isinstance(row, dict) or "maintenance" not in row:
                return False, False, False
            value = Decimal(str(row["maintenance"]))
            if not value.is_finite() or value < 0:
                raise ContractNotReady("native maintenance margin fact is invalid")
            maintenance += value
        equity = Decimal(str(snapshot.equity))
    except (InvalidOperation, TypeError, ValueError, AttributeError) as exc:
        raise ContractNotReady("native solvency facts cannot be parsed") from exc
    if not equity.is_finite():
        raise ContractNotReady("native equity fact is non-finite")
    return True, maintenance > 0 and equity <= maintenance, bool(liquidations)


def _native_order_status_rows(snapshot: Any) -> tuple[dict[str, str], ...]:
    """Detach the stable M2.2 order identity/status fields from one native snapshot."""
    orders = getattr(snapshot, "all_orders", None)
    if not isinstance(orders, (tuple, list)):
        raise ContractNotReady("native M2.2 all-order snapshot unavailable")
    rows = []
    identities: set[str] = set()
    for order in orders:
        if not isinstance(order, dict):
            raise ContractNotReady("native M2.2 order row is malformed")
        client_order_id = order.get("client_order_id")
        status = order.get("status")
        side = order.get("side")
        quantity = order.get("quantity")
        filled_qty = order.get("filled_qty")
        if (not isinstance(client_order_id, str) or not client_order_id.strip()
                or not isinstance(status, str) or not status.strip()
                or not isinstance(side, str) or not side.strip()
                or not isinstance(quantity, str) or not quantity.strip()
                or not isinstance(filled_qty, str) or not filled_qty.strip()):
            raise ContractNotReady("native M2.2 order identity/status facts are invalid")
        if side not in {"BUY", "SELL"} or client_order_id in identities:
            raise ContractNotReady("native M2.2 order identity/status facts are invalid")
        row: dict[str, str] = {
            "client_order_id": client_order_id, "status": status, "side": side,
            "quantity": quantity, "filled_qty": filled_qty,
        }
        identities.add(row["client_order_id"])
        rows.append(row)
    rows.sort(key=lambda row: row["client_order_id"])
    return tuple(rows)


def _native_order_status_receipt(before: tuple[dict[str, str], ...],
                                 after: tuple[dict[str, str], ...]) -> str:
    payload = json.dumps({"before": before, "after": after}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(payload).hexdigest()


def _native_policy_fills(before: Any, after: Any) -> tuple[dict[str, Any], ...]:
    """Return exact native fill events created in this policy-barrier interval."""
    previous = getattr(before, "fills", None)
    current = getattr(after, "fills", None)
    start_ns = getattr(before, "decision_ts_ns", None)
    end_ns = getattr(after, "decision_ts_ns", None)
    if (not isinstance(previous, (tuple, list)) or not isinstance(current, (tuple, list))
            or type(start_ns) is not int or type(end_ns) is not int or end_ns <= start_ns):
        raise ContractNotReady("native M2.2 fill interval is unavailable")
    prior_ids = {str(row.get("event_id", "")) for row in previous if isinstance(row, dict)}
    result = []
    seen: set[str] = set()
    for fill in current:
        if not isinstance(fill, dict):
            raise ContractNotReady("native M2.2 fill row is malformed")
        identity = fill.get("event_id")
        timestamp = fill.get("ts_event", fill.get("ts_event_ns"))
        if (not isinstance(identity, str) or not identity.strip()
                or type(timestamp) is not int or timestamp < 0):
            raise ContractNotReady("native M2.2 fill identity/clock is invalid")
        if identity in prior_ids or not start_ns < timestamp <= end_ns:
            continue
        if identity in seen:
            raise ContractNotReady("native M2.2 fill event is duplicated")
        seen.add(identity)
        result.append(dict(fill))
    result.sort(key=lambda row: str(row["event_id"]))
    return tuple(result)


def _terminal_settlement_facts(before: Any, after: Any, *, policy_exposure: float) -> dict[str, Any]:
    """Describe native account facts created by the forced episode settlement."""
    before_orders = {str(row["client_order_id"]): row for row in before.all_orders}
    after_orders = {str(row["client_order_id"]): row for row in after.all_orders}
    if len(before_orders) != len(before.all_orders) or len(after_orders) != len(after.all_orders):
        raise ContractNotReady("terminal settlement order identities are not unique")
    changed_orders = [dict(row) for key, row in after_orders.items()
                      if before_orders.get(key) != row]
    before_fill_ids = {str(row["event_id"]) for row in before.fills}
    terminal_fills = [dict(row) for row in after.fills
                      if str(row["event_id"]) not in before_fill_ids]
    before_funding_ids = {str(row["event_id"]) for row in before.funding}
    terminal_funding = [dict(row) for row in after.funding
                        if str(row["event_id"]) not in before_funding_ids]
    before_equity, after_equity = Decimal(str(before.equity)), Decimal(str(after.equity))
    if (not before_equity.is_finite() or not after_equity.is_finite()
            or type(before.decision_ts_ns) is not int or type(after.decision_ts_ns) is not int
            or after.decision_ts_ns < before.decision_ts_ns
            or not np.isfinite(policy_exposure)):
        raise ContractNotReady("terminal settlement account facts are invalid")
    return {
        "policy_barrier_ts_ns": before.decision_ts_ns,
        "settlement_barrier_ts_ns": after.decision_ts_ns,
        "equity_before": str(before_equity),
        "equity_after": str(after_equity),
        "equity_delta": str(after_equity - before_equity),
        "equity_currency": after.currency,
        "pre_terminal_net_exposure": float(policy_exposure),
        "post_terminal_open_position_count": len(after.positions),
        "orders": changed_orders,
        "fills": terminal_fills,
        "funding": terminal_funding,
    }


def _new_native_records(before: tuple[dict, ...], after: tuple[dict, ...]) -> list[dict]:
    """Return detached, replay-stable ledger events first visible after a barrier."""
    known = {str(row.get("event_id")) for row in before}
    records = []
    volatile = {"event_id", "trade_id", "venue_order_id", "causation_id"}
    for row in after:
        if str(row.get("event_id")) in known:
            continue
        stable = {key: value for key, value in row.items() if key not in volatile}
        stable["event_id"] = sha256(json.dumps(
            stable, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")).hexdigest()
        records.append(stable)
    return records


class NautilusBarrierEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, episode_factory: Callable[..., Any], observation_builder: ObservationBuilder,
                 reward_config: RewardConfig, max_steps: int,
                 transition_hash: str | None = None) -> None:
        super().__init__()
        if not callable(episode_factory) or not isinstance(observation_builder, ObservationBuilder):
            raise TypeError("reviewed episode factory and observation builder required")
        if not isinstance(reward_config, RewardConfig) or type(max_steps) is not int or max_steps <= 0:
            raise ValueError("finite reward config and positive max steps required")
        self._factory = episode_factory
        if transition_hash is not None and re.fullmatch(r"[0-9a-f]{64}", transition_hash) is None:
            raise ValueError("frozen transition hash must be SHA-256 lowercase hex")
        self._transition_hash = transition_hash
        self._observation_builder = observation_builder
        self._reward = RewardCalculator(reward_config)
        self._max_steps = max_steps
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                           shape=(len(observation_builder.schema.names),), dtype=np.float32)
        self._episode: Any | None = None
        self._step_count = 0
        self._done = True
        self._last_observation: np.ndarray | None = None

    @property
    def observation_schema(self) -> ObservationSchema:
        """Frozen ordered observation contract for external actor adapters."""
        return self._observation_builder.schema

    def _episode_boundary_value(self, method_name: str) -> str:
        if self._episode is None:
            raise ContractNotReady("L3 boundary proof requested without an active episode")
        method = getattr(self._episode, method_name, None)
        if not callable(method):
            raise ContractNotReady(f"episode lacks required L4 boundary method {method_name}()")
        value = method()
        if type(value) is not str or not value:
            raise ContractNotReady(f"episode returned invalid {method_name}")
        return value

    @property
    def rles_transition_hash(self) -> str:
        """Read the current L1/L2/engine transition fingerprint at each barrier."""
        if self._episode is None and self._transition_hash is not None:
            # SB3 constructs the VecEnv and checks its recipe before first reset.
            return self._transition_hash
        value = self._episode_boundary_value("transition_fingerprint_hash")
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ContractNotReady("transition fingerprint must be SHA-256 lowercase hex")
        if self._transition_hash is not None and value != self._transition_hash:
            raise ContractNotReady("active episode differs from frozen transition recipe")
        return value

    @property
    def rles_boundary_token(self) -> str:
        """Opaque L3 token proving a restorable, settled episode boundary."""
        return self._episode_boundary_value("checkpoint_boundary_token")

    @property
    def rles_boundary_observation(self) -> np.ndarray:
        """Copy of the exact raw observation associated with the current token."""
        if self._episode is None or self._last_observation is None:
            raise ContractNotReady("L3 boundary observation requested without a stable episode")
        if not self.rles_boundary_token:
            raise ContractNotReady("L3 boundary token is empty")
        return self._last_observation.copy()

    def rles_export_replay(self) -> dict[str, Any]:
        """Export the action journal needed to rebuild this native episode at its boundary."""
        if self._episode is None:
            raise ContractNotReady("L3 replay export requested without an active episode")
        export = getattr(self._episode, "export_replay", None)
        if not callable(export):
            raise ContractNotReady("episode lacks required export_replay() checkpoint method")
        replay = export()
        if (type(replay) is not EpisodeReplay
                or replay.transition_hash != self.rles_transition_hash
                or replay.boundary_token != self.rles_boundary_token):
            raise ContractNotReady("episode replay does not match the active native boundary")
        try:
            return replay.to_dict()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ContractNotReady("episode replay is not serializable") from exc

    def rles_visible_context_inputs(self) -> dict[str, Any]:
        """Expose only the raw facts visible at the current policy barrier."""
        if self._episode is None or self._done:
            raise ContractNotReady("visible context requires an active decision barrier")
        reader = getattr(self._episode, "visible_context_inputs", None)
        if not callable(reader):
            raise ContractNotReady("episode lacks reviewed visible-context access")
        context = reader()
        if (not isinstance(context, dict)
                or context.get("decision_ts_ns") != self._episode.previous_decision_ts_ns):
            raise ContractNotReady("visible context clock differs from the native decision barrier")
        return context

    def rles_project_target(self, requested_exposure: float | Decimal) -> dict[str, Any]:
        """Read-only host projection at the current completed native barrier.

        The projection runs the same soft risk, hard risk, and quantity planner
        used by ``step``. It does not set a pending action or advance the native
        session. Non-submit dispositions resolve to the exposure already held
        at this barrier, never to a fictional fill.
        """
        if self._episode is None or self._done:
            raise ContractNotReady("target projection requires an active decision barrier")
        project = getattr(self._episode, "project_action", None)
        if not callable(project):
            raise ContractNotReady("episode lacks read-only native action projection")
        review = project(requested_exposure)
        plan = getattr(review, "plan", None)
        disposition = getattr(plan, "action", None)
        if disposition not in {"SUBMIT", "NOOP", "WAIT", "CANCEL"}:
            raise ContractNotReady("native projection returned an unknown plan disposition")
        if disposition == "SUBMIT":
            target = getattr(review, "rounded_target_exposure", None)
            if target is None:
                raise ContractNotReady("native submit projection omitted rounded target exposure")
        else:
            snapshot = getattr(self._episode, "_snapshot", None)
            exposure_at = getattr(self._episode, "exposure_at", None)
            if snapshot is None or not callable(exposure_at):
                raise ContractNotReady("native current exposure is unavailable for non-submit projection")
            target = exposure_at(snapshot)
        try:
            target_value = Decimal(str(target))
            requested_value = Decimal(str(requested_exposure))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ContractNotReady("native projected exposure is malformed") from exc
        if (not target_value.is_finite() or not -1 <= target_value <= 1
                or not requested_value.is_finite() or not -1 <= requested_value <= 1):
            raise ContractNotReady("native projected exposure is outside normalized bounds")
        return {
            "decision_ts_ns": self._episode.previous_decision_ts_ns,
            "requested_exposure": float(requested_value),
            "target": float(target_value),
            "disposition": disposition,
            "execution_recipe": self.rles_transition_hash,
        }

    def _observation(self, snapshot: Any, *, guard_intervention: float = 0.0) -> np.ndarray:
        if self._episode is None:
            raise ContractNotReady("episode missing while building observation")
        inputs = self._episode.observation_inputs(
            snapshot=snapshot, reward_state=self._reward,
            guard_intervention=guard_intervention,
        )
        obs = self._observation_builder.build(**inputs)
        if not self.observation_space.contains(obs):
            raise ContractNotReady("observation violates immutable Gymnasium space")
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.close()
        if seed is None:
            seed = int(self.np_random.integers(0, 2**32))
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dictionary")
        self._episode = self._factory(seed=seed, options={} if options is None else options)
        try:
            first = self._episode.bootstrap()
            episode_reward = getattr(self._episode, "reward_config", self._reward.config)
            if episode_reward != self._reward.config:
                raise ContractNotReady("episode replay and Gym reward configuration differ")
            if type(first.decision_ts_ns) is not int or first.decision_ts_ns <= 0:
                raise ContractNotReady("invalid initial decision timestamp")
            self._reward.reset(float(first.equity))
            self._step_count = 0
            guard = 0.0
            restore = getattr(self._episode, "restored_env_state", None)
            state = restore() if callable(restore) else None
            if state is not None:
                if (type(state) is not EpisodeResumeState or type(state.step_count) is not int or
                        not 0 <= state.step_count < self._max_steps or
                        not np.isfinite(state.guard_intervention) or not 0 <= state.guard_intervention <= 2):
                    raise ContractNotReady("invalid restored Gym episode state")
                self._reward.restore(state.reward)
                if state.reward.previous_equity != float(first.equity):
                    raise ContractNotReady("restored reward differs from native account equity")
                self._step_count, guard = state.step_count, state.guard_intervention
            obs = self._observation(first, guard_intervention=guard)
            self._done = False
            self._last_observation = obs.copy()
            # Prove that the first state is already bound to the complete
            # transition recipe and an L3-owned restore boundary.
            self.rles_transition_hash
            self.rles_boundary_token
            first_inputs = self._episode.observation_inputs(
                snapshot=first, reward_state=self._reward, guard_intervention=guard,
            )
            return obs, {
                "observation_schema_hash": self._observation_builder.schema.hash,
                "decision_ts_ns": first.decision_ts_ns,
                "initial_equity": str(first.equity),
                "episode_window": self._episode.episode_window,
                "transition_hash": self.rles_transition_hash,
                "native_identity": self._episode.native_identity,
                "split": self._observation_builder.split,
                "market_features": {name: float(feature.value)
                                    for name, feature in first_inputs["market"].items()},
            }
        except BaseException:
            self.close()
            raise

    def step(self, action: Any):
        if self._done or self._episode is None:
            raise ContractNotReady("reset required before step")
        try:
            vector = np.asarray(action)
            if vector.shape != (1,) or vector.dtype.kind not in "fi" or not np.isfinite(vector).all():
                raise ValueError("action must be finite shape=(1,) numeric vector")
            raw = float(vector[0])
            if not -1 <= raw <= 1:
                raise ValueError("raw action outside Box bounds")
            # The reviewed episode implementation must execute soft -> hard ->
            # quantity planner -> official native engine, never action -> fill.
            plan, guard = self._episode.action_to_plan(raw)
            risk_review = self._episode.pending_risk_review
            pending_snapshot = getattr(self._episode, "_snapshot", None)
            pending_order_ids = tuple(sorted(
                row["client_order_id"]
                for row in getattr(pending_snapshot, "orders", ())
                if isinstance(row.get("client_order_id"), str)
            ))
            related_order_ids = (plan.cancel_order_ids if plan.action == "CANCEL" else
                                 pending_order_ids if plan.action == "WAIT" else ())
            guard = float(guard)
            if not np.isfinite(guard) or not 0 <= guard <= 2:
                raise ContractNotReady("guard intervention invalid")
            previous_ts = self._episode.previous_decision_ts_ns
            result = self._episode.advance(plan)
            if type(result.decision_ts_ns) is not int or result.decision_ts_ns <= previous_ts:
                raise ContractNotReady("native decision timestamp failed to advance")
            if not isinstance(result.evidence_hash, str) or not result.evidence_hash:
                raise ContractNotReady("native step evidence is missing")
            self._step_count += 1
            truncated = bool(result.truncated or self._step_count >= self._max_steps)
            terminated = bool(result.terminated or float(result.snapshot.equity) <= 0)
            margin_facts_known, maintenance_breach, liquidation_detected = (
                _native_constraint_facts(result.snapshot)
            )
            evidence_hash = result.evidence_hash
            snapshot = result.snapshot
            terminal_settlement = None
            if terminated or truncated:
                # Even a bankruptcy must not receive a cost-free terminal reward.
                # If liquidation/flatten cannot be proven, the attempt is invalid.
                terminal = self._episode.drain_and_close("force_flatten_with_cost")
                if terminal is None or terminal.unsettled_orders or terminal.open_positions:
                    raise ContractNotReady("native terminal close/ACK not proven")
                if not isinstance(terminal.evidence_hash, str) or not terminal.evidence_hash:
                    raise ContractNotReady("missing native terminal settlement evidence")
                snapshot = terminal.snapshot
                terminal_settlement = _terminal_settlement_facts(
                    result.snapshot, snapshot, policy_exposure=float(result.executed_action),
                )
                terminal_margin_known, terminal_margin_breach, terminal_liquidation = (
                    _native_constraint_facts(terminal.snapshot)
                )
                margin_facts_known = margin_facts_known and terminal_margin_known
                maintenance_breach = maintenance_breach or terminal_margin_breach
                liquidation_detected = liquidation_detected or terminal_liquidation
                if type(snapshot.decision_ts_ns) is not int or snapshot.decision_ts_ns < result.decision_ts_ns:
                    raise ContractNotReady("terminal snapshot clock rewound")
                evidence_hash = sha256((result.evidence_hash + "|" + terminal.evidence_hash).encode()).hexdigest()
                # This is a finite, settled trading task, including max_steps.
                # No account/position continues past this terminal observation.
                terminated, truncated = True, False
            breakdown = self._reward.calculate(native_equity=float(snapshot.equity), guard_intervention=guard)
            terminated = terminated or breakdown.insolvency
            transition_inputs = self._episode.observation_inputs(
                snapshot=snapshot, reward_state=self._reward,
                guard_intervention=guard,
            )
            obs = self._observation_builder.build(**transition_inputs)
            if not self.observation_space.contains(obs):
                raise ContractNotReady("observation violates immutable Gymnasium space")
            self._last_observation = obs.copy()
            statuses_before = _native_order_status_rows(result.previous_snapshot)
            statuses_after = _native_order_status_rows(result.snapshot)
            pending_before = tuple(sorted(
                str(row["client_order_id"]) for row in result.previous_snapshot.orders
            ))
            pending_after = tuple(sorted(
                str(row["client_order_id"]) for row in result.snapshot.orders
            ))
            info = {
                "r_econ": breakdown.r_econ, "r_shape": breakdown.r_shape,
                "r_train": breakdown.r_total,
                "native_equity": str(snapshot.equity), "drawdown": breakdown.drawdown,
                "native_equity_before": str(result.previous_snapshot.equity),
                "policy_equity_before": str(result.previous_snapshot.equity),
                "policy_equity_after": str(result.snapshot.equity),
                "policy_equity_change": str(
                    Decimal(str(result.snapshot.equity))
                    - Decimal(str(result.previous_snapshot.equity))
                ),
                "policy_equity_currency": result.snapshot.currency,
                "settlement_after_ns": snapshot.decision_ts_ns,
                "native_pending_order_ids_before": pending_before,
                "native_pending_order_ids_after": pending_after,
                "native_order_statuses_before": statuses_before,
                "native_order_statuses_after": statuses_after,
                "native_order_status_receipt": _native_order_status_receipt(
                    statuses_before, statuses_after,
                ),
                "native_policy_fills": _native_policy_fills(
                    result.previous_snapshot, result.snapshot,
                ),
                "native_net_exposure": float(self._episode.exposure_at(result.snapshot)),
                "native_exposure_ts_ns": result.snapshot.decision_ts_ns,
                "native_decision_ts_before_ns": result.previous_snapshot.decision_ts_ns,
                "native_decision_ts_after_ns": result.snapshot.decision_ts_ns,
                "native_fills": _new_native_records(
                    result.previous_snapshot.fills, result.snapshot.fills,
                ),
                "native_funding": _new_native_records(
                    result.previous_snapshot.funding, result.snapshot.funding,
                ),
                "requested_action": raw, "risk_guard_intervention": guard,
                "risk_soft_adjustment": float(risk_review.soft_adjustment),
                "risk_hard_adjustment": float(risk_review.hard_adjustment),
                "risk_intervention_components_known": risk_review.components_known,
                "insolvency": bool(breakdown.insolvency or float(result.snapshot.equity) <= 0),
                "margin_facts_known": margin_facts_known,
                "maintenance_margin_breach": maintenance_breach,
                "native_liquidation_detected": liquidation_detected,
                "risk_soft_target": None if risk_review.soft_target is None else float(risk_review.soft_target),
                "risk_adjusted_target": None if risk_review.risk_target is None else float(risk_review.risk_target),
                "quantity_rounded_target_exposure": (
                    None if risk_review.rounded_target_exposure is None
                    else float(risk_review.rounded_target_exposure)),
                "execution_plan_action": plan.action,
                "execution_plan_reason": plan.reason,
                "execution_decision_id": plan.decision_id,
                "execution_plan_id": plan.plan_id,
                "execution_idempotency_key": plan.idempotency_key,
                "execution_delta_qty": str(plan.delta_qty),
                "execution_cancel_order_ids": plan.cancel_order_ids,
                "execution_related_order_ids": related_order_ids,
                "execution_client_order_id": (
                    "RLES-" + sha256(plan.idempotency_key.encode("utf-8")).hexdigest()
                    if plan.action == "SUBMIT" else None
                ),
                "executed_action": result.executed_action, "evidence_hash": evidence_hash,
                # This is the policy barrier time. Terminal settlement may
                # advance the native account clock through additional drain
                # frames; those are not extra RL decision samples.
                "decision_ts_ns": result.decision_ts_ns,
                "terminal_settled": bool(terminated or truncated),
                "terminal_value_semantics": "finite_settlement_no_bootstrap_v1",
                "policy_executed_action_pre_terminal": result.executed_action,
                "episode_window": dict(self._episode.episode_window),
                "market_features": {
                    name: float(feature.value)
                    for name, feature in transition_inputs["market"].items()
                },
                "net_exposure": float(
                    transition_inputs["account"].get("net_exposure", 0.0)
                ),
            }
            if terminal_settlement is not None:
                info["terminal_settlement"] = terminal_settlement
            if terminated or truncated:
                self._done = True
                self.close()
            return obs, float(breakdown.r_total), terminated, truncated, info
        except BaseException:
            self._done = True
            self.close()
            raise

    def close(self) -> None:
        if self._episode is not None:
            try:
                self._episode.close()
            finally:
                self._episode = None
                self._last_observation = None
                self._done = True
