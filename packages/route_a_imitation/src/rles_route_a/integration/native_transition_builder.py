"""Nonterminal native Gym barrier -> bounded offline RL transition.

    The decision action is the policy's behavior command. Its mapping through
    SoftRisk, HardRisk, quantity planning, and orders remains in the native trace.
    The next state preserves actual exposure and pending orders, so the learner
    does not assume that a submitted target filled immediately.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.datasets.transitions import OfflineTransition
from .host_gym_projection import HostGymProjection, TerminalL6Receipt
from .native_economics import NativeRewardReceipt


@dataclass(frozen=True, slots=True)
class EncodedNativeState:
    observation: PointInTimeObservation
    vector: tuple[float, ...]
    frozen_encoder_id: str
    source_identity: str
    settlement_receipt_id: str = ""

    def __post_init__(self) -> None:
        if (not isinstance(self.observation, PointInTimeObservation) or
                not isinstance(self.frozen_encoder_id, str) or not self.frozen_encoder_id.strip() or
                not isinstance(self.source_identity, str) or not self.source_identity.strip() or
                not isinstance(self.settlement_receipt_id, str) or
                (self.settlement_receipt_id and not self.settlement_receipt_id.strip())):
            raise ValueError("frozen encoded observation identity missing")
        if not isinstance(self.vector, tuple) or not self.vector or any(
            not isinstance(x, (int, float)) or isinstance(x, bool) or not isfinite(x)
            for x in self.vector):
            raise ValueError("invalid native frozen state embedding")


def native_nonterminal_transition(pre: EncodedNativeState,
                                  post: EncodedNativeState,
                                  native: HostGymProjection, *,
                                  decision_period_ns: int,
                                  sample_id: str,
                                  exposure_tolerance: float = 1e-6) -> OfflineTransition:
    """Require a genuine same-episode native transition and L3 equity receipt.

    No terminal conversion here: the host must instead use complete original L6
    settlement facts to handle terminal log reward and bootstrap semantics.
    """
    if type(decision_period_ns) is not int or decision_period_ns <= 0:
        raise ValueError("positive decision_period_ns required")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("native transition sample identity required")
    if not isinstance(native, HostGymProjection):
        raise TypeError("nonterminal native host projection required")
    b = native.teacher_barrier
    before, after = pre.observation, post.observation
    if (before != b.observation or before.fold_id != after.fold_id or
            before.episode_id != after.episode_id or
            before.fold_id == ""):
        raise PermissionError("native offline transition cannot switch registered Fold/episode")
    if not (pre.frozen_encoder_id == post.frozen_encoder_id and
            pre.source_identity == post.source_identity == b.source_identity):
        raise PermissionError("native offline state source/encoder identities changed")
    if len(pre.vector) != len(post.vector):
        raise ValueError("native encoded state dimensions changed")
    delta = after.decision_ts_ns - before.decision_ts_ns
    if delta < decision_period_ns or delta % decision_period_ns:
        raise ValueError("native decision interval is not a positive multiple of frozen clock")
    if after.decision_ts_ns != native.advance_ts_ns:
        raise ValueError("L3 post-advance clock does not match actual next observation")
    if "net_exposure" not in after.account or abs(
            after.account_value("net_exposure") - b.trace.actual_exposure_after) > exposure_tolerance:
        raise ValueError("post-native observation disagrees with actual executed exposure")
    if tuple(after.pending_order_ids) != b.trace.pending_order_ids_after:
        raise ValueError("post-native observation pending orders disagree with native trace")
    b.reward.validate()
    return OfflineTransition(
        state=pre.vector, requested_action=b.trace.requested_action,
        executed_target=b.trace.planner_target,
        native_log_reward=b.reward.recorded_r_econ, next_state=post.vector,
        terminated=False, truncated=False, bootstrap_allowed=True,
        provenance=f"{native.native_evidence_receipt}:{sample_id}",
        split="train", fold_id=before.fold_id, episode_id=before.episode_id,
        decision_ts_ns=before.decision_ts_ns,
        next_decision_ts_ns=after.decision_ts_ns,
        discount_steps=delta // decision_period_ns,
        native_execution_recipe=b.execution_recipe,
        action_trace=b.trace,
        source_identity=pre.source_identity,
        frozen_encoder_id=pre.frozen_encoder_id,
    )


def native_terminal_transition(pre: EncodedNativeState,
                               settled: EncodedNativeState,
                               native: HostGymProjection,
                               terminal_receipt: TerminalL6Receipt, *,
                               decision_period_ns: int,
                               sample_id: str,
                               terminated: bool,
                               truncated: bool = False) -> OfflineTransition:
    """Build an L6-bound terminal replay row using the actual settled state.

    The post-state must be encoded from the host after its forced settlement;
    it cannot be copied from the pre-terminal policy observation. Terminal
    replay never bootstraps, including collection truncations.
    """
    if type(decision_period_ns) is not int or decision_period_ns <= 0:
        raise ValueError("positive decision_period_ns required")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("native transition sample identity required")
    if type(terminated) is not bool or type(truncated) is not bool or terminated == truncated:
        raise ValueError("exactly one native terminal or truncation flag is required")
    if not isinstance(native, HostGymProjection):
        raise TypeError("terminal native host projection required")
    if not isinstance(terminal_receipt, TerminalL6Receipt):
        raise TypeError("host-issued terminal L6 receipt required")
    b = native.teacher_barrier
    before, after = pre.observation, settled.observation
    if (before != b.observation or before.fold_id != after.fold_id
            or before.episode_id != after.episode_id or not before.fold_id):
        raise PermissionError("terminal offline transition cannot switch registered Fold/episode")
    if not (pre.frozen_encoder_id == settled.frozen_encoder_id and
            pre.source_identity == settled.source_identity == b.source_identity == terminal_receipt.source_identity):
        raise PermissionError("terminal native state/receipt source or encoder identities changed")
    if len(pre.vector) != len(settled.vector):
        raise ValueError("terminal native encoded state dimensions changed")
    if (b.terminal_l6_receipt != terminal_receipt.receipt_id
            or settled.settlement_receipt_id != terminal_receipt.receipt_id
            or terminal_receipt.policy_barrier_ts_ns != before.decision_ts_ns
            or terminal_receipt.settlement_barrier_ts_ns != after.decision_ts_ns
            or terminal_receipt.equity_before != b.reward.prior_equity
            or terminal_receipt.settled_equity_after != b.reward.next_equity
            or terminal_receipt.recorded_r_econ != b.reward.recorded_r_econ):
        raise ValueError("terminal native projection, state clock, and L6 receipt disagree")
    if not isinstance(b.reward, NativeRewardReceipt):
        raise TypeError("native terminal reward receipt required")
    b.reward.validate()
    if "net_exposure" not in after.account or abs(after.account_value("net_exposure")) > 1e-6:
        raise ValueError("terminal post-state must reflect the settled flat account")
    if after.pending_order_ids:
        raise ValueError("terminal post-state cannot retain pending native orders")
    delta = after.decision_ts_ns - before.decision_ts_ns
    if delta <= 0 or delta % decision_period_ns:
        raise ValueError("terminal settlement clock is not a positive multiple of frozen decision clock")
    return OfflineTransition(
        state=pre.vector, requested_action=b.trace.requested_action,
        executed_target=b.trace.planner_target,
        native_log_reward=b.reward.recorded_r_econ, next_state=settled.vector,
        terminated=terminated, truncated=truncated, bootstrap_allowed=False,
        provenance=f"{native.native_evidence_receipt}:{terminal_receipt.receipt_id}:{sample_id}",
        split="train", fold_id=before.fold_id, episode_id=before.episode_id,
        decision_ts_ns=before.decision_ts_ns, next_decision_ts_ns=after.decision_ts_ns,
        discount_steps=delta // decision_period_ns,
        native_execution_recipe=b.execution_recipe, action_trace=b.trace,
        source_identity=pre.source_identity,
        frozen_encoder_id=pre.frozen_encoder_id,
    )
