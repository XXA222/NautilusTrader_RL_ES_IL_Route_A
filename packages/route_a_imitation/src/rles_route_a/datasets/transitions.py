"""Offline native account transitions; no surrogate PnL or synthetic fill events.

    The learned action is the behavior command issued at the policy barrier.
    Its native risk/planner/order mapping remains attached as an audit trace,
    while the next state contains actual post-native exposure and pending orders.
    Stage-RA4 requires a registered fold, causal clock and execution identity.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Sequence
import numpy as np
import torch
from rles_route_a.contracts import NativeActionTrace


@dataclass(frozen=True)
class OfflineTransition:
    state: tuple[float, ...]
    requested_action: float
    native_log_reward: float
    next_state: tuple[float, ...]
    terminated: bool
    truncated: bool
    bootstrap_allowed: bool
    provenance: str
    split: str = "train"
    fold_id: str = ""              # required by RA4 strict registered production gate
    episode_id: str = ""
    decision_ts_ns: int = 0
    next_decision_ts_ns: int = 0
    discount_steps: int = 1       # actual decision intervals for gamma^steps
    native_execution_recipe: str = ""
    # Exact L3 trace for the command->risk->planner->order mapping. Kept on the
    # replay row so registered production training can audit command aliasing.
    action_trace: NativeActionTrace | None = None
    # Final native planner target retained for action-mapping diagnostics.
    # IQL/TD3+BC learn requested_action, the behavior command in the trace.
    executed_target: float | None = None
    source_identity: str = ""
    frozen_encoder_id: str = ""

    def __post_init__(self) -> None:
        if self.split != "train":
            raise PermissionError("offline RL may only train on registered Train samples")
        if not self.state or len(self.state) != len(self.next_state):
            raise ValueError("state/next_state frozen encoded dimensions mismatch")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not isfinite(v)
               for v in (*self.state, *self.next_state, self.requested_action, self.native_log_reward)):
            raise ValueError("nonfinite offline transition")
        if not -1 <= self.requested_action <= 1:
            raise ValueError("offline requested action outside current command domain")
        if (self.executed_target is None
                or not isinstance(self.executed_target, (int, float))
                or isinstance(self.executed_target, bool)
                or not isfinite(self.executed_target)
                or not -1 <= self.executed_target <= 1):
            raise ValueError("offline native executed target must be finite and within [-1, 1]")
        if not self.provenance:
            raise ValueError("native transition provenance required")
        if ((self.source_identity and
             (not isinstance(self.source_identity, str) or not self.source_identity.strip()))
                or (self.frozen_encoder_id and
                    (not isinstance(self.frozen_encoder_id, str)
                     or not self.frozen_encoder_id.strip()))):
            raise ValueError("invalid offline source or frozen encoder identity")
        if type(self.discount_steps) is not int or self.discount_steps < 1:
            raise ValueError("positive discount_steps required")
        if (self.terminated or self.truncated) and self.bootstrap_allowed:
            raise ValueError("cannot bootstrap at true terminal or truncated native boundary")
        if self.terminated and self.truncated:
            raise ValueError("must disambiguate terminal from collection truncation")
        if self.decision_ts_ns or self.next_decision_ts_ns:
            if (type(self.decision_ts_ns) is not int or type(self.next_decision_ts_ns) is not int
                    or self.decision_ts_ns <= 0 or self.next_decision_ts_ns <= self.decision_ts_ns):
                raise ValueError("offline transition clocks must be causal and increasing")
        if self.fold_id and (not isinstance(self.fold_id, str) or not self.fold_id.strip()):
            raise ValueError("invalid registered fold")
        if self.action_trace is not None:
            if not isinstance(self.action_trace, NativeActionTrace):
                raise TypeError("offline action mapping must be a native action trace")
            if self.action_trace.requested_action != self.requested_action:
                raise ValueError("offline requested action differs from native action trace")
            if self.action_trace.planner_target != self.executed_target:
                raise ValueError("offline executed target differs from native planner target")


class OfflineBatch:
    def __init__(self, transitions: Sequence[OfflineTransition], *, device: str = "cpu",
                 reward_scale: float = 1.0):
        if not transitions:
            raise ValueError("empty offline replay")
        if not isinstance(reward_scale, (int,float)) or isinstance(reward_scale, bool) or not isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("finite positive numerical reward_scale required")
        # Use np.float64 intermediate to reject float32 overflow before allocating
        # the potentially GPU-resident learner data.
        arrays = [np.asarray([getattr(t, name) for t in transitions], dtype=np.float64)
                  for name in ("state", "next_state")]
        if arrays[0].ndim != 2 or arrays[0].shape != arrays[1].shape:
            raise ValueError("offline embedded tensor dimensions mismatch")
        float_max = np.finfo(np.float32).max
        if any(not np.isfinite(x).all() or np.any(np.abs(x) > float_max) for x in arrays):
            raise ValueError("frozen encoder values cannot be represented as float32")
        scaled_reward = np.asarray([t.native_log_reward * reward_scale for t in transitions], np.float64)
        if not np.isfinite(scaled_reward).all() or np.any(np.abs(scaled_reward) > float_max):
            raise ValueError("scaled economic reward cannot be represented as float32")
        self.state = torch.tensor(arrays[0], dtype=torch.float32, device=device)
        self.next_state = torch.tensor(arrays[1], dtype=torch.float32, device=device)
        self.action = torch.tensor([[t.requested_action] for t in transitions], dtype=torch.float32, device=device)
        self.reward = torch.tensor(scaled_reward[:,None], dtype=torch.float32, device=device)
        self.bootstrap = torch.tensor([[float(t.bootstrap_allowed)] for t in transitions], dtype=torch.float32, device=device)
        self.discount_steps = torch.tensor([[float(t.discount_steps)] for t in transitions], dtype=torch.float32, device=device)

    @property
    def size(self) -> int:
        return self.state.shape[0]

    def indices(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"s": self.state[indices], "a": self.action[indices],
                "r": self.reward[indices], "ns": self.next_state[indices],
                "bootstrap": self.bootstrap[indices],
                "discount_steps": self.discount_steps[indices]}
