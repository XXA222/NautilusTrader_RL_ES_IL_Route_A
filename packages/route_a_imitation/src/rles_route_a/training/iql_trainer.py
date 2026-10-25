"""Independent PyTorch IQL for FROZEN history embeddings and native rewards.

This is an original small implementation of published IQL math. It does NOT
pretend to be the official JAX implementation, nor infer executable fills.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
from math import log, isfinite
import torch
from torch import nn
from torch.nn import functional as F
from .offline_common import mlp, validate_batch, gamma_factor, safe_update


@dataclass(frozen=True)
class IQLHyper:
    gamma: float = 0.99
    expectile: float = 0.7
    beta: float = 3.
    max_weight: float = 100.
    lr: float = 3e-4
    tau_target: float = 0.005
    hidden_size: int = 128

    def __post_init__(self) -> None:
        if (not all(type(x) in (int,float) and isfinite(x) for x in
                    (self.gamma,self.expectile,self.beta,self.max_weight,self.lr,self.tau_target))
                or type(self.hidden_size) is not int or not (1 <= self.hidden_size <= 2048)
                or not (0 < self.gamma <= 1 and 0 < self.expectile < 1 and self.beta > 0
                        and self.max_weight >= 1 and 0 < self.lr < 1 and 0 < self.tau_target <= 1)):
            raise ValueError("invalid IQL config")


class SquashedGaussian(nn.Module):
    def __init__(self, state_dim: int, hidden: int):
        super().__init__()
        self.mu = mlp(state_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.tensor([-1.]))

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mu(state), self.log_std.clamp(-6, 2).exp())

    def log_prob(self, s: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Boundary commands a=+-1 exist in trading. Their Gaussian density is
        # undefined after exact tanh; evaluate a dequantized boundary proxy at
        # 1-1e-6, never return inf or silently alter execution commands.
        a = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        z = torch.atanh(a)
        dist = self.distribution(s)
        # Stable log |d tanh(z)/dz| = 2*(ln 2-z-softplus(-2z)).
        log_jacobian = 2. * (log(2.) - z - F.softplus(-2. * z))
        return (dist.log_prob(z) - log_jacobian).sum(-1, keepdim=True)

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        return self.mu(state).tanh()


class IQLTrainer:
    def __init__(self, state_dim: int, cfg: IQLHyper = IQLHyper(), *, device: str = "cpu"):
        self.cfg = cfg
        self.state_dim = state_dim
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.q1 = mlp(state_dim + 1, 1, cfg.hidden_size).to(self.device)
        self.q2 = mlp(state_dim + 1, 1, cfg.hidden_size).to(self.device)
        self.value = mlp(state_dim, 1, cfg.hidden_size).to(self.device)
        self.actor = SquashedGaussian(state_dim, cfg.hidden_size).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).eval()
        self.q2_target = copy.deepcopy(self.q2).eval()
        self.q_opt = torch.optim.Adam((*self.q1.parameters(), *self.q2.parameters()), lr=cfg.lr)
        self.v_opt = torch.optim.Adam(self.value.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)

    def train_step(self, b: dict[str, torch.Tensor]) -> dict[str, float]:
        dim, device = validate_batch(b)
        if device != self.device or dim != self.value[0].in_features:
            raise ValueError("offline frozen embedding schema/device mismatch")
        s, a, r, ns, bootstrap = (b[k] for k in ("s", "a", "r", "ns", "bootstrap"))
        with torch.no_grad():
            q_min = torch.min(self.q1_target(torch.cat([s, a], -1)),
                              self.q2_target(torch.cat([s, a], -1)))
        # IQL expectile asymmetry: τ when Q−V>0, 1−τ otherwise.
        v = self.value(s)
        residual = q_min - v
        weight = torch.where(residual > 0, self.cfg.expectile, 1 - self.cfg.expectile)
        v_loss = (weight * residual.square()).mean()
        safe_update(v_loss, self.v_opt, list(self.value.parameters()), "IQL-V")
        with torch.no_grad():
            target = r + gamma_factor(b, self.cfg.gamma) * bootstrap * self.value(ns)
        sa = torch.cat([s, a], -1)
        q_loss = F.mse_loss(self.q1(sa), target) + F.mse_loss(self.q2(sa), target)
        safe_update(q_loss, self.q_opt, list(self.q1.parameters()) + list(self.q2.parameters()), "IQL-Q")
        with torch.no_grad():
            advantage = q_min - self.value(s)
            weights = torch.exp((self.cfg.beta * advantage).clamp(max=log(self.cfg.max_weight)))
        a_loss = -(weights * self.actor.log_prob(s, a)).mean()
        safe_update(a_loss, self.actor_opt, list(self.actor.parameters()), "IQL-actor")
        with torch.no_grad():
            for q, target_q in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(q.parameters(), target_q.parameters()):
                    tp.lerp_(p, self.cfg.tau_target)
        result = {"v_loss": float(v_loss.item()), "q_loss": float(q_loss.item()),
                  "actor_loss": float(a_loss.item()), "mean_weight": float(weights.mean().item())}
        if not all(torch.isfinite(torch.tensor(v)) for v in result.values()):
            raise FloatingPointError("IQL encountered nonfinite loss")
        return result

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor.deterministic_action(state.to(self.device))
