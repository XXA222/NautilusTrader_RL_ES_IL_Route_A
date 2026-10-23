"""TD3+BC offline control baseline, using same frozen native replay batches."""
from __future__ import annotations
from dataclasses import dataclass
import copy
from math import isfinite
import torch
from torch import nn
from torch.nn import functional as F
from .offline_common import mlp, validate_batch, gamma_factor, safe_update


@dataclass(frozen=True)
class TD3BCHyper:
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 2.5
    lr: float = 3e-4
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    hidden_size: int = 128

    def __post_init__(self) -> None:
        if (not all(type(x) in (int,float) and isfinite(x) for x in
                    (self.gamma,self.tau,self.alpha,self.lr,self.policy_noise,self.noise_clip))
                or type(self.policy_delay) is not int or self.policy_delay < 1
                or type(self.hidden_size) is not int or not 1 <= self.hidden_size <= 2048
                or not (0 < self.gamma <= 1 and 0 < self.tau <= 1 and self.alpha > 0
                        and self.lr > 0 and self.policy_noise >= 0 and self.noise_clip >= 0)):
            raise ValueError("invalid TD3+BC hyperparameters")


class TD3BCTrainer:
    def __init__(self, state_dim: int, cfg: TD3BCHyper = TD3BCHyper(), *, device: str = "cpu", seed: int = 31):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("invalid TD3+BC target noise seed")
        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.noise_seed = seed
        self._noise_generator = torch.Generator(device=self.device).manual_seed(seed)
        self.actor = nn.Sequential(mlp(state_dim, 1, cfg.hidden_size), nn.Tanh()).to(self.device)
        self.q1 = mlp(state_dim + 1, 1, cfg.hidden_size).to(self.device)
        self.q2 = mlp(state_dim + 1, 1, cfg.hidden_size).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).eval()
        self.q1_target = copy.deepcopy(self.q1).eval()
        self.q2_target = copy.deepcopy(self.q2).eval()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam((*self.q1.parameters(), *self.q2.parameters()), lr=cfg.lr)
        self.steps = 0
        self.state_dim = state_dim

    def train_step(self, b: dict[str, torch.Tensor]) -> dict[str, float | None]:
        dim, device = validate_batch(b)
        if dim != self.state_dim or device != self.device:
            raise ValueError("invalid frozen offline embeddings/device")
        s, a, r, ns, bootstrap = (b[k] for k in ("s", "a", "r", "ns", "bootstrap"))
        self.steps += 1
        with torch.no_grad():
            noise = (torch.randn(a.shape, device=a.device, dtype=a.dtype,
                                 generator=self._noise_generator) * self.cfg.policy_noise).clamp(
                                     -self.cfg.noise_clip, self.cfg.noise_clip)
            next_a = (self.actor_target(ns) + noise).clamp(-1, 1)
            target_q = torch.min(self.q1_target(torch.cat([ns, next_a], -1)),
                                 self.q2_target(torch.cat([ns, next_a], -1)))
            target = r + gamma_factor(b, self.cfg.gamma) * bootstrap * target_q
        sa = torch.cat([s, a], -1)
        loss_q = F.mse_loss(self.q1(sa), target) + F.mse_loss(self.q2(sa), target)
        safe_update(loss_q, self.critic_opt,
                    list(self.q1.parameters()) + list(self.q2.parameters()), "TD3BC-critic")
        metrics: dict[str, float | None] = {"q_loss": float(loss_q.item()), "actor_loss": None}
        if self.steps % self.cfg.policy_delay == 0:
            pred = self.actor(s)
            q_pred = self.q1(torch.cat([s, pred], -1))
            scale = self.cfg.alpha / q_pred.detach().abs().mean().clamp_min(1e-6)
            loss_a = -(scale * q_pred).mean() + F.mse_loss(pred, a)
            safe_update(loss_a, self.actor_opt, list(self.actor.parameters()), "TD3BC-actor")
            metrics["actor_loss"] = float(loss_a.item())
            with torch.no_grad():
                for net, target_net in ((self.actor, self.actor_target),
                                        (self.q1, self.q1_target), (self.q2, self.q2_target)):
                    for p, tp in zip(net.parameters(), target_net.parameters()):
                        tp.lerp_(p, self.cfg.tau)
        if not torch.isfinite(loss_q):
            raise FloatingPointError("nonfinite TD3 critic")
        return metrics

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state.to(self.device))
