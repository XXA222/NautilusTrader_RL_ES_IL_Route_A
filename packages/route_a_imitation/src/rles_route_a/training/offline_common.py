"""Shared offline learner validation and numerically safe updates."""
from __future__ import annotations
import torch
from torch import nn


def mlp(input_dim: int, output_dim: int, hidden: int = 128) -> nn.Sequential:
    if any(type(x) is not int or x < 1 for x in (input_dim, output_dim, hidden)):
        raise ValueError("invalid positive model dimensions")
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))


def validate_batch(batch: dict[str, torch.Tensor]) -> tuple[int, torch.device]:
    s, a, r, ns, boot = (batch[k] for k in ("s", "a", "r", "ns", "bootstrap"))
    if (s.ndim != 2 or ns.shape != s.shape or a.shape != (s.shape[0], 1)
            or r.shape != a.shape or boot.shape != a.shape or s.shape[0] < 1):
        raise ValueError("invalid [B,Z], [B,1] offline tensors")
    items = [s, a, r, ns, boot]
    if "discount_steps" in batch:
        steps = batch["discount_steps"]
        if steps.shape != a.shape or not bool((steps >= 1).all().item()) or not bool((steps == steps.floor()).all().item()):
            raise ValueError("invalid discount_steps")
        items.append(steps)
    if any(x.device != s.device or x.dtype != torch.float32 or not bool(torch.isfinite(x).all().item()) for x in items):
        raise ValueError("nonfinite, non-float32 or device-mixed offline batch")
    if not bool(((a >= -1) & (a <= 1)).all().item()) or not bool(((boot == 0) | (boot == 1)).all().item()):
        raise ValueError("command out of bounds or invalid bootstrap flag")
    return s.shape[1], s.device


def gamma_factor(batch: dict[str, torch.Tensor], gamma: float) -> torch.Tensor:
    r = batch["r"]
    steps = batch.get("discount_steps")
    if steps is None:
        steps = torch.ones_like(r)  # v0.1 batch compatibility for isolated tests
    return torch.pow(torch.full_like(r, gamma), steps)


def safe_update(loss: torch.Tensor, opt: torch.optim.Optimizer,
                params: list[torch.nn.Parameter], name: str, max_grad_norm: float = 10.) -> None:
    if not bool(torch.isfinite(loss).all().item()):
        raise FloatingPointError(f"{name}: nonfinite optimizer loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
    if not bool(torch.isfinite(norm).item()):
        raise FloatingPointError(f"{name}: nonfinite gradient")
    opt.step()
    if any(not bool(torch.isfinite(p).all().item()) for p in params):
        raise FloatingPointError(f"{name}: optimizer produced nonfinite parameters")
