"""Frozen encoder + trained IQL/TD3+BC actor; inference only.

The host L3 still maps normalized targets through its own risk and execution.
"""
from __future__ import annotations
import numpy as np
import torch
from rles_route_a.models.encoder import MultiScaleGRU


@torch.no_grad()
def encode_frozen(encoder: MultiScaleGRU, x: torch.Tensor,
                  lengths: torch.Tensor) -> torch.Tensor:
    if encoder.training:
        raise ValueError("IQL v1 only accepts FROZEN BC encoder")
    if any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("freeze all encoder params before offline RL")
    return encoder(x, lengths).detach()


@torch.no_grad()
def predict_offline(actor: object, frozen_embedding: torch.Tensor) -> float:
    value = actor.predict(frozen_embedding.reshape(1, -1))  # type: ignore[attr-defined]
    action = float(value.reshape(-1)[0].item())
    if not np.isfinite(action) or not -1 <= action <= 1:
        raise ValueError("offline actor target outside NETTING range")
    return action
