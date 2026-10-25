"""BC-RNN with masked, confidence-weighted intent and requested-action losses."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from rles_route_a.contracts import INTENTS
from .encoder import MultiScaleGRU


class BCGRUPolicy(nn.Module):
    def __init__(self, feature_names: tuple[str, ...], hidden_size: int = 128, layers: int = 2):
        super().__init__()
        self.encoder = MultiScaleGRU(feature_names, hidden_size, layers)
        self.intent_head = nn.Linear(hidden_size, len(INTENTS))
        self.exposure_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, lengths)
        return self.intent_head(z), self.exposure_head(z).tanh().squeeze(-1)


def bc_losses(logits: torch.Tensor, exposure: torch.Tensor, *,
              target_intent: torch.Tensor, target_exposure: torch.Tensor,
              exposure_weight: float = 1., intent_weight: float = 0.25,
              sample_weight: torch.Tensor | None = None,
              intent_sample_weight: torch.Tensor | None = None,
              intent_class_weights: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if logits.ndim != 2 or logits.shape[-1] != len(INTENTS):
        raise ValueError("invalid intent logits")
    size = logits.shape[0]
    if (exposure.shape != (size,) or target_intent.shape != (size,)
            or target_exposure.shape != (size,)):
        raise ValueError("BC target tensors shape mismatch")
    if sample_weight is None:
        weights = torch.ones_like(exposure)
    else:
        if sample_weight.shape != (size,) or not bool(torch.isfinite(sample_weight).all()) or not bool(((sample_weight >= 0) & (sample_weight <= 1)).all()):
            raise ValueError("invalid sample weights")
        weights = sample_weight
    if intent_sample_weight is None:
        intent_weights = weights
    else:
        if (intent_sample_weight.shape != (size,)
                or not bool(torch.isfinite(intent_sample_weight).all())
                or not bool(((intent_sample_weight >= 0) & (intent_sample_weight <= 1)).all())):
            raise ValueError("invalid intent sample weights")
        intent_weights = intent_sample_weight
    # A fractional confidence weight changes a sample's contribution relative
    # to other rows, but must not shrink the entire minibatch loss when its
    # total weight is below one. Keep the all-zero batch finite for callers
    # that inspect it, while training skips such batches.
    denom = weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    # The exposure target is the teacher's requested action. Its sample weight
    # is selected by the recipe: legacy recipes require execution evidence,
    # while teacher_request_v2 supervises every audited teacher command. Native
    # realized exposure remains a separate L3 fact and is never fabricated here.
    if intent_class_weights is not None:
        if (intent_class_weights.shape != (logits.shape[-1],)
                or intent_class_weights.device != logits.device
                or not bool(torch.isfinite(intent_class_weights).all())
                or not bool((intent_class_weights >= 0).all())
                or not bool((intent_class_weights.sum() > 0).item())):
            raise ValueError("invalid intent class weights")
    intent_denom = intent_weights.sum().clamp_min(torch.finfo(intent_weights.dtype).tiny)
    intent = (
        F.cross_entropy(
            logits,
            target_intent,
            weight=intent_class_weights,
            reduction="none",
        )
        * intent_weights
    ).sum() / intent_denom
    position = (F.smooth_l1_loss(exposure, target_exposure, reduction="none") * weights).sum() / denom
    return {"loss": exposure_weight * position + intent_weight * intent,
            "intent_ce": intent, "exposure_huber": position,
            "effective_samples": torch.maximum(weights.sum(), intent_weights.sum()).detach(),
            "effective_exposure_samples": weights.sum().detach(),
            "effective_intent_samples": intent_weights.sum().detach()}
