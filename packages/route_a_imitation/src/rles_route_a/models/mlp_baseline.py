"""Simple last-observation MLP baseline to compare with GRU sequence model."""
from __future__ import annotations
import torch
from torch import nn
from rles_route_a.contracts import INTENTS


class BCMLPBaseline(nn.Module):
    def __init__(self, feature_names: tuple[str, ...], hidden: int = 64):
        super().__init__()
        self.feature_names = feature_names
        num_features = len(feature_names)
        self.layers = nn.Sequential(nn.Linear(num_features, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU())
        self.intent_head = nn.Linear(hidden, len(INTENTS))
        self.exposure_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.layers[0].in_features:
            raise ValueError("expected [B,T,F] for MLP baseline")
        h = self.layers(x[:, -1, :])
        return self.intent_head(h), self.exposure_head(h).tanh().squeeze(-1)
