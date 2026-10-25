"""Small multi-timescale GRU with explicit left-padding / episode boundaries."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def feature_groups(names: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    groups: dict[str, list[int]] = {"minute": [], "hour": [], "day": [], "account": []}
    for i, k in enumerate(names):
        if k.startswith("account."):
            groups["account"].append(i)
        elif k.startswith("market."):
            if "1d" in k:
                groups["day"].append(i)
            elif "1h" in k:
                groups["hour"].append(i)
            else:
                groups["minute"].append(i)
        else:
            raise ValueError("expected market./account. frozen feature prefixes")
    return {k: tuple(v) for k, v in groups.items() if v}


class MultiScaleGRU(nn.Module):
    def __init__(self, names: tuple[str, ...], hidden_size: int = 128, layers: int = 2):
        super().__init__()
        self.feature_names = names
        self.groups = feature_groups(names)
        dims = {"minute": 64, "hour": 32, "day": 16, "account": 32}
        self.projections = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(len(indices), dims[k]), nn.LayerNorm(dims[k]), nn.Tanh())
            for k, indices in self.groups.items()
        })
        self.gru = nn.GRU(sum(dims[k] for k in self.groups), hidden_size,
                          num_layers=layers, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != len(self.feature_names):
            raise ValueError("expected [B,T,F] in exact frozen feature order")
        if lengths.ndim != 1 or lengths.shape[0] != x.shape[0] or x.shape[1] < 1:
            raise ValueError("invalid sequence lengths")
        if torch.any(lengths < 1) or torch.any(lengths > x.shape[1]):
            raise ValueError("sequence length outside context")
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite observations")
        b, t, _ = x.shape
        # Input is LEFT padded; pack_padded_sequence expects RIGHT padding.
        idx = torch.arange(t, device=x.device)[None, :].expand(b, -1)
        src = (t - lengths.to(device=x.device)[:, None] + idx) % t
        ordered = x.gather(1, src[:, :, None].expand(-1, -1, x.shape[2]))
        parts = [module(ordered[:, :, indices]) for k, module in self.projections.items()
                 for indices in (self.groups[k],)]
        projected = torch.cat(parts, dim=-1)
        # No training labels can be drawn from left-padding; true episode sequences
        # are gathered into contiguous prefixes and masked by pack_padded_sequence.
        packed = pack_padded_sequence(projected, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return hidden[-1]
