"""Actor-only offline policy export: JSON + no-pickle NPZ, not a training resume.

IQL/TD3+BC critics, replay buffers and optimizers are NOT exported. Record this
as a deployment/inference artifact only, with frozen encoder provenance.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
import numpy as np
import torch
from torch import nn
from rles_route_a import BASELINE_COMMIT
from .archive_safety import validate_weight_archive, finite_weight_state
from rles_route_a.training.iql_trainer import SquashedGaussian
from rles_route_a.training.offline_common import mlp


class DeterministicOfflineActor:
    def __init__(self, net: nn.Module, *, kind: str, device: str):
        self.net, self.kind, self.device = net.to(device).eval(), kind, device

    @torch.no_grad()
    def predict(self, encoded_state: torch.Tensor) -> torch.Tensor:
        s = encoded_state.to(self.device)
        if self.kind == "iql":
            if not isinstance(self.net, SquashedGaussian):
                raise TypeError("IQL actor bundle must contain a squashed Gaussian policy")
            return self.net.deterministic_action(s)
        return self.net(s)


def save_offline_actor(*, policy: nn.Module, kind: Literal["iql", "td3bc"],
                       state_dim: int, hidden_size: int, encoder_bundle_ref: str,
                       native_train_receipt: str, path: str | Path) -> Path:
    if not encoder_bundle_ref or not native_train_receipt or state_dim < 1 or hidden_size < 1:
        raise ValueError("frozen encoder and native training evidence required")
    dest = Path(path)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="route-a-offline-", dir=dest.parent) as tmp:
        source = Path(tmp)
        finite_weight_state(policy.state_dict())
        tensor_map = {k: x.detach().cpu().numpy() for k, x in policy.state_dict().items()}
        np.savez_compressed(source / "actor.npz", **tensor_map)
        raw = (source / "actor.npz").read_bytes()
        manifest = {"schema": "route_a_offline_actor_v1", "baseline": BASELINE_COMMIT,
                    "kind": kind, "action_schema": "single_exposure_v1",
                    "state_dim": state_dim, "hidden_size": hidden_size,
                    "encoder_bundle_ref": encoder_bundle_ref,
                    "native_train_receipt": native_train_receipt,
                    "actor_sha256": hashlib.sha256(raw).hexdigest(),
                    "economic_status": "UNVERIFIED",
                    "scope": "ACTOR_ONLY_NO_OPTIMIZER_OR_REPLAY"}
        (source / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.rename(source, dest)
    return dest


def load_offline_actor(path: str | Path, *, device: str = "cpu") -> tuple[DeterministicOfflineActor, dict]:
    d = Path(path)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "route_a_offline_actor_v1" or manifest.get("baseline") != BASELINE_COMMIT:
        raise ValueError("unsupported offline bundle/snapshot")
    if manifest.get("action_schema") != "single_exposure_v1" or manifest.get("scope") != "ACTOR_ONLY_NO_OPTIMIZER_OR_REPLAY":
        raise ValueError("invalid action/bundle provenance")
    kind = manifest.get("kind")
    width, hidden = manifest.get("state_dim"), manifest.get("hidden_size")
    if kind not in ("iql", "td3bc") or type(width) is not int or type(hidden) is not int or not 1 <= width <= 8192 or not 1 <= hidden <= 2048:
        raise ValueError("invalid offline actor shape/kind")
    actor = SquashedGaussian(width, hidden) if kind == "iql" else nn.Sequential(mlp(width, 1, hidden), nn.Tanh())
    path_weights = d / "actor.npz"
    if hashlib.sha256(path_weights.read_bytes()).hexdigest() != manifest.get("actor_sha256"):
        raise ValueError("offline actor bundle checksum mismatch")
    expected = actor.state_dict()
    validate_weight_archive(path_weights, {name: tuple(tensor.shape)
                                           for name, tensor in expected.items()})
    with np.load(path_weights, allow_pickle=False) as payload:
        if set(payload.files) != set(expected):
            raise ValueError("offline actor tensor shape/schema mismatch")
        weights = {}
        for name, tensor in expected.items():
            arr = payload[name]
            if arr.shape != tuple(tensor.shape) or arr.dtype.kind not in "fi" or not np.isfinite(arr).all():
                raise ValueError("offline actor contains invalid tensor")
            weights[name] = torch.tensor(arr, dtype=tensor.dtype)
    actor.load_state_dict(weights, strict=True)
    return DeterministicOfflineActor(actor, kind=kind, device=device), manifest
