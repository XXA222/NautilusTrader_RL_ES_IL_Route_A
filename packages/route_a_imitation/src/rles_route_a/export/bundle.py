"""No-pickle model NPZ + strict JSON manifest, with frozen baseline/feature order."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from rles_route_a import BASELINE_COMMIT
from rles_route_a.config import BCConfig
from rles_route_a.models.bc import BCGRUPolicy
from .archive_safety import validate_weight_archive, finite_weight_state


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_bc_bundle(model: BCGRUPolicy, cfg: BCConfig, path: str | Path) -> Path:
    """Save only float/int tensor weights; caller controls trusted destination."""
    dest = Path(path)
    if dest.exists():
        raise FileExistsError(dest)
    if tuple(cfg.feature_names) != tuple(model.encoder.feature_names):
        raise ValueError("configuration and model feature order disagree")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write both assets together; atomic directory move on same filesystem.
    with TemporaryDirectory(prefix="route-a-", dir=dest.parent) as tmp:
        temp = Path(tmp)
        finite_weight_state(model.state_dict())
        state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        np.savez_compressed(temp / "weights.npz", **state)
        manifest = {
            "schema": "route_a_bc_bundle_v1",
            "baseline_commit": BASELINE_COMMIT,
            "action_schema": "single_exposure_v1",
            "model": "BCGRUPolicy",
            "config": asdict(cfg),
            "weights_sha256": _digest((temp / "weights.npz").read_bytes()),
            "economics": "UNVERIFIED",
        }
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                      sort_keys=True, indent=2), encoding="utf-8")
        # Directory must not exist; prevents accidental replacement of past evidence.
        os.rename(temp, dest)
    return dest


def load_bc_bundle(path: str | Path, *, device: str = "cpu") -> tuple[BCGRUPolicy, BCConfig]:
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != {"schema", "baseline_commit", "action_schema", "model", "config", "weights_sha256", "economics"}:
        raise ValueError("unsupported/altered bundle manifest")
    if (manifest["schema"] != "route_a_bc_bundle_v1"
            or manifest["baseline_commit"] != BASELINE_COMMIT
            or manifest["action_schema"] != "single_exposure_v1"
            or manifest["model"] != "BCGRUPolicy"):
        raise ValueError("bundle incompatible with the frozen snapshot/actor")
    content = (directory / "weights.npz").read_bytes()
    if _digest(content) != manifest["weights_sha256"]:
        raise ValueError("model weights integrity mismatch")
    cfg = BCConfig.parse(manifest["config"])
    if len(cfg.feature_names) > 2048 or cfg.hidden_size > 2048 or cfg.layers > 8:
        raise ValueError("unreasonably large/unknown model configuration")
    model = BCGRUPolicy(cfg.feature_names, cfg.hidden_size, cfg.layers)
    expected = model.state_dict()
    validate_weight_archive(directory / "weights.npz",
                            {name: tuple(value.shape) for name, value in expected.items()})
    with np.load(directory / "weights.npz", allow_pickle=False) as weights:
        if set(weights.files) != set(expected):
            raise ValueError("unexpected/missing model tensor")
        state = {}
        for key, default in expected.items():
            arr = weights[key]
            if arr.shape != tuple(default.shape) or arr.dtype.kind not in "fi" or not np.isfinite(arr).all():
                raise ValueError(f"invalid model tensor: {key}")
            state[key] = torch.tensor(arr, dtype=default.dtype)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cfg
