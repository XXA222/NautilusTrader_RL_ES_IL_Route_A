"""Stable identities shared by student screening and WFO promotion."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from typing import Any, cast

import numpy as np


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def bc_actor_sha256(model: Any) -> str:
    """Hash every named tensor in a frozen BC actor independent of archive format."""
    state = getattr(model, "state_dict", None)
    if not callable(state):
        raise TypeError("BC actor must expose a state_dict")
    tensors = state()
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("BC actor state must be a nonempty tensor mapping")
    digest = sha256(b"rles.route_a.bc_actor_state.v1\0")
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(name, str) or not name:
            raise ValueError("BC actor state contains an invalid tensor name")
        value = tensor.detach().cpu().contiguous().numpy()
        if value.dtype.kind not in "fi" or not value.size:
            raise ValueError("BC actor state contains an unsupported tensor")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError("BC actor state contains a nonfinite tensor")
        header = json.dumps(
            {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        payload = value.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def bc_observation_schema_sha256(config: Any) -> str:
    """Bind sequence input identity, ordering, context, and discontinuity policy."""
    feature_names = getattr(config, "feature_names", None)
    context_bars = getattr(config, "context_bars", None)
    max_gap_ns = getattr(config, "max_gap_ns", None)
    if (
        not isinstance(feature_names, tuple)
        or not feature_names
        or any(not isinstance(name, str) or not name for name in feature_names)
        or type(context_bars) is not int
        or context_bars < 1
        or (max_gap_ns is not None and (type(max_gap_ns) is not int or max_gap_ns < 1))
    ):
        raise ValueError("BC sequence observation schema is invalid")
    return _canonical_sha256(
        {
            "schema": "rles.route_a.bc_sequence_observation.v1",
            "feature_names": list(feature_names),
            "context_bars": context_bars,
            "max_gap_ns": max_gap_ns,
        }
    )


def bc_action_schema_sha256() -> str:
    """Identity of the single target-exposure action accepted by native Route A."""
    return _canonical_sha256(
        {
            "schema": "single_exposure_v1",
            "domain": [-1.0, 1.0],
            "semantics": "target_net_exposure",
        }
    )


def bc_student_recipe_sha256(config: Any) -> str:
    """Hash the model and optimization recipe used to create a Route A student."""
    if not is_dataclass(config):
        raise TypeError("BC student recipe must be a dataclass config")
    return _canonical_sha256(
        {
            "schema": "rles.route_a.bc_student_recipe.v1",
            "model": "BCGRUPolicy",
            "config": asdict(cast(Any, config)),
        }
    )
