"""Portable, non-pickle persistence of frozen entry models."""
from __future__ import annotations

from dataclasses import fields
import json
from math import isfinite
from pathlib import Path

import numpy as np

from .entry_meta_policy import FrozenEntryMetaPolicy

SCHEMA = "rles.route-a.frozen-entry-meta-policy.v1"
_ARRAYS = ("mean", "scale", "lower", "upper", "logistic", "value_weights")


def policy_document(policy: FrozenEntryMetaPolicy) -> dict:
    if not isinstance(policy, FrozenEntryMetaPolicy):
        raise TypeError("frozen entry policy required")
    result = {}
    for field in fields(policy):
        value = getattr(policy, field.name)
        result[field.name] = value.tolist() if isinstance(value, np.ndarray) else value
    # Validate the exact object that will be written, including dimensions and economics.
    policy_from_document({"schema": SCHEMA, "policy": result})
    return {"schema": SCHEMA, "policy": result}


def policy_from_document(doc: dict) -> FrozenEntryMetaPolicy:
    if (not isinstance(doc, dict) or set(doc) != {"schema", "policy"}
            or doc["schema"] != SCHEMA or not isinstance(doc["policy"], dict)
            or set(doc["policy"]) != {field.name for field in fields(FrozenEntryMetaPolicy)}):
        raise ValueError("invalid frozen policy schema")
    data = dict(doc["policy"])
    names = data["feature_names"]
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("invalid frozen feature names")
    data["feature_names"] = tuple(names)
    size = len(names) + 1  # includes side
    for name in _ARRAYS:
        try:
            values = np.asarray(data[name], dtype=np.float64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"invalid frozen {name}") from exc
        required = size + 1 if name in ("logistic", "value_weights") else size
        if values.shape != (required,) or not np.isfinite(values).all():
            raise ValueError(f"invalid frozen {name} shape or values")
        values = np.frombuffer(values.tobytes(), dtype=np.float64)
        data[name] = values
    if np.any(data["scale"] <= 0) or np.any(data["lower"] > data["upper"]):
        raise ValueError("invalid feature support or scaling")
    calibrator = data["calibrator"]
    if (not isinstance(calibrator, (tuple, list)) or len(calibrator) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
                   for v in calibrator)):
        raise ValueError("invalid probability calibrator")
    data["calibrator"] = tuple(calibrator)
    if (any(not isinstance(data[name], str) or not data[name].strip()
            for name in ("source_id", "fold_id", "feature_spec_hash", "return_basis_id"))
            or len(data["feature_spec_hash"]) != 64
            or any(c not in "0123456789abcdef" for c in data["feature_spec_hash"])
            or data["return_basis_id"] not in ("LOG_REFERENCE_ENTRY_NOTIONAL_BPS",
                                                "LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS")):
        raise ValueError("invalid source or return identity")
    recipe = data["label_recipe"]
    if (not isinstance(recipe, (tuple, list)) or len(recipe) != 3
            or any(not isinstance(value, str) or not value.strip() for value in recipe)):
        raise ValueError("invalid frozen label recipe")
    data["label_recipe"] = tuple(recipe)
    if (type(data["available_after_ns"]) is not int or data["available_after_ns"] <= 0
            or type(data["fit_count"]) is not int or data["fit_count"] < 4
            or type(data["calibration_count"]) is not int or data["calibration_count"] < 4):
        raise ValueError("invalid training sample counts or availability clock")
    for name in ("margin_bps", "calibration_mae_bps", "calibration_brier"):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f"invalid frozen {name}")
    if data["calibration_brier"] > 1:
        raise ValueError("invalid calibration Brier score")
    fee, slip, cost = (data["fee_bps_per_leg"], data["slippage_bps_per_leg"], data["cost_bps"])
    for value in (fee, slip, cost):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not isfinite(value) or value < 0):
            raise ValueError("invalid frozen transaction costs")
    if (fee is None) != (slip is None) or (fee is not None and
            (cost is None or abs(cost - 2 * (fee + slip)) > 1e-9)):
        raise ValueError("inconsistent frozen per-leg transaction costs")
    return FrozenEntryMetaPolicy(**data)


def save_frozen_entry_policy(policy: FrozenEntryMetaPolicy, path: str | Path) -> None:
    encoded = json.dumps(policy_document(policy), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as output:
        output.write(encoded)


def load_frozen_entry_policy(path: str | Path) -> FrozenEntryMetaPolicy:
    with Path(path).open(encoding="utf-8") as source:
        return policy_from_document(json.load(source))
