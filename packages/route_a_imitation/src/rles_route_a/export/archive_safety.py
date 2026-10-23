"""Resource-limited, pickle-free NumPy weight archive validation."""
from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import numpy as np

_MAX_TOTAL_UNCOMPRESSED = 128 * 1024 * 1024
_MAX_COMPRESSED = 64 * 1024 * 1024
_MAX_TENSORS = 1024


def validate_weight_archive(path: Path, expected_shapes: dict[str, tuple[int, ...]]) -> None:
    if path.stat().st_size > _MAX_COMPRESSED:
        raise ValueError("model archive exceeds serialized weight limit")
    if len(expected_shapes) > _MAX_TENSORS:
        raise ValueError("model architecture exceeds weight count cap")
    try:
        with ZipFile(path, "r") as archive:
            names = archive.namelist()
            expected = {key + ".npy" for key in expected_shapes}
            if set(names) != expected or len(names) != len(expected):
                raise ValueError("unexpected/missing model weight member")
            total = 0
            for member in archive.infolist():
                if member.file_size > _MAX_TOTAL_UNCOMPRESSED or member.compress_size > _MAX_COMPRESSED:
                    raise ValueError("model tensor exceeds size cap")
                total += member.file_size
                if total > _MAX_TOTAL_UNCOMPRESSED:
                    raise ValueError("model archive uncompressed payload exceeds size cap")
    except BadZipFile as exc:
        raise ValueError("model weights are not a valid NPZ archive") from exc


def finite_weight_state(state: dict[str, object]) -> None:
    for name, tensor in state.items():
        arr = tensor.detach().cpu().numpy()  # type: ignore[attr-defined]
        if arr.dtype.kind not in "fi" or not np.isfinite(arr).all():
            raise ValueError(f"cannot serialize nonfinite model tensor: {name}")
