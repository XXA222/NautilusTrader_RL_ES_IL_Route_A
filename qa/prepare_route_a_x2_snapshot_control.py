"""Build an isolated, hash-verified pre-fix teacher package for native pairing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = ROOT / "packages/route_a_imitation/src/rles_route_a"
TEACHER_BASE = SOURCE_PACKAGE / "teachers_1m/base.py"
CONTROL_ROOT = ROOT / ".artifacts/route-a-x2-snapshot-fix-task3/controls/pre_fix"
OVERLAY_ROOT = CONTROL_ROOT / "pythonpath"
OVERLAY_PACKAGE = OVERLAY_ROOT / "rles_route_a"
CONTROL_BASE = OVERLAY_PACKAGE / "teachers_1m/base.py"
CONTROL_SNAPSHOT = CONTROL_ROOT / "base.py"
RECEIPT = CONTROL_ROOT / "overlay-receipt.json"

PRE_FIX_SHA256 = "55b77d36116ada91fd79a89956915b7d9da8b33fb3ab63a980c18b5ac36db887"
POST_FIX_SHA256 = "740eadc7719c021e263f30e42d1d0ff2522aa45a64f34e311bf360e158d71af8"

ASSIGNMENT_BLOCK = '''        atr_signal = (
            self._atr_exit.evaluate(observation, current_exposure)
            if self._atr_exit is not None
            else None
        )
'''
OLD_EARLY_RETURN = '''        if self._atr_exit is not None:
            atr_signal = self._atr_exit.evaluate(observation, current_exposure)
            if atr_signal is not None:
                self._last_signal = atr_signal
                return advice(
                    observation,
                    self.expert_id,
                    "EXIT",
                    0.0,
                    "ATR_TRAILING_EXIT",
                    atr_signal.reason_code,
                )
'''
NEW_EXIT_RETURN_NESTED = '''            if atr_signal is not None:
                self._pending_reversal = None
                self._last_signal = atr_signal
                return advice(
                    observation,
                    self.expert_id,
                    "EXIT",
                    0.0,
                    "ATR_TRAILING_EXIT",
                    atr_signal.reason_code,
                )
'''
NEW_EXIT_RETURN_TOP_LEVEL = '''        if atr_signal is not None:
            self._pending_reversal = None
            self._last_signal = atr_signal
            return advice(
                observation,
                self.expert_id,
                "EXIT",
                0.0,
                "ATR_TRAILING_EXIT",
                atr_signal.reason_code,
            )
'''


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reconstruct_pre_fix(current: bytes) -> bytes:
    text = current.decode("utf-8")
    if text.count(ASSIGNMENT_BLOCK) != 1:
        raise ValueError("expected exactly one current ATR snapshot assignment")
    if (text.count(NEW_EXIT_RETURN_NESTED) != 2
            or text.count(NEW_EXIT_RETURN_TOP_LEVEL) != 1):
        raise ValueError("expected two nested and one top-level ATR exit-return branch")
    text = text.replace(ASSIGNMENT_BLOCK, OLD_EARLY_RETURN, 1)
    text = text.replace(NEW_EXIT_RETURN_NESTED, "", 2)
    text = text.replace(NEW_EXIT_RETURN_TOP_LEVEL, "", 1)
    return text.encode("utf-8")


def _write_receipt(data: dict[str, Any]) -> None:
    RECEIPT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CONTROL_ROOT.exists():
        raise FileExistsError(f"control overlay already exists: {CONTROL_ROOT}")
    current = TEACHER_BASE.read_bytes()
    current_sha = _sha256(current)
    if current_sha != POST_FIX_SHA256:
        raise ValueError(f"post-fix source hash changed: {current_sha}")
    pre_fix = _reconstruct_pre_fix(current)
    pre_fix_sha = _sha256(pre_fix)
    if pre_fix_sha != PRE_FIX_SHA256:
        raise ValueError(f"reconstructed pre-fix source hash differs: {pre_fix_sha}")

    CONTROL_ROOT.mkdir(parents=True)
    CONTROL_SNAPSHOT.write_bytes(pre_fix)
    shutil.copytree(
        SOURCE_PACKAGE,
        OVERLAY_PACKAGE,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    CONTROL_BASE.write_bytes(pre_fix)
    if _sha256(CONTROL_BASE.read_bytes()) != PRE_FIX_SHA256:
        raise RuntimeError("isolated control base file failed its source hash check")
    _write_receipt({
        "schema": "route-a.x2-snapshot-fix-control-overlay.v1",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "pre_fix_base_sha256": PRE_FIX_SHA256,
        "post_fix_base_sha256": POST_FIX_SHA256,
        "control_base_path": str(CONTROL_BASE.relative_to(ROOT)),
        "source_package_path": str(SOURCE_PACKAGE.relative_to(ROOT)),
        "overlay_pythonpath": str(OVERLAY_ROOT.relative_to(ROOT)),
        "method": "copy current package and reverse only the ATR early-return snapshot update; reconstructed pre-fix base hash must match the saved pre-run hash",
        "working_tree_sources_modified": False,
    })
    print(json.dumps({
        "control_base_sha256": PRE_FIX_SHA256,
        "post_fix_base_sha256": POST_FIX_SHA256,
        "overlay_pythonpath": str(OVERLAY_ROOT.relative_to(ROOT)),
        "receipt": str(RECEIPT.relative_to(ROOT)),
        "working_tree_sources_modified": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
