from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


_GENERATOR = Path(__file__).resolve().parents[1] / "qa" / "source_manifest.py"


def _generate(root: Path, output: Path) -> dict:
    subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(root), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_source_identity_tracks_workspace_inputs_and_ignores_generated_reports(tmp_path: Path) -> None:
    (tmp_path / "packages" / "demo").mkdir(parents=True)
    (tmp_path / "packages" / "demo" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text("def test_demo(): pass\n", encoding="utf-8")
    (tmp_path / "qa" / "reports").mkdir(parents=True)
    (tmp_path / "qa" / "research_trial_inventory.py").write_text(
        "VALUE = 'trial inventory'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "train_signal_baselines.py").write_text(
        "VALUE = 'signal baseline runner'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_stage58_train_coverage.py").write_text(
        "VALUE = 'train coverage'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_stage59_train_diagnostics.py").write_text(
        "VALUE = 'stage59 analysis'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "preregister_stage59_train_diagnostics.py").write_text(
        "VALUE = 'stage59 preregistration'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_stage60_fixed_regimes.py").write_text(
        "VALUE = 'stage60 fixed regime analysis'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "preregister_stage60_fixed_regimes.py").write_text(
        "VALUE = 'stage60 fixed regime preregistration'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_m33_action_value.py").write_text(
        "VALUE = 'M3.3 action value analysis'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "preregister_m33_action_value.py").write_text(
        "VALUE = 'M3.3 action value preregistration'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "m13_stage_contribution_20261017.py").write_text(
        "VALUE = 'M1.3 stage contribution runner'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_m13_stage_contribution.py").write_text(
        "VALUE = 'M1.3 stage contribution independent analysis'\n", encoding="utf-8",
    )
    manifest_path = tmp_path / "qa" / "reports" / "source_manifest_current.json"

    first = _generate(tmp_path, manifest_path)
    second = _generate(tmp_path, manifest_path)
    assert first["source_identity_sha256"] == second["source_identity_sha256"]
    assert [item["path"] for item in first["files"]] == [
        "packages/demo/module.py", "qa/analyze_m13_stage_contribution.py",
        "qa/analyze_m33_action_value.py",
        "qa/analyze_stage58_train_coverage.py",
        "qa/analyze_stage59_train_diagnostics.py",
        "qa/analyze_stage60_fixed_regimes.py",
        "qa/m13_stage_contribution_20261017.py",
        "qa/preregister_m33_action_value.py",
        "qa/preregister_stage59_train_diagnostics.py",
        "qa/preregister_stage60_fixed_regimes.py",
        "qa/research_trial_inventory.py", "qa/train_signal_baselines.py",
        "tests/test_demo.py",
    ]

    (tmp_path / "packages" / "demo" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = _generate(tmp_path, manifest_path)
    assert changed["source_identity_sha256"] != first["source_identity_sha256"]


def test_source_identity_tracks_isolated_workers_but_excludes_their_runtime_outputs(tmp_path: Path) -> None:
    worker = tmp_path / "research_workers" / "offline"
    (worker / "src" / "worker").mkdir(parents=True)
    (worker / "src" / "worker" / "train.py").write_text("VALUE = 1\n", encoding="utf-8")
    (worker / ".venv" / "Lib").mkdir(parents=True)
    (worker / ".venv" / "Lib" / "runtime.pyc").write_bytes(b"runtime")
    (worker / ".tmp" / "toy-run").mkdir(parents=True)
    (worker / ".tmp" / "toy-run" / "metrics.json").write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "qa" / "reports" / "source_manifest_current.json"

    result = _generate(tmp_path, manifest_path)
    paths = {item["path"] for item in result["files"]}
    assert "research_workers/offline/src/worker/train.py" in paths
    assert not any("/.venv/" in f"/{path}/" or "/.tmp/" in f"/{path}/" for path in paths)
    assert "research_workers" in result["scope"]["included"]
    assert ".venv*" in result["scope"]["excluded_generated"]
