"""Run same-size fixed LONG/SHORT controls on the consumed S06 native windows."""
# mypy: disable-error-code=import-untyped
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

import run_route_a_1m_teacher_native as native  # noqa: E402
from rles_route_a.teachers_1m.base import BaseTeacher1m  # noqa: E402
from rles_route_a.teachers_1m.contracts import TeacherRecipe1m, TeacherSignal  # noqa: E402

OUTPUT = ROOT / ".artifacts/route-a-fixed-controls-task4"
FEATURE_RECIPE = ROOT / (
    ".artifacts/route-a-roc-x2-costaware-task3/"
    "feature-recipe-27f03e698a58dde7aa2c09cf161b83a491a20e0de167608f482d6073aa67818e.json"
)
WINDOWS = ("inner-screen-1", "inner-screen-2", "inner-screen-3")
SCENARIOS = ("base", "stress")
SIDES: tuple[tuple[str, Literal[-1, 1]], ...] = (
    ("FIXED_LONG_0.3", 1),
    ("FIXED_SHORT_0.3", -1),
)


class FixedSideControlTeacher(BaseTeacher1m):
    """Hold one fixed 0.3 target side; native risk and settlement stay enabled."""

    required_fields = ("bar_close_1m",)

    def __init__(self, control_id: str, side: Literal[-1, 1]) -> None:
        if control_id not in {"FIXED_LONG_0.3", "FIXED_SHORT_0.3"}:
            raise ValueError("unknown fixed-side control")
        if (control_id == "FIXED_LONG_0.3") != (side == 1):
            raise ValueError("fixed-side control ID and direction disagree")
        self._side = side
        direction = "LONG" if side > 0 else "SHORT"
        recipe = TeacherRecipe1m(
            teacher_id=control_id,
            indicator_recipe_id=f"route_a.control.{direction.lower()}.v1",
            parameters=(("control_target_side", side),),
            warmup_bars=1,
            entry_rule_id=f"FIXED_{direction}_CONTROL_V1",
            indicator_exit_rule_id="NO_SIGNAL_EXIT_V1",
            htf_mode="FEATURE_ONLY",
            target_exposure=0.3,
            cooldown_policy="NEW_EVENT_ONLY_NO_TIME_COOLDOWN",
            order_policy="MARKET_NATIVE_L3_NETTING",
            risk_recipe_id="NATIVE_ROUTE_A_HARD_SOFT_RISK",
            scenario_cost_ids=("base", "stress"),
        )
        super().__init__(recipe)

    def entry_signal(
        self, current: dict[str, float], previous: dict[str, float],
    ) -> TeacherSignal:
        del current, previous
        if self._side > 0:
            return TeacherSignal("LONG_ENTRY", 1, "FIXED_LONG_CONTROL")
        return TeacherSignal("SHORT_ENTRY", -1, "FIXED_SHORT_CONTROL")

    def exit_signal(
        self,
        current_exposure: float,
        current: dict[str, float],
        previous: dict[str, float],
    ) -> TeacherSignal | None:
        del current_exposure, current, previous
        return None


def _existing_report(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    report_path = path / "native-run.json"
    if not path.exists():
        return None
    if not report_path.is_file():
        raise RuntimeError(f"partial immutable control run exists: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = (
        report.get("protocol_id") == expected["protocol_id"],
        report.get("teacher") == expected["teacher"],
        report.get("execution", {}).get("scenario") == expected["scenario"],
        report.get("source", {}).get("window_id") == expected["window_id"],
        report.get("source", {}).get("snapshot_hash") == expected["snapshot_hash"],
        report.get("source", {}).get("marker_sha256") == expected["marker_sha256"],
        report.get("decisions", {}).get("observed")
        == report.get("decisions", {}).get("m22_verified"),
        report.get("account", {}).get("liquidations") == 0,
    )
    if not all(checks):
        raise RuntimeError(f"existing fixed control run does not match: {path}")
    return report


def run() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    doc, prereg_sha = native.verify_preregistration(
        native.ROOT_DEFAULT_PREREG, feature_recipe_override=FEATURE_RECIPE,
    )
    native.FEATURE_RECIPE = FEATURE_RECIPE
    scaler, scaler_receipt = native.fit_development_scaler(
        doc, OUTPUT / "preparation",
    )
    results: list[dict[str, Any]] = []
    for window_id in WINDOWS:
        for scenario in SCENARIOS:
            prepared = native.prepare_window_factory(
                doc, OUTPUT / "preparation", window_id=window_id,
                scenario=scenario, scaler=scaler,
            )
            config, factory = prepared
            del config
            start_ns, end_ns, block_id = native._window_bounds(doc, window_id)
            for control_id, side in SIDES:
                teacher = FixedSideControlTeacher(control_id, side)
                run_dir = OUTPUT / window_id / scenario / control_id
                expected = {
                    "protocol_id": doc["protocol_id"],
                    "teacher": control_id,
                    "scenario": scenario,
                    "window_id": window_id,
                    "snapshot_hash": doc["source"]["snapshot_hash"],
                    "marker_sha256": doc["source"]["marker_sha256"],
                }
                report = _existing_report(run_dir, expected)
                if report is None:
                    run_dir.mkdir(parents=True, exist_ok=False)
                    report = native._run_native_window(
                        doc, prereg_sha, run_dir,
                        teacher_id=control_id,
                        exit_variant="X1",
                        scenario=scenario,
                        window_id=window_id,
                        prepared=prepared,
                        bounds_override=(start_ns, end_ns, block_id),
                        teacher_override=teacher,
                    )
                results.append({
                    "window_id": window_id,
                    "scenario": scenario,
                    "control_id": control_id,
                    "native_run": str(run_dir.relative_to(ROOT) / "native-run.json"),
                    "account": report["account"],
                    "decisions": report["decisions"],
                })
            factory.snapshots.clear()
            factory.teacher_snapshots.clear()
            factory.fill_references.clear()
    summary = {
        "schema": "rles.route-a.fixed-side-controls-s06.v1",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "economic_status": "SCENARIO_ONLY",
        "protocol_id": doc["protocol_id"],
        "preregistration_sha256": prereg_sha,
        "source_snapshot_hash": doc["source"]["snapshot_hash"],
        "feature_recipe_sha256": native._sha256(FEATURE_RECIPE),
        "risk_exposure_target": 0.3,
        "scaler_hash": scaler.hash,
        "scaler_fit_window_ns": scaler_receipt["fit_window_ns"],
        "results": results,
    }
    destination = OUTPUT / "fixed_side_controls.json"
    destination.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2, allow_nan=False))
