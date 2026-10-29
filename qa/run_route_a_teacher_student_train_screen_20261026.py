"""Run the in-scope fresh-Train teacher screen through native execution.

The frozen protocol defines T01-T08, but the active project scope evaluates
T01-T04 plus the frozen cost-aware ROC X1/X7 candidates. Controls, scenarios,
seeds, source scope, and the acceptance rule remain protocol-bound. This runner
does not fit on a scored window and never starts student training.
"""
# mypy: disable-error-code=import-untyped
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Literal, cast
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

import run_route_a_1m_teacher_native as native  # noqa: E402
import run_route_a_train_student_screen_20261020 as harness  # noqa: E402
from route_a_run_identity import RunIdentityV1, canonical_sha, require_identity  # noqa: E402
from rles_orchestrator.research_access import (  # noqa: E402
    ACTIVE_OWNER,
    market_key,
    reject_protected_monthly_archive_source,
)
from rles_route_a.teachers_1m.base import BaseTeacher1m  # noqa: E402
from rles_route_a.teachers_1m.contracts import TeacherRecipe1m, TeacherSignal  # noqa: E402
from rles_route_a.teachers_1m.registry import (  # noqa: E402
    ROC_COST_HYSTERESIS_TEACHER_ID,
    build_teacher_1m,
)

PREREG = ROOT / "configs/route_a_teacher_student_train_prereg_20261026.json"
ERRATA = ROOT / "configs/route_a_teacher_student_train_prereg_20261026_errata.json"
SOURCE_ERRATA = ROOT / "configs/route_a_teacher_student_train_prereg_20261026_source_errata.json"
OUTPUT_ROOT = ROOT / ".artifacts/route-a-teacher-student-fresh-train-20261026/teacher-screen"
SCENARIO_RULES = ROOT / ".artifacts/audit-development-20261016/data/scenario-rules.json"
EXECUTION_TEMPLATE = ROOT / "configs/ethusdt_20240101_real.json"
FEATURE_RECIPE = ROOT / "configs/route_a_1m_teacher_library_v1.json"
INSTRUMENT = "ETHUSDT-PERP.SIM"
MINUTE_NS = 60_000_000_000
SOURCE_MODE = "whole_snapshot_isolated_v1"
FEATURE_PROFILE = "causal_mtf_v4"
SCENARIO_ONLY_ECONOMIC_STATUS = "SCENARIO_ONLY"
DEVELOPMENT_QUALIFICATION = "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2"
NativeExitVariant = Literal["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
EmaReversalMode = Literal["EXIT_ONLY_V1", "REVERSE_AFTER_FLAT_V1"]
IN_SCOPE_TEACHER_IDS = frozenset({
    "T01_EMA_9_21",
    "T02_ROC_20_60",
    ROC_COST_HYSTERESIS_TEACHER_ID,
    "T03_DONCHIAN_20",
    "T04_ZSCORE_60",
})


def _digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _ns(value: str) -> int:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("pre-registered timestamps must include a timezone")
    delta = stamp.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def _protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    prereg = _read_json(PREREG)
    errata = _read_json(ERRATA)
    source_errata = _read_json(SOURCE_ERRATA)
    execution_scope = source_errata.get("teacher_execution_scope", {})
    registered_teacher_ids = {
        row["teacher_id"] for row in errata["corrected_candidate_matrix"]
    }
    expected_additional_candidates = [{
        "teacher_id": ROC_COST_HYSTERESIS_TEACHER_ID,
        "reversal_mode": "EXIT_ONLY_V1",
        "exit_variant": "X1",
    }, {
        "teacher_id": ROC_COST_HYSTERESIS_TEACHER_ID,
        "reversal_mode": "EXIT_ONLY_V1",
        "exit_variant": "X7",
    }]
    additional_candidates = execution_scope.get("additional_frozen_candidates", [])
    if (set(execution_scope.get("included_teacher_ids", [])) != IN_SCOPE_TEACHER_IDS
            or set(execution_scope.get("excluded_teacher_ids", []))
            != registered_teacher_ids - IN_SCOPE_TEACHER_IDS
            or additional_candidates != expected_additional_candidates
            or execution_scope.get("scored_teacher_outcomes_observed") is not False):
        raise RuntimeError("teacher execution scope differs from the active task or candidate registry")
    base_costs = source_errata["cost_scenarios"]["base"]
    base_round_trip_cost_bps = 2 * (
        float(base_costs["taker_fee_bps_per_leg"])
        + float(base_costs["slippage_bps_per_leg"])
    )
    build_teacher_1m(
        ROC_COST_HYSTERESIS_TEACHER_ID,
        exit_variant="X1",
        reversal_mode="EXIT_ONLY_V1",
        roc_round_trip_cost_bps=base_round_trip_cost_bps,
    )
    x7 = build_teacher_1m(
        ROC_COST_HYSTERESIS_TEACHER_ID,
        exit_variant="X7",
        reversal_mode="EXIT_ONLY_V1",
        roc_round_trip_cost_bps=base_round_trip_cost_bps,
    )
    if dict(x7.recipe.parameters).get("entry_ema_cross_age_min_bars") != 21:
        raise RuntimeError("frozen X7 EMA cross-age minimum changed")
    if dict(x7.recipe.parameters).get("entry_ema_cross_age_max_bars") != 60:
        raise RuntimeError("frozen X7 EMA cross-age maximum changed")
    prereg_sha = _digest(PREREG)
    if PREREG.with_suffix(PREREG.suffix + ".sha256").read_text().strip() != prereg_sha:
        raise RuntimeError("fresh-Train pre-registration SHA mismatch")
    if ERRATA.with_suffix(ERRATA.suffix + ".sha256").read_text().strip() != _digest(ERRATA):
        raise RuntimeError("teacher errata SHA mismatch")
    if SOURCE_ERRATA.with_suffix(SOURCE_ERRATA.suffix + ".sha256").read_text().strip() != _digest(SOURCE_ERRATA):
        raise RuntimeError("source errata SHA mismatch")
    if (prereg.get("protocol_id") != "route-a-teacher-student-train-20261023-20261130-v1"
            or errata.get("supersedes_prereg_sha256") != prereg_sha
            or source_errata.get("supersedes_prereg_sha256") != prereg_sha
            or source_errata.get("supersedes_previous_errata_sha256") != _digest(ERRATA)):
        raise RuntimeError("fresh-Train protocol and errata chain do not match")
    return prereg, errata, source_errata, prereg_sha


def _candidate_id(candidate: dict[str, Any]) -> str:
    mode = candidate["reversal_mode"]
    suffix = "" if mode == "EXIT_ONLY_V1" else f"--{mode}"
    exit_variant = candidate.get("exit_variant", "X1")
    if exit_variant != "X1":
        suffix += f"--{exit_variant}"
    return f"{candidate['teacher_id']}{suffix}"


def _candidate_rows_in_scope(
    errata: dict[str, Any], source_errata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = errata["corrected_candidate_matrix"]
    selected = [row for row in rows if row["teacher_id"] in IN_SCOPE_TEACHER_IDS]
    if source_errata is not None:
        additional = source_errata.get("teacher_execution_scope", {}).get(
            "additional_frozen_candidates", [],
        )
        selected.extend(
            row for row in additional
            if row["teacher_id"] in IN_SCOPE_TEACHER_IDS
        )
    if not selected:
        raise ValueError("frozen candidate matrix has no active-goal teachers")
    return selected


def _teacher_recipe_document(
    teacher: BaseTeacher1m, exit_variant: NativeExitVariant,
) -> dict[str, Any]:
    return {
        "indicator_recipe_id": teacher.recipe.indicator_recipe_id,
        "parameters": [list(parameter) for parameter in teacher.recipe.parameters],
        "entry_rule_id": teacher.recipe.entry_rule_id,
        "exit_variant": exit_variant,
        "exit_policy_id": teacher.recipe.indicator_exit_rule_id,
        "ema_reversal_mode": getattr(teacher, "reversal_mode", None),
        "higher_timeframe_mode": teacher.recipe.htf_mode,
        "target_exposure": teacher.recipe.target_exposure,
    }


class FixedExposureControlTeacher(BaseTeacher1m):
    """Frozen no-trade/constant-side account control for native pairing."""

    required_fields = ("bar_close_1m",)

    def __init__(self, control_id: str, target: Literal[-1, 0, 1]) -> None:
        if control_id not in ("FLAT", "FIXED_LONG_0.3", "FIXED_SHORT_0.3"):
            raise ValueError("unknown frozen teacher screen control")
        if (control_id == "FLAT") != (target == 0):
            raise ValueError("control identity and target side disagree")
        self._control_target = target
        side = "FLAT" if target == 0 else "LONG" if target > 0 else "SHORT"
        recipe = TeacherRecipe1m(
            teacher_id=control_id,
            indicator_recipe_id=f"route_a.control.{side.lower()}.v1",
            parameters=(("control_target_side", target),),
            warmup_bars=1,
            entry_rule_id=f"FIXED_{side}_CONTROL_V1",
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
        self, current: Mapping[str, float], previous: Mapping[str, float],
    ) -> TeacherSignal:
        del current, previous
        if self._control_target > 0:
            return TeacherSignal("LONG_ENTRY", 1, "FIXED_LONG_CONTROL")
        if self._control_target < 0:
            return TeacherSignal("SHORT_ENTRY", -1, "FIXED_SHORT_CONTROL")
        return TeacherSignal("NONE", 0, "FIXED_FLAT_CONTROL")

    def exit_signal(
        self,
        current_exposure: float,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> TeacherSignal | None:
        del current_exposure, current, previous
        return None


def _control(control_id: str) -> FixedExposureControlTeacher:
    target: Literal[-1, 0, 1] = {
        "FLAT": 0,
        "FIXED_LONG_0.3": 1,
        "FIXED_SHORT_0.3": -1,
    }[control_id]  # type: ignore[assignment]
    return FixedExposureControlTeacher(control_id, target)


def _verified_source(
    prereg: dict[str, Any], source_errata: dict[str, Any], prereg_sha: str,
    import_spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    registered = source_errata["import_invocation"]
    if (registered["development_protocol_sha256"] != prereg_sha
            or registered["development_access_key"]
            != "access:" + prereg["protocol_id"]):
        raise RuntimeError("Train import is not bound to the pre-registered access row")
    relative_root = Path(import_spec["output"])
    if relative_root.is_absolute() or ".." in relative_root.parts:
        raise RuntimeError("staged Train import output escapes the canonical project")
    import_root = (ROOT / relative_root).resolve()
    try:
        import_root.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("staged Train import output escapes the canonical project") from exc
    market = import_root / "market.parquet"
    marker = import_root / "market.parquet.COMPLETED.json"
    manifest_path = import_root / "import.json"
    if not all(path.is_file() for path in (market, marker, manifest_path)):
        raise FileNotFoundError(
            f"complete registered staged Train import is not present: {import_spec['output']}"
        )
    manifest = _read_json(manifest_path)
    marker_doc = _read_json(marker)
    reject_protected_monthly_archive_source(market, market_key(INSTRUMENT))
    scope = prereg["scope"]
    start_ns = _ns(scope["source_start_utc_inclusive"])
    end_ns = _ns(scope["end_utc_exclusive"])
    expected_first = datetime.fromisoformat(import_spec["start_day"]).replace(
        tzinfo=timezone.utc,
    )
    expected_after = datetime.fromisoformat(import_spec["end_day_inclusive"]).replace(
        tzinfo=timezone.utc,
    ) + timedelta(days=1)
    first_ns = _ns(expected_first.isoformat().replace("+00:00", "Z"))
    after_ns = _ns(expected_after.isoformat().replace("+00:00", "Z"))
    if (manifest.get("schema") != "rles.binance-archive-range.v1"
            or manifest.get("start_day") != import_spec["start_day"]
            or manifest.get("end_day") != import_spec["end_day_inclusive"]
            or import_spec["funding_api_months"] != sorted(set(
                import_spec["funding_api_months"]
            ))
            or not start_ns <= int(manifest.get("start_ns", -1)) < int(manifest.get("end_ns", -1)) <= end_ns
            or int(manifest.get("start_ns", -1)) != first_ns + MINUTE_NS
            or int(manifest.get("end_ns", -1)) != after_ns
            or manifest.get("event_count") != marker_doc.get("events")
            or marker_doc.get("schema_version") not in {"0.4.0", "0.4.1"}
            or marker_doc.get("file_sha256") != _digest(market)):
        raise RuntimeError("staged Train import scope or completion marker does not match the frozen source")
    row_groups = marker_doc.get("row_groups")
    if (marker_doc.get("schema_version") == "0.4.1"
            and isinstance(row_groups, list) and row_groups):
        read_scope_start_ns = min(
            min(row["min_ts_event_ns"], row["min_available_at_ns"])
            for row in row_groups
        )
        read_scope_end_ns = max(
            1 + max(
                max(row["max_ts_event_ns"], row["max_available_at_ns"])
                for row in row_groups
            ),
            int(manifest["end_ns"]),
        )
    else:
        read_scope_start_ns = int(manifest["start_ns"])
        read_scope_end_ns = int(manifest["end_ns"])
    if not start_ns <= read_scope_start_ns < read_scope_end_ns <= end_ns:
        raise RuntimeError("staged row-group read scope escapes the registered Train range")
    return {
        "market_path": str(market.relative_to(ROOT)),
        "marker_path": str(marker.relative_to(ROOT)),
        "import_manifest_path": str(manifest_path.relative_to(ROOT)),
        "marker_sha256": _digest(marker),
        "snapshot_hash": marker_doc["snapshot_hash"],
        "events": marker_doc["events"],
        "read_scope_start_ns": read_scope_start_ns,
        "read_scope_end_ns": read_scope_end_ns,
        "read_mode": SOURCE_MODE,
    }, manifest


def _run_doc(
    prereg: dict[str, Any], source: dict[str, Any], source_errata: dict[str, Any],
) -> dict[str, Any]:
    template = _read_json(EXECUTION_TEMPLATE)["environment"]
    return {
        "protocol_id": prereg["protocol_id"],
        "source": source,
        "market": {"instrument": INSTRUMENT},
        "roles": {"smoke_warmup_bars": 61},
        "rules": {"path": str(SCENARIO_RULES.relative_to(ROOT))},
        "native_execution_template": {
            "path": str(EXECUTION_TEMPLATE.relative_to(ROOT)),
            "starting_balance": template["starting_balance"],
            "risk_budget_notional": template["risk_budget_notional"],
            "hard_risk_max_leverage": template["hard_risk"]["max_leverage"],
        },
        "feature_recipe": {
            "path": str(FEATURE_RECIPE.relative_to(ROOT)),
            "sha256": _digest(FEATURE_RECIPE),
        },
        "cost_scenarios": source_errata["cost_scenarios"],
    }


def _configure_harness(source: dict[str, Any], prereg: dict[str, Any], output: Path) -> None:
    harness.MARKET = ROOT / source["market_path"]
    harness.MARKER = ROOT / source["marker_path"]
    harness.IMPORT = ROOT / source["import_manifest_path"]
    harness.RULES = SCENARIO_RULES
    harness.CONFIG = EXECUTION_TEMPLATE
    harness.OUT_BASE = output
    harness.FEATURE_PROFILE = FEATURE_PROFILE
    harness.FOLD_ID = prereg["protocol_id"]
    native.FEATURE_RECIPE = FEATURE_RECIPE


def _prepare_factory(
    source: dict[str, Any], manifest: dict[str, Any], prereg: dict[str, Any],
    output: Path, *, session_id: str, start_ns: int, end_ns: int,
    scenario: str, scaler: Any,
) -> tuple[dict[str, Any], Any]:
    _configure_harness(source, prereg, output)
    label = f"{prereg['protocol_id']}-{session_id}-{start_ns}-{scenario}"
    owner = ACTIVE_OWNER.set(prereg["protocol_id"])
    try:
        config, factory = harness.make_config(
            label=label,
            marker_sha=source["marker_sha256"],
            scope_start=int(source["read_scope_start_ns"]),
            scope_end=int(source["read_scope_end_ns"]),
            start_ns=start_ns,
            end_ns=end_ns,
            mode=scenario,
            fit_scaler=False,
            scaler=scaler,
            prepare_factory=True,
            source_mode=SOURCE_MODE,
            known_funding_calendar=(FEATURE_PROFILE == "causal_mtf_v4"),
        )
    finally:
        ACTIVE_OWNER.reset(owner)
    if factory is None or factory.scaler.hash != scaler.hash:
        raise RuntimeError("native screen factory did not reuse the frozen warmup scaler")
    return config, factory


def _fit_warmup_scaler(
    source: dict[str, Any], manifest: dict[str, Any], prereg: dict[str, Any],
    source_errata: dict[str, Any], output: Path, *, session_id: str,
) -> tuple[Any, dict[str, Any]]:
    bounds = source_errata_fit_window(prereg, source_errata)
    _configure_harness(source, prereg, output)
    label = f"{prereg['protocol_id']}-{session_id}-warmup-fit"
    owner = ACTIVE_OWNER.set(prereg["protocol_id"])
    try:
        _config, factory = harness.make_config(
            label=label,
            marker_sha=source["marker_sha256"],
            scope_start=int(source["read_scope_start_ns"]),
            scope_end=int(source["read_scope_end_ns"]),
            start_ns=bounds[0],
            end_ns=bounds[1],
            mode="base",
            fit_scaler=True,
            prepare_factory=True,
            source_mode=SOURCE_MODE,
            known_funding_calendar=(FEATURE_PROFILE == "causal_mtf_v4"),
        )
    finally:
        ACTIVE_OWNER.reset(owner)
    if factory is None or factory.scaler.training_only is not True:
        raise RuntimeError("pre-scored warmup did not produce a Train-only scaler")
    receipt = {
        "schema": "rles.route-a.teacher-screen-warmup-scaler.v1",
        "protocol_id": prereg["protocol_id"],
        "fit_window_ns": list(bounds),
        "scaler_hash": factory.scaler.hash,
        "train_window_hash": factory.scaler.train_window_hash,
        "feature_hash": factory.scaler.feature_hash,
        "sample_count": factory.scaler.sample_count,
        "training_only": factory.scaler.training_only,
        "quality_grade": factory.quality.grade,
        "quality_findings": list(factory.quality.findings),
        "scored_data_used_for_fit": False,
    }
    destination = output / "warmup-scaler.json"
    encoded = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("existing warmup scaler receipt differs from current frozen fit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")
    return factory.scaler, receipt


def source_errata_fit_window(
    prereg: dict[str, Any], source_errata: dict[str, Any],
) -> tuple[int, int]:
    fit = source_errata["warmup_scaler_fit_window_utc"]
    bounds = (_ns(fit[0]), _ns(fit[1]))
    source_start = _ns(prereg["scope"]["source_start_utc_inclusive"])
    scored_start = _ns(prereg["scope"]["scored_start_utc_inclusive"])
    if bounds != (source_start + 122 * MINUTE_NS, scored_start):
        raise RuntimeError("warmup scaler fit must be the complete pre-scored Train interval")
    return bounds


def _run_key(window_index: int, scenario: str, seed: int, candidate_id: str) -> str:
    safe_id = candidate_id.replace("/", "_")
    return f"window-{window_index + 1}/{scenario}/seed-{seed}/{safe_id}"


def _run_spec(
    row: dict[str, Any], candidate_id: str, scenario: str, seed: int,
    *, round_trip_cost_bps: float | None = None,
) -> tuple[str, BaseTeacher1m, NativeExitVariant, EmaReversalMode]:
    if "teacher_id" in row:
        reversal = row["reversal_mode"]
        teacher_id = row["teacher_id"]
        if teacher_id == ROC_COST_HYSTERESIS_TEACHER_ID:
            if round_trip_cost_bps is None:
                raise ValueError("cost-aware ROC requires the registered scenario round-trip cost")
            teacher = build_teacher_1m(
                teacher_id,
                exit_variant=cast(NativeExitVariant, row["exit_variant"]),
                reversal_mode=reversal,
                roc_round_trip_cost_bps=round_trip_cost_bps,
            )
        else:
            teacher = build_teacher_1m(
                teacher_id,
                exit_variant=cast(NativeExitVariant, row["exit_variant"]),
                reversal_mode=reversal,
            )
        return (
            candidate_id,
            teacher,
            cast(NativeExitVariant, row["exit_variant"]),
            cast(EmaReversalMode, reversal),
        )
    control = _control(row["control_id"])
    return candidate_id, control, "X1", "EXIT_ONLY_V1"


def _require_scenario_only_development_result(
    report: dict[str, Any], path: Path,
) -> None:
    if (report.get("economic_status") != SCENARIO_ONLY_ECONOMIC_STATUS
            or report.get("qualification") != DEVELOPMENT_QUALIFICATION):
        raise RuntimeError(
            f"native run is not bound to the frozen scenario-only development status: {path}"
        )


def _identity_context(prereg_sha: str, scaler_sha: str, warmup_marker_sha: str) -> dict[str, str]:
    # Bind the actual runnable source rather than trusting a self-reported revision.
    source_files = sorted(
        path for path in (ROOT / "packages").glob("*/src/**/*.py")
        if "__pycache__" not in path.parts
    )
    source_files += [ROOT / name for name in (
        "qa/route_a_run_identity.py",
        "qa/run_route_a_teacher_student_train_screen_20261026.py",
        "qa/run_route_a_1m_teacher_native.py",
        "qa/run_route_a_train_student_screen_20261020.py",
        "qa/validate_route_a_teacher_student_prereg_20261026.py",
        "pyproject.toml", "uv.lock",
    )]
    code = {str(path.relative_to(ROOT)).replace("\\", "/"): _digest(path)
            for path in source_files}
    return {
        "protocol_sha256": prereg_sha,
        "errata_sha256": _digest(ERRATA),
        "source_errata_sha256": _digest(SOURCE_ERRATA),
        "code_manifest_sha256": canonical_sha(code),
        "feature_recipe_sha256": _digest(FEATURE_RECIPE),
        "scaler_sha256": scaler_sha,
        "warmup_marker_sha256": warmup_marker_sha,
        "execution_template_sha256": _digest(EXECUTION_TEMPLATE),
        "instrument_rules_sha256": _digest(SCENARIO_RULES),
    }


def _make_identity(
    context: dict[str, str], source: dict[str, Any], *, candidate_id: str,
    recipe: dict[str, Any], scenario: str, costs: dict[str, Any], seed: int,
    window_ns: tuple[int, int],
) -> RunIdentityV1:
    return RunIdentityV1(
        **context,
        market_marker_sha256=source["marker_sha256"],
        market_snapshot_sha256=source["snapshot_hash"],
        import_manifest_sha256=source["import_manifest_sha256"],
        cost_recipe_sha256=canonical_sha(costs),
        teacher_recipe_sha256=canonical_sha(recipe),
        candidate_id=candidate_id,
        exit_variant=recipe["exit_variant"],
        scenario=scenario, seed=seed, window_ns=window_ns,
    )


def _summary_source(spec: dict[str, Any]) -> dict[str, str]:
    # Metadata only: do not decode or re-read already consumed market rows.
    rel = Path(spec["output"])
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError("registered summary source escapes the project")
    directory = (ROOT / rel).resolve()
    if not directory.is_relative_to(ROOT.resolve()):
        raise RuntimeError("registered summary source escapes the project")
    marker = directory / "market.parquet.COMPLETED.json"
    manifest = directory / "import.json"
    marker_doc, manifest_doc = _read_json(marker), _read_json(manifest)
    if (manifest_doc.get("start_day") != spec["start_day"]
            or manifest_doc.get("end_day") != spec["end_day_inclusive"]
            or manifest_doc.get("event_count") != marker_doc.get("events")):
        raise RuntimeError("summary source metadata differs from the registered window")
    return {"marker_sha256": _digest(marker),
            "snapshot_hash": marker_doc["snapshot_hash"],
            "import_manifest_sha256": _digest(manifest)}


def _completed_run(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    report_path = path / "native-run.json"
    if not path.exists():
        return None
    if not report_path.is_file():
        raise RuntimeError(f"partial native run exists and will not be overwritten: {path}")
    report = _read_json(report_path)
    checks = (
        report.get("protocol_id") == expected["protocol_id"],
        report.get("teacher") == expected["teacher"],
        report.get("teacher_recipe") == expected["teacher_recipe"],
        report.get("source", {}).get("marker_sha256") == expected["marker_sha256"],
        report.get("source", {}).get("window_ns") == expected["window_ns"],
        report.get("execution", {}).get("scenario") == expected["scenario"],
        report.get("execution", {}).get("seed") == expected["seed"],
    )
    if not all(checks):
        raise RuntimeError(f"existing native run does not match the frozen run key: {path}")
    _require_scenario_only_development_result(report, path)
    require_identity(report, expected["run_identity"])
    decisions = report.get("decisions", {})
    if (type(decisions.get("observed")) is not int or decisions["observed"] <= 0
            or decisions.get("observed") != decisions.get("m22_verified")):
        raise RuntimeError(f"existing run lacks complete M2.2 verification: {path}")
    if float(report.get("account", {}).get("liquidations", -1)) != 0:
        raise RuntimeError(f"existing run reports liquidation: {path}")
    return report


def _write_screen_summary(
    output: Path,
    prereg: dict[str, Any],
    errata: dict[str, Any],
    source_errata: dict[str, Any] | None = None,
    *, identity_context: dict[str, str], window_sources: dict[int, dict[str, str]],
) -> dict[str, Any]:
    preregistered_rows = errata["corrected_candidate_matrix"]
    teacher_rows = _candidate_rows_in_scope(errata, source_errata)
    candidates = {_candidate_id(row): row for row in teacher_rows}
    preregistered_candidate_ids = sorted(
        _candidate_id(row) for row in preregistered_rows
    )
    additional_frozen_candidate_ids = sorted(
        _candidate_id(row)
        for row in (
            source_errata.get("teacher_execution_scope", {}).get(
                "additional_frozen_candidates", [],
            ) if source_errata is not None else []
        )
        if _candidate_id(row) in candidates
    )
    excluded_candidate_ids = sorted(
        set(preregistered_candidate_ids) - set(candidates)
    )
    controls = {
        control_id: {"control_id": control_id}
        for control_id in prereg["teacher_screen"]["controls"]
    }
    window_count = len(prereg["teacher_screen"]["windows"])
    scenarios = tuple(prereg["teacher_screen"]["scenarios"])
    seeds = tuple(prereg["teacher_screen"]["seeds"])
    expected_teacher_ids = {
        candidate_id: row["teacher_id"] for candidate_id, row in candidates.items()
    }
    expected_teacher_ids.update({control_id: control_id for control_id in controls})
    expected_teacher_recipes: dict[tuple[str, str], dict[str, Any]] = {}
    scenario_costs = source_errata.get("cost_scenarios", {}) if source_errata else {}
    for scenario in scenarios:
        scenario_cost = scenario_costs.get(scenario)
        round_trip_cost_bps = (
            2 * (
                float(scenario_cost["taker_fee_bps_per_leg"])
                + float(scenario_cost["slippage_bps_per_leg"])
            )
            if scenario_cost is not None else None
        )
        for candidate_id, row in candidates.items():
            if (row.get("teacher_id") == ROC_COST_HYSTERESIS_TEACHER_ID
                    and round_trip_cost_bps is None):
                raise RuntimeError(
                    "cost-aware ROC recipe verification requires registered scenario costs"
                )
            _, teacher, exit_variant, _ = _run_spec(
                row, candidate_id, scenario, int(seeds[0]),
                round_trip_cost_bps=round_trip_cost_bps,
            )
            expected_teacher_recipes[(candidate_id, scenario)] = (
                _teacher_recipe_document(teacher, exit_variant)
            )
        for control_id in controls:
            expected_teacher_recipes[(control_id, scenario)] = (
                _teacher_recipe_document(_control(control_id), "X1")
            )
    report: dict[str, Any] = {
        "schema": "rles.route-a.teacher-student-fresh-train-screen.v1",
        "classification": "TRAIN_INNER_SCENARIO_ONLY",
        "protocol_id": prereg["protocol_id"],
        "qualification": DEVELOPMENT_QUALIFICATION,
        "economic_status": "SCENARIO_ONLY",
        "source_read_scope": prereg["scope"],
        "candidate_acceptance_rule": prereg["teacher_screen"]["acceptance"],
        "candidate_execution_scope": {
            "teacher_ids": sorted(IN_SCOPE_TEACHER_IDS),
            "evaluated_candidate_ids": sorted(candidates),
            "additional_frozen_candidate_ids": additional_frozen_candidate_ids,
            "excluded_candidate_ids": excluded_candidate_ids,
            "preregistered_candidate_ids": preregistered_candidate_ids,
        },
        "student_training_started": False,
        "completed_runs": [],
        "candidate_summary": {},
        "screen_complete": False,
        "qualifying_candidates": [],
    }
    source_markers_by_window: dict[int, str] = {}
    for path in sorted(output.glob("window-*/**/native-run.json")):
        run = _read_json(path)
        key_parts = path.relative_to(output).parts
        window_index = int(key_parts[0].split("-")[1]) - 1
        if not 0 <= window_index < window_count:
            raise RuntimeError(f"run directory has an unregistered teacher window: {path}")
        scenario = key_parts[1]
        seed = int(key_parts[2].removeprefix("seed-"))
        candidate_id = key_parts[3]
        if candidate_id not in expected_teacher_ids:
            raise RuntimeError(f"run directory is outside the frozen candidate/control matrix: {path}")
        expected_window = prereg["teacher_screen"]["windows"][window_index]
        expected_bounds = [_ns(expected_window[0]), _ns(expected_window[1])]
        marker_sha = run.get("source", {}).get("marker_sha256")
        observed = run.get("decisions", {}).get("observed")
        m22_verified = run.get("decisions", {}).get("m22_verified")
        prior_marker_sha = source_markers_by_window.setdefault(window_index, marker_sha)
        if (run.get("protocol_id") != prereg["protocol_id"]
                or run.get("teacher") != expected_teacher_ids[candidate_id]
                or run.get("source", {}).get("window_ns") != expected_bounds
                or marker_sha != prior_marker_sha
                or run.get("execution", {}).get("scenario") != scenario
                or scenario not in scenarios
                or run.get("execution", {}).get("seed") != seed
                or seed not in seeds
                or run.get("teacher_recipe")
                != expected_teacher_recipes.get((candidate_id, scenario))
                or run.get("economic_status") != SCENARIO_ONLY_ECONOMIC_STATUS
                or run.get("qualification") != DEVELOPMENT_QUALIFICATION
                or not isinstance(marker_sha, str) or len(marker_sha) != 64
                or type(observed) is not int or observed <= 0
                or observed != m22_verified
                or float(run.get("account", {}).get("liquidations", -1)) != 0):
            raise RuntimeError(f"native run identity or execution evidence is invalid: {path}")
        if scenario not in scenario_costs or window_index not in window_sources:
            raise RuntimeError(f"registered identity evidence is missing: {path}")
        expected_identity = _make_identity(
            identity_context, window_sources[window_index],
            candidate_id=candidate_id,
            recipe=expected_teacher_recipes[(candidate_id, scenario)],
            scenario=scenario, costs=scenario_costs[scenario], seed=seed,
            window_ns=(expected_bounds[0], expected_bounds[1]),
        )
        require_identity(run, expected_identity)
        report["completed_runs"].append({
            "candidate_id": candidate_id,
            "window_index": window_index,
            "scenario": scenario,
            "seed": seed,
            "net_return": run["account"]["net_return"],
            "max_drawdown": run["account"]["max_drawdown"],
            "fees": run["account"]["fees"],
            "slippage_cost": run["account"]["slippage_cost"],
            "funding": run["account"]["funding"],
            "liquidations": run["account"]["liquidations"],
            "m22_verified": run["decisions"]["observed"] == run["decisions"]["m22_verified"],
            "native_report": str(path.relative_to(output)),
        })
    run_index = {
        (row["candidate_id"], row["window_index"], row["scenario"], row["seed"]): row
        for row in report["completed_runs"]
    }
    if len(run_index) != len(report["completed_runs"]):
        raise RuntimeError("duplicate candidate/window/scenario/seed native run detected")
    expected_slots = window_count * len(scenarios) * len(seeds)
    all_ids = [*candidates, *controls]
    for candidate_id in candidates:
        slots = []
        qualifies = True
        for window_index in range(window_count):
            for scenario in scenarios:
                for seed in seeds:
                    candidate = run_index.get((candidate_id, window_index, scenario, seed))
                    comparison = {
                        control_id: run_index.get((control_id, window_index, scenario, seed))
                        for control_id in controls
                    }
                    if candidate is None or any(row is None for row in comparison.values()):
                        qualifies = False
                        continue
                    beat_controls = {
                        control_id: candidate["net_return"] > row["net_return"]
                        for control_id, row in comparison.items() if row is not None
                    }
                    slot_pass = (
                        candidate["net_return"] > 0
                        and all(beat_controls.values())
                        and candidate["max_drawdown"] <= 0.30
                        and candidate["liquidations"] == 0
                        and candidate["m22_verified"]
                    )
                    qualifies = qualifies and slot_pass
                    slots.append({
                        "window_index": window_index,
                        "scenario": scenario,
                        "seed": seed,
                        "net_return": candidate["net_return"],
                        "max_drawdown": candidate["max_drawdown"],
                        "beats_each_control": beat_controls,
                        "slot_pass": slot_pass,
                    })
        present = sum(
            (candidate_id, window_index, scenario, seed) in run_index
            for window_index in range(window_count)
            for scenario in scenarios
            for seed in seeds
        )
        report["candidate_summary"][candidate_id] = {
            "completed_slots": present,
            "expected_slots": expected_slots,
            "all_frozen_slots_pass": present == expected_slots and qualifies,
            "slots": slots,
        }
        if present == expected_slots and qualifies:
            report["qualifying_candidates"].append(candidate_id)
    actual_run_keys = set(run_index)
    active_expected_run_keys = {
        (candidate_id, window_index, scenario, seed)
        for candidate_id in all_ids
        for window_index in range(window_count)
        for scenario in scenarios
        for seed in seeds
    }
    preregistered_expected_run_keys = {
        (candidate_id, window_index, scenario, seed)
        for candidate_id in (*preregistered_candidate_ids, *controls)
        for window_index in range(window_count)
        for scenario in scenarios
        for seed in seeds
    }
    report["screen_complete"] = actual_run_keys == active_expected_run_keys
    report["full_preregistered_candidate_matrix_complete"] = (
        preregistered_expected_run_keys.issubset(actual_run_keys)
    )
    def paired_comparison(
        baseline_id: str, candidate_id: str,
        baseline_label: str, candidate_label: str,
    ) -> dict[str, Any]:
        pairs = []
        for window_index in range(window_count):
            for scenario in scenarios:
                for seed in seeds:
                    baseline = run_index.get(
                        (baseline_id, window_index, scenario, seed),
                    )
                    candidate = run_index.get(
                        (candidate_id, window_index, scenario, seed),
                    )
                    if baseline is None or candidate is None:
                        continue
                    pairs.append({
                        "window_index": window_index,
                        "scenario": scenario,
                        "seed": seed,
                        f"{baseline_label}_net_return": baseline["net_return"],
                        f"{candidate_label}_net_return": candidate["net_return"],
                        "net_return_delta": (
                            candidate["net_return"] - baseline["net_return"]
                        ),
                        f"{baseline_label}_max_drawdown": baseline["max_drawdown"],
                        f"{candidate_label}_max_drawdown": candidate["max_drawdown"],
                        "max_drawdown_delta": (
                            candidate["max_drawdown"] - baseline["max_drawdown"]
                        ),
                        "fees_delta": candidate["fees"] - baseline["fees"],
                        "slippage_cost_delta": (
                            candidate["slippage_cost"] - baseline["slippage_cost"]
                        ),
                        "funding_delta": candidate["funding"] - baseline["funding"],
                    })
        return {
            "baseline_candidate_id": baseline_id,
            "candidate_id": candidate_id,
            "expected_pairs": expected_slots,
            "completed_pairs": len(pairs),
            "pairs": pairs,
        }

    legacy_roc_id = "T02_ROC_20_60"
    cost_aware_roc_id = ROC_COST_HYSTERESIS_TEACHER_ID
    cost_aware_roc_x7_id = f"{ROC_COST_HYSTERESIS_TEACHER_ID}--X7"
    report["paired_candidate_comparisons"] = {
        "cost_aware_roc_vs_legacy_roc": paired_comparison(
            legacy_roc_id, cost_aware_roc_id, "legacy", "cost_aware",
        ),
        "cost_aware_roc_x7_vs_x1": paired_comparison(
            cost_aware_roc_id, cost_aware_roc_x7_id, "x1", "x7",
        ),
    }
    report["source_markers_by_window"] = {
        str(index): marker for index, marker in sorted(source_markers_by_window.items())
    }
    report["student_training_eligible"] = (
        report["screen_complete"] and bool(report["qualifying_candidates"])
    )
    destination = output / "teacher-screen-summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def run_window(window_index: int) -> dict[str, Any]:
    if not 0 <= window_index < 4:
        raise ValueError("window index must select one of the four frozen teacher windows")
    # Read-only protocol/ledger preflight must pass before inspecting imported market bytes.
    import validate_route_a_teacher_student_prereg_20261026 as validation

    validation.main()
    prereg, errata, source_errata, prereg_sha = _protocol()
    source_spec = source_errata["teacher_window_imports"][window_index]
    source, manifest = _verified_source(
        prereg, source_errata, prereg_sha, source_spec,
    )
    warmup_source, warmup_manifest = _verified_source(
        prereg, source_errata, prereg_sha, source_errata["warmup_scaler_import"],
    )
    window = prereg["teacher_screen"]["windows"][window_index]
    start_ns, end_ns = _ns(window[0]), _ns(window[1])
    block_id = f"teacher-{window[0][:10].replace('-', '')}_{window[1][:10].replace('-', '')}"
    output = OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    session_id = uuid4().hex[:12]

    scaler, scaler_receipt = _fit_warmup_scaler(
        warmup_source, warmup_manifest, prereg, source_errata, output,
        session_id=session_id,
    )
    report_doc = _run_doc(prereg, source, source_errata)
    identity_context = _identity_context(
        prereg_sha, scaler.hash, warmup_source["marker_sha256"],
    )
    identity_source = {
        "marker_sha256": source["marker_sha256"],
        "snapshot_hash": source["snapshot_hash"],
        "import_manifest_sha256": _digest(ROOT / source["import_manifest_path"]),
    }
    run_rows: list[dict[str, Any]] = []
    candidate_specs = [
        (_candidate_id(row), row)
        for row in _candidate_rows_in_scope(errata, source_errata)
    ]
    candidate_specs.extend(
        (control_id, {"control_id": control_id})
        for control_id in prereg["teacher_screen"]["controls"]
    )
    for scenario in prereg["teacher_screen"]["scenarios"]:
        config, factory = _prepare_factory(
            source, manifest, prereg, output,
            session_id=session_id,
            start_ns=start_ns, end_ns=end_ns, scenario=scenario, scaler=scaler,
        )
        prepared = (config, factory)
        scenario_costs = source_errata["cost_scenarios"][scenario]
        round_trip_cost_bps = 2 * (
            float(scenario_costs["taker_fee_bps_per_leg"])
            + float(scenario_costs["slippage_bps_per_leg"])
        )
        for seed in prereg["teacher_screen"]["seeds"]:
            for candidate_id, row in candidate_specs:
                _, teacher, exit_variant, reversal = _run_spec(
                    row, candidate_id, scenario, int(seed),
                    round_trip_cost_bps=round_trip_cost_bps,
                )
                run_dir = output / _run_key(
                    window_index, scenario, int(seed), candidate_id,
                )
                recipe = _teacher_recipe_document(teacher, exit_variant)
                identity = _make_identity(
                    identity_context, identity_source, candidate_id=candidate_id,
                    recipe=recipe, scenario=scenario, costs=scenario_costs,
                    seed=int(seed), window_ns=(start_ns, end_ns),
                )
                expected = {
                    "run_identity": identity,
                    "protocol_id": prereg["protocol_id"],
                    "teacher": teacher.expert_id,
                    "teacher_recipe": _teacher_recipe_document(teacher, exit_variant),
                    "marker_sha256": source["marker_sha256"],
                    "window_ns": [start_ns, end_ns],
                    "scenario": scenario,
                    "seed": int(seed),
                }
                if not run_dir.exists():
                    run_dir.mkdir(parents=True, exist_ok=False)
                else:
                    completed = _completed_run(run_dir, expected)
                    if completed is not None:
                        continue
                if not (run_dir / "native-run.json").is_file():
                    run = native._run_native_window(
                        report_doc,
                        prereg_sha,
                        run_dir,
                        teacher_id=teacher.expert_id,
                        exit_variant=exit_variant,
                        scenario=scenario,
                        window_id=f"teacher-screen-{window_index + 1}",
                        prepared=prepared,
                        ema_reversal_mode=reversal,
                        bounds_override=(start_ns, end_ns, block_id),
                        seed=int(seed),
                        teacher_override=teacher,
                        run_identity=identity,
                    )
                    run_rows.append(run)
        factory.snapshots.clear()
        factory.teacher_snapshots.clear()
        factory.fill_references.clear()
    populated_windows = {
        int(path.relative_to(output).parts[0].split("-")[1]) - 1
        for path in output.glob("window-*/**/native-run.json")
    }
    window_sources = {
        index: _summary_source(source_errata["teacher_window_imports"][index])
        for index in populated_windows
    }
    summary = _write_screen_summary(
        output, prereg, errata, source_errata,
        identity_context=identity_context, window_sources=window_sources,
    )
    return {
        "window_index": window_index,
        "block_id": block_id,
        "new_runs": len(run_rows),
        "scaler_hash": scaler.hash,
        "scaler_fit_window_ns": scaler_receipt["fit_window_ns"],
        "screen_complete": summary["screen_complete"],
        "student_training_eligible": summary["student_training_eligible"],
        "summary_path": str(output / "teacher-screen-summary.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-index", type=int, required=True)
    args = parser.parse_args(argv)
    result = run_window(args.window_index)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
