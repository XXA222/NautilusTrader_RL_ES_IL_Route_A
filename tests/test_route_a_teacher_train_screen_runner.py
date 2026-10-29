from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qa.run_route_a_teacher_student_train_screen_20261026 import (
    FixedExposureControlTeacher,
    _candidate_id,
    _candidate_rows_in_scope,
    _completed_run,
    _control,
    _ns,
    _protocol as _load_registered_protocol,
    _run_spec,
    _teacher_recipe_document,
    _write_screen_summary as _real_write_screen_summary,
    _make_identity,
    source_errata_fit_window,
)
import qa.run_route_a_train_student_screen_20261020 as harness
from rles_route_a.teachers_1m.contracts import TeacherSignal


CONTEXT = {name: "a" * 64 for name in (
    "protocol_sha256", "errata_sha256", "source_errata_sha256",
    "code_manifest_sha256", "feature_recipe_sha256", "scaler_sha256",
    "warmup_marker_sha256", "execution_template_sha256", "instrument_rules_sha256",
)}
COSTS = {
    "base": {"taker_fee_bps_per_leg": 5, "slippage_bps_per_leg": 1},
    "stress": {"taker_fee_bps_per_leg": 10, "slippage_bps_per_leg": 5},
}


def _source(marker: str) -> dict[str, str]:
    return {"marker_sha256": marker, "snapshot_hash": "b" * 64,
            "import_manifest_sha256": "c" * 64}


def _identity(candidate: str, scenario: str, seed: int, start: int, end: int,
              marker: str, recipe: dict[str, Any]):
    return _make_identity(CONTEXT, _source(marker), candidate_id=candidate,
                          recipe=recipe, scenario=scenario, costs=COSTS[scenario],
                          seed=seed, window_ns=(start, end))


def _write_screen_summary(output, prereg, errata, source_errata=None):
    source_errata = dict(source_errata or {})
    source_errata.setdefault("cost_scenarios", COSTS)
    sources = {i: _source("a" * 63 + str(i)) for i in range(4)}
    return _real_write_screen_summary(output, prereg, errata, source_errata,
                                      identity_context=CONTEXT, window_sources=sources)


def _recipe_document(candidate: str, scenario: str, seed: int) -> dict[str, Any]:
    control_targets = {
        "FLAT": 0,
        "FIXED_LONG_0.3": 1,
        "FIXED_SHORT_0.3": -1,
    }
    if candidate in control_targets:
        return _teacher_recipe_document(_control(candidate), "X1")
    teacher_id = candidate.split("--", 1)[0]
    reversal_mode = (
        "REVERSE_AFTER_FLAT_V1"
        if candidate.endswith("--REVERSE_AFTER_FLAT_V1")
        else "EXIT_ONLY_V1"
    )
    exit_variant = "X7" if candidate.endswith("--X7") else "X1"
    row = {
        "teacher_id": teacher_id,
        "reversal_mode": reversal_mode,
        "exit_variant": exit_variant,
    }
    round_trip_cost_bps = (
        {"base": 12.0, "stress": 30.0}[scenario]
        if teacher_id == "T02_ROC_20_60_COST_HYSTERESIS_V1"
        else None
    )
    _, teacher, resolved_exit, _ = _run_spec(
        row, candidate, scenario, seed, round_trip_cost_bps=round_trip_cost_bps,
    )
    return _teacher_recipe_document(teacher, resolved_exit)


def _native_run(
    *, candidate: str, scenario: str, seed: int, net_return: float,
    window_start: int, window_end: int, marker_sha: str,
    recipe_candidate: str | None = None,
) -> dict[str, Any]:
    teacher_id = candidate.split("--", 1)[0]
    recipe = _recipe_document(recipe_candidate or candidate, scenario, seed)
    identity = _identity(candidate, scenario, seed, window_start, window_end,
                         marker_sha, recipe)
    return {
        "run_identity": identity.document(),
        "preregistration_sha256": CONTEXT["protocol_sha256"],
        "verified_feature_recipe_sha256": CONTEXT["feature_recipe_sha256"],
        "protocol_id": "frozen-protocol",
        "teacher": teacher_id,
        "teacher_recipe": _recipe_document(recipe_candidate or candidate, scenario, seed),
        "source": {
            "marker_sha256": marker_sha,
            "snapshot_hash": "b" * 64,
            "window_ns": [window_start, window_end],
        },
        "execution": {"scenario": scenario, "seed": seed,
                      "scaler_hash": CONTEXT["scaler_sha256"],
                      "scenario_costs": COSTS[scenario]},
        "qualification": "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2",
        "economic_status": "SCENARIO_ONLY",
        "account": {
            "net_return": net_return,
            "max_drawdown": 0.1,
            "fees": 1.0,
            "slippage_cost": 0.5,
            "funding": 0.0,
            "liquidations": 0,
        },
        "decisions": {"observed": 100, "m22_verified": 100},
    }


def _write_run(
    root: Path, candidate: str, window_index: int, scenario: str, seed: int,
    net_return: float, window_start: int, window_end: int,
    marker_sha: str | None = None, recipe_candidate: str | None = None,
) -> None:
    path = root / f"window-{window_index + 1}" / scenario / f"seed-{seed}" / candidate
    path.mkdir(parents=True)
    (path / "native-run.json").write_text(
        json.dumps(_native_run(
            candidate=candidate,
            scenario=scenario,
            seed=seed,
            net_return=net_return,
            window_start=window_start,
            window_end=window_end,
            marker_sha=marker_sha or ("a" * 63 + str(window_index)),
            recipe_candidate=recipe_candidate,
        )),
        encoding="utf-8",
    )


def _protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = {
        "protocol_id": "frozen-protocol",
        "scope": {"source_start_utc_inclusive": "2026-10-23T00:00:00Z"},
        "teacher_screen": {
            "windows": [["2026-10-25T00:00:00Z", "2026-10-30T00:00:00Z"]] * 4,
            "scenarios": ["base", "stress"],
            "seeds": [20261026, 20261027],
            "controls": ["FLAT", "FIXED_LONG_0.3", "FIXED_SHORT_0.3"],
            "acceptance": "positive versus every control in every paired slot",
        },
    }
    errata = {
        "corrected_candidate_matrix": [
            {"teacher_id": "T01_EMA_9_21", "reversal_mode": "EXIT_ONLY_V1", "exit_variant": "X1"},
            {"teacher_id": "T01_EMA_9_21", "reversal_mode": "REVERSE_AFTER_FLAT_V1", "exit_variant": "X1"},
            {"teacher_id": "T02_ROC_20_60", "reversal_mode": "EXIT_ONLY_V1", "exit_variant": "X1"},
            {"teacher_id": "T03_DONCHIAN_20", "reversal_mode": "EXIT_ONLY_V1", "exit_variant": "X1"},
            {"teacher_id": "T04_ZSCORE_60", "reversal_mode": "EXIT_ONLY_V1", "exit_variant": "X1"},
            {"teacher_id": "T05_MACD_12_26_9", "reversal_mode": "EXIT_ONLY_V1", "exit_variant": "X1"},
            {"teacher_id": "T06_RSI_14", "reversal_mode": "EXIT_ONLY_V1", "exit_variant": "X1"},
            {"teacher_id": "T07_BOLL_20_2", "reversal_mode": "EXIT_ONLY_V1", "exit_variant": "X1"},
            {"teacher_id": "T08_PULLBACK_1M", "reversal_mode": "EXIT_ONLY_V1", "exit_variant": "X1"},
        ],
    }
    return prereg, errata


def test_fixed_exposure_controls_hold_the_frozen_side() -> None:
    flat = FixedExposureControlTeacher("FLAT", 0)
    long = FixedExposureControlTeacher("FIXED_LONG_0.3", 1)
    short = FixedExposureControlTeacher("FIXED_SHORT_0.3", -1)

    assert flat.entry_signal({}, {}).raw_signal == "NONE"
    assert long.entry_signal({}, {}).raw_signal == "LONG_ENTRY"
    assert short.entry_signal({}, {}).raw_signal == "SHORT_ENTRY"
    assert long.exit_signal(0.3, {}, {}) is None
    assert short.exit_signal(-0.3, {}, {}) is None
    assert isinstance(flat.entry_signal({}, {}), TeacherSignal)


def test_candidate_id_keeps_ema_reversal_variants_distinct() -> None:
    assert _candidate_id({
        "teacher_id": "T01_EMA_9_21",
        "reversal_mode": "EXIT_ONLY_V1",
    }) == "T01_EMA_9_21"
    assert _candidate_id({
        "teacher_id": "T01_EMA_9_21",
        "reversal_mode": "REVERSE_AFTER_FLAT_V1",
    }) == "T01_EMA_9_21--REVERSE_AFTER_FLAT_V1"
    assert _candidate_id({
        "teacher_id": "T02_ROC_20_60_COST_HYSTERESIS_V1",
        "reversal_mode": "EXIT_ONLY_V1",
        "exit_variant": "X7",
    }) == "T02_ROC_20_60_COST_HYSTERESIS_V1--X7"


def test_active_screen_excludes_the_remaining_four_teacher_families() -> None:
    matrix = [
        {"teacher_id": "T01_EMA_9_21", "reversal_mode": "EXIT_ONLY_V1"},
        {"teacher_id": "T01_EMA_9_21", "reversal_mode": "REVERSE_AFTER_FLAT_V1"},
        {"teacher_id": "T02_ROC_20_60", "reversal_mode": "EXIT_ONLY_V1"},
        {"teacher_id": "T03_DONCHIAN_20", "reversal_mode": "EXIT_ONLY_V1"},
        {"teacher_id": "T04_ZSCORE_60", "reversal_mode": "EXIT_ONLY_V1"},
        {"teacher_id": "T05_MACD_12_26_9", "reversal_mode": "EXIT_ONLY_V1"},
        {"teacher_id": "T06_RSI_14", "reversal_mode": "EXIT_ONLY_V1"},
        {"teacher_id": "T07_BOLL_20_2", "reversal_mode": "EXIT_ONLY_V1"},
        {"teacher_id": "T08_PULLBACK_1M", "reversal_mode": "EXIT_ONLY_V1"},
    ]

    selected = _candidate_rows_in_scope({"corrected_candidate_matrix": matrix})

    assert [_candidate_id(row) for row in selected] == [
        "T01_EMA_9_21",
        "T01_EMA_9_21--REVERSE_AFTER_FLAT_V1",
        "T02_ROC_20_60",
        "T03_DONCHIAN_20",
        "T04_ZSCORE_60",
    ]


def test_registered_execution_scope_matches_the_active_task() -> None:
    _, errata, source_errata, _ = _load_registered_protocol()

    assert source_errata["teacher_execution_scope"]["included_teacher_ids"] == [
        "T01_EMA_9_21",
        "T02_ROC_20_60",
        "T02_ROC_20_60_COST_HYSTERESIS_V1",
        "T03_DONCHIAN_20",
        "T04_ZSCORE_60",
    ]
    assert set(source_errata["teacher_execution_scope"]["excluded_teacher_ids"]) == {
        "T05_MACD_12_26_9",
        "T06_RSI_14",
        "T07_BOLL_20_2",
        "T08_PULLBACK_1M",
    }
    assert source_errata["teacher_execution_scope"]["additional_frozen_candidates"] == [
        {
            "teacher_id": "T02_ROC_20_60_COST_HYSTERESIS_V1",
            "reversal_mode": "EXIT_ONLY_V1",
            "exit_variant": "X1",
        },
        {
            "teacher_id": "T02_ROC_20_60_COST_HYSTERESIS_V1",
            "reversal_mode": "EXIT_ONLY_V1",
            "exit_variant": "X7",
        },
    ]
    assert len(errata["corrected_candidate_matrix"]) == 9


def test_cost_aware_roc_uses_registered_base_and_stress_costs() -> None:
    row = {
        "teacher_id": "T02_ROC_20_60_COST_HYSTERESIS_V1",
        "reversal_mode": "EXIT_ONLY_V1",
        "exit_variant": "X1",
    }
    for scenario, round_trip_cost in (("base", 12.0), ("stress", 30.0)):
        _, teacher, exit_variant, _ = _run_spec(
            row,
            "T02_ROC_20_60_COST_HYSTERESIS_V1",
            scenario,
            20261026,
            round_trip_cost_bps=round_trip_cost,
        )
        assert dict(teacher.recipe.parameters)["round_trip_cost_bps"] == round_trip_cost
        assert teacher.recipe.entry_rule_id == "ROC_COST_AWARE_CONSENSUS_HYSTERESIS_V1"
        assert exit_variant == "X1"


def test_frozen_roc_x7_keeps_ema_age_gate_and_x1_exit() -> None:
    row = {
        "teacher_id": "T02_ROC_20_60_COST_HYSTERESIS_V1",
        "reversal_mode": "EXIT_ONLY_V1",
        "exit_variant": "X7",
    }
    for scenario_cost in (12.0, 30.0):
        _, teacher, exit_variant, _ = _run_spec(
            row,
            _candidate_id(row),
            "base" if scenario_cost == 12.0 else "stress",
            20261026,
            round_trip_cost_bps=scenario_cost,
        )
        parameters = dict(teacher.recipe.parameters)
        assert parameters["round_trip_cost_bps"] == scenario_cost
        assert parameters["entry_ema_cross_age_min_bars"] == 21
        assert parameters["entry_ema_cross_age_max_bars"] == 60
        assert teacher.recipe.entry_rule_id == "ROC_COST_AWARE_EMA_CROSS_AGE_21_60_V1"
        assert exit_variant == "X7"


def test_warmup_scaler_is_frozen_to_pre_scored_interval() -> None:
    prereg = {
        "scope": {
            "source_start_utc_inclusive": "2026-10-23T00:00:00Z",
            "scored_start_utc_inclusive": "2026-10-25T00:00:00Z",
        },
    }
    errata = {
        "warmup_scaler_fit_window_utc": [
            "2026-10-23T02:02:00Z", "2026-10-25T00:00:00Z",
        ],
    }
    start, end = source_errata_fit_window(prereg, errata)
    assert start == _ns("2026-10-23T02:02:00Z")
    assert end == _ns("2026-10-25T00:00:00Z")


def test_known_funding_calendar_is_explicit_and_opt_in() -> None:
    config, factory = harness.make_config(
        label="known-calendar-unit-test",
        marker_sha="a" * 64,
        scope_start=_ns("2026-10-23T00:00:00Z"),
        scope_end=_ns("2026-10-25T00:00:00Z"),
        start_ns=_ns("2026-10-23T02:02:00Z"),
        end_ns=_ns("2026-10-25T00:00:00Z"),
        mode="base",
        fit_scaler=True,
        prepare_factory=False,
        known_funding_calendar=True,
    )
    assert factory is None
    policy = config["environment"]["coverage_policy"]
    assert policy["known_schedule_times_ns"] == policy["settlement_times_ns"]
    legacy, _ = harness.make_config(
        label="known-calendar-unit-test",
        marker_sha="a" * 64,
        scope_start=_ns("2026-10-23T00:00:00Z"),
        scope_end=_ns("2026-10-25T00:00:00Z"),
        start_ns=_ns("2026-10-23T02:02:00Z"),
        end_ns=_ns("2026-10-25T00:00:00Z"),
        mode="base",
        fit_scaler=True,
        prepare_factory=False,
    )
    assert "known_schedule_times_ns" not in legacy["environment"]["coverage_policy"]


def test_acceptance_requires_every_candidate_control_pair_in_every_slot(tmp_path: Path) -> None:
    prereg, errata = _protocol()
    source_errata = {
        "teacher_execution_scope": {
            "additional_frozen_candidates": [{
                "teacher_id": "T02_ROC_20_60_COST_HYSTERESIS_V1",
                "reversal_mode": "EXIT_ONLY_V1",
                "exit_variant": "X1",
            }, {
                "teacher_id": "T02_ROC_20_60_COST_HYSTERESIS_V1",
                "reversal_mode": "EXIT_ONLY_V1",
                "exit_variant": "X7",
            }],
        },
        "cost_scenarios": {
            "base": {"taker_fee_bps_per_leg": 5, "slippage_bps_per_leg": 1},
            "stress": {"taker_fee_bps_per_leg": 10, "slippage_bps_per_leg": 5},
        },
    }
    start = _ns("2026-10-25T00:00:00Z")
    end = _ns("2026-10-30T00:00:00Z")
    candidates = [
        _candidate_id(row)
        for row in _candidate_rows_in_scope(errata, source_errata)
    ]
    controls = prereg["teacher_screen"]["controls"]
    for window_index in range(4):
        for scenario in prereg["teacher_screen"]["scenarios"]:
            for seed in prereg["teacher_screen"]["seeds"]:
                for candidate in candidates:
                    _write_run(tmp_path, candidate, window_index, scenario, seed, 0.02, start, end)
                for control, result in zip(controls, (0.0, 0.01, -0.01), strict=True):
                    _write_run(tmp_path, control, window_index, scenario, seed, result, start, end)

    report = _write_screen_summary(tmp_path, prereg, errata, source_errata)
    assert report["screen_complete"] is True
    assert report["qualification"] == "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2"
    assert report["economic_status"] == "SCENARIO_ONLY"
    assert report["full_preregistered_candidate_matrix_complete"] is False
    assert set(report["qualifying_candidates"]) == set(candidates)
    assert "T02_ROC_20_60_COST_HYSTERESIS_V1" in candidates
    assert "T02_ROC_20_60_COST_HYSTERESIS_V1--X7" in candidates
    comparison = report["paired_candidate_comparisons"]["cost_aware_roc_vs_legacy_roc"]
    assert comparison["completed_pairs"] == comparison["expected_pairs"] == 16
    assert all(pair["net_return_delta"] == 0 for pair in comparison["pairs"])
    x7_comparison = report["paired_candidate_comparisons"]["cost_aware_roc_x7_vs_x1"]
    assert x7_comparison["baseline_candidate_id"] == "T02_ROC_20_60_COST_HYSTERESIS_V1"
    assert x7_comparison["candidate_id"] == "T02_ROC_20_60_COST_HYSTERESIS_V1--X7"
    assert x7_comparison["completed_pairs"] == x7_comparison["expected_pairs"] == 16
    assert all(pair["net_return_delta"] == 0 for pair in x7_comparison["pairs"])
    assert report["candidate_execution_scope"]["excluded_candidate_ids"] == [
        "T05_MACD_12_26_9",
        "T06_RSI_14",
        "T07_BOLL_20_2",
        "T08_PULLBACK_1M",
    ]
    assert report["student_training_eligible"] is True
    assert report["source_markers_by_window"] == {
        str(index): "a" * 63 + str(index) for index in range(4)
    }


def test_screen_summary_rejects_recipe_mismatch_under_candidate_key(tmp_path: Path) -> None:
    prereg, errata = _protocol()
    source_errata = {
        "teacher_execution_scope": {
            "additional_frozen_candidates": [{
                "teacher_id": "T02_ROC_20_60_COST_HYSTERESIS_V1",
                "reversal_mode": "EXIT_ONLY_V1",
                "exit_variant": "X1",
            }, {
                "teacher_id": "T02_ROC_20_60_COST_HYSTERESIS_V1",
                "reversal_mode": "EXIT_ONLY_V1",
                "exit_variant": "X7",
            }],
        },
        "cost_scenarios": {
            "base": {"taker_fee_bps_per_leg": 5, "slippage_bps_per_leg": 1},
            "stress": {"taker_fee_bps_per_leg": 10, "slippage_bps_per_leg": 5},
        },
    }
    x1_id = "T02_ROC_20_60_COST_HYSTERESIS_V1"
    x7_id = f"{x1_id}--X7"
    _write_run(
        tmp_path, x7_id, 0, "base", 20261026, 0.5,
        _ns("2026-10-25T00:00:00Z"), _ns("2026-10-30T00:00:00Z"),
        recipe_candidate=x1_id,
    )

    with pytest.raises(RuntimeError, match="identity or execution evidence"):
        _write_screen_summary(tmp_path, prereg, errata, source_errata)


def test_resume_rejects_recipe_mismatch_under_candidate_key(tmp_path: Path) -> None:
    x1_id = "T02_ROC_20_60_COST_HYSTERESIS_V1"
    x7_id = f"{x1_id}--X7"
    path = tmp_path / "run"
    path.mkdir()
    report = _native_run(
        candidate=x7_id,
        recipe_candidate=x1_id,
        scenario="base",
        seed=20261026,
        net_return=0.1,
        window_start=_ns("2026-10-25T00:00:00Z"),
        window_end=_ns("2026-10-30T00:00:00Z"),
        marker_sha="a" * 64,
    )
    (path / "native-run.json").write_text(json.dumps(report), encoding="utf-8")
    expected = {
        "protocol_id": "frozen-protocol",
        "teacher": x1_id,
        "teacher_recipe": _recipe_document(x7_id, "base", 20261026),
        "marker_sha256": "a" * 64,
        "window_ns": [
            _ns("2026-10-25T00:00:00Z"),
            _ns("2026-10-30T00:00:00Z"),
        ],
        "scenario": "base",
        "seed": 20261026,
    }

    with pytest.raises(RuntimeError, match="does not match the frozen run key"):
        _completed_run(path, expected)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (("economic_status", "G1_QUALIFIED"), ("qualification", "G1_G2_QUALIFIED")),
)
def test_resume_rejects_native_run_with_non_diagnostic_authority(
    tmp_path: Path, field: str, wrong_value: str,
) -> None:
    candidate = "T01_EMA_9_21"
    path = tmp_path / "run"
    path.mkdir()
    report = _native_run(
        candidate=candidate,
        scenario="base",
        seed=20261026,
        net_return=0.1,
        window_start=_ns("2026-10-25T00:00:00Z"),
        window_end=_ns("2026-10-30T00:00:00Z"),
        marker_sha="a" * 64,
    )
    report[field] = wrong_value
    (path / "native-run.json").write_text(json.dumps(report), encoding="utf-8")
    expected = {
        "protocol_id": "frozen-protocol",
        "teacher": candidate,
        "teacher_recipe": _recipe_document(candidate, "base", 20261026),
        "marker_sha256": "a" * 64,
        "window_ns": [
            _ns("2026-10-25T00:00:00Z"),
            _ns("2026-10-30T00:00:00Z"),
        ],
        "scenario": "base",
        "seed": 20261026,
    }

    with pytest.raises(RuntimeError, match="scenario-only development status"):
        _completed_run(path, expected)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (("economic_status", "G1_QUALIFIED"), ("qualification", "G1_G2_QUALIFIED")),
)
def test_screen_summary_rejects_native_run_with_non_diagnostic_authority(
    tmp_path: Path, field: str, wrong_value: str,
) -> None:
    prereg, errata = _protocol()
    candidate = _candidate_id(errata["corrected_candidate_matrix"][0])
    start = _ns("2026-10-25T00:00:00Z")
    end = _ns("2026-10-30T00:00:00Z")
    _write_run(tmp_path, candidate, 0, "base", 20261026, 0.5, start, end)
    path = (
        tmp_path / "window-1" / "base" / "seed-20261026"
        / candidate / "native-run.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    report[field] = wrong_value
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="identity or execution evidence"):
        _write_screen_summary(tmp_path, prereg, errata)


def test_window_summary_rejects_mismatched_source_marker_within_pair(tmp_path: Path) -> None:
    prereg, errata = _protocol()
    candidate = _candidate_id(errata["corrected_candidate_matrix"][0])
    start = _ns("2026-10-25T00:00:00Z")
    end = _ns("2026-10-30T00:00:00Z")
    _write_run(
        tmp_path, candidate, 0, "base", 20261026, 0.5, start, end,
        marker_sha="a" * 64,
    )
    _write_run(
        tmp_path, "FLAT", 0, "base", 20261026, 0.0, start, end,
        marker_sha="b" * 64,
    )
    with pytest.raises(RuntimeError, match="identity"):
        _write_screen_summary(tmp_path, prereg, errata)


def test_missing_controls_cannot_pass_the_teacher_screen(tmp_path: Path) -> None:
    prereg, errata = _protocol()
    candidate = _candidate_id(errata["corrected_candidate_matrix"][0])
    _write_run(
        tmp_path, candidate, 0, "base", 20261026, 0.5,
        _ns("2026-10-25T00:00:00Z"), _ns("2026-10-30T00:00:00Z"),
    )

    report = _write_screen_summary(tmp_path, prereg, errata)
    assert report["screen_complete"] is False
    assert report["qualifying_candidates"] == []
    assert report["student_training_eligible"] is False



@pytest.mark.parametrize("field", list(CONTEXT) + [
    "market_marker_sha256", "market_snapshot_sha256", "import_manifest_sha256",
    "cost_recipe_sha256", "teacher_recipe_sha256", "candidate_id",
    "exit_variant", "scenario", "seed", "window_ns",
])
def test_identity_tampering_rejected_by_resume_and_summary(tmp_path, field):
    candidate = "T01_EMA_9_21"
    start = _ns("2026-10-25T00:00:00Z")
    end = _ns("2026-10-30T00:00:00Z")
    marker = "a" * 63 + "0"
    report = _native_run(candidate=candidate, scenario="base", seed=20261026,
                         net_return=0.1, window_start=start, window_end=end,
                         marker_sha=marker)
    expected_identity = _identity(candidate, "base", 20261026, start, end,
                                  marker, report["teacher_recipe"])
    expected = {"protocol_id": "frozen-protocol", "teacher": candidate,
                "teacher_recipe": report["teacher_recipe"], "marker_sha256": marker,
                "window_ns": [start, end], "scenario": "base", "seed": 20261026,
                "run_identity": expected_identity}
    path = tmp_path / "window-1" / "base" / "seed-20261026" / candidate
    path.mkdir(parents=True)
    file = path / "native-run.json"
    file.write_text(json.dumps(report), encoding="utf-8")
    assert _completed_run(path, expected) == report
    report["run_identity"][field] = "changed"
    file.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity"):
        _completed_run(path, expected)
    prereg, errata = _protocol()
    with pytest.raises(RuntimeError, match="identity"):
        _write_screen_summary(tmp_path, prereg, errata)


def test_legacy_report_without_identity_rejected(tmp_path):
    start = _ns("2026-10-25T00:00:00Z")
    end = _ns("2026-10-30T00:00:00Z")
    marker = "a" * 63 + "0"
    report = _native_run(candidate="T01_EMA_9_21", scenario="base", seed=20261026,
                         net_return=0.1, window_start=start, window_end=end,
                         marker_sha=marker)
    identity = _identity("T01_EMA_9_21", "base", 20261026, start, end,
                         marker, report["teacher_recipe"])
    del report["run_identity"]
    path = tmp_path / "window-1" / "base" / "seed-20261026" / "T01_EMA_9_21"
    path.mkdir(parents=True)
    (path / "native-run.json").write_text(json.dumps(report), encoding="utf-8")
    expected = {"protocol_id": "frozen-protocol", "teacher": "T01_EMA_9_21",
                "teacher_recipe": report["teacher_recipe"], "marker_sha256": marker,
                "window_ns": [start, end], "scenario": "base", "seed": 20261026,
                "run_identity": identity}
    with pytest.raises(RuntimeError, match="LEGACY_INCOMPATIBLE"):
        _completed_run(path, expected)
    prereg, errata = _protocol()
    with pytest.raises(RuntimeError, match="LEGACY_INCOMPATIBLE"):
        _write_screen_summary(tmp_path, prereg, errata)


@pytest.mark.parametrize(("section", "field", "value"), [
    ("execution", "scaler_hash", "d" * 64),
    ("execution", "scenario_costs", {"taker_fee_bps_per_leg": 0}),
    ("source", "snapshot_hash", "d" * 64),
    ("source", "marker_sha256", "d" * 64),
    ("teacher_recipe", "target_exposure", 0.9),
])
def test_identity_also_binds_report_facts(tmp_path, section, field, value):
    start = _ns("2026-10-25T00:00:00Z")
    end = _ns("2026-10-30T00:00:00Z")
    marker = "a" * 63 + "0"
    report = _native_run(candidate="T01_EMA_9_21", scenario="base", seed=20261026,
                         net_return=0.1, window_start=start, window_end=end,
                         marker_sha=marker)
    identity = _identity("T01_EMA_9_21", "base", 20261026, start, end,
                         marker, report["teacher_recipe"])
    expected = {"protocol_id": "frozen-protocol", "teacher": "T01_EMA_9_21",
                "teacher_recipe": report["teacher_recipe"].copy(), "marker_sha256": marker,
                "window_ns": [start, end], "scenario": "base", "seed": 20261026,
                "run_identity": identity}
    report[section][field] = value
    path = tmp_path / "run"
    path.mkdir()
    (path / "native-run.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity|Identity|frozen run key"):
        _completed_run(path, expected)
