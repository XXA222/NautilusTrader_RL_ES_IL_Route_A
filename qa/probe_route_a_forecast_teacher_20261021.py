"""Fit a causal cost-hurdle forecast teacher on Train and replay it natively."""

from __future__ import annotations

from hashlib import sha256
import json
import os

import numpy as np

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.evaluation import NativePair, screen_teacher
from rles_route_a.teachers.forecast import RidgeForecastTeacher
from rles_route_a.teachers.wait import WaitTeacher
from rles_route_a.integration.l2_adapter import (
    CAUSAL_MTF_V5_ROUTE_A_FEATURES,
    CAUSAL_MTF_V6_ROUTE_A_FEATURES,
)

import run_route_a_trainfold_screen_20261021 as trainfold
import run_route_a_train_student_screen_20261020 as runner


RUN_ID = os.environ.get("ROUTE_A_FORECAST_RUN", "run01")
HORIZON_MINUTES = int(os.environ.get("ROUTE_A_FORECAST_HORIZON_MINUTES", "60"))
OUT = runner.ROOT / ".artifacts" / f"route-a-forecast-teacher-20261021-{RUN_ID}"
FIT_START = "2024-04-16T00:00:00+00:00"
FIT_END = "2024-05-15T00:00:00+00:00"
HORIZON_NS = HORIZON_MINUTES * runner.MINUTE_NS
SAMPLE_INTERVAL_MINUTES = int(os.environ.get("ROUTE_A_FORECAST_SAMPLE_MINUTES", "5"))
SAMPLE_NS = SAMPLE_INTERVAL_MINUTES * runner.MINUTE_NS
LABEL_MAX_LAG_NS = 2 * runner.MINUTE_NS
RIDGE_ALPHA = float(os.environ.get("ROUTE_A_FORECAST_RIDGE_ALPHA", "0.1"))
FEATURE_SET = os.environ.get("ROUTE_A_FORECAST_FEATURE_SET", "v6")
DECISION_INTERVAL_MINUTES = int(
    os.environ.get("ROUTE_A_FORECAST_DECISION_INTERVAL_MINUTES", "0")
)
ENTRY_HURDLE_BPS = 30.0  # Base 12 bps; Stress 30 bps under scenario fee/slippage assumptions.
EXIT_HURDLE_BPS = 15.0
TEACHER_SIZE = 0.2
V6_FEATURE_MAP = (
    ("trend_1h", "closed_return_1h"),
    ("trend_1d", "closed_return_1d"),
    ("vol_1m", "volatility_1m"),
    ("zscore_1m", "zscore_1m"),
    ("efficiency_15m", "trend_efficiency_15m"),
    ("efficiency_1h", "trend_efficiency_60m"),
    ("return_1m", "return_1m"),
    ("return_5m", "return_5m"),
    ("return_15m", "return_15m"),
    ("return_60m", "return_60m"),
    ("volatility_5m", "volatility_5m"),
    ("volatility_15m", "volatility_15m"),
    ("volatility_60m", "volatility_60m"),
    ("volume_5m", "volume_5m"),
    ("basis", "basis"),
    ("last_settled_funding", "last_settled_funding"),
)
FEATURE_MAP = (
    tuple(CAUSAL_MTF_V5_ROUTE_A_FEATURES.items())
    if FEATURE_SET == "v5" else V6_FEATURE_MAP
)
MARKET_FEATURE_MAP = (
    CAUSAL_MTF_V5_ROUTE_A_FEATURES
    if FEATURE_SET == "v5" else CAUSAL_MTF_V6_ROUTE_A_FEATURES
)
if FEATURE_SET not in {"v5", "v6"}:
    raise ValueError("ROUTE_A_FORECAST_FEATURE_SET must be v5 or v6")
WINDOWS = (
    ("jun04_h00", "2024-06-04T00:00:00+00:00", "2024-06-04T12:00:00+00:00"),
    ("jun07_h00", "2024-06-07T00:00:00+00:00", "2024-06-07T12:00:00+00:00"),
    ("jun12_h00", "2024-06-12T00:00:00+00:00", "2024-06-12T12:00:00+00:00"),
    ("jun14_h00", "2024-06-14T00:00:00+00:00", "2024-06-14T12:00:00+00:00"),
)
SEEDS = (11, 29)
SCENARIOS = ("base", "stress")
EPISODE_STEPS = 700


def fit_model(factory) -> tuple[RidgeForecastTeacher, dict]:
    field_names = tuple(field.name for field in factory.features.fields)
    field_index = {host_name: field_names.index(host_name) for _, host_name in FEATURE_MAP}
    close_index = field_names.index("close_1m")
    rows = factory.rows
    decision_times = np.asarray([row.decision_ts_ns for row in rows], dtype=np.int64)
    feature_rows = np.asarray(
        [[row.values[field_index[host_name]] for _, host_name in FEATURE_MAP] for row in rows],
        dtype=np.float64,
    )
    closes = np.asarray([row.values[close_index] for row in rows], dtype=np.float64)
    if (feature_rows.ndim != 2 or not np.isfinite(feature_rows).all()
            or not np.isfinite(closes).all() or np.any(np.diff(decision_times) <= 0)):
        raise RuntimeError("causal Train feature rows are invalid or unordered")

    grid = np.arange(
        runner.ns(FIT_START), runner.ns(FIT_END) - HORIZON_NS,
        SAMPLE_NS, dtype=np.int64,
    )
    selected = np.searchsorted(decision_times, grid, side="left")
    keep = selected < len(rows)
    selected = selected[keep]
    grid = grid[keep]
    aligned = ((decision_times[selected] >= grid)
               & (decision_times[selected] - grid <= LABEL_MAX_LAG_NS)
               & (decision_times[selected] + HORIZON_NS <= runner.ns(FIT_END)))
    selected = selected[aligned]
    minimum_rows = max(100, 20 * HORIZON_MINUTES // SAMPLE_INTERVAL_MINUTES)
    if selected.size < minimum_rows:
        raise RuntimeError(f"insufficient purged Train forecast rows: {selected.size}")

    future_indices = np.searchsorted(decision_times, decision_times[selected] + HORIZON_NS,
                                     side="left")
    valid = future_indices < len(rows)
    lag = np.full(selected.size, np.iinfo(np.int64).max, dtype=np.int64)
    valid_positions = np.flatnonzero(valid)
    lag[valid_positions] = (
        decision_times[future_indices[valid_positions]]
        - (decision_times[selected[valid_positions]] + HORIZON_NS)
    )
    valid &= lag <= LABEL_MAX_LAG_NS
    selected, future_indices = selected[valid], future_indices[valid]
    if (selected.size < minimum_rows
            or np.any(decision_times[future_indices] > runner.ns(FIT_END))):
        raise RuntimeError("forward label escaped the purged Train fit boundary")

    x = feature_rows[selected]
    y = np.log(closes[future_indices] / closes[selected]) * 10_000.0
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    z = (x - mean) / std
    design = np.column_stack((np.ones(z.shape[0]), z))
    gram = design.T @ design / design.shape[0]
    rhs = design.T @ y / design.shape[0]
    penalty = np.eye(gram.shape[0]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(gram + penalty, rhs)
    prediction = design @ coefficients
    selected_long = prediction >= ENTRY_HURDLE_BPS
    selected_short = prediction <= -ENTRY_HURDLE_BPS
    diagnostics = {
        "sample_interval_minutes": SAMPLE_INTERVAL_MINUTES,
        "horizon_minutes": HORIZON_MINUTES,
        "purged_fit_end_ns": runner.ns(FIT_END),
        "samples": int(x.shape[0]),
        "features": [name for name, _ in FEATURE_MAP],
        "feature_projection": f"CAUSAL_MTF_{FEATURE_SET.upper()}_ROUTE_A_FEATURES",
        "decision_interval_minutes": DECISION_INTERVAL_MINUTES,
        "feature_means": mean.tolist(),
        "feature_stds": std.tolist(),
        "ridge_alpha": RIDGE_ALPHA,
        "coefficients_intercept_then_features": coefficients.tolist(),
        "label_mean_bps": float(np.mean(y)),
        "label_std_bps": float(np.std(y)),
        "prediction_mean_bps_train_only": float(np.mean(prediction)),
        "prediction_std_bps_train_only": float(np.std(prediction)),
        "train_fraction_above_entry_hurdle": float(np.mean(selected_long)),
        "train_fraction_below_entry_hurdle": float(np.mean(selected_short)),
        "train_selected_gross_bps": (
            float(np.mean(y[selected_long])) if np.any(selected_long) else None
        ),
        "train_selected_short_gross_bps": (
            float(np.mean(-y[selected_short])) if np.any(selected_short) else None
        ),
        "train_metrics_scope": "fit diagnostics only; not generalization evidence",
    }
    teacher = RidgeForecastTeacher(
        feature_names=tuple(name for name, _ in FEATURE_MAP),
        means=tuple(map(float, mean)),
        stds=tuple(map(float, std)),
        coefficients=tuple(map(float, coefficients)),
        entry_hurdle_bps=ENTRY_HURDLE_BPS,
        exit_hurdle_bps=EXIT_HURDLE_BPS,
        size=TEACHER_SIZE,
        forecast_horizon_minutes=HORIZON_MINUTES,
        decision_interval_minutes=DECISION_INTERVAL_MINUTES,
        expert_id=(f"ridge_{HORIZON_MINUTES}m_forecast_30bp_hurdle_v4_"
                   f"{FEATURE_SET}_{SAMPLE_INTERVAL_MINUTES}m_alpha{RIDGE_ALPHA:g}_"
                   f"decision{DECISION_INTERVAL_MINUTES}m"),
    )
    return teacher, diagnostics


def main() -> int:
    runner.configure_torch_threads(runner.torch)
    trainfold.configure()
    if not 1 <= HORIZON_MINUTES <= 1_440:
        raise ValueError("forecast horizon must be within the one-day causal feature scope")
    if OUT.exists():
        raise FileExistsError(f"immutable forecast teacher output already exists: {OUT}")
    score_start, score_end = (runner.ns("2024-04-16T00:00:00+00:00"),
                              runner.ns("2024-06-15T00:00:00+00:00"))
    for window_id, start, end in WINDOWS:
        if runner.ns(start) < score_start or runner.ns(end) > score_end:
            raise PermissionError(f"forecast screen outside frozen Train interval: {window_id}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    _, fit_factory = runner.make_config(
        label="forecast-fit-apr16-may15",
        marker_sha=marker_sha,
        scope_start=scope_start,
        scope_end=scope_end,
        start_ns=runner.ns(FIT_START),
        end_ns=runner.ns(FIT_END),
        mode="base",
        fit_scaler=True,
    )
    teacher, fit_diagnostics = fit_model(fit_factory)
    scaler = fit_factory.scaler
    scaler_hash = scaler.hash
    baseline = WaitTeacher(expert_id="wait_flat_baseline_v1")
    baseline_hash = runner.teacher_config_hash(baseline)
    teacher_hash = runner.teacher_config_hash(teacher)
    del fit_factory

    owner = ACTIVE_OWNER.set("route-a-forecast-teacher-probe-20261021")
    try:
        results = []
        pairs = []
        evaluation_recipe_hashes = {}
        for window_id, start, end in WINDOWS:
            for scenario in SCENARIOS:
                _, factory = runner.make_config(
                    label=f"forecast-{window_id}-{scenario}",
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=runner.ns(start),
                    end_ns=runner.ns(end),
                    mode=scenario,
                    fit_scaler=False,
                    scaler=scaler,
                )
                for seed in SEEDS:
                    label = f"{window_id}-{scenario}-seed{seed}"
                    teacher_run = runner.run_episode(
                        factory, teacher, marker_sha, label + "-teacher", seed, EPISODE_STEPS,
                        market_feature_map=MARKET_FEATURE_MAP,
                    )
                    baseline_run = runner.run_episode(
                        factory, baseline, marker_sha, label + "-baseline", seed, EPISODE_STEPS,
                        market_feature_map=MARKET_FEATURE_MAP,
                    )
                    matched = (teacher_run["barriers"] == baseline_run["barriers"] == EPISODE_STEPS)
                    evidence_sha = {
                        role: runner.write_verified(
                            OUT / "native-evidence" / f"{label}-{role}.json", result["evidence"]
                        )
                        for role, result in (("teacher", teacher_run), ("baseline", baseline_run))
                    }
                    pairs.append(NativePair(
                        window_id=window_id,
                        scenario=scenario,
                        seed=seed,
                        teacher_recipe=teacher_hash,
                        baseline_recipe=baseline_hash,
                        source_identity=marker_sha,
                        teacher_net_return=teacher_run["net_return"],
                        baseline_net_return=baseline_run["net_return"],
                        teacher_max_drawdown=teacher_run["max_drawdown"],
                        baseline_max_drawdown=baseline_run["max_drawdown"],
                        teacher_filled_orders=teacher_run["filled_orders"],
                        teacher_liquidations=teacher_run["liquidations"],
                        matched_native_barriers=matched,
                        scenario_only=True,
                        nonoverlap_verified=True,
                    ))
                    results.append({
                        "window_id": window_id,
                        "scenario": scenario,
                        "seed": seed,
                        "teacher_net_return": teacher_run["net_return"],
                        "baseline_net_return": baseline_run["net_return"],
                        "teacher_max_drawdown": teacher_run["max_drawdown"],
                        "teacher_filled_orders": teacher_run["filled_orders"],
                        "teacher_liquidations": teacher_run["liquidations"],
                        "matched_native_barriers": matched,
                        "evidence_sha256": evidence_sha,
                    })
                    print(f"completed {label}", flush=True)
                evaluation_recipe_hashes[scenario] = factory.route_a_execution_recipe_hash
                del factory

        teacher_screen = screen_teacher(pairs)
        report = {
            "schema": "rles.route_a.forecast_teacher_probe.v1",
            "purpose": "Causal cost-hurdle model fitted on earlier Train; diagnostic replay only, not independent promotion evidence",
            "source_marker_sha256": marker_sha,
            "wfo_report_sha256": sha256(trainfold.WFO_REPORT.read_bytes()).hexdigest(),
            "fit_window_ns": [runner.ns(FIT_START), runner.ns(FIT_END)],
            "fit_diagnostics": fit_diagnostics,
            "fit_scaler_hash": scaler_hash,
            "teacher_recipe_sha256": teacher_hash,
            "evaluation_execution_recipe_sha256_by_scenario": evaluation_recipe_hashes,
            "cost_hurdle": {
                "entry_bps": ENTRY_HURDLE_BPS,
                "exit_bps": EXIT_HURDLE_BPS,
                "base_round_trip_scenario_bps": 12.0,
                "stress_round_trip_scenario_bps": 30.0,
                "status": "scenario-only assumptions; not historical cost evidence",
            },
            "windows": [
                {"id": window_id, "start_ns": runner.ns(start), "end_ns": runner.ns(end)}
                for window_id, start, end in WINDOWS
            ],
            "scenarios": list(SCENARIOS),
            "seeds": list(SEEDS),
            "episode_steps": EPISODE_STEPS,
            "teacher_screen": {
                "status": teacher_screen.status,
                "economic_status": teacher_screen.economic_status,
                "pair_count": teacher_screen.pair_count,
                "mean_excess_return": teacher_screen.mean_excess_return,
                "worst_excess_return": teacher_screen.worst_excess_return,
                "reasons": list(teacher_screen.reasons),
                "receipt_sha256": teacher_screen.receipt_sha256,
            },
            "results": results,
            "economic_status": "SCENARIO_ONLY",
        }
        runner.write_verified(OUT / "forecast-teacher-report.json", report)
        print(json.dumps({
            "output": str(OUT),
            "teacher_status": teacher_screen.status,
            "reasons": list(teacher_screen.reasons),
            "mean_excess_return": teacher_screen.mean_excess_return,
            "worst_excess_return": teacher_screen.worst_excess_return,
        }, indent=2))
        return 0
    finally:
        ACTIVE_OWNER.reset(owner)


if __name__ == "__main__":
    raise SystemExit(main())
