"""Nested chronological confidence-gate calibration for the Route A event GRU.

Within each expanding Train fold, an early block fits the model, the last five
training days select an OPEN probability threshold, and the next outer block
measures it. Threshold selection requires at least five calibration trades and
an approximate one-sided 90% lower bound on after-cost return above zero.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

import probe_route_a_temporal_event_classifier_teacher_20261021 as temporal
import run_route_a_trainfold_screen_20261021 as trainfold
import run_route_a_train_student_screen_20261020 as runner
from rles_orchestrator.research_access import ACTIVE_OWNER


RUN_ID = os.environ.get("ROUTE_A_NESTED_CALIBRATION_RUN", "run01")
OUT = runner.ROOT / ".artifacts" / f"route-a-nested-event-calibration-20261021-{RUN_ID}"
CALIBRATION_DAYS = 5
MINIMUM_CALIBRATION_TRADES = 5
LOWER_BOUND_Z = 1.645
THRESHOLDS = tuple(np.round(np.arange(0.50, 0.901, 0.05), 2).tolist())
TRAIN_GRID_MINUTES = temporal.TRAIN_SAMPLE_MINUTES
EVALUATION_GRID_MINUTES = temporal.EVALUATION_SAMPLE_MINUTES


def rows_with_history(start_ns: int, end_ns: int, sample_minutes: int, times, closes, features):
    entries, returns, decision_times = temporal.anchor_rows(
        times, closes, start_ns, end_ns, sample_minutes
    )
    sequences, retained_entries = temporal.make_sequences(entries, times, features)
    retained_times = times[retained_entries]
    keep = np.isin(decision_times, retained_times)
    return sequences, returns[keep], retained_times


def threshold_metrics(probabilities: np.ndarray, returns_bps: np.ndarray, threshold: float) -> dict:
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities[:, 1:], axis=1)
    selected = (predicted != 0) & (confidence >= threshold)
    side = np.where(predicted == 1, 1.0, -1.0)
    net = returns_bps[selected] * side[selected] - temporal.ENTRY_COST_HURDLE_BPS
    count = int(np.sum(selected))
    mean = float(np.mean(net)) if count else None
    std = float(np.std(net, ddof=1)) if count > 1 else None
    lower_bound = (
        mean - LOWER_BOUND_Z * std / np.sqrt(count)
        if count > 1 and std is not None else None
    )
    return {
        "threshold": threshold,
        "selected_count": count,
        "selected_long_count": int(np.sum(selected & (predicted == 1))),
        "selected_short_count": int(np.sum(selected & (predicted == 2))),
        "net_mean_bps_after_hurdle": mean,
        "net_std_bps": std,
        "approx_one_sided_90pct_lower_bound_bps": lower_bound,
        "positive_net_count": int(np.sum(net > 0)),
        "eligible": bool(
            count >= MINIMUM_CALIBRATION_TRADES
            and lower_bound is not None and lower_bound > 0
        ),
    }


def evaluate_at_threshold(probabilities: np.ndarray, returns_bps: np.ndarray,
                          threshold: float | None) -> dict:
    if threshold is None:
        return {
            "status": "NO_CALIBRATION_THRESHOLD",
            "selected_count": 0,
            "selected_net_mean_bps_after_hurdle": None,
            "selected_net_positive_count": 0,
        }
    result = threshold_metrics(probabilities, returns_bps, threshold)
    return {"status": "CALIBRATED_THRESHOLD_APPLIED", **result}


def main() -> int:
    import torch

    runner.configure_torch_threads(torch)
    torch.use_deterministic_algorithms(True)
    trainfold.configure()
    if OUT.exists():
        raise FileExistsError(f"immutable nested calibration output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner = ACTIVE_OWNER.set("route-a-nested-event-calibration-20261021")
    try:
        _, factory = runner.make_config(
            label="nested-event-calibration-apr16-may15",
            marker_sha=marker_sha,
            scope_start=scope_start,
            scope_end=scope_end,
            start_ns=runner.ns(temporal.FIT_START),
            end_ns=runner.ns(temporal.ANALYSIS_END),
            mode="base",
            fit_scaler=True,
        )
    finally:
        ACTIVE_OWNER.reset(owner)

    rows = factory.rows
    times = np.asarray([row.decision_ts_ns for row in rows], dtype=np.int64)
    field_names = tuple(field.name for field in factory.features.fields)
    feature_indices = [field_names.index(name) for name in temporal.FEATURES]
    close_index = field_names.index("close_1m")
    features = np.asarray(
        [[row.values[index] for index in feature_indices] for row in rows], dtype=np.float64
    )
    closes = np.asarray([row.values[close_index] for row in rows], dtype=np.float64)
    if (not np.isfinite(features).all() or not np.isfinite(closes).all()
            or np.any(np.diff(times) <= 0)):
        raise RuntimeError("causal Train feature rows are non-finite or unordered")

    folds = []
    for fold_id, train_start_text, train_end_text, eval_start_text, eval_end_text in temporal.FOLDS:
        train_start, train_end = runner.ns(train_start_text), runner.ns(train_end_text)
        eval_start, eval_end = runner.ns(eval_start_text), runner.ns(eval_end_text)
        calibration_start = train_end - CALIBRATION_DAYS * 1_440 * runner.MINUTE_NS

        x_fit, y_fit, fit_times = rows_with_history(
            train_start, calibration_start, TRAIN_GRID_MINUTES, times, closes, features
        )
        x_cal, y_cal, cal_times = rows_with_history(
            calibration_start, train_end, EVALUATION_GRID_MINUTES, times, closes, features
        )
        if len(x_fit) < 500 or len(x_cal) < MINIMUM_CALIBRATION_TRADES:
            raise RuntimeError(f"insufficient purged nested calibration rows in {fold_id}")
        if (int(np.max(fit_times) + temporal.HORIZON_MINUTES * runner.MINUTE_NS)
                > calibration_start):
            raise RuntimeError(f"model-fit labels overlap calibration block in {fold_id}")
        fit_labels = temporal.event_labels(y_fit)
        calibration_probabilities, _ = temporal.fit_classifier(x_fit, fit_labels, x_cal)
        candidates = [threshold_metrics(calibration_probabilities, y_cal, threshold)
                      for threshold in THRESHOLDS]
        eligible = [row for row in candidates if row["eligible"]]
        chosen = (
            max(eligible, key=lambda row: (
                row["approx_one_sided_90pct_lower_bound_bps"],
                row["net_mean_bps_after_hurdle"],
                row["selected_count"],
                row["threshold"],
            )) if eligible else None
        )
        selected_threshold = chosen["threshold"] if chosen else None

        x_full, y_full, full_times = rows_with_history(
            train_start, train_end, TRAIN_GRID_MINUTES, times, closes, features
        )
        x_eval, y_eval, eval_times = rows_with_history(
            eval_start, eval_end, EVALUATION_GRID_MINUTES, times, closes, features
        )
        if len(x_full) < 500 or len(x_eval) < 10:
            raise RuntimeError(f"insufficient outer evaluation rows in {fold_id}")
        if int(np.max(full_times) + temporal.HORIZON_MINUTES * runner.MINUTE_NS) > eval_start:
            raise RuntimeError(f"outer fit labels overlap evaluation in {fold_id}")
        full_probabilities, full_fit_meta = temporal.fit_classifier(
            x_full, temporal.event_labels(y_full), x_eval
        )
        calibrated_eval = evaluate_at_threshold(
            full_probabilities, y_eval, selected_threshold
        )
        fixed_eval = evaluate_at_threshold(
            full_probabilities, y_eval, temporal.OPEN_PROBABILITY_THRESHOLD
        )
        folds.append({
            "fold_id": fold_id,
            "train_start_ns": train_start,
            "train_end_ns_exclusive": train_end,
            "calibration_start_ns": calibration_start,
            "calibration_end_ns_exclusive": train_end,
            "evaluation_start_ns": eval_start,
            "evaluation_end_ns_exclusive": eval_end,
            "fit_label_end_max_ns": int(
                np.max(fit_times) + temporal.HORIZON_MINUTES * runner.MINUTE_NS
            ),
            "calibration_label_end_max_ns": int(
                np.max(cal_times) + temporal.HORIZON_MINUTES * runner.MINUTE_NS
            ),
            "outer_fit_label_end_max_ns": int(
                np.max(full_times) + temporal.HORIZON_MINUTES * runner.MINUTE_NS
            ),
            "fit_class_counts": {
                name: int(count) for name, count in zip(
                    temporal.CLASS_NAMES,
                    np.bincount(fit_labels, minlength=len(temporal.CLASS_NAMES)),
                    strict=True,
                )
            },
            "calibration_sample_count": int(len(y_cal)),
            "calibration_event_class_counts": {
                name: int(count) for name, count in zip(
                    temporal.CLASS_NAMES,
                    np.bincount(temporal.event_labels(y_cal),
                                minlength=len(temporal.CLASS_NAMES)),
                    strict=True,
                )
            },
            "threshold_candidates": candidates,
            "selected_threshold": selected_threshold,
            "selection_status": "CALIBRATED" if chosen else "NO_THRESHOLD_MEETS_TRAIN_CRITERIA",
            "selected_calibration_metrics": chosen,
            "outer_fit_samples": int(len(y_full)),
            "outer_model_fit": full_fit_meta,
            "calibrated_outer_evaluation": calibrated_eval,
            "fixed_0_60_comparison": fixed_eval,
        })

    report = {
        "schema": "rles.route-a.nested-event-calibration.v1",
        "purpose": "Nested chronological Train-inner threshold selection; not native execution or promotion evidence",
        "economic_status": "SCENARIO_ONLY",
        "source_marker_sha256": marker_sha,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "wfo_report_sha256": sha256(trainfold.WFO_REPORT.read_bytes()).hexdigest(),
        "lookback_minutes": temporal.LOOKBACK_MINUTES,
        "history_sample_minutes": temporal.HISTORY_SAMPLE_MINUTES,
        "event_target_bps": temporal.EVENT_TARGET_BPS,
        "cost_hurdle_bps": temporal.ENTRY_COST_HURDLE_BPS,
        "calibration_policy": {
            "calibration_tail_days": CALIBRATION_DAYS,
            "threshold_grid": list(THRESHOLDS),
            "minimum_trades": MINIMUM_CALIBRATION_TRADES,
            "selection_metric": "net_mean_bps - 1.645 * sample_std_bps / sqrt(n)",
            "required_approx_lower_bound_bps": "> 0",
            "warning": "approximate one-sided 90% bound; dependent observations and threshold search limit inferential meaning",
        },
        "folds": folds,
    }
    output = OUT / "nested-event-calibration-report.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest, encoding="ascii")
    calibrated = [row["calibrated_outer_evaluation"] for row in folds]
    traded = [row for row in calibrated if row.get("selected_count", 0)]
    trade_count = sum(row["selected_count"] for row in traded)
    aggregate = {
        "folds_with_selected_threshold": sum(row["selected_threshold"] is not None for row in folds),
        "folds_with_outer_trades": len(traded),
        "outer_trades": trade_count,
        "outer_net_bps_trade_weighted": (
            float(np.average(
                [row["net_mean_bps_after_hurdle"] for row in traded],
                weights=[row["selected_count"] for row in traded],
            )) if traded else None
        ),
        "outer_positive_folds": sum(
            row.get("net_mean_bps_after_hurdle") is not None
            and row["net_mean_bps_after_hurdle"] > 0 for row in traded
        ),
    }
    print(json.dumps({"output": str(OUT), "report_sha256": digest, "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
