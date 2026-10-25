"""Purged walk-forward diagnostics for Route A forecast inputs and Ridge shrinkage.

All fits and validation blocks stay inside the frozen Train interval. Labels
are sampled every four hours for a four-hour horizon to avoid overlapping
forward-return outcomes in the primary diagnostic.
"""

from __future__ import annotations

from hashlib import sha256
import json
import numpy as np

import run_route_a_trainfold_screen_20261021 as trainfold
import run_route_a_train_student_screen_20261020 as runner
from rles_orchestrator.research_access import ACTIVE_OWNER


OUT = runner.ROOT / ".artifacts" / "route-a-forecast-walkforward-20261021-run02"
FIT_START = "2024-04-16T00:00:00+00:00"
ANALYSIS_END = "2024-05-15T00:00:00+00:00"
HORIZON_MINUTES = 240
SAMPLE_MINUTES = 240
MAX_ALIGNMENT_LAG_MINUTES = 2
ALPHAS = (0.1, 1.0, 10.0, 100.0)
ENTRY_HURDLE_BPS = 30.0
FEATURE_SETS = {
    "v5_six": (
        "closed_return_1h", "closed_return_1d", "volatility_1m", "zscore_1m",
        "trend_efficiency_15m", "trend_efficiency_60m",
    ),
    "v6_sixteen": (
        "closed_return_1h", "closed_return_1d", "volatility_1m", "zscore_1m",
        "trend_efficiency_15m", "trend_efficiency_60m", "return_1m", "return_5m",
        "return_15m", "return_60m", "volatility_5m", "volatility_15m",
        "volatility_60m", "volume_5m", "basis", "last_settled_funding",
    ),
}
FOLDS = (
    ("fold01", "2024-04-16T00:00:00+00:00", "2024-04-26T00:00:00+00:00",
     "2024-04-26T00:00:00+00:00", "2024-05-01T00:00:00+00:00"),
    ("fold02", "2024-04-16T00:00:00+00:00", "2024-05-01T00:00:00+00:00",
     "2024-05-01T00:00:00+00:00", "2024-05-06T00:00:00+00:00"),
    ("fold03", "2024-04-16T00:00:00+00:00", "2024-05-06T00:00:00+00:00",
     "2024-05-06T00:00:00+00:00", "2024-05-11T00:00:00+00:00"),
    ("fold04", "2024-04-16T00:00:00+00:00", "2024-05-11T00:00:00+00:00",
     "2024-05-11T00:00:00+00:00", "2024-05-15T00:00:00+00:00"),
)


def aligned_rows(times: np.ndarray, closes: np.ndarray, features: np.ndarray,
                 start_ns: int, end_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_ns = SAMPLE_MINUTES * runner.MINUTE_NS
    horizon_ns = HORIZON_MINUTES * runner.MINUTE_NS
    lag_ns = MAX_ALIGNMENT_LAG_MINUTES * runner.MINUTE_NS
    grid = np.arange(start_ns, end_ns - horizon_ns, step_ns, dtype=np.int64)
    entry = np.searchsorted(times, grid, side="left")
    keep = entry < len(times)
    grid, entry = grid[keep], entry[keep]
    aligned = ((times[entry] >= grid)
               & (times[entry] - grid <= lag_ns)
               & (times[entry] + horizon_ns <= end_ns))
    entry = entry[aligned]
    exit_idx = np.searchsorted(times, times[entry] + horizon_ns, side="left")
    valid = exit_idx < len(times)
    lag = np.full(entry.size, np.iinfo(np.int64).max, dtype=np.int64)
    positions = np.flatnonzero(valid)
    lag[positions] = times[exit_idx[positions]] - (times[entry[positions]] + horizon_ns)
    valid &= lag <= lag_ns
    entry, exit_idx = entry[valid], exit_idx[valid]
    return features[entry], np.log(closes[exit_idx] / closes[entry]) * 10_000.0, times[entry]


def fit_predict(x_train: np.ndarray, y_train: np.ndarray,
                x_eval: np.ndarray, alpha: float) -> tuple[np.ndarray, dict]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    train_z = (x_train - mean) / std
    eval_z = (x_eval - mean) / std
    train_design = np.column_stack((np.ones(len(train_z)), train_z))
    eval_design = np.column_stack((np.ones(len(eval_z)), eval_z))
    gram = train_design.T @ train_design / len(train_design)
    rhs = train_design.T @ y_train / len(train_design)
    penalty = np.eye(gram.shape[0]) * alpha
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(gram + penalty, rhs)
    return eval_design @ coefficient, {
        "means": mean.tolist(), "stds": std.tolist(),
        "coefficients_intercept_then_features": coefficient.tolist(),
    }


def summarize(prediction: np.ndarray, labels: np.ndarray) -> dict:
    long = prediction >= ENTRY_HURDLE_BPS
    short = prediction <= -ENTRY_HURDLE_BPS
    selected = long | short
    realized = np.where(long, labels, -labels)
    net = realized[selected] - ENTRY_HURDLE_BPS
    return {
        "n": int(len(labels)),
        "prediction_mean_bps": float(np.mean(prediction)) if len(prediction) else None,
        "prediction_std_bps": float(np.std(prediction)) if len(prediction) else None,
        "rmse_bps": float(np.sqrt(np.mean((prediction - labels) ** 2)))
        if len(labels) else None,
        "selected_count": int(np.sum(selected)),
        "selected_long_count": int(np.sum(long)),
        "selected_short_count": int(np.sum(short)),
        "selected_gross_bps": float(np.mean(realized[selected])) if np.any(selected) else None,
        "selected_after_30bp_hurdle_bps": float(np.mean(net)) if net.size else None,
        "selected_positive_after_hurdle": int(np.sum(net > 0)),
    }


def main() -> int:
    trainfold.configure()
    if OUT.exists():
        raise FileExistsError(f"immutable walk-forward output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    start_ns, end_ns = runner.ns(FIT_START), runner.ns(ANALYSIS_END)
    owner = ACTIVE_OWNER.set("route-a-forecast-walkforward-20261021")
    try:
        _, factory = runner.make_config(
            label="walkforward-full-train-features",
            marker_sha=marker_sha,
            scope_start=scope_start,
            scope_end=scope_end,
            start_ns=start_ns,
            end_ns=end_ns,
            mode="base",
            fit_scaler=True,
        )
    finally:
        ACTIVE_OWNER.reset(owner)

    rows = factory.rows
    times = np.asarray([row.decision_ts_ns for row in rows], dtype=np.int64)
    field_names = tuple(field.name for field in factory.features.fields)
    close_index = field_names.index("close_1m")
    closes = np.asarray([row.values[close_index] for row in rows], dtype=np.float64)
    variants = {}
    for name, fields in FEATURE_SETS.items():
        indices = [field_names.index(field) for field in fields]
        x = np.asarray([[row.values[i] for i in indices] for row in rows], dtype=np.float64)
        if not np.isfinite(x).all():
            raise RuntimeError(f"non-finite Train-only feature values in {name}")
        variants[name] = (fields, x)

    fold_results = []
    for fold_id, train_start_text, train_end_text, eval_start_text, eval_end_text in FOLDS:
        train_start, train_end = runner.ns(train_start_text), runner.ns(train_end_text)
        eval_start, eval_end = runner.ns(eval_start_text), runner.ns(eval_end_text)
        for feature_set, (fields, feature_matrix) in variants.items():
            x_train, y_train, train_times = aligned_rows(
                times, closes, feature_matrix, train_start, train_end
            )
            x_eval, y_eval, eval_times = aligned_rows(
                times, closes, feature_matrix, eval_start, eval_end
            )
            if not len(x_train) or not len(x_eval):
                raise RuntimeError(f"empty purged walk-forward fold: {fold_id}/{feature_set}")
            for alpha in ALPHAS:
                prediction, fit_meta = fit_predict(x_train, y_train, x_eval, alpha)
                fold_results.append({
                    "fold_id": fold_id,
                    "train_start_ns": train_start,
                    "train_end_ns_exclusive": train_end,
                    "eval_start_ns": eval_start,
                    "eval_end_ns_exclusive": eval_end,
                    "feature_set": feature_set,
                    "features": list(fields),
                    "ridge_alpha": alpha,
                    "train_samples_nonoverlap": int(len(x_train)),
                    "eval_samples_nonoverlap": int(len(x_eval)),
                    "train_label_end_max_ns": int(np.max(train_times) + HORIZON_MINUTES * runner.MINUTE_NS),
                    "eval_label_end_max_ns": int(np.max(eval_times) + HORIZON_MINUTES * runner.MINUTE_NS),
                    "train_label_mean_bps": float(np.mean(y_train)),
                    "evaluation": summarize(prediction, y_eval),
                    "fit_parameters": fit_meta,
                })

    grouped = {}
    for row in fold_results:
        key = (row["feature_set"], row["ridge_alpha"])
        grouped.setdefault(key, []).append(row)
    aggregates = []
    for (feature_set, alpha), folds in grouped.items():
        chosen = [f["evaluation"] for f in folds if f["evaluation"]["selected_count"] > 0]
        aggregates.append({
            "feature_set": feature_set,
            "ridge_alpha": alpha,
            "fold_count": len(folds),
            "folds_with_any_selected_trade": len(chosen),
            "selected_trade_count": sum(x["selected_count"] for x in chosen),
            "selected_after_30bp_hurdle_bps_trade_weighted": (
                float(np.average(
                    [x["selected_after_30bp_hurdle_bps"] for x in chosen],
                    weights=[x["selected_count"] for x in chosen],
                )) if chosen else None
            ),
            "positive_net_folds": sum(
                x["selected_after_30bp_hurdle_bps"] is not None
                and x["selected_after_30bp_hurdle_bps"] > 0 for x in chosen
            ),
        })

    report = {
        "schema": "rles.route-a.forecast-walkforward.v1",
        "purpose": "Train-inner purged chronological model diagnostics; no promotion evidence",
        "economic_status": "SCENARIO_ONLY",
        "source_marker_sha256": marker_sha,
        "wfo_report_sha256": sha256(trainfold.WFO_REPORT.read_bytes()).hexdigest(),
        "horizon_minutes": HORIZON_MINUTES,
        "sample_interval_minutes": SAMPLE_MINUTES,
        "fit_cost_hurdle_bps": ENTRY_HURDLE_BPS,
        "purge_policy": "training labels end no later than evaluation start; labels are sampled at the same non-overlapping interval as horizon",
        "folds": fold_results,
        "aggregates": aggregates,
    }
    output = OUT / "forecast-walkforward-report.json"
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    output.write_text(payload, encoding="utf-8")
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest, encoding="ascii")
    print(json.dumps({"output": str(OUT), "aggregates": aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
