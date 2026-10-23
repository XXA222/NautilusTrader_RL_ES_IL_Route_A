"""Chronological Train-only nonlinear forecast diagnostic for Route A.

The MLP is fitted to causal features and future close returns, with labels used
only as targets. Evaluation anchors do not overlap at the 240-minute horizon;
each fit label ends no later than the corresponding evaluation start. A fixed
30 bps hurdle represents the existing conservative Stress scenario assumption.
This is a teacher discovery screen, not native execution or promotion evidence.
"""

from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import torch
from torch import nn

import probe_route_a_forecast_walkforward_20261021 as ridge_probe
import run_route_a_trainfold_screen_20261021 as trainfold
import run_route_a_train_student_screen_20261020 as runner
from rles_orchestrator.research_access import ACTIVE_OWNER


OUT = runner.ROOT / ".artifacts" / "route-a-nonlinear-forecast-teacher-20261021-run01"
FIT_START = "2024-04-16T00:00:00+00:00"
ANALYSIS_END = "2024-05-15T00:00:00+00:00"
HORIZON_MINUTES = 240
TRAIN_SAMPLE_MINUTES = 5
EVALUATION_SAMPLE_MINUTES = 240
MAX_ALIGNMENT_LAG_MINUTES = 2
ENTRY_HURDLE_BPS = 30.0
RIDGE_ALPHA = 1.0
SEED = 31
HIDDEN_SIZES = (32, 16)
BATCH_SIZE = 256
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
FOLDS = ridge_probe.FOLDS
FEATURES = ridge_probe.FEATURE_SETS["v6_sixteen"]


def aligned_rows(
    times: np.ndarray,
    closes: np.ndarray,
    features: np.ndarray,
    start_ns: int,
    end_ns: int,
    sample_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizon_ns = HORIZON_MINUTES * runner.MINUTE_NS
    step_ns = sample_minutes * runner.MINUTE_NS
    lag_ns = MAX_ALIGNMENT_LAG_MINUTES * runner.MINUTE_NS
    grid = np.arange(start_ns, end_ns - horizon_ns, step_ns, dtype=np.int64)
    entry = np.searchsorted(times, grid, side="left")
    keep = entry < len(times)
    entry, grid = entry[keep], grid[keep]
    aligned = (times[entry] >= grid) & (times[entry] - grid <= lag_ns)
    entry = entry[aligned]
    exit_idx = np.searchsorted(times, times[entry] + horizon_ns, side="left")
    valid = exit_idx < len(times)
    lag = np.full(entry.size, np.iinfo(np.int64).max, dtype=np.int64)
    positions = np.flatnonzero(valid)
    lag[positions] = times[exit_idx[positions]] - (times[entry[positions]] + horizon_ns)
    valid &= lag <= lag_ns
    entry, exit_idx = entry[valid], exit_idx[valid]
    x = features[entry]
    y = np.log(closes[exit_idx] / closes[entry]) * 10_000.0
    return x, y, times[entry], entry


def fit_mlp(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, dict]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    target_mean = float(np.mean(y))
    target_std = float(np.std(y))
    if target_std <= 1e-12:
        target_std = 1.0
    x_train = torch.as_tensor((x - mean) / std, dtype=torch.float32)
    y_train = torch.as_tensor(((y - target_mean) / target_std)[:, None], dtype=torch.float32)
    x_valid = torch.as_tensor((x_eval - mean) / std, dtype=torch.float32)

    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(x.shape[1], HIDDEN_SIZES[0]),
        nn.GELU(),
        nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
        nn.GELU(),
        nn.Linear(HIDDEN_SIZES[1], 1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    generator = torch.Generator().manual_seed(SEED)
    epoch_losses = []
    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(x_train), generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
        epoch_losses.append(total_loss / seen)
    model.eval()
    with torch.no_grad():
        prediction = model(x_valid).squeeze(-1).cpu().numpy() * target_std + target_mean
    return prediction, {
        "train_samples": int(len(x)),
        "target_mean_bps": target_mean,
        "target_std_bps": target_std,
        "feature_means": mean.tolist(),
        "feature_stds": std.tolist(),
        "epoch_loss_first": epoch_losses[0],
        "epoch_loss_last": epoch_losses[-1],
        "epoch_losses": epoch_losses,
    }


def trade_summary(prediction: np.ndarray, realized: np.ndarray) -> dict:
    long = prediction >= ENTRY_HURDLE_BPS
    short = prediction <= -ENTRY_HURDLE_BPS
    selected = long | short
    gross = np.where(long, realized, -realized)
    net = gross[selected] - ENTRY_HURDLE_BPS
    return {
        "anchor_count": int(len(realized)),
        "selected_count": int(np.sum(selected)),
        "long_count": int(np.sum(long)),
        "short_count": int(np.sum(short)),
        "gross_mean_bps": float(np.mean(gross[selected])) if np.any(selected) else None,
        "net_mean_bps_after_hurdle": float(np.mean(net)) if net.size else None,
        "net_positive_count": int(np.sum(net > 0)),
        "net_worst_bps": float(np.min(net)) if net.size else None,
        "prediction_mean_bps": float(np.mean(prediction)) if len(prediction) else None,
        "prediction_std_bps": float(np.std(prediction)) if len(prediction) else None,
    }


def main() -> int:
    runner.configure_torch_threads(torch)
    torch.use_deterministic_algorithms(True)
    trainfold.configure()
    if OUT.exists():
        raise FileExistsError(f"immutable nonlinear teacher output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner = ACTIVE_OWNER.set("route-a-nonlinear-forecast-teacher-20261021")
    try:
        _, factory = runner.make_config(
            label="nonlinear-forecast-fit-apr16-may15",
            marker_sha=marker_sha,
            scope_start=scope_start,
            scope_end=scope_end,
            start_ns=runner.ns(FIT_START),
            end_ns=runner.ns(ANALYSIS_END),
            mode="base",
            fit_scaler=True,
        )
    finally:
        ACTIVE_OWNER.reset(owner)

    rows = factory.rows
    times = np.asarray([row.decision_ts_ns for row in rows], dtype=np.int64)
    field_names = tuple(field.name for field in factory.features.fields)
    feature_indices = [field_names.index(name) for name in FEATURES]
    close_index = field_names.index("close_1m")
    matrix = np.asarray(
        [[row.values[index] for index in feature_indices] for row in rows], dtype=np.float64
    )
    closes = np.asarray([row.values[close_index] for row in rows], dtype=np.float64)
    if (not np.isfinite(matrix).all() or not np.isfinite(closes).all()
            or np.any(np.diff(times) <= 0)):
        raise RuntimeError("causal Train feature rows are non-finite or unordered")

    folds = []
    for fold_id, train_start_text, train_end_text, eval_start_text, eval_end_text in FOLDS:
        train_start, train_end = runner.ns(train_start_text), runner.ns(train_end_text)
        eval_start, eval_end = runner.ns(eval_start_text), runner.ns(eval_end_text)
        x_train, y_train, train_times, _ = aligned_rows(
            times, closes, matrix, train_start, train_end, TRAIN_SAMPLE_MINUTES
        )
        x_eval, y_eval, eval_times, eval_indices = aligned_rows(
            times, closes, matrix, eval_start, eval_end, EVALUATION_SAMPLE_MINUTES
        )
        if len(x_train) < 500 or len(x_eval) < 10:
            raise RuntimeError(f"insufficient purged temporal rows in {fold_id}")
        if int(np.max(train_times) + HORIZON_MINUTES * runner.MINUTE_NS) > eval_start:
            raise RuntimeError(f"training labels overlap evaluation in {fold_id}")

        mlp_prediction, fit_meta = fit_mlp(x_train, y_train, x_eval)
        ridge_prediction, ridge_meta = ridge_probe.fit_predict(x_train, y_train, x_eval, RIDGE_ALPHA)
        past_idx = np.searchsorted(times, eval_times - HORIZON_MINUTES * runner.MINUTE_NS, side="right") - 1
        if np.any(past_idx < 0):
            raise RuntimeError(f"missing causal momentum rows in {fold_id}")
        momentum = np.log(closes[eval_indices] / closes[past_idx]) * 10_000.0
        folds.append({
            "fold_id": fold_id,
            "train_start_ns": train_start,
            "train_end_ns_exclusive": train_end,
            "evaluation_start_ns": eval_start,
            "evaluation_end_ns_exclusive": eval_end,
            "train_sample_interval_minutes": TRAIN_SAMPLE_MINUTES,
            "evaluation_sample_interval_minutes": EVALUATION_SAMPLE_MINUTES,
            "train_samples": int(len(x_train)),
            "evaluation_nonoverlap_samples": int(len(x_eval)),
            "max_train_label_end_ns": int(np.max(train_times) + HORIZON_MINUTES * runner.MINUTE_NS),
            "max_evaluation_label_end_ns": int(np.max(eval_times) + HORIZON_MINUTES * runner.MINUTE_NS),
            "mlp_fit": fit_meta,
            "mlp": trade_summary(mlp_prediction, y_eval),
            "ridge_alpha_1": trade_summary(ridge_prediction, y_eval),
            "same_horizon_momentum": trade_summary(momentum, y_eval),
            "same_horizon_reversal": trade_summary(-momentum, y_eval),
            "features": list(FEATURES),
            "ridge_fit": ridge_meta,
        })

    report = {
        "schema": "rles.route-a.nonlinear-forecast-teacher.v1",
        "purpose": "Chronological Train-inner MLP discovery screen; not native execution or promotion evidence",
        "economic_status": "SCENARIO_ONLY",
        "source_marker_sha256": marker_sha,
        "wfo_report_sha256": sha256(trainfold.WFO_REPORT.read_bytes()).hexdigest(),
        "horizon_minutes": HORIZON_MINUTES,
        "train_sample_interval_minutes": TRAIN_SAMPLE_MINUTES,
        "evaluation_sample_interval_minutes": EVALUATION_SAMPLE_MINUTES,
        "max_alignment_lag_minutes": MAX_ALIGNMENT_LAG_MINUTES,
        "entry_hurdle_bps": ENTRY_HURDLE_BPS,
        "cost_scope": "conservative Base/Stress scenario hurdle; not historical fee certification",
        "model": {
            "kind": "two-hidden-layer causal-feature MLP regressor",
            "hidden_sizes": list(HIDDEN_SIZES),
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "SmoothL1 on train-standardized forward log-return bps",
        },
        "purge_policy": "train labels end no later than evaluation start; evaluation labels are 240-minute non-overlapping returns",
        "folds": folds,
    }
    output = OUT / "nonlinear-forecast-teacher-report.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest, encoding="ascii")
    aggregate = {}
    for key in ("mlp", "ridge_alpha_1", "same_horizon_momentum", "same_horizon_reversal"):
        selected = [fold[key] for fold in folds if fold[key]["selected_count"]]
        aggregate[key] = {
            "folds_with_trades": len(selected),
            "trades": sum(row["selected_count"] for row in selected),
            "net_bps_trade_weighted": (
                float(np.average(
                    [row["net_mean_bps_after_hurdle"] for row in selected],
                    weights=[row["selected_count"] for row in selected],
                )) if selected else None
            ),
            "net_positive_folds": sum(
                row["net_mean_bps_after_hurdle"] > 0 for row in selected
            ),
        }
    print(json.dumps({"output": str(OUT), "report_sha256": digest, "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
