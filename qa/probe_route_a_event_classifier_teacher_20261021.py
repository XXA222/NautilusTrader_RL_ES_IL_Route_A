"""Train-only high-quality trade-event classifier for Route A.

Labels use future 240-minute close returns only: >= +60 bps is LONG, <= -60
bps is SHORT, and the remainder is FLAT. Causal feature rows train a small
classifier in purged expanding time folds. Evaluation uses non-overlapping
240-minute anchors and a fixed 30 bps scenario cost hurdle. No execution or
promotion claim is made by this diagnostic.
"""

from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import torch
from torch import nn

import probe_route_a_forecast_walkforward_20261021 as fold_protocol
import run_route_a_trainfold_screen_20261021 as trainfold
import run_route_a_train_student_screen_20261020 as runner
from rles_orchestrator.research_access import ACTIVE_OWNER


OUT = runner.ROOT / ".artifacts" / "route-a-event-classifier-teacher-20261021-run01"
FIT_START = "2024-04-16T00:00:00+00:00"
ANALYSIS_END = "2024-05-15T00:00:00+00:00"
HORIZON_MINUTES = 240
TRAIN_SAMPLE_MINUTES = 5
EVALUATION_SAMPLE_MINUTES = 240
MAX_ALIGNMENT_LAG_MINUTES = 2
EVENT_TARGET_BPS = 60.0
ENTRY_COST_HURDLE_BPS = 30.0
OPEN_PROBABILITY_THRESHOLD = 0.60
SEED = 31
HIDDEN_SIZES = (32, 16)
BATCH_SIZE = 256
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
FOLDS = fold_protocol.FOLDS
FEATURES = fold_protocol.FEATURE_SETS["v6_sixteen"]
CLASS_NAMES = ("FLAT", "OPEN_LONG", "OPEN_SHORT")


def aligned_rows(
    times: np.ndarray,
    closes: np.ndarray,
    features: np.ndarray,
    start_ns: int,
    end_ns: int,
    sample_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon_ns = HORIZON_MINUTES * runner.MINUTE_NS
    step_ns = sample_minutes * runner.MINUTE_NS
    lag_ns = MAX_ALIGNMENT_LAG_MINUTES * runner.MINUTE_NS
    grid = np.arange(start_ns, end_ns - horizon_ns, step_ns, dtype=np.int64)
    entry = np.searchsorted(times, grid, side="left")
    keep = entry < len(times)
    entry, grid = entry[keep], grid[keep]
    valid = (times[entry] >= grid) & (times[entry] - grid <= lag_ns)
    entry = entry[valid]
    exits = np.searchsorted(times, times[entry] + horizon_ns, side="left")
    valid = exits < len(times)
    lag = np.full(entry.size, np.iinfo(np.int64).max, dtype=np.int64)
    positions = np.flatnonzero(valid)
    lag[positions] = times[exits[positions]] - (times[entry[positions]] + horizon_ns)
    valid &= lag <= lag_ns
    entry, exits = entry[valid], exits[valid]
    x = features[entry]
    y_bps = np.log(closes[exits] / closes[entry]) * 10_000.0
    return x, y_bps, times[entry]


def event_labels(y_bps: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(y_bps), dtype=np.int64)
    labels[y_bps >= EVENT_TARGET_BPS] = 1
    labels[y_bps <= -EVENT_TARGET_BPS] = 2
    return labels


def fit_classifier(
    x: np.ndarray,
    labels: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, dict]:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if np.any(counts == 0):
        raise RuntimeError(f"training fold lacks an event class: {counts.tolist()}")
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    x_train = torch.as_tensor((x - mean) / std, dtype=torch.float32)
    y_train = torch.as_tensor(labels, dtype=torch.long)
    x_valid = torch.as_tensor((x_eval - mean) / std, dtype=torch.float32)
    class_weights = np.sqrt(len(labels) / counts.astype(np.float64))
    class_weights /= np.mean(class_weights)
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)

    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(x.shape[1], HIDDEN_SIZES[0]),
        nn.GELU(),
        nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
        nn.GELU(),
        nn.Linear(HIDDEN_SIZES[1], len(CLASS_NAMES)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
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
        probabilities = torch.softmax(model(x_valid), dim=-1).cpu().numpy()
    return probabilities, {
        "train_samples": int(len(x)),
        "train_class_counts": {
            name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "class_weights": {
            name: float(class_weights[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "feature_means": mean.tolist(),
        "feature_stds": std.tolist(),
        "epoch_loss_first": epoch_losses[0],
        "epoch_loss_last": epoch_losses[-1],
        "epoch_losses": epoch_losses,
    }


def summarize(probabilities: np.ndarray, y_bps: np.ndarray) -> dict:
    proposed = np.argmax(probabilities, axis=1)
    strength = np.max(probabilities[:, 1:], axis=1)
    chosen = (proposed != 0) & (strength >= OPEN_PROBABILITY_THRESHOLD)
    side = np.where(proposed == 1, 1.0, -1.0)
    net_bps = y_bps[chosen] * side[chosen] - ENTRY_COST_HURDLE_BPS
    return {
        "sample_count": int(len(y_bps)),
        "actual_event_class_counts": {
            name: int(count) for name, count in zip(
                CLASS_NAMES, np.bincount(event_labels(y_bps), minlength=len(CLASS_NAMES)), strict=True
            )
        },
        "predicted_class_counts": {
            name: int(np.sum(proposed == i)) for i, name in enumerate(CLASS_NAMES)
        },
        "classification_accuracy": float(np.mean(proposed == event_labels(y_bps))),
        "selected_count": int(np.sum(chosen)),
        "selected_long_count": int(np.sum(chosen & (proposed == 1))),
        "selected_short_count": int(np.sum(chosen & (proposed == 2))),
        "selected_gross_mean_bps": (
            float(np.mean(y_bps[chosen] * side[chosen])) if np.any(chosen) else None
        ),
        "selected_net_mean_bps_after_hurdle": (
            float(np.mean(net_bps)) if net_bps.size else None
        ),
        "selected_net_positive_count": int(np.sum(net_bps > 0)),
        "selected_net_worst_bps": float(np.min(net_bps)) if net_bps.size else None,
        "selected_threshold": OPEN_PROBABILITY_THRESHOLD,
    }


def main() -> int:
    runner.configure_torch_threads(torch)
    torch.use_deterministic_algorithms(True)
    trainfold.configure()
    if OUT.exists():
        raise FileExistsError(f"immutable event classifier output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner = ACTIVE_OWNER.set("route-a-event-classifier-teacher-20261021")
    try:
        _, factory = runner.make_config(
            label="event-classifier-fit-apr16-may15",
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
    features = np.asarray(
        [[row.values[index] for index in feature_indices] for row in rows], dtype=np.float64
    )
    closes = np.asarray([row.values[close_index] for row in rows], dtype=np.float64)
    if (not np.isfinite(features).all() or not np.isfinite(closes).all()
            or np.any(np.diff(times) <= 0)):
        raise RuntimeError("causal Train feature rows are non-finite or unordered")

    fold_rows = []
    for fold_id, train_start_text, train_end_text, eval_start_text, eval_end_text in FOLDS:
        train_start, train_end = runner.ns(train_start_text), runner.ns(train_end_text)
        eval_start, eval_end = runner.ns(eval_start_text), runner.ns(eval_end_text)
        x_train, y_train_bps, train_times = aligned_rows(
            times, closes, features, train_start, train_end, TRAIN_SAMPLE_MINUTES
        )
        x_eval, y_eval_bps, eval_times = aligned_rows(
            times, closes, features, eval_start, eval_end, EVALUATION_SAMPLE_MINUTES
        )
        if len(x_train) < 500 or len(x_eval) < 10:
            raise RuntimeError(f"insufficient purged event rows in {fold_id}")
        if int(np.max(train_times) + HORIZON_MINUTES * runner.MINUTE_NS) > eval_start:
            raise RuntimeError(f"training event labels overlap evaluation in {fold_id}")
        train_labels, eval_labels = event_labels(y_train_bps), event_labels(y_eval_bps)
        probabilities, fit_meta = fit_classifier(x_train, train_labels, x_eval)
        fold_rows.append({
            "fold_id": fold_id,
            "train_start_ns": train_start,
            "train_end_ns_exclusive": train_end,
            "evaluation_start_ns": eval_start,
            "evaluation_end_ns_exclusive": eval_end,
            "max_train_label_end_ns": int(
                np.max(train_times) + HORIZON_MINUTES * runner.MINUTE_NS
            ),
            "max_evaluation_label_end_ns": int(
                np.max(eval_times) + HORIZON_MINUTES * runner.MINUTE_NS
            ),
            "features": list(FEATURES),
            "fit": fit_meta,
            "evaluation": summarize(probabilities, y_eval_bps),
            "true_eval_event_counts": {
                name: int(count) for name, count in zip(
                    CLASS_NAMES,
                    np.bincount(eval_labels, minlength=len(CLASS_NAMES)),
                    strict=True,
                )
            },
        })

    report = {
        "schema": "rles.route-a.event-classifier-teacher.v1",
        "purpose": "Train-inner high-quality outcome-event discovery; not native execution or promotion evidence",
        "economic_status": "SCENARIO_ONLY",
        "source_marker_sha256": marker_sha,
        "wfo_report_sha256": sha256(trainfold.WFO_REPORT.read_bytes()).hexdigest(),
        "horizon_minutes": HORIZON_MINUTES,
        "training_sample_interval_minutes": TRAIN_SAMPLE_MINUTES,
        "evaluation_sample_interval_minutes": EVALUATION_SAMPLE_MINUTES,
        "event_label": {
            "long_minimum_future_close_log_return_bps": EVENT_TARGET_BPS,
            "short_maximum_future_close_log_return_bps": -EVENT_TARGET_BPS,
            "otherwise": "FLAT",
        },
        "fixed_net_cost_hurdle_bps": ENTRY_COST_HURDLE_BPS,
        "model": {
            "kind": "weighted two-hidden-layer causal-feature event classifier",
            "hidden_sizes": list(HIDDEN_SIZES),
            "class_weight": "inverse square-root train-class frequency, normalized to mean 1",
            "probability_open_threshold": OPEN_PROBABILITY_THRESHOLD,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "purge_policy": "training future-label end no later than evaluation start; evaluation anchors are non-overlapping at the 240-minute horizon",
        "folds": fold_rows,
    }
    output = OUT / "event-classifier-teacher-report.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest, encoding="ascii")
    selected = [row["evaluation"] for row in fold_rows if row["evaluation"]["selected_count"]]
    total_selected = sum(row["selected_count"] for row in selected)
    aggregate = {
        "folds_with_trades": len(selected),
        "trades": total_selected,
        "net_bps_trade_weighted": (
            float(np.average(
                [row["selected_net_mean_bps_after_hurdle"] for row in selected],
                weights=[row["selected_count"] for row in selected],
            )) if selected else None
        ),
        "positive_net_folds": sum(
            row["selected_net_mean_bps_after_hurdle"] > 0 for row in selected
        ),
        "open_threshold": OPEN_PROBABILITY_THRESHOLD,
        "event_target_bps": EVENT_TARGET_BPS,
    }
    print(json.dumps({"output": str(OUT), "report_sha256": digest, "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
