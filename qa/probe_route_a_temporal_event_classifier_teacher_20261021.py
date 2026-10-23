"""Temporal event-label teacher discovery on purged Route A Train folds.

The model sees a causal 60-minute history of 5-minute feature snapshots and
classifies 240-minute close-return events. Future labels never enter inputs.
Evaluation anchors are 240 minutes apart; all economic metrics use a fixed
30 bps scenario cost hurdle and remain Train-inner diagnostics.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os

import numpy as np
import torch
from torch import nn

import probe_route_a_forecast_walkforward_20261021 as fold_protocol
import run_route_a_trainfold_screen_20261021 as trainfold
import run_route_a_train_student_screen_20261020 as runner
from rles_orchestrator.research_access import ACTIVE_OWNER


RUN_ID = os.environ.get("ROUTE_A_TEMPORAL_EVENT_RUN", "run01")
OUT = runner.ROOT / ".artifacts" / f"route-a-temporal-event-classifier-teacher-20261021-{RUN_ID}"
FIT_START = "2024-04-16T00:00:00+00:00"
ANALYSIS_END = "2024-05-15T00:00:00+00:00"
HORIZON_MINUTES = 240
TRAIN_SAMPLE_MINUTES = 5
EVALUATION_SAMPLE_MINUTES = 240
LOOKBACK_MINUTES = int(os.environ.get("ROUTE_A_TEMPORAL_LOOKBACK_MINUTES", "60"))
HISTORY_SAMPLE_MINUTES = 5
MAX_ALIGNMENT_LAG_MINUTES = 2
EVENT_TARGET_BPS = 60.0
ENTRY_COST_HURDLE_BPS = 30.0
OPEN_PROBABILITY_THRESHOLD = 0.60
SEED = 31
HIDDEN_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
FOLDS = fold_protocol.FOLDS
FEATURES = fold_protocol.FEATURE_SETS["v6_sixteen"]
CLASS_NAMES = ("FLAT", "OPEN_LONG", "OPEN_SHORT")
HISTORY_LENGTH = LOOKBACK_MINUTES // HISTORY_SAMPLE_MINUTES
HISTORY_OFFSETS = np.arange(
    -(HISTORY_LENGTH - 1) * HISTORY_SAMPLE_MINUTES, 1, HISTORY_SAMPLE_MINUTES,
    dtype=np.int64,
)


def anchor_rows(
    times: np.ndarray,
    closes: np.ndarray,
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
    aligned = (times[entry] >= grid) & (times[entry] - grid <= lag_ns)
    entry = entry[aligned]
    exit_idx = np.searchsorted(times, times[entry] + horizon_ns, side="left")
    valid = exit_idx < len(times)
    lag = np.full(entry.size, np.iinfo(np.int64).max, dtype=np.int64)
    positions = np.flatnonzero(valid)
    lag[positions] = times[exit_idx[positions]] - (times[entry[positions]] + horizon_ns)
    valid &= lag <= lag_ns
    entry, exit_idx = entry[valid], exit_idx[valid]
    y = np.log(closes[exit_idx] / closes[entry]) * 10_000.0
    return entry, y, times[entry]


def make_sequences(
    entries: np.ndarray,
    times: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ids = entries[:, None] + HISTORY_OFFSETS[None, :]
    valid = ids[:, 0] >= 0
    ids = ids[valid]
    entries = entries[valid]
    if not len(entries):
        return np.empty((0, HISTORY_LENGTH, features.shape[1])), entries
    expected = HISTORY_SAMPLE_MINUTES * runner.MINUTE_NS
    contiguous = np.all(np.diff(times[ids], axis=1) == expected, axis=1)
    ids = ids[contiguous]
    entries = entries[contiguous]
    return features[ids], entries


def event_labels(y_bps: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(y_bps), dtype=np.int64)
    labels[y_bps >= EVENT_TARGET_BPS] = 1
    labels[y_bps <= -EVENT_TARGET_BPS] = 2
    return labels


def fit_classifier(
    sequences: np.ndarray,
    labels: np.ndarray,
    sequences_eval: np.ndarray,
) -> tuple[np.ndarray, dict]:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if np.any(counts == 0):
        raise RuntimeError(f"training fold lacks an event class: {counts.tolist()}")
    mean = np.mean(sequences, axis=(0, 1))
    std = np.std(sequences, axis=(0, 1))
    std = np.where(std > 1e-12, std, 1.0)
    sequences = (sequences - mean[None, None, :]) / std[None, None, :]
    sequences_eval = (sequences_eval - mean[None, None, :]) / std[None, None, :]
    x_train = torch.as_tensor(sequences, dtype=torch.float32)
    y_train = torch.as_tensor(labels, dtype=torch.long)
    x_valid = torch.as_tensor(sequences_eval, dtype=torch.float32)
    class_weights = np.sqrt(len(labels) / counts.astype(np.float64))
    class_weights /= np.mean(class_weights)
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)

    torch.manual_seed(SEED)
    model = nn.GRU(
        input_size=sequences.shape[2], hidden_size=HIDDEN_SIZE, batch_first=True
    )
    head = nn.Linear(HIDDEN_SIZE, len(CLASS_NAMES))
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *head.parameters()],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    generator = torch.Generator().manual_seed(SEED)
    epoch_losses = []
    model.train()
    head.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(x_train), generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            _, hidden = model(x_train[batch])
            logits = head(hidden[-1])
            loss = loss_fn(logits, y_train[batch])
            loss.backward()
            nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
        epoch_losses.append(total_loss / seen)
    model.eval()
    head.eval()
    with torch.no_grad():
        _, hidden = model(x_valid)
        probabilities = torch.softmax(head(hidden[-1]), dim=-1).cpu().numpy()
    return probabilities, {
        "train_samples": int(len(x_train)),
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
    actual = event_labels(y_bps)
    return {
        "sample_count": int(len(y_bps)),
        "actual_event_class_counts": {
            name: int(count) for name, count in zip(
                CLASS_NAMES, np.bincount(actual, minlength=len(CLASS_NAMES)), strict=True
            )
        },
        "predicted_class_counts": {
            name: int(np.sum(proposed == i)) for i, name in enumerate(CLASS_NAMES)
        },
        "classification_accuracy": float(np.mean(proposed == actual)),
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
    if (type(LOOKBACK_MINUTES) is not int or LOOKBACK_MINUTES < HISTORY_SAMPLE_MINUTES
            or LOOKBACK_MINUTES % HISTORY_SAMPLE_MINUTES):
        raise ValueError("lookback must be a positive multiple of the history sample interval")
    trainfold.configure()
    if OUT.exists():
        raise FileExistsError(f"immutable temporal event output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner = ACTIVE_OWNER.set("route-a-temporal-event-classifier-teacher-20261021")
    try:
        _, factory = runner.make_config(
            label="temporal-event-fit-apr16-may15",
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

    folds = []
    for fold_id, train_start_text, train_end_text, eval_start_text, eval_end_text in FOLDS:
        train_start, train_end = runner.ns(train_start_text), runner.ns(train_end_text)
        eval_start, eval_end = runner.ns(eval_start_text), runner.ns(eval_end_text)
        train_entries, y_train, train_times = anchor_rows(
            times, closes, train_start, train_end, TRAIN_SAMPLE_MINUTES
        )
        eval_entries, y_eval, eval_times = anchor_rows(
            times, closes, eval_start, eval_end, EVALUATION_SAMPLE_MINUTES
        )
        x_train, train_entries = make_sequences(train_entries, times, features)
        x_eval, eval_entries = make_sequences(eval_entries, times, features)
        if len(x_train) < 500 or len(x_eval) < 10:
            raise RuntimeError(f"insufficient temporal event samples in {fold_id}")
        if int(np.max(train_times) + HORIZON_MINUTES * runner.MINUTE_NS) > eval_start:
            raise RuntimeError(f"training event labels overlap evaluation in {fold_id}")
        # Sequence filtering only removes history-incomplete/gapped anchors; map
        # the retained causal decision times back to their future-return targets.
        train_time_set = set(int(times[index]) for index in train_entries)
        eval_time_set = set(int(times[index]) for index in eval_entries)
        train_mask = np.asarray([int(value) in train_time_set for value in train_times])
        eval_mask = np.asarray([int(value) in eval_time_set for value in eval_times])
        y_train, y_eval = y_train[train_mask], y_eval[eval_mask]
        train_labels = event_labels(y_train)
        probabilities, fit_meta = fit_classifier(x_train, train_labels, x_eval)
        folds.append({
            "fold_id": fold_id,
            "train_start_ns": train_start,
            "train_end_ns_exclusive": train_end,
            "evaluation_start_ns": eval_start,
            "evaluation_end_ns_exclusive": eval_end,
            "training_label_end_max_ns": int(
                np.max(train_times[train_mask]) + HORIZON_MINUTES * runner.MINUTE_NS
            ),
            "evaluation_label_end_max_ns": int(
                np.max(eval_times[eval_mask]) + HORIZON_MINUTES * runner.MINUTE_NS
            ),
            "features": list(FEATURES),
            "lookback_minutes": LOOKBACK_MINUTES,
            "history_sample_interval_minutes": HISTORY_SAMPLE_MINUTES,
            "fit": fit_meta,
            "evaluation": summarize(probabilities, y_eval),
        })

    report = {
        "schema": "rles.route-a.temporal-event-classifier-teacher.v1",
        "purpose": "Train-inner temporal event-label discovery; not native execution or promotion evidence",
        "economic_status": "SCENARIO_ONLY",
        "source_marker_sha256": marker_sha,
        "wfo_report_sha256": sha256(trainfold.WFO_REPORT.read_bytes()).hexdigest(),
        "horizon_minutes": HORIZON_MINUTES,
        "training_sample_interval_minutes": TRAIN_SAMPLE_MINUTES,
        "evaluation_sample_interval_minutes": EVALUATION_SAMPLE_MINUTES,
        "history": {
            "lookback_minutes": LOOKBACK_MINUTES,
            "sample_interval_minutes": HISTORY_SAMPLE_MINUTES,
            "sequence_length": HISTORY_LENGTH,
        },
        "event_label": {
            "long_minimum_future_close_log_return_bps": EVENT_TARGET_BPS,
            "short_maximum_future_close_log_return_bps": -EVENT_TARGET_BPS,
            "otherwise": "FLAT",
        },
        "fixed_net_cost_hurdle_bps": ENTRY_COST_HURDLE_BPS,
        "model": {
            "kind": "causal GRU event classifier",
            "hidden_size": HIDDEN_SIZE,
            "class_weight": "inverse square-root train-class frequency, normalized to mean 1",
            "probability_open_threshold": OPEN_PROBABILITY_THRESHOLD,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "purge_policy": "training future-label end no later than evaluation start; evaluation anchors are non-overlapping at the 240-minute horizon",
        "folds": folds,
    }
    output = OUT / "temporal-event-classifier-teacher-report.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest, encoding="ascii")
    selected = [fold["evaluation"] for fold in folds if fold["evaluation"]["selected_count"]]
    aggregate = {
        "folds_with_trades": len(selected),
        "trades": sum(fold["selected_count"] for fold in selected),
        "net_bps_trade_weighted": (
            float(np.average(
                [fold["selected_net_mean_bps_after_hurdle"] for fold in selected],
                weights=[fold["selected_count"] for fold in selected],
            )) if selected else None
        ),
        "positive_net_folds": sum(
            fold["selected_net_mean_bps_after_hurdle"] > 0 for fold in selected
        ),
        "probability_threshold": OPEN_PROBABILITY_THRESHOLD,
    }
    print(json.dumps({"output": str(OUT), "report_sha256": digest, "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
