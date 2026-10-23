"""Train-only GRU value teacher for net OHLC barrier-path outcomes."""

from __future__ import annotations

from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
from time import perf_counter, process_time
import copy
from typing import Any

import numpy as np

import probe_route_a_barrier_label_teacher_20261021 as barrier
import probe_route_a_temporal_event_classifier_teacher_20261021 as temporal
import run_route_a_trainfold_screen_20261021 as trainfold
import run_route_a_train_student_screen_20261020 as runner
from rles_gym_bridge.historical import historical_feature_spec
from route_a_training_runtime import configure_torch_threads
from rles_orchestrator.research_access import ACTIVE_OWNER


RUN_ID = os.environ.get("ROUTE_A_BARRIER_VALUE_RUN", "run01")
DEFAULT_OUT = (
    runner.ROOT / ".artifacts"
    / f"route-a-barrier-value-teacher-20261021-{RUN_ID}"
)
OUT = Path(os.environ.get("ROUTE_A_BARRIER_VALUE_OUTPUT_DIR", str(DEFAULT_OUT))).resolve()
VALUE_THRESHOLDS_BPS = (0.0, 5.0, 10.0, 15.0, 20.0)
EVALUATION_COSTS_BPS = (12.0, 20.0, 30.0)
FLOW_FEATURE_SOURCE = os.environ.get("ROUTE_A_VALUE_FLOW_SOURCE", "kline")
TARGET_MODE = os.environ.get("ROUTE_A_VALUE_TARGET_MODE", "barrier_path")
AGGTRADE_FLOW_FEATURE_NAMES = (
    "agg_taker_buy_imbalance_5m",
    "agg_taker_buy_imbalance_15m",
    "log_agg_trade_count_5m",
)
EVALUATION_SAMPLE_MINUTES = int(os.environ.get("ROUTE_A_VALUE_EVAL_STEP_MINUTES", "240"))
HIDDEN_SIZE = int(os.environ.get("ROUTE_A_VALUE_HIDDEN_SIZE", "32"))
BATCH_SIZE = 512
EPOCHS = int(os.environ.get("ROUTE_A_VALUE_MAX_EPOCHS", "200"))
MIN_EPOCHS = int(os.environ.get("ROUTE_A_VALUE_MIN_EPOCHS", "20"))
TARGET_TRAIN_LOSS = float(os.environ.get("ROUTE_A_VALUE_TARGET_TRAIN_LOSS", "0.005"))
PARALLEL_WORKERS = int(os.environ.get("ROUTE_A_VALUE_PARALLEL_WORKERS", "4"))
CALIBRATION_FRACTION = 0.15
EARLY_STOP_PATIENCE = int(os.environ.get("ROUTE_A_VALUE_EARLY_STOP_PATIENCE", "10"))
SEED = 31

if EPOCHS < MIN_EPOCHS:
    raise ValueError("ROUTE_A_VALUE_MAX_EPOCHS must be at least MIN_EPOCHS")
if MIN_EPOCHS < 1:
    raise ValueError("ROUTE_A_VALUE_MIN_EPOCHS must be positive")
if EARLY_STOP_PATIENCE < 1:
    raise ValueError("ROUTE_A_VALUE_EARLY_STOP_PATIENCE must be positive")
if HIDDEN_SIZE < 1:
    raise ValueError("ROUTE_A_VALUE_HIDDEN_SIZE must be positive")
if FLOW_FEATURE_SOURCE not in {"kline", "aggtrades_1s"}:
    raise ValueError("ROUTE_A_VALUE_FLOW_SOURCE must be kline or aggtrades_1s")
if TARGET_MODE not in {"barrier_path", "fixed_horizon"}:
    raise ValueError("ROUTE_A_VALUE_TARGET_MODE must be barrier_path or fixed_horizon")
if not np.isfinite(TARGET_TRAIN_LOSS) or TARGET_TRAIN_LOSS <= 0:
    raise ValueError("ROUTE_A_VALUE_TARGET_TRAIN_LOSS must be finite and positive")
if PARALLEL_WORKERS < 1:
    raise ValueError("ROUTE_A_VALUE_PARALLEL_WORKERS must be positive")


def fit_value_model(x_train: np.ndarray, y_train_bps: np.ndarray,
                    x_eval: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    import torch
    from torch import nn

    runtime = configure_torch_threads(torch)
    cpu_count = runtime["logical_cpu_count"]
    requested_device = os.environ.get("ROUTE_A_TRAIN_DEVICE", "auto").lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("ROUTE_A_TRAIN_DEVICE must be auto, cpu, or cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch runtime has no CUDA")
    device = torch.device(
        "cuda" if requested_device == "cuda"
        or (requested_device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    calibration_start = int(len(x_train) * (1.0 - CALIBRATION_FRACTION))
    purge_samples = int(np.ceil(
        temporal.HORIZON_MINUTES / temporal.TRAIN_SAMPLE_MINUTES
    ))
    fit_end = calibration_start - purge_samples
    if fit_end < 500 or len(x_train) - calibration_start < 100:
        raise RuntimeError("not enough purged temporal data for fit/calibration")
    x_fit = x_train[:fit_end]
    y_fit_bps = y_train_bps[:fit_end]
    x_calibration = x_train[calibration_start:]
    y_calibration = y_train_bps[calibration_start:] / 100.0
    mean = np.mean(x_fit, axis=(0, 1))
    std = np.std(x_fit, axis=(0, 1))
    std = np.where(std > 1e-12, std, 1.0)
    x_fit = (x_fit - mean[None, None, :]) / std[None, None, :]
    x_calibration = (
        x_calibration - mean[None, None, :]
    ) / std[None, None, :]
    x_eval = (x_eval - mean[None, None, :]) / std[None, None, :]
    targets = y_fit_bps / 100.0
    train_x = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    calibration_x = torch.as_tensor(
        x_calibration, dtype=torch.float32, device=device
    )
    calibration_y = torch.as_tensor(
        y_calibration, dtype=torch.float32, device=device
    )
    eval_x = torch.as_tensor(x_eval, dtype=torch.float32, device=device)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    encoder = nn.GRU(x_train.shape[2], HIDDEN_SIZE, batch_first=True).to(device)
    head = nn.Linear(HIDDEN_SIZE, 2).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *head.parameters()], lr=1e-3, weight_decay=1e-3
    )
    loss_fn = nn.SmoothL1Loss()
    generator = torch.Generator(device=device).manual_seed(SEED)
    epoch_losses = []
    calibration_losses = []
    best_calibration_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_checkpoint_train_loss = float("inf")
    epochs_without_calibration_gain = 0
    early_stop_reason = "max_epochs_reached"
    encoder.train()
    head.train()
    started = perf_counter()
    cpu_started = process_time()
    target_reached = False
    for epoch in range(EPOCHS):
        order = torch.randperm(len(train_x), generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            _, hidden = encoder(train_x[batch])
            prediction = head(hidden[-1])
            loss = loss_fn(prediction, train_y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_([*encoder.parameters(), *head.parameters()], max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
        epoch_loss = total_loss / seen
        epoch_losses.append(epoch_loss)
        target_reached = target_reached or epoch_loss <= TARGET_TRAIN_LOSS
        encoder.eval()
        head.eval()
        with torch.no_grad():
            _, calibration_hidden = encoder(calibration_x)
            calibration_prediction = head(calibration_hidden[-1])
            calibration_loss = float(loss_fn(
                calibration_prediction, calibration_y
            ).item())
        calibration_losses.append(calibration_loss)
        if calibration_loss < best_calibration_loss - 1e-7:
            best_calibration_loss = calibration_loss
            best_epoch = epoch + 1
            best_checkpoint_train_loss = epoch_loss
            best_state = (
                copy.deepcopy(encoder.state_dict()),
                copy.deepcopy(head.state_dict()),
            )
            epochs_without_calibration_gain = 0
        else:
            epochs_without_calibration_gain += 1
        encoder.train()
        head.train()
        # Do not spend the remaining epoch budget chasing a training-loss
        # target after purged calibration has plateaued; that only deepens
        # overfit. The target remains explicit in metadata and may be unmet.
        if (epoch + 1 >= MIN_EPOCHS
                and epochs_without_calibration_gain >= EARLY_STOP_PATIENCE):
            early_stop_reason = (
                "calibration_plateau_after_target"
                if target_reached else "calibration_plateau_before_target"
            )
            break
    if best_state is None:
        raise RuntimeError("GRU training did not produce a calibration checkpoint")
    encoder.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    encoder.eval()
    head.eval()
    with torch.no_grad():
        _, hidden = encoder(eval_x)
        predictions_bps = (head(hidden[-1]).cpu().numpy() * 100.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = perf_counter() - started
    elapsed_cpu_seconds = process_time() - cpu_started
    model_bundle = {
        "encoder": encoder,
        "head": head,
        "mean": mean,
        "std": std,
        "device": device,
    }
    metadata = {
        "train_samples": int(len(x_train)),
        "fit_samples": int(len(train_x)),
        "calibration_samples": int(len(calibration_x)),
        "calibration_purge_samples": purge_samples,
        "feature_means": mean.tolist(),
        "feature_stds": std.tolist(),
        "target_long_mean_bps": float(np.mean(y_train_bps[:, 0])),
        "target_short_mean_bps": float(np.mean(y_train_bps[:, 1])),
        "epoch_loss_first": epoch_losses[0],
        "epoch_loss_last": epoch_losses[-1],
        "epoch_loss_min": min(epoch_losses),
        "best_checkpoint_train_loss": best_checkpoint_train_loss,
        "epoch_losses": epoch_losses,
        "calibration_loss_best": best_calibration_loss,
        "calibration_loss_last": calibration_losses[-1],
        "best_epoch": best_epoch,
        "target_train_loss": TARGET_TRAIN_LOSS,
        "target_train_loss_reached": target_reached,
        "early_stop_reason": early_stop_reason,
        "best_checkpoint_target_reached": (
            best_checkpoint_train_loss <= TARGET_TRAIN_LOSS
        ),
        "epochs_completed": len(epoch_losses),
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "device": str(device),
        **runtime,
        "fit_seconds": elapsed_seconds,
        "fit_process_cpu_seconds": elapsed_cpu_seconds,
        "fit_average_logical_cpu_utilization_percent": (
            100.0 * elapsed_cpu_seconds / elapsed_seconds / cpu_count
            if elapsed_seconds > 0 else None
        ),
    }
    return predictions_bps, metadata, model_bundle


def predict_value_model(model_bundle: dict, x_eval: np.ndarray) -> np.ndarray:
    import torch

    mean, std = model_bundle["mean"], model_bundle["std"]
    x_eval = (x_eval - mean[None, None, :]) / std[None, None, :]
    eval_x = torch.as_tensor(
        x_eval, dtype=torch.float32, device=model_bundle["device"]
    )
    model_bundle["encoder"].eval()
    model_bundle["head"].eval()
    with torch.no_grad():
        _, hidden = model_bundle["encoder"](eval_x)
        return model_bundle["head"](hidden[-1]).cpu().numpy() * 100.0


def _fit_value_fold_worker(
    x_train: np.ndarray, y_train_bps: np.ndarray, x_eval: np.ndarray,
    worker_threads: int,
) -> tuple[np.ndarray, dict]:
    """Fit an isolated fold in a spawned process with a bounded CPU budget."""
    os.environ["ROUTE_A_TORCH_NUM_THREADS"] = str(worker_threads)
    os.environ["ROUTE_A_TORCH_INTEROP_THREADS"] = "1"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(worker_threads)
    import torch

    torch.use_deterministic_algorithms(True)
    predictions, metadata, _ = fit_value_model(x_train, y_train_bps, x_eval)
    return predictions, metadata


def _verified_aggtrade_archive_hashes(
    archive_manifest: Path, rows: tuple[Any, ...],
) -> set[str]:
    try:
        manifest = json.loads(archive_manifest.read_text(encoding="utf-8"))
        start_day = date.fromisoformat(manifest["agg_trades_start_day"])
        end_day = date.fromisoformat(manifest["agg_trades_end_day_inclusive"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("aggTrades source manifest is missing or invalid") from exc
    if (manifest.get("schema") != "rles.route-a-train-snapshot.v1"
            or manifest.get("agg_trades_included") is not True
            or end_day < start_day or not rows):
        raise RuntimeError("Train manifest does not certify a bounded aggTrades source")
    required_start = datetime.fromtimestamp(
        (min(int(row.decision_ts_ns) for row in rows) - 15 * runner.MINUTE_NS) / 1e9,
        tz=timezone.utc,
    ).date()
    required_end = datetime.fromtimestamp(
        max(int(row.decision_ts_ns) for row in rows) / 1e9, tz=timezone.utc,
    ).date()
    if start_day > required_start or end_day < required_end:
        raise RuntimeError("verified aggTrades archives do not cover the causal feature window")
    expected_days: set[date] = set()
    current = start_day
    while current <= end_day:
        expected_days.add(current)
        current += timedelta(days=1)
    actual_days: list[date] = []
    source_hashes: set[str] = set()
    for source in manifest.get("sources", []):
        if not isinstance(source, dict) or "/aggTrades/" not in str(source.get("url", "")):
            continue
        match = re.search(
            r"ETHUSDT-aggTrades-(\d{4}-\d{2}-\d{2})\.zip$", source["url"]
        )
        source_hash = source.get("sha256")
        if (match is None or not isinstance(source_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None):
            raise RuntimeError("aggTrades archive identity or SHA-256 is malformed")
        actual_days.append(date.fromisoformat(match.group(1)))
        source_hashes.add(source_hash)
    if (len(actual_days) != len(set(actual_days))
            or set(actual_days) != expected_days or not source_hashes):
        raise RuntimeError("Train manifest has missing or duplicate aggTrades archive days")
    return source_hashes


def registered_aggtrade_flow_features(
    factory: Any, rows: tuple[Any, ...], *, archive_manifest: Path | None = None,
) -> np.ndarray:
    """Build causal flow features from registered one-second aggTrades.

    Empty seconds carry zero trade flow, not missing data. The production caller
    therefore requires a checksummed, complete daily archive manifest. Synthetic
    callers without that provenance retain the conservative 99% density gate.
    """
    archive_hashes = (
        _verified_aggtrade_archive_hashes(archive_manifest, rows)
        if archive_manifest is not None else None
    )
    events = sorted(
        (event for event in factory.store.events
         if event.kind == "trade_flow_1s"
         and event.payload.get("instrument") == runner.INSTRUMENT),
        key=lambda event: event.ts_event_ns,
    )
    if not events or any(event.available_at_ns is None for event in events):
        raise RuntimeError("registered Train source lacks timestamped aggTrades flow")
    if archive_hashes is not None and any(
        getattr(event, "source_hash", None) not in archive_hashes for event in events
    ):
        raise RuntimeError("aggTrades events do not match the verified Train archive manifest")
    event_times = np.asarray([int(event.ts_event_ns) for event in events], dtype=np.int64)
    available = np.asarray([int(event.available_at_ns) for event in events], dtype=np.int64)
    if (np.any(np.diff(event_times) <= 0) or np.any(np.diff(available) <= 0)
            or np.any(available < event_times)):
        raise RuntimeError("registered aggTrades flow clocks are not strictly ordered")
    buy = np.asarray([
        float(event.payload["aggressor_buy_notional"]) for event in events
    ], dtype=np.float64)
    sell = np.asarray([
        float(event.payload["aggressor_sell_notional"]) for event in events
    ], dtype=np.float64)
    counts = np.asarray([
        int(event.payload["trade_count"]) for event in events
    ], dtype=np.int64)
    if (not np.isfinite(buy).all() or not np.isfinite(sell).all()
            or np.any(buy < 0) or np.any(sell < 0) or np.any(counts <= 0)):
        raise RuntimeError("registered aggTrades flow payload is invalid")
    buy_cum = np.concatenate(([0.0], np.cumsum(buy, dtype=np.float64)))
    sell_cum = np.concatenate(([0.0], np.cumsum(sell, dtype=np.float64)))
    count_cum = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    output = np.empty((len(rows), len(AGGTRADE_FLOW_FEATURE_NAMES)), dtype=np.float64)
    minute_ns = runner.MINUTE_NS
    decision_times = np.fromiter(
        (int(row.decision_ts_ns) for row in rows), dtype=np.int64, count=len(rows)
    )
    ends = np.searchsorted(available, decision_times, side="right")
    if np.any(ends == 0):
        raise RuntimeError("aggTrades flow has no data available at decision time")
    for feature_index, minutes in enumerate((5, 15)):
        starts = np.searchsorted(
            event_times, decision_times - minutes * minute_ns, side="right"
        )
        event_counts = ends - starts
        if np.any(event_counts <= 0):
            raise RuntimeError("aggTrades flow window contains no observed trades")
        if (archive_hashes is None
                and np.any(event_counts < int(np.ceil(minutes * 60 * 0.99)))):
            raise RuntimeError(
                f"aggTrades flow coverage below 99% in causal {minutes}m window"
            )
        total_buy = buy_cum[ends] - buy_cum[starts]
        total_sell = sell_cum[ends] - sell_cum[starts]
        total_notional = total_buy + total_sell
        if not np.isfinite(total_notional).all() or np.any(total_notional <= 0):
            raise RuntimeError("aggTrades flow window has no positive notional")
        output[:, feature_index] = (total_buy - total_sell) / total_notional
        if feature_index == 0:
            output[:, 2] = np.log1p(count_cum[ends] - count_cum[starts])
    if not np.isfinite(output).all():
        raise RuntimeError("derived aggTrades features contain non-finite values")
    return output


def evaluate_value_predictions(predictions_bps: np.ndarray, outcomes_bps: np.ndarray) -> dict:
    selected_side = np.argmax(predictions_bps, axis=1)
    expected_value = np.max(predictions_bps, axis=1)
    row_indices = np.arange(len(outcomes_bps))
    selected_realized = outcomes_bps[row_indices, selected_side]
    residual = expected_value - selected_realized
    prediction_correlation = None
    if (len(expected_value) > 1 and np.std(expected_value) > 0.0
            and np.std(selected_realized) > 0.0):
        prediction_correlation = float(np.corrcoef(
            expected_value, selected_realized
        )[0, 1])
    rows = []
    for threshold in VALUE_THRESHOLDS_BPS:
        selected = expected_value >= threshold
        indexes = np.flatnonzero(selected)
        net = outcomes_bps[indexes, selected_side[indexes]]
        rows.append({
            "minimum_predicted_net_value_bps": threshold,
            "trade_count": int(len(indexes)),
            "long_count": int(np.sum(selected_side[indexes] == 0)),
            "short_count": int(np.sum(selected_side[indexes] == 1)),
            "predicted_value_mean_bps": (
                float(np.mean(expected_value[indexes])) if len(indexes) else None
            ),
            "realized_net_mean_bps": float(np.mean(net)) if len(net) else None,
            "realized_net_median_bps": float(np.median(net)) if len(net) else None,
            "realized_net_std_bps": (
                float(np.std(net, ddof=1)) if len(net) > 1 else None
            ),
            "realized_net_positive_count": int(np.sum(net > 0.0)),
            "realized_net_worst_bps": float(np.min(net)) if len(net) else None,
        })
    return {
        "sample_count": int(len(outcomes_bps)),
        "selected_side_prediction_mae_bps": (
            float(np.mean(np.abs(residual))) if len(residual) else None
        ),
        "selected_side_prediction_bias_bps": (
            float(np.mean(residual)) if len(residual) else None
        ),
        "selected_side_prediction_realized_pearson_r": prediction_correlation,
        "selected_side_positive_sign_accuracy": (
            float(np.mean((expected_value > 0.0) == (selected_realized > 0.0)))
            if len(expected_value) else None
        ),
        "thresholds": rows,
    }


def exit_minutes(entries: np.ndarray, opens: np.ndarray, highs: np.ndarray,
                 lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Conservative per-side exit offsets for 1m OHLC first-touch paths."""
    horizon = temporal.HORIZON_MINUTES
    ids = entries[:, None] + np.arange(1, horizon + 1, dtype=np.int64)[None, :]
    entry_close = closes[entries, None]
    high_bps = np.log(highs[ids] / entry_close) * 10_000.0
    low_bps = np.log(lows[ids] / entry_close) * 10_000.0

    def first_hit(condition: np.ndarray) -> np.ndarray:
        hit = np.any(condition, axis=1)
        return np.where(hit, np.argmax(condition, axis=1), horizon)

    long_tp = first_hit(high_bps >= barrier.TAKE_PROFIT_BPS)
    long_sl = first_hit(low_bps <= -barrier.STOP_LOSS_BPS)
    short_tp = first_hit(low_bps <= -barrier.TAKE_PROFIT_BPS)
    short_sl = first_hit(high_bps >= barrier.STOP_LOSS_BPS)
    def offset(first_a: np.ndarray, first_b: np.ndarray) -> np.ndarray:
        first = np.minimum(first_a, first_b)
        return np.where(first == horizon, horizon, first + 1)

    return np.column_stack((offset(long_tp, long_sl), offset(short_tp, short_sl)))


def fixed_horizon_outcomes(
    entries: np.ndarray,
    times: np.ndarray,
    closes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Net long/short outcomes for entry-close to 240m exit-close execution."""
    horizon = temporal.HORIZON_MINUTES
    exit_ids = entries + horizon
    valid = exit_ids < len(times)
    entries = entries[valid]
    exit_ids = exit_ids[valid]
    if not len(entries):
        return np.empty((0, 2), dtype=np.float64), entries
    contiguous = times[exit_ids] - times[entries] == horizon * runner.MINUTE_NS
    entries = entries[contiguous]
    exit_ids = exit_ids[contiguous]
    if not len(entries):
        return np.empty((0, 2), dtype=np.float64), entries
    gross_bps = np.log(closes[exit_ids] / closes[entries]) * 10_000.0
    outcomes = np.column_stack((gross_bps, -gross_bps)) - barrier.ROUND_TRIP_COST_BPS
    if not np.isfinite(outcomes).all():
        raise RuntimeError("fixed-horizon outcomes contain non-finite values")
    return outcomes, entries


def registered_feature_read_scope(
    marker_path: Path,
    *,
    requested_start_ns: int,
    requested_end_ns: int,
    source_start_ns: int,
    source_end_ns: int,
) -> dict[str, Any]:
    """Expand the feature window to complete verified row groups when indexed."""
    if (type(requested_start_ns) is not int or type(requested_end_ns) is not int
            or requested_start_ns >= requested_end_ns
            or requested_start_ns < source_start_ns or requested_end_ns > source_end_ns):
        raise ValueError("feature read window is outside the registered source interval")
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "0.4.1":
        return {
            "source_mode": "whole_snapshot_isolated_v1",
            "start_ns": source_start_ns,
            "end_ns": source_end_ns,
            "row_group_count": None,
            "event_count": payload.get("events"),
        }
    groups = payload.get("row_groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("indexed Train snapshot has no row-group metadata")

    def bounds(group: dict[str, Any]) -> tuple[int, int, int, int]:
        names = (
            "min_ts_event_ns", "max_ts_event_ns",
            "min_available_at_ns", "max_available_at_ns",
        )
        if (not isinstance(group, dict)
                or any(type(group.get(name)) is not int for name in names)
                or type(group.get("events")) is not int or group["events"] <= 0):
            raise RuntimeError("indexed Train row-group bounds are malformed")
        event_start, event_end, available_start, available_end = (
            group[name] for name in names
        )
        if event_start > event_end or available_start > available_end:
            raise RuntimeError("indexed Train row-group bounds are inverted")
        return event_start, event_end, available_start, available_end

    selected_indexes = set()
    for index, group in enumerate(groups):
        event_start, event_end, available_start, available_end = bounds(group)
        touches = (
            event_end >= requested_start_ns and event_start < requested_end_ns
        ) or (
            available_end >= requested_start_ns and available_start < requested_end_ns
        )
        if touches:
            selected_indexes.add(index)
    if not selected_indexes:
        raise RuntimeError("feature window does not intersect indexed Train row groups")
    while True:
        selected_bounds = [bounds(groups[index]) for index in selected_indexes]
        scope_start_ns = min(min(row[0], row[2]) for row in selected_bounds)
        scope_end_ns = 1 + max(max(row[1], row[3]) for row in selected_bounds)
        newly_touched = set()
        for index, group in enumerate(groups):
            if index in selected_indexes:
                continue
            event_start, event_end, available_start, available_end = bounds(group)
            if (
                event_end >= scope_start_ns and event_start < scope_end_ns
            ) or (
                available_end >= scope_start_ns and available_start < scope_end_ns
            ):
                newly_touched.add(index)
        if not newly_touched:
            break
        selected_indexes.update(newly_touched)

    if scope_start_ns < source_start_ns or scope_end_ns > source_end_ns:
        raise RuntimeError("expanded feature scope escaped the verified source interval")
    scoped_groups = [(index, groups[index]) for index in sorted(selected_indexes)]
    return {
        "source_mode": "row_group_scoped_v1",
        "start_ns": scope_start_ns,
        "end_ns": scope_end_ns,
        "row_group_count": len(scoped_groups),
        "row_group_indexes": [index for index, _ in scoped_groups],
        "event_count": sum(group["events"] for _, group in scoped_groups),
    }


def evaluate_sequential_policy(
    predictions_bps: np.ndarray,
    outcomes_bps: np.ndarray,
    entries: np.ndarray,
    exits_after_minutes: np.ndarray,
) -> dict:
    """Simulate one position at a time over the five-minute decision lane."""
    selected_side = np.argmax(predictions_bps, axis=1)
    expected_value = np.max(predictions_bps, axis=1)
    thresholds = []
    for threshold in VALUE_THRESHOLDS_BPS:
        last_exit_index = -1
        realized: list[float] = []
        hold_minutes: list[int] = []
        for index, entry_index in enumerate(entries):
            if int(entry_index) <= last_exit_index:
                continue
            if expected_value[index] < threshold:
                continue
            side = int(selected_side[index])
            held = int(exits_after_minutes[index, side])
            realized.append(float(outcomes_bps[index, side]))
            hold_minutes.append(held)
            last_exit_index = int(entry_index) + held
        values = np.asarray(realized, dtype=np.float64)
        thresholds.append({
            "minimum_predicted_net_value_bps": threshold,
            "trade_count": int(values.size),
            "net_mean_bps": float(np.mean(values)) if values.size else None,
            "net_median_bps": float(np.median(values)) if values.size else None,
            "net_std_bps": float(np.std(values, ddof=1)) if values.size > 1 else None,
            "net_positive_count": int(np.sum(values > 0.0)),
            "net_worst_bps": float(np.min(values)) if values.size else None,
            "mean_holding_minutes": float(np.mean(hold_minutes)) if hold_minutes else None,
        })
    return {"candidate_anchor_count": int(len(entries)), "thresholds": thresholds}


def main() -> int:
    import torch

    training_runtime = configure_torch_threads(torch)
    torch.use_deterministic_algorithms(True)
    if (type(EVALUATION_SAMPLE_MINUTES) is not int or EVALUATION_SAMPLE_MINUTES < 5
            or temporal.HORIZON_MINUTES % EVALUATION_SAMPLE_MINUTES):
        raise ValueError("evaluation cadence must divide the 240-minute label horizon")
    source_override = os.environ.get("ROUTE_A_TRAIN_SOURCE_DIR")
    if not source_override:
        raise RuntimeError("barrier value research requires ROUTE_A_TRAIN_SOURCE_DIR")
    source_path = Path(source_override).resolve()
    if not source_path.is_relative_to(runner.ROOT.resolve()):
        raise PermissionError("flow Train snapshot must remain inside the Route A project")
    trainfold.SOURCE_DIR = source_path
    trainfold.configure()
    if OUT.exists():
        raise FileExistsError(f"immutable barrier value output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, source_scope_start, source_scope_end = runner.load_source_identity()
    feature_end_text = os.environ.get(
        "ROUTE_A_VALUE_FEATURE_END_UTC", temporal.ANALYSIS_END
    )
    feature_end_ns = runner.ns(feature_end_text)
    analysis_end_ns = runner.ns(temporal.ANALYSIS_END)
    if feature_end_ns < analysis_end_ns:
        raise ValueError("feature window cannot end before the frozen fitting window")
    environment_template = json.loads(
        runner.CONFIG.read_text(encoding="utf-8")
    )["environment"]
    feature_spec = historical_feature_spec(
        environment_template["coverage_policy"]["maximum_age_ns"],
        profile=runner.FEATURE_PROFILE,
    )
    desired_read_start_ns = (
        runner.ns(temporal.FIT_START) - feature_spec.max_lookback_ns
        - 2 * runner.MINUTE_NS
    )
    feature_read_scope = registered_feature_read_scope(
        runner.MARKER,
        requested_start_ns=desired_read_start_ns,
        requested_end_ns=analysis_end_ns,
        source_start_ns=source_scope_start,
        source_end_ns=source_scope_end,
    )
    owner = ACTIVE_OWNER.set("route-a-barrier-value-teacher-20261021")
    try:
        _, factory = runner.make_config(
            label="barrier-value-fit-apr16-may15",
            marker_sha=marker_sha,
            scope_start=feature_read_scope["start_ns"],
            scope_end=feature_read_scope["end_ns"],
            start_ns=runner.ns(temporal.FIT_START),
            end_ns=runner.ns(temporal.ANALYSIS_END),
            mode="base",
            fit_scaler=True,
            source_mode=feature_read_scope["source_mode"],
        )
    finally:
        ACTIVE_OWNER.reset(owner)

    feature_factory = factory
    extended_read_scope = None
    if feature_end_ns > analysis_end_ns:
        extended_read_scope = registered_feature_read_scope(
            runner.MARKER,
            requested_start_ns=desired_read_start_ns,
            requested_end_ns=feature_end_ns,
            source_start_ns=source_scope_start,
            source_end_ns=source_scope_end,
        )
        feature_owner = ACTIVE_OWNER.set("route-a-barrier-value-teacher-20261021")
        try:
            _, feature_factory = runner.make_config(
                label="barrier-value-causal-features-through-evaluation",
                marker_sha=marker_sha,
                scope_start=extended_read_scope["start_ns"],
                scope_end=extended_read_scope["end_ns"],
                start_ns=runner.ns(temporal.FIT_START),
                end_ns=feature_end_ns,
                mode="base",
                fit_scaler=False,
                scaler=factory.scaler,
                source_mode=extended_read_scope["source_mode"],
            )
        finally:
            ACTIVE_OWNER.reset(feature_owner)

    rows = feature_factory.rows
    times = np.asarray([row.decision_ts_ns for row in rows], dtype=np.int64)
    field_names = tuple(field.name for field in feature_factory.features.fields)
    feature_indices = [field_names.index(name) for name in temporal.FEATURES]
    close_index = field_names.index("close_1m")
    base_features = np.asarray(
        [[row.values[index] for index in feature_indices] for row in rows], dtype=np.float64
    )
    if FLOW_FEATURE_SOURCE == "aggtrades_1s":
        flow_features = registered_aggtrade_flow_features(
            feature_factory, rows,
            archive_manifest=trainfold.SOURCE_DIR / "import.json",
        )
        flow_feature_names = AGGTRADE_FLOW_FEATURE_NAMES
    else:
        flow_features = barrier.registered_flow_features(feature_factory, rows)
        flow_feature_names = barrier.FLOW_FEATURE_NAMES
    features = np.column_stack((base_features, flow_features))
    bars = getattr(feature_factory, "_bars", ())
    if len(bars) != len(rows) or any(
        row.decision_ts_ns != int(bar.available_at_ns or -1)
        for row, bar in zip(rows, bars, strict=True)
    ):
        raise RuntimeError("registered Train feature rows do not align with source OHLC bars")
    opens = np.asarray([float(bar.payload["open"]) for bar in bars], dtype=np.float64)
    highs = np.asarray([float(bar.payload["high"]) for bar in bars], dtype=np.float64)
    lows = np.asarray([float(bar.payload["low"]) for bar in bars], dtype=np.float64)
    closes = np.asarray([row.values[close_index] for row in rows], dtype=np.float64)
    if not np.isfinite(features).all() or np.any(np.diff(times) != runner.MINUTE_NS):
        raise RuntimeError("barrier value input rows are non-finite or unordered")

    pending_folds = []
    worker_count = min(PARALLEL_WORKERS, len(temporal.FOLDS))
    logical_cpu_count = os.cpu_count() or 1
    default_worker_threads = max(1, min(8, logical_cpu_count // worker_count))
    worker_threads = int(os.environ.get(
        "ROUTE_A_VALUE_WORKER_THREADS", str(default_worker_threads)
    ))
    if not 1 <= worker_threads <= logical_cpu_count:
        raise ValueError("ROUTE_A_VALUE_WORKER_THREADS is outside the CPU range")
    training_started = perf_counter()
    scenario_case_rows = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for fold_id, train_start_text, train_end_text, eval_start_text, eval_end_text in temporal.FOLDS:
            train_start, train_end = runner.ns(train_start_text), runner.ns(train_end_text)
            eval_start, eval_end = runner.ns(eval_start_text), runner.ns(eval_end_text)
            train_entries, _, train_times = temporal.anchor_rows(
                times, closes, train_start, train_end, temporal.TRAIN_SAMPLE_MINUTES
            )
            eval_entries, _, eval_times = temporal.anchor_rows(
                times, closes, eval_start, eval_end, EVALUATION_SAMPLE_MINUTES
            )
            outcome_builder = (
                fixed_horizon_outcomes
                if TARGET_MODE == "fixed_horizon" else barrier.barrier_outcomes
            )
            if TARGET_MODE == "fixed_horizon":
                train_outcomes, train_entries = outcome_builder(
                    train_entries, times, closes
                )
                eval_outcomes, eval_entries = outcome_builder(
                    eval_entries, times, closes
                )
            else:
                train_outcomes, train_entries = outcome_builder(
                    train_entries, times, opens, highs, lows, closes
                )
                eval_outcomes, eval_entries = outcome_builder(
                    eval_entries, times, opens, highs, lows, closes
                )
            x_train, train_entries = temporal.make_sequences(train_entries, times, features)
            x_eval, eval_entries = temporal.make_sequences(eval_entries, times, features)
            train_time_set = set(int(times[index]) for index in train_entries)
            eval_time_set = set(int(times[index]) for index in eval_entries)
            train_mask = np.asarray(
                [int(value) in train_time_set for value in train_times], dtype=np.bool_
            )
            eval_mask = np.asarray(
                [int(value) in eval_time_set for value in eval_times], dtype=np.bool_
            )
            train_outcomes = train_outcomes[train_mask]
            eval_outcomes = eval_outcomes[eval_mask]
            if len(x_train) < 500 or len(x_eval) < 10:
                raise RuntimeError(f"insufficient barrier value samples in {fold_id}")
            label_end = int(np.max(train_times[train_mask])
                            + temporal.HORIZON_MINUTES * runner.MINUTE_NS)
            if label_end > eval_start:
                raise RuntimeError(f"training path labels overlap evaluation in {fold_id}")
            future = executor.submit(
                _fit_value_fold_worker, x_train, train_outcomes, x_eval, worker_threads
            )
            pending_folds.append({
                "future": future,
                "fold_id": fold_id,
                "train_start_ns": train_start,
                "train_end_ns_exclusive": train_end,
                "evaluation_start_ns": eval_start,
                "evaluation_end_ns_exclusive": eval_end,
                "training_label_end_max_ns": label_end,
                "train_samples": int(len(train_outcomes)),
                "eval_outcomes": eval_outcomes,
                "eval_entries": eval_entries,
                "eval_sequences": x_eval,
                "eval_exit_minutes": (
                    np.full(
                        (len(eval_entries), 2), temporal.HORIZON_MINUTES,
                        dtype=np.int64,
                    )
                    if TARGET_MODE == "fixed_horizon"
                    else exit_minutes(eval_entries, opens, highs, lows, closes)
                ),
            })

    training_wall_seconds = perf_counter() - training_started
    folds = []
    pooled_predictions_by_cost = {cost: [] for cost in EVALUATION_COSTS_BPS}
    pooled_outcomes_by_cost = {cost: [] for cost in EVALUATION_COSTS_BPS}
    for pending in pending_folds:
        predictions, fit_meta = pending["future"].result()
        eval_outcomes = pending["eval_outcomes"]
        eval_entries = pending["eval_entries"]
        eval_sequences = pending["eval_sequences"]
        predicted_sides = np.argmax(predictions, axis=1)
        cost_rows = []
        cost_sensitivity = []
        for evaluation_cost_bps in EVALUATION_COSTS_BPS:
            cost_adjustment = barrier.ROUND_TRIP_COST_BPS - evaluation_cost_bps
            adjusted_predictions = predictions + cost_adjustment
            adjusted_outcomes = eval_outcomes + cost_adjustment
            pooled_predictions_by_cost[evaluation_cost_bps].append(adjusted_predictions)
            pooled_outcomes_by_cost[evaluation_cost_bps].append(adjusted_outcomes)
            selected_values = adjusted_predictions[
                np.arange(len(adjusted_predictions)), predicted_sides
            ]
            realized_values = adjusted_outcomes[
                np.arange(len(adjusted_outcomes)), predicted_sides
            ]
            cost_rows.append({
                "round_trip_cost_bps": evaluation_cost_bps,
                "predicted_long_short_net_value_bps": adjusted_predictions.tolist(),
                "realized_long_short_ohlc_outcome_bps": adjusted_outcomes.tolist(),
                "selected_predicted_net_value_bps": selected_values.tolist(),
                "selected_realized_ohlc_outcome_bps": realized_values.tolist(),
            })
            cost_sensitivity.append({
                "round_trip_cost_bps": evaluation_cost_bps,
                "value_prediction_diagnostics": evaluate_value_predictions(
                    adjusted_predictions, adjusted_outcomes
                ),
                "evaluation": evaluate_sequential_policy(
                    adjusted_predictions, adjusted_outcomes,
                    eval_entries, pending["eval_exit_minutes"],
                ),
            })
        for index, entry in enumerate(eval_entries):
            scenario_case_rows.append({
                "fold_id": pending["fold_id"],
                "decision_ts_ns": int(times[entry]),
                "selected_side": "long" if predicted_sides[index] == 0 else "short",
                "training_cost_bps": barrier.ROUND_TRIP_COST_BPS,
                "training_cost_predicted_long_short_net_value_bps": (
                    predictions[index].tolist()
                ),
                "training_cost_realized_long_short_ohlc_outcome_bps": (
                    eval_outcomes[index].tolist()
                ),
                "exit_after_minutes_long_short": (
                    pending["eval_exit_minutes"][index].tolist()
                ),
                "cost_sensitivity": [
                    {
                        "round_trip_cost_bps": row["round_trip_cost_bps"],
                        "predicted_long_short_net_value_bps": (
                            row["predicted_long_short_net_value_bps"][index]
                        ),
                        "realized_long_short_ohlc_outcome_bps": (
                            row["realized_long_short_ohlc_outcome_bps"][index]
                        ),
                    }
                    for row in cost_rows
                ],
                "causal_feature_sequence": eval_sequences[index].tolist(),
                "case_kind": (
                    "FIXED_HORIZON_SCENARIO_CANDIDATE"
                    if TARGET_MODE == "fixed_horizon"
                    else "OHLC_BARRIER_SCENARIO_CANDIDATE"
                ),
                "target_mode": TARGET_MODE,
                "economic_status": "SCENARIO_ONLY",
                "native_execution_verified": False,
                "student_eligible": False,
            })
        folds.append({
            "fold_id": pending["fold_id"],
            "train_start_ns": pending["train_start_ns"],
            "train_end_ns_exclusive": pending["train_end_ns_exclusive"],
            "evaluation_start_ns": pending["evaluation_start_ns"],
            "evaluation_end_ns_exclusive": pending["evaluation_end_ns_exclusive"],
            "training_label_end_max_ns": pending["training_label_end_max_ns"],
            "train_samples": pending["train_samples"],
            "fit": fit_meta,
            "fit_cache_reused": False,
            "fit_reference_fold": pending["fold_id"],
            "value_prediction_diagnostics": evaluate_value_predictions(
                predictions, eval_outcomes
            ),
            "evaluation": evaluate_sequential_policy(
                predictions, eval_outcomes, eval_entries,
                pending["eval_exit_minutes"],
            ),
            "cost_sensitivity": cost_sensitivity,
        })

    candidate_path = OUT / "case-candidates.jsonl"
    candidate_bytes = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in scenario_case_rows
    ).encode("utf-8")
    candidate_path.write_bytes(candidate_bytes)
    candidate_digest = sha256(candidate_bytes).hexdigest()
    pooled_prediction_quality = {
        str(cost): evaluate_value_predictions(
            np.vstack(pooled_predictions_by_cost[cost]),
            np.vstack(pooled_outcomes_by_cost[cost]),
        )
        for cost in EVALUATION_COSTS_BPS
    }

    report = {
        "schema": "rles.route-a.barrier-value-teacher.v1",
        "purpose": (
            "Train-inner GRU value diagnostic; not native execution or promotion evidence"
        ),
        "experiment_class": "training_performance_diagnostic",
        "promotion_evidence": False,
        "training_protocol": {
            "max_epochs": EPOCHS,
            "minimum_epochs": MIN_EPOCHS,
            "target_train_loss": TARGET_TRAIN_LOSS,
            "target_policy": (
            "target is a reported fit objective; stop when purged calibration "
            "patience is exhausted and select the best calibration checkpoint"
            ),
        "parallel_training": {
            "mode": "spawned process per independent walk-forward fold",
            "workers": worker_count,
            "torch_threads_per_worker": worker_threads,
            "fold_training_wall_seconds": training_wall_seconds,
            "sum_worker_cpu_seconds": sum(
                fold["fit"]["fit_process_cpu_seconds"] for fold in folds
            ),
            "aggregate_logical_cpu_utilization_percent": (
                100.0 * sum(fold["fit"]["fit_process_cpu_seconds"] for fold in folds)
                / training_wall_seconds / training_runtime["logical_cpu_count"]
                if training_wall_seconds > 0 else None
            ),
        },
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "calibration_fraction": CALIBRATION_FRACTION,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            **training_runtime,
            "requested_device": os.environ.get("ROUTE_A_TRAIN_DEVICE", "auto"),
            "flow_feature_source": FLOW_FEATURE_SOURCE,
            "target_mode": TARGET_MODE,
        },
        "registered_market_read_scope": {
            "fit_factory": feature_read_scope,
            "extended_feature_factory": extended_read_scope,
        },
        "economic_status": "SCENARIO_ONLY",
        "case_candidate_artifact": {
            "path": candidate_path.name,
            "sha256": candidate_digest,
            "row_count": len(scenario_case_rows),
            "all_student_eligible": False,
            "warning": (
                "OHLC path scenarios are diagnostic examples only; no native "
                "orders, fills, queue position, or impact are verified"
            ),
        },
        "pooled_prediction_quality_by_cost_bps": pooled_prediction_quality,
        "source_marker_sha256": marker_sha,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "classifier_report_sha256": sha256(
            (runner.ROOT / ".artifacts" / "route-a-barrier-label-teacher-20261021-run04"
             / "barrier-label-teacher-report.json").read_bytes()
        ).hexdigest(),
        "features": list(temporal.FEATURES + flow_feature_names),
        "value_target": (
            "two side-specific 240m entry-close to exit-close returns with round-trip "
            "cost deducted"
            if TARGET_MODE == "fixed_horizon"
            else "two separate 240m close-path first-hit OHLC net outcomes, round-trip cost already deducted"
        ),
        "label_action_alignment": (
            "position enters at decision close and exits at the 240m close; no "
            "intrabar stop/target assumption"
            if TARGET_MODE == "fixed_horizon"
            else "first-touch barrier label is a scenario proxy and requires a separate native stop/target action mapping"
        ),
        "cost_sensitivity": {
            "training_round_trip_cost_bps": barrier.ROUND_TRIP_COST_BPS,
            "evaluation_round_trip_cost_bps": list(EVALUATION_COSTS_BPS),
            "method": (
                "constant cost-hurdle translation of model predictions and realized "
                "OHLC outcomes; scenario-only, not native fill validation"
            ),
        },
        "evaluation_policy": {
            "candidate_decision_cadence_minutes": EVALUATION_SAMPLE_MINUTES,
            "position_limit": 1,
            "reentry": "next eligible five-minute decision strictly after prior exit bar",
            "entry_thresholds_are_fixed_diagnostics": list(VALUE_THRESHOLDS_BPS),
        },
        "feature_window_ns": [runner.ns(temporal.FIT_START), feature_end_ns],
        "thresholds_are_fixed_diagnostics": list(VALUE_THRESHOLDS_BPS),
        "folds": folds,
    }
    output = OUT / "barrier-value-teacher-report.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest, encoding="ascii")
    aggregate = []
    for threshold_index, threshold in enumerate(VALUE_THRESHOLDS_BPS):
        rows_at_threshold = [fold["evaluation"]["thresholds"][threshold_index] for fold in folds]
        count = sum(row["trade_count"] for row in rows_at_threshold)
        mean = (
            sum(row["net_mean_bps"] * row["trade_count"]
                for row in rows_at_threshold if row["trade_count"]) / count if count else None
        )
        aggregate.append({
            "minimum_predicted_net_value_bps": threshold,
            "trade_count": count,
            "trade_weighted_net_mean_bps": mean,
            "positive_folds": sum(
                row["net_mean_bps"] is not None
                and row["net_mean_bps"] > 0 for row in rows_at_threshold
            ),
        })
    cost_aggregates = []
    for cost_index, evaluation_cost_bps in enumerate(EVALUATION_COSTS_BPS):
        thresholds = []
        for threshold_index, threshold in enumerate(VALUE_THRESHOLDS_BPS):
            rows_at_threshold = [
                fold["cost_sensitivity"][cost_index]["evaluation"]["thresholds"][threshold_index]
                for fold in folds
            ]
            count = sum(row["trade_count"] for row in rows_at_threshold)
            mean = (
                sum(row["net_mean_bps"] * row["trade_count"]
                    for row in rows_at_threshold if row["trade_count"]) / count
                if count else None
            )
            thresholds.append({
                "minimum_predicted_net_value_bps": threshold,
                "trade_count": count,
                "trade_weighted_net_mean_bps": mean,
                "positive_folds": sum(
                    row["net_mean_bps"] is not None and row["net_mean_bps"] > 0
                    for row in rows_at_threshold
                ),
            })
        cost_aggregates.append({
            "round_trip_cost_bps": evaluation_cost_bps,
            "thresholds": thresholds,
        })
    print(json.dumps({"output": str(OUT), "report_sha256": digest,
                      "threshold_aggregates": aggregate,
                      "cost_sensitivity_aggregates": cost_aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
