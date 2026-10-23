"""Train-only event teacher using cost-aware, close-path barrier outcomes.

For each causal decision, simulate long and short outcomes over a 240-minute
1m OHLC path. A +60 bps take-profit or adverse 30 bps stop triggers an exit;
when both touch in the same minute, the stop is assumed first. Stop gaps use
the worse open. Otherwise the position exits at the horizon close. A 30 bps
round-trip hurdle is deducted before assigning LONG, SHORT, or FLAT labels.
This is an OHLC label diagnostic, not a substitute for native execution.
"""

from __future__ import annotations

from hashlib import sha256
from bisect import bisect_right
import json
import os
from pathlib import Path

import numpy as np

import probe_route_a_temporal_event_classifier_teacher_20261021 as temporal
import run_route_a_trainfold_screen_20261021 as trainfold
import run_route_a_train_student_screen_20261020 as runner
from rles_orchestrator.research_access import ACTIVE_OWNER


RUN_ID = os.environ.get("ROUTE_A_BARRIER_LABEL_RUN", "run01")
OUT = runner.ROOT / ".artifacts" / f"route-a-barrier-label-teacher-20261021-{RUN_ID}"
TAKE_PROFIT_BPS = 60.0
STOP_LOSS_BPS = 30.0
ROUND_TRIP_COST_BPS = 30.0
THRESHOLDS = (0.0, 0.50, 0.60, 0.70, 0.80)
NAMES = temporal.CLASS_NAMES
FLOW_FEATURE_NAMES = (
    "taker_buy_imbalance_5m",
    "taker_buy_imbalance_15m",
    "log_trade_count_5m",
)
FEATURE_NAMES = temporal.FEATURES + FLOW_FEATURE_NAMES


def registered_flow_features(factory, rows) -> np.ndarray:
    """Derive point-in-time kline flow from the registered Train snapshot."""
    events = sorted(
        (event for event in factory.store.events
         if event.kind == "ohlcv_1m" and event.payload.get("instrument") == runner.INSTRUMENT),
        key=lambda event: event.ts_event_ns,
    )
    if not events or any(event.available_at_ns is None for event in events):
        raise RuntimeError("registered Train source lacks timestamped OHLCV events")
    available = [int(event.available_at_ns) for event in events]
    if any(
        not {"quote_asset_volume", "number_of_trades", "taker_buy_quote_asset_volume"}
        <= set(event.payload)
        for event in events
    ):
        raise RuntimeError("registered Train source lacks preserved kline flow fields")
    quote = np.asarray([float(event.payload["quote_asset_volume"]) for event in events])
    taker_buy = np.asarray([
        float(event.payload["taker_buy_quote_asset_volume"]) for event in events
    ])
    trades = np.asarray([int(event.payload["number_of_trades"]) for event in events])
    quote_cum = np.concatenate(([0.0], np.cumsum(quote)))
    buy_cum = np.concatenate(([0.0], np.cumsum(taker_buy)))
    trade_cum = np.concatenate(([0], np.cumsum(trades)))

    output = np.empty((len(rows), len(FLOW_FEATURE_NAMES)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        event_index = bisect_right(available, row.decision_ts_ns) - 1
        if (event_index < 0 or available[event_index] != row.decision_ts_ns
                or events[event_index].available_at_ns != row.decision_ts_ns):
            raise RuntimeError("flow event is not aligned with the causal feature decision")
        for output_index, window in enumerate((5, 15)):
            start = max(0, event_index + 1 - window)
            total_quote = quote_cum[event_index + 1] - quote_cum[start]
            total_buy = buy_cum[event_index + 1] - buy_cum[start]
            output[row_index, output_index] = (
                2.0 * total_buy / total_quote - 1.0 if total_quote > 0.0 else 0.0
            )
        start = max(0, event_index + 1 - 5)
        total_trades = trade_cum[event_index + 1] - trade_cum[start]
        output[row_index, 2] = np.log1p(total_trades)
    if not np.isfinite(output).all():
        raise RuntimeError("derived causal flow features contain non-finite values")
    return output


def barrier_outcomes(
    entries: np.ndarray,
    times: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
):
    """Return strict 1m-contiguous OHLC barrier outcomes and valid entry rows."""
    horizon = temporal.HORIZON_MINUTES
    offsets = np.arange(1, horizon + 1, dtype=np.int64)
    path_ids = entries[:, None] + offsets[None, :]
    in_bounds = path_ids[:, -1] < len(times)
    entries = entries[in_bounds]
    path_ids = path_ids[in_bounds]
    if not len(entries):
        return np.empty((0, 2), dtype=np.float64), entries

    expected_ns = runner.MINUTE_NS
    contiguous = np.all(np.diff(times[path_ids], axis=1) == expected_ns, axis=1)
    entries = entries[contiguous]
    path_ids = path_ids[contiguous]
    if not len(entries):
        return np.empty((0, 2), dtype=np.float64), entries

    entry_prices = closes[entries, None]
    open_bps = np.log(opens[path_ids] / entry_prices) * 10_000.0
    high_bps = np.log(highs[path_ids] / entry_prices) * 10_000.0
    low_bps = np.log(lows[path_ids] / entry_prices) * 10_000.0
    close_bps = np.log(closes[path_ids] / entry_prices) * 10_000.0
    def first_hit(condition: np.ndarray) -> np.ndarray:
        hit = np.any(condition, axis=1)
        first = np.argmax(condition, axis=1)
        return np.where(hit, first, horizon)

    long_tp = first_hit(high_bps >= TAKE_PROFIT_BPS)
    long_sl = first_hit(low_bps <= -STOP_LOSS_BPS)
    short_tp = first_hit(low_bps <= -TAKE_PROFIT_BPS)
    short_sl = first_hit(high_bps >= STOP_LOSS_BPS)

    row_ids = np.arange(len(entries), dtype=np.int64)
    long_exit = np.minimum(long_tp, long_sl)
    long_open_at_exit = open_bps[row_ids, np.minimum(long_exit, horizon - 1)]
    long_stop_first = long_sl <= long_tp
    long_underlying_return = np.where(
        long_exit == horizon,
        close_bps[:, -1],
        np.where(
            long_stop_first,
            np.minimum(-STOP_LOSS_BPS, long_open_at_exit),
            np.maximum(TAKE_PROFIT_BPS, long_open_at_exit),
        ),
    )

    short_exit = np.minimum(short_tp, short_sl)
    short_open_at_exit = open_bps[row_ids, np.minimum(short_exit, horizon - 1)]
    short_stop_first = short_sl <= short_tp
    short_underlying_return = np.where(
        short_exit == horizon,
        close_bps[:, -1],
        np.where(
            short_stop_first,
            np.maximum(STOP_LOSS_BPS, short_open_at_exit),
            np.minimum(-TAKE_PROFIT_BPS, short_open_at_exit),
        ),
    )
    long_gross = long_underlying_return
    short_gross = -short_underlying_return
    outcomes = np.column_stack((long_gross, short_gross)) - ROUND_TRIP_COST_BPS
    return outcomes, entries


def path_labels(outcomes: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(outcomes), dtype=np.int64)
    choose_long = (outcomes[:, 0] > 0.0) & (outcomes[:, 0] >= outcomes[:, 1])
    choose_short = (outcomes[:, 1] > 0.0) & (outcomes[:, 1] > outcomes[:, 0])
    labels[choose_long] = 1
    labels[choose_short] = 2
    return labels


def summarize(probabilities: np.ndarray, outcomes: np.ndarray) -> dict:
    proposed = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities[:, 1:], axis=1)
    threshold_rows = []
    for threshold in THRESHOLDS:
        selected = (proposed != 0) & (confidence >= threshold)
        indexes = np.flatnonzero(selected)
        net = np.asarray([
            outcomes[index, proposed[index] - 1] for index in indexes
        ], dtype=np.float64)
        threshold_rows.append({
            "confidence_threshold": threshold,
            "trade_count": int(len(indexes)),
            "long_count": int(np.sum(proposed[indexes] == 1)),
            "short_count": int(np.sum(proposed[indexes] == 2)),
            "net_mean_bps": float(np.mean(net)) if len(net) else None,
            "net_median_bps": float(np.median(net)) if len(net) else None,
            "net_std_bps": float(np.std(net, ddof=1)) if len(net) > 1 else None,
            "net_positive_count": int(np.sum(net > 0.0)),
            "net_worst_bps": float(np.min(net)) if len(net) else None,
        })
    truth = path_labels(outcomes)
    return {
        "sample_count": int(len(outcomes)),
        "realized_outcome_class_counts": {
            name: int(count) for name, count in zip(
                NAMES, np.bincount(truth, minlength=len(NAMES)), strict=True
            )
        },
        "predicted_class_counts": {
            name: int(np.sum(proposed == index)) for index, name in enumerate(NAMES)
        },
        "classification_accuracy": float(np.mean(proposed == truth)),
        "predicted_trade_thresholds": threshold_rows,
        "mean_long_net_bps_all_anchors": float(np.mean(outcomes[:, 0])),
        "mean_short_net_bps_all_anchors": float(np.mean(outcomes[:, 1])),
    }


def main() -> int:
    import torch

    runner.configure_torch_threads(torch)
    torch.use_deterministic_algorithms(True)
    source_override = os.environ.get("ROUTE_A_TRAIN_SOURCE_DIR")
    if not source_override:
        raise RuntimeError("flow-label research requires ROUTE_A_TRAIN_SOURCE_DIR")
    source_path = Path(source_override).resolve()
    if not source_path.is_relative_to(runner.ROOT.resolve()):
        raise PermissionError("flow Train snapshot must remain inside the Route A project")
    trainfold.SOURCE_DIR = source_path
    trainfold.configure()
    if OUT.exists():
        raise FileExistsError(f"immutable barrier-label output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    owner = ACTIVE_OWNER.set("route-a-barrier-label-teacher-20261021")
    try:
        _, factory = runner.make_config(
            label="barrier-label-fit-apr16-may15",
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
    base_features = np.asarray(
        [[row.values[index] for index in feature_indices] for row in rows], dtype=np.float64
    )
    features = np.column_stack((base_features, registered_flow_features(factory, rows)))
    bars = getattr(factory, "_bars", ())
    if len(bars) != len(rows) or any(
        row.decision_ts_ns != int(bar.available_at_ns or -1)
        for row, bar in zip(rows, bars, strict=True)
    ):
        raise RuntimeError("registered Train feature rows do not align with source OHLC bars")
    opens = np.asarray([float(bar.payload["open"]) for bar in bars], dtype=np.float64)
    highs = np.asarray([float(bar.payload["high"]) for bar in bars], dtype=np.float64)
    lows = np.asarray([float(bar.payload["low"]) for bar in bars], dtype=np.float64)
    closes = np.asarray([row.values[close_index] for row in rows], dtype=np.float64)
    if (not np.isfinite(features).all() or not np.isfinite(closes).all()
            or not np.isfinite(opens).all() or not np.isfinite(highs).all()
            or not np.isfinite(lows).all() or np.any(np.diff(times) != runner.MINUTE_NS)):
        raise RuntimeError("barrier labels require finite causal features and contiguous 1m rows")

    folds = []
    for fold_id, train_start_text, train_end_text, eval_start_text, eval_end_text in temporal.FOLDS:
        train_start, train_end = runner.ns(train_start_text), runner.ns(train_end_text)
        eval_start, eval_end = runner.ns(eval_start_text), runner.ns(eval_end_text)
        train_entries, _, train_times = temporal.anchor_rows(
            times, closes, train_start, train_end, temporal.TRAIN_SAMPLE_MINUTES
        )
        eval_entries, _, eval_times = temporal.anchor_rows(
            times, closes, eval_start, eval_end, temporal.EVALUATION_SAMPLE_MINUTES
        )
        train_outcomes, train_entries = barrier_outcomes(
            train_entries, times, opens, highs, lows, closes
        )
        eval_outcomes, eval_entries = barrier_outcomes(
            eval_entries, times, opens, highs, lows, closes
        )
        x_train, train_entries = temporal.make_sequences(train_entries, times, features)
        x_eval, eval_entries = temporal.make_sequences(eval_entries, times, features)

        train_time_set = set(int(times[index]) for index in train_entries)
        eval_time_set = set(int(times[index]) for index in eval_entries)
        train_mask = np.asarray([int(value) in train_time_set for value in train_times])
        eval_mask = np.asarray([int(value) in eval_time_set for value in eval_times])
        train_outcomes = train_outcomes[train_mask]
        eval_outcomes = eval_outcomes[eval_mask]
        train_labels = path_labels(train_outcomes)
        if len(x_train) < 500 or len(x_eval) < 10:
            raise RuntimeError(f"insufficient barrier-label samples in {fold_id}")
        train_label_end = int(np.max(train_times[train_mask])
                              + temporal.HORIZON_MINUTES * runner.MINUTE_NS)
        if train_label_end > eval_start:
            raise RuntimeError(f"training path labels overlap evaluation in {fold_id}")

        probabilities, fit_meta = temporal.fit_classifier(x_train, train_labels, x_eval)
        folds.append({
            "fold_id": fold_id,
            "train_start_ns": train_start,
            "train_end_ns_exclusive": train_end,
            "evaluation_start_ns": eval_start,
            "evaluation_end_ns_exclusive": eval_end,
            "training_label_end_max_ns": train_label_end,
            "train_anchor_count": int(len(train_labels)),
            "training_path_class_counts": {
                name: int(count) for name, count in zip(
                    NAMES, np.bincount(train_labels, minlength=len(NAMES)), strict=True
                )
            },
            "fit": fit_meta,
            "evaluation": summarize(probabilities, eval_outcomes),
        })

    report = {
        "schema": "rles.route-a.barrier-label-teacher.v1",
        "purpose": "Train-inner close-path barrier outcome predictability; not native execution or promotion evidence",
        "economic_status": "SCENARIO_ONLY",
        "source_marker_sha256": marker_sha,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "wfo_report_sha256": sha256(trainfold.WFO_REPORT.read_bytes()).hexdigest(),
        "lookback_minutes": temporal.LOOKBACK_MINUTES,
        "history_sample_minutes": temporal.HISTORY_SAMPLE_MINUTES,
        "features": list(FEATURE_NAMES),
        "evaluation_anchor_minutes": temporal.EVALUATION_SAMPLE_MINUTES,
        "horizon_minutes": temporal.HORIZON_MINUTES,
        "label_definition": {
            "take_profit_bps": TAKE_PROFIT_BPS,
            "stop_loss_bps": STOP_LOSS_BPS,
            "round_trip_cost_hurdle_bps": ROUND_TRIP_COST_BPS,
            "path_source": "registered 1-minute OHLC; source bar aligned to causal feature row",
            "same_minute_collision": "stop is assumed first when target and stop are both touched in one minute",
            "stop_gap_fill": "worse of stop threshold and observed minute open",
            "timeout_exit": "240-minute close",
            "class_rule": "choose the better side only when its simulated net bps is positive; otherwise FLAT",
        },
        "purge_policy": "training path label end no later than evaluation start; evaluation anchors spaced at horizon",
        "folds": folds,
    }
    output = OUT / "barrier-label-teacher-report.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest, encoding="ascii")
    aggregates = []
    for threshold in THRESHOLDS:
        rows_at_threshold = [
            fold["evaluation"]["predicted_trade_thresholds"][THRESHOLDS.index(threshold)]
            for fold in folds
        ]
        counts = [row["trade_count"] for row in rows_at_threshold]
        total = sum(counts)
        mean = (
            sum(row["net_mean_bps"] * row["trade_count"] for row in rows_at_threshold
                if row["trade_count"]) / total if total else None
        )
        aggregates.append({
            "confidence_threshold": threshold,
            "trade_count": total,
            "trade_weighted_net_mean_bps": mean,
            "positive_folds": sum(
                row["net_mean_bps"] is not None and row["net_mean_bps"] > 0
                for row in rows_at_threshold
            ),
        })
    print(json.dumps({"output": str(OUT), "report_sha256": digest,
                      "threshold_aggregates": aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
