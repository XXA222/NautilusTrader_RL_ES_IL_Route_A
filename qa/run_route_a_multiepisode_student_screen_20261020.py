"""Train BC on repeated flat-reset Train episodes, then screen later Train days.

The purpose is to test whether the single-episode BC fit omitted enough OPEN
examples to generalize from a flat reset. All windows remain registered Train;
the screen is development evidence and cannot certify account profitability.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import gc
import os
from time import perf_counter, process_time

import numpy as np
import torch
from torch.utils.data import DataLoader

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.config import BCConfig
from rles_route_a.contracts import INTENTS
from rles_route_a.datasets.sequence_dataset import EpisodeSequenceDataset
from rles_route_a.evaluation import (
    NativePair,
    StudentPair,
    bc_action_schema_sha256,
    bc_actor_sha256,
    bc_observation_schema_sha256,
    bc_student_recipe_sha256,
    screen_student,
    screen_teacher,
    verify_student_screen,
)
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.datasets.daggerset import DAggerSet
from rles_route_a.integration.native_rollout import teacher_native_rollout
from rles_route_a.integration.gym_rollout import NativeGymDaggerTrainProvider
from rles_route_a.models.bc import BCGRUPolicy
from rles_route_a.teachers.trend import TrendTeacher
from rles_route_a.teachers.wait import WaitTeacher
from rles_route_a.training.bc_trainer import train_bc
from rles_route_a.training.dagger_collector import DAggerCollector
from rles_route_a.training.dagger_trainer import fine_tune_bc_on_daggerset
from rles_route_a.training.native_bc import _native_behavior_summary

import run_route_a_train_student_screen_20261020 as runner
from probe_route_a_stop_cooldown_20261021 import StopCooldownTeacher
from route_a_training_runtime import configure_torch_threads


EXPERIMENT_RUN = os.environ.get("ROUTE_A_EXPERIMENT_RUN", "run35")
INTENT_CLASS_WEIGHTING = os.environ.get("ROUTE_A_INTENT_CLASS_WEIGHTING", "none")
INTENT_SAMPLING = os.environ.get("ROUTE_A_INTENT_SAMPLING", "balanced_intent_v1")
TEACHER_MODE = os.environ.get("ROUTE_A_TEACHER_MODE", "countertrend_v1")
INNER_VALIDATION_DATES = tuple(
    value.strip().replace("-", "")
    for value in os.environ.get("ROUTE_A_INNER_VALIDATION_DATES", "").split(",")
    if value.strip()
)
TRAIN_EPOCHS = int(os.environ.get("ROUTE_A_TRAIN_EPOCHS", "20"))
STOP_LOSS_FRACTION = float(os.environ.get("ROUTE_A_STOP_LOSS_FRACTION", "0"))
STOP_COOLDOWN_NS = int(os.environ.get("ROUTE_A_STOP_COOLDOWN_NS", "0"))
MINIMUM_POSITION_HOLD_NS = int(os.environ.get("ROUTE_A_MINIMUM_POSITION_HOLD_NS", "0"))
OPEN_INTENT_MIN_CONFIDENCE = float(
    os.environ.get("ROUTE_A_OPEN_INTENT_MIN_CONFIDENCE", "0")
)
CALIBRATE_OPEN_CONFIDENCE = os.environ.get(
    "ROUTE_A_CALIBRATE_OPEN_CONFIDENCE", "0"
) == "1"
DAGGER_ROLLOUTS = os.environ.get("ROUTE_A_DAGGER_ROLLOUTS", "0") == "1"
DAGGER_SEED = int(os.environ.get("ROUTE_A_DAGGER_SEED", "101"))
DAGGER_EXPERT_FRACTION = float(os.environ.get("ROUTE_A_DAGGER_EXPERT_FRACTION", "0.7"))
OUT = runner.ROOT / ".artifacts" / f"route-a-multiepisode-student-screen-20261020-{EXPERIMENT_RUN}"
_DEFAULT_FIT_DAYS = (
    "2024-02-01", "2024-02-08", "2024-02-15", "2024-02-22",
    "2024-03-01", "2024-03-08", "2024-03-15", "2024-03-22",
)
_FIT_DAYS_RAW = os.environ.get("ROUTE_A_FIT_DAYS")
FIT_DAYS = tuple(
    value.strip() for value in _FIT_DAYS_RAW.split(",") if value.strip()
) if _FIT_DAYS_RAW is not None else _DEFAULT_FIT_DAYS
if (not FIT_DAYS or len(set(FIT_DAYS)) != len(FIT_DAYS)
        or any(datetime.fromisoformat(value).date().isoformat() != value
               for value in FIT_DAYS)):
    raise ValueError("ROUTE_A_FIT_DAYS must be unique ISO calendar dates")

_DEFAULT_SCREEN_WINDOWS = (
    ("april13", "2024-04-13T00:00:00+00:00", "2024-04-14T00:00:00+00:00"),
    ("april14", "2024-04-14T00:00:00+00:00", "2024-04-15T00:00:00+00:00"),
)
_SCREEN_WINDOWS_RAW = os.environ.get("ROUTE_A_SCREEN_WINDOWS")
if _SCREEN_WINDOWS_RAW is None:
    SCREEN_WINDOWS = _DEFAULT_SCREEN_WINDOWS
else:
    parsed_screen_windows = []
    for item in _SCREEN_WINDOWS_RAW.split(";"):
        parts = tuple(value.strip() for value in item.split(","))
        if len(parts) != 3 or any(not value for value in parts):
            raise ValueError(
                "ROUTE_A_SCREEN_WINDOWS entries must be id,start_iso,end_iso separated by semicolons"
            )
        window_id, start_iso, end_iso = parts
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        if (start.tzinfo is None or end.tzinfo is None or start >= end):
            raise ValueError("Route A screen windows require ordered timezone-aware boundaries")
        parsed_screen_windows.append((window_id, start.isoformat(), end.isoformat()))
    SCREEN_WINDOWS = tuple(parsed_screen_windows)
if (not SCREEN_WINDOWS
        or len({window_id for window_id, _, _ in SCREEN_WINDOWS}) != len(SCREEN_WINDOWS)):
    raise ValueError("Route A screen windows must have unique nonempty identities")
_ORDERED_SCREEN_INTERVALS = sorted(
    (datetime.fromisoformat(start), datetime.fromisoformat(end))
    for _, start, end in SCREEN_WINDOWS
)
if any(left_end > right_start for (_, left_end), (right_start, _) in zip(
        _ORDERED_SCREEN_INTERVALS, _ORDERED_SCREEN_INTERVALS[1:])):
    raise ValueError("Route A screen windows must not overlap")
SEEDS = (11, 29)
EPISODE_STEPS = 1_400
FIT_EPISODE_STEPS = 350
FEATURES = (
    "market.trend_1h",
    "market.trend_1d",
    "market.vol_1m",
    "market.zscore_1m",
    "market.efficiency_15m",
    "market.efficiency_1h",
    "account.net_exposure",
    "account.drawdown",
    "account.unrealized_pnl_fraction",
) + (
    (
        "market.taker_buy_imbalance_5m",
        "market.taker_buy_imbalance_15m",
        "market.log_trade_count_5m",
    )
    if runner.FEATURE_PROFILE == "causal_mtf_v6"
    else ()
)


def fit_windows() -> tuple[tuple[str, str, str], ...]:
    windows = []
    for day in FIT_DAYS:
        midnight = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        for slot in range(4):
            start = midnight + timedelta(hours=slot * 6)
            end = start + timedelta(hours=6)
            windows.append(
                (f"{day.replace('-', '')}-h{slot * 6:02d}", start.isoformat(), end.isoformat())
            )
    return tuple(windows)


def make_teacher() -> object:
    primary = TrendTeacher(
        size=0.8,
        threshold=0.004,
        hold_on_neutral=True,
        direction=-1,
        expert_id="countertrend_hold_neutral_v1",
    )
    if TEACHER_MODE == "countertrend_v1":
        return primary
    if TEACHER_MODE == "stop_cooldown_5m_v1":
        return StopCooldownTeacher(
            primary=primary,
            stop_loss_fraction=0.001,
            cooldown_ns=5 * 60_000_000_000,
            expert_id="countertrend_stop_cooldown_5m_v1",
        )
    raise ValueError(f"unsupported teacher mode: {TEACHER_MODE}")


def intent_diagnostics(model: BCGRUPolicy, dataset: EpisodeSequenceDataset,
                       batch_size: int) -> dict:
    model.eval()
    actual = []
    predicted = []
    predicted_confidence = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits, _ = model(batch["x"], batch["length"])
            probabilities = torch.softmax(logits, dim=-1)
            actual.extend(int(value) for value in batch["intent"].tolist())
            predicted.extend(int(value) for value in logits.argmax(dim=-1).tolist())
            predicted_confidence.extend(
                float(value) for value in probabilities.max(dim=-1).values.tolist()
            )
    actual_names = [INTENTS[index] for index in actual]
    predicted_names = [INTENTS[index] for index in predicted]
    actual_open = {index for index, name in enumerate(INTENTS) if name.startswith("OPEN_")}
    true_open = sum(index in actual_open for index in actual)
    correct_open = sum(
        expected in actual_open and observed in actual_open
        for expected, observed in zip(actual, predicted, strict=True)
    )
    open_confidence = [
        confidence for index, confidence in zip(predicted, predicted_confidence, strict=True)
        if index in actual_open
    ]
    return {
        "actual_intent_counts": dict(sorted(Counter(actual_names).items())),
        "predicted_intent_counts": dict(sorted(Counter(predicted_names).items())),
        "accuracy": sum(a == p for a, p in zip(actual, predicted, strict=True)) / len(actual),
        "open_label_count": true_open,
        "open_recall": correct_open / true_open if true_open else None,
        "predicted_open_count": len(open_confidence),
        "predicted_open_confidence_mean": (
            float(np.mean(open_confidence)) if open_confidence else None
        ),
        "predicted_open_confidence_quantiles": (
            {str(q): float(np.quantile(open_confidence, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
            if open_confidence else {}
        ),
        "predicted_open_below_confidence_count": {
            str(threshold): sum(value < threshold for value in open_confidence)
            for threshold in (0.5, 0.6, 0.7, 0.8, 0.9)
        },
    }


def calibrate_open_intent_gate(
        model: BCGRUPolicy, dataset: EpisodeSequenceDataset, batch_size: int,
        *, minimum_recall: float = 0.4, minimum_accepted: int = 5,
) -> dict:
    """Fit scalar temperature and select a conservative OPEN gate on later Train.

    The calibration data must be chronologically later than every training and
    DAgger episode. This report is still Train-inner development evidence.
    """
    if not 0 < minimum_recall <= 1 or minimum_accepted < 1:
        raise ValueError("invalid OPEN calibration constraints")
    model.eval()
    logits_rows, labels = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits, _ = model(batch["x"], batch["length"])
            logits_rows.extend(logits.cpu().numpy())
            labels.extend(int(value) for value in batch["intent"].tolist())
    logits_np = np.asarray(logits_rows, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    open_indices = np.asarray(
        [i for i, name in enumerate(INTENTS) if name.startswith("OPEN_")], dtype=np.int64
    )
    if (logits_np.ndim != 2 or logits_np.shape[0] != len(labels_np)
            or logits_np.shape[1] != len(INTENTS) or len(labels_np) == 0):
        raise ValueError("invalid held-out Train logits for OPEN calibration")
    actual_open = np.isin(labels_np, open_indices)
    teacher_open_count = int(np.sum(actual_open))
    if teacher_open_count < minimum_accepted:
        return {
            "method": "later_train_temperature_and_f0_5_open_gate_v1",
            "status": "insufficient_teacher_open_labels",
            "sample_count": int(len(labels_np)),
            "teacher_open_count": teacher_open_count,
            "predicted_open_count_before_gate": int(np.sum(
                np.isin(np.argmax(logits_np, axis=1), open_indices)
            )),
            "minimum_recall": minimum_recall,
            "minimum_accepted": minimum_accepted,
            "temperature": 1.0,
            "selected_threshold": 0.0,
            "selected_validation_metrics": None,
            "candidate_count": 0,
            "calibration_scope": (
                "later Train windows; insufficient teacher OPEN labels, so no "
                "temperature or OPEN gate was fitted and defaults were retained"
            ),
        }

    temperatures = np.exp(np.linspace(np.log(0.25), np.log(4.0), 101))
    nll_by_temperature = []
    for temperature in temperatures:
        scaled = logits_np / temperature
        row_max = np.max(scaled, axis=1, keepdims=True)
        log_normalizer = row_max[:, 0] + np.log(
            np.exp(scaled - row_max).sum(axis=1)
        )
        nll_by_temperature.append(float(np.mean(
            log_normalizer - scaled[np.arange(len(labels_np)), labels_np]
        )))
    selected_temperature = float(temperatures[int(np.argmin(nll_by_temperature))])
    scaled = logits_np / selected_temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = np.argmax(logits_np, axis=1)
    predicted_is_open = np.isin(predicted, open_indices)
    confidence = probabilities[np.arange(len(labels_np)), predicted]
    thresholds = np.unique(np.concatenate(([0.0], confidence[predicted_is_open], [1.0])))
    candidates = []
    for threshold in thresholds:
        accepted = predicted_is_open & (confidence >= threshold)
        accepted_count = int(np.sum(accepted))
        true_positive = int(np.sum(accepted & actual_open))
        precision = true_positive / accepted_count if accepted_count else 0.0
        recall = true_positive / int(np.sum(actual_open))
        beta_squared = 0.25
        f05 = (
            (1 + beta_squared) * precision * recall
            / (beta_squared * precision + recall)
            if precision + recall > 0 else 0.0
        )
        eligible = recall >= minimum_recall and accepted_count >= minimum_accepted
        candidates.append({
            "threshold": float(threshold),
            "accepted": accepted_count,
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "f0_5": f05,
            "eligible": eligible,
        })
    eligible = [row for row in candidates if row["eligible"]]
    if eligible:
        chosen = max(eligible, key=lambda row: (row["f0_5"], row["precision"], row["threshold"]))
        selected_threshold = chosen["threshold"]
    else:
        chosen = None
        selected_threshold = 1.0
    return {
        "method": "later_train_temperature_and_f0_5_open_gate_v1",
        "status": "calibrated",
        "sample_count": int(len(labels_np)),
        "teacher_open_count": int(np.sum(actual_open)),
        "predicted_open_count_before_gate": int(np.sum(predicted_is_open)),
        "minimum_recall": minimum_recall,
        "minimum_accepted": minimum_accepted,
        "temperature_grid": [float(temperatures[0]), float(temperatures[-1]), len(temperatures)],
        "temperature": selected_temperature,
        "nll_before": nll_by_temperature[int(np.argmin(
            np.abs(temperatures - 1.0)
        ))],
        "nll_after": min(nll_by_temperature),
        "open_confidence_quantiles": {
            str(q): float(np.quantile(confidence[predicted_is_open], q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        } if np.any(predicted_is_open) else {},
        "selected_threshold": selected_threshold,
        "selected_validation_metrics": chosen,
        "candidate_count": len(candidates),
        "calibration_scope": "later Train windows; same data selects temperature and OPEN threshold; not independent evaluation",
    }


class _UnusedProjection:
    """Guard the legacy path; native DAgger rollouts precompute projection."""

    def project_native_target(self, _observation, _requested_exposure):
        raise RuntimeError("DAgger sample omitted its pre-action native projection")


def rollout_actor_identity(model: BCGRUPolicy) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value).tobytes(order="C"))
    return f"route_a_bc:{digest.hexdigest()}"


def main() -> int:
    training_runtime = configure_torch_threads(torch)
    run_wall_started = perf_counter()
    run_cpu_started = process_time()
    if OUT.exists():
        raise FileExistsError(f"immutable multi-episode output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    runner.WINDOWS = SCREEN_WINDOWS
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    source_identity = marker_sha
    teacher = make_teacher()
    baseline = WaitTeacher(expert_id="wait_flat_baseline_v1")
    teacher_hash = runner.teacher_config_hash(teacher)
    baseline_hash = runner.teacher_config_hash(baseline)
    owner_token = ACTIVE_OWNER.set("route-a-multiepisode-screen-20261020")
    try:
        all_training_windows = fit_windows()
        all_dates = {window_id[:8] for window_id, _, _ in all_training_windows}
        if any(date not in all_dates for date in INNER_VALIDATION_DATES):
            raise ValueError("inner validation dates must be selected from the frozen Train fit dates")
        validation_windows = tuple(
            row for row in all_training_windows if row[0][:8] in INNER_VALIDATION_DATES
        )
        training_windows = tuple(
            row for row in all_training_windows if row[0][:8] not in INNER_VALIDATION_DATES
        )
        if INNER_VALIDATION_DATES and (
            not validation_windows or not training_windows
            or max(row[0][:8] for row in training_windows)
            >= min(row[0][:8] for row in validation_windows)
        ):
            raise ValueError("inner validation must be a nonempty strictly later Train period")
        scaler_start = runner.ns(min(row[1] for row in training_windows))
        scaler_end = runner.ns(max(row[2] for row in training_windows))
        _, scaler_factory = runner.make_config(
            label="multi-fit-scaler-feb01-mar23",
            marker_sha=marker_sha,
            scope_start=scope_start,
            scope_end=scope_end,
            start_ns=scaler_start,
            end_ns=scaler_end,
            mode="base",
            fit_scaler=True,
        )
        scaler = scaler_factory.scaler
        execution_recipe_sha = scaler_factory.route_a_execution_recipe_hash
        del scaler_factory
        gc.collect()

        # Evaluate the teacher before paying for BC/DAgger training. A student
        # cannot be promoted when the frozen teacher gate already rejects.
        teacher_pairs = []
        pair_metrics = []
        for window_id, start, end in SCREEN_WINDOWS:
            _, factory = runner.make_config(
                label=f"multi-screen-{window_id}-base",
                marker_sha=marker_sha,
                scope_start=scope_start,
                scope_end=scope_end,
                start_ns=runner.ns(start),
                end_ns=runner.ns(end),
                mode="base",
                fit_scaler=False,
                scaler=scaler,
            )
            for scenario in ("base", "stress"):
                scenario_factory = factory
                if scenario != "base":
                    _, scenario_factory = runner.make_config(
                        label=f"multi-screen-{window_id}-{scenario}",
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
                    key = f"{window_id}-{scenario}-seed-{seed}"
                    teacher_run = runner.run_episode(
                        scenario_factory, make_teacher(), source_identity,
                        key, seed, EPISODE_STEPS,
                    )
                    baseline_run = runner.run_episode(
                        scenario_factory, baseline, source_identity,
                        key + "-baseline", seed, EPISODE_STEPS,
                    )
                    evidence_hashes = {}
                    for role, run in (("teacher", teacher_run), ("baseline", baseline_run)):
                        evidence_hashes[role] = runner.write_verified(
                            OUT / "native-evidence" / f"{key}-{role}.json", run["evidence"],
                        )
                    matched = (
                        teacher_run["barriers"] == baseline_run["barriers"] == EPISODE_STEPS
                    )
                    pair_metrics.append({
                        "window_id": window_id,
                        "scenario": scenario,
                        "seed": seed,
                        "teacher_net_return": teacher_run["net_return"],
                        "baseline_net_return": baseline_run["net_return"],
                        "student_net_return": None,
                        "teacher_max_drawdown": teacher_run["max_drawdown"],
                        "student_max_drawdown": None,
                        "teacher_filled_orders": teacher_run["filled_orders"],
                        "student_filled_orders": None,
                        "student_liquidations": None,
                        "matched_native_barriers": matched,
                        "evidence_sha256": evidence_hashes,
                    })
                    teacher_pairs.append(NativePair(
                        window_id=window_id,
                        scenario=scenario,
                        seed=seed,
                        teacher_recipe=teacher_hash,
                        baseline_recipe=baseline_hash,
                        source_identity=source_identity,
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
                if scenario_factory is not factory:
                    del scenario_factory
            del factory
        teacher_screen = screen_teacher(teacher_pairs)
        if teacher_screen.status != "TRAIN_INNER_SCREEN_PASS":
            preflight_wall_seconds = perf_counter() - run_wall_started
            preflight_cpu_seconds = process_time() - run_cpu_started
            logical_cpu_count = training_runtime["logical_cpu_count"]
            preflight_report = {
                "schema": "rles.route_a.multiepisode_student_screen.v1",
                "purpose": (
                    "Teacher preflight rejected; student BC/DAgger fit was skipped. "
                    "All windows are Train-only scenario development evidence."
                ),
                "source_marker_sha256": marker_sha,
                "fit_decisions": 0,
                "dagger_training": None,
                "screen_windows": [
                    {"id": window, "start_ns": runner.ns(start), "end_ns": runner.ns(end)}
                    for window, start, end in SCREEN_WINDOWS
                ],
                "scenarios": ["base", "stress"],
                "seeds": list(SEEDS),
                "pair_metrics": pair_metrics,
                "teacher_screen": asdict(teacher_screen),
                "training_runtime": {
                    **training_runtime,
                    "device": "cpu",
                    "wall_seconds": preflight_wall_seconds,
                    "process_cpu_seconds": preflight_cpu_seconds,
                    "average_logical_cpu_utilization_percent": (
                        100.0 * preflight_cpu_seconds / preflight_wall_seconds
                        / logical_cpu_count if preflight_wall_seconds > 0 else None
                    ),
                    "student_fit_skipped": True,
                },
                "student_screen": None,
                "student_gate_status": "NOT_ELIGIBLE_TEACHER_GATE_REJECTED",
                "wfo_actor_exported": False,
                "economic_status": "SCENARIO_ONLY",
            }
            runner.write_verified(OUT / "screen-report.json", preflight_report)
            print(json.dumps({
                "output": str(OUT),
                "teacher_status": teacher_screen.status,
                "teacher_reasons": teacher_screen.reasons,
                "student_training_skipped": True,
                "economic_status": "SCENARIO_ONLY",
            }, indent=2), flush=True)
            return 0

        fit_steps_by_window = {}
        fit_rows = []
        for index, (window_id, start, end) in enumerate(all_training_windows):
            _, factory = runner.make_config(
                label=f"multi-fit-{window_id}-base",
                marker_sha=marker_sha,
                scope_start=scope_start,
                scope_end=scope_end,
                start_ns=runner.ns(start),
                end_ns=runner.ns(end),
                mode="base",
                fit_scaler=False,
                scaler=scaler,
            )
            provider, grant = runner.make_provider(
                factory, make_teacher(), source_identity, f"multi-fit-{window_id}"
            )
            barriers, rollout_report = provider.rollout_with_m22_trace(
                factory=factory,
                host_rollout=runner.application.rollout,
                grant=grant,
                seed=11,
                rollout_id=f"multi-fit-{window_id}",
                episode_steps=FIT_EPISODE_STEPS,
                reset_options={"rles_episode_index": 11 + index},
            )
            episode_steps = teacher_native_rollout(
                lambda episode_barriers=barriers, **_kwargs: iter(episode_barriers),
                fold_id=grant.fold_id,
                grant=grant,
            )
            if len(episode_steps) != FIT_EPISODE_STEPS:
                raise RuntimeError(f"fit episode {window_id} lost registered Train decisions")
            fit_steps_by_window[window_id] = tuple(episode_steps)
            metrics = runner.metric_row(barriers, dict(rollout_report), expected_source=source_identity)
            evidence_sha = runner.write_verified(
                OUT / "native-evidence" / f"fit-{window_id}.json", metrics["evidence"]
            )
            fit_rows.append(
                {
                    "window_id": window_id,
                    "role": (
                        "inner_validation" if window_id[:8] in INNER_VALIDATION_DATES
                        else "fit"
                    ),
                    "start_ns": runner.ns(start),
                    "end_ns": runner.ns(end),
                    "decisions": len(episode_steps),
                    "teacher_net_return": metrics["net_return"],
                    "teacher_max_drawdown": metrics["max_drawdown"],
                    "teacher_filled_orders": metrics["filled_orders"],
                    "native_evidence_sha256": evidence_sha,
                }
            )
            print(f"collected {window_id}: {len(episode_steps)} labels", flush=True)
            del factory, provider, grant, barriers, rollout_report

        config = BCConfig(
            feature_names=FEATURES,
            context_bars=120,
            max_gap_ns=61_000_000_000,
            hidden_size=32,
            layers=1,
            batch_size=128,
            learning_rate=0.001,
            epochs=TRAIN_EPOCHS,
            seed=31,
            intent_weight=1.0,
            inference_mode="intent_aware_v1",
            exposure_target_mode="teacher_request_v2",
            stop_loss_fraction=STOP_LOSS_FRACTION,
            stop_cooldown_ns=STOP_COOLDOWN_NS,
            minimum_position_hold_ns=MINIMUM_POSITION_HOLD_NS,
            open_intent_min_confidence=(
                0.0 if CALIBRATE_OPEN_CONFIDENCE else OPEN_INTENT_MIN_CONFIDENCE
            ),
            intent_class_weighting=INTENT_CLASS_WEIGHTING,
            intent_sampling=INTENT_SAMPLING,
        )
        dataset = EpisodeSequenceDataset(
            tuple(step for row in training_windows for step in fit_steps_by_window[row[0]]),
            feature_names=config.feature_names,
            context_bars=config.context_bars,
            max_gap_ns=config.max_gap_ns,
        )
        validation_steps = tuple(
            step for row in validation_windows for step in fit_steps_by_window[row[0]]
        )
        inner_validation_dataset = (
            EpisodeSequenceDataset(
                validation_steps,
                feature_names=config.feature_names,
                context_bars=config.context_bars,
                max_gap_ns=config.max_gap_ns,
            ) if validation_steps else None
        )
        if dataset.source_identity != source_identity or dataset.fold_id != runner.FOLD_ID:
            raise PermissionError("multi-episode sequence set lost registered Train identity")
        if inner_validation_dataset is not None and (
            inner_validation_dataset.source_identity != source_identity
            or inner_validation_dataset.fold_id != runner.FOLD_ID
        ):
            raise PermissionError("inner validation dataset lost registered Train identity")
        fit_steps = dataset.steps
        counts = Counter(
            row.advice.intent
            for row in fit_steps
            if row.intent_supervision_weight > 0
        )
        print(f"training on {len(dataset)} decisions across {len(training_windows)} resets", flush=True)
        model = BCGRUPolicy(config.feature_names, config.hidden_size, config.layers)
        history = train_bc(
            model, dataset, config, device="cpu",
            inner_validation=(None if CALIBRATE_OPEN_CONFIDENCE else inner_validation_dataset),
        )
        policy = BCInferencePolicy(model, config, device="cpu")
        class_diagnostics = intent_diagnostics(model, dataset, config.batch_size)
        behavior = _native_behavior_summary(
            model,
            dataset,
            batch_size=config.batch_size,
            device="cpu",
            inference_mode=config.inference_mode,
        )
        dagger_training = None
        if DAGGER_ROLLOUTS:
            if type(DAGGER_SEED) is not int or not 0 <= DAGGER_SEED < 2**32:
                raise ValueError("DAgger rollout seed must be a frozen non-negative integer")
            if not 0 < DAGGER_EXPERT_FRACTION < 1:
                raise ValueError("DAgger expert fraction must be between zero and one")
            aggregate = DAggerSet(dataset.steps)
            authorized_grants = set()
            collector = DAggerCollector(
                teacher=teacher,
                projector=_UnusedProjection(),
                train_grant_verifier=lambda candidate: candidate in authorized_grants,
            )
            actor_identity = rollout_actor_identity(model)
            dagger_rollout_rows = []
            for index, (window_id, start, end) in enumerate(training_windows):
                _, dagger_factory = runner.make_config(
                    label=f"dagger-fit-{window_id}-base",
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=runner.ns(start),
                    end_ns=runner.ns(end),
                    mode="base",
                    fit_scaler=False,
                    scaler=scaler,
                )
                dagger_teacher_provider, dagger_grant = runner.make_provider(
                    dagger_factory,
                    make_teacher(),
                    source_identity,
                    f"dagger-fit-{window_id}",
                )
                authorized_grants.add(dagger_grant)
                dagger_provider = NativeGymDaggerTrainProvider(
                    teacher_provider=dagger_teacher_provider,
                    factory=dagger_factory,
                    host_rollout=runner.application.rollout,
                    seed=DAGGER_SEED,
                    decision_policy=policy,
                    rollout_options={
                        "episode_steps": FIT_EPISODE_STEPS,
                        "reset_options": {"rles_episode_index": 10_000 + index},
                    },
                )
                collected = collector.collect(
                    provider=dagger_provider,
                    grant=dagger_grant,
                    student_policy_id=actor_identity,
                    iteration=1,
                    aggregate=aggregate,
                )
                if not isinstance(dagger_provider.last_report, dict):
                    raise RuntimeError("DAgger provider did not retain its verified M2.2 report")
                native_report_sha = runner.write_verified(
                    OUT / "dagger-evidence" / f"{window_id}-m22-report.json",
                    dagger_provider.last_report,
                )
                if (not isinstance(dagger_provider.last_visits, tuple)
                        or len(dagger_provider.last_visits) != collected):
                    raise RuntimeError("DAgger M2.2 report and retained visit rows differ")
                visit_rows = []
                for visit in dagger_provider.last_visits:
                    observation = visit.observation
                    visit_rows.append({
                        "observation": {
                            "episode_id": observation.episode_id,
                            "decision_ts_ns": observation.decision_ts_ns,
                            "market": dict(observation.market),
                            "feature_available_at_ns": dict(observation.feature_available_at_ns),
                            "account": dict(observation.account),
                            "pending_order_ids": list(observation.pending_order_ids),
                            "fold_id": observation.fold_id,
                        },
                        "teacher_label": asdict(visit.teacher_label),
                        "projected_teacher_target": asdict(visit.projected_teacher_target),
                        "native_action_trace": asdict(visit.native_action_trace),
                    })
                visit_sha = runner.write_verified(
                    OUT / "dagger-evidence" / f"{window_id}-student-visits.json",
                    {
                        "schema": "rles.route_a.dagger.visits.v1",
                        "split": "train",
                        "fold_id": dagger_grant.fold_id,
                        "source_identity": dagger_grant.source_identity,
                        "student_policy_id": actor_identity,
                        "execution_recipe": dagger_provider.projection_recipe,
                        "visits": visit_rows,
                    },
                )
                dagger_rollout_rows.append({
                    "window_id": window_id,
                    "decisions": collected,
                    "grant_reservation_id": dagger_grant.reservation_id,
                    "m22_report_sha256": native_report_sha,
                    "student_visit_evidence_sha256": visit_sha,
                })
                print(f"collected DAgger {window_id}: {collected} states", flush=True)
                del dagger_factory, dagger_teacher_provider, dagger_grant, dagger_provider
                gc.collect()

            fine_tune = fine_tune_bc_on_daggerset(
                policy=policy,
                aggregate=aggregate,
                grant=next(iter(authorized_grants)),
                train_grant_verifier=lambda candidate: candidate in authorized_grants,
                device="cpu",
                patience=8,
                expert_fraction=DAGGER_EXPERT_FRACTION,
            )
            policy = fine_tune.policy
            model = policy.model
            dataset = EpisodeSequenceDataset(
                aggregate.training_steps(),
                feature_names=config.feature_names,
                context_bars=config.context_bars,
                max_gap_ns=config.max_gap_ns,
            )
            behavior = _native_behavior_summary(
                model,
                dataset,
                batch_size=config.batch_size,
                device="cpu",
                inference_mode=config.inference_mode,
            )
            class_diagnostics = intent_diagnostics(model, dataset, config.batch_size)
            dagger_training = {
                "iteration": 1,
                "student_policy_id": actor_identity,
                "dagger_samples": aggregate.collected,
                "original_teacher_samples": len(aggregate.original),
                "training_samples_after_aggregation": len(dataset),
                "rollouts": dagger_rollout_rows,
                "requested_expert_fraction": fine_tune.requested_expert_fraction,
                "sampled_expert_fraction": fine_tune.sampled_expert_fraction,
                "samples_per_epoch": fine_tune.samples_per_epoch,
                "fine_tune_epochs": [asdict(row) for row in fine_tune.history],
                "fine_tuned_actor_sha256": bc_actor_sha256(model),
            }
            print(
                f"DAgger fine-tuned with {aggregate.collected} verified student states",
                flush=True,
            )

        inference_config = config
        open_intent_calibration = None
        if CALIBRATE_OPEN_CONFIDENCE:
            if inner_validation_dataset is None:
                raise ValueError("OPEN confidence calibration requires later Train windows")
            open_intent_calibration = calibrate_open_intent_gate(
                model, inner_validation_dataset, config.batch_size
            )
            inference_config = replace(
                config,
                intent_temperature=open_intent_calibration["temperature"],
                open_intent_min_confidence=open_intent_calibration["selected_threshold"],
            )
            policy = BCInferencePolicy(model, inference_config, device="cpu")

        student_screen = None
        if teacher_screen.status == "TRAIN_INNER_SCREEN_PASS":
            actor_sha = bc_actor_sha256(model)
            student_recipe_sha = bc_student_recipe_sha256(inference_config)
            observation_schema_sha = bc_observation_schema_sha256(inference_config)
            action_schema_sha = bc_action_schema_sha256()
            student_pairs = []
            for window_id, start, end in SCREEN_WINDOWS:
                for scenario in ("base", "stress"):
                    _, factory = runner.make_config(
                        label=f"multi-screen-{window_id}-{scenario}",
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
                        key = f"{window_id}-{scenario}-seed-{seed}"
                        student_run = runner.run_episode(
                            factory, make_teacher(), source_identity,
                            key + "-multi-episode-student", seed, EPISODE_STEPS,
                            policy=policy,
                        )
                        row = next(item for item in pair_metrics if (
                            item["window_id"], item["scenario"], item["seed"]
                        ) == (window_id, scenario, seed))
                        row["student_net_return"] = student_run["net_return"]
                        row["student_max_drawdown"] = student_run["max_drawdown"]
                        row["student_filled_orders"] = student_run["filled_orders"]
                        row["student_liquidations"] = student_run["liquidations"]
                        row["matched_native_barriers"] = (
                            row["matched_native_barriers"]
                            and student_run["barriers"] == EPISODE_STEPS
                        )
                        row["evidence_sha256"]["student"] = runner.write_verified(
                            OUT / "native-evidence" / f"{key}-student.json",
                            student_run["evidence"],
                        )
                    del factory
            for row in pair_metrics:
                teacher_record = next(
                    item for item in teacher_pairs
                    if (item.window_id, item.scenario, item.seed)
                    == (row["window_id"], row["scenario"], row["seed"])
                )
                student_pairs.append(
                    StudentPair(
                        window_id=row["window_id"],
                        scenario=row["scenario"],
                        seed=row["seed"],
                        teacher_recipe=teacher_hash,
                        student_recipe=student_recipe_sha,
                        source_identity=source_identity,
                        teacher_net_return=teacher_record.teacher_net_return,
                        student_net_return=row["student_net_return"],
                        teacher_max_drawdown=teacher_record.teacher_max_drawdown,
                        student_max_drawdown=row["student_max_drawdown"],
                        student_filled_orders=row["student_filled_orders"],
                        student_liquidations=row["student_liquidations"],
                        matched_native_barriers=row["matched_native_barriers"],
                        teacher_gate_sha256=teacher_screen.receipt_sha256,
                        actor_sha256=actor_sha,
                        observation_schema_sha256=observation_schema_sha,
                        action_schema_sha256=action_schema_sha,
                        execution_recipe_sha256=execution_recipe_sha,
                        teacher_evidence_sha256=row["evidence_sha256"]["teacher"],
                        student_evidence_sha256=row["evidence_sha256"]["student"],
                        scenario_only=True,
                        nonoverlap_verified=True,
                    )
                )
            student_screen = screen_student(
                student_pairs, teacher_screen=teacher_screen
            )
            if not verify_student_screen(student_screen):
                raise RuntimeError("student screen receipt failed recomputation")
        runtime_wall_seconds = perf_counter() - run_wall_started
        runtime_cpu_seconds = process_time() - run_cpu_started
        logical_cpu_count = os.cpu_count() or 1
        report = {
            "schema": "rles.route_a.multiepisode_student_screen.v1",
            "purpose": (
                "Train-only development screen testing repeated flat-reset teacher episodes; "
                "all listed windows are development data, not independent certification."
            ),
            "source_marker_sha256": marker_sha,
            "fit_windows": fit_rows,
            "training_windows": [row[0] for row in training_windows],
            "inner_validation_windows": [row[0] for row in validation_windows],
            "inner_validation_decisions": len(validation_steps),
            "fit_scaler_window_ns": [scaler_start, scaler_end],
            "fit_decisions": len(dataset),
            "fit_executable_intent_counts": dict(sorted(counts.items())),
            "student_training_recipe": asdict(config),
            "student_training_recipe_sha256": runner.bc_student_recipe_sha256(config),
            "student_inference_recipe": asdict(inference_config),
            "student_inference_recipe_sha256": runner.bc_student_recipe_sha256(inference_config),
            "open_intent_calibration": open_intent_calibration,
            "training_behavior": asdict(behavior),
            "training_intent_diagnostics": class_diagnostics,
            "training_epochs": [asdict(row) for row in history],
            "training_runtime": {
                **training_runtime,
                "device": "cpu",
                "wall_seconds": runtime_wall_seconds,
                "process_cpu_seconds": runtime_cpu_seconds,
                "logical_cpu_count": logical_cpu_count,
                "average_logical_cpu_utilization_percent": (
                    100.0 * runtime_cpu_seconds / runtime_wall_seconds / logical_cpu_count
                    if runtime_wall_seconds > 0 else None
                ),
                "student_fit_skipped": False,
            },
            "dagger_training": dagger_training,
            "screen_windows": [
                {"id": w, "start_ns": runner.ns(start), "end_ns": runner.ns(end)}
                for w, start, end in SCREEN_WINDOWS
            ],
            "scenarios": ["base", "stress"],
            "seeds": list(SEEDS),
            "pair_metrics": pair_metrics,
            "teacher_screen": asdict(teacher_screen),
            "teacher_policy_mode": TEACHER_MODE,
            "student_screen": asdict(student_screen) if student_screen is not None else None,
            "student_gate_status": (
                student_screen.status if student_screen is not None
                else "NOT_ELIGIBLE_TEACHER_GATE_REJECTED"
            ),
            "wfo_actor_exported": False,
            "economic_status": "SCENARIO_ONLY",
        }
        runner.write_verified(OUT / "screen-report.json", report)
        print(
            json.dumps(
                {
                    "output": str(OUT),
                    "fit_decisions": len(dataset),
                    "fit_intents": dict(sorted(counts.items())),
                    "teacher_status": teacher_screen.status,
                    "teacher_reasons": teacher_screen.reasons,
                    "student_gate_status": report["student_gate_status"],
                    "wfo_actor_exported": False,
                    "economic_status": "SCENARIO_ONLY",
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
