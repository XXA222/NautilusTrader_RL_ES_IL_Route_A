"""Package a Train-only Route A BC fit for the host's WFO artifact contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from rles_route_a.export.bundle import load_bc_bundle, save_bc_bundle
from rles_route_a.evaluation.promotion_identity import (
    bc_action_schema_sha256,
    bc_actor_sha256,
    bc_observation_schema_sha256,
    bc_student_recipe_sha256,
)
from rles_route_a.evaluation.teacher_gate import verify_student_screen
from rles_route_a.training.native_bc import NativeBCTrainingResult
from .execution_contract import require_route_a_flat_close


def make_route_a_wfo_train(
    *,
    fit_candidate: Callable[[dict], tuple[NativeBCTrainingResult, Any]],
    validation_backtest: Callable[..., dict],
    write_verified: Callable[[Path, Any], str],
    digest: Callable[[Any], str],
    save_actor: Callable[..., Path] = save_bc_bundle,
) -> Callable[..., dict]:
    """Build the Route A WFO ``train_fn`` for Validation and final retrain.

    ``fit_candidate`` must build its Train factory through the host's registered
    reader and return the native BC result plus that frozen factory. Validation
    is evaluated only after the bundle is written and the WFO ValidationAccess
    is consumed by ``validation_backtest``. ``evaluate=False`` performs the
    final pre-Holdout retrain and writes no evaluation report.
    """
    if not all(
        callable(value)
        for value in (fit_candidate, validation_backtest, write_verified, digest, save_actor)
    ):
        raise TypeError(
            "Route A WFO training, validation, bundle, and digest callbacks are required"
        )

    def train(run: dict, *, evaluate: bool = True, validation_access: Any = None) -> dict:
        if type(evaluate) is not bool:
            raise TypeError("WFO evaluate flag must be Boolean")
        if evaluate and validation_access is None:
            raise PermissionError("WFO ValidationAccess is required for candidate evaluation")
        if not evaluate and validation_access is not None:
            raise ValueError("final retrain cannot carry a Validation capability")
        output = Path(run["output"]).resolve()
        if output.exists() or output.is_symlink():
            raise FileExistsError("WFO candidate output must be a fresh immutable directory")
        if (
            not isinstance(run.get("environment"), dict)
            or type(run["environment"].get("start_ns")) is not int
            or type(run["environment"].get("end_ns")) is not int
            or run["environment"]["start_ns"] >= run["environment"]["end_ns"]
        ):
            raise ValueError("frozen WFO Train environment is required")
        require_route_a_flat_close(
            run["environment"].get("execution"), context="Route A WFO Train recipe"
        )
        output.mkdir(parents=True, exist_ok=False)

        result, train_factory = fit_candidate(run)
        require_route_a_flat_close(
            getattr(train_factory, "execution", None), context="Route A WFO Train factory"
        )
        if not isinstance(result, NativeBCTrainingResult):
            raise TypeError("Route A WFO fitter must return a verified native BC result")
        trade_case_facts: dict[str, tuple[str, float, int]] = {}
        for label in result.labeled_steps:
            if not label.economic_case_status.startswith("OBSERVED_NATIVE_TRADE_"):
                continue
            if label.case_outcome_bps is None or type(label.case_end_ts_ns) is not int:
                raise PermissionError("Route A labels contain an incomplete native trade-case receipt")
            fact = (
                label.economic_case_status,
                float(label.case_outcome_bps),
                label.case_end_ts_ns,
            )
            prior = trade_case_facts.setdefault(label.case_fact_sha256, fact)
            if prior != fact:
                raise PermissionError("Route A labels contain conflicting native trade-case facts")
        case_counts = {
            status: sum(fact[0] == status for fact in trade_case_facts.values())
            for status in (
                "OBSERVED_NATIVE_TRADE_POSITIVE",
                "OBSERVED_NATIVE_TRADE_NEGATIVE",
                "OBSERVED_NATIVE_TRADE_NEUTRAL",
            )
        }
        inferred_avoidance_count = sum(
            label.label_origin == "INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE"
            for label in result.labeled_steps
        )
        if (
            len(result.labeled_steps) != result.label_count
            or sum(
                label.supervision_weight is not None and label.supervision_weight > 0
                for label in result.labeled_steps
            )
            != result.executable_label_count
            or case_counts["OBSERVED_NATIVE_TRADE_POSITIVE"]
            != result.native_summary.positive_native_trade_case_count
            or case_counts["OBSERVED_NATIVE_TRADE_NEGATIVE"]
            != result.native_summary.negative_native_trade_case_count
            or case_counts["OBSERVED_NATIVE_TRADE_NEUTRAL"]
            != result.native_summary.neutral_native_trade_case_count
            or sum(case_counts.values())
            != result.native_summary.completed_native_trade_case_count
            or inferred_avoidance_count
            != result.native_summary.negative_entry_avoidance_label_count
        ):
            raise PermissionError(
                "Route A WFO case summary does not match audited teacher labels"
            )
        if (
            type(result.executable_label_count) is not int
            or result.executable_label_count < 1
            or type(result.native_summary.positive_native_trade_case_count) is not int
            or result.native_summary.positive_native_trade_case_count < 1
            or result.native_summary.positive_native_trade_case_count
            > result.native_summary.completed_native_trade_case_count
        ):
            raise PermissionError(
                "Route A WFO requires at least one completed positive native teacher case "
                "and executable labels before candidate packaging"
            )
        train_start = run["environment"]["start_ns"]
        train_end = run["environment"]["end_ns"]
        fit_window = result.train_window_ns
        factory_window = (
            getattr(train_factory, "start_ns", None),
            getattr(train_factory, "end_ns", None),
        )
        if (
            fit_window is None
            or len(fit_window) != 2
            or any(type(value) is not int for value in fit_window)
            or not train_start <= fit_window[0] < fit_window[1] <= train_end
            or factory_window != fit_window
            or getattr(train_factory, "transition_hash", None) != result.transition_hash
            or getattr(train_factory, "route_a_execution_recipe_hash", None)
            != result.execution_recipe_hash
        ):
            raise PermissionError(
                "Route A fit result is outside frozen Train or differs from its factory"
            )
        assert fit_window is not None
        if any(
            label.source_identity != result.source_identity
            or label.observation.fold_id != result.native_summary.fold_id
            or not fit_window[0] <= label.observation.decision_ts_ns < fit_window[1]
            or (
                label.economic_case_status.startswith("OBSERVED_NATIVE_TRADE_")
                and (
                    label.case_end_ts_ns is None
                    or label.case_end_ts_ns > fit_window[1]
                )
            )
            or (
                label.economic_case_status == "OPEN_CASE_CENSORED"
                and (
                    label.case_censor_ts_ns is None
                    or label.case_censor_ts_ns > fit_window[1]
                )
            )
            for label in result.labeled_steps
        ):
            raise PermissionError(
                "Route A audited teacher labels escape fitted Train source, Fold, or time window"
            )
        behavior = result.behavior_summary
        selection = result.independent_evaluation
        if selection is None:
            raise ValueError(
                "Route A WFO requires a separately registered and purged Train-inner BC evaluation"
            )
        if (
            selection.fold_id != result.native_summary.fold_id
            or selection.source_identity != result.source_identity
            or selection.execution_recipe_hash != result.execution_recipe_hash
            or selection.feature_hash != result.feature_hash
            or selection.teacher_hash != result.teacher_hash
            or type(selection.purge_ns) is not int
            or selection.purge_ns < 0
            or selection.native_summary.fold_id != selection.fold_id
            or selection.native_summary.decisions != selection.behavior_summary.sample_count
            or selection.window_ns[0] < fit_window[1] + selection.purge_ns
            or selection.window_ns[0] >= selection.window_ns[1]
            or selection.window_ns[1] > train_end
            or selection.window_ns[0] < train_start
        ):
            raise PermissionError(
                "Route A independent evaluation receipt differs from the fitted Train source, "
                "Fold, recipe, or purged time window"
            )
        selection_behavior = selection.behavior_summary
        student_behavior = selection.student_behavior_summary
        if (
            behavior.collapsed_to_flat
            or selection_behavior.collapsed_to_flat
            or student_behavior.collapsed_to_flat
            or selection.student_native_summary.fold_id != selection.fold_id
            or selection.student_native_summary.decisions != student_behavior.sample_count
        ):
            raise ValueError(
                "Route A BC actor collapsed into the configured flat-action deadband on fit, "
                "teacher selection, or native student replay; "
                "refusing to publish the WFO candidate"
            )
        promotion = result.student_screen
        if promotion is None:
            raise PermissionError(
                "Route A WFO requires a passed multi-window student-beats-teacher receipt"
            )
        promotion_identity_ok = (
            verify_student_screen(promotion)
            and promotion.status == "TRAIN_INNER_SCREEN_PASS"
            and promotion.pair_count >= 8
            and promotion.window_count >= 2
            and promotion.source_identity == result.source_identity
            and promotion.teacher_recipe == result.teacher_hash
            and promotion.student_recipe == bc_student_recipe_sha256(result.config)
            and promotion.actor_sha256 == bc_actor_sha256(result.policy.model)
            and promotion.observation_schema_sha256 == bc_observation_schema_sha256(result.config)
            and promotion.action_schema_sha256 == bc_action_schema_sha256()
            and promotion.execution_recipe_sha256 == result.execution_recipe_hash
            and promotion.worst_excess_return > 0
            and promotion.mean_excess_return > 0
            and promotion.receipt_sha256 == promotion.receipt_sha256.lower()
            and len(promotion.receipt_sha256) == 64
            and all(char in "0123456789abcdef" for char in promotion.receipt_sha256)
            and len(promotion.teacher_gate_sha256) == 64
            and all(char in "0123456789abcdef" for char in promotion.teacher_gate_sha256)
        )
        if not promotion_identity_ok:
            raise PermissionError(
                "Route A student promotion receipt failed status or frozen actor/source/schema/recipe checks"
            )
        promotion_payload = {
            "schema": "rles.route_a.wfo_student_promotion.v1",
            "screen": {
                "status": promotion.status,
                "economic_status": promotion.economic_status,
                "receipt_sha256": promotion.receipt_sha256,
                "teacher_gate_sha256": promotion.teacher_gate_sha256,
                "reasons": list(promotion.reasons),
                "thresholds": dict(promotion.thresholds),
                "teacher_thresholds": dict(promotion.teacher_thresholds),
                "pair_count": promotion.pair_count,
                "window_count": promotion.window_count,
                "worst_excess_return": promotion.worst_excess_return,
                "mean_excess_return": promotion.mean_excess_return,
            },
            "teacher_pairs": [asdict(row) for row in promotion.teacher_records],
            "student_pairs": [asdict(row) for row in promotion.paired_records],
            "identity": {
                "source_identity": promotion.source_identity,
                "teacher_recipe": promotion.teacher_recipe,
                "student_recipe": promotion.student_recipe,
                "actor_state_sha256": promotion.actor_sha256,
                "observation_schema_sha256": promotion.observation_schema_sha256,
                "action_schema_sha256": promotion.action_schema_sha256,
                "execution_recipe_sha256": promotion.execution_recipe_sha256,
            },
        }
        promotion_path = output / "student-promotion.json"
        promotion_sha = write_verified(promotion_path, promotion_payload)
        if (
            promotion_path.is_symlink()
            or not promotion_path.is_file()
            or sha256(promotion_path.read_bytes()).hexdigest() != promotion_sha
        ):
            raise ValueError(
                "Route A student promotion writer returned a mismatched artifact digest"
            )
        fit_lookback_ns = getattr(getattr(train_factory, "features", None), "max_lookback_ns", 0)
        if type(fit_lookback_ns) is not int or fit_lookback_ns < 0:
            raise ValueError("Route A fit factory has an invalid feature lookback")
        evidence_specs = (
            (
                "native-fit-teacher.json",
                result.native_barrier_evidence,
                result.native_summary.decisions,
                "fit_teacher",
                result.native_summary,
            ),
            (
                "native-selection-teacher.json",
                selection.teacher_barrier_evidence,
                selection.native_summary.decisions,
                "selection_teacher",
                selection.native_summary,
            ),
            (
                "native-selection-bc-student.json",
                selection.student_barrier_evidence,
                selection.student_native_summary.decisions,
                "selection_bc_student",
                selection.student_native_summary,
            ),
        )
        evidence_artifacts = {}
        for filename, rows, decisions, stage, summary in evidence_specs:
            required = {
                "decision_id",
                "decision_ts_ns",
                "requested_exposure",
                "actual_exposure_after",
                "native_equity_before",
                "native_equity_after",
                "reward_econ",
                "planner_action",
                "planner_reason",
                "fills",
                "fee_currency",
                "scenario_only",
            }
            if (
                len(rows) != decisions
                or not rows
                or any(not isinstance(row, Mapping) or not required <= set(row) for row in rows)
            ):
                raise ValueError(
                    f"Route A {stage} native barrier evidence does not cover every decision"
                )
            if len({row["decision_id"] for row in rows}) != len(rows):
                raise ValueError(f"Route A {stage} evidence has duplicate decision identities")
            payload = {
                "schema": "rles.route-a-native-barrier-evidence.v1",
                "stage": stage,
                "scope": {
                    "fold_id": summary.fold_id,
                    "source_identity": result.source_identity,
                    "transition_hash": (
                        result.transition_hash
                        if stage == "fit_teacher"
                        else selection.transition_hash
                    ),
                    "execution_recipe_hash": result.execution_recipe_hash,
                    "feature_hash": result.feature_hash,
                    "teacher_hash": result.teacher_hash,
                    "window_ns": list(fit_window)
                    if stage == "fit_teacher"
                    else list(selection.window_ns),
                    "scenario_only": summary.economic_status == "SCENARIO_ONLY",
                },
                "summary": {
                    "decisions": summary.decisions,
                    "native_submits": summary.native_submits,
                    "native_fill_order_mentions": summary.native_fills,
                    "blocked_label_count": summary.blocked_label_count,
                    "executable_label_count": summary.executable_label_count,
                    "intent_supervised_count": summary.intent_supervised_count,
                    "completed_native_trade_case_count": (
                        summary.completed_native_trade_case_count
                    ),
                    "positive_native_trade_case_count": (
                        summary.positive_native_trade_case_count
                    ),
                    "negative_native_trade_case_count": (
                        summary.negative_native_trade_case_count
                    ),
                    "neutral_native_trade_case_count": (
                        summary.neutral_native_trade_case_count
                    ),
                    "negative_entry_avoidance_label_count": (
                        summary.negative_entry_avoidance_label_count
                    ),
                    "native_equity_start": summary.native_equity_start,
                    "native_equity_end": summary.native_equity_end,
                    "economic_status": summary.economic_status,
                },
                "decisions": [dict(row) for row in rows],
            }
            # Keep only the projected barrier trace; raw observations and
            # account snapshots are intentionally absent from these artifacts.
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            evidence_path = output / filename
            evidence_sha = write_verified(evidence_path, payload)
            if (
                evidence_path.is_symlink()
                or not evidence_path.is_file()
                or sha256(evidence_path.read_bytes()).hexdigest() != evidence_sha
            ):
                raise ValueError(f"Route A {stage} evidence writer returned a mismatched digest")
            evidence_artifacts[stage] = {
                "path": filename,
                "sha256": evidence_sha,
                "decision_count": decisions,
            }
        metadata_fn = getattr(train_factory, "metadata", None)
        if not callable(metadata_fn):
            raise TypeError("host Train factory must expose frozen preprocessing metadata")
        preprocessing = metadata_fn()
        if (
            not isinstance(preprocessing, dict)
            or not isinstance(preprocessing.get("scaler"), dict)
            or not {"mean", "std", "feature_hash", "train_window_hash", "sample_count"}
            <= set(preprocessing["scaler"])
        ):
            raise ValueError("host Train factory metadata lacks a WFO-compatible frozen scaler")

        actor_dir = output / "route_a_bc"
        save_actor(result.policy.model, result.config, actor_dir)
        actor_manifest = actor_dir / "manifest.json"
        if actor_manifest.is_symlink() or not actor_manifest.is_file():
            raise ValueError("Route A BC actor manifest is missing or unsafe")
        actor_manifest_sha = sha256(actor_manifest.read_bytes()).hexdigest()
        saved_actor, _saved_config = load_bc_bundle(actor_dir, device="cpu")
        if bc_actor_sha256(saved_actor) != promotion.actor_sha256:
            raise PermissionError(
                "serialized Route A actor weights differ from the screened frozen actor"
            )
        bundle = {
            "schema": "rles.historical-bundle.v1",
            "actor": {
                "kind": "route_a_bc",
                "path": "route_a_bc",
                "manifest_sha256": actor_manifest_sha,
                "action_schema": "single_exposure_v1",
            },
            "soft_multipliers": dict(run.get("soft_multipliers", {})),
            "reward_scale": run.get("reward_scale", 1.0),
            "environment": dict(run["environment"]),
            "training_end_ns": run["environment"]["end_ns"],
            "preprocessing": preprocessing,
            "lineage": [
                {
                    "stage": "route_a_bc_gru",
                    "actor_hash": actor_manifest_sha,
                    "usage": {
                        "train_samples": result.label_count,
                        "executable_labels": result.executable_label_count,
                        "epochs": len(result.history),
                        "fit_window_ns": list(fit_window),
                        "research_data_use_window_ns": [
                            max(0, fit_window[0] - fit_lookback_ns),
                            selection.window_ns[1],
                        ],
                        "fit_transition_hash": result.transition_hash,
                        "selection_transition_hash": selection.transition_hash,
                        "execution_recipe_hash": result.execution_recipe_hash,
                        "train_behavior": {
                            "label_nonzero_fraction": behavior.label_nonzero_fraction,
                            "prediction_nonzero_fraction": behavior.prediction_nonzero_fraction,
                            "prediction_mean": behavior.prediction_mean,
                            "prediction_std": behavior.prediction_std,
                            "mae": behavior.mae,
                            "mse": behavior.mse,
                            "deadband": behavior.deadband,
                            "collapsed_to_flat": behavior.collapsed_to_flat,
                        },
                        "selection_window_ns": list(selection.window_ns),
                        "selection_behavior": {
                            "label_nonzero_fraction": selection_behavior.label_nonzero_fraction,
                            "prediction_nonzero_fraction": selection_behavior.prediction_nonzero_fraction,
                            "prediction_mean": selection_behavior.prediction_mean,
                            "prediction_std": selection_behavior.prediction_std,
                            "mae": selection_behavior.mae,
                            "mse": selection_behavior.mse,
                            "deadband": selection_behavior.deadband,
                            "collapsed_to_flat": selection_behavior.collapsed_to_flat,
                        },
                        "student_native_behavior": {
                            "sample_count": student_behavior.sample_count,
                            "executable_label_count": student_behavior.executable_label_count,
                            "prediction_nonzero_fraction": student_behavior.prediction_nonzero_fraction,
                            "prediction_mean": student_behavior.prediction_mean,
                            "prediction_std": student_behavior.prediction_std,
                            "mae": student_behavior.mae,
                            "mse": student_behavior.mse,
                            "deadband": student_behavior.deadband,
                            "collapsed_to_flat": student_behavior.collapsed_to_flat,
                        },
                        "student_promotion": {
                            "status": promotion.status,
                            "economic_status": promotion.economic_status,
                            "receipt_sha256": promotion.receipt_sha256,
                            "teacher_gate_sha256": promotion.teacher_gate_sha256,
                            "actor_state_sha256": promotion.actor_sha256,
                            "observation_schema_sha256": promotion.observation_schema_sha256,
                            "action_schema_sha256": promotion.action_schema_sha256,
                            "execution_recipe_sha256": promotion.execution_recipe_sha256,
                            "pair_count": promotion.pair_count,
                            "window_count": promotion.window_count,
                            "worst_excess_return": promotion.worst_excess_return,
                            "mean_excess_return": promotion.mean_excess_return,
                            "artifact_path": promotion_path.name,
                            "artifact_sha256": promotion_sha,
                        },
                        "native_barrier_evidence": evidence_artifacts,
                    },
                }
            ],
        }
        bundle_path = output / "bundle.json"
        bundle_sha = write_verified(bundle_path, bundle)
        if (
            bundle_path.is_symlink()
            or not bundle_path.is_file()
            or sha256(bundle_path.read_bytes()).hexdigest() != bundle_sha
        ):
            raise ValueError("Route A WFO bundle writer returned a mismatched artifact digest")

        report_sha: str | None = None
        if evaluate:
            identity = getattr(validation_access, "identity", None)
            if (
                not isinstance(identity, dict)
                or identity.get("split") != "validation"
                or identity.get("candidate_id") != run.get("wfo_candidate_id")
                or identity.get("fold_id") != run.get("wfo_fold_id")
                or not isinstance(identity.get("query_id"), str)
                or not identity["query_id"]
            ):
                raise PermissionError(
                    "WFO Validation capability differs from this Route A candidate"
                )
            report = validation_backtest(
                run,
                output,
                validation_access=validation_access,
            )
            report_scope = report.get("evaluation_scope") if isinstance(report, dict) else None
            if (
                not isinstance(report, dict)
                or report.get("execution_mode") != "base"
                or not isinstance(report_scope, dict)
                or report_scope.get("split") != "validation"
                or report_scope.get("mode", "base") != "base"
                or report_scope.get("query_id") != identity["query_id"]
                or report_scope.get("fold_id") != identity["fold_id"]
            ):
                raise ValueError(
                    "Route A Validation callback returned the wrong scope or cost mode"
                )
            if getattr(validation_access, "bundle_sha256", None) != bundle_sha:
                raise PermissionError(
                    "Route A Validation callback did not consume access for the frozen candidate bundle"
                )
            report_sha = write_verified(output / "report.json", report)

        recipe = {
            key: value
            for key, value in run.items()
            if key not in {"output", "resume_checkpoint", "wfo_candidate_id", "wfo_fold_id"}
        }
        run_id = str(uuid4())
        write_verified(
            output / "run.json",
            {
                "schema": "rles.historical-run.v1",
                "run_id": run_id,
                "attempt_id": run_id,
                "recipe_hash": digest(recipe),
            },
        )
        write_verified(
            output / "COMPLETED.json",
            {
                "bundle_sha256": bundle_sha,
                "report_sha256": report_sha,
                "route": run.get("route", "route_a"),
                "algorithm": run.get("algorithm", "route_a_bc_gru"),
            },
        )
        return {
            "output": str(output),
            "route": run.get("route", "route_a"),
            "algorithm": run.get("algorithm", "route_a_bc_gru"),
            "training_samples": result.label_count,
            "executable_labels": result.executable_label_count,
        }

    return train
