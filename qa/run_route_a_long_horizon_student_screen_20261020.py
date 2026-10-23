"""Fit BC on multi-day Train examples and screen it on later Train windows."""

from __future__ import annotations

from dataclasses import asdict, replace
from collections import Counter
import json

import torch

from rles_orchestrator.research_access import ACTIVE_OWNER
from rles_route_a.config import BCConfig
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
from rles_route_a.teachers.trend import TrendTeacher
from rles_route_a.teachers.wait import WaitTeacher
from rles_route_a.training.native_bc import (
    evaluate_bc_on_registered_train_inner_rollout,
    train_bc_from_registered_native_rollout,
)
from rles_route_a.integration.wfo_train import make_route_a_wfo_train

import run_route_a_train_student_screen_20261020 as runner
from route_a_training_runtime import configure_torch_threads


OUT = runner.ROOT / ".artifacts" / "route-a-long-horizon-student-screen-20261020-run22"
FIT = ("2024-02-01T00:00:00+00:00", "2024-02-10T00:00:00+00:00")
WINDOWS = (
    ("april13", "2024-04-13T00:00:00+00:00", "2024-04-14T00:00:00+00:00"),
    ("april14", "2024-04-14T00:00:00+00:00", "2024-04-15T00:00:00+00:00"),
)
FIT_STEPS = 12_000
SCREEN_STEPS = 1_400


def main() -> int:
    training_runtime = configure_torch_threads(torch)
    if OUT.exists():
        raise FileExistsError(f"immutable long-horizon student output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.OUT_BASE = OUT
    runner.FIT = FIT
    runner.WINDOWS = WINDOWS
    marker_sha, scope_start, scope_end = runner.load_source_identity()
    source_identity = marker_sha
    teacher = TrendTeacher(
        size=0.8,
        threshold=0.004,
        hold_on_neutral=True,
        direction=-1,
        expert_id="countertrend_hold_neutral_v1",
    )
    baseline = WaitTeacher(expert_id="wait_flat_baseline_v1")
    teacher_hash = runner.teacher_config_hash(teacher)
    baseline_hash = runner.teacher_config_hash(baseline)
    owner_token = ACTIVE_OWNER.set("route-a-long-student-screen-20261020")
    try:
        fit_config, fit_factory = runner.make_config(
            label="fit-feb01-feb10-base",
            marker_sha=marker_sha,
            scope_start=scope_start,
            scope_end=scope_end,
            start_ns=runner.ns(FIT[0]),
            end_ns=runner.ns(FIT[1]),
            mode="base",
            fit_scaler=True,
        )
        fit_provider, fit_grant = runner.make_provider(
            fit_factory,
            teacher,
            source_identity,
            "fit-feb01-feb10-countertrend",
        )
        feature_names = (
            "market.trend_1h",
            "market.trend_1d",
            "market.vol_1m",
            "account.net_exposure",
            "account.drawdown",
            "account.unrealized_pnl_fraction",
        )
        bc_config = BCConfig(
            feature_names=feature_names,
            context_bars=120,
            max_gap_ns=61_000_000_000,
            hidden_size=32,
            layers=1,
            batch_size=128,
            learning_rate=0.001,
            epochs=20,
            seed=31,
        )
        print(f"fitting {FIT_STEPS} registered Train decisions", flush=True)
        training_result = train_bc_from_registered_native_rollout(
            factory=fit_factory,
            provider=fit_provider,
            host_rollout=runner.application.rollout,
            grant=fit_grant,
            config=bc_config,
            seed=31,
            rollout_options={
                "episode_steps": FIT_STEPS,
                "reset_options": {"rles_episode_index": 31},
            },
        )
        if training_result.behavior_summary.collapsed_to_flat:
            raise RuntimeError("BC fit collapsed into the flat-action deadband")

        teacher_pairs = []
        student_pairs = []
        pair_metrics = []
        factories = {}
        for window_id, start, end in WINDOWS:
            for scenario in ("base", "stress"):
                _config, factory = runner.make_config(
                    label=f"screen-{window_id}-{scenario}",
                    marker_sha=marker_sha,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=runner.ns(start),
                    end_ns=runner.ns(end),
                    mode=scenario,
                    fit_scaler=False,
                    scaler=fit_factory.scaler,
                )
                factories[(window_id, scenario)] = factory

        for (window_id, scenario), factory in factories.items():
            for seed in runner.SEEDS:
                episode = f"{window_id}-{scenario}"
                teacher_run = runner.run_episode(
                    factory, teacher, source_identity, episode, seed, SCREEN_STEPS
                )
                baseline_run = runner.run_episode(
                    factory, baseline, source_identity, episode + "-baseline", seed, SCREEN_STEPS
                )
                student_run = runner.run_episode(
                    factory,
                    teacher,
                    source_identity,
                    episode + "-student",
                    seed,
                    SCREEN_STEPS,
                    policy=training_result.policy,
                )
                key = f"{window_id}-{scenario}-seed-{seed}"
                evidence_hashes = {}
                for role, run in (
                    ("teacher", teacher_run),
                    ("baseline", baseline_run),
                    ("student", student_run),
                ):
                    evidence_hashes[role] = runner.write_verified(
                        OUT / "native-evidence" / f"{key}-{role}.json", run["evidence"]
                    )
                barriers_match = (
                    teacher_run["barriers"]
                    == baseline_run["barriers"]
                    == student_run["barriers"]
                    == SCREEN_STEPS
                )
                pair_metrics.append(
                    {
                        "window_id": window_id,
                        "scenario": scenario,
                        "seed": seed,
                        "teacher_net_return": teacher_run["net_return"],
                        "baseline_net_return": baseline_run["net_return"],
                        "student_net_return": student_run["net_return"],
                        "teacher_max_drawdown": teacher_run["max_drawdown"],
                        "student_max_drawdown": student_run["max_drawdown"],
                        "teacher_filled_orders": teacher_run["filled_orders"],
                        "student_filled_orders": student_run["filled_orders"],
                        "teacher_liquidations": teacher_run["liquidations"],
                        "student_liquidations": student_run["liquidations"],
                        "matched_native_barriers": barriers_match,
                        "evidence_sha256": evidence_hashes,
                    }
                )
                teacher_pairs.append(
                    NativePair(
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
                        matched_native_barriers=barriers_match,
                        scenario_only=True,
                        nonoverlap_verified=True,
                    )
                )
                student_pairs.append(
                    StudentPair(
                        window_id=window_id,
                        scenario=scenario,
                        seed=seed,
                        teacher_recipe=teacher_hash,
                        student_recipe=bc_student_recipe_sha256(bc_config),
                        source_identity=source_identity,
                        teacher_net_return=teacher_run["net_return"],
                        student_net_return=student_run["net_return"],
                        teacher_max_drawdown=teacher_run["max_drawdown"],
                        student_max_drawdown=student_run["max_drawdown"],
                        student_filled_orders=student_run["filled_orders"],
                        student_liquidations=student_run["liquidations"],
                        matched_native_barriers=(
                            teacher_run["barriers"] == student_run["barriers"] == SCREEN_STEPS
                        ),
                        teacher_gate_sha256="0" * 64,
                        actor_sha256=bc_actor_sha256(training_result.policy.model),
                        observation_schema_sha256=bc_observation_schema_sha256(bc_config),
                        action_schema_sha256=bc_action_schema_sha256(),
                        execution_recipe_sha256=fit_factory.route_a_execution_recipe_hash,
                        teacher_evidence_sha256=evidence_hashes["teacher"],
                        student_evidence_sha256=evidence_hashes["student"],
                        scenario_only=True,
                        nonoverlap_verified=True,
                    )
                )

        teacher_screen = screen_teacher(teacher_pairs)
        student_screen = None
        if teacher_screen.status == "TRAIN_INNER_SCREEN_PASS":
            student_pairs = [
                replace(pair, teacher_gate_sha256=teacher_screen.receipt_sha256)
                for pair in student_pairs
            ]
            student_screen = screen_student(student_pairs, teacher_screen=teacher_screen)
            if not verify_student_screen(student_screen):
                raise RuntimeError("host could not recompute the Train promotion receipt")

        result = replace(training_result, student_screen=student_screen)
        experiment = {
            "schema": "rles.route_a.long_horizon_student_screen.v1",
            "source_marker_sha256": marker_sha,
            "source_mode": "whole_snapshot_isolated_v1",
            "fit_window_ns": [runner.ns(FIT[0]), runner.ns(FIT[1])],
            "fit_episode_steps": FIT_STEPS,
            "fit_teacher_recipe_sha256": teacher_hash,
            "student_training_recipe": asdict(bc_config),
            "student_training_recipe_sha256": bc_student_recipe_sha256(bc_config),
            "training_native_summary": asdict(training_result.native_summary),
            "training_behavior": asdict(training_result.behavior_summary),
            "training_runtime": training_runtime,
            "training_intent_counts": dict(
                sorted(
                    Counter(
                        step.advice.intent
                        for step in training_result.labeled_steps
                        if step.intent_supervision_weight > 0
                    ).items()
                )
            ),
            "screen_windows": [
                {"id": w, "start_ns": runner.ns(a), "end_ns": runner.ns(b)} for w, a, b in WINDOWS
            ],
            "screen_episode_steps": SCREEN_STEPS,
            "scenarios": ["base", "stress"],
            "seeds": list(runner.SEEDS),
            "pair_metrics": pair_metrics,
            "teacher_screen": asdict(teacher_screen),
            "student_screen": None if student_screen is None else asdict(student_screen),
            "student_screen_reverified": (
                student_screen is not None and verify_student_screen(student_screen)
            ),
            "economic_status": "SCENARIO_ONLY",
        }
        runner.write_verified(OUT / "screen-report.json", experiment)
        summary = {
            "output": str(OUT),
            "teacher_status": teacher_screen.status,
            "teacher_reasons": teacher_screen.reasons,
            "student_status": (
                student_screen.status if student_screen is not None else "TEACHER_GATE_REJECT"
            ),
            "student_reasons": (
                student_screen.reasons if student_screen is not None else teacher_screen.reasons
            ),
            "student_mean_excess_return": (
                student_screen.mean_excess_return if student_screen is not None else None
            ),
            "wfo_actor_exported": False,
            "economic_status": "SCENARIO_ONLY",
        }
        if student_screen is not None and student_screen.status == "TRAIN_INNER_SCREEN_PASS":
            _selection_cfg, selection_factory = runner.make_config(
                label="independent-selection-teacher-april13",
                marker_sha=marker_sha,
                scope_start=scope_start,
                scope_end=scope_end,
                start_ns=runner.ns(WINDOWS[0][1]),
                end_ns=runner.ns(WINDOWS[0][2]),
                mode="base",
                fit_scaler=False,
                scaler=fit_factory.scaler,
            )
            _student_cfg, student_factory = runner.make_config(
                label="independent-selection-student-april13",
                marker_sha=marker_sha,
                scope_start=scope_start,
                scope_end=scope_end,
                start_ns=runner.ns(WINDOWS[0][1]),
                end_ns=runner.ns(WINDOWS[0][2]),
                mode="base",
                fit_scaler=False,
                scaler=fit_factory.scaler,
            )
            selection_provider, selection_grant = runner.make_provider(
                selection_factory, teacher, source_identity, "independent-selection-april13"
            )
            student_provider, student_grant = runner.make_provider(
                student_factory, teacher, source_identity, "independent-student-april13"
            )
            result = evaluate_bc_on_registered_train_inner_rollout(
                training_result=result,
                fit_factory=fit_factory,
                selection_factory=selection_factory,
                selection_provider=selection_provider,
                student_factory=student_factory,
                student_provider=student_provider,
                host_rollout=runner.application.rollout,
                grant=selection_grant,
                seed=41,
                student_grant=student_grant,
                student_seed=43,
                rollout_options={
                    "episode_steps": SCREEN_STEPS,
                    "reset_options": {"rles_episode_index": 41},
                },
            )
            result = replace(result, student_screen=student_screen)
            wfo_factory = make_route_a_wfo_train(
                fit_candidate=lambda _run: (result, fit_factory),
                validation_backtest=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("Train-only candidate must not open Validation")
                ),
                write_verified=runner.write_verified,
                digest=runner.canonical_digest,
            )
            wfo_run = {
                "output": str(OUT / "wfo-candidate"),
                "route": "route_a",
                "algorithm": "route_a_bc_gru",
                "environment": {
                    "start_ns": runner.ns("2024-01-01T00:00:00+00:00"),
                    "end_ns": scope_end,
                    "execution": fit_config["environment"]["execution"],
                },
            }
            summary["wfo_candidate"] = wfo_factory(wfo_run, evaluate=False)
            summary["wfo_actor_exported"] = (OUT / "wfo-candidate" / "bundle.json").is_file()
            experiment["wfo_candidate"] = summary["wfo_candidate"]
            runner.write_verified(OUT / "screen-report-final.json", experiment)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["student_status"] == "TRAIN_INNER_SCREEN_PASS" else 2
    finally:
        ACTIVE_OWNER.reset(owner_token)


if __name__ == "__main__":
    raise SystemExit(main())
