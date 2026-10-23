"""Capability-bound Route A BC evaluation through the host WFO rollout hook."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .native_inference import load_route_a_predictor_factory
from .execution_contract import require_route_a_flat_close


def make_route_a_wfo_backtest(
        *, factory_builder: Callable[[dict, dict, Any], Any],
        host_rollout: Callable[..., tuple[dict, np.ndarray]],
        episode_id: Callable[[Any, Mapping[str, Any]], str],
        market_key: Callable[[str], str], device: str = "cpu",
        visible_context: Callable[[Any], Mapping[str, Any]] | None = None,
        market_feature_map: Mapping[str, str] | None = None,
) -> Callable[..., dict]:
    """Create a host-WFO custom backtest callback for Route A BC bundles.

    The callback requires the WFO EvaluationAccess capability and consumes it
    before ``factory_builder`` can open or prepare the evaluation window. The
    builder receives the frozen run, WFO bundle metadata, and consumed access;
    it must return the host's native historical factory for that exact scope.
    """
    if not callable(factory_builder) or not callable(host_rollout):
        raise TypeError("host factory builder and native rollout callable required")
    if not callable(episode_id) or not callable(market_key):
        raise TypeError("episode and market identity resolvers required")
    if not isinstance(device, str) or not device:
        raise ValueError("inference device is required")

    def backtest(run: dict, bundle_root: str | Path, *,
                 evaluation_access: Any = None, validation_access: Any = None) -> dict:
        if (evaluation_access is None) == (validation_access is None):
            raise PermissionError("exactly one host WFO evaluation capability is required")
        capability = (evaluation_access if evaluation_access is not None
                      else validation_access)
        is_validation = validation_access is not None
        root = Path(bundle_root).resolve()
        bundle_file = root / "bundle.json"
        if bundle_file.is_symlink() or not bundle_file.is_file():
            raise ValueError("host WFO bundle.json is missing or unsafe")
        bundle_bytes = bundle_file.read_bytes()
        bundle_sha = sha256(bundle_bytes).hexdigest()
        try:
            bundle = json.loads(bundle_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("host WFO bundle.json is malformed") from exc
        if (not isinstance(bundle, dict) or bundle.get("schema") != "rles.historical-bundle.v1"
                or not isinstance(bundle.get("actor"), dict)
                or bundle["actor"].get("kind") != "route_a_bc"):
            raise ValueError("WFO artifact does not contain a Route A BC actor")
        identity = getattr(capability, "identity", None)
        consume = getattr(capability, "consume", None)
        if not isinstance(identity, dict) or not callable(consume):
            raise PermissionError("host WFO evaluation capability required")
        evaluation = run.get("evaluation")
        environment = run.get("environment")
        if (not isinstance(evaluation, dict) or not isinstance(environment, dict)
                or not isinstance(environment.get("instrument"), str)):
            raise ValueError("WFO run lacks a frozen evaluation environment")
        require_route_a_flat_close(environment.get("execution"),
                                   context="Route A WFO replay")
        split = evaluation.get("split")
        start_ns, end_ns = evaluation.get("start_ns"), evaluation.get("end_ns")
        mode = evaluation.get("mode", "base")
        if (not isinstance(split, str) or type(start_ns) is not int or type(end_ns) is not int
                or start_ns >= end_ns or not isinstance(mode, str)):
            raise ValueError("WFO evaluation window is invalid")
        market = market_key(environment["instrument"])
        # This is the authorization boundary: consume the one-shot capability
        # before creating the evaluation factory or touching its market reader.
        if is_validation:
            if (identity.get("split") != "validation"
                    or not isinstance(identity.get("query_id"), str)
                    or identity.get("candidate_id") != run.get("wfo_candidate_id")
                    or identity.get("fold_id") != run.get("wfo_fold_id")
                    or identity.get("mode", "base") != mode
                    or not isinstance(identity.get("source_sha256"), str)
                    or identity.get("market") != market
                    or identity.get("start_ns") != start_ns
                    or identity.get("end_ns") != end_ns):
                raise PermissionError("WFO Validation capability differs from candidate/fold window")

        relative_actor_path = bundle["actor"].get("path")
        if (not isinstance(relative_actor_path, str) or not relative_actor_path.strip()
                or Path(relative_actor_path).is_absolute()
                or ".." in Path(relative_actor_path).parts):
            raise ValueError("Route A actor path must be a safe bundle-relative directory")
        actor_candidate = root / relative_actor_path
        if actor_candidate.is_symlink():
            raise ValueError("Route A actor bundle directory may not be a symlink")
        actor_root = actor_candidate.resolve()
        if actor_root == root or root not in actor_root.parents:
            raise ValueError("Route A actor bundle escapes the WFO artifact directory")
        if (not actor_root.is_dir()
                or any((actor_root / name).is_symlink() or not (actor_root / name).is_file()
                       for name in ("manifest.json", "weights.npz"))):
            raise ValueError("Route A BC bundle assets are missing or unsafe")
        manifest_sha = sha256((actor_root / "manifest.json").read_bytes()).hexdigest()
        if bundle["actor"].get("manifest_sha256") != manifest_sha:
            raise ValueError("WFO bundle does not bind the immutable Route A actor manifest")
        # Validate the complete immutable actor artifact before spending the
        # one-shot research query, then consume before any evaluation reader is
        # constructed by the host factory builder.
        if is_validation:
            consume(bundle_sha256=bundle_sha)
            if getattr(validation_access, "bundle_sha256", None) != bundle_sha:
                raise PermissionError("WFO Validation capability does not bind this actor bundle")
        else:
            consume(bundle_sha=bundle_sha, split=split, start_ns=start_ns, end_ns=end_ns,
                    market=market, mode=mode)
        factory = factory_builder(run, bundle, capability)
        require_route_a_flat_close(getattr(factory, "execution", None),
                                   context="Route A WFO replay factory")
        if any(getattr(factory, name, None) != expected for name, expected in {
            "split": split, "start_ns": start_ns, "end_ns": end_ns,
            "mode": mode, "instrument": environment["instrument"],
        }.items()):
            raise ValueError("host factory differs from the consumed WFO evaluation scope")
        fold_id = identity.get("fold_id") if is_validation else identity.get("query_id")
        query_id = identity.get("query_id")
        if not isinstance(fold_id, str) or not fold_id.strip():
            raise PermissionError("WFO capability has no registered Fold/query identity")
        _policy, _config, predictor = load_route_a_predictor_factory(
            actor_root, fold_id=f"evaluation:{fold_id}", episode_id=episode_id,
            device=device, visible_context=visible_context,
            market_feature_map=market_feature_map,
        )
        report, _observations = host_rollout(
            factory, lambda _batch: (_ for _ in ()).throw(
                RuntimeError("Route A rollout must use its bound recurrent predictor")),
            seed=run.get("seed", 31), soft_parameters=bundle.get("soft_multipliers"),
            predictor_factory=predictor, include_execution_trace=True,
        )
        if not isinstance(report, dict):
            raise TypeError("host native rollout must return a report mapping")
        report["dataset_quality"] = asdict(factory.quality)
        report["execution_mode"] = factory.mode
        report["evaluation_scope"] = {
            "split": split, "formal": split in {"test", "holdout"},
            "mode": mode,
            "project_access_key": getattr(evaluation_access, "key", None),
            "query_id": query_id,
            "fold_id": fold_id,
        }
        return report

    return backtest


def make_route_a_wfo_stress_validation(
        *, factory_builder: Callable[[dict, dict, Any], Any],
        host_rollout: Callable[..., tuple[dict, np.ndarray]],
        episode_id: Callable[[Any, Mapping[str, Any]], str],
        market_key: Callable[[str], str], device: str = "cpu",
        visible_context: Callable[[Any], Mapping[str, Any]] | None = None,
        market_feature_map: Mapping[str, str] | None = None,
) -> Callable[..., dict]:
    """Create paired stress Validation using the host's stress cost factory.

    The callback shape matches ``run_wfo(stress_validation_fn=...)``. It
    requires the host-issued, one-shot Validation capability bound to the
    candidate bundle and verifies the factory covers exactly the paired
    Validation window in stress mode. ``factory_builder`` owns the host
    execution-cost stress configuration and native data access.
    """
    run_backtest = make_route_a_wfo_backtest(
        factory_builder=factory_builder, host_rollout=host_rollout,
        episode_id=episode_id, market_key=market_key, device=device,
        visible_context=visible_context, market_feature_map=market_feature_map,
    )

    def stress_validation(
            run: dict, bundle_root: str | Path, *, start_ns: int, end_ns: int,
            coverage_policy: dict, validation_access: Any = None,
    ) -> dict:
        evaluation = run.get("evaluation")
        if (not isinstance(evaluation, dict)
                or evaluation.get("split") != "validation"
                or evaluation.get("mode") != "stress"
                or evaluation.get("start_ns") != start_ns
                or evaluation.get("end_ns") != end_ns
                or evaluation.get("coverage_policy") != coverage_policy):
            raise PermissionError("stress Validation run differs from the host's frozen paired window")
        if validation_access is None:
            raise PermissionError("host WFO stress Validation capability is required")
        report = run_backtest(
            run, bundle_root, validation_access=validation_access,
        )
        if report.get("execution_mode") != "stress":
            raise ValueError("host stress Validation factory did not execute stress costs")
        return report

    return stress_validation
