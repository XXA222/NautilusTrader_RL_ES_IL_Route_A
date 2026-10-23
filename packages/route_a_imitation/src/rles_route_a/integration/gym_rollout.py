"""Collect Route A teacher barriers from the host's native Gym environment.

The provider owns no market reader or execution simulator. It calls the
already-authorized Gym factory and requires separate terminal L6 handling.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import numpy as np

from rles_route_a.contracts import ExpertAdvice, ExpertTeacher
from rles_route_a.export.inference import BCInferencePolicy
from .host_gym_projection import (
    AuditedTeacherBarrier,
    L3OrderFacts,
    TerminalL6Receipt,
    l3_order_facts_from_gym_info,
    project_nonterminal_teacher_barrier,
    project_terminal_teacher_barrier,
)
from .l2_adapter import (
    CAUSAL_MTF_V5_ROUTE_A_FEATURES,
    CAUSAL_MTF_V6_ROUTE_A_FEATURES,
    project_host_visible_context,
)
from .native_rollout import NativeTeacherFact
from .scope import HostTrainGrant
from .m22_trace import verify_m22_decision_trace, verify_m22_order_facts
from .execution_contract import require_route_a_flat_close
from rles_route_a.teachers.protocol import classify_target
from .live_dagger_projection import LiveNautilusGymProjection


def _callable_identity(callback: Callable[..., Any] | None) -> str | None:
    if callback is None:
        return None
    module = getattr(callback, "__module__", type(callback).__module__)
    name = getattr(callback, "__qualname__", type(callback).__qualname__)
    return f"{module}.{name}"

if TYPE_CHECKING:
    from rles_route_a.training.dagger_collector import (
        ProjectedTeacherTarget, VisitedStudentState,
    )


class NativeGymTeacherProvider:
    """Run one host Gym episode and yield audited native teacher facts.

    Terminal steps require either ``terminal_receipt_provider`` to return the
    host's L6 terminal cashflow/equity receipt or a host ``terminal_projector``.
    Without one, the provider raises instead of losing the final decision.

    Historical/synthetic execution remains ``scenario_only`` by default. The
    optional economic evidence authority may clear that classification only
    after verifying the provenance of the market events, real/supported fill
    evidence, and fee-inclusive accounting represented by this exact barrier.
    """

    def __init__(self, *, env_factory: Callable[..., Any], teacher: ExpertTeacher,
                 fold_id: str, execution_recipe: str, source_identity: str,
                 native_order_facts_verifier: Callable[[Mapping[str, Any], str,
                                                        L3OrderFacts], bool],
                 no_external_cashflow_receipt: Callable[[Mapping[str, Any]], str],
                 no_external_cashflow_verifier: Callable[[Mapping[str, Any], str, str], bool],
                 native_economic_evidence_verifier: Callable[[Mapping[str, Any], str,
                                                              L3OrderFacts], bool] | None = None,
                 episode_id: Callable[[Any, Mapping[str, Any]], str],
                 visible_context: Callable[[Any], Mapping[str, Any]] | None = None,
                 market_feature_map: Mapping[str, str] | None = None,
                 terminal_receipt_provider: Callable[[Mapping[str, Any], str],
                                                     TerminalL6Receipt] | None = None,
                 terminal_receipt_verifier: Callable[[Mapping[str, Any], str,
                                                      TerminalL6Receipt], bool] | None = None,
                 terminal_projector: Callable[[Any, Any, Mapping[str, Any], L3OrderFacts,
                                               str, str], AuditedTeacherBarrier] | None = None):
        for name, value in (("fold_id", fold_id), ("execution_recipe", execution_recipe),
                            ("source_identity", source_identity)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        if (not callable(env_factory) or not callable(native_order_facts_verifier)
                or not callable(no_external_cashflow_receipt)
                or not callable(no_external_cashflow_verifier)):
            raise TypeError("authorized Gym factory and native-order/L6 verifiers are required")
        if (native_economic_evidence_verifier is not None
                and not callable(native_economic_evidence_verifier)):
            raise TypeError("native economic evidence verifier must be callable")
        if terminal_receipt_provider is not None and not callable(terminal_receipt_verifier):
            raise TypeError("terminal L6 receipt provider requires its authority verifier")
        if not callable(episode_id):
            raise TypeError("episode identity callback is required")
        self.env_factory = env_factory
        self.teacher = teacher
        self.fold_id = fold_id
        self.execution_recipe = execution_recipe
        self.source_identity = source_identity
        self.native_order_facts_verifier = native_order_facts_verifier
        self.no_external_cashflow_receipt = no_external_cashflow_receipt
        self.no_external_cashflow_verifier = no_external_cashflow_verifier
        self.native_economic_evidence_verifier = native_economic_evidence_verifier
        self.episode_id = episode_id
        self.visible_context = visible_context or (lambda env: env.rles_visible_context_inputs())
        self.market_feature_map = (dict(CAUSAL_MTF_V5_ROUTE_A_FEATURES)
                                   if market_feature_map is None else dict(market_feature_map))
        self.terminal_receipt_provider = terminal_receipt_provider
        self.terminal_receipt_verifier = terminal_receipt_verifier
        self.terminal_projector = terminal_projector


    @classmethod
    def from_prepared_factory(
            cls, factory: Any, *, fold_id: str, teacher: ExpertTeacher,
            native_order_facts_verifier: Callable[[Mapping[str, Any], str,
                                                   L3OrderFacts], bool],
            no_external_cashflow_receipt: Callable[[Mapping[str, Any]], str],
            no_external_cashflow_verifier: Callable[[Mapping[str, Any], str, str], bool],
            native_economic_evidence_verifier: Callable[[Mapping[str, Any], str,
                                                         L3OrderFacts], bool] | None = None,
            episode_id: Callable[[Any, Mapping[str, Any]], str],
            visible_context: Callable[[Any], Mapping[str, Any]] | None = None,
            market_feature_map: Mapping[str, str] | None = None,
            terminal_receipt_provider: Callable[[Mapping[str, Any], str],
                                                TerminalL6Receipt] | None = None,
            terminal_receipt_verifier: Callable[[Mapping[str, Any], str,
                                                 TerminalL6Receipt], bool] | None = None,
            terminal_projector: Callable[[Any, Any, Mapping[str, Any], L3OrderFacts,
                                          str, str], AuditedTeacherBarrier] | None = None
    ) -> tuple["NativeGymTeacherProvider", HostTrainGrant]:
        """Bind a prepared host factory and its registered Train receipt.

        This is the supported assembly path from ``application.prepare()``:
        an ordinary or evaluation factory without its retained host read
        receipt is rejected before the first environment is created.
        """
        require_route_a_flat_close(getattr(factory, "execution", None),
                                   context="Route A Train Gym")
        registered_read = getattr(factory, "application_registered_train_read", None)
        verifier = getattr(factory, "application_registered_train_read_verifier", None)
        if not callable(verifier) or verifier(registered_read) is not True:
            raise PermissionError("host project registry did not verify the Train read receipt")
        grant = HostTrainGrant.from_registered_read(registered_read, fold_id=fold_id)
        env_factory = getattr(factory, "make_env", None)
        execution_recipe = getattr(factory, "transition_hash", None)
        if not callable(env_factory):
            raise TypeError("prepared historical factory must expose make_env()")
        if not isinstance(execution_recipe, str) or not execution_recipe.strip():
            raise ValueError("prepared historical factory has no frozen transition recipe")
        if market_feature_map is None:
            profile = getattr(factory, "feature_profile", "legacy")
            market_feature_map = (
                CAUSAL_MTF_V6_ROUTE_A_FEATURES
                if profile == "causal_mtf_v6"
                else CAUSAL_MTF_V5_ROUTE_A_FEATURES
            )
        provider = cls(
            env_factory=env_factory, teacher=teacher, fold_id=fold_id,
            execution_recipe=execution_recipe, source_identity=grant.source_identity,
            native_order_facts_verifier=native_order_facts_verifier,
            no_external_cashflow_receipt=no_external_cashflow_receipt,
            no_external_cashflow_verifier=no_external_cashflow_verifier,
            native_economic_evidence_verifier=native_economic_evidence_verifier,
            episode_id=episode_id, visible_context=visible_context,
            market_feature_map=market_feature_map,
            terminal_receipt_provider=terminal_receipt_provider,
            terminal_receipt_verifier=terminal_receipt_verifier,
            terminal_projector=terminal_projector,
        )
        return provider, grant

    def rollout(self, *, fold_id: str, grant: HostTrainGrant, seed: int,
                rollout_id: str | None = None
                ) -> Iterator[NativeTeacherFact]:
        if fold_id != self.fold_id or not isinstance(grant, HostTrainGrant):
            raise PermissionError("matching host-registered Train grant required")
        if grant.fold_id != fold_id or grant.source_identity != self.source_identity:
            raise PermissionError("Train grant does not bind the rollout source and fold")
        if type(seed) is not int:
            raise TypeError("deterministic integer rollout seed required")
        trajectory_id = self._trajectory_identity(grant, seed, rollout_id)
        env = self.env_factory()
        try:
            _observation, reset_info = env.reset(seed=seed)
            episode = self.episode_id(env, reset_info)
            reset_teacher = getattr(self.teacher, "reset", None)
            if callable(reset_teacher):
                reset_teacher(episode)
            while True:
                context = self.visible_context(env)
                observation = project_host_visible_context(
                    context=context, fold_id=fold_id, episode_id=episode,
                    market_feature_map=self.market_feature_map,
                )
                advice = self.teacher.advise(observation)
                result = env.step([advice.requested_exposure])
                if not isinstance(result, tuple) or len(result) != 5:
                    raise TypeError("host Gym step must return observation, reward, flags, info")
                _next_observation, _reward, terminated, truncated, info = result
                if not isinstance(info, Mapping):
                    raise ValueError("host Gym step info mapping is required")
                order_facts = l3_order_facts_from_gym_info(info)
                barrier = self._project_step(
                    observation, advice, info, order_facts,
                    terminated=bool(terminated), truncated=bool(truncated),
                    trajectory_id=trajectory_id,
                )
                yield barrier
                if terminated or truncated:
                    return
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def _project_step(self, observation: Any, advice: Any, info: Mapping[str, Any],
                      order_facts: L3OrderFacts, *, terminated: bool,
                      truncated: bool, trajectory_id: str) -> AuditedTeacherBarrier:
        if self.native_order_facts_verifier(
                info, self.source_identity, order_facts) is not True:
            raise PermissionError("independent M2.2 authority rejected native order facts")
        scenario_only = True
        economic_authority = None
        if self.native_economic_evidence_verifier is not None:
            if self.native_economic_evidence_verifier(
                    info, self.source_identity, order_facts) is not True:
                raise PermissionError(
                    "native economic authority did not verify observed market/execution evidence"
                )
            scenario_only = False
            economic_authority = _callable_identity(
                self.native_economic_evidence_verifier
            )
        if terminated or truncated:
            if self.terminal_projector is not None:
                barrier = self.terminal_projector(
                    observation, advice, info, order_facts,
                    self.execution_recipe, self.source_identity,
                )
            elif self.terminal_receipt_provider is not None:
                terminal_receipt = self.terminal_receipt_provider(info, self.source_identity)
                if (self.terminal_receipt_verifier is None
                        or self.terminal_receipt_verifier(
                            info, self.source_identity, terminal_receipt,
                        ) is not True):
                    raise PermissionError("L6 authority did not verify terminal receipt")
                barrier = project_terminal_teacher_barrier(
                    observation=observation, advice=advice, gym_info=info,
                    l3_orders=order_facts, terminal_receipt=terminal_receipt,
                    execution_recipe=self.execution_recipe,
                    source_identity=self.source_identity,
                    scenario_only=scenario_only,
                ).teacher_barrier
            else:
                raise ValueError("terminal decision requires a native L6 terminal projector")
            if (not isinstance(barrier, AuditedTeacherBarrier)
                    or not barrier.terminal_l6_receipt):
                raise ValueError("terminal projector did not retain its L6 receipt")
            if barrier.trace.scenario_only != scenario_only:
                raise PermissionError("terminal projector economic evidence class disagrees with authority")
            barrier = replace(
                barrier,
                native_economic_evidence_authority=economic_authority,
            )
            return self._bind_trajectory(barrier, trajectory_id)

        no_flow_receipt = self.no_external_cashflow_receipt(info)
        if self.no_external_cashflow_verifier(
                info, self.source_identity, no_flow_receipt) is not True:
            raise PermissionError("L6 authority did not verify no-cashflow receipt")
        barrier = project_nonterminal_teacher_barrier(
            observation=observation, advice=advice, gym_info=info,
            l3_orders=order_facts, execution_recipe=self.execution_recipe,
            source_identity=self.source_identity,
            no_external_cashflow_receipt=no_flow_receipt,
            scenario_only=scenario_only,
        ).teacher_barrier
        barrier = replace(
            barrier,
            native_economic_evidence_authority=economic_authority,
        )
        return self._bind_trajectory(barrier, trajectory_id)

    def _trajectory_identity(self, grant: HostTrainGrant, seed: int,
                             rollout_id: str | None,
                             actor_identity: str | None = None) -> str:
        if rollout_id is not None and (
                not isinstance(rollout_id, str) or not rollout_id.strip()):
            raise ValueError("rollout_id must be a nonempty string when supplied")
        if actor_identity is None:
            actor_type = type(self.teacher)
            actor_identity = f"teacher:{actor_type.__module__}.{actor_type.__qualname__}"
        payload = "\0".join((grant.fold_id, grant.source_identity,
                              self.execution_recipe, str(seed), actor_identity,
                              rollout_id or ""))
        return "routea-" + sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _bind_trajectory(barrier: AuditedTeacherBarrier,
                         trajectory_id: str) -> AuditedTeacherBarrier:
        if barrier.trajectory_id and barrier.trajectory_id != trajectory_id:
            raise ValueError("projected teacher barrier has a conflicting trajectory identity")
        return replace(barrier, trajectory_id=trajectory_id)

    def rollout_with_m22_trace(
            self, *, factory: Any, host_rollout: Callable[..., Any],
            grant: HostTrainGrant, seed: int, rollout_id: str | None = None,
            decision_policy: BCInferencePolicy | None = None,
            student_policy_id: str | None = None,
            dagger_projection_recipe: str | None = None,
            dagger_state_sink: Callable[[tuple[VisitedStudentState, ...]], None] | None = None,
            **rollout_options: Any
    ) -> tuple[tuple[AuditedTeacherBarrier, ...], Mapping[str, Any]]:
        """Run teacher through the host rollout and verify its same-run M2.2 trace.

        ``host_rollout`` is the existing orchestrator ``rollout`` function with
        the generic predictor/decision observer hooks. The method buffers facts
        until its returned M2.2 decision trace has been reconciled.
        """
        if not callable(host_rollout):
            raise TypeError("host M2.2 rollout callable required")
        if (not isinstance(grant, HostTrainGrant) or grant.fold_id != self.fold_id
                or grant.source_identity != self.source_identity):
            raise PermissionError("matching registered Train grant required")
        if getattr(factory, "make_env", None) is None:
            raise TypeError("host historical factory with make_env() required")
        require_route_a_flat_close(getattr(factory, "execution", None),
                                   context="Route A Train Gym")
        if getattr(factory, "transition_hash", None) != self.execution_recipe:
            raise ValueError("M2.2 factory execution recipe differs from prepared grant")
        if decision_policy is not None and not isinstance(decision_policy, BCInferencePolicy):
            raise TypeError("native student replay requires a frozen Route A BC inference policy")
        actor_identity = None
        active_student_policy_id: str = ""
        if decision_policy is not None:
            digest = sha256()
            for name, tensor in sorted(decision_policy.model.state_dict().items()):
                value = tensor.detach().cpu().contiguous()
                digest.update(name.encode("utf-8"))
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(np.asarray(value).tobytes(order="C"))
            actor_identity = f"route_a_bc:{digest.hexdigest()}"
        if (dagger_state_sink is None) != (student_policy_id is None):
            raise ValueError("DAgger state sink and student policy identity must be supplied together")
        if (dagger_state_sink is None) != (dagger_projection_recipe is None):
            raise ValueError("DAgger state sink and frozen projection recipe must be supplied together")
        if dagger_state_sink is not None:
            if decision_policy is None or not callable(dagger_state_sink):
                raise TypeError("DAgger collection requires the frozen student policy and a state sink")
            if not isinstance(student_policy_id, str) or not student_policy_id:
                raise ValueError("DAgger collection requires the frozen student policy identity")
            active_student_policy_id = student_policy_id
            if student_policy_id != actor_identity:
                raise ValueError("DAgger student identity must equal the frozen actor hash")
            if (not isinstance(dagger_projection_recipe, str)
                    or len(dagger_projection_recipe) != 64
                    or any(character not in "0123456789abcdef"
                           for character in dagger_projection_recipe)):
                raise ValueError("DAgger requires a window-independent execution recipe hash")
            from rles_route_a.training.dagger_collector import VisitedStudentState
        trajectory_id = self._trajectory_identity(
            grant, seed, rollout_id, actor_identity=actor_identity,
        )

        barriers: list[AuditedTeacherBarrier] = []
        dagger_visits: list[VisitedStudentState] = []
        collected_order_facts = []
        policy_barrier_equity_by_id: dict[str, Mapping[str, Any]] = {}
        current: list[tuple[Any, Any, ExpertAdvice | None,
                             ProjectedTeacherTarget | None]] = []

        def predictor_factory(env: Any, reset_info: Mapping[str, Any]):
            episode = self.episode_id(env, reset_info)
            if decision_policy is not None:
                decision_policy.reset(episode)
            reset_teacher = getattr(self.teacher, "reset", None)
            if callable(reset_teacher):
                reset_teacher(episode)

            def predict(_batch: Any):
                if current:
                    raise RuntimeError("previous teacher command has no native step result")
                observation = project_host_visible_context(
                    context=self.visible_context(env), fold_id=self.fold_id,
                    episode_id=episode, market_feature_map=self.market_feature_map,
                )
                if decision_policy is None:
                    advice = self.teacher.advise(observation)
                    teacher_label = None
                    projected_target = None
                else:
                    requested = decision_policy.predict_exposure(observation)
                    current_exposure = observation.account_value("net_exposure")
                    advice = ExpertAdvice(
                        "bc_student_native_eval", observation.decision_ts_ns,
                        classify_target(current_exposure, requested), requested,
                        ("BC_STUDENT_NATIVE_EVAL",),
                    )
                    if dagger_state_sink is not None:
                        teacher_label = self.teacher.advise(observation)
                        projector = LiveNautilusGymProjection(
                            env,
                            execution_recipe=self.execution_recipe,
                            projection_recipe=dagger_projection_recipe,
                        )
                        projected_target = projector.project_native_target(
                            observation, teacher_label.requested_exposure,
                        )
                    else:
                        teacher_label = None
                        projected_target = None
                current.append((observation, advice, teacher_label, projected_target))
                return [advice.requested_exposure]

            return predict

        def observe_step(env: Any, _decision_observation: Any, info: Any,
                         terminated: bool, truncated: bool) -> None:
            if not isinstance(info, Mapping):
                raise TypeError("host M2.2 step observer received malformed Gym info")
            if len(current) != 1:
                raise ValueError("host rollout step has no unique Route A teacher decision")
            decision_id = info.get("execution_decision_id")
            if (not isinstance(decision_id, str) or not decision_id.strip()
                    or decision_id in policy_barrier_equity_by_id):
                raise ValueError("host Gym step has missing or duplicate decision identity")
            policy_barrier_equity_by_id[decision_id] = {
                key: info.get(key) for key in (
                    "policy_equity_before", "policy_equity_after", "policy_equity_change",
                )
            }
            observation, advice, teacher_label, projected_target = current.pop()
            facts = l3_order_facts_from_gym_info(info)
            collected_order_facts.append(facts)
            barrier = self._project_step(
                observation, advice, info, facts,
                terminated=terminated, truncated=truncated,
                trajectory_id=trajectory_id,
            )
            barriers.append(barrier)
            if dagger_state_sink is not None:
                if teacher_label is None or projected_target is None:
                    raise RuntimeError("DAgger pre-action label/projection was not retained")
                dagger_visits.append(VisitedStudentState(
                    observation=observation,
                    native_action_trace=barrier.trace,
                    student_policy_id=active_student_policy_id,
                    source_identity=self.source_identity,
                    split="train",
                    teacher_label=teacher_label,
                    projected_teacher_target=projected_target,
                ))

        def unused_predict(_batch: Any):
            raise RuntimeError("host rollout did not use the bound Route A predictor")

        result = host_rollout(
            factory, unused_predict, seed=seed, include_execution_trace=True,
            predictor_factory=predictor_factory, decision_observer=observe_step,
            **rollout_options,
        )
        if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], Mapping):
            raise TypeError("host M2.2 rollout must return report and observations")
        report = dict(result[0])
        execution_trace = report.get("execution_decision_trace")
        if not isinstance(execution_trace, (tuple, list)):
            raise ValueError("host M2.2 report omitted its opt-in execution decision trace")
        verify_m22_decision_trace(
            barriers, execution_trace,
            policy_barrier_equity_by_id=policy_barrier_equity_by_id,
        )
        native_snapshots = getattr(factory, "snapshots", None)
        if not isinstance(native_snapshots, Sequence):
            raise PermissionError("host M2.2 factory omitted full retained order snapshots")
        order_verification = verify_m22_order_facts(
            barriers, collected_order_facts, native_snapshots,
        )
        # Only this exact returned trajectory has completed independent M2.2
        # order/fill reconciliation, so only these barriers may participate in
        # full trade-lifecycle attribution during teacher auditing.
        barriers = [replace(barrier, native_order_lifecycle_verified=True)
                    for barrier in barriers]
        report["route_a_native_order_verification"] = {
            "authority": "retained_m22_session_snapshots",
            "barriers": order_verification.barriers,
            "order_status_facts": order_verification.order_status_facts,
            "policy_fill_events": order_verification.policy_fill_events,
        }
        report["route_a_economic_evidence"] = {
            "classification": (
                "OBSERVED_NATIVE_AUTHORIZED"
                if self.native_economic_evidence_verifier is not None
                else "SCENARIO_ONLY"
            ),
            "decisions": len(barriers),
            "authority": (
                _callable_identity(self.native_economic_evidence_verifier)
            ),
        }
        if dagger_state_sink is not None:
            # DAgger samples become visible only after the entire trajectory's
            # decision and retained native order traces reconcile successfully.
            dagger_state_sink(tuple(dagger_visits))
        return tuple(barriers), report


class NativeGymDaggerTrainProvider:
    """Adapt one authorized host rollout into M2.2-verified DAgger visits."""

    def __init__(self, *, teacher_provider: NativeGymTeacherProvider,
                 factory: Any, host_rollout: Callable[..., Any], seed: int,
                 decision_policy: BCInferencePolicy,
                 rollout_options: Mapping[str, Any] | None = None) -> None:
        if not isinstance(teacher_provider, NativeGymTeacherProvider):
            raise TypeError("native Route A teacher provider required")
        if (not callable(getattr(factory, "make_env", None))
                or not callable(host_rollout) or type(seed) is not int
                or not isinstance(decision_policy, BCInferencePolicy)):
            raise TypeError("prepared Gym factory, host rollout, seed, and frozen BC policy required")
        self.teacher_provider = teacher_provider
        self.factory = factory
        self.host_rollout = host_rollout
        self.seed = seed
        self.decision_policy = decision_policy
        self.rollout_options = dict(rollout_options or {})
        self.last_report: dict[str, Any] | None = None
        self.last_visits: tuple[VisitedStudentState, ...] | None = None
        self.projection_recipe = getattr(factory, "route_a_execution_recipe_hash", None)
        if (not isinstance(self.projection_recipe, str) or len(self.projection_recipe) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.projection_recipe)):
            raise ValueError("prepared Train factory lacks its frozen execution recipe hash")

    def collect_train_states(self, *, grant: HostTrainGrant,
                             student_policy_id: str) -> tuple[VisitedStudentState, ...]:
        visits: list[VisitedStudentState] = []
        _barriers, report = self.teacher_provider.rollout_with_m22_trace(
            factory=self.factory,
            host_rollout=self.host_rollout,
            grant=grant,
            seed=self.seed,
            rollout_id=f"dagger-{student_policy_id}",
            decision_policy=self.decision_policy,
            student_policy_id=student_policy_id,
            dagger_projection_recipe=self.projection_recipe,
            dagger_state_sink=visits.extend,
            **self.rollout_options,
        )
        self.last_report = dict(report)
        self.last_visits = tuple(visits)
        return self.last_visits
