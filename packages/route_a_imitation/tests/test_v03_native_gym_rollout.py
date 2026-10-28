from __future__ import annotations

from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from rles_route_a.contracts import ExpertAdvice
from rles_route_a.config import BCConfig
from rles_route_a.evaluation.teacher_funnel_1m import TeacherPreDecision
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.integration.gym_rollout import NativeGymTeacherProvider
from rles_route_a.integration.m22_trace import _snapshot_fills, _snapshot_status_rows
from rles_route_a.integration.host_gym_projection import TerminalL6Receipt
from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.models.bc import BCGRUPolicy
from rles_gym_bridge.historical import (
    _AppendOnlyNativeHistory,
    _VersionedNativeOrdersView,
)
from test_v03_host_context import _context, _feature


class _TerminalEnv:
    def reset(self, *, seed):
        self.seed = seed
        return None, {}

    def rles_visible_context_inputs(self):
        return _context()

    def step(self, _action):
        before, after = (), ({
            "client_order_id": "RLES-terminal-order", "status": "FILLED",
            "side": "BUY", "quantity": "1", "filled_qty": "1",
        },)
        receipt = sha256(json.dumps(
            {"before": before, "after": after}, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        info = {
            "execution_plan_action": "SUBMIT",
            "execution_client_order_id": "RLES-terminal-order",
            "execution_related_order_ids": (),
            "native_pending_order_ids_before": (),
            "native_pending_order_ids_after": (),
            "native_policy_fills": ({
                "event_id": "terminal-fill-1", "client_order_id": "RLES-terminal-order",
                "ts_event": _context()["decision_ts_ns"] + 30,
                "last_qty": "1", "last_px": "100", "commission": "0.1 USDT",
            },),
            "native_order_statuses_before": before,
            "native_order_statuses_after": after,
            "native_order_status_receipt": receipt,
            "terminal_settled": True,
        }
        return None, 0.0, True, False, info

    def close(self):
        self.closed = True


class _NonterminalEnv(_TerminalEnv):
    def step(self, _action):
        status_receipt = sha256(json.dumps(
            {"before": (), "after": ()}, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        return None, 0.0, False, False, {
            "decision_ts_ns": _context()["decision_ts_ns"] + 60,
            "terminal_settled": False,
            "execution_plan_action": "NOOP",
            "execution_client_order_id": None,
            "execution_related_order_ids": (),
            "native_pending_order_ids_before": (),
            "native_pending_order_ids_after": (),
            "native_policy_fills": (),
            "native_order_statuses_before": (),
            "native_order_statuses_after": (),
            "native_order_status_receipt": status_receipt,
        }


def test_native_rollout_requires_terminal_receipt_instead_of_dropping_last_decision():
    env = _TerminalEnv()
    teacher = SimpleNamespace(advise=lambda obs: ExpertAdvice(
        "trend", obs.decision_ts_ns, "OPEN_LONG", 0.2, ("native",),
    ))
    provider = NativeGymTeacherProvider(
        env_factory=lambda: env, teacher=teacher, fold_id="fold-1",
        execution_recipe="exec-v1", source_identity="source-marker",
        native_order_facts_verifier=lambda _info, _source, _facts: True,
        no_external_cashflow_receipt=lambda _info: "no-flow-receipt",
        no_external_cashflow_verifier=lambda _info, _source, _receipt: True,
        episode_id=lambda _env, _info: "episode-1",
    )
    grant = HostTrainGrant("fold-1", "owner", "reservation", "source-marker", "train", True)

    with pytest.raises(ValueError, match="terminal decision requires a native L6"):
        tuple(provider.rollout(fold_id="fold-1", grant=grant, seed=7))
    assert env.closed


def test_native_teacher_receives_action_after_m22_and_l6_verification():
    class VerifiedNonterminalEnv(_NonterminalEnv):
        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            info.update({
                "execution_decision_id": "decision-1",
                "execution_plan_id": "plan-1",
                "execution_idempotency_key": "idempotency-1",
                "execution_plan_reason": "already-at-target",
                "evidence_hash": "e" * 64,
                "policy_equity_currency": "USDT",
                "requested_action": action[0],
                "risk_soft_target": action[0],
                "risk_adjusted_target": action[0],
                "quantity_rounded_target_exposure": action[0],
                "policy_executed_action_pre_terminal": action[0],
                "policy_equity_before": "1000",
                "policy_equity_after": "1000",
                "policy_equity_change": "0",
                "r_econ": 0.0,
                "execution_cancel_order_ids": (),
            })
            return observation, reward, terminated, truncated, info

    env = VerifiedNonterminalEnv()
    events = []

    class Teacher:
        def advise(self, observation):
            return ExpertAdvice(
                "trend", observation.decision_ts_ns, "OPEN_LONG", 0.2, ("native",),
            )

        def observe_native_action(self, action, trace, order_facts):
            events.append(("teacher", action, trace, order_facts))

    def verify_native(_info, _source, order_facts):
        events.append(("m22", order_facts))
        return True

    def verify_no_cashflow(_info, _source, receipt):
        events.append(("l6", receipt))
        return True

    def visible_context(_env):
        context = _context()
        context["pending_order_ids"] = ()
        context["execution"] = {"pending_order_count": 0.0}
        return context

    provider = NativeGymTeacherProvider(
        env_factory=lambda: env, teacher=Teacher(), fold_id="fold-1",
        execution_recipe="exec-v1", source_identity="source-marker",
        native_order_facts_verifier=verify_native,
        no_external_cashflow_receipt=lambda _info: "no-flow-receipt",
        no_external_cashflow_verifier=verify_no_cashflow,
        episode_id=lambda _env, _info: "episode-1",
        visible_context=visible_context,
    )
    grant = HostTrainGrant("fold-1", "owner", "reservation", "source-marker", "train", True)

    rollout = provider.rollout(fold_id="fold-1", grant=grant, seed=7)
    barrier = next(rollout)
    rollout.close()

    assert env.closed
    assert [event[0] for event in events] == ["m22", "l6", "teacher"]
    assert events[0][1] is events[2][3]
    assert events[1][1] == "no-flow-receipt"
    assert events[2][1].intent == "OPEN_LONG"
    assert events[2][2] == barrier.trace
    assert events[2][3].native_order_status_receipt


def test_m22_rollout_feedback_follows_per_barrier_m22_and_l6_checks():
    class StopAfterObservation(Exception):
        pass

    class VerifiedNonterminalEnv(_NonterminalEnv):
        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            info.update({
                "execution_decision_id": "decision-1",
                "execution_plan_id": "plan-1",
                "execution_idempotency_key": "idempotency-1",
                "execution_plan_reason": "already-at-target",
                "evidence_hash": "e" * 64,
                "policy_equity_currency": "USDT",
                "requested_action": action[0],
                "risk_soft_target": action[0],
                "risk_adjusted_target": action[0],
                "quantity_rounded_target_exposure": action[0],
                "policy_executed_action_pre_terminal": action[0],
                "policy_equity_before": "1000",
                "policy_equity_after": "1000",
                "policy_equity_change": "0",
                "r_econ": 0.0,
                "execution_cancel_order_ids": (),
            })
            return observation, reward, terminated, truncated, info

    env = VerifiedNonterminalEnv()
    events = []

    class Teacher:
        def advise(self, observation):
            return ExpertAdvice(
                "trend", observation.decision_ts_ns, "OPEN_LONG", 0.2, ("native",),
            )

        def observe_native_action(self, action, trace, order_facts):
            events.append(("teacher", action, trace, order_facts))

    def verify_native(_info, _source, order_facts):
        events.append(("m22", order_facts))
        return True

    def verify_no_cashflow(_info, _source, receipt):
        events.append(("l6", receipt))
        return True

    def visible_context(_env):
        context = _context()
        context["pending_order_ids"] = ()
        context["execution"] = {"pending_order_count": 0.0}
        return context

    provider = NativeGymTeacherProvider(
        env_factory=lambda: env, teacher=Teacher(), fold_id="fold-1",
        execution_recipe="exec-v1", source_identity="source-marker",
        native_order_facts_verifier=verify_native,
        no_external_cashflow_receipt=lambda _info: "no-flow-receipt",
        no_external_cashflow_verifier=verify_no_cashflow,
        episode_id=lambda _env, _info: "episode-1",
        visible_context=visible_context,
    )
    grant = HostTrainGrant("fold-1", "owner", "reservation", "source-marker", "train", True)

    def host_rollout(_factory, _predict, *, seed, predictor_factory,
                     decision_observer, **_kwargs):
        _observation, reset_info = env.reset(seed=seed)
        predict = predictor_factory(env, reset_info)
        action = predict(None)
        _next_observation, _reward, terminated, truncated, info = env.step(action)
        decision_observer(env, None, info, terminated, truncated)
        env.close()
        raise StopAfterObservation

    factory = SimpleNamespace(
        transition_hash="exec-v1",
        execution=SimpleNamespace(zero_target_policy="close"),
        make_env=lambda: env,
    )
    with pytest.raises(StopAfterObservation):
        provider.rollout_with_m22_trace(
            factory=factory, host_rollout=host_rollout, grant=grant, seed=7,
        )

    assert env.closed
    assert [event[0] for event in events] == ["m22", "l6", "teacher"]
    assert events[0][1] is events[2][3]
    assert events[1][1] == "no-flow-receipt"
    assert events[2][1].intent == "OPEN_LONG"
    assert events[2][2].requested_action == events[2][1].requested_exposure


def test_prepared_factory_assembly_requires_registered_train_receipt():
    factory = SimpleNamespace(
        make_env=lambda: None, transition_hash="recipe-v1",
        execution=SimpleNamespace(zero_target_policy="close"),
    )
    teacher = SimpleNamespace(advise=lambda _obs: None)

    with pytest.raises(PermissionError, match="registry did not verify"):
        NativeGymTeacherProvider.from_prepared_factory(
            factory, fold_id="fold-1", teacher=teacher,
            native_order_facts_verifier=lambda _info, _source, _facts: True,
            no_external_cashflow_receipt=lambda _info: "receipt",
            no_external_cashflow_verifier=lambda _info, _source, _receipt: True,
            episode_id=lambda _env, _info: "episode-1",
        )


def test_m22_teacher_rollout_uses_the_provider_feature_mapping():
    class StopAfterPrediction(Exception):
        pass

    env = _TerminalEnv()
    observed = []
    teacher = SimpleNamespace(advise=lambda obs: observed.append(obs) or ExpertAdvice(
        "trend", obs.decision_ts_ns, "HOLD", 0.0, ("native",),
    ))
    mapping = {"trend_1h": "closed_return_1h"}
    provider = NativeGymTeacherProvider(
        env_factory=lambda: env, teacher=teacher, fold_id="fold-1",
        execution_recipe="exec-v1", source_identity="source-marker",
        native_order_facts_verifier=lambda _info, _source, _facts: True,
        no_external_cashflow_receipt=lambda _info: "receipt",
        no_external_cashflow_verifier=lambda _info, _source, _receipt: True,
        episode_id=lambda _env, _info: "episode-1",
        market_feature_map=mapping,
    )
    grant = HostTrainGrant("fold-1", "owner", "reservation", "source-marker", "train", True)

    def host_rollout(_factory, _predict, *, predictor_factory, **_kwargs):
        predict = predictor_factory(env, {})
        predict(None)
        raise StopAfterPrediction

    with pytest.raises(StopAfterPrediction):
        provider.rollout_with_m22_trace(
            factory=SimpleNamespace(
                transition_hash="exec-v1",
                execution=SimpleNamespace(zero_target_policy="close"),
                make_env=lambda: env,
            ),
            host_rollout=host_rollout, grant=grant, seed=7,
        )
    assert len(observed) == 1
    assert dict(observed[0].market) == {"trend_1h": 0.01}


def test_teacher_funnel_hook_captures_predecision_before_native_step():
    class StopAfterPrediction(Exception):
        pass

    env = _TerminalEnv()
    observations = []
    teacher = SimpleNamespace(advise=lambda obs: observations.append(obs) or ExpertAdvice(
        "trend", obs.decision_ts_ns, "FLAT", 0.0, ("synthetic",),
    ))
    provider = NativeGymTeacherProvider(
        env_factory=lambda: env, teacher=teacher, fold_id="fold-1",
        execution_recipe="exec-v1", source_identity="source-marker",
        native_order_facts_verifier=lambda _info, _source, _facts: True,
        no_external_cashflow_receipt=lambda _info: "receipt",
        no_external_cashflow_verifier=lambda _info, _source, _receipt: True,
        episode_id=lambda _env, _info: "episode-1",
        visible_context=lambda _env: _context(),
        market_feature_map={"trend_1h": "closed_return_1h"},
    )
    grant = HostTrainGrant("fold-1", "owner", "reservation", "source-marker", "train", True)

    class Hook:
        def __init__(self):
            self.pre = []

        def capture_pre_decision(self, observation, advice, _context, _teacher):
            self.pre.append((observation.decision_ts_ns, advice.intent))
            close_ts = observation.decision_ts_ns - 60_000_000_000
            return TeacherPreDecision(
                teacher_id=advice.expert_id,
                recipe_id="route_a_1m_multiteacher_v1",
                fold_id=observation.fold_id,
                source_identity=observation.source_id,
                decision_ts_ns=observation.decision_ts_ns,
                latest_1m_close_ts_ns=close_ts,
                latest_1m_available_at_ns=observation.decision_ts_ns - 1,
                bar_status="READY",
                indicator_values={},
                raw_signal="NONE",
                raw_signal_reason="NO_SIGNAL",
                position_before=observation.account_value("net_exposure"),
                native_pending_orders=len(observation.pending_order_ids),
                risk_veto_reason=None,
                order_state_veto_reason="PENDING_ACK",
                exit_policy_id="X1_TEST",
                scenario_id="base",
                economic_status="SCENARIO_ONLY",
                block_id="synthetic-block",
            )

        def capture_post_decision(self, *_args):
            pytest.fail("post-decision native facts were requested before env.step")

        def build_native_cases(self, *_args):
            pytest.fail("native cases cannot be built before native order reconciliation")

        def commit(self, *_args):
            pytest.fail("research output cannot be committed before M2.2 reconciliation")

    hook = Hook()

    def host_rollout(_factory, _predict, *, predictor_factory, **_kwargs):
        predictor_factory(env, {})(None)
        raise StopAfterPrediction

    with pytest.raises(StopAfterPrediction):
        provider.rollout_with_m22_trace(
            factory=SimpleNamespace(
                transition_hash="exec-v1",
                execution=SimpleNamespace(zero_target_policy="close"),
                make_env=lambda: env,
            ),
            host_rollout=host_rollout, grant=grant, seed=7,
            teacher_funnel_hook=hook,
        )
    assert hook.pre == [(observations[0].decision_ts_ns, "FLAT")]


def test_candidate_provider_binds_registered_source_feature_spec_and_snapshot():
    from rles_route_a.integration.l2_adapter import CAUSAL_MTF_V7_ROUTE_A_FEATURES
    from rles_route_a.teacher_research import CausalOpportunityGenerator

    class StopAfterPrediction(Exception):
        pass

    context = _context()
    context["market"]["breakout_1h"] = _feature(.02, context["decision_ts_ns"] - 1)
    for host_name in CAUSAL_MTF_V7_ROUTE_A_FEATURES.values():
        context["market"].setdefault(
            host_name, _feature(.001, context["decision_ts_ns"] - 2)
        )
    observed = []
    env = _TerminalEnv()
    provider = NativeGymTeacherProvider(
        env_factory=lambda: env,
        teacher=SimpleNamespace(advise=lambda obs: observed.append(obs) or ExpertAdvice(
            "candidate", obs.decision_ts_ns, "FLAT", 0.0, ("audit",),
        )),
        fold_id="fold-1", execution_recipe="exec-v1", source_identity="source-marker",
        native_order_facts_verifier=lambda _info, _source, _facts: True,
        no_external_cashflow_receipt=lambda _info: "receipt",
        no_external_cashflow_verifier=lambda _info, _source, _receipt: True,
        episode_id=lambda _env, _info: "episode-candidate",
        visible_context=lambda _env: context,
        market_feature_map=CAUSAL_MTF_V7_ROUTE_A_FEATURES,
        opportunity_generator=CausalOpportunityGenerator(),
    )
    factory = SimpleNamespace(
        transition_hash="exec-v1",
        execution=SimpleNamespace(zero_target_policy="close"),
        make_env=lambda: env,
        features=SimpleNamespace(hash="c" * 64),
    )
    grant = HostTrainGrant("fold-1", "owner", "reservation", "source-marker", "train", True)

    def host_rollout(_factory, _predict, *, predictor_factory, **_kwargs):
        predictor_factory(env, {})(None)
        raise StopAfterPrediction

    with pytest.raises(StopAfterPrediction):
        provider.rollout_with_m22_trace(
            factory=factory, host_rollout=host_rollout, grant=grant, seed=11,
        )

    observation, = observed
    candidates = CausalOpportunityGenerator().candidates(observation)
    assert candidates and candidates[0].family == "breakout"
    assert observation.source_id == "source-marker"
    assert observation.feature_spec_hash == "c" * 64
    assert len(observation.snapshot_id) == 64


def test_native_p0_teacher_receives_candidate_provenance_without_audit_sink():
    from rles_route_a.integration.l2_adapter import CAUSAL_MTF_V7_ROUTE_A_FEATURES
    from rles_route_a.teachers.causal_position import CausalPositionTeacher

    class StopAfterPrediction(Exception):
        pass

    context = _context()
    context["account"]["net_exposure"] = 0.0
    context["pending_order_ids"] = ()
    context["market"]["breakout_1h"] = _feature(.02, context["decision_ts_ns"] - 1)
    for host_name in CAUSAL_MTF_V7_ROUTE_A_FEATURES.values():
        context["market"].setdefault(
            host_name, _feature(.001, context["decision_ts_ns"] - 2)
        )
    env = _TerminalEnv()
    position_teacher = CausalPositionTeacher()
    observed = []
    advice_rows = []

    def advise(observation):
        observed.append(observation)
        result = position_teacher.advise(observation)
        advice_rows.append(result)
        return result

    provider = NativeGymTeacherProvider(
        env_factory=lambda: env, teacher=SimpleNamespace(advise=advise),
        fold_id="fold-1", execution_recipe="exec-v1", source_identity="source-marker",
        native_order_facts_verifier=lambda _info, _source, _facts: True,
        no_external_cashflow_receipt=lambda _info: "receipt",
        no_external_cashflow_verifier=lambda _info, _source, _receipt: True,
        episode_id=lambda _env, _info: "episode-p0",
        visible_context=lambda _env: context,
        market_feature_map=CAUSAL_MTF_V7_ROUTE_A_FEATURES,
    )
    factory = SimpleNamespace(
        transition_hash="exec-v1",
        execution=SimpleNamespace(zero_target_policy="close"),
        make_env=lambda: env,
        features=SimpleNamespace(hash="d" * 64),
    )
    grant = HostTrainGrant("fold-1", "owner", "reservation", "source-marker", "train", True)
    action = []

    def host_rollout(_factory, _predict, *, predictor_factory, **_kwargs):
        action.extend(predictor_factory(env, {})(None))
        raise StopAfterPrediction

    with pytest.raises(StopAfterPrediction):
        provider.rollout_with_m22_trace(
            factory=factory, host_rollout=host_rollout, grant=grant, seed=13,
        )
    observation, = observed
    assert observation.source_id == "source-marker"
    assert observation.feature_spec_hash == "d" * 64
    assert observation.market["breakout_1h"] == .02
    assert observation.market["trend_1d"] > 0
    assert position_teacher.generator.candidates(observation)
    assert advice_rows[0].reason_codes == ("CANDIDATE_BREAKOUT",)
    assert action == [.3]


def test_m22_student_rollout_uses_the_frozen_bc_policy_instead_of_teacher():
    class StopAfterPrediction(Exception):
        pass

    env = _TerminalEnv()
    teacher = SimpleNamespace(advise=lambda _obs: pytest.fail("student rollout called teacher"))
    provider = NativeGymTeacherProvider(
        env_factory=lambda: env, teacher=teacher, fold_id="fold-1",
        execution_recipe="exec-v1", source_identity="source-marker",
        native_order_facts_verifier=lambda _info, _source, _facts: True,
        no_external_cashflow_receipt=lambda _info: "receipt",
        no_external_cashflow_verifier=lambda _info, _source, _receipt: True,
        episode_id=lambda _env, _info: "student-episode",
    )
    policy = BCInferencePolicy(
        BCGRUPolicy(("market.trend_1h",), hidden_size=8, layers=1),
        BCConfig(feature_names=("market.trend_1h",), hidden_size=8, layers=1),
    )
    seen = []
    policy.predict_exposure = lambda observation: seen.append(observation) or 0.375
    grant = HostTrainGrant("fold-1", "owner", "reservation", "source-marker", "train", True)

    def host_rollout(_factory, _predict, *, predictor_factory, **_kwargs):
        predict = predictor_factory(env, {})
        assert predict(None) == [0.375]
        raise StopAfterPrediction

    with pytest.raises(StopAfterPrediction):
        provider.rollout_with_m22_trace(
            factory=SimpleNamespace(
                transition_hash="exec-v1",
                execution=SimpleNamespace(zero_target_policy="close"),
                make_env=lambda: env,
            ),
            host_rollout=host_rollout, grant=grant, seed=19,
            decision_policy=policy,
        )

    assert len(seen) == 1
    assert seen[0].episode_id == "student-episode"


def test_prepared_factory_requires_explicit_close_policy_before_registry_read():
    factory = SimpleNamespace(
        make_env=lambda: None, transition_hash="recipe-v1",
        execution=SimpleNamespace(zero_target_policy="deadband"),
        application_registered_train_read=object(),
        application_registered_train_read_verifier=lambda _read: pytest.fail(
            "execution policy must be checked before the Train reader receipt"),
    )
    with pytest.raises(PermissionError, match="zero_target_policy='close'"):
        NativeGymTeacherProvider.from_prepared_factory(
            factory, fold_id="fold-1", teacher=SimpleNamespace(advise=lambda _obs: None),
            native_order_facts_verifier=lambda _info, _source, _facts: True,
            no_external_cashflow_receipt=lambda _info: "receipt",
            no_external_cashflow_verifier=lambda _info, _source, _receipt: True,
            episode_id=lambda _env, _info: "episode-1",
        )


@pytest.mark.parametrize(
    ("native_order_verified", "cashflow_verified", "message"),
    ((False, True, "independent M2.2"), (True, False, "L6 authority")),
)
def test_nonterminal_rollout_requires_independent_m2_and_l6_verification(
        native_order_verified, cashflow_verified, message):
    env = _NonterminalEnv()
    observed_actions = []

    class Teacher:
        def advise(self, observation):
            return ExpertAdvice(
                "trend", observation.decision_ts_ns, "HOLD", 0.0, ("native",),
            )

        def observe_native_action(self, *args):
            observed_actions.append(args)

    provider = NativeGymTeacherProvider(
        env_factory=lambda: env, teacher=Teacher(), fold_id="fold-1",
        execution_recipe="exec-v1", source_identity="source-marker",
        native_order_facts_verifier=lambda _info, _source, _facts: native_order_verified,
        no_external_cashflow_receipt=lambda _info: "unverified-no-flow",
        no_external_cashflow_verifier=lambda _info, _source, _receipt: cashflow_verified,
        episode_id=lambda _env, _info: "episode-1",
    )
    grant = HostTrainGrant("fold-1", "owner", "reservation", "source-marker", "train", True)

    with pytest.raises(PermissionError, match=message):
        tuple(provider.rollout(fold_id="fold-1", grant=grant, seed=7))
    assert env.closed
    assert observed_actions == []


def test_terminal_rollout_requires_l6_to_verify_settlement_receipt():
    env = _TerminalEnv()
    teacher = SimpleNamespace(advise=lambda obs: ExpertAdvice(
        "trend", obs.decision_ts_ns, "OPEN_LONG", 0.2, ("native",),
    ))
    receipt = TerminalL6Receipt(
        receipt_id="unverified-terminal", source_identity="source-marker",
        settlement_sha256="a" * 64, policy_barrier_ts_ns=100,
        settlement_barrier_ts_ns=100, equity_before=1000.,
        policy_equity_after=1000., settled_equity_after=1000., currency="USDT",
        gym_recorded_r_econ=0., recorded_r_econ=0.,
    )
    provider = NativeGymTeacherProvider(
        env_factory=lambda: env, teacher=teacher, fold_id="fold-1",
        execution_recipe="exec-v1", source_identity="source-marker",
        native_order_facts_verifier=lambda _info, _source, _facts: True,
        no_external_cashflow_receipt=lambda _info: "no-flow-receipt",
        no_external_cashflow_verifier=lambda _info, _source, _receipt: True,
        episode_id=lambda _env, _info: "episode-1",
        terminal_receipt_provider=lambda _info, _source: receipt,
        terminal_receipt_verifier=lambda _info, _source, _receipt: False,
    )
    grant = HostTrainGrant("fold-1", "owner", "reservation", "source-marker", "train", True)

    with pytest.raises(PermissionError, match="L6 authority did not verify terminal"):
        tuple(provider.rollout(fold_id="fold-1", grant=grant, seed=7))
    assert env.closed


def test_shared_full_native_snapshot_views_preserve_exact_barrier_facts():
    order_before = {
        "client_order_id": "RLES-order-1", "status": "ACCEPTED", "side": "BUY",
        "quantity": "1", "filled_qty": "0", "instrument_id": "ETHUSDT-PERP.SIM",
    }
    order_after = {**order_before, "status": "FILLED", "filled_qty": "1"}
    order_versions = {"RLES-order-1": ([0, 2], [order_before, order_after])}
    before_orders = _VersionedNativeOrdersView(order_versions, 1)
    after_orders = _VersionedNativeOrdersView(order_versions, 2)
    fill_history = _AppendOnlyNativeHistory()
    fill_start = len(fill_history.rows)
    fill = {
        "event_id": "fill-1", "client_order_id": "RLES-order-1", "ts_event": 20,
        "order_side": "BUY", "last_qty": "1", "last_px": "100",
        "commission": "0.1 USDT",
    }
    fill_history.append(fill, timestamp_ns=20)
    fill_view = fill_history.view()

    assert _snapshot_status_rows(
        SimpleNamespace(all_orders=before_orders), {"RLES-order-1"}, label="pre-barrier",
    ) == ({
        "client_order_id": "RLES-order-1", "status": "ACCEPTED", "side": "BUY",
        "quantity": "1", "filled_qty": "0",
    },)
    assert _snapshot_status_rows(
        SimpleNamespace(all_orders=after_orders), {"RLES-order-1"}, label="post-barrier",
    )[0]["status"] == "FILLED"
    assert list(fill_history.view(end=fill_start)) == []
    assert _snapshot_fills(
        SimpleNamespace(fills=fill_view), start_ns=10, end_ns=20,
    ) == (fill,)
    assert _snapshot_fills(
        SimpleNamespace(fills=fill_view), start_ns=20, end_ns=30,
    ) == ()


def test_ema_reverse_lifecycle_completes_through_native_gym_provider_feedback():
    from rles_route_a.indicators_1m import MINUTE_NS
    from rles_route_a.teachers_1m import build_teacher_1m

    contexts = []
    for minute, fast, slow, exposure, pending in (
        (1, 99.0, 100.0, 0.0, ()),
        (2, 101.0, 100.0, 0.0, ()),
        (3, 99.0, 100.0, 0.3, ()),
        (4, 99.0, 100.0, 0.3, ("close-1",)),
        (5, 99.0, 100.0, 0.0, ()),
    ):
        close_ts = minute * MINUTE_NS
        decision_ts = close_ts + 1_000_000_000
        context = _context(decision_ts)
        context["source_id"] = "source-marker"
        context["market"] = {
            "ema_fast_1m": _feature(fast, close_ts),
            "ema_slow_1m": _feature(slow, close_ts),
        }
        context["exact_market_clocks_ns"] = (
            ("latest_1m_close_ts_ns", close_ts),
        )
        context["account"]["net_exposure"] = exposure
        context["pending_order_ids"] = pending
        context["execution"]["pending_order_count"] = float(len(pending))
        contexts.append(context)

    class ReversalEnv:
        def __init__(self):
            self.index = 0
            self.statuses = {}
            self.closed = False

        def reset(self, *, seed):
            self.seed = seed
            self.index = 0
            self.statuses = {}
            return None, {}

        def rles_visible_context_inputs(self):
            return contexts[self.index]

        def step(self, action):
            index = self.index
            expected_targets = (0.0, 0.3, 0.0, 0.3, -0.3)
            assert action[0] == pytest.approx(expected_targets[index])

            before = tuple(
                dict(self.statuses[key]) for key in sorted(self.statuses)
            )
            planner_actions = ("NOOP", "SUBMIT", "SUBMIT", "WAIT", "SUBMIT")
            planner_action = planner_actions[index]
            client_order_id = None
            related_order_ids = ()
            fills = ()
            pending_after = ()
            next_exposure = (0.0, 0.3, 0.3, 0.0, -0.3)[index]
            if index == 1:
                client_order_id = "long-1"
                self.statuses[client_order_id] = {
                    "client_order_id": client_order_id, "status": "FILLED",
                    "side": "BUY", "quantity": "0.3", "filled_qty": "0.3",
                }
                fills = ({
                    "event_id": "fill-long-1", "client_order_id": client_order_id,
                    "ts_event": contexts[index]["decision_ts_ns"] + 1,
                    "last_qty": "0.3", "last_px": "100", "commission": "0 USDT",
                },)
            elif index == 2:
                client_order_id = "close-1"
                self.statuses[client_order_id] = {
                    "client_order_id": client_order_id, "status": "NEW",
                    "side": "SELL", "quantity": "0.3", "filled_qty": "0",
                }
                pending_after = ("close-1",)
            elif index == 3:
                related_order_ids = ("close-1",)
                self.statuses["close-1"] = {
                    "client_order_id": "close-1", "status": "FILLED",
                    "side": "SELL", "quantity": "0.3", "filled_qty": "0.3",
                }
                fills = ({
                    "event_id": "fill-close-1", "client_order_id": "close-1",
                    "ts_event": contexts[index]["decision_ts_ns"] + 1,
                    "last_qty": "0.3", "last_px": "100", "commission": "0 USDT",
                },)
            elif index == 4:
                client_order_id = "short-1"
                self.statuses[client_order_id] = {
                    "client_order_id": client_order_id, "status": "FILLED",
                    "side": "SELL", "quantity": "0.3", "filled_qty": "0.3",
                }
                fills = ({
                    "event_id": "fill-short-1", "client_order_id": client_order_id,
                    "ts_event": contexts[index]["decision_ts_ns"] + 1,
                    "last_qty": "0.3", "last_px": "100", "commission": "0 USDT",
                },)

            after = tuple(
                dict(self.statuses[key]) for key in sorted(self.statuses)
            )
            status_receipt = sha256(json.dumps(
                {"before": [dict(row) for row in before],
                 "after": [dict(row) for row in after]},
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest()
            decision_ts = contexts[index]["decision_ts_ns"]
            info = {
                "decision_ts_ns": decision_ts + MINUTE_NS,
                "terminal_settled": False,
                "execution_plan_action": planner_action,
                "execution_client_order_id": client_order_id,
                "execution_related_order_ids": related_order_ids,
                "native_pending_order_ids_before": contexts[index]["pending_order_ids"],
                "native_pending_order_ids_after": pending_after,
                "native_policy_fills": fills,
                "native_order_statuses_before": before,
                "native_order_statuses_after": after,
                "native_order_status_receipt": status_receipt,
                "execution_decision_id": f"decision-{index}",
                "execution_plan_id": f"plan-{index}",
                "execution_idempotency_key": f"idempotency-{index}",
                "execution_plan_reason": "synthetic-provider-lifecycle",
                "evidence_hash": "e" * 64,
                "policy_equity_currency": "USDT",
                "requested_action": action[0],
                "risk_soft_target": action[0],
                "risk_adjusted_target": action[0],
                "quantity_rounded_target_exposure": action[0],
                "policy_executed_action_pre_terminal": next_exposure,
                "policy_equity_before": "1000",
                "policy_equity_after": "1000",
                "policy_equity_change": "0",
                "r_econ": 0.0,
                "execution_cancel_order_ids": (),
            }
            self.index += 1
            return None, 0.0, False, False, info

        def close(self):
            self.closed = True

    env = ReversalEnv()
    teacher = build_teacher_1m(
        "T01_EMA_9_21", reversal_mode="REVERSE_AFTER_FLAT_V1",
    )
    teacher_actions = []
    native_feedback = []
    advise = teacher.advise
    observe = teacher.observe_native_action

    def record_advice(observation):
        action = advise(observation)
        teacher_actions.append(action)
        return action

    def record_feedback(action, trace, order_facts):
        native_feedback.append((action, trace, order_facts))
        observe(action, trace, order_facts)

    teacher.advise = record_advice
    teacher.observe_native_action = record_feedback
    provider = NativeGymTeacherProvider(
        env_factory=lambda: env, teacher=teacher, fold_id="fold-1",
        execution_recipe="exec-v1", source_identity="source-marker",
        native_order_facts_verifier=lambda _info, _source, _facts: True,
        no_external_cashflow_receipt=lambda _info: "synthetic-no-flow-receipt",
        no_external_cashflow_verifier=lambda _info, _source, _receipt: True,
        episode_id=lambda _env, _info: "episode-1",
        market_feature_map={
            "ema_9": "ema_fast_1m", "ema_21": "ema_slow_1m",
        },
    )
    grant = HostTrainGrant(
        "fold-1", "owner", "reservation", "source-marker", "train", True,
    )

    rollout = provider.rollout(fold_id="fold-1", grant=grant, seed=7)
    barriers = [next(rollout) for _ in contexts]
    rollout.close()

    assert env.closed
    assert [action.intent for action in teacher_actions] == [
        "FLAT", "OPEN_LONG", "EXIT", "HOLD", "OPEN_SHORT",
    ]
    assert teacher_actions[2].reason_codes[-1] == "EMA_REVERSE_CROSS"
    assert teacher_actions[3].reason_codes == ("PENDING_ACK",)
    assert teacher_actions[4].reason_codes == ("EMA_REVERSAL_RECHECK",)
    assert barriers[2].trace.submitted_order_ids == ("close-1",)
    assert barriers[3].trace.related_existing_order_ids == ("close-1",)
    assert barriers[3].trace.filled_order_ids == ("close-1",)
    assert len(native_feedback) == len(contexts)
    assert all(barrier.trace.scenario_only for barrier in barriers)
