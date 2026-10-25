from __future__ import annotations

from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from rles_route_a.contracts import ExpertAdvice
from rles_route_a.config import BCConfig
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.integration.gym_rollout import NativeGymTeacherProvider
from rles_route_a.integration.host_gym_projection import TerminalL6Receipt
from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.models.bc import BCGRUPolicy
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
    teacher = SimpleNamespace(advise=lambda obs: ExpertAdvice(
        "trend", obs.decision_ts_ns, "HOLD", 0.0, ("native",),
    ))
    provider = NativeGymTeacherProvider(
        env_factory=lambda: env, teacher=teacher, fold_id="fold-1",
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
