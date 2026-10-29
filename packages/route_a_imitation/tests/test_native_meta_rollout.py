from types import SimpleNamespace

import pytest

from rles_route_a.contracts import NativeActionTrace
from rles_route_a.integration.gym_rollout import NativeGymTeacherProvider
from rles_route_a.integration.host_gym_projection import L3OrderFacts
from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.teacher_research.meta_gated_teacher import MetaGatedTeacher
from rles_route_a.teacher_research.native_meta_rollout import run_registered_meta_rollout
from test_candidate_collector import observation
from test_meta_gated_teacher import Recovery, trained_policy


def test_meta_runner_requires_scope_cost_and_complete_m22_trace(tmp_path):
    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    provider = NativeGymTeacherProvider(
        env_factory=lambda: None, teacher=teacher, fold_id="train", source_identity="synthetic",
        execution_recipe="native", native_order_facts_verifier=lambda *_: True,
        no_external_cashflow_receipt=lambda *_: "receipt",
        no_external_cashflow_verifier=lambda *_: True, episode_id=lambda *_: "e")
    grant = HostTrainGrant("train", "owner", "reservation", "synthetic", "train", True)
    obs = observation(10)
    action = teacher.advise(obs)
    trace = NativeActionTrace(obs.decision_ts_ns, 0., 0., 0., 0., 10000., 10000., "NOOP")
    teacher.observe_native_action(action, trace, L3OrderFacts(native_evidence_receipt="r"))
    provider.rollout_with_m22_trace = lambda **kwargs: ((SimpleNamespace(trace=trace),),
        {"route_a_native_order_verification": {"barriers": 1,
                                               "authority": "retained_m22_session_snapshots"}})
    kwargs = dict(provider=provider, teacher=teacher, factory=object(), host_rollout=lambda: None,
                  grant=grant, verify_grant=lambda _: True, scenario="base", seed=7,
                  rollout_id="synthetic", episode_steps=1, trace_path=tmp_path / "trace.jsonl")
    with pytest.raises(ValueError, match="cost scenario"):
        run_registered_meta_rollout(**{**kwargs, "scenario": "stress"})
    with pytest.raises(PermissionError, match="grant"):
        run_registered_meta_rollout(**{**kwargs, "verify_grant": lambda _: False})
    barriers, native = run_registered_meta_rollout(**kwargs)
    assert len(barriers) == 1 and (tmp_path / "trace.jsonl").exists()
    assert native["route_a_native_order_verification"]["barriers"] == 1
