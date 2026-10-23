from dataclasses import replace
import pytest
import torch
from rles_route_a.config import BCConfig
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.training.factories import new_bc_policy, new_iql
from rles_route_a.training.offline_train_loop import OfflineReadiness, fit_offline
from rles_route_a.training.iql_trainer import IQLHyper
from rles_route_a.datasets.transitions import OfflineTransition
from rles_route_a.contracts import NativeActionTrace
from rles_route_a.evaluation.diagnostics import audit_action_identifiability
from rles_route_a.evaluation.teacher_gate import NativePair,screen_teacher
from rles_route_a.datasets.sequence_dataset import LabeledStep
from rles_route_a.datasets.daggerset import DAggerSet
from rles_route_a.teachers.wait import WaitTeacher
from rles_route_a.training.dagger_collector import DAggerCollector, VisitedStudentState, ProjectedTeacherTarget
from rles_route_a.integration.scope import HostTrainGrant


def test_bc_failed_inference_does_not_consume_timestamp(observed):
    cfg=BCConfig(feature_names=("market.trend_1h","account.net_exposure"),context_bars=2,hidden_size=8,layers=1)
    policy=BCInferencePolicy(new_bc_policy(cfg),cfg)
    o=observed()
    policy.reset(o.episode_id)
    class BrokenModel(torch.nn.Module):
        def forward(self,x,lengths):
            return torch.zeros(1,7),torch.tensor([float("nan")])
    original=policy.model
    policy.model=BrokenModel()
    with pytest.raises(ValueError,match="actor output"):
        policy.predict_exposure(o)
    assert policy._last_ts==-1 and len(policy._series)==0
    policy.model=original
    assert -1 <= policy.predict_exposure(o) <= 1
    assert len(policy._series)==1


def test_dagger_noop_deadband_projection_has_zero_weight(observed):
    o=observed(pos=.5)
    base=LabeledStep(o, WaitTeacher().advise(o),.5, action_status="NOOP",supervision_weight=0.,
                     source_identity="source")
    data=DAggerSet((base,))
    v=observed(ts=2_000_000_000,pos=.4,episode="learner")
    class Provider:
        def collect_train_states(self,*,grant,student_policy_id):
            yield VisitedStudentState(v, NativeActionTrace(v.decision_ts_ns,0.,0.,0.,.4,100.,100.,"NOOP"),
                                      student_policy_id,grant.source_identity)
    class Projector:
        def project_native_target(self,obs,requested_exposure):
            assert requested_exposure == 0.
            return ProjectedTeacherTarget(.4,"exec","NOOP")
    grant=HostTrainGrant("f1","owner","reservation","source","train",True)
    DAggerCollector(WaitTeacher(),Projector(),lambda _grant: True).collect(provider=Provider(),grant=grant,
        student_policy_id="p",iteration=1,aggregate=data)
    label=data.training_steps()[-1]
    assert label.supervision_weight==0 and label.executable_exposure==.4


def transition(i, *, dt=60_000_000_000, discount_steps=1, terminated=False):
    return OfflineTransition((.1,.2),.3,.001,(.2,.3),terminated,False,not terminated,
        f"native-{i}",fold_id="f",episode_id="ep",decision_ts_ns=100_000_000_000+i*dt,
        next_decision_ts_ns=100_000_000_000+(i+1)*dt,discount_steps=discount_steps,
        native_execution_recipe="exec",
        action_trace=NativeActionTrace(100_000_000_000+i*dt,.3,.3,.3,.2,1000.,999.,"NOOP"),
        executed_target=.3, source_identity="source", frozen_encoder_id="encoder-v1")


def readiness(**k):
    return OfflineReadiness(True,True,True,True,"exec","receipt",registered_fold_id="f",
                             decision_period_ns=60_000_000_000,
                             registered_source_identity="source",
                             frozen_encoder_identity="encoder-v1",**k)


def verify_readiness(_readiness, _transitions):
    return True


def test_native_offline_discount_clock_must_match():
    learner=new_iql(2,IQLHyper(hidden_size=8),seed=5)
    good=tuple(transition(i) for i in range(6))
    assert len(fit_offline(learner,good,readiness=readiness(),updates=1,batch_size=3,seed=5,
                           readiness_verifier=verify_readiness))==1
    with pytest.raises(ValueError,match="discount"):
        fit_offline(learner,(transition(0,discount_steps=2),),readiness=readiness(),
                    updates=1,batch_size=1,seed=5,readiness_verifier=verify_readiness)


def test_production_offline_requires_exact_host_verification_and_one_identity():
    learner=new_iql(2,IQLHyper(hidden_size=8),seed=5)
    rows=(transition(0),transition(1))
    with pytest.raises(PermissionError,match="host authority"):
        fit_offline(learner,rows,readiness=readiness(),updates=1,batch_size=1,seed=5)
    with pytest.raises(PermissionError,match="host authority"):
        fit_offline(learner,rows,readiness=readiness(),updates=1,batch_size=1,seed=5,
                    readiness_verifier=lambda _ready,_rows: False)
    mixed=(rows[0],replace(rows[1],provenance="mixed-source",
                           source_identity="other-source"))
    with pytest.raises(PermissionError,match="provenance|mix registered"):
        fit_offline(learner,mixed,readiness=readiness(),updates=1,batch_size=1,seed=5,
                    readiness_verifier=verify_readiness)
    mixed_encoder=(rows[0],replace(rows[1],provenance="mixed-encoder",
                                   frozen_encoder_id="other-encoder"))
    with pytest.raises(PermissionError,match="provenance|mix registered"):
        fit_offline(learner,mixed_encoder,readiness=readiness(),updates=1,batch_size=1,seed=5,
                    readiness_verifier=verify_readiness)


def test_offline_readiness_rejects_truthy_strings_and_boolean_reward_scale():
    with pytest.raises(TypeError,match="strict boolean"):
        replace(readiness(),ra3_recovery_accepted="true").require()
    with pytest.raises(ValueError,match="reward scale"):
        replace(readiness(),reward_scale=True).require()


def test_native_offline_duplicate_time_and_implicit_episode_gap_rejected():
    learner=new_iql(2,IQLHyper(hidden_size=8),seed=5)
    a=transition(0)
    with pytest.raises(ValueError,match="repeated decision"):
        fit_offline(learner,(a,replace(a,provenance="other")),readiness=readiness(),
                    updates=1,batch_size=1,seed=5,readiness_verifier=verify_readiness)
    b=replace(transition(2),decision_ts_ns=220_000_000_000,next_decision_ts_ns=280_000_000_000)
    with pytest.raises(ValueError,match="unobserved episode gap"):
        fit_offline(learner,(a,b),readiness=readiness(),updates=1,batch_size=1,seed=5,
                    readiness_verifier=verify_readiness)


def test_action_identifiability_flags_same_native_outcome_for_distinct_commands():
    a=transition(0)
    b=replace(a,provenance="alias",requested_action=-.3,
        action_trace=NativeActionTrace(a.decision_ts_ns,-.3,.3,.3,.2,1000.,999.,"NOOP"))
    report=audit_action_identifiability((a,b))
    assert report["distinct_command_pairs_compared"]==1
    assert report["aliased_pair_count"]==1
    assert report["status"]=="DIAGNOSTIC_ONLY"

    separated=replace(b,next_state=(.9,.8))
    assert audit_action_identifiability((a,separated))["aliased_pair_count"]==0


def test_teacher_must_make_absolute_native_profit_not_merely_lose_less():
    records=[NativePair(w,s,3,"expert","baseline","source",-.01,-.10,.10,.20,4,0,True)
             for w in ("w1","w2") for s in ("base","stress")]
    verdict=screen_teacher(records)
    assert verdict.status=="TRAIN_INNER_SCREEN_REJECT"
    assert "TEACHER_NONPOSITIVE_NET_RETURN" in verdict.reasons
