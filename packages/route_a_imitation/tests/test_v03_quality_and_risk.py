from dataclasses import replace
import pytest
import torch
from rles_route_a.config import BCConfig
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.training.factories import new_bc_policy, new_iql
from rles_route_a.training.offline_train_loop import OfflineReadiness, fit_offline
from rles_route_a.training.iql_trainer import IQLHyper
from rles_route_a.datasets.transitions import OfflineTransition
from rles_route_a.contracts import ExpertAdvice, NativeActionTrace
from rles_route_a.evaluation.diagnostics import (
    OfflinePolicyActionQuery,
    audit_action_identifiability,
    audit_offline_policy_action_support,
)
from rles_route_a.evaluation.outcome_supervision import (
    build_outcome_supervision_arms,
    outcome_ablation_sampler,
)
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


def test_offline_policy_action_support_is_state_conditioned_and_stratified():
    train=(transition(0),transition(1))
    query=OfflinePolicyActionQuery(
        state=(.1,.2), action=.9, fold_id="heldout", episode_id="heldout-ep",
        decision_ts_ns=500_000_000_000, source_identity="source",
        frozen_encoder_id="encoder-v1", execution_recipe="exec",
        stratum="volatile|long|held-30m",
    )
    report=audit_offline_policy_action_support(
        train,(query,),training_strata=("volatile|long|held-30m",)*2,
        max_state_distance=.1,action_tolerance=.05,neighbors=2,
    )
    assert report["status"]=="DIAGNOSTIC_ONLY"
    assert report["state_supported_fraction"]==1.
    assert report["action_supported_fraction"]==0.
    assert report["strata"]["volatile|long|held-30m"]["mean_nearest_action_gap"]==pytest.approx(.6)

    unsupported=replace(query,stratum="calm|short|flat")
    no_stratum=audit_offline_policy_action_support(
        train,(unsupported,),training_strata=("volatile|long|held-30m",)*2,
        max_state_distance=.1,action_tolerance=.05,
    )
    assert no_stratum["state_supported_fraction"]==0.
    assert no_stratum["action_supported_fraction"]==0.


def test_offline_policy_action_support_rejects_episode_leak_and_mixed_provenance():
    train=(transition(0),transition(1))
    query=OfflinePolicyActionQuery(
        state=(.1,.2),action=.3,fold_id="heldout",episode_id="ep",
        decision_ts_ns=500_000_000_000,source_identity="source",
        frozen_encoder_id="encoder-v1",execution_recipe="exec",stratum="state-1",
    )
    with pytest.raises(PermissionError,match="episodes disjoint"):
        audit_offline_policy_action_support(
            train,(query,),training_strata=("state-1",)*2,
            max_state_distance=1.,action_tolerance=.1,
        )
    separated=replace(query,episode_id="heldout-ep",execution_recipe="other")
    with pytest.raises(PermissionError,match="mixes Fold, source, encoder, or execution recipe"):
        audit_offline_policy_action_support(
            train,(separated,),training_strata=("state-1",)*2,
            max_state_distance=1.,action_tolerance=.1,
        )


def test_teacher_must_make_absolute_native_profit_not_merely_lose_less():
    records=[NativePair(w,s,3,"expert","baseline","source",-.01,-.10,.10,.20,4,0,True)
             for w in ("w1","w2") for s in ("base","stress")]
    verdict=screen_teacher(records)
    assert verdict.status=="TRAIN_INNER_SCREEN_REJECT"
    assert "TEACHER_NONPOSITIVE_NET_RETURN" in verdict.reasons


def test_outcome_supervision_arms_preserve_teacher_behavior_and_isolate_counterfactuals(observed):
    positive_case="a"*64
    negative_case="b"*64

    def native_step(index, *, case_id, outcome, exposure, intent, origin="OBSERVED_TEACHER_ACTION",
                    supervision=1., intent_weight=1., reason="native_teacher"):
        o=observed(ts=10_000_000_000+index*1_000_000_000,episode="teacher-ep")
        advice=ExpertAdvice("teacher",o.decision_ts_ns,intent,exposure,(reason,))
        status=("OBSERVED_NATIVE_TRADE_POSITIVE" if outcome>0
                else "OBSERVED_NATIVE_TRADE_NEGATIVE")
        return LabeledStep(
            o,advice,exposure,action_status="FILLED",source_identity="source",
            trajectory_id="trajectory",supervision_weight=supervision,
            intent_supervision_weight=intent_weight,economic_case_status=status,
            case_outcome_bps=outcome,case_end_ts_ns=100_000_000_000,
            case_fact_sha256=case_id,label_origin=origin,
        )

    positive_a=native_step(0,case_id=positive_case,outcome=12.,exposure=.5,
                           intent="OPEN_LONG")
    positive_b=native_step(1,case_id=positive_case,outcome=12.,exposure=.7,
                           intent="ADD")
    negative_observed=native_step(2,case_id=negative_case,outcome=-8.,exposure=.4,
                                 intent="OPEN_LONG")
    negative_avoid=native_step(
        3,case_id=negative_case,outcome=-8.,exposure=0.,intent="FLAT",
        origin="INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE",supervision=0.,
        reason="negative_native_trade_entry_avoidance",
    )
    full,positive_only,positive_plus_avoid=build_outcome_supervision_arms(
        (positive_a,positive_b,negative_observed,negative_avoid)
    )

    assert [arm.name for arm in (full,positive_only,positive_plus_avoid)]==[
        "full_teacher_behavior","positive_native_case_weighted",
        "positive_plus_negative_entry_avoidance",
    ]
    assert all(arm.fold_id=="f1" and arm.source_identity=="source" for arm in
               (full,positive_only,positive_plus_avoid))
    assert [step.advice.intent for step in full.steps[:3]]==["OPEN_LONG","ADD","OPEN_LONG"]
    assert [step.supervision_weight for step in positive_only.steps]==[.5,.5,0.,0.]
    assert [step.intent_supervision_weight for step in positive_only.steps]==[.5,.5,0.,0.]
    assert positive_plus_avoid.steps[2].intent_supervision_weight==0.
    assert positive_plus_avoid.steps[3].supervision_weight==0.
    assert positive_plus_avoid.steps[3].intent_supervision_weight==1.
    assert positive_only.positive_case_count==1 and positive_only.negative_case_count==1
    assert list(outcome_ablation_sampler(range(len(full.steps)),seed=41))==list(
        outcome_ablation_sampler(range(len(positive_plus_avoid.steps)),seed=41)
    )
    with pytest.raises(ValueError,match="conflicting case outcomes"):
        build_outcome_supervision_arms((
            positive_a,replace(positive_b,case_outcome_bps=13.),
            negative_observed,negative_avoid,
        ))
    with pytest.raises(ValueError,match="requires a native negative-entry avoidance"):
        build_outcome_supervision_arms((positive_a,positive_b,negative_observed))


def test_outcome_supervision_ablation_requires_native_positive_case(observed):
    o=observed()
    step=LabeledStep(o,WaitTeacher().advise(o),0.,source_identity="source")
    with pytest.raises(ValueError,match="at least one completed positive native trade"):
        build_outcome_supervision_arms((step,))
