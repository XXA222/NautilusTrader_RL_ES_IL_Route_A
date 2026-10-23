"""High-impact edge-case regression tests for Route A v0.2 hardening."""
from __future__ import annotations
from dataclasses import replace
from math import isfinite, log
import pytest
import torch
from rles_route_a.contracts import PointInTimeObservation, ExpertAdvice, NativeActionTrace
from rles_route_a.config import BCConfig
from rles_route_a.datasets.sequence_dataset import EpisodeSequenceDataset, LabeledStep
from rles_route_a.datasets.daggerset import DAggerSet, AggregatedLabel
from rles_route_a.datasets.transitions import OfflineTransition, OfflineBatch
from rles_route_a.models.bc import BCGRUPolicy, bc_losses
from rles_route_a.training.bc_trainer import train_bc
from rles_route_a.training.offline_common import gamma_factor
from rles_route_a.training.iql_trainer import IQLHyper
from rles_route_a.training.td3bc_trainer import TD3BCHyper
from rles_route_a.training.factories import new_bc_policy, new_iql, new_td3bc
from rles_route_a.training.offline_train_loop import OfflineReadiness, fit_offline
from rles_route_a.integration.native_rollout import native_teacher_step, TeacherFact
from rles_route_a.integration.native_economics import NativeRewardReceipt
from rles_route_a.integration.trace_audit import AuditedTeacherBarrier, audit_teacher_barriers
from rles_route_a.export.bundle import save_bc_bundle
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.teachers.breakout import BreakoutTeacher
from rles_route_a.teachers.mean_reversion import MeanReversionTeacher
from rles_route_a.teachers.trend import TrendTeacher
from rles_route_a.teachers.selector import FrozenTeacherSelector
from rles_route_a.teachers.wait import WaitTeacher
from rles_route_a.teachers.recovery import RecoveryTeacher

NAMES = ("market.trend_1h", "account.net_exposure")


def label(o, target=0., *, status="FILLED", episode_id=None):
    a = ExpertAdvice("test", o.decision_ts_ns, "EXIT" if target == 0 else "HOLD", target, ("TEST",))
    return LabeledStep(o,a,target,action_status=status,source_identity="registered-source")


def test_observation_copies_mapping_sources(observed):
    market={"price_1m":1.}
    ts={"price_1m":100}
    account={"net_exposure":0.}
    obs=PointInTimeObservation("ep", 200, market, ts, account,fold_id="f1")
    market["price_1m"]=100.;ts["price_1m"]=999999999999
    account["net_exposure"]=-1.
    assert obs.market_value("price_1m")==1.
    assert obs.feature_available_at_ns["price_1m"]==100
    assert obs.account_value("net_exposure")==0.
    with pytest.raises(TypeError):
        obs.account["net_exposure"]=1.


def test_gap_resets_history_and_inference(observed):
    t0=2_000_000_000
    steps=[label(observed(ts=t0,episode="ep")),
           label(observed(ts=t0+60,episode="ep")),
           label(observed(ts=t0+600,episode="ep"))]
    ds=EpisodeSequenceDataset(steps,feature_names=NAMES,context_bars=3,max_gap_ns=100)
    assert [ds[i]["length"].item() for i in range(3)]==[1,2,1]
    cfg=BCConfig(feature_names=NAMES,context_bars=3,max_gap_ns=100,hidden_size=8,layers=1,epochs=1,batch_size=2)
    policy=BCInferencePolicy(new_bc_policy(cfg),cfg)
    policy.reset("ep")
    for step in steps:
        policy.predict_exposure(step.observation)
    assert len(policy._series)==1


def test_bc_supervision_masks_unexecuted_and_rejects_all_blocked(observed):
    o=observed()
    blocked=label(o,0.,status="WAIT")
    assert blocked.supervision_weight==0
    with pytest.raises(ValueError,match="unexecuted"):
        replace(blocked, supervision_weight=1.)
    logits=torch.randn(2,7,requires_grad=True)
    outputs=torch.tensor([.2,-.5],requires_grad=True)
    intent=torch.tensor([0,1]);target=torch.tensor([0.,.5])
    masked=bc_losses(logits, outputs, target_intent=intent,target_exposure=target,
                     sample_weight=torch.tensor([1.,0.]))
    only=bc_losses(logits[:1],outputs[:1],target_intent=intent[:1],target_exposure=target[:1])
    assert torch.allclose(masked["loss"],only["loss"])
    masked["loss"].backward()
    assert torch.allclose(outputs.grad[1:],torch.zeros(1))
    ds=EpisodeSequenceDataset([blocked],feature_names=NAMES,context_bars=2)
    cfg=BCConfig(feature_names=NAMES,context_bars=2,hidden_size=8,layers=1,epochs=1)
    with pytest.raises(ValueError,match="no teacher actions"):
        train_bc(BCGRUPolicy(NAMES,8,1),ds,cfg)


def test_noop_blocked_exit_not_taught_as_correct_position(observed):
    obs=observed(pos=.5)
    advice=WaitTeacher().advise(obs)
    trace=NativeActionTrace(obs.decision_ts_ns,0.,0.,0.,.5,1000.,1000.,"NOOP")
    step=native_teacher_step(TeacherFact(obs,advice,trace))
    assert step.supervision_weight==0 and step.executable_exposure==.5


def test_dagger_atomic_duplicate_and_trajectory_isolation(observed):
    base_obs=observed(episode="shared",ts=1_000_000_000)
    base=label(base_obs)
    data=DAggerSet((base,))
    student_obs=observed(episode="shared",ts=2_000_000_000,pos=-.5)
    candidate=AggregatedLabel(1,"student",label(student_obs,-.2))
    with pytest.raises(ValueError,match="duplicate"):
        data.extend((candidate,candidate))
    assert data.collected==0
    data.append(candidate)
    data.append(AggregatedLabel(1,"student",label(observed(episode="shared",ts=3_000_000_000),0.)))
    allsteps=data.training_steps()
    ds=EpisodeSequenceDataset(allsteps,feature_names=NAMES,context_bars=4)
    assert ds[1]["length"].item()==1
    assert ds[2]["length"].item()==2
    assert ds[1]["x"][-1,1].item()==pytest.approx(-.5)


@pytest.mark.parametrize("factory",[
    lambda:BreakoutTeacher(threshold=float('nan')),
    lambda:MeanReversionTeacher(size=float('inf')),
    lambda:TrendTeacher(threshold=float('nan')),
    lambda:RecoveryTeacher(stop_loss_fraction=float('inf')),
    lambda:FrozenTeacherSelector(WaitTeacher(), RecoveryTeacher(), max_active_vol=float('nan')),
])
def test_nonfinite_teacher_settings_rejected(factory):
    with pytest.raises(ValueError): factory()


def test_seeded_factories_are_reproducible_and_gru_isolated():
    cfg=BCConfig(feature_names=NAMES,hidden_size=16,layers=1,seed=991)
    p1=new_bc_policy(cfg)
    _=torch.randn(100)
    p2=new_bc_policy(cfg)
    assert all(torch.equal(a,b) for a,b in zip(p1.parameters(),p2.parameters()))
    p3=new_bc_policy(replace(cfg,seed=992))
    assert any(not torch.equal(a,b) for a,b in zip(p1.parameters(),p3.parameters()))
    t1=new_td3bc(2,TD3BCHyper(hidden_size=8),seed=47)
    t2=new_td3bc(2,TD3BCHyper(hidden_size=8),seed=47)
    assert all(torch.equal(a,b) for a,b in zip(t1.actor.parameters(),t2.actor.parameters()))
    i1=new_iql(2,IQLHyper(hidden_size=8),seed=47)
    i2=new_iql(2,IQLHyper(hidden_size=8),seed=47)
    assert all(torch.equal(a,b) for a,b in zip(i1.actor.parameters(),i2.actor.parameters()))


def test_gamma_time_gap_not_silent_one_step():
    one={"r":torch.zeros(2,1,dtype=torch.float32),
         "discount_steps":torch.tensor([[1.],[5.]],dtype=torch.float32)}
    factors=gamma_factor(one,0.9)
    assert factors[0].item()==pytest.approx(.9)
    assert factors[1].item()==pytest.approx(.9**5)


def transition(i,*,fold="train2026",recipe="exec2"):
    return OfflineTransition(
        (i/100.,.2), 0.4,-0.0002,((i+1)/100.,.2),False,False,True,
        f"native/{fold}/{i}",fold_id=fold,episode_id="ep",decision_ts_ns=1000+i*60,
        next_decision_ts_ns=1060+i*60,native_execution_recipe=recipe,discount_steps=1,
        action_trace=NativeActionTrace(1000+i*60,.4,.4,.4,.2,1000.,999.,"NOOP"),
        executed_target=.4, source_identity="source-2026", frozen_encoder_id="bc-encoder-v1")


def test_strict_native_offline_fold_and_recipe_receipts():
    data=tuple(transition(i) for i in range(12))
    gate=OfflineReadiness(True,True,True,True,"exec2","receipt",registered_fold_id="train2026",
        reward_scale=1000.,registered_source_identity="source-2026",
        frozen_encoder_identity="bc-encoder-v1")
    learner=new_iql(2,IQLHyper(hidden_size=12),seed=31)
    authorize=lambda _readiness,_rows: True
    stats=fit_offline(learner,data,readiness=gate,updates=3,batch_size=6,seed=31,
                      readiness_verifier=authorize)
    assert len(stats)==3 and all(isfinite(s["actor_loss"]) for s in stats)
    with pytest.raises(PermissionError,match="Fold"):
        fit_offline(learner,data[:-1]+(transition(11,fold="other"),),
                    readiness=gate,updates=1,batch_size=4,seed=31,readiness_verifier=authorize)
    with pytest.raises(PermissionError,match="execution provenance"):
        fit_offline(learner,data[:-1]+(transition(11,recipe="evil"),),
                    readiness=gate,updates=1,batch_size=4,seed=31,readiness_verifier=authorize)
    with pytest.raises(PermissionError,match="action mapping"):
        fit_offline(learner,data[:-1]+(replace(data[-1],action_trace=None),),
                    readiness=gate,updates=1,batch_size=4,seed=31,readiness_verifier=authorize)
    with pytest.raises(PermissionError,match="Fold"):
        fit_offline(learner,data,readiness=replace(gate,registered_fold_id="other"),
                    updates=1,batch_size=4,seed=31,readiness_verifier=authorize)


def test_reward_float32_overflow_blocked():
    sample=OfflineTransition((1e100,),0.,0.,(0.,),False,False,True,"x",executed_target=0.)
    with pytest.raises(ValueError,match="float32"):
        OfflineBatch([sample])
    sample=OfflineTransition((0.,),0.,1e100,(0.,),False,False,True,"x",executed_target=0.)
    with pytest.raises(ValueError,match="scaled economic"):
        OfflineBatch([sample])


def test_offline_boundary_log_prob_finite():
    p=new_iql(2,IQLHyper(hidden_size=8),seed=7)
    x=torch.randn(4,2)
    a=torch.tensor([[-1.],[1.],[-.5],[.5]])
    lp=p.actor.log_prob(x,a)
    assert lp.shape==(4,1) and bool(torch.isfinite(lp).all())


def test_native_audit_equity_and_teacher_provenance(observed):
    ts=2_000_000_000
    o=observed(ts=ts,episode="episode")
    advice=WaitTeacher().advise(o)
    r=NativeRewardReceipt(100.,99.,log(99./100.),False)
    native=NativeActionTrace(ts,0.,0.,0.,0.,100.,99.,"NOOP")
    case=AuditedTeacherBarrier(o,advice,native,r,"recipeA","sourceA")
    labels,summary=audit_teacher_barriers([case])
    # This synthetic legacy trace has no end-time/source-bound case facts, so
    # its reward cannot qualify it as a positive native teaching example.
    assert summary.decisions==1 and summary.executable_label_count==0
    assert labels[0].economic_case_status == "UNASSESSED"
    assert labels[0].advice.requested_exposure==0
    broken=replace(case,reward=NativeRewardReceipt(100.,99.,42.,False))
    with pytest.raises(ValueError,match="conservation"):
        audit_teacher_barriers([broken])


def test_weight_bundle_rejects_nan(tmp_path):
    cfg=BCConfig(feature_names=NAMES,hidden_size=8,layers=1)
    model=new_bc_policy(cfg)
    with torch.no_grad():
        next(model.parameters()).fill_(float('nan'))
    with pytest.raises(ValueError,match="nonfinite"):
        save_bc_bundle(model,cfg,tmp_path/"no-nan")


def test_observation_and_train_dataset_are_spawn_pickleable(observed):
    import pickle
    o=observed()
    encoded=pickle.dumps(o, protocol=4)
    decoded=pickle.loads(encoded)
    assert decoded.market_value("trend_1h")==o.market_value("trend_1h")
    ds=EpisodeSequenceDataset([label(o)],feature_names=NAMES,context_bars=2)
    rebuilt=pickle.loads(pickle.dumps(ds,protocol=4))
    assert torch.equal(ds[0]["x"],rebuilt[0]["x"])
