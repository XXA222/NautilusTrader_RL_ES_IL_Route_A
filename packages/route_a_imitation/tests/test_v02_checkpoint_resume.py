"""True optimizer + sampler + stochastic TD3 target-noise recovery tests."""
from __future__ import annotations
import pytest
import torch
from rles_route_a.datasets.transitions import OfflineTransition
from rles_route_a.contracts import NativeActionTrace
from rles_route_a.training.iql_trainer import IQLHyper
from rles_route_a.training.td3bc_trainer import TD3BCHyper
from rles_route_a.training.offline_train_loop import OfflineReadiness, ReplaySampler, fit_offline
from rles_route_a.training.factories import new_iql, new_td3bc
from rles_route_a.training.checkpoints import save_training_checkpoint, load_training_checkpoint


def dataset():
    return tuple(OfflineTransition((i/100., 0.2), .25, -0.001,
        ((i+1)/100., 0.2), i==20, False, i!=20, f"ep/a/{i}",
        fold_id="fold-x",episode_id="ep/a",decision_ts_ns=1000+60*i,
        next_decision_ts_ns=1060+60*i,native_execution_recipe="risk-cc",
        discount_steps=1 if i % 2 else 3,
        action_trace=NativeActionTrace(1000+60*i,.25,.25,.25,.2,1000.,999.,"NOOP"),
        executed_target=.25, source_identity="source-d", frozen_encoder_id="bc-pretrained")
        for i in range(21))


def ready():
    return OfflineReadiness(True,True,True,True,"risk-cc","receipt-z",
                            registered_fold_id="fold-x",reward_scale=1000.,
                            registered_source_identity="source-d",
                            frozen_encoder_identity="bc-pretrained")


def verify_readiness(_readiness, _transitions):
    return True


def save_and_reload(learner, path, sampler):
    save_training_checkpoint(learner,path, registered_fold_id="fold-x",
                             execution_recipe="risk-cc", dataset_identity="source-d",
                             frozen_encoder_identity="bc-pretrained", reward_scale=1000.,
                             update_index=3,sampler_state=sampler.state())
    return load_training_checkpoint(path,expected_fold_id="fold-x",
            expected_execution_recipe="risk-cc",expected_dataset_identity="source-d",
            expected_frozen_encoder_identity="bc-pretrained",expected_reward_scale=1000.)


def tensors_equal(a,b):
    return all(torch.allclose(pa,pb,atol=0,rtol=0)
               for pa,pb in zip(a.parameters(),b.parameters()))


def test_iql_exact_optimizer_recovery_and_actor_weights(tmp_path):
    torch.set_num_threads(1)
    hyper=IQLHyper(hidden_size=12)
    first=new_iql(2,hyper,seed=44)
    split=new_iql(2,hyper,seed=44)
    full_rng=ReplaySampler(101)
    split_rng=ReplaySampler(101)
    fit_offline(first,dataset(),readiness=ready(),updates=6,batch_size=8,seed=101,sampler=full_rng,
                readiness_verifier=verify_readiness)
    fit_offline(split,dataset(),readiness=ready(),updates=3,batch_size=8,seed=101,sampler=split_rng,
                readiness_verifier=verify_readiness)
    loaded,index,state=save_and_reload(split,tmp_path/"iql-resume",split_rng)
    assert index==3 and state is not None
    resumed=ReplaySampler(0);resumed.restore(state)
    fit_offline(loaded,dataset(),readiness=ready(),updates=3,batch_size=8,seed=101,sampler=resumed,
                readiness_verifier=verify_readiness)
    assert tensors_equal(first.actor,loaded.actor)
    assert tensors_equal(first.q1,loaded.q1)
    assert tensors_equal(first.q2,loaded.q2)
    assert tensors_equal(first.value,loaded.value)
    with pytest.raises(PermissionError,match="provenance"):
        load_training_checkpoint(tmp_path/"iql-resume",expected_fold_id="another",
            expected_execution_recipe="risk-cc",expected_dataset_identity="source-d",
            expected_frozen_encoder_identity="bc-pretrained",expected_reward_scale=1000.)


def test_td3bc_resume_restores_target_noise_and_delayed_policy(tmp_path):
    torch.set_num_threads(1)
    hyper=TD3BCHyper(hidden_size=12,policy_delay=2)
    first=new_td3bc(2,hyper,seed=99)
    split=new_td3bc(2,hyper,seed=99)
    sampler_a=ReplaySampler(31);sampler_b=ReplaySampler(31)
    fit_offline(first,dataset(),readiness=ready(),updates=7,batch_size=8,seed=31,sampler=sampler_a,
                readiness_verifier=verify_readiness)
    fit_offline(split,dataset(),readiness=ready(),updates=3,batch_size=8,seed=31,sampler=sampler_b,
                readiness_verifier=verify_readiness)
    loaded,index,state=save_and_reload(split,tmp_path/"td3-resume",sampler_b)
    assert index==3 and loaded.steps==3
    restored=ReplaySampler(0);restored.restore(state)
    fit_offline(loaded,dataset(),readiness=ready(),updates=4,batch_size=8,seed=31,sampler=restored,
                readiness_verifier=verify_readiness)
    assert loaded.steps==first.steps==7
    assert tensors_equal(loaded.actor,first.actor)
    assert tensors_equal(loaded.q1,first.q1)
    assert tensors_equal(loaded.q2,first.q2)
    assert tensors_equal(loaded.q1_target,first.q1_target)
    assert torch.equal(loaded._noise_generator.get_state(),first._noise_generator.get_state())


def test_checkpoint_tamper_detection(tmp_path):
    model=new_iql(2,IQLHyper(hidden_size=8),seed=12)
    sampler=ReplaySampler(1)
    path=tmp_path/"corrupt"
    save_training_checkpoint(model,path,registered_fold_id="fold-x",execution_recipe="risk-cc",
             dataset_identity="source-d",frozen_encoder_identity="bc-pretrained",
             reward_scale=1000.,update_index=0,sampler_state=sampler.state())
    (path/"state.npz").write_bytes(b"attacker")
    with pytest.raises(ValueError,match="checksum"):
        load_training_checkpoint(path,expected_fold_id="fold-x",
            expected_execution_recipe="risk-cc",expected_dataset_identity="source-d",
            expected_frozen_encoder_identity="bc-pretrained",expected_reward_scale=1000.)
