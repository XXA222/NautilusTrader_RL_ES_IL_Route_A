import math
import pytest
import torch
from rles_route_a.datasets.transitions import OfflineTransition
from rles_route_a.training.iql_trainer import IQLTrainer, IQLHyper
from rles_route_a.training.td3bc_trainer import TD3BCTrainer, TD3BCHyper
from rles_route_a.training.offline_train_loop import OfflineReadiness, fit_offline
from rles_route_a.export.offline_actor_bundle import save_offline_actor, load_offline_actor


def samples():
    return tuple(OfflineTransition((i/10., 0.2), 0.3 * (-1)**i, -0.0001,
                                   ((i+1)/10., 0.2), i==8, False, i!=8,
                                   f"native-episode-e1-step-{i}", executed_target=.1) for i in range(9))


def ready():
    return OfflineReadiness(True, True, True, True, "snapshot80194a/exec-close", "host-train-receipt", allow_synthetic_unit_only=True)


def test_conditional_offline_training(tmp_path):
    torch.set_num_threads(1)
    t = samples()
    iql = IQLTrainer(2, IQLHyper(hidden_size=16))
    with pytest.raises(PermissionError):
        fit_offline(iql, t, readiness=OfflineReadiness(False, True, True, True, "p", "receipt"),
                    updates=3, batch_size=4, seed=31)
    iql_stats = fit_offline(iql, t, readiness=ready(), updates=3, batch_size=4, seed=31)
    assert len(iql_stats) == 3 and all(math.isfinite(x["q_loss"]) for x in iql_stats)
    path = tmp_path / "iql"
    save_offline_actor(policy=iql.actor, kind="iql", state_dim=2, hidden_size=16,
                       encoder_bundle_ref="frozen-bc-id", native_train_receipt="host-receipt", path=path)
    actor, manifest = load_offline_actor(path)
    assert manifest["economic_status"] == "UNVERIFIED"
    with torch.no_grad():
        assert torch.allclose(actor.predict(torch.zeros(1,2)), iql.predict(torch.zeros(1,2)))
    with pytest.raises(FileExistsError):
        save_offline_actor(policy=iql.actor, kind="iql", state_dim=2, hidden_size=16,
                           encoder_bundle_ref="frozen-bc-id", native_train_receipt="receipt", path=path)
    td = TD3BCTrainer(2, TD3BCHyper(hidden_size=16, policy_delay=2))
    td_stats = fit_offline(td, t, readiness=ready(), updates=4, batch_size=6, seed=31)
    assert td_stats[0]["actor_loss"] is None and td_stats[1]["actor_loss"] is not None
    path_td = tmp_path / "td3bc"
    save_offline_actor(policy=td.actor, kind="td3bc", state_dim=2, hidden_size=16,
                       encoder_bundle_ref="frozen-bc-id", native_train_receipt="host-receipt", path=path_td)
    loaded_td, _ = load_offline_actor(path_td)
    with torch.no_grad():
        assert torch.allclose(loaded_td.predict(torch.zeros(1,2)), td.predict(torch.zeros(1,2)))


def test_cross_fold_rejected(observed):
    from rles_route_a.contracts import ExpertAdvice
    from rles_route_a.datasets.sequence_dataset import EpisodeSequenceDataset, LabeledStep
    from rles_route_a.datasets.daggerset import DAggerSet, AggregatedLabel
    names = ("market.trend_1h", "account.net_exposure")
    one = observed(fold="f1", episode="e1")
    two = observed(fold="f2", episode="e2", ts=2_000_000_000)
    a = LabeledStep(one, ExpertAdvice("w", one.decision_ts_ns, "FLAT", 0., ("NOOP",)), 0.,
                    source_identity="source")
    b = LabeledStep(two, ExpertAdvice("w", two.decision_ts_ns, "FLAT", 0., ("NOOP",)), 0.)
    with pytest.raises(ValueError, match="ONE registered Fold"):
        EpisodeSequenceDataset([a,b], feature_names=names, context_bars=2)
    data = DAggerSet((a,))
    with pytest.raises(PermissionError, match="different Folds"):
        data.append(AggregatedLabel(1,"student",b))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_cuda_training_accepts_unspecified_device_index():
    transitions = samples()
    gate = ready()
    for learner in (
        IQLTrainer(2, IQLHyper(hidden_size=8), device="cuda"),
        TD3BCTrainer(2, TD3BCHyper(hidden_size=8, policy_delay=1), device="cuda"),
    ):
        assert learner.device.index == torch.cuda.current_device()
        stats = fit_offline(learner, transitions, readiness=gate,
                            updates=2, batch_size=4, seed=31)
        assert len(stats) == 2
        assert all(value is not None and math.isfinite(value)
                   for row in stats for value in row.values())
