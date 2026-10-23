import math
import pytest
import torch
from rles_route_a.datasets.transitions import OfflineTransition, OfflineBatch
from rles_route_a.training.iql_trainer import IQLTrainer, IQLHyper
from rles_route_a.training.td3bc_trainer import TD3BCTrainer, TD3BCHyper
from rles_route_a.export.offline_policy import encode_frozen
from rles_route_a.models.encoder import MultiScaleGRU


def fixture_transitions(n=12):
    out = []
    for i in range(n):
        out.append(OfflineTransition((i/10., .2), requested_action=(-1)**i * .4,
                                     native_log_reward=-.0001 * (i%3),
                                     next_state=((i+1)/10., .2),
                                     terminated=i==n-1, truncated=False,
                                     bootstrap_allowed=i<n-1, provenance=f"native-{i}",
                                     executed_target=.2))
    return out


def test_offline_action_uses_behavior_command_and_retains_native_mapping():
    t = fixture_transitions()
    b = OfflineBatch(t)
    assert b.action[0,0].item() == pytest.approx(.4)
    assert t[0].executed_target == pytest.approx(.2)
    assert t[0].requested_action == pytest.approx(.4)
    assert b.bootstrap[-1,0].item() == 0
    assert b.state.shape == (len(t), 2)


def test_invalid_transition_blocks():
    with pytest.raises(ValueError, match="terminal"):
        OfflineTransition((.1,), 0., 0., (.2,), True, False, True, "native", executed_target=0.)
    with pytest.raises(PermissionError):
        OfflineTransition((.1,), 0., 0., (.2,), False, False, True, "native", split="test", executed_target=0.)


def test_iql_td3bc_update_and_predict():
    torch.set_num_threads(1)
    b = OfflineBatch(fixture_transitions()).indices(torch.arange(12))
    iql = IQLTrainer(2, IQLHyper(hidden_size=12))
    results = [iql.train_step(b) for _ in range(2)]
    assert all(math.isfinite(v) for row in results for v in row.values())
    assert iql.predict(b["s"]).shape == (12,1)
    td = TD3BCTrainer(2, TD3BCHyper(hidden_size=12, policy_delay=2))
    td.train_step(b)
    assert math.isfinite(td.train_step(b)["actor_loss"])
    assert td.predict(b["s"]).abs().max() <= 1


def test_encoder_frozen_required():
    names = ("market.trend_1h", "account.net_exposure")
    enc = MultiScaleGRU(names, hidden_size=8, layers=1)
    x = torch.randn(2, 4, 2)
    length = torch.tensor([2,4])
    with pytest.raises(ValueError, match="FROZEN"):
        encode_frozen(enc, x, length)
    enc.eval()
    enc.requires_grad_(False)
    out = encode_frozen(enc, x, length)
    assert out.shape == (2,8) and not out.requires_grad
