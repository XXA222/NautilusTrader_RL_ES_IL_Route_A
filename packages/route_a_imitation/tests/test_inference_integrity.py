from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from rles_route_a.config import BCConfig
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.models.bc import BCGRUPolicy


def _policy(*, temperature=1.0):
    names = ("market.trend_1h", "account.net_exposure")
    cfg = BCConfig(feature_names=names, hidden_size=8, layers=1, context_bars=3,
                   inference_mode="intent_aware_v1", open_intent_min_confidence=0.9,
                   intent_temperature=temperature)
    model = BCGRUPolicy(names, 8, 1)
    policy = BCInferencePolicy(model, cfg)
    policy.reset("e1")
    return policy


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_intent_logits_fail_without_advancing_history(observed, bad):
    policy = _policy()
    logits = torch.zeros((1, 7))
    logits[0, 1] = bad
    with patch.object(policy.model, "forward", return_value=(logits, torch.tensor([0.5]))):
        with pytest.raises(ValueError, match="intent logits"):
            policy.predict_exposure(observed())
    assert policy._last_ts == -1
    assert policy._observation_identity is None
    assert len(policy._series) == 0


def test_wrong_logits_shape_and_invalid_softmax_are_rejected(observed):
    policy = _policy(temperature=1e-320)
    with patch.object(policy.model, "forward",
                      return_value=(torch.tensor([[0., 1., 0., 0., 0., 0., 0.]]),
                                    torch.tensor([0.5]))):
        with pytest.raises(ValueError, match="intent probabilities"):
            policy.predict_exposure(observed())
    with patch.object(policy.model, "forward",
                      return_value=(torch.zeros((1, 2)), torch.tensor([0.5]))):
        with pytest.raises(ValueError, match="intent logits"):
            policy.predict_exposure(observed())
    assert policy._last_ts == -1 and len(policy._series) == 0


@pytest.mark.parametrize("field,value", [
    ("source_id", "different-source"), ("fold_id", "different-fold"),
    ("feature_spec_hash", "a" * 64),
])
def test_source_fold_and_schema_change_requires_reset(observed, field, value):
    policy = _policy()
    obs = observed()
    policy.predict_exposure(obs)
    modified = replace(obs, decision_ts_ns=obs.decision_ts_ns + 1, **{field: value})
    with pytest.raises(ValueError, match="identity changed"):
        policy.predict_exposure(modified)
    assert policy._last_ts == obs.decision_ts_ns and len(policy._series) == 1
    policy.reset("e1")
    policy.predict_exposure(modified)
