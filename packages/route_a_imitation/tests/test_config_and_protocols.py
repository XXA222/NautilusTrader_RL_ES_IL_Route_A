import json
from pathlib import Path
import pytest
from rles_route_a.config import RouteAConfig, BCConfig, DAggerConfig
from rles_route_a.integration.l6_research import HostEvaluationGrant, evaluate_registered
from rles_route_a.integration.native_rollout import teacher_native_rollout
from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.datasets.sequence_dataset import LabeledStep
from rles_route_a.datasets.daggerset import DAggerSet, AggregatedLabel
from rles_route_a.models.mlp_baseline import BCMLPBaseline
from rles_route_a.training.bc_trainer import train_bc
from rles_route_a.contracts import ExpertAdvice
from rles_route_a.datasets.sequence_dataset import EpisodeSequenceDataset


def test_load_config_strict():
    path = Path(__file__).resolve().parents[1] / "configs" / "route_a_v1.json"
    cfg = RouteAConfig.load(path)
    assert cfg.action_schema == "single_exposure_v1"
    assert cfg.dagger.iterations == 3
    assert cfg.dagger.expert_fraction == pytest.approx(.7)
    with pytest.raises(PermissionError, match="only be collected from Train"):
        DAggerConfig(sample_split="validation")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["invalid_option"] = True
    with pytest.raises(ValueError, match="unknown"):
        BCConfig.parse({**raw["bc"], "invalid_option": True})


@pytest.mark.parametrize(("section", "key", "value", "error"), [
    ("market", "bar_interval", "1m", ValueError),
    ("teacher", "strategies", ["trend", "future_optimized"], ValueError),
    ("teacher", "require_economic_gate", False, PermissionError),
    ("iql", "enabled_only_if", "always", PermissionError),
    ("iql", "expectile", float("nan"), ValueError),
    ("evaluation", "scenarios", ["base", "stress"], ValueError),
    ("evaluation", "final_holdout_once", False, PermissionError),
])
def test_route_a_config_rejects_unsafe_nested_mutations(
        tmp_path, section, key, value, error):
    source = Path(__file__).resolve().parents[1] / "configs" / "route_a_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload[section][key] = value
    path = tmp_path / "route_a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(error):
        RouteAConfig.load(path)


def test_route_a_config_rejects_market_bc_context_drift(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs" / "route_a_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["market"]["context_bars"] = 60
    path = tmp_path / "route_a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must match"):
        RouteAConfig.load(path)


def test_native_rollout_must_have_registered_grant():
    with pytest.raises(PermissionError):
        teacher_native_rollout(lambda **kwargs: (), fold_id="f", grant=None)
    grant = HostTrainGrant("f", "owner", "receipt", "source", "train", True)
    assert teacher_native_rollout(lambda **kwargs: (), fold_id="f", grant=grant) == ()


def test_train_grant_binds_host_registered_read_receipt():
    from types import SimpleNamespace

    receipt = SimpleNamespace(
        split="train", registered_before_read=True, owner="research-owner",
        reservation_id="reservation-7",
        source_identity={"marker_sha256": "a" * 64},
    )
    grant = HostTrainGrant.from_registered_read(receipt, fold_id="fold-train")
    assert grant == HostTrainGrant(
        "fold-train", "research-owner", "reservation-7", "a" * 64, "train", True,
    )


def test_train_grant_rejects_malformed_receipt_identity():
    from types import SimpleNamespace

    with pytest.raises(PermissionError, match="registered TRAIN-only"):
        HostTrainGrant("f", "owner", "reservation", "source", "train", 1)
    with pytest.raises(PermissionError, match="invalid source marker"):
        HostTrainGrant.from_registered_read(SimpleNamespace(
            split="train", registered_before_read=True, owner="owner",
            reservation_id="reservation", source_identity={"marker_sha256": "bad"},
        ), fold_id="fold-train")
    with pytest.raises(PermissionError, match="missing host research scope"):
        HostTrainGrant("f", "owner", "reservation", object(), "train", True)


def test_daggerset_requires_one_registered_source_identity(observed):
    obs = observed()
    advice = ExpertAdvice("teacher", obs.decision_ts_ns, "FLAT", 0., ("NOOP",))
    with pytest.raises(ValueError, match="one registered Train source"):
        DAggerSet((LabeledStep(obs, advice, 0.),))
    valid = LabeledStep(obs, advice, 0., source_identity="registered-source")
    data = DAggerSet((valid,))
    foreign = LabeledStep(observed(episode="foreign", ts=2_000_000_000),
                          ExpertAdvice("teacher", 2_000_000_000, "FLAT", 0., ("NOOP",)),
                          0., source_identity="other-source")
    with pytest.raises(PermissionError, match="provenance"):
        data.append(AggregatedLabel(1, "student", foreign, "frozen-recipe"))


def test_l6_formal_eval_requires_host_evaluation_grant():
    class Eval:
        def evaluate_native(self, *, policy, frozen_recipe_id, grant):
            assert grant.split == "validation"
            return {"status": "HOST_REPORT_ONLY", "equity": None}
    with pytest.raises(PermissionError):
        HostEvaluationGrant("f", "train", "proto", "query", "receipt")
    v = evaluate_registered(Eval(), policy=object(), recipe_id="frozen", grant=
                            HostEvaluationGrant("f", "validation", "proto", "query", "receipt"))
    assert v["equity"] is None


def test_mlp_uses_same_bc_training_contract(observed):
    import torch
    torch.set_num_threads(1)
    names = ("market.trend_1h", "account.net_exposure")
    rows = []
    for i in range(4):
        o = observed(ts=1_000_000_000 + i*60_000_000_000)
        rows.append(LabeledStep(o, ExpertAdvice("teacher", o.decision_ts_ns,
                                               "HOLD", 0., ("WAIT",)), 0.))
    ds = EpisodeSequenceDataset(rows, feature_names=names, context_bars=3)
    cfg = BCConfig(names, context_bars=3, epochs=1, batch_size=2)
    model = BCMLPBaseline(names, hidden=8)
    report = train_bc(model, ds, cfg)
    assert len(report) == 1 and report[0].train_loss >= 0
