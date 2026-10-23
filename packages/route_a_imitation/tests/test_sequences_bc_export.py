import numpy as np
import pytest
import torch
from dataclasses import replace
from rles_route_a.config import BCConfig
from rles_route_a.contracts import ExpertAdvice
from rles_route_a.datasets.sequence_dataset import LabeledStep, EpisodeSequenceDataset
from rles_route_a.models.bc import BCGRUPolicy, bc_losses
from rles_route_a.models.mlp_baseline import BCMLPBaseline
from rles_route_a.training.bc_trainer import _native_case_balanced_sampler, train_bc
from rles_route_a.training.native_bc import _native_behavior_summary
from rles_route_a.export.bundle import save_bc_bundle, load_bc_bundle
from rles_route_a.export.inference import BCInferencePolicy

NAMES = ("market.trend_1h", "market.trend_1d", "market.vol_1m", "account.net_exposure")


def make_steps(observed):
    output = []
    for ep in ("a", "b"):
        for i in range(4):
            o = observed(ts=1_000_000_000 + (i + 1) * 60_000_000_000, episode=ep,
                         pos=i / 20., hourly=(i+1)*.01)
            advice = ExpertAdvice("expert", o.decision_ts_ns, "ADD", .5, ("causal",))
            output.append(LabeledStep(o, advice, .4))
    return output


def test_episode_padding_and_reset(observed):
    ds = EpisodeSequenceDataset(make_steps(observed), feature_names=NAMES, context_bars=3)
    # The student action is a pre-risk request; .4 is the retained native outcome.
    assert ds[0]["exposure"].item() == pytest.approx(.5)
    assert ds[0]["case_assessed"].item() is False
    assert ds[0]["case_outcome_bps"].item() == 0.0
    assert ds.steps[0].executable_exposure == pytest.approx(.4)
    assert ds[0]["length"].item() == 1
    assert ds[0]["valid_mask"].tolist() == [False, False, True]
    assert ds[4]["length"].item() == 1  # cannot leak end of previous episode
    assert ds[3]["length"].item() == 3
    assert float(ds[4]["x"][-1][0]) == pytest.approx(.01)
    assert torch.count_nonzero(ds[4]["x"][:2]) == 0


def test_distinct_rollouts_with_same_episode_id_do_not_share_sequence_context(observed):
    base = make_steps(observed)[:2]
    first = [replace(row, observation=replace(row.observation, episode_id="same-episode"),
                     trajectory_id="run-1") for row in base]
    second = [replace(row, observation=replace(row.observation, episode_id="same-episode"),
                      trajectory_id="run-2") for row in base]

    ds = EpisodeSequenceDataset(
        [*first, *second], feature_names=NAMES, context_bars=3,
    )

    assert ds[1]["length"].item() == 2
    assert ds[2]["length"].item() == 1
    assert ds[3]["length"].item() == 2
    assert torch.count_nonzero(ds[2]["x"][:-1]) == 0


def test_dataset_rejects_nontrain_or_noncausal(observed):
    step = make_steps(observed)[0]
    with pytest.raises(PermissionError):
        LabeledStep(step.observation, step.advice, .4, split="test")
    with pytest.raises(ValueError, match="non-monotonic"):
        EpisodeSequenceDataset([step, step], feature_names=NAMES, context_bars=4)


def test_bc_forward_grad_export_and_inference(tmp_path, observed):
    torch.set_num_threads(1)
    ds = EpisodeSequenceDataset(make_steps(observed), feature_names=NAMES, context_bars=3)
    cfg = BCConfig(feature_names=NAMES, context_bars=3, hidden_size=12, layers=1,
                   batch_size=4, epochs=2)
    model = BCGRUPolicy(NAMES, 12, 1)
    hist = train_bc(model, ds, cfg)
    assert len(hist) == 2 and all(np.isfinite(x.train_loss) for x in hist)
    sample = ds[0]
    with torch.no_grad():
        logits, a = model(sample["x"].unsqueeze(0), sample["length"].reshape(1))
    assert logits.shape == (1, 7) and a.shape == (1,) and a.abs().max() <= 1
    path = tmp_path / "bc_bundle"
    save_bc_bundle(model, cfg, path)
    loaded, loaded_cfg = load_bc_bundle(path)
    with torch.no_grad():
        new = loaded(sample["x"].unsqueeze(0), sample["length"].reshape(1))
    assert loaded_cfg == cfg
    assert torch.allclose(a, new[1], atol=1e-6)
    policy = BCInferencePolicy(loaded, loaded_cfg)
    policy.reset("a")
    first = make_steps(observed)[0].observation
    assert -1 <= policy.predict_exposure(first) <= 1
    with pytest.raises(ValueError, match="clock"):
        policy.predict_exposure(first)
    with pytest.raises(ValueError, match="reset"):
        policy.predict_exposure(observed(episode="new", ts=first.decision_ts_ns+1))
    with pytest.raises(FileExistsError):
        save_bc_bundle(model, cfg, path)


def test_native_bc_behavior_summary_detects_flat_action_collapse(observed):
    ds = EpisodeSequenceDataset(make_steps(observed), feature_names=NAMES, context_bars=3)
    model = BCGRUPolicy(NAMES, hidden_size=8, layers=1)
    with torch.no_grad():
        model.exposure_head.weight.zero_()
        model.exposure_head.bias.zero_()

    report = _native_behavior_summary(
        model, ds, batch_size=4, device="cpu",
    )

    assert report.sample_count == len(ds)
    assert report.executable_label_count == len(ds)
    assert report.label_nonzero_fraction == 1.0
    assert report.prediction_nonzero_fraction == 0.0
    assert report.prediction_std == 0.0
    assert report.mae == pytest.approx(0.5)
    assert report.collapsed_to_flat is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_bc_bundle_cpu_cuda_inference_parity_and_determinism(tmp_path, observed):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    cfg = BCConfig(feature_names=NAMES, context_bars=3, hidden_size=8, layers=1)
    path = tmp_path / "cross_device_bundle"
    save_bc_bundle(BCGRUPolicy(NAMES, 8, 1), cfg, path)
    cpu_model, cpu_cfg = load_bc_bundle(path, device="cpu")
    cuda_model, cuda_cfg = load_bc_bundle(path, device="cuda")
    assert cpu_cfg == cuda_cfg == cfg

    cpu_policy = BCInferencePolicy(cpu_model, cpu_cfg, device="cpu")
    cuda_policy = BCInferencePolicy(cuda_model, cuda_cfg, device="cuda")
    cpu_policy.reset("a")
    cuda_policy.reset("a")
    rows = make_steps(observed)[:4]
    cpu_outputs = []
    cuda_outputs = []
    for row in rows:
        cpu_outputs.append(cpu_policy.predict_exposure(row.observation))
        cuda_outputs.append(cuda_policy.predict_exposure(row.observation))
    torch.cuda.synchronize()
    # CPU and CuDNN GRU kernels differ slightly; bound the action error to
    # 3e-4 absolute in the normalized [-1, 1] action domain.
    assert np.max(np.abs(np.asarray(cpu_outputs) - np.asarray(cuda_outputs))) <= 3e-4

    cuda_policy.reset("a")
    repeated = cuda_policy.predict_exposure(rows[0].observation)
    assert repeated == pytest.approx(cuda_outputs[0], rel=1e-6, abs=1e-6)


def test_corrupted_bundle_rejected(tmp_path):
    cfg = BCConfig(feature_names=NAMES, hidden_size=8, layers=1)
    d = tmp_path / "model"
    save_bc_bundle(BCGRUPolicy(NAMES, 8, 1), cfg, d)
    (d / "weights.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        load_bc_bundle(d)


def test_mlp_baseline_and_left_padding_invariance(observed):
    ds = EpisodeSequenceDataset(make_steps(observed), feature_names=NAMES, context_bars=3)
    x = ds[0]["x"].unsqueeze(0)
    length = ds[0]["length"].reshape(1)
    mlp = BCMLPBaseline(NAMES)
    assert mlp(x, length)[1].shape == (1,)
    model = BCGRUPolicy(NAMES, 8, 1).eval()
    with torch.no_grad():
        one = model(x, length)[1]
        x2 = torch.cat([torch.zeros(1, 2, len(NAMES)), x], dim=1)
        two = model(x2, length)[1]
    assert torch.allclose(one, two, atol=1e-6)


def test_training_sequences_match_online_prefix_inference(observed):
    torch.manual_seed(41)
    torch.set_num_threads(1)
    rows = make_steps(observed)[:4]
    ds = EpisodeSequenceDataset(rows, feature_names=NAMES, context_bars=3)
    cfg = BCConfig(feature_names=NAMES, context_bars=3, hidden_size=8, layers=1)
    policy = BCInferencePolicy(BCGRUPolicy(NAMES, 8, 1), cfg)
    policy.reset("a")

    for index, row in enumerate(rows):
        sample = ds[index]
        with torch.no_grad():
            _logits, trained_exposure = policy.model(
                sample["x"].unsqueeze(0), sample["length"].reshape(1),
            )
        online_exposure = policy.predict_exposure(row.observation)
        assert online_exposure == pytest.approx(
            float(trained_exposure.item()), rel=1e-6, abs=1e-6,
        )


def test_fractional_supervision_weight_preserves_weighted_mean():
    logits = torch.tensor([[1.0, -0.5, 0.2, 0.0, -0.3, 0.1, -1.0]])
    exposure = torch.tensor([0.1])
    intent = torch.tensor([2])
    target_exposure = torch.tensor([0.7])
    full = bc_losses(logits, exposure, target_intent=intent,
                     target_exposure=target_exposure, sample_weight=torch.tensor([1.0]))
    fractional = bc_losses(logits, exposure, target_intent=intent,
                           target_exposure=target_exposure,
                           sample_weight=torch.tensor([0.2]))
    zero = bc_losses(logits, exposure, target_intent=intent,
                     target_exposure=target_exposure, sample_weight=torch.tensor([0.0]))

    assert torch.allclose(fractional["loss"], full["loss"])
    assert torch.allclose(fractional["intent_ce"], full["intent_ce"])
    assert torch.allclose(fractional["exposure_huber"], full["exposure_huber"])
    assert fractional["effective_samples"].item() == pytest.approx(0.2)
    assert zero["loss"].item() == 0.0


def test_native_case_sampler_equalizes_case_mass_and_excludes_unassessed(observed):
    rows = make_steps(observed)
    labels = []
    for index, step in enumerate(rows):
        if index == 7:
            labels.append(step)
            continue
        case_index = "a" if index < 3 else "b"
        labels.append(replace(
            step,
            economic_case_status="OBSERVED_NATIVE_TRADE_POSITIVE",
            case_outcome_bps=1.0,
            case_end_ts_ns=step.observation.decision_ts_ns + 1,
            case_fact_sha256=case_index * 64,
        ))
    dataset = EpisodeSequenceDataset(
        labels, feature_names=NAMES, context_bars=3,
    )

    sampler = _native_case_balanced_sampler(
        dataset, seed=13, exposure_target_mode="executed_only_v1",
    )

    assert sampler is not None
    assert sampler.num_samples == 7
    assert sampler.weights[:3].sum().item() == pytest.approx(1.0)
    assert sampler.weights[3:7].sum().item() == pytest.approx(1.0)
    assert sampler.weights[7].item() == 0.0
    assert len(list(sampler)) == 7


def test_native_case_sampler_includes_only_explicit_losing_entry_counterexample(observed):
    rows = make_steps(observed)
    labels = [
        replace(
            step,
            economic_case_status="OBSERVED_NATIVE_TRADE_NEGATIVE",
            case_outcome_bps=-2.0,
            case_end_ts_ns=step.observation.decision_ts_ns + 1,
            case_fact_sha256="d" * 64,
        )
        for step in rows[:3]
    ]
    labels[0] = replace(
        labels[0],
        advice=replace(
            labels[0].advice,
            intent="FLAT",
            requested_exposure=0.0,
            reason_codes=("negative_native_trade_entry_avoidance",),
        ),
        supervision_weight=0.0,
        intent_supervision_weight=1.0,
        label_origin="INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE",
    )
    dataset = EpisodeSequenceDataset(
        [*labels, *rows[3:]], feature_names=NAMES, context_bars=3,
    )

    sampler = _native_case_balanced_sampler(
        dataset, seed=17, exposure_target_mode="teacher_request_v2",
    )

    assert sampler is not None
    assert sampler.num_samples == 1
    assert sampler.weights[0].item() == pytest.approx(1.0)
    assert torch.count_nonzero(sampler.weights[1:]).item() == 0
