from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from rles_il_core.bc import (
    BCFinetuneBlocked, FrozenBCPolicy, behavior_report, compare_bc_to_random_ppo,
    compare_four_native_arms, evaluate_bc_candidates, finetune_bc_actor,
    prepare_bc_pairs,
)
from rles_il_core.experts import FixedExposureExpert, FrozenVisibleContext
from test_core import trajectory


class _Model:
    def __init__(self, action: float):
        self.action = action

    def predict(self, observation, *, deterministic):
        assert deterministic is True
        assert observation.dtype == np.float32
        shape = (1,) if observation.ndim == 1 else (len(observation), 1)
        return np.full(shape, self.action, dtype=np.float32), None


def _active_episode():
    item = trajectory("trend_v1")
    transitions = tuple(replace(t, action=0.5) for t in item.transitions)
    return replace(
        item, transitions=transitions,
        episode_metadata={"split": "train_inner", "expert_id": "trend_v1", "seed": 101},
    )


def _episode_in_window(item, start_ns):
    window = {
        "start_ns": start_ns, "end_ns": start_ns + 180,
        "decision_steps": len(item.transitions), "episode_index": 0, "start_offset": 0,
    }
    transitions = tuple(replace(
        transition, episode_window=window,
        decision_ts_ns=start_ns + index * 60,
        next_decision_ts_ns=start_ns + (index + 1) * 60,
    ) for index, transition in enumerate(item.transitions))
    return replace(item, transitions=transitions)


def _four_arm_fit_and_selection(native_factory, teacher_factory):
    from rles_il_core.collector import collect_episodes

    collected = collect_episodes(native_factory, teacher_factory, seeds=(7,))
    fit = (_episode_in_window(collected[0], 1_000_000_000),)
    selection = (_episode_in_window(collected[0], 2_000_000_000),)
    return fit, selection, (2_000_000_000, 2_000_000_180)


def test_prepare_bc_pairs_preserves_all_requested_action_labels():
    item = _active_episode()
    observations, actions, episodes = prepare_bc_pairs((item,))
    assert episodes == (item,)
    assert observations.shape == (2, 2)
    assert observations.dtype == np.float32
    assert actions.tolist() == [[0.5], [0.5]]


def test_bc_report_marks_flat_collapse_even_with_valid_low_mse():
    item = _active_episode()
    report = behavior_report((item,), np.zeros((2, 1), dtype=np.float32))
    assert report.mse == 0.25
    assert report.prediction_deadband_fraction == 1.0
    assert report.collapsed_to_flat
    assert "within the configured deadband" in report.warnings[0]


def test_bc_report_keeps_noncollapsed_actor_visible():
    item = _active_episode()
    report = behavior_report((item,), np.full((2, 1), 0.4, dtype=np.float32))
    assert not report.collapsed_to_flat
    assert report.direction_accuracy == 1.0
    assert report.prediction_nonzero_fraction == 1.0


def test_a04_frozen_candidate_sweep_uses_disjoint_train_inner_selection_block_and_publishes(tmp_path):
    from dataclasses import replace as dc_replace
    from rles_rl_sb3.provider import RLProvider
    from test_l4_provider_smoke import _small_pretraining_provider

    fit = _episode_in_window(_active_episode(), 60)
    selection = _episode_in_window(_active_episode(), 300)
    prototype = _small_pretraining_provider(normalize=False)
    prototype_config = prototype.config
    env_factory, gate_authorizer = prototype.env_factory, prototype.gate_authorizer
    prototype.close()

    def provider_factory(seed):
        seeds = dc_replace(
            prototype_config.seeds, algorithm=seed, environments=(seed + 1,),
            noise=seed + 2, replay=seed + 3,
        )
        return RLProvider(dc_replace(prototype_config, seeds=seeds), env_factory, gate_authorizer)

    manifest = {
        "schema": "rles.il.dataset.v1", "split": "train_inner",
        "account_truth": "nautilus_native", "source_data_hash": "a" * 64,
        "fold_hashes": ["b" * 64], "source_environment_config_hash": "c" * 64,
        "transition_hash": "d" * 64, "observation_schema_hash": "e" * 64,
        "allowed_time_range_ns": [60, 480], "seeds": [101],
        "expert_ids": ["trend_v1"], "all_episodes_retained": True,
    }
    artifact = tmp_path / "candidate-selected.zip"
    result = evaluate_bc_candidates(
        provider_factory, (fit,), (selection,), selection_window_ns=(300, 500),
        artifact_path=artifact, dataset_manifest=manifest,
        epochs_candidates=(60,), learning_rate_candidates=(3e-4,), seeds=(11,),
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].fit_behavior.transitions == 2
    assert result.candidates[0].selection_behavior.transitions == 2
    assert result.selected_candidate_id == result.candidates[0].candidate_id
    assert result.artifact_path == str(artifact.resolve())
    metadata = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
    plan = json.loads(Path(result.selection_plan_path).read_text(encoding="utf-8"))
    assert metadata["training"]["selection_window_ns"] == [300, 500]
    assert metadata["training"]["economic_performance_claim"] is False
    assert metadata["training"]["selection_plan_sha256"] == result.selection_plan_sha256
    assert plan["schema"] == "rles.il.bc_sweep_plan.v1"
    assert plan["epochs_candidates"] == [60]
    assert plan["selection_window_ns"] == [300, 500]


def test_a04_candidate_sweep_rejects_fit_windows_inside_selection_block_before_training(tmp_path):
    item = _episode_in_window(_active_episode(), 300)

    def forbidden_factory(_seed):
        pytest.fail("candidate training must not start on overlapping time blocks")

    with pytest.raises(ValueError, match="fit episodes must end before"):
        evaluate_bc_candidates(
            forbidden_factory, (item,), (item,), selection_window_ns=(300, 500),
            artifact_path=tmp_path / "forbidden.zip",
            dataset_manifest={"split": "train_inner", "account_truth": "nautilus_native"},
            epochs_candidates=(10,), learning_rate_candidates=(1e-4,), seeds=(11,),
        )


def test_a04_all_collapsed_candidates_persist_a_negative_report(tmp_path, monkeypatch):
    import rles_il_core.bc as bc_module
    from test_l4_provider_smoke import _small_pretraining_provider

    fit = _episode_in_window(_active_episode(), 60)
    selection = _episode_in_window(_active_episode(), 300)
    flat_report = behavior_report((selection,), np.zeros((2, 1), dtype=np.float32))
    monkeypatch.setattr(
        bc_module, "train_behavioral_clone",
        lambda *_args, **_kwargs: ({"max_abs_error": 0.5}, flat_report),
    )
    monkeypatch.setattr(bc_module, "behavior_report", lambda *_args, **_kwargs: flat_report)
    manifest = {
        "split": "train_inner", "account_truth": "nautilus_native",
        "source_data_hash": "a" * 64, "fold_hashes": ["b" * 64],
        "source_environment_config_hash": "c" * 64, "transition_hash": "d" * 64,
        "observation_schema_hash": "e" * 64, "allowed_time_range_ns": [60, 480],
        "seeds": [101], "expert_ids": ["trend_v1"], "all_episodes_retained": True,
    }
    artifact = tmp_path / "all-flat.zip"
    with pytest.raises(bc_module.BCCandidateSweepBlocked) as blocked:
        evaluate_bc_candidates(
            lambda _seed: _small_pretraining_provider(normalize=False), (fit,), (selection,),
            selection_window_ns=(300, 500), artifact_path=artifact, dataset_manifest=manifest,
            epochs_candidates=(10,), learning_rate_candidates=(1e-4,), seeds=(11,),
        )
    assert blocked.value.report_path == artifact.with_suffix(".failed.json")
    failure = json.loads(blocked.value.report_path.read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED"
    assert failure["candidates"][0]["selection_behavior"]["collapsed_to_flat"] is True
    assert failure["economic_performance_claim"] is False


def test_bc_rejects_validation_and_mixed_frozen_identities():
    item = _active_episode()
    validation = replace(item, episode_metadata={"split": "validation", "expert_id": "trend_v1"})
    with pytest.raises(ValueError, match="Train-inner"):
        prepare_bc_pairs((validation,))
    different_dataset = replace(
        item,
        transitions=tuple(replace(t, native_identity={**t.native_identity, "dataset_hash": "f" * 64})
                          for t in item.transitions),
    )
    with pytest.raises(ValueError, match="identities"):
        prepare_bc_pairs((item, different_dataset))


def test_frozen_bc_policy_is_deterministic_and_bounded():
    policy = FrozenBCPolicy(_Model(-0.25))
    policy.reset({"split": "train_inner"})
    action = policy.act(observation=np.asarray([1.0, 2.0], dtype=np.float32),
                        context=FrozenVisibleContext(60, {}))
    assert action == -0.25
    with pytest.raises(ValueError, match="finite policy-domain"):
        policy.act(observation=np.asarray([np.nan, 0.0], dtype=np.float32),
                   context=FrozenVisibleContext(60, {}))


def test_f1_bc_to_l4_finetuning_executes_same_fixed_rl_budget():
    from test_l4_provider_smoke import _small_pretraining_provider

    provider = _small_pretraining_provider(normalize=False)
    try:
        provider.create()
        with torch.no_grad():
            provider.model.policy.action_net.weight.zero_()
            provider.model.policy.action_net.bias.fill_(0.4)
        result = finetune_bc_actor(
            provider, (_active_episode(),), rl_steps=2, bc_epochs=600,
            bc_batch_size=2, bc_learning_rate=1e-3,
        )
        assert result.rl_steps == result.training_performance.samples == 2
        assert result.behavior_before_rl.transitions == 2
        assert provider.model.num_timesteps == 2
    finally:
        provider.close()


def test_f1_refuses_rl_updates_when_bc_actor_collapses():
    from test_l4_provider_smoke import _small_pretraining_provider

    provider = _small_pretraining_provider(normalize=False)
    try:
        provider.create()
        with torch.no_grad():
            provider.model.policy.action_net.weight.zero_()
            provider.model.policy.action_net.bias.zero_()
        with pytest.raises(BCFinetuneBlocked) as blocked:
            finetune_bc_actor(
                provider, (_active_episode(),), rl_steps=2, bc_epochs=1,
                bc_batch_size=2, bc_learning_rate=1e-6,
            )
        assert blocked.value.report.collapsed_to_flat
        assert provider.model.num_timesteps == 0
    finally:
        provider.close()


def test_f1_paired_ppo_comparison_equalizes_samples_and_replays_l1_checkpoints(tmp_path):
    from rles_rl_sb3.checkpoint import Boundary, checkpoint_episode_replays, checkpoint_verify
    from rles_rl_sb3.provider import RLProvider
    from test_l4_provider_smoke import ProviderFixtureEnv, _small_pretraining_provider

    checkpoints = {}
    configs = {}
    attempt_ids = {"random_ppo": "random-attempt", "bc_ppo": "bc-attempt"}
    providers = iter(("random_ppo", "bc_ppo"))

    def write_checkpoint(arm, provider, samples):
        assert provider.model.num_timesteps == samples
        tokens = tuple(provider.vec.get_attr("rles_boundary_token"))
        boundary = Boundary("R1", samples, tokens, True, True)
        path = tmp_path / arm / f"step-{samples}"
        saved = provider.save(path, boundary, attempt_id=attempt_ids[arm])
        checkpoints[arm] = saved
        return saved

    def replay_checkpoint(arm, path, observations):
        config = configs[arm]
        manifest = checkpoint_verify(path, config, trusted_source=True)
        boundary = Boundary.parse(manifest["boundary"])
        tapes = checkpoint_episode_replays(path, config, trusted_source=True)
        with np.load(path / "continuation.npz", allow_pickle=False) as continuation:
            restored_obs = continuation["raw"][0].copy()
        replay_state = (int(restored_obs[0]), restored_obs, tapes[0]["actions"])
        resumed = RLProvider(
            config,
            lambda _rank, _seed, _phase: ProviderFixtureEnv(
                config.fingerprint.hash, replay_state,
            ),
            lambda _gates, _config: True,
        )
        try:
            resumed.load(path, trusted_source=True, restored_boundary=boundary)
            predictions, _ = resumed.model.predict(observations, deterministic=True)
            return predictions
        finally:
            resumed.close()

    def make_provider():
        arm = next(providers)
        provider = _small_pretraining_provider(normalize=False)
        configs[arm] = provider.config
        return provider

    result = compare_bc_to_random_ppo(
        make_provider, (_active_episode(),), sample_milestones=(2, 4),
        checkpoint_writer=write_checkpoint, checkpoint_replayer=replay_checkpoint,
        attempt_ids=attempt_ids,
        bc_epochs=600, bc_batch_size=2, bc_learning_rate=1e-3,
    )
    assert result.seed == 11
    assert result.points[0].samples == 2
    assert result.points[0].random_performance.samples == 2
    assert result.points[0].bc_performance.samples == 2
    assert result.points[0].random_checkpoint != result.points[0].bc_checkpoint
    assert result.points[0].random_checkpoint_max_abs_error == 0
    assert result.points[0].bc_checkpoint_max_abs_error == 0
    assert result.points[1].samples == 4
    assert result.points[1].random_performance.samples == 2
    assert result.points[1].bc_performance.samples == 2
    assert result.points[1].random_checkpoint_max_abs_error == 0
    assert result.points[1].bc_checkpoint_max_abs_error == 0
    assert set(checkpoints) == {"random_ppo", "bc_ppo"}


def test_bc_actor_replays_through_native_collector_not_a_synthetic_ledger():
    from rles_il_core.collector import collect_episode
    from test_native_training_episode import _env

    env = _env(max_steps=2)
    try:
        item = collect_episode(env, FrozenBCPolicy(_Model(0.5)), seed=29)
    finally:
        env.close()
    assert item.expert_id == "bc_sb3_v1"
    assert len(item.transitions) == 2
    assert item.transitions[0].execution["requested_action"] == 0.5
    assert item.transitions[-1].terminal_settlement["fills"]


def test_four_arm_comparison_uses_one_native_input_domain_and_equal_ppo_budget(tmp_path):
    from rles_rl_sb3.config import PolicySpec, RLConfig, SeedSpec, TransitionFingerprint
    from rles_rl_sb3.provider import RLProvider
    from test_native_training_episode import _SCHEMA, _env

    fingerprint = TransitionFingerprint(**{
        name: f"{index + 1:064x}"
        for index, name in enumerate(TransitionFingerprint.__dataclass_fields__)
    })
    config = RLConfig(
        algorithm="PPO",
        policy=PolicySpec(
            observation_dim=6, hidden_layers=(8,), value_layers=(8,),
            observation_schema_hash=_SCHEMA.hash, action_schema_hash="a" * 64,
        ),
        fingerprint=fingerprint,
        seeds=SeedSpec(algorithm=11, environments=(12,), noise=13, replay=14),
        hyperparameters={"n_steps": 2, "batch_size": 2, "n_epochs": 1},
        device="cpu", normalize_observation=False,
    )

    class NativeFactory:
        split = "train_inner"
        transition_hash = fingerprint.hash
        features = SimpleNamespace(max_lookback_ns=5)

        @staticmethod
        def make_env(transition_hash):
            assert transition_hash == fingerprint.hash
            return _env(transition=transition_hash, max_steps=2)

    native_factory = NativeFactory()
    teacher_factory = lambda: FixedExposureExpert(target_exposure=0.5, entry_ts_ns=1)
    fit, selection, selection_window = _four_arm_fit_and_selection(
        native_factory, teacher_factory,
    )

    def provider_factory():
        return RLProvider(
            config,
            lambda _rank, _seed, _phase: _env(
                transition=fingerprint.hash, max_steps=2,
            ),
            lambda _gates, _config: True,
        )

    destination = tmp_path / "four-arm.json"
    result = compare_four_native_arms(
        provider_factory, fit, selection_trajectories=selection,
        selection_window_ns=selection_window, selection_purge_ns=5,
        rule_teacher_factory=teacher_factory,
        native_env_factory=native_factory, native_seeds=(17, 29),
        report_path=destination, rl_steps=2, bc_epochs=600,
        bc_batch_size=2, bc_learning_rate=1e-3,
    )

    report = json.loads(destination.read_text(encoding="utf-8"))
    assert result.sample_budget == 2
    assert report["schema"] == "rles.il.four_arm_native_comparison.v1"
    assert report["scope"] == "train_inner_scenario_only"
    assert report["economic_performance_claim"] is False
    assert report["sample_budget"] == 2
    assert report["policy_input_domain"]["l4_vecnormalize_enabled"] is False
    assert report["training_identity"] == dict(fit[0].transitions[0].native_identity)
    assert report["fit_episode_windows"] == [list((
        fit[0].transitions[0].episode_window["start_ns"],
        fit[0].transitions[0].episode_window["end_ns"],
    ))]
    assert report["bc_selection"]["window_ns"] == list(selection_window)
    assert report["bc_selection"]["episode_windows"] == [list(selection_window)]
    assert report["bc_selection"]["pure_bc_parity"]["passed"] is True
    assert report["bc_selection"]["bc_ppo_parity"]["passed"] is True
    assert report["bc_initialization"]["pure_bc_parity"]["method"] == "behavioral_distillation"
    assert report["bc_initialization"]["bc_ppo_parity"]["method"] == "behavioral_distillation"
    assert report["bc_initialization"]["pure_bc_parity"]["samples"] == 2
    assert report["bc_initialization"]["pure_bc_parity"]["passed"] is True
    assert report["bc_initialization"]["bc_ppo_parity"]["passed"] is True
    assert report["arms"] == {
        "rule_teacher": "fixed_long_v1", "pure_bc": "pure_bc",
        "pure_ppo": "pure_ppo", "bc_ppo": "bc_ppo",
    }
    assert report["bc_initialization"]["pure_bc"]["collapsed_to_flat"] is False
    assert report["bc_initialization"]["bc_ppo"]["collapsed_to_flat"] is False
    assert report["ppo_training"]["pure_ppo"]["samples"] == 2
    assert report["ppo_training"]["bc_ppo"]["samples"] == 2
    assert report["paired_episode_identity_sha256"] == result.paired_episode_identity_sha256
    assert sum(
        arm["episodes"] for arm in report["native_report"]["experts"].values()
    ) == 8


def test_four_arm_comparison_blocks_ppo_when_bc_action_parity_fails(tmp_path, monkeypatch):
    from rles_rl_sb3.config import PolicySpec, RLConfig, SeedSpec, TransitionFingerprint
    from rles_rl_sb3.provider import RLProvider
    from test_native_training_episode import _SCHEMA, _env

    fingerprint = TransitionFingerprint(**{
        name: f"{index + 1:064x}"
        for index, name in enumerate(TransitionFingerprint.__dataclass_fields__)
    })
    config = RLConfig(
        algorithm="PPO",
        policy=PolicySpec(
            observation_dim=6, hidden_layers=(8,), value_layers=(8,),
            observation_schema_hash=_SCHEMA.hash, action_schema_hash="a" * 64,
        ),
        fingerprint=fingerprint,
        seeds=SeedSpec(algorithm=11, environments=(12,), noise=13, replay=14),
        hyperparameters={"n_steps": 2, "batch_size": 2, "n_epochs": 1},
        device="cpu", normalize_observation=False,
    )

    class NativeFactory:
        split = "train_inner"
        transition_hash = fingerprint.hash
        features = SimpleNamespace(max_lookback_ns=5)

        @staticmethod
        def make_env(transition_hash):
            assert transition_hash == fingerprint.hash
            return _env(transition=transition_hash, max_steps=2)

    native_factory = NativeFactory()
    teacher_factory = lambda: FixedExposureExpert(target_exposure=0.5, entry_ts_ns=1)
    fit, selection, selection_window = _four_arm_fit_and_selection(
        native_factory, teacher_factory,
    )
    providers = []
    train_calls = []
    original_train = RLProvider.train

    def track_train(provider, samples, **kwargs):
        train_calls.append(samples)
        return original_train(provider, samples, **kwargs)

    monkeypatch.setattr(RLProvider, "train", track_train)

    def provider_factory():
        provider = RLProvider(
            config,
            lambda _rank, _seed, _phase: _env(
                transition=fingerprint.hash, max_steps=2,
            ),
            lambda _gates, _config: True,
        )
        providers.append(provider)
        return provider

    fit_window = fit[0].transitions[0].episode_window
    short_start = fit_window["end_ns"] + 4
    with pytest.raises(ValueError, match="overlap"):
        compare_four_native_arms(
            provider_factory, fit,
            selection_trajectories=(_episode_in_window(fit[0], short_start),),
            selection_window_ns=(short_start, short_start + 180),
            rule_teacher_factory=teacher_factory,
            native_env_factory=native_factory, native_seeds=(17, 29),
            report_path=tmp_path / "purge-four-arm.json", rl_steps=2,
            bc_epochs=20, bc_batch_size=2, bc_learning_rate=1e-3,
        )
    assert providers == []
    with pytest.raises(ValueError, match="overlap"):
        compare_four_native_arms(
            provider_factory, fit, selection_trajectories=fit,
            selection_window_ns=(fit_window["start_ns"], fit_window["end_ns"]),
            selection_purge_ns=5, rule_teacher_factory=teacher_factory,
            native_env_factory=native_factory, native_seeds=(17, 29),
            report_path=tmp_path / "overlap-four-arm.json", rl_steps=2,
            bc_epochs=20, bc_batch_size=2, bc_learning_rate=1e-3,
        )
    assert providers == []

    with pytest.raises(BCFinetuneBlocked, match="action parity"):
        compare_four_native_arms(
            provider_factory, fit, selection_trajectories=selection,
            selection_window_ns=selection_window, selection_purge_ns=5,
            rule_teacher_factory=teacher_factory,
            native_env_factory=native_factory, native_seeds=(17, 29),
            report_path=tmp_path / "blocked-four-arm.json", rl_steps=2,
            bc_epochs=20, bc_batch_size=2, bc_learning_rate=1e-3,
        )
    assert len(providers) == 3
    assert train_calls == []
    assert not (tmp_path / "blocked-four-arm.json").exists()
def test_ppo_arm_training_is_independent_of_previous_arm_random_draws():
    from hashlib import sha256
    import torch
    from rles_il_core.bc import _reset_ppo_arm_random_stream
    from test_l4_provider_smoke import _small_pretraining_provider

    def trained_hash(*, preceding_arm: bool) -> str:
        prior = _small_pretraining_provider(normalize=False)
        target = _small_pretraining_provider(normalize=False)
        try:
            prior.create()
            target.create()
            if preceding_arm:
                _reset_ppo_arm_random_stream(prior)
                prior.train(2)
                prior.close()
            _reset_ppo_arm_random_stream(target)
            target.train(2)
            digest = sha256()
            for name, tensor in sorted(target.model.policy.state_dict().items()):
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        finally:
            prior.close()
            target.close()

    torch.set_num_threads(1)
    assert trained_hash(preceding_arm=False) == trained_hash(preceding_arm=True)
