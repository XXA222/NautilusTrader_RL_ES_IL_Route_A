from __future__ import annotations

from hashlib import sha256
import json
from math import log

import numpy as np
import pytest

from rles_il_core.bc import evaluate_bc_native_rollouts
from rles_il_core.collector import collect_episode
from rles_il_core.diagnostics import native_rollout_report
from rles_il_core.iql import evaluate_iql_native_rollouts

from test_native_training_episode import _SCHEMA, _env


class _FixedExpert:
    expert_id = "fixed_long_fixture"

    def reset(self, _metadata):
        return None

    def act(self, *, observation, context):
        assert context.decision_ts_ns > 0
        assert np.isfinite(observation).all()
        return 0.5


def test_native_three_state_episode_records_plan_fill_and_terminal_settlement():
    env = _env(max_steps=2)
    try:
        item = collect_episode(env, _FixedExpert(), seed=19)
    finally:
        env.close()
    assert len(item.observations) == len(item.transitions) + 1 == 3
    assert item.transitions[-1].terminated
    assert item.transitions[-1].terminal_settlement["policy_barrier_ts_ns"] < item.transitions[-1].terminal_settlement["settlement_barrier_ts_ns"]
    assert item.transitions[0].execution["execution_plan_action"] in {"SUBMIT", "NOOP", "WAIT", "CANCEL"}
    assert item.transitions[0].execution["requested_action"] == 0.5
    assert item.transitions[0].execution["risk_soft_adjustment"] > 0
    assert item.transitions[0].native_fills
    assert "commission" in item.transitions[0].native_fills[0]
    assert float(item.transitions[0].native_equity_after) < float(item.transitions[0].native_equity_before)
    assert np.isclose(
        item.transitions[0].r_econ,
        log(float(item.transitions[0].native_equity_after)
            / float(item.transitions[0].native_equity_before)),
    )
    assert item.transitions[0].native_identity["transition_hash"] == "d" * 64
    assert item.transitions[-1].native_equity_after
    assert item.transitions[-1].terminal_settlement["fills"]
    assert item.transitions[-1].terminal_settlement["funding"] == []
    report = native_rollout_report((item,))["experts"][item.expert_id]
    assert report["submit_count"] >= 1
    assert report["native_fill_count"] >= 1
    assert report["zero_position_fraction"] < 1.0
    assert report["native_net_return"] < 0.0
    assert report["economic_claim_eligible"] is False


def test_native_report_counts_reused_fill_ids_in_independent_accounts():
    episodes = []
    for seed in (11, 23):
        env = _env(max_steps=2)
        try:
            episodes.append(collect_episode(env, _FixedExpert(), seed=seed))
        finally:
            env.close()
    individual = [native_rollout_report((episode,))["experts"][_FixedExpert.expert_id]
                  for episode in episodes]
    combined = native_rollout_report(episodes)["experts"][_FixedExpert.expert_id]
    assert combined["native_fill_count"] == sum(row["native_fill_count"] for row in individual)
    from decimal import Decimal
    for currency in {key for row in individual for key in row["native_commissions_by_currency"]}:
        expected = sum((Decimal(row["native_commissions_by_currency"].get(currency, "0"))
                        for row in individual), Decimal(0))
        assert Decimal(combined["native_commissions_by_currency"][currency]) == expected


def test_bc_native_rollout_loads_hash_bound_actor_and_publishes_immutable_report(tmp_path, monkeypatch):
    from stable_baselines3 import PPO

    class _Actor:
        def predict(self, observation, *, deterministic):
            assert deterministic
            return np.asarray([0.5], dtype=np.float32), None

    class _Factory:
        split = "train_inner"
        transition_hash = "d" * 64

        def make_env(self, transition_hash):
            assert transition_hash == self.transition_hash
            return _env(max_steps=2)

    monkeypatch.setattr(PPO, "load", lambda _path, device: _Actor())
    actor = tmp_path / "bc.zip"
    actor.write_bytes(b"synthetic-checkpoint")
    manifest = {
        "split": "train_inner", "account_truth": "nautilus_native",
        "source_data_hash": "a" * 64, "fold_hashes": ["b" * 64],
        "source_environment_config_hash": "c" * 64, "transition_hash": "d" * 64,
        "observation_schema_hash": _SCHEMA.hash,
    }
    manifest_digest = sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    metadata = {
        "schema": "rles.il.bc_actor.v1", "account_truth": "nautilus_native",
        "split": "train_inner", "dataset_manifest": manifest,
        "dataset_manifest_sha256": manifest_digest,
        "checkpoint_sha256": sha256(actor.read_bytes()).hexdigest(),
        "actor_mapping": {"schema": "rles.l4.actor.mapping.v1", "algorithm": "PPO",
                          "actor_signature": "f" * 64},
    }
    actor.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    result = evaluate_bc_native_rollouts(
        actor, _Factory(), seeds=(7,), report_path=tmp_path / "native-report.json",
    )
    assert result.episode_count == 1
    assert result.native_report["experts"]["bc_ppo_ffffffffffff"]["native_fill_count"] >= 1
    saved = json.loads((tmp_path / "native-report.json").read_text(encoding="utf-8"))
    assert saved["actor_sha256"] == result.actor_sha256
    assert saved["economic_performance_claim"] is False
    with pytest.raises(FileExistsError):
        evaluate_bc_native_rollouts(
            actor, _Factory(), seeds=(7,), report_path=tmp_path / "native-report.json",
        )


def test_iql_native_rollout_loads_only_hash_bound_native_run_and_reserved_windows(tmp_path):
    import torch

    class _Fixed(torch.nn.Module):
        def forward(self, observations):
            return torch.full((observations.shape[0], 1), 0.5)

    class _Factory:
        split = "train_inner"
        transition_hash = "d" * 64

        def make_env(self, transition_hash):
            assert transition_hash == self.transition_hash
            return _env(transition=transition_hash, max_steps=2)

    run = tmp_path / "iql-run"
    run.mkdir()
    policy_path = run / "policy.pt"
    torch.jit.trace(_Fixed(), torch.zeros((1, len(_SCHEMA.names)))).save(str(policy_path))
    parity_path = run / "parity_inputs.npz"
    parity_observations = np.zeros((10_000, len(_SCHEMA.names)), dtype=np.float32)
    np.savez_compressed(
        parity_path, observations=parity_observations,
        expected_actions=np.full((10_000, 1), 0.5, dtype=np.float32),
    )
    identity = {"dataset_hash": "a" * 64, "fold_hash": "b" * 64,
                "config_hash": "c" * 64, "transition_hash": "d" * 64,
                "schema_hash": _SCHEMA.hash}
    source = {"split": "train_inner", "account_truth": "nautilus_native"}
    training = {"native_identity": identity, "source_manifest": source}
    selection = {"native_identity": identity, "source_manifest": source}
    import hashlib
    (run / "manifest.json").write_text(json.dumps({
        "schema": "rles.il.iql_run.v1", "purpose": "native_research",
        "historical_economic_claim": False,
        "dataset_manifest": training, "selection_dataset_manifest": selection,
        "training_episode_windows_ns": [[1, 10]],
        "selection_episode_windows_ns": [[10, 20]],
        "artifacts_sha256": {
            "policy.pt": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "parity_inputs.npz": hashlib.sha256(parity_path.read_bytes()).hexdigest(),
        },
    }), encoding="utf-8")
    (run / "COMPLETED").write_text("mechanism only\n", encoding="utf-8")

    result = evaluate_iql_native_rollouts(
        run, _Factory(), seeds=(17,), report_path=tmp_path / "iql-native.json",
    )
    assert result.episode_count == 1
    report = json.loads((tmp_path / "iql-native.json").read_text(encoding="utf-8"))
    assert report["evaluation_windows_ns"][0][0] > 20
    assert report["torchscript_cross_process_parity"]["samples"] == 10_000
    assert report["torchscript_cross_process_parity"]["max_abs_error"] == 0.0
    assert report["economic_performance_claim"] is False
