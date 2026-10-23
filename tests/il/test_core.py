from __future__ import annotations

import numpy as np
import pytest

from rles_il_core.dataset import (
    EpisodeTrajectory, Transition, assign_time_blocks, read_parquet_dataset,
    to_offline_episode_arrays, verify_parquet_dataset, write_parquet_dataset,
)
from rles_il_core.diagnostics import behavior_support_report
from rles_il_core.experts import (
    E1TrendExpert, E2Config, E2ReversionExpert, E3Config, E3CostAwareExpert,
    FixedExposureExpert, FlatExpert, FrozenVisibleContext, RecoveryExpert,
)


IDENTITY = {"dataset_hash": "a" * 64, "fold_hash": "b" * 64, "config_hash": "c" * 64,
            "transition_hash": "d" * 64, "schema_hash": "e" * 64}
WINDOW = {"start_ns": 60, "end_ns": 240, "decision_steps": 2,
          "episode_index": 0, "start_offset": 0}


def trajectory(expert_id: str = "flat_v1", *, start: int = 60) -> EpisodeTrajectory:
    window = {**WINDOW, "start_ns": start, "end_ns": start + 180}
    states = ((0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    transitions = tuple(Transition(
        observation=states[i], action=0.0, next_observation=states[i + 1],
        r_econ=0.0, r_shape=0.0, r_train=0.0,
        terminated=i == 1, truncated=False,
        decision_ts_ns=start + i * 60, next_decision_ts_ns=start + (i + 1) * 60,
        native_equity_before="1000", native_equity_after="1000", expert_id=expert_id,
        native_identity=IDENTITY, episode_window=window,
        execution={"executed_action": 0.0, "risk_guard_intervention": 0.0},
        terminal_settlement=(
            {"equity_delta": "0", "fills": [], "orders": []} if i == 1 else None
        ),
    ) for i in range(2))
    return EpisodeTrajectory(expert_id, states, transitions, {"split": "train_inner"})


def test_e1_is_causal_hysteretic_and_requires_raw_trend_features():
    policy = E1TrendExpert()
    policy.reset({})
    state = np.zeros(3, dtype=np.float32)
    long = FrozenVisibleContext(60, {"return_15m": .01, "return_60m": .02,
                                     "trend_efficiency_60m": .5})
    reverse = FrozenVisibleContext(120, {"return_15m": -.01, "return_60m": -.02,
                                         "trend_efficiency_60m": .8})
    assert policy.act(observation=state, context=long) == .5
    assert policy.act(observation=state, context=reverse) == 0.0
    with pytest.raises(ValueError, match="causal_mtf_v2"):
        policy.act(observation=state, context=FrozenVisibleContext(180, {"return_15m": .1}))


def test_flat_baseline_is_constant_zero():
    policy = FlatExpert()
    policy.reset({})
    context = FrozenVisibleContext(60, {})
    assert policy.act(observation=np.zeros(2), context=context) == 0.0


def test_e2_uses_only_visible_closed_prices_resets_on_gaps_and_enforces_holding_exit():
    policy = E2ReversionExpert(E2Config(lookback_minutes=4, max_holding_minutes=2))
    policy.reset({})
    state = np.zeros(2, dtype=np.float32)
    action = 0.0
    for minute in range(4):
        context = FrozenVisibleContext(
            (minute + 1) * 60_000_000_000,
            {"close_1m": 100.0 if minute < 3 else 110.0,
             "return_15m": 0.0, "return_60m": 0.0},
        )
        action = policy.act(observation=state, context=context)
    assert action == -0.4
    assert policy.act(
        observation=state,
        context=FrozenVisibleContext(
            5 * 60_000_000_000,
            {"close_1m": 110.0, "return_15m": 0.0, "return_60m": 0.0},
        ),
    ) == -0.4
    assert policy.act(
        observation=state,
        context=FrozenVisibleContext(
            6 * 60_000_000_000,
            {"close_1m": 110.0, "return_15m": 0.0, "return_60m": 0.0},
        ),
    ) == 0.0
    # A clock gap invalidates the rolling sample and returns the teacher to flat.
    gap = FrozenVisibleContext(
        10 * 60_000_000_000,
        {"close_1m": 100.0, "return_15m": 0.0, "return_60m": 0.0},
    )
    assert policy.act(observation=state, context=gap) == 0.0
    with pytest.raises(ValueError, match="causal_mtf_v4"):
        policy.act(observation=state, context=FrozenVisibleContext(11 * 60_000_000_000, {}))


def test_e3_cost_veto_fixed_direction_baselines_and_teacher_deadband():
    state = np.zeros(2, dtype=np.float32)
    policy = E3CostAwareExpert(E3Config(max_volatility_60m=0.2))
    policy.reset({})
    features = {"return_15m": 0.01, "return_60m": 0.02,
                "trend_efficiency_60m": 0.5, "volatility_60m": 0.3}
    assert policy.act(observation=state, context=FrozenVisibleContext(60, features)) == 0.0
    features["volatility_60m"] = 0.1
    assert policy.act(observation=state, context=FrozenVisibleContext(120, features)) == 0.5
    long = FixedExposureExpert(target_exposure=0.25, entry_ts_ns=120)
    short = FixedExposureExpert(target_exposure=-0.25, entry_ts_ns=120)
    assert long.expert_id == "fixed_long_v1" and short.expert_id == "fixed_short_v1"
    assert long.act(observation=state, context=FrozenVisibleContext(60, {})) == 0.0
    assert short.act(observation=state, context=FrozenVisibleContext(120, {})) == -0.25


def test_t3_recovery_teacher_uses_native_drawdown_pnl_and_exposure():
    teacher = RecoveryExpert(max_abs_exposure=0.6, stop_loss_fraction=0.04,
                             max_drawdown=0.15)
    observation = np.zeros(2, dtype=np.float32)
    teacher.reset({})
    safe = FrozenVisibleContext(
        60, {}, {"net_exposure": 0.5, "drawdown": 0.01,
                 "unrealized_pnl_pct": -0.01}, (),
    )
    assert teacher.act(observation=observation, context=safe) == 0.5
    oversize = FrozenVisibleContext(
        120, {}, {"net_exposure": -0.9, "drawdown": 0.01,
                  "unrealized_pnl_pct": 0.0}, (),
    )
    assert teacher.act(observation=observation, context=oversize) == -0.6
    stop = FrozenVisibleContext(
        180, {}, {"net_exposure": 0.5, "drawdown": 0.01,
                  "unrealized_pnl_pct": -0.05}, (),
    )
    assert teacher.act(observation=observation, context=stop) == 0.0
    drawdown = FrozenVisibleContext(
        240, {}, {"net_exposure": -0.3, "drawdown": 0.2,
                  "unrealized_pnl_pct": 0.0}, (),
    )
    assert teacher.act(observation=observation, context=drawdown) == 0.0


def test_t_plus_one_conversion_preserves_terminal_mask_and_economic_reward():
    item = trajectory()
    arrays = to_offline_episode_arrays(item)
    assert arrays["observations"].shape == (3, 2)
    assert arrays["actions"].shape == (2, 1)
    assert arrays["terminated"].tolist() == [False, True]
    assert arrays["truncated"].tolist() == [False, False]
    assert arrays["rewards"].tolist() == [0.0, 0.0]


def test_same_window_experts_stay_in_one_time_block():
    first, second = trajectory("flat_v1"), trajectory("trend_v1")
    split = assign_time_blocks((first, second), {"train": (1, 1000), "validation": (1000, 2000)})
    assert split["train"] == (first, second)
    assert not split["validation"]
    with pytest.raises(ValueError, match="crosses or falls outside"):
        assign_time_blocks((first,), {"train": (1, 100), "validation": (100, 1000)})


def test_support_report_does_not_drop_flat_or_loss_rows():
    report = behavior_support_report((trajectory(),))
    assert report["experts"]["flat_v1"]["transitions"] == 2
    assert report["experts"]["flat_v1"]["zero_target_fraction"] == 1.0


def test_parquet_dataset_is_atomic_and_checksummed(tmp_path):
    manifest = {
        "run_label": "synthetic-core-test", "split": "train_inner",
        "account_truth": "nautilus_native", "instrument": "BTC/USD.SIM",
        "market_timeframe": "1m", "feature_profile": "synthetic",
        "source_data_hash": IDENTITY["dataset_hash"], "fold_hashes": [IDENTITY["fold_hash"]],
        "source_environment_config_hash": IDENTITY["config_hash"],
        "transition_hash": IDENTITY["transition_hash"],
        "observation_schema_hash": IDENTITY["schema_hash"], "scaler_hash": "f" * 64,
        "dependencies": {"python": "3.14"}, "seeds": [1],
        "allowed_time_range_ns": [60, 240], "all_episodes_retained": True,
        "reward_source": "native_equity_delta",
    }
    target = write_parquet_dataset(tmp_path / "set", (trajectory(),), manifest)
    manifest = verify_parquet_dataset(target)
    assert manifest["episode_count"] == 1
    assert manifest["transition_count"] == 2
    assert (target / "episodes.parquet").is_file()
    assert (target / "transitions.parquet").is_file()
    assert (target / "support.json").is_file()
    restored, restored_manifest = read_parquet_dataset(target)
    assert restored == (trajectory(),)
    assert restored_manifest == manifest | {
        "schema": "rles.il.dataset.v1", "episode_count": 1, "transition_count": 2,
        "files_sha256": restored_manifest["files_sha256"],
    }
    with pytest.raises(FileExistsError):
        write_parquet_dataset(target, (trajectory(),), {})


def test_validator_rejects_bootstrapable_finite_terminal():
    item = trajectory()
    last = item.transitions[-1]
    bad_last = Transition(
        **{**{name: getattr(last, name) for name in last.__dataclass_fields__},
           "terminated": False, "truncated": False}
    )
    with pytest.raises(ValueError, match="explicit termination"):
        EpisodeTrajectory(item.expert_id, item.observations, (item.transitions[0], bad_last), {})
