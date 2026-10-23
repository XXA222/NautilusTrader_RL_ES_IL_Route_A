# A-07 F1: BC-initialized SB3 fine-tuning

`finetune_bc_actor()` connects the frozen Train-inner BC baseline to the existing L4 `RLProvider.train()` path for PPO or SAC. It requires a pristine learner, consistent frozen BC dataset identities, and disabled VecNormalize; it trains BC first, then requires both non-collapse and a passed frozen action-parity tolerance before executing one explicit RL sample budget. It reports post-RL imitation behavior on the original training observations.

`compare_bc_to_random_ppo()` provides the paired F1 experiment: it creates independent pristine PPO providers from one factory, verifies identical frozen L4 config/transition identities and disabled VecNormalize, initializes only the treatment actor from BC, requires a non-collapsed actor and passed action-parity tolerance before any RL sample, then trains both arms to the same increasing cumulative sample milestones. At each milestone, the required L1 writer saves a checkpoint under that arm's frozen `attempt_id`; the runner calls the existing checkpoint verifier and checks the exact attempt and cumulative sample cursor. A required replay adapter must restore the checkpoint through L1 `load()` using the native environment's replay tapes and return float32 deterministic actions; the runner enforces action parity against the live actor. The audit snapshots and restores Python/NumPy/Torch RNG states around each checkpoint load so replay verification cannot perturb either continuing training arm. The result records both L4 native training performance blocks, checkpoint paths, checkpoint action parity errors, and Train-inner action behavior at each milestone. These behavior and training diagnostics are sample-efficiency evidence, not financial performance claims.

`compare_four_native_arms()` closes the engineering comparison loop after those
model gates. It requires disjoint fit and Train-inner selection trajectories,
with selection beginning after the native feature-lookback purge. It creates
independent pure-BC, pure-PPO and BC-to-PPO providers from one frozen raw
policy-domain recipe, clones the same teacher labels into both BC actors, then
checks both actors on the independent selection observations. Flat collapse or
failed action parity blocks before PPO interaction; the two PPO arms receive the
exact same sample budget only after both gates pass. It then replays the rule
teacher and all three learned policies through fresh native Gym accounts using
the same seeds, episode windows and decision clocks. The immutable JSON report
binds fit and selection identities/windows, BC fit/selection parity and
behavior receipts, paired episode identity, and native equity, fee, funding,
fill, exposure, risk and reward diagnostics. Its scope is explicitly
`train_inner_scenario_only` and `economic_performance_claim` remains false.

The included synthetic contract tests prove that BC initialization consumes no environment steps, paired arms receive equal fixed budgets, both L1 checkpoints pass integrity/attempt/cursor validation and restore with exact action parity, native collector replay works, overlapping fit/selection windows are rejected, and underfit BC is stopped before PPO interaction. The comparison runner is implemented, but its real experiment and independent native historical evaluation still require the registered Train-inner dataset and valid historical rule/fee evidence. No economic result is inferred from the toy fixtures.
