# Route A: imitation learning core

This folder tracks Route A implementation evidence and gates. The mechanism layer uses the existing `NautilusBarrierEnv` and native account as the only source of order, fill, fee, funding and equity truth. The collection path is serial by default because each environment owns an isolated Nautilus session.

## Work-package status

| Package | Current state |
|---|---|
| A-00 | Inventory complete; full-period historical instrument-rule, account-fee, margin and liquidation evidence is missing. Economic selection remains closed. |
| A-01–A-03 | E1/E2/E3, Flat and fixed-exposure causal teachers, full native episodes, immutable dataset and offline conversion are implemented; synthetic mechanism/contract evidence is in `tests/il`. E2 uses the new identity-bound `causal_mtf_v4` profile. |
| A-04 | L4 BC, frozen behavior diagnostics, RFC candidate sweep and immutable winner publication are implemented; no eligible historical ExpertBC run is available. |
| A-05 | Isolated d3rlpy worker now enforces later non-overlapping fit/selection windows, emits held-out support/Q calibration, and exposes native replay; actual IQL training remains toy-only without an admissible historical dataset. |
| A-06 | Deferred: the A-05 evidence does not include native evaluation proving systematic IQL value overestimation, so the RFC trigger for CQL is absent. |
| A-07 | Paired random-PPO/BC-PPO runner checks fixed budgets and L1 checkpoint save/restore action parity on synthetic trajectories; real Train-inner training/evaluation awaits admissible inputs. |
| A-08 | Existing L6 WFO/ES flow covers frozen folds, stress gates, test selection and one-shot Holdout lifecycle. Synthetic WFO suite passes; historical economics remain gated by A-00. |

Details: [A-00 inventory](A00_inventory.md), [A-01 teachers](A01_teachers.md), [A-04 BC](A04_bc.md), [A-05 IQL worker](A05_iql_worker.md), [A-07 fine-tuning](A07_finetune.md), [A-08 WFO gate](A08_wfo.md).

## Interfaces

- `rles_il_core.experts.ExpertPolicy` receives the frozen observation plus a `FrozenVisibleContext` containing raw L2 features and the current decision clock. `E1TrendExpert` uses only `return_15m`, `return_60m` and `trend_efficiency_60m`; use `causal_mtf_v2` or later. `FlatExpert` is a zero-target plumbing baseline.
- Teacher IDs, complete parameter defaults, causal input fields and data-mixing rules are documented in [A-01 teachers](A01_teachers.md). E2 requires `causal_mtf_v4` to provide only the completed 1m close; its rolling buffer clears at a clock gap.
- `rles_il_core.collector.collect_episode(env, expert, seed=...)` runs one complete episode. The helper `collect_episodes(factory, expert_factory, seeds=...)` creates one new native environment and account for each episode. It refuses non-Train-inner factories.
- `EpisodeTrajectory` stores exactly T+1 observations and T transitions. The last transition contains the native terminal settlement facts. Raw target, soft/hard risk adjustments, rounded target, order plan, native fills, funding events, economic/shaped/training reward and source identities are kept separately.
- `assign_time_blocks` assigns entire exact-window groups to predeclared, disjoint time blocks. Multiple experts on the same window stay in the same split, but their account states remain independent trajectories.
- `behavior_support_report` reports action coverage and raw-to-executed gaps. It does not remove or reweight losing episodes.
- `write_parquet_dataset` publishes `episodes.parquet`, `transitions.parquet`, `support.json`, `manifest.json` and `COMPLETED` atomically; `verify_parquet_dataset` checks physical-file checksums and row counts; `read_parquet_dataset` rebuilds and revalidates trajectories. The target must not already exist.
- `to_offline_episode_arrays` exposes the verified T+1/T arrays for a future isolated algorithm worker. It does not install or import d3rlpy, fit a scaler or map shaped rewards to economic rewards.

## Data gate

Run `uv run --package rles-il-core rles-il-audit --root .` to generate `docs/il/A00_inventory.json` and the readable summary `docs/il/A00_inventory.md`. The audit opens `.artifacts/research-access.sqlite` read-only, exports each registered UTC-nanosecond interval and state, and verifies the archived market/rule files against their existing SHA-256 manifest. It reports 214 registered intervals in the copied inventory, including 209 `development:recorded`, 3 `test:reserved` and 2 `holdout:reserved`; the legacy inventory is known to be incomplete, so these records are not proof that unlisted historical intervals are fresh.

The current audit says historical ETHUSDT instrument rules, account fee history and margin/liquidation history are not established for the full local period. The existing material supports synthetic mechanism validation and explicitly labeled Train-inner engineering diagnostics. It does not support historical economic claims or fresh Test/Holdout selection. Current exchange rules must not be used to fill historical gaps.

The teacher thresholds and cooldowns are reproducible engineering candidates, not claims that any teacher is profitable. E1/E3 use causal trend/volatility features; E2's `causal_mtf_v4` close field is a new feature/schema identity. No BC, IQL, CQL, world model, new matching engine or limit-order path is included in this A-00 to A-03 package.
