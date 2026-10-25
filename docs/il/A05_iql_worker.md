# A-05 isolated d3rlpy IQL worker

The IQL worker lives in `research_workers/il_d3rlpy`, outside the Python 3.14 workspace. It is locked to Python 3.12 and d3rlpy 2.8.1. The main workspace remains unchanged by d3rlpy, Gymnasium, or Torch resolution. A safe NPZ/JSON transport preserves original `T+1/T` observations and transitions, verifies the payload SHA-256, retains terminated/truncated separately, and refuses non-Train-inner inputs.

The worker imports each native trajectory as one d3rlpy `Episode` and checks every sampled transition against source state/action/reward arrays before training. The checksummed NPZ carries one native time window per episode, preserving shared windows across expert teachers while rejecting partial overlap. Native research requires a separate reserved Train-inner selection dataset with the same native source/fold/config/transition/schema identity, strictly later and non-overlapping fit windows, and fully settled episodes. It computes held-out policy-to-action support distances and calibrates dataset-action Q estimates against realized discounted native returns in Q-ranked bins. Fit-only action/Q statistics remain labeled as training diagnostics. The worker applies reward scaling once, disables observation/action/reward scalers, requires an explicit truncation policy, and fails if requested gradient updates are not executed exactly. Native and synthetic-smoke data have distinct manifests and run purposes. Synthetic output is marked `historical_economic_claim=false`.

The worker saves a hashed fixed-state parity payload beside the TorchScript actor. Native runs use up to 10,000 held-out selection observations; `rles_il_core.verify_iql_export_parity()` reloads the actor in the main Python 3.14 process and requires 10,000 finite states with action error at most `1e-5`. A short dataset is reported as an incomplete compatibility gate rather than padded with duplicated market states. The small synthetic smoke may repeat its tiny toy states for a runtime contract check, explicitly without market-coverage meaning.

`rles_il_core.evaluate_iql_native_rollouts()` loads only a completed `native_research` run whose TorchScript policy hash matches its run manifest and whose 10,000-state cross-process parity passes. It verifies same-source native identity and requires new Train-inner replay windows after the fit and selection blocks, then sends the policy through the existing Nautilus collector and writes a hash-bound immutable account/behavior report. The isolated worker still cannot access Nautilus or native market data directly.

## Verification

- Python 3.12.10, d3rlpy 2.8.1, Torch 2.14.1+cpu, Gymnasium 1.0.0, NumPy 2.5.4 import successfully in the isolated environment.
- A 4-update CPU IQL toy run completed using both an exported `EpisodeTrajectory` and the worker's self-contained synthetic smoke command.
- d3rlpy model save/load action parity and TorchScript export parity passed during both runs.
- A worker run exported 10,000 fixed toy-state/action pairs; the main Python 3.14 process loaded the worker TorchScript file and verified action parity with max/mean absolute error `0.0`. The toy states are repeated fixtures and do not satisfy market/position coverage.
- `tests/test_conversion.py` covers timeout-bootstrap next-state preservation, explicit no-bootstrap terminalization, reward scaling, and rejection of implicit timeout behavior.
- Main-package exporter tests cover checksummed pickle-free NPZ, Train-inner scope, and T+1/T field counts.
- Worker tests cover native episode window separation and held-out Q-versus-realized-return calibration; these are mechanism checks, not evidence of economic value.

These results prove worker mechanics only. IQL has not yet been trained on an admissible historical trading dataset; A-00's rule/fee/margin evidence blocker still prohibits historical economic conclusions or model selection.
