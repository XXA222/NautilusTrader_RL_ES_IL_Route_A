# Isolated d3rlpy IQL worker

This project is deliberately outside the Python 3.14 `packages/*` workspace. Its runtime target is Python 3.12 and it pins d3rlpy 2.8.1. It exchanges only a checksummed `.npz` payload and JSON manifest with the main process; it imports neither NautilusTrader nor `rles_il_core`.

Input arrays use one row per native transition: `observations`, `next_observations`, `actions`, native `r_econ`, `terminated`, `truncated`, and episode offsets; `episode_windows_ns` stores each complete market window inside the checksummed NPZ. NPZ loading always disables pickle. Episode counts, finite values, dtypes, source hashes, Train-inner scope, and T+1/T adjacency are validated before adaptation.

The d3rlpy adapter builds one Episode per native account Episode. Native terminal episodes map to d3rlpy terminal episodes (no value bootstrap). Truncated episodes require an explicit config: `terminal_no_bootstrap` treats the last transition as terminal, while `timeout_bootstrap` preserves the native next observation through a dummy trailing row that is outside the T real transitions. The adapter compares every sampled d3rlpy transition against the source arrays and fails before training if any state, action, reward, count, or mask differs. Reward scaling is applied once to the copied reward array; observation/action scalers are explicitly disabled. A `native_research` config must name `selection_dataset`; it must share the frozen native identity, follow fit time windows without overlap, and contain only natively settled episodes. The worker reports held-out support distance and Q-to-realized-discounted-return calibration, then exports 10,000 held-out states and d3rlpy expected actions for main-process TorchScript parity. A synthetic smoke config sets `selection_dataset` to `null`; its repeated toy parity states test runtime compatibility only and have no market coverage or historical claim.

Run from this directory after `uv sync --locked`:

```powershell
uv run --locked python -m il_worker.toy --output .tmp/il_iql_toy_smoke
```

The toy command creates its own deterministic synthetic T+1/T dataset and config. It is a mechanism-only integration check, not a market result. Successful runs save `iql_model.d3`, `policy.pt`, `metrics.json`, `manifest.json`, and `COMPLETED`; any post-start failure writes `FAILED`. Output directories are immutable. Native runs record separate fit and selection manifests/windows plus held-out Q calibration. The core package exposes `evaluate_iql_native_rollouts()` to verify a completed native run and perform a fresh Train-inner Nautilus replay. Synthetic output always sets `historical_economic_claim=false`.



