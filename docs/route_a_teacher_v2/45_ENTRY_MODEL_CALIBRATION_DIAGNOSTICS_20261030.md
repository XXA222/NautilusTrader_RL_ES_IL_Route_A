# Entry model calibration diagnostics — 2026-10-30

Status: `CALIBRATION_DIAGNOSTICS_ADDED; INDEPENDENT_TRAIN_SCREEN_BLOCKED_BY_FUTURE_SOURCE`.

## Change

The Train-only Logistic + Ridge pipeline now emits calibration-set summaries for predicted net returns and positive-return probabilities. The report contains weighted prediction bias, MAE and RMSE, Brier and log-loss, effective weighted sample count, and stable equal-count bins with predicted/observed net return and predicted/observed positive rate. Diagnostics do not modify model parameters or lower the frozen acceptance margin.

The frozen feature recipe was checked: it includes completed 1m price/indicator and ATR context, account exposure/drawdown/unrealized PnL, side (appended by the model), all eight teacher indicators, consensus count, and round-trip cost.

Evidence scope is explicit. Ridge is fitted only on Train-fit, so its net-return errors here are measured on the later Train-calibration block. Platt is fitted on Train-calibration, so the probability Brier/log-loss and bins on that same block are in-sample Platt diagnostics, **not independent probability validation**. The future Train-inner native screen remains the first independent economic evaluation; no thresholds were changed to improve apparent fit.

## Verification

- Ubuntu 24.04 / WSL2 / Python 3.14.7: relevant package tests plus the native-engine and full native-training end-to-end tests: **574 passed in 69.08s**.
- Ruff passed on changed code and tests; `git diff --check` passed.
- The full synthetic study still completes its 48-cell comparison while returning `NOT_ESTABLISHED` qualification.
- Read-only real-source preflight still reports `REGISTERED_FUTURE_SOURCE`, zero completed windows and `market_payload_read: false` for `route-a-meta-x1-train-20261201-20261219-v1` (protocol SHA-256 `b731f1e9f3d0eb7bb62f2fcdf4a76b609ff6452e3e8c146a8538b2fc25cc1845`).

## Limit

Synthetic data verifies the training/reporting path only. Real Train calibration and Base/Stress native economic comparison remain unavailable until the registered source interval is complete and the source preflight permits execution.
