# Train-inner prediction quality diagnostics — 2026-10-30

Status: `FROZEN_MODEL_SCREEN_DIAGNOSTICS_EXPANDED; REAL_TRAIN_SCREEN_BLOCKED_BY_FUTURE_SOURCE`.

## Change

Candidate-quality evaluation now reports prediction-to-outcome quality for all scored candidates, executable flat candidates, and candidates blocked by an open position or pending order. It preserves unscored candidates and outcome-status coverage, including right-censored labels, in each population.

For complete scored outcomes, the frozen screen report includes predicted versus observed net return and positive-return probability; MAE, bias, RMSE, Brier and log-loss; fixed probability bands plus stable equal-count return/probability bins; and overlap-uniqueness-weighted estimates so overlapping candidate labels do not count as independent trials. Accepted/rejected and per-teacher/side/consensus slices continue to show outcome quality and explicit reject reasons.

These are diagnostics on the independent Train-inner screen after the model and thresholds are frozen. They do not refit Platt/Ridge or modify the acceptance margin. Blocked account candidates are reported separately from executable candidates; censored labels remain outside realized-return statistics.

## Verification

- Ubuntu 24.04 / WSL2 / Python 3.14.7 full package suite plus native Meta engine and synthetic Train-to-screen end-to-end tests: **575 passed in 68.65s**.
- Ruff and `git diff --check` passed for changed modules.
- The synthetic full study still produces a complete 48-cell comparison while retaining `qualification: NOT_ESTABLISHED`.
- Latest read-only source preflight: `REGISTERED_FUTURE_SOURCE`, `completed_screen_windows: 0`, `market_payload_read: false`; registered interval `2026-12-01T00:00:00Z` through `2026-12-19T00:00:00Z`; protocol SHA-256 `b731f1e9f3d0eb7bb62f2fcdf4a76b609ff6452e3e8c146a8538b2fc25cc1845`.

## Evidence boundary

The new diagnostics are implementation evidence, not economic results. The frozen independent market source has not started, so this report contains no real Train-inner predictions or profitability claims.
