# Executable candidate population for Meta fitting (2026-10-30)

Status: `ENGINEERING_FIX_VERIFIED; ECONOMIC_QUALIFICATION_BLOCKED_BY_FUTURE_SOURCE`.

## Correction

The native candidate denominator intentionally includes opportunities observed while a position or pending order prevents entry. The live Meta policy returns `NO_TRADE` for those account states, so predictions on those rows can never drive an entry. Before this correction, the training pipeline retained those rows in fit and calibration after label completion, despite separately reporting their account-state coverage.

The pipeline now keeps all candidates, labels, censor states, and coverage counts for audit, but the model fitting and Platt calibration population is restricted to candidates that are:

1. Complete and retained after the existing time-boundary purge.
2. Flat at the native account snapshot and free of pending orders.
3. Complete on every configured model feature.

The direct `fit_entry_meta_policy` API enforces the same flat/no-pending condition, so callers cannot bypass the training-pipeline filter. Diagnostics report account-ineligible completed candidates separately for fit and calibration. Feature-missing counts remain available over the entire complete-label population, with separate counts after the account gate. `fit_complete_used` and `calibration_complete_used` now count only rows that pass all population gates. Rejected and censored rows remain in the saved outcome denominator.

## Validation

- Added regression coverage proving position-blocked and pending-order candidates remain in the outcomes but do not enter fit/calibration; the full minimum-sample check fails closed when the executable population is too small.
- Direct model fitting now rejects a blocked candidate.
- WSL Ubuntu 24.04 / Python 3.14.7 Route A opportunity, label, limit-entry, Meta-policy, native funnel and portfolio-replay suite: **116 passed**.
- The actual Nautilus/Gym synthetic-market Meta native-engine test and synthetic native-training end-to-end test: **2 passed**.
- Data preflight rechecked the registered Meta protocol `route-a-meta-x1-train-20261201-20261219-v1`: `REGISTERED_FUTURE_SOURCE`, completed native-screen windows `0`, `market_payload_read=false`, protocol SHA-256 `b731f1e9f3d0eb7bb62f2fcdf4a76b609ff6452e3e8c146a8538b2fc25cc1845`.

This validates the candidate population and engineering behavior only. No source payload was read, no real fit/calibration or native economic screen was run, and no teacher profitability or qualification is claimed.
