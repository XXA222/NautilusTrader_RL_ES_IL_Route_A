# Calibration-error-adjusted entry margin — 2026-10-30

## Change

The native-facing frozen entry policy now requires both:

1. predicted net return above the frozen positive margin; and
2. predicted net return minus the later Train-calibration MAE above that same
   margin.

Candidates that clear the predicted-return margin but fail the calibration
error check receive `CALIBRATION_ERROR_EXCEEDS_EDGE`. `EntryScore` retains the
MAE used for the decision, and `summarize_entry_scores()` reports the MAE,
calibration-adjusted expected-net distribution, and its effect on the
sequential candidate funnel. The MAE is an empirical calibration error, not a
conditional quantile or confidence interval.

The acceptance rule does not tune or lower the configured margin to force
trades. Native arbitration, pending-order checks, position ownership, and hard
risk handling remain in the existing MetaGatedTeacher path.

## Verification

`test_entry_meta_policy.py`, `test_meta_gated_teacher.py`, `test_policy_io.py`,
`test_route_a_meta_native_engine.py`, and
`test_route_a_meta_native_training_e2e.py`: **37 passed** on Ubuntu 24.04 /
Python 3.14.7. Ruff passed for the changed policy and tests.

## Evidence boundary

This strengthens the entry decision using the already measured later
calibration error. It is engineering evidence only; there is no claim of
positive Train or independent economic performance.
