# Entry score output contract — 2026-10-30

## Change

`EntryScore` now validates its own decision output before it can enter a funnel or native trace: action vocabulary, finite expected net return, probability in `[0, 1]`, boolean support status, nonempty reason, unique unsupported feature names, and finite nonnegative calibration MAE. This makes malformed model outputs fail at the score boundary instead of being counted as ordinary accepts or rejects.

## Verification

- Invalid action, NaN probability, out-of-range probability, infinite expected return, non-boolean support, and negative calibration error are rejected.
- Entry policy, gated teacher, native funnel, and native integration regressions: **44 passed**. Ruff and `git diff --check` passed.

## Evidence boundary

This validates score integrity only. Predictive quality and positive net economics still require the registered future Train/calibration and native screen windows.
