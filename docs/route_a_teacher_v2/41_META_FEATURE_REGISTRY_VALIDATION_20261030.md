# Meta feature registry validation — 2026-10-30

## Change

The frozen entry policy now validates feature names against the supported feature grammar and the current eight-teacher registry. A misspelled or retired `teacher.*` feature raises an explicit error instead of silently becoming a constant zero indicator. Market/account features remain point-in-time observation lookups; missing fields are reported as not ready. Direction remains appended from the candidate side, and consensus/cost features retain their explicit encodings.

The independently frozen 2026-10-29 protocol already contains 1m indicators including ATR, native account state, all T01–T08 source indicators, consensus count, and round-trip cost. Its bytes and preregistration hash were not changed.

## Evidence

- Changed `entry_meta_policy.py`; changed training fixtures to use registered `T01_EMA_9_21` identity.
- Added a regression case for unknown teacher names.
- Ruff passed. Entry policy, registered training, gated teacher, native training data and native Meta runner/engine/e2e tests passed: `54 passed in 44.28s`.

## Boundary

This rejects invalid feature schemas; it does not establish predictive value. The independently registered source remains future-dated and unavailable, so no real Train calibration, native screen, or post-cost profitability result is claimed.
