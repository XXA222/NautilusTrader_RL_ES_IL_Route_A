# Group outcome receipt and censor clocks — 2026-10-30

## Change

`GroupOutcome` now requires the simulator receipt that produced each frozen-exit label and preserves `observed_until_ns` for completed, right-censored, unfilled, cancelled, and missed candidates. No-entry results retain the replay receipt and observation clock instead of losing them when converted from `NoEntryReplay`. Open trade replays retain their censor/coverage clock.

The outcome contract now rejects malformed candidate SHA-256 identities, non-positive decision clocks, missing recipe/receipt identity, native-evidence claims on counterfactual outcomes, unknown statuses, non-finite complete returns, incomplete outcomes with a return or exit time, and inconsistent censor/observation clocks. Counterfactual labels remain `MODEL_COUNTERFACTUAL` / `SCENARIO_ONLY`; incomplete outcomes are not converted into zero returns.

## Evidence

- Changed `conditional_outcome_labels.py` and constructor fixtures across the meta-training tests.
- Added assertions for receipt propagation, all four no-entry statuses, finite complete returns, and matching right-censor observation clocks.
- Ruff passed for all changed outcome and policy test modules.
- Package suite plus native Meta study runner/engine/e2e suite passed after the final contract assertions: `578 passed in 69.87s`.
- Follow-on identity-clock checks passed Ruff and the affected group-label, policy, training, replay and native-engine tests: `66 passed in 44.39s`.
- The retained native case `native-case-047b9e12-31d2-4dfc-931f-e32b441a5c14` remains available in prior Train diagnostic artifacts and prior reports for actual-fill cashflow reconciliation. This change does not claim a new replay-to-native economic result.

## Evidence boundary and next stage

The independent Meta protocol preflight passed its registration checks but returned `REGISTERED_FUTURE_SOURCE`, with `completed_screen_windows=0` and `market_payload_read=false` for `2026-12-01` through `2026-12-19`. No new market model fit, calibration, native screen, or profitability claim is available from this run. Continue engineering work against the frozen contracts; run the registered study only when its complete authorized source is available and the preflight confirms it.
