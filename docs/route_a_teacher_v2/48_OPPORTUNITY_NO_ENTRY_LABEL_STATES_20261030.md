# Opportunity no-entry labels — 2026-10-30

## Finding

The eight-teacher `conditional_outcome_labels` path already retained `UNFILLED`, `CANCELLED`, `MISSED`, and `RIGHT_CENSORED`, but the general `opportunity_labels` path accepted only a filled trade replay. Its outcome DTO represented every missing return as censoring. That erased limit-order disposition and could make the two research label paths disagree about the candidate denominator.

## Change

`NoEntryOpportunityReplay` is now the shared replay receipt used by both label paths. `OpportunityOutcome` records complete outcomes, right-censored outcomes, and no-entry statuses separately, with a bounded `observed_until_ts_ns`. No-entry outcomes cannot carry a return, exit time, fees, slippage, or funding. Right-censored outcomes retain a separate censor clock. Batch counts now distinguish complete, censored, no-entry, and all incomplete candidates. Model fitting still uses complete outcomes only.

## Verification

- No-entry state regression cases cover `UNFILLED`, `CANCELLED`, `MISSED`, and `RIGHT_CENSORED`; all remain in the denominator and none is assigned a synthetic zero return.
- Label, model, candidate, training, and native Meta integration tests: **55 passed**. Ruff and `git diff --check` passed.
- Read-only preregistration preflight: `REGISTERED_FUTURE_SOURCE`, 0 completed screen windows, and `market_payload_read: false`.

## Evidence boundary

This aligns label representation and preserves candidate states. It is not a market result, does not validate the limit-touch execution assumption against a native limit fill, and does not qualify a teacher.
