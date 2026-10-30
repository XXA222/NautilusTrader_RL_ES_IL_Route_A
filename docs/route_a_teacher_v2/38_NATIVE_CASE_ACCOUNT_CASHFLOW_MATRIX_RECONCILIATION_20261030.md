# Reconcile native cases against account cash flows — 2026-10-30

## Change

The preregistered native comparison previously computed trade-case metrics and
attached account-level fees, slippage, funding, and equity change without
checking that the two evidence paths agreed. Each matrix cell with retained
account metrics now reconciles all case net PnL, fees, slippage, and funding
against the account snapshot totals. The case sum includes terminal-censored
cases because their settlement cash flows are part of the native account.

A cell with any difference above the cash tolerance is rejected before it can
be marked `AVAILABLE` or included in strategy comparisons. The output records
the reconciled totals and per-field deltas.

## Verification

- Ruff passed for the changed comparison module and tests.
- Ubuntu 24.04 / Python 3.14.7: candidate quality and comparison tests, Meta
  study runner, native engine, and native training end-to-end tests: **20
  passed**.
- Regression coverage verifies both a reconciled account/case cell and a
  mismatched fee total that must fail closed.

## Evidence boundary

This validates internal consistency between native trade cases and the account
ledger. It does not establish positive net returns. Future preregistered Train
screen windows remain required for economic comparisons.
