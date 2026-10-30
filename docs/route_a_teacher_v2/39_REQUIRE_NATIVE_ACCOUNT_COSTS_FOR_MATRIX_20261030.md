# Require native account costs in comparison matrix — 2026-10-30

## Change

The native comparison matrix previously allowed a run without account-level
fee, slippage, and funding totals to be marked `AVAILABLE`. The comparator now
rejects any matrix cell that lacks all three native account cost facts before
computing its economic metrics. It then reconciles those account totals and
equity change with all retained trade cases.

Synthetic comparison fixtures now include explicit zero account costs where
they represent a truly flat account. The window-consistency fixture that
previously claimed a positive equity change with no trade case was corrected
to retain the matching native case.

## Verification

- Ruff passed for the changed comparison module and tests.
- Ubuntu 24.04 / Python 3.14.7: comparison, study runner, native engine, and
  native training end-to-end tests: **20 passed**.
- Regression coverage rejects both absent account-cost facts and account/case
  cash-flow mismatches.

## Evidence boundary

This prevents incomplete cost records from entering the comparison matrix. It
does not establish positive returns; new preregistered Train windows are still
required for economic evaluation.

