# Candidate outcome status coverage — 2026-10-30

## Change

Candidate-quality segments now report their complete-outcome coverage and separate counts for every observed outcome status, including `RIGHT_CENSORED`, `UNFILLED`, `CANCELLED`, and `MISSED`. The accepted/rejected, side, teacher, consensus, and disposition slices share this accounting. Incomplete outcomes remain excluded from return averages and are not coerced to zero.

## Evidence

- Updated `meta_evaluation.py` and `test_meta_evaluation.py`.
- Ruff passed; candidate quality and native Meta runner/engine/e2e tests passed: `20 passed in 42.57s`.
- The regression fixture confirms accepted candidates can have one complete outcome and one right-censored outcome, with 50% complete coverage and the censor reason visible in the accepted slice.

## Boundary

This improves denominator interpretation; it does not establish economic advantage. The independent Train market source is still unavailable, so no real fit, calibration, screen, or profitability result is claimed.
