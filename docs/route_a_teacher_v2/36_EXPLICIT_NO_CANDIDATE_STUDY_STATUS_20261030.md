# Explicit no-candidate study status — 2026-10-30

## Change

When all eight teachers have complete signal coverage but none proposes an
entry, the Meta study summary now records
`candidate_quality_status=NO_CANDIDATES`, emits an empty outcome list, and
includes explicit null values for paired Base/Stress candidate comparison and
native case reconciliation. This distinguishes a valid zero-candidate
population from missing diagnostic fields.

The summary also gives candidate quality a separate status for incomplete
teacher coverage, candidate collection errors, and available quality replay.
The score funnel remains present, so scored `NO_TRADE` candidates continue to
report their predictions, sample count, and rejection reasons.

## Verification

- Ruff on the changed runner and test passed (`E402` ignored for the runner's
  existing import-path bootstrap).
- Ubuntu 24.04 / Python 3.14.7:
  `tests/test_route_a_meta_study_runner.py`,
  `tests/test_route_a_meta_native_engine.py`, and
  `tests/test_route_a_meta_native_training_e2e.py`: **12 passed**.
- Regression assertions cover explicit zero-candidate output fields.

## Evidence boundary

The zero-candidate status is a population diagnostic, not an economic result.
It does not establish that any candidate model or teacher is profitable.
