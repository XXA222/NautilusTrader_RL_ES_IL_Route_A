# Incomplete teacher coverage blocks candidate quality — 2026-10-30

## Change

The native meta-study runner previously recorded incomplete T01–T08 signal
coverage but continued to replay and summarize candidate counterfactual
outcomes. A missing teacher changes the candidate population, so accepted vs.
rejected quality summaries from that run are not a complete eight-teacher
comparison.

`_score_diagnostics()` now retains the per-minute signal/error coverage and
candidate score funnel, sets `quality_status` to
`INCOMPLETE_EIGHT_TEACHER_SIGNAL_COVERAGE`, and returns before constructing
candidate outcome labels. Complete coverage is explicitly marked
`COMPLETE_EIGHT_TEACHER_SIGNAL_COVERAGE` before quality replay.

## Verification

- Ruff on the changed runner and test: passed (`E402` ignored for the runner's
  existing import-path bootstrap).
- Ubuntu 24.04 / Python 3.14.7:
  `tests/test_route_a_meta_study_runner.py`,
  `tests/test_route_a_meta_native_engine.py`, and
  `tests/test_route_a_meta_native_training_e2e.py`: **11 passed**.
- Regression case asserts that incomplete coverage preserves disposition
  counts while never calling the outcome replay path.

## Evidence boundary

This prevents a partial teacher population from being reported as complete
counterfactual quality evidence. It does not establish positive expected
returns or native economic performance.

