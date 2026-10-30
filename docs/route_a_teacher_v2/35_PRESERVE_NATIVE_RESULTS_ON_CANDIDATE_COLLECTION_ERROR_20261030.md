# Preserve native results on candidate collection error — 2026-10-30

## Change

The Meta study runner previously raised immediately when a decision contained
`candidate_error`. That discarded the study cell's serialized native account
result even when the rollout and M2.2 order verification had completed.

The runner now records candidate error counts and error types in the candidate
summary. Candidate collection errors prevent counterfactual quality replay and
set `quality_status` to `CANDIDATE_COLLECTION_ERROR` when teacher signal
coverage is otherwise complete. If coverage is incomplete, the more specific
`INCOMPLETE_EIGHT_TEACHER_SIGNAL_COVERAGE` status takes precedence. The native
account evidence can still be serialized with the failure clearly attached.

## Verification

- Ruff on the changed runner and tests passed (`E402` ignored for the runner's
  existing import-path bootstrap).
- Ubuntu 24.04 / Python 3.14.7:
  `tests/test_route_a_meta_study_runner.py`,
  `tests/test_route_a_meta_native_engine.py`, and
  `tests/test_route_a_meta_native_training_e2e.py`: **12 passed**.
- Regression test confirms a collection error is retained and does not enter
  outcome replay.

## Evidence boundary

This preserves diagnostic and native account records after a research-side
collection error. A cell with the error remains incomplete research evidence
and cannot establish candidate quality or profitability.
