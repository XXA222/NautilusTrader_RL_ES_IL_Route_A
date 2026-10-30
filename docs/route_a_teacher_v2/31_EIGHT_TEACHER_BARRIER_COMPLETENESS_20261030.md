# Eight-teacher native barrier completeness — 2026-10-30

## Change

`NativeCandidateRecorder.require_complete()` now checks every retained native
barrier for exactly one signal row from each registered T01–T08 teacher and
rejects any barrier with a teacher signal error. A recorder can still retain
and export the failed barrier and unaffected candidates for diagnosis, but the
episode cannot pass the complete-candidate-dataset gate.

This closes a gap where `require_complete()` previously rejected missing
barriers but could accept a barrier whose `CandidateCollection` contained a
teacher exception. The aggregate coverage report already marked that case
incomplete; the episode-level gate now agrees with it.

## Verification

- `test_native_candidate_recorder.py` and `test_candidate_collector.py`: **13 passed**.
- `test_meta_gated_teacher.py`, `test_meta_native_gym.py`,
  `test_route_a_meta_study_runner.py`, and `test_route_a_meta_native_engine.py`:
  **30 passed**.
- A regression case forces T01 to fail, verifies other teacher candidates are
  retained, and verifies the recorder refuses to certify the data as complete.

## Evidence boundary

This is an engineering completeness check, not a market-result or profitability
claim. Native source screening remains subject to the active registered data
window and its access preflight.
