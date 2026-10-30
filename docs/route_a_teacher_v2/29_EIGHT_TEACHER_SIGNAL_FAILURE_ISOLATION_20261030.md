# Eight-teacher signal failure isolation — 2026-10-30

## Change

The one-minute candidate collector previously let one teacher exception abort the full eight-teacher pass. It now records an explicit `ERROR` signal and error detail for that teacher, continues evaluating the other seven, and preserves their candidates for research diagnostics.

Meta entry is fail-closed on an incomplete teacher union: candidates remain in the trace with `TEACHER_SIGNAL_INCOMPLETE`, are not scored, and cannot be selected for a native order. Existing-position owner and hard-risk paths remain active. Candidate coverage output now states whether all eight teachers supplied one signal row per barrier and lists per-teacher errors.

## Verification

- WSL targeted suite: `pytest packages/route_a_imitation/tests/test_candidate_collector.py packages/route_a_imitation/tests/test_meta_gated_teacher.py packages/route_a_imitation/tests/test_native_candidate_recorder.py packages/route_a_imitation/tests/test_native_meta_rollout.py` — **33 passed**.
- Ruff on changed Python modules/tests — **passed**; study runner checked with its existing dynamic-import `E402` pattern ignored — **passed**.
- `git diff --check` — **passed**.

Synthetic failure injection confirms a failing EMA adapter does not suppress a valid ROC candidate; the resulting Meta action is FLAT, retains the candidate and error details, and does not call the model scorer. Native coverage marks the failed teacher and the union incomplete.

This is signal-pipeline resilience evidence only. It does not establish profitability or complete real-data coverage. The preregistered independent Train source remains future-dated and has not been read.
