# Meta teacher scope isolation (2026-10-30)

Status: `D1_DEVELOPMENT_DIAGNOSTIC_ONLY`. This change fixes state isolation in the Meta/native adapter. It does not change candidate recipes, model thresholds, market scope, or economic qualification.

## Change

- Meta state scope now binds `episode_id`, `source_id`, `fold_id`, and `feature_spec_hash`.
- A scope transition while flat with no pending order resets all candidate and owner teacher rolling state, while retaining the current decision/native receipt trace.
- A scope transition with an active position or pending order fails closed when no recovery intervention is due.
- If hard recovery is due during an active-state scope transition, the recovery action remains executable; candidate and teacher diagnostics are skipped and the reason is recorded. Rolling state is not advanced with observations from the new scope.
- Candidates whose source/fold/schema differs from the frozen policy, or whose decision precedes calibration availability, are retained as explicit `REJECTED_MODEL_SCOPE_MISMATCH` / `REJECTED_MODEL_NOT_AVAILABLE` dispositions. They do not raise a model-scoring error or enter arbitration.
- For a selected consensus candidate, execution ownership is chosen from all contributing teachers whose existing native lifecycle is ready for the candidate direction. A not-ready first-listed teacher no longer blocks another ready contributor from executing the same grouped candidate.
- Native entry receipts and the existing case funnel now record that actual selected owner and its reason code, while trade cases continue to retain the full contributing teacher set. This keeps execution provenance consistent when a consensus owner is not the first sorted teacher.
- Native receipts for a hard-risk action during an active-state scope transition are still retained, but they are not sent into the old owner teacher's lifecycle callback. This prevents new-scope execution facts from mutating the previous scope's pending reversal state.
- The standalone eight-teacher `OneMinuteCandidateCollector` now independently binds episode/source/fold/feature-schema and resets all rolling indicator state on any scope transition. This also protects the native candidate recorder and Train collection path when it runs outside `MetaGatedTeacher`.
- Clarified types in the ordinary scoring and native-owner branches; this fixes existing Mypy errors in the module without changing runtime selection behavior.

## Validation

- Ubuntu 24.04 / Python 3.14.7 targeted candidate, label, model, Meta, native recorder, rollout, and Gym suite: `58 passed`.
- Ruff on the changed source and test modules: passed.
- Mypy on `meta_gated_teacher.py`: passed (`Success: no issues found in 1 source file`).
- Regression cases cover flat feature-schema reset with trace retention, independent collector schema reset, hard-risk priority on active scope transition without advancing teacher state or invoking the old owner's native callback, fail-closed behavior when the same transition has no recovery override, explicit rejection of out-of-scope or not-yet-available model scores, fallback to a ready native owner within a grouped consensus candidate, and propagation of that owner into native receipts and the case funnel.

The read-only protocol preflight reported `REGISTERED_FUTURE_SOURCE`, with `completed_screen_windows: 0` and `market_payload_read: false` for protocol `route-a-meta-x1-train-20261201-20261219-v1`.

## Limits

These checks establish state-scope behavior and synthetic/native-engine integration only. No new market payload was read; no real Train fit, calibration, independent economic screen, or positive net-return evidence was produced. Teacher qualification remains unproven.
