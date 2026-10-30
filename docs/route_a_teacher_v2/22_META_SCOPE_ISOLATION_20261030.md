# Meta teacher scope isolation (2026-10-30)

Status: `D1_DEVELOPMENT_DIAGNOSTIC_ONLY`. This change fixes state isolation in the Meta/native adapter. It does not change candidate recipes, model thresholds, market scope, or economic qualification.

## Change

- Meta state scope now binds `episode_id`, `source_id`, `fold_id`, and `feature_spec_hash`.
- A scope transition while flat with no pending order resets all candidate and owner teacher rolling state, while retaining the current decision/native receipt trace.
- A scope transition with an active position or pending order fails closed when no recovery intervention is due.
- If hard recovery is due during an active-state scope transition, the recovery action remains executable; candidate and teacher diagnostics are skipped and the reason is recorded. Rolling state is not advanced with observations from the new scope.
- Clarified types in the ordinary scoring and native-owner branches; this fixes existing Mypy errors in the module without changing runtime selection behavior.

## Validation

- Ubuntu 24.04 / Python 3.14.7 targeted candidate, label, model, Meta, native recorder, rollout, and Gym suite: `53 passed`.
- Ruff on the changed source and test modules: passed.
- Mypy on `meta_gated_teacher.py`: passed (`Success: no issues found in 1 source file`).
- Regression cases cover flat feature-schema reset with trace retention, hard-risk priority on active scope transition without advancing teacher state, and fail-closed behavior when the same transition has no recovery override.

## Limits

These checks establish state-scope behavior and synthetic/native-engine integration only. No new market payload was read; no real Train fit, calibration, independent economic screen, or positive net-return evidence was produced. Teacher qualification remains unproven.
