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
- The shared `BaseTeacher1m` state machine also tracks `feature_spec_hash`; direct original-teacher runs now reset rolling/cross/exit state when the observation schema changes, not only when wrapped by Meta or the candidate collector.
- Clarified types in the ordinary scoring and native-owner branches; this fixes existing Mypy errors in the module without changing runtime selection behavior.
- The normal Meta path now handles candidate research failures independently from native position management. A per-candidate scoring exception records `REJECTED_SCORING_ERROR` and its exception detail, while other candidates continue through arbitration. If candidate collection fails, a flat account stays flat; an already-owned position still receives its original teacher HOLD/EXIT decision. Hard recovery is still checked first.
- Entry-score diagnostics now include unscored candidates in the total denominator and overall coverage. In addition to retaining the existing overlapping gate counts, they report the ordered `scored → account-eligible → in-support → above-frozen-margin → model-gate-accepted → native-selected` funnel, first failed-stage counts, stage coverage, and explicit unscored rejection dispositions. The policy margin and support rule are unchanged.
- The diagnostics distinguish model-gate acceptance from the final single-position native selection. Opposing directions or multiple candidates may pass the model gate, but only the selected candidate counts toward native-selected coverage; arbitration and owner-readiness losses are reported separately with per-candidate dispositions.

## Validation

- Ubuntu 24.04 / Python 3.14.7 targeted eight-teacher lifecycle/signal/ROC, candidate, label, model, Meta, native recorder, rollout, and Gym suite: `141 passed`.
- After the candidate error-path change, the same joint lifecycle/candidate/Meta/native rollout suite: `144 passed`.
- Cross-module candidate, counterfactual-label, limit-entry/censor, model fit/calibration, native case funnel, native engine, study runner and full synthetic end-to-end screen suite: `88 passed`.
- After adding the ordered gate-funnel diagnostic and runner integration, focused policy, Meta, evaluation, study-runner and synthetic native end-to-end tests: `40 passed`.
- After separating model-gate passes from the final native-selected candidate, the same focused policy/Meta/evaluation/runner/end-to-end suite: `41 passed`.
- Ruff on the changed source and test modules: passed.
- Mypy on `meta_gated_teacher.py`: passed (`Success: no issues found in 1 source file`).
- For this gate-funnel change, Ruff and Mypy pass on `entry_meta_policy.py` and its tests; Ruff passes on the QA runner with its existing `E402` path-setup imports ignored. Standalone Mypy on the runner reports existing external dependencies without `py.typed` metadata (`rles_orchestrator` / `rles_research`).
- Regression cases also cover candidate scoring exceptions, collection failure with an active-position exit, and fail-closed collection failure for a flat account. The complete suite retains coverage for direct base-teacher schema reset, independent collector reset, hard-risk priority on active scope transition, out-of-scope/not-yet-available model rejection, consensus owner fallback and native receipt propagation.

The read-only protocol preflight on 2026-10-30 reported `REGISTERED_FUTURE_SOURCE`, with `completed_screen_windows: 0`, `market_payload_read: false`, and the registered interval `2026-12-01T00:00:00Z` to `2026-12-19T00:00:00Z` for protocol `route-a-meta-x1-train-20261201-20261219-v1`. Protocol SHA-256: `b731f1e9f3d0eb7bb62f2fcdf4a76b609ff6452e3e8c146a8538b2fc25cc1845`.

The retained development artifact `.artifacts/route-a-meta-development-20261029/native-vs-counterfactual.json` identifies the same completed case `native-case-047b9e12-31d2-4dfc-931f-e32b441a5c14`: modeled reference-price net cash `-0.422069582 USDT`, native actual-fill net cash `-0.423483520 USDT`, and reconciled difference `-0.001413938 USDT`. It records the reference/actual fill-price, modeled slippage, fee and funding components separately. The case is consumed historical development evidence with `SCENARIO_ONLY` status; it is an economic-basis reconciliation, not an independent profitability result.

## Limits

These checks establish state-scope behavior and synthetic/native-engine integration only. No new market payload was read; no real Train fit, calibration, independent economic screen, or positive net-return evidence was produced. Teacher qualification remains unproven.
