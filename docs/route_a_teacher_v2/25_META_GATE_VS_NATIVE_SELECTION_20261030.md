# Meta gate versus native selection (2026-10-30)

Status: `ENGINEERING_DIAGNOSTIC_FIX_VERIFIED; ECONOMIC_SCREEN_NOT_YET_AVAILABLE`.

## Correction

Candidate-quality summaries previously classified only `CandidateDecision.selected` rows as accepted. That flag means the candidate survived both the model gate and single-position native arbitration. As a result, a model-approved candidate that lost arbitration was included in the model-rejected economic segment, which could bias the central accepted-versus-rejected quality comparison.

The summary now reports separate populations:

- `accepted`: candidate passed the frozen model gate (`LONG` or `SHORT`).
- `rejected`: candidate did not pass the model gate or had a scoring/scope rejection.
- `accepted_not_native_selected`: model-approved candidate removed by native arbitration/owner readiness.
- `native_selected`: the single candidate bound to the native entry action.

A native-selected candidate that did not pass the model gate is treated as an invariant violation. Per-disposition breakdowns and prediction metrics remain available; paired Base/Stress execution comparison still applies only to native-selected candidates.

## Validation

- Ubuntu 24.04 / Python 3.14.7: `test_meta_evaluation.py` and `test_route_a_meta_native_engine.py`: **6 passed**.
- The regression includes a model-rejected candidate, a model-approved arbitration loss, and a censored arbitration loss; each lands in the correct distinct denominator.
- Read-only independent Meta source preflight: `REGISTERED_FUTURE_SOURCE`, zero completed screen windows, `market_payload_read=false`, SHA-256 `b731f1e9f3d0eb7bb62f2fcdf4a76b609ff6452e3e8c146a8538b2fc25cc1845`.

This corrects diagnostic attribution only. It adds no market evidence and makes no teacher-profitability claim.
