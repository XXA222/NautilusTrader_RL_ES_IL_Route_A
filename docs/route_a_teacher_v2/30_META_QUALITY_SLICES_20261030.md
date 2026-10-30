# Meta candidate economic quality slices — 2026-10-30

## Change

`meta_evaluation.summarize_candidate_quality` now breaks complete model-counterfactual outcome quality into model-accepted and model-rejected slices by:

- candidate side (LONG / SHORT);
- each contributing teacher ID;
- teacher consensus count.

Every slice uses the same explicit denominator fields as the top-level summary: candidate count, complete outcomes, no-outcome/censored count, positive/negative/breakeven counts, mean positive/negative net bps, overall mean net bps, and win rate. Teacher slices are flagged as overlapping because one consensus event can appear under multiple teachers. These rows remain `MODEL_COUNTERFACTUAL`, never native trade evidence.

Each slice also reports complete disposition and `REJECTED_*` reason counts, separating model rejection such as `OUT_OF_SUPPORT` from downstream arbitration such as `BY_ARBITRATION`.

## Verification

- `pytest packages/route_a_imitation/tests/test_meta_evaluation.py tests/test_route_a_meta_native_engine.py tests/test_route_a_meta_study_runner.py` — **16 passed**.
- Ruff on the changed implementation and test — **passed**.
- `git diff --check` — **passed**.

Tests verify the accepted and rejected LONG slices, empty SHORT slice, censored accepted denominator, per-teacher overlap semantics, and consensus-count partition.

No new market data was read. This enables the preregistered economic analysis to distinguish signal quality by direction, teacher, and consensus; it does not provide profitability evidence by itself.
