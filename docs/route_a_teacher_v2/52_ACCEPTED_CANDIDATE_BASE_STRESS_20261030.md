# Base/Stress evaluation of accepted opportunities — 2026-10-30

## Finding

The native Meta screen previously paired Base/Stress counterfactual paths only for the candidate ultimately selected by the one-position arbitrator. Model-accepted alternatives, including same-minute alternatives and candidates whose native owner was not ready, were absent from this cost-resilience view. That made it harder to distinguish model ranking quality from execution selection.

## Change

The screen now replays every direction-consistent model-accepted candidate under both registered cost schedules. The report separately aggregates the complete model-accepted set and its native-selected subset, retains accepted-but-not-selected and censored rows, and explicitly states that overlapping candidate labels are not independent trials. Direction-mismatch rejections are excluded from the accepted set.

## Verification

- Added a case with two accepted candidates, one native-selected candidate, and one right-censored unselected candidate; both remain in the paired denominator while selected metrics remain separate.
- Meta evaluation, study runner, native engine, and training integration tests: **25 passed**. Ruff (with the runner's existing E402 path-bootstrap exception) and `git diff --check` passed.

## Evidence boundary

This repairs the evaluation denominator. No registered source payload was read, and this is not evidence of Base/Stress profitability.
