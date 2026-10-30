# Post-gate native override attribution (2026-10-30)

Status: `DIAGNOSTIC_ATTRIBUTION_FIX_VERIFIED; NO_MARKET_RESULT`.

## Correction

The Meta evaluation now keeps a positive model-gate score classified as accepted even when a higher-priority hard-risk override prevents an order. Those candidates are reported separately as accepted-but-blocked. The sequential score funnel records post-gate native constraints before single-position arbitration, so a risk override is not mislabeled as a model rejection or as an arbitration loss.

The native-selected population is required to be a subset of model-accepted candidates that survived post-gate overrides. The study runner supplies per-reason override counts, including `HARD_RISK_OVERRIDE` and owner-readiness failures.

## Validation

- Ubuntu 24.04 / Python 3.14.7 score, economic quality, native-engine and training end-to-end suite: **20 passed**.
- Ruff passed on the changed package modules/tests and on the study runner (`E402` ignored for its existing path bootstrap).
- `git diff --check` passed.

Regression coverage verifies a model-approved candidate blocked by hard risk remains in model acceptance, receives a separate override attribution, and cannot count as native selected. This is diagnostic correctness only; no market payload or profitability evidence was produced.
