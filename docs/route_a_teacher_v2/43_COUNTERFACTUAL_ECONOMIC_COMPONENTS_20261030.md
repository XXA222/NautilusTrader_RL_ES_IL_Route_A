# Counterfactual outcome economics — 2026-10-30

Status: `LABEL_ECONOMICS_DECOMPOSITION_VERIFIED; INDEPENDENT_ECONOMIC_SCREEN_BLOCKED_BY_FUTURE_SOURCE`.

## Change

The frozen-exit opportunity-label path now retains each complete `OpportunityOutcome` as reference-price gross return, fees, modeled slippage, signed funding cashflow, and net return, all in the declared reference-entry-notional bps basis. It enforces:

`net = gross - fees - modeled_slippage + funding_cashflow`

Fees and modeled slippage must be finite and nonnegative. A censored outcome cannot carry complete-trade components or be converted to zero. The grouped candidate label path retains the same decomposition, and candidate-quality summaries report component means and per-candidate Base/Stress components alongside net outcomes.

The counterfactual reference prices exclude slippage, so modeled slippage is subtracted once. Native cashflow reconciliation continues to use actual fill prices, which already include slippage, and does not subtract it again. Both outcome paths remain `MODEL_COUNTERFACTUAL` / `SCENARIO_ONLY`.

## Verification

- Ubuntu 24.04 / WSL2 / Python 3.14.7 full `packages/route_a_imitation/tests` run completed with exit code 0 (100% of collected tests passed).
- Ruff passed for all modified source and test modules.
- `git diff --check` passed.
- The retained, consumed Train case `native-case-047b9e12-31d2-4dfc-931f-e32b441a5c14` still reconciles: native net cash `-0.423483520 USDT`; reference-price modeled net cash `-0.422069582 USDT`; difference `-0.001413938 USDT`. Actual fills include slippage. This is a same-case cashflow-basis check, not evidence of profitability.
- Read-only protocol preflight returned `REGISTERED_FUTURE_SOURCE`, `completed_screen_windows: 0`, and `market_payload_read: false` for `route-a-meta-x1-train-20261201-20261219-v1` (protocol SHA-256 `b731f1e9f3d0eb7bb62f2fcdf4a76b609ff6452e3e8c146a8538b2fc25cc1845`).

## Limits and next work

This closes the itemized economics contract for the frozen-exit label DTOs. It does not establish that the Logistic + Ridge gate improves trade selection or earns positive net returns. The independent source is future-dated, so model fitting/calibration and native economic screening remain pending. Continue engineering against the saved candidate, label, model, and native lifecycle contracts; run the preregistered study only after its complete authorized source becomes available and preflight permits it.
