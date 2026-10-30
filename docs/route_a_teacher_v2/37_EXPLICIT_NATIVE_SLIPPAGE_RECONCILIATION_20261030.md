# Explicit native slippage in case reconciliation — 2026-10-30

## Change

`reconcile_counterfactual_to_native()` now requires and reports the native
case's finite, nonnegative `slippage_cost`. The output explicitly labels this
amount as already included in the actual fill prices and therefore not
subtracted from native cash flow a second time.

The regression uses retained completed Train case
`native-case-047b9e12-31d2-4dfc-931f-e32b441a5c14`. It checks the known native
net cash `-0.423483520 USDT`, reported slippage `0.05014 USDT`, modeled
reference-price net cash `-0.422069582 USDT`, and difference
`-0.001413938 USDT`, including the existing price/reference, cost, and funding
decomposition.

## Verification

- Ruff passed for the reconciliation module and test.
- Ubuntu 24.04 / Python 3.14.7: native case economics, native Meta funnel,
  native engine, and native training end-to-end tests: **13 passed**.
- The retained native case matches the archived reconciliation artifact.

## Evidence boundary

This checks that counterfactual and native cash-flow bases are reconciled
without double-counting slippage. The case is consumed historical Train
diagnostic evidence (`SCENARIO_ONLY`), not an independent profitability result.

