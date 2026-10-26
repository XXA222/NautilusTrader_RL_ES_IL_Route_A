# S06 — Native executioner and bounded smoke

Status: **PASS — engineering/accounting smoke only**. This completes S06; it is not an alpha, G1, or G2 result. No Test/Holdout market rows or raw monthly packages were opened.

## Frozen run identity

- Protocol: `route-a-1m-multiteacher-train-diagnostic-20261026-v1`
- Pre-registration: `configs/route_a_1m_teacher_native_prereg_20261026.json`
- Pre-registration SHA-256: `96a5f9f8c5eaee9c65830a4bf9e2e44a79b0b6bede721b4813522a859070c036`
- Teacher/exit/scenario: `T01_EMA_9_21 / X1 / base`
- Smoke window: `2024-01-02T00:00:00Z`–`2024-01-03T00:00:00Z`; 61-bar warmup; 1,437 scored decision barriers (three terminal/drain bars reserved)
- Source snapshot: `62239c22c3d069657ecb39ea109cffd1032a3dd039e3b46ee57db5fac0a77d21`, marker SHA-256 `9e5fb5a628f4ca849aeafbdc0dce50545b1d43c2fa1642bab005cef1469728b2`
- Registered physical read scope: `[1703894460000000000, 1713225660000000000)`; source monthly packages were 2023-12 through 2024-04. The existing Test/Holdout monthly-source guard passed before row decoding.
- Source audit: `DQ-B`, finding `conservative_availability`. The account fee/margin assumptions remain scenario rules; economics are `SCENARIO_ONLY`.

The legacy host's `causal_mtf_v7` profile requires Binance flow fields absent from this 0.4.0 snapshot. The runner therefore uses the existing `causal_mtf_v5` host profile internally, while the teacher receives only the dedicated causal 1m indicator map from `StreamingIndicators1m`. No 5m/15m/1h/4h/1d alignment filter is used. The host action and account path remains the existing Gym → Route A provider → M2.2/Nautilus engine; no second order matcher or ledger was introduced.

## Files added

- `qa/run_route_a_1m_teacher_native.py` — source preflight, registered Train read, bounded native smoke, and staged S05 hook.
- `qa/verify_route_a_1m_teacher_results.py` — fail-closed decision-clock, M2.2, lifecycle, funnel, and native-account reconciliation.
- `qa/test_route_a_1m_teacher_native.py` — preregistration integrity and native fill/slippage checks.

## Commands and actual results

Ubuntu environment: Python 3.14.7 in `.venv-ubuntu`.

```bash
.venv-ubuntu/bin/python qa/run_route_a_1m_teacher_native.py \
  --dry-validate-source --teacher T01_EMA_9_21 --exit X1 --scenario base \
  --prereg configs/route_a_1m_teacher_native_prereg_20261026.json \
  --output .artifacts/route-a-1m-teacher-native-s06-source-audit-run01
```

Actual: registered source audit completed; grade `DQ-B`; finding `conservative_availability`; protected-month guard passed. The reader reserved the pre-registered Train scope before opening rows.

```bash
.venv-ubuntu/bin/python qa/run_route_a_1m_teacher_native.py \
  --teacher T01_EMA_9_21 --exit X1 --scenario base \
  --prereg configs/route_a_1m_teacher_native_prereg_20261026.json \
  --output .artifacts/route-a-1m-teacher-native-s06-run06
.venv-ubuntu/bin/python qa/verify_route_a_1m_teacher_results.py \
  --output .artifacts/route-a-1m-teacher-native-s06-run06
```

Actual final smoke/verifier result: 1,437/1,437 decision rows and M2.2 barriers verified; 57 submitted and filled order IDs; zero liquidations; 29 cases (28 ordinary complete, one terminal-censored). Every decision had a one-minute interval, `latest_1m_close_ts_ns <= latest_1m_available_at_ns <= decision_ts_ns`, and each execution clock followed feature availability. The report and CSVs are in `.artifacts/route-a-1m-teacher-native-s06-run06/`.

Per-minute signal/action totals: 29 `SHORT_ENTRY`, 28 `EXIT_SHORT`, 1,380 `NONE`; 29 `OPEN_SHORT`, 692 `HOLD`, 28 `EXIT`, 688 `FLAT`; 57 `SUBMIT`, 1,380 `NOOP`. This run recorded no risk or order-state veto. Case/account fees, slippage, funding, and equity change reconciled within the verifier tolerance.

| Native account measure | Base/X1 smoke |
|---|---:|
| Starting equity | 10,000.00000000 USDT |
| Ending equity | 9,994.71211331 USDT |
| Net return | -0.0528789% |
| Maximum drawdown | 0.0753612% |
| Fees | 7.21928930 USDT |
| Slippage cost | 1.46204000 USDT |
| Funding | +0.15804261 USDT |

These are historical replay outcomes under the pre-registered scenario recipe, not realized exchange economics and not an eligible-alpha finding. The negative smoke return is reported as observed; no teacher or threshold was selected from it.

Targeted test matrix:

```bash
.venv-ubuntu/bin/python -m pytest -q \
  qa/test_route_a_1m_teacher_native.py \
  packages/route_a_imitation/tests/test_teacher_1m_funnel.py \
  packages/route_a_imitation/tests/test_teachers_1m_signals.py
.venv-ubuntu/bin/ruff check packages qa tests
.venv-ubuntu/bin/python -m mypy packages
```

Results: `31 passed in 18.83s`; Ruff `All checks passed!`; MyPy `Success: no issues found in 244 source files`. The verifier returned `VERIFIED_SCENARIO_ONLY_SMOKE`.

## Stopped/rejected attempts retained as evidence

- The first native attempt stopped before replay because `causal_mtf_v7` required source flow fields that this snapshot does not preserve. It was not reclassified as a data-quality pass; the unused host profile was changed to the compatible v5 profile and the 1m teacher recipe stayed fixed.
- An early hook attempt failed same-step planner reconciliation because it used the Gym float32 action projection instead of the full-precision teacher advice in the S05 requested-exposure field. The record now retains the teacher request; risk attribution compares against the host trace's own requested action and only calls a veto when an entry is actually suppressed without a pending order.
- Earlier diagnostic output directories (`run01`–`run05`) were not used for acceptance. The immutable accepted evidence is `run06`, whose funnel, decision rows, native cases, and account totals pass the verifier.

## Gate disposition

S06 demonstrates that one 1m teacher can be bound to the existing native account path with causal features and reconciled per-minute evidence. It does not establish Alpha eligibility, and the output stays `D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2` / `SCENARIO_ONLY`. Continue only to S07's preregistered Train comparison; do not promote this smoke, train a student, or infer a historical Oracle.
