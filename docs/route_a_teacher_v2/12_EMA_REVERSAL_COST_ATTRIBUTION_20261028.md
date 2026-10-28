# 12 — EMA reversal side and cost attribution (2026-10-28)

## Scope

This is a re-aggregation of the existing consumed-S06 paired report at `.artifacts/route-a-ema-reversal-task1/paired_base_stress_summary.json`; it does not rerun the native engine or read a market source. All figures remain `DEVELOPMENT_DIAGNOSTIC_ONLY` / `SCENARIO_ONLY`.

Each cell summarizes ordinary complete cases as `trade count / gross cash PnL before fees and funding / fees / funding / net case PnL` in USDT. Account endpoint returns, which can also include terminal-censored mark-to-market, remain in the paired account report. Gross cash PnL includes execution slippage; the slippage field is an attribution and must not be subtracted a second time. Completed-case net reconciles as gross cash PnL minus fees plus funding.

| Window | Scenario | Legacy X1 active side | Reverse LONG | Reverse SHORT |
|---|---|---|---|---|
| February | Base | LONG: 978 / +12.37 / 243.19 / −1.71 / −232.53 | 868 / −6.04 / 215.80 / −1.30 / −223.14 | 867 / −74.65 / 215.55 / +1.43 / −288.78 |
| February | Stress | LONG: 978 / −182.39 / 486.38 / −1.71 / −670.49 | 868 / −182.40 / 431.59 / −1.02 / −615.02 | 867 / −244.98 / 431.11 / +1.07 / −675.01 |
| March | Base | LONG: 1,098 / −54.64 / 272.48 / −3.66 / −330.78 | 969 / −35.17 / 240.47 / −3.58 / −279.21 | 968 / −66.36 / 240.22 / +3.96 / −302.61 |
| March | Stress | LONG: 1,098 / −272.26 / 544.96 / −3.61 / −820.83 | 969 / −226.60 / 480.94 / −3.18 / −710.72 | 968 / −254.21 / 480.43 / +3.41 / −731.23 |
| April first half | Base | SHORT: 457 / +40.42 / 113.44 / +0.82 / −72.20 | 402 / +4.93 / 99.81 / −0.43 / −95.31 | 402 / +38.54 / 99.78 / +0.70 / −60.54 |
| April first half | Stress | SHORT: 457 / −50.29 / 226.89 / +0.82 / −276.36 | 402 / −74.89 / 199.62 / −0.43 / −274.94 | 402 / −41.24 / 199.56 / +0.70 / −240.10 |

## What the paired cases show

- In Base February and March, both sides of the reversal teacher have negative gross cash PnL before fees and funding. The reversal signal does not show a positive price edge in those blocks.
- In Base April, both reversal sides have positive gross cash PnL, but fees alone exceed that amount. These samples show cost erosion, not a net edge.
- In Stress, both reversal sides have negative gross cash PnL in all three blocks, before subtracting fees.
- No reversal side is net profitable in any of the six block/scenario pairs. The account-level reversal policy also underperformed its paired exit-only path in all six pairs, as recorded in `10_DEVELOPMENT_STATUS_20261027.md`.

This descriptive split attributes the failure to both weak/inconsistent signal quality and fees: fees erase the small Base-April gross gains, while negative gross results dominate the other windows, especially Stress. It does not identify causality or support a profitability claim. The frozen new-source candidate remains unchanged; only a completed, preregistered Train window can provide the next economic evaluation.

## Evidence

- Source report: `.artifacts/route-a-ema-reversal-task1/paired_base_stress_summary.json`
- Source report SHA-256: `84b9a81e4aff80e26a2b8b5b2715e5dc08bea3a1e81ef68156a3b759d13e21b5`
- Recomputed all six paired reports: every side's completed-case identity reconciles; all six Stress reversal-side gross cash PnLs are negative; reversal account net underperforms exit-only in all six pairs.
- Current preregistered window-0 scoring command, after its complete source import and fresh preflight: `.venv-ubuntu/bin/python qa/run_route_a_teacher_student_train_screen_20261026.py --window-index 0`
- The source-access audit records that the complete import must include dates through 2026-10-29; no partial-window score is valid.
