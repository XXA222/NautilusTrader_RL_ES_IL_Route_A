# F06 — New-source native research

Status: `RUNNER_READY_WARMUP_IMPORTED_FIRST_WINDOW_PENDING`.

A new Train source scope is frozen and registered under `route-a-teacher-student-train-20261023-20261130-v1`; see `05_SOURCE_ACCESS_AUDIT.md` and its parent preregistration/errata files. The two unscored warmup days were acquired and imported under that registration, and a DQ-B scaler was fit only on the frozen warmup interval. The runner for the frozen nine-candidate, four-window, two-scenario, two-seed matrix is implemented. One of five staged imports is present; the first complete scored window ends on 2026-10-30 and has not been scored. The read-only preflight verifies the parent/errata SHA chain, exact teacher registry IDs, split windows, code hashes, and ledger reservation without reading market payload. Its latest status is `PREFLIGHT_PASS_STAGED_SOURCES_PARTIAL_METADATA_NOT_READ`.

Before the new protocol's first scored window, the X2 ATR-exit return path was corrected to store that minute's indicator snapshot before exiting the teacher decision. Without this update, a later bar could compare against a pre-exit snapshot and emit a stale EMA cross. The focused lifecycle test, Ruff, MyPy, and the source preflight pass. The source erratum binds the corrected bytes. The new protocol's scored outcomes remain unobserved.

The correction was then checked separately on two already-consumed S06 Train windows using the frozen 2026-10-26 native preregistration and the exact pre-fix `base.py` hash as the control. Both paired runs use the same snapshot, feature recipe, scaler, account, Base costs, and X2 recipe within each window. All 41,757 and 44,637 decisions respectively passed M2.2; neither run liquidated, and trade-case cashflows reconcile exactly to account equity.

| S06 window | Pre-fix net / max DD | Corrected net / max DD | Net change | Fills, pre-fix → corrected | Unpaired completed-trade net PnL, pre-fix → corrected |
|---|---:|---:|---:|---:|---:|
| `inner-screen-1` | −3.875% / 4.013% | −3.478% / 3.567% | +0.396 pp; DD −0.446 pp | 2,679 → 2,373 | −$58.67 → −$19.06 |
| `inner-screen-2` | −4.625% / 4.638% | −4.021% / 4.035% | +0.603 pp; DD −0.603 pp | 2,709 → 2,385 | −$95.83 → −$35.50 |

In both windows the first changed entry signal was a stale EMA cross immediately after an ATR exit; indicator snapshots match on every compared decision row. The corrected policy avoids 306 fills in the first window and 324 in the second. Shared completed entries had exactly equal holding minutes and net return differences below `1e-8` bps; all observed account gains came from avoiding later unpaired trades, whose aggregate net PnL remained negative but was less negative under the correction. This is a replicated mechanism correction and a favorable D1 account delta, but both corrected account runs still lose money. It is not evidence of profitability, teacher qualification, G1/G2, or generalization. Reports: `.artifacts/route-a-x2-snapshot-fix-task3/comparison.json` and `comparison-inner-screen-2.json`.

The same hash-exact pairing was run under the preregistered Stress costs:

| S06 window | Pre-fix net / max DD | Corrected net / max DD | Net change | Fills, pre-fix → corrected | Fees, pre-fix → corrected |
|---|---:|---:|---:|---:|---:|
| `inner-screen-1` | −9.883% / 9.906% | −8.795% / 8.809% | +1.088 pp; DD −1.096 pp | 3,445 → 2,701 | $666.33 → $590.25 |
| `inner-screen-2` | −10.583% / 10.594% | −9.302% / 9.313% | +1.281 pp; DD −1.281 pp | 3,720 → 2,943 | $672.47 → $592.07 |

Stress also improves the full-account result in both consumed windows, while remaining deeply negative. The common-entry subset has identical holding times and zero median net-return change, but execution economics differ on many matched cases under Stress: in window 1, 197/1,115 matched cases favor the pre-fix run and 118 favor the correction; in window 2, 234/1,088 favor pre-fix and 96 favor the correction. The fixed run's unpaired completed trade sets still lose $51.40 and $81.69 respectively, versus $161.05 and $211.75 for the pre-fix sets. These are selected post-hoc path diagnostics, not independent validation. Reports: `.artifacts/route-a-x2-snapshot-fix-task3/comparison-stress-inner-screen-1.json` and `comparison-stress-inner-screen-2.json`.

Because the snapshot update is in the shared teacher base, the corrected X2 path was screened against exact pre-fix overlays for ROC, Donchian, and Z-score on `inner-screen-1` in both scenarios:

| Teacher | Scenario | Pre-fix net / max DD | Corrected net / max DD | Net change | Fills, pre-fix → corrected |
|---|---|---:|---:|---:|---:|
| ROC | Base | −4.316% / 4.362% | −4.102% / 4.144% | +0.214 pp | 2,980 → 2,878 |
| ROC | Stress | −10.970% / 10.970% | −10.540% / 10.540% | +0.430 pp | 4,318 → 3,895 |
| Donchian | Base | −4.331% / 4.417% | −4.267% / 4.331% | +0.064 pp | 2,905 → 2,857 |
| Donchian | Stress | −10.853% / 10.866% | −10.691% / 10.699% | +0.163 pp | 4,129 → 3,998 |
| Z-score | Base | −2.053% / 2.099% | −2.114% / 2.161% | −0.061 pp | 1,160 → 1,182 |
| Z-score | Stress | −4.649% / 4.680% | −4.759% / 4.790% | −0.110 pp | 1,160 → 1,182 |

All six corrected runs passed M2.2 on all 41,757 decisions, had zero liquidations, and reconciled trade-case cashflows to the native account. The X2 correction modestly improves ROC and Donchian but worsens Z-score; all remain net losers. Thus the code fix repairs the shared snapshot transition, while economic effects depend on the teacher and do not qualify any teacher. Reports are `comparison-{roc,donchian,zscore}-x2-{base,stress}-inner-screen-1.json` under `.artifacts/route-a-x2-snapshot-fix-task3/`.

The native case cashflows were also decomposed to distinguish price direction from execution costs. Estimated pre-slippage price PnL is `net PnL + fees + slippage attribution − funding`; slippage is already embedded in fill prices, so this adds it back once for attribution only.

| Corrected teacher/window | Scenario | Estimated price PnL before slippage, fees, funding | Fees | Slippage attribution | Funding | Net account PnL |
|---|---|---:|---:|---:|---:|---:|
| EMA, window 1 | Base | +$7.82 | $295.13 | $60.18 | −$0.36 | −$347.85 |
| EMA, window 1 | Stress | +$7.41 | $590.25 | $296.30 | −$0.32 | −$879.46 |
| EMA, window 2 | Base | −$47.01 | $296.03 | $60.07 | +$0.97 | −$402.15 |
| EMA, window 2 | Stress | −$42.22 | $592.07 | $296.86 | +$0.95 | −$930.20 |
| ROC, window 1 | Base | +$20.27 | $357.79 | $72.98 | +$0.27 | −$410.23 |
| ROC, window 1 | Stress | +$20.57 | $715.57 | $359.20 | +$0.23 | −$1,053.97 |
| Donchian, window 1 | Base | +$1.27 | $355.33 | $72.45 | −$0.21 | −$426.72 |
| Donchian, window 1 | Stress | −$1.51 | $710.66 | $356.71 | −$0.19 | −$1,069.07 |
| Z-score, window 1 | Base | −$34.29 | $146.94 | $29.94 | −$0.22 | −$211.40 |
| Z-score, window 1 | Stress | −$34.29 | $293.89 | $147.49 | −$0.22 | −$475.89 |

This points to both weak/inconsistent signal edge and overwhelming costs: in window 1, EMA/ROC/Donchian have only $1–$20 estimated gross price PnL before costs, while fees plus slippage exceed $350 Base and $650 Stress. Z-score is already negative before costs, especially on shorts. EMA window 2 is also negative before costs on both sides. These short samples diagnose the current loss mechanism; they do not prove a stable cost-free edge.

Teacher selection and student comparison have not started. No Test/Holdout source was opened.

Historical S07 results remain D1 diagnostics only. Even a passing Train screen remains `SCENARIO_ONLY` and does not prove account-specific profitability or formal G1/G2 qualification.


## S06 D1: cost-aware ROC with X2 ATR exit and opposite-consensus rearm (X6)

X6 was formed from the observed X2 re-entry churn on already consumed S06 Train data. After an ATR trail exit, it waits until flat with no pending orders and requires two consecutive opposite ROC20/ROC60 consensus bars before another entry. It has no fixed holding-time limit. X6 was not included in the frozen S06 X1/X2 exit list, so these are exploratory development diagnostics, not a preregistered qualification result.

| Scenario | Window | X1 net / DD / cases | X2 net / DD / cases | X6 net / DD / cases | X6 minus X1 | X6 minus X2 |
|---|---|---:|---:|---:|---:|---:|
| Base | inner-screen-1 | −1.055% / 1.185% / 381 | −1.824% / 1.898% / 667 | −1.211% / 1.260% / 459 | −0.156 pp | +0.613 pp |
| Base | inner-screen-2 | −1.576% / 1.600% / 463 | −2.376% / 2.383% / 842 | −1.817% / 1.825% / 585 | −0.240 pp | +0.560 pp |
| Base | inner-screen-3 | −0.471% / 0.888% / 205 | −0.707% / 1.045% / 368 | −0.600% / 0.881% / 255 | −0.129 pp | +0.107 pp |
| Stress | inner-screen-1 | −1.674% / 1.739% / 209 | −2.142% / 2.169% / 301 | −1.639% / 1.660% / 237 | +0.035 pp | +0.503 pp |
| Stress | inner-screen-2 | −2.415% / 2.436% / 307 | −3.117% / 3.121% / 466 | −2.483% / 2.487% / 355 | −0.069 pp | +0.634 pp |
| Stress | inner-screen-3 | −0.666% / 0.927% / 129 | −1.012% / 1.140% / 189 | −0.836% / 1.065% / 148 | −0.170 pp | +0.176 pp |

| Scenario chain | X1 | X2 | X6 |
|---|---:|---:|---:|
| Base net return | −3.074% | −4.835% | −3.588% |
| Stress net return | −4.687% | −6.152% | −4.883% |

X6 improves on X2 in all six pairs and reduces trades and fees plus slippage: Base trade cases/costs move from 1,877/$560.88 on X2 to 1,299/$388.15 on X6; Stress moves from 956/$712.90 to 740/$551.81. X1 still has fewer trades and lower costs, and beats X6's chain in both scenarios. All six X6 windows remain net losers. This confirms that opposite-consensus rearming mitigates ATR-exit churn but does not repair the ROC signal's economics.

All 18 native runs (X1, X2, X6 across three windows and two scenarios) passed M2.2 on every requested minute, had zero liquidations, and reconciled case-account returns. Source snapshot, feature recipe, scaler, account risk, costs, protocol and registered window matched within each pair. Results remain SCENARIO_ONLY and DEVELOPMENT_DIAGNOSTIC_ONLY; they do not establish profitability, teacher qualification, student superiority, or G1/G2.

Reproducible comparison: .artifacts/route-a-roc-x6-atr-rearm-task3/paired_x1_x2_x6_results.json. X6 native runs: .artifacts/route-a-roc-x6-atr-rearm-task3/native-x6/. The frozen feature snapshot used for these diagnostics is .artifacts/route-a-roc-x2-costaware-task3/feature-recipe-27f03e698a58dde7aa2c09cf161b83a491a20e0de167608f482d6073aa67818e.json.

## S06 D1: ROC/X6 entry-edge and exit-capture diagnosis

A reproducible point-in-time join connected 2,039 completed X6 cases to exactly one native entry decision each across the three consumed S06 windows and both cost scenarios. There were zero unmatched or ambiguous entry rows. This is DEVELOPMENT_DIAGNOSTIC_ONLY; cohorts are post-hoc, and a cohort filter can change later positions and account paths.

| Scenario | Completed cases | Mean pre-slippage gross return | Mean observed round-trip fees + slippage | Mean MFE | Cases where MFE exceeded realized costs | Those cases ending net negative |
|---|---:|---:|---:|---:|---:|---:|
| Base | 1,299 | 0.77 bps | 12.03 bps | 31.0 bps | 676 | 378 |
| Stress | 740 | 3.03 bps | 30.03 bps | 37.1 bps | 260 | 135 |

X6 exits realized only a small share of the recorded MFE; MFE is based on minute-bar highs/lows and is an upper bound, not proof a threshold was executable at a decision close or fill. The forward path also shows weak entry economics. Mean directional close return from the entry decision close rises from −0.18 bps at 5 minutes to +4.16 bps at 60 minutes in Base; estimated after-cost returns remain −12.21 to −7.88 bps. Stress means are +0.75 to +5.68 bps while estimated after-cost returns remain −29.28 to −24.32 bps. The after-cost horizon estimate uses each completed case’s realized fee and slippage rate; it is not a counterfactual native fill or account simulation.

The same 60-minute comparison is negative after observed costs in every window:

| Window | Scenario | Cases | Mean directional return at 60m | Estimated after-cost return at 60m | Positive after-cost cases | Native X6 net PnL |
|---|---|---:|---:|---:|---:|---:|
| inner-screen-1 | Base | 459 | 4.33 bps | −7.71 bps | 38% | −$121.13 |
| inner-screen-1 | Stress | 237 | 3.50 bps | −26.54 bps | 25% | −$163.88 |
| inner-screen-2 | Base | 585 | 2.79 bps | −9.24 bps | 40% | −$181.66 |
| inner-screen-2 | Stress | 355 | 5.60 bps | −24.43 bps | 29% | −$248.34 |
| inner-screen-3 | Base | 255 | 6.99 bps | −5.04 bps | 33% | −$59.98 |
| inner-screen-3 | Stress | 148 | 9.34 bps | −20.68 bps | 34% | −$83.58 |

Feature splits did not establish a robust entry gate. ROC acceleration versus deceleration was nearly even in Base and changed by window in Stress. EMA direction alignment was effectively tautological for these consensus entries. MACD-discordant entries were sparse and remained loss-making in pooled Base and Stress. Entries 21–60 minutes after the last direction-aligned EMA crossover had less negative pooled case PnL than the full cohort, but remained negative in both scenarios, and their fixed-horizon returns still failed to clear costs. No filter or candidate is promoted from this post-hoc result.

Reproducible analysis script and outputs: .artifacts/route-a-roc-entry-economics-task3/analyze_entry_cohorts.py, entry_cohorts.json, and entry_cohorts.md. All inputs are hashed in the JSON; source rows are from already-consumed S06 Train only.
## S06 D1: X7 EMA-cross-age native comparison

X7 was added as an unregistered exploratory ROC entry filter after the X6 cohort split showed a less-negative 21–60 minute EMA-cross-age cohort. It keeps the X1 indicator exit and admits trend ROC entries only 21–60 completed minutes after a direction-aligned EMA9/21 cross. This was derived from already-consumed S06 Train evidence; it is not a preregistered or independent qualification result.

| Window | Scenario | X1 net / DD / cases | X7 net / DD / cases | X7 net change |
|---|---|---:|---:|---:|
| inner-screen-1 | Base | −1.055% / 1.185% / 381 | −0.689% / 0.799% / 269 | +0.366 pp |
| inner-screen-1 | Stress | −1.674% / 1.739% / 209 | −1.155% / 1.191% / 150 | +0.518 pp |
| inner-screen-2 | Base | −1.576% / 1.600% / 463 | −1.115% / 1.126% / 333 | +0.461 pp |
| inner-screen-2 | Stress | −2.415% / 2.436% / 307 | −2.244% / 2.253% / 234 | +0.171 pp |
| inner-screen-3 | Base | −0.471% / 0.888% / 205 | −0.203% / 0.597% / 157 | +0.268 pp |
| inner-screen-3 | Stress | −0.666% / 0.927% / 129 | −0.460% / 0.848% / 103 | +0.206 pp |

The compounded Base account loss falls from 3.074% on X1 to 1.996% on X7; Stress falls from 4.687% to 3.818%. Both remain negative. Matched-entry analysis found a 0 bps median net-return difference on common entries in every pair. X7-only entries had a more negative mean per-case result than X1-only entries in five of six pairs. The observed account improvement therefore does not demonstrate improved predictive quality or better teaching examples; fewer cases and lower fees/slippage explain much of the reduction, with later account paths also changing. X7 is not promoted.

All six native pairs passed M2.2 decision coverage, had zero liquidations, reconciled case cashflows to account equity, and matched the intended source, feature, scaler, protocol, risk, cost, and window identities. The added Z-score, Donchian-breakout, and Bollinger splits in the entry-cohort report also failed to identify a directionally stable, after-cost gate across Base and Stress. These post-hoc splits are diagnostic only.

Reproducible comparison: `.artifacts/route-a-roc-ema-age-x7-task3/paired_x1_x7_results.md` and `paired_x1_x7_results.json`; native X7 runs: `.artifacts/route-a-roc-ema-age-x7-task3/native-x7/`. The X6 entry-cohort update remains at `.artifacts/route-a-roc-entry-economics-task3/entry_cohorts.md` and `.json`.
### ATR-normalized ROC consensus strength follow-up

The latest point-in-time join also bins the weaker of direction-signed ROC20/ROC60 by ATR14, as a causal signal-to-volatility diagnostic. The broad 2–4 and 4–8 buckets contain most cases and remain negative in both scenarios: mean native net per case is −$0.302/−$0.695 for 2–4 and −$0.242/−$0.654 for 4–8 (Base/Stress). The only Base bucket with positive 60-minute estimated after-cost direction return is 1–2, with just 21 cases and negative realized native net per case; its Stress counterpart has one case and is sharply negative. The ≥8 bucket has only 2 Base and 7 Stress cases and is negative in Stress. No robust, sufficiently populated after-cost entry threshold emerged, so no ROC strength gate is promoted. Full per-window values and case-level inputs are in `entry_cohorts.json`.
## S06 D1: X8 ATR-to-indicator handoff exit

The exact-entry X1/X2 analysis motivated X8: the X1 ROC indicator exit remains active throughout; an additional 3×ATR exit is also active while completed-bar favorable excursion is below 4× the registered Base/Stress round-trip cost. Once the excursion reaches that level, X8 disables the ATR exit and relies on X1. ATR exits still require the two-opposite-consensus re-entry rule. It is an unregistered development candidate using consumed S06 Train only.

All six native runs used the immutable feature recipe and identical source, scaler, protocol, account risk, leverage, cost and window identities. M2.2 coverage was complete on all 213,104 requested decisions across the six runs; case PnL reconciled exactly to account equity within $0.02, and no liquidation occurred. Per-window results are in `paired_x1_x2_x6_x8_results.md`.

| Scenario | Window | X1 net / DD / cases | X2 net / DD / cases | X6 net / DD / cases | X8 net / DD / cases | X8−X1 |
|---|---|---:|---:|---:|---:|---:|
| Base | inner-screen-1 | −1.055% / 1.185% / 381 | −1.824% / 1.898% / 667 | −1.211% / 1.260% / 459 | −1.452% / 1.578% / 447 | −0.397 pp |
| Stress | inner-screen-1 | −1.674% / 1.739% / 209 | −2.142% / 2.169% / 301 | −1.639% / 1.660% / 237 | −1.650% / 1.672% / 233 | +0.024 pp |
| Base | inner-screen-2 | −1.576% / 1.600% / 463 | −2.376% / 2.383% / 842 | −1.817% / 1.825% / 585 | −1.560% / 1.570% / 543 | +0.016 pp |
| Stress | inner-screen-2 | −2.415% / 2.436% / 307 | −3.117% / 3.121% / 466 | −2.483% / 2.487% / 355 | −2.531% / 2.535% / 348 | −0.116 pp |
| Base | inner-screen-3 | −0.471% / 0.888% / 205 | −0.707% / 1.045% / 368 | −0.600% / 0.881% / 255 | −0.175% / 0.624% / 240 | +0.296 pp |
| Stress | inner-screen-3 | −0.666% / 0.927% / 129 | −1.012% / 1.140% / 189 | −0.836% / 1.065% / 148 | −0.593% / 0.929% / 145 | +0.073 pp |

X8 compounds to −3.160% Base and −4.708% Stress, slightly worse than X1 (−3.074%/−4.687%), although it improves on X2 and X6 chains. Its large W1 Base loss and W2 Stress loss outweigh W3 improvements. LONG and SHORT PnL are both negative in five of six pairs; neither direction supports an eligibility filter. X8 is not promoted: the exit handoff does not restore profitability or robustly beat X1.

Reproducible identity checks, per-side funnels/economics, case/account residuals, quantile holding time, MFE/MAE and paired tables: `.artifacts/route-a-roc-x2-handoff-x8-task4/paired_x1_x2_x6_x8_results.json` and `.md`. Native runs: `.artifacts/route-a-roc-x2-handoff-x8-task4/native-x8/`. Candidate source/tests: `teachers_1m/exits.py`, registry/native runner, and focused lifecycle/ROC tests.

## S06 D1: causal ROC-slope and multi-factor entry screen

The entry cohort analyzer now compares signed ROC20/ROC60 changes against the immediately preceding contiguous completed 1m bar, extends forward-close diagnostics from 60 to 240 minutes, and evaluates all 56 two/three-factor conjunctions among seven point-in-time directional confirmations. This is a post-hoc screen over the same 2,039 completed X6 cases, with zero ambiguous or unmatched joins; all outcomes remain DEVELOPMENT_DIAGNOSTIC_ONLY.

The dominant `both_rising` slope cohort contains 1,031 Base and 622 Stress cases. Its mean native net per case is −$0.291 and −$0.646, respectively; its estimated 60-minute after-cost direction continuation is −9.19 and −23.05 bps. No two/three-factor conjunction had at least 20 observations in each of the six window/scenario cells while remaining positive after estimated costs at 60, 120, or 240 minutes. These horizon figures use realized case cost and observed future closes, not counterfactual fills or account paths. No slope or confluence gate is promoted; reduced entries alone would not establish an economic improvement.

Reproducible output and hashed inputs: `.artifacts/route-a-roc-entry-economics-task3/entry_cohorts.json`, `entry_cohorts.md`; analysis source: `analyze_entry_cohorts.py`. Ubuntu Ruff passed for the updated analyzer.

## S06 D1: fixed-side native controls against ROC/X1

To determine whether the ROC/X1 losses represented useful direction selection or mostly costs, fixed 0.3-target LONG and SHORT controls were run through the same native account engine on all three S06 windows and both cost scenarios. All 12 runs matched the X1 source snapshot, development scaler, starting balance, $1,000 risk budget, 1× leverage, seed 20261026, and Base/Stress cost settings; every requested 1m decision passed M2.2 and all accounts had zero liquidations. The historical X1 receipts did not serialize the seed, but the native research call omitted a seed argument and therefore used its recorded runner default, 20261026.

ROC/X1 lost in all six pairs (−1.055% to −2.415%). Fixed LONG outperformed X1 in all six; fixed SHORT outperformed it in five. However, the fixed-side results reversed with the window: LONG earned +1.10% in window 1 and +0.13% in window 2, then lost about 0.35% in window 3; SHORT showed the opposite direction. Neither side is a stable teacher, and selecting a side after seeing these outcomes would be hindsight bias.

The gross price/cost split identifies both signal inconsistency and fee drag. X1 gross price PnL was +$8.12, −$11.47, −$19.67, −$12.56, +$14.08, and +$29.56 across the six pairs, while fees plus slippage ranged from $61.23 to $228.79; the resulting net was negative each time. Fixed controls retain one open position until each block ends, so their sole case is `TERMINAL_CENSORED` and reported account return includes native end-of-window mark-to-market, fees, slippage, and funding; these are directional controls rather than closed-trade comparisons.

No candidate is promoted. Evidence indicates the ROC edge is absent or inconsistent in several windows, while transaction costs overwhelm the small positive gross edge in others. Reproducible pair table, per-case metrics, and identity checks: `.artifacts/route-a-fixed-controls-task4/paired_fixed_side_vs_x1.md` and `.json`; native control reports: `.artifacts/route-a-fixed-controls-task4/<window>/<scenario>/<control>/native-run.json`; runner and summarizer: `qa/run_route_a_fixed_side_controls_s06.py`, `qa/summarize_route_a_fixed_side_controls_s06.py`.

The same 12 controls also pair exactly with the EMA reverse-after-flat runs: source, feature recipe, scaler, risk budget, leverage, costs, target exposure, and M2.2 checks all match. Both fixed directions outperform the reversal candidate in all six cells. EMA reversal loses 5.12%–14.43% while its fees plus slippage are $240–$1,444; fixed controls cost less than $1 per block and remain direction-sensitive, so this confirms reversal-driven turnover is damaging without identifying a prospectively profitable side. The EMA candidate remains an engineering lifecycle fix with failed D1 economics. Paired report: `.artifacts/route-a-fixed-controls-task4/paired_fixed_side_vs_ema_reversal.md` and `.json`.

## S06 D1: ROC entry edge surplus over cost and noise buffer

For each of the 2,039 completed X6 cases, the entry-time buffer was reconstructed from the registered scenario cost plus `max(0.25 × ATR14, 0.5 × RV20)`. RV20 uses 20 contiguous completed 1m log returns; their population standard deviation is scaled by `sqrt(20)` to a 20-bar horizon. Edge surplus is the weaker direction-signed ROC20/ROC60 magnitude minus this cost-plus-noise threshold. Buckets divide that surplus by the registered scenario cost: `<0.25×`, `0.25–0.5×`, `0.5–1×`, `1–2×`, and `≥2×`. This is a post-hoc decomposition of selected X6 entries, not a causal policy or an account replay. Forward 60/120/240m outcomes subtract each case's realized round-trip cost from observed directional close returns; they are not counterfactual fills.

| Surplus / scenario-cost bucket | Cases | Mean native net / case | 60m after-cost | 120m after-cost | 240m after-cost | Cells with ≥20 cases | Positive horizon cells among those (60/120/240m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `<0.25×` | 959 | −$0.467 | −13.5 bps | −13.6 bps | −16.5 bps | 6/6 | 0/1/0 |
| `0.25–0.5×` | 459 | −$0.451 | −20.5 bps | −21.9 bps | −20.9 bps | 6/6 | 0/0/0 |
| `0.5–1×` | 366 | −$0.355 | −10.2 bps | −19.3 bps | −15.5 bps | 4/6 | 0/0/1 |
| `1–2×` | 204 | −$0.435 | −12.2 bps | −19.2 bps | −16.7 bps | 3/6 | 0/0/0 |
| `≥2×` | 51 | +$0.291 | +6.0 bps | −39.3 bps | −12.2 bps | 1/6 | 1/1/1 |

No adequately populated bucket is positive across windows, Base/Stress costs, and all tested horizons. The apparent `≥2×` benefit is concentrated in one of six cells and reverses at longer pooled horizons; it is too sparse and unstable to support a gate. The broad, adequately populated buckets remain net-negative. No entry threshold is promoted and ROC/X6 remains unqualified. Per-cell values and the full case-level join are in `.artifacts/route-a-roc-entry-economics-task3/entry_cohorts.md` and `.json` (schema v4).
