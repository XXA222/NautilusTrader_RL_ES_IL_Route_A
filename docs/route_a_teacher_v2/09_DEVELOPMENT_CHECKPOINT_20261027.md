# Route A teacher development checkpoint — 2026-10-27

**Classification:** `DEVELOPMENT_DIAGNOSTIC_ONLY`  
**Baseline:** `bd313c5241c31f1ccff96fa4bf76a7966be48767`  
**Canonical root:** `D:\Program Files\NautilusTrader_RL_ES_IL_Route_A`

This checkpoint records development work on already consumed S06 Train windows. It does not claim G1/G2, fresh out-of-sample performance, student superiority, or profitability. No Test/Holdout source was read.

## Acceptance answers

- **Can the teachers emit native, traceable cases?** Yes at the engineering level. The native runner retains cases, minute decisions, fees, slippage, funding, censor states, and M2.2 evidence.
- **Are the current cases qualified teaching examples for profitable behavior?** No. Every tested candidate has negative Base and Stress chain returns.
- **Has a student learned from these cases and beaten its teacher?** No student training or paired student evaluation was started.
- **Is trading profitable?** No. No tested teacher or candidate established positive after-cost returns.

The project is **not accepted** against those economic and student-learning criteria. The active improvement goal remains open.

## Work and evidence

| Workstream | Status | Result |
|---|---|---|
| EMA reversal lifecycle | Engineering pass; economics fail | Multi-step close acknowledgement, flat/no-pending wait, next-minute signal recheck, symmetric long/short tests. Native reversal candidate worsened the Base chain from −6.24% to −12.04% and Stress from −16.73% to −29.32%. |
| ROC churn and cost-aware hysteresis | Candidate improves baseline; qualification fail | Cost-aware ROC X1 chain: Base −3.07%, Stress −4.69%, versus legacy ROC X1: Base about −22.14%, Stress about −49.51%. This is a large loss reduction, not profitability. |
| X1/X2 exit economics | Analysis complete; economic result mixed | Twenty-four paired teacher/window/scenario comparisons; case/account reconciliation passed and completed-case MFE path mismatches were zero. X2 improved ROC in every block, but remained negative; X2 worsened EMA, Donchian, and Z-score in every block. |
| Cost-aware ROC X2 ATR rearm (X6) | Candidate improves X2; qualification fail | Six native pairs: X6 beats X2 in all pairs, but Base/Stress chains are −3.59%/−4.88% versus X1 at −3.07%/−4.69%. Lower churn does not restore profitability. X6 was not in the frozen X1/X2 exit list and remains an exploratory D1 candidate. |
| X6 entry-edge and exit-capture diagnosis | Diagnostic complete; no robust gate | 2,039 completed cases joined with zero errors. Base mean gross return 0.77 bps vs 12.03 bps costs; Stress 3.03 bps vs 30.03 bps. Fixed 60m after-cost returns remained negative in both scenarios. |
| X3 cost-scaled profit protection | Candidate failed | X3 worsened cost-aware ROC X1 in all six windows. Base/Stress chain: −4.79%/−5.59%. |
| X4 opposite-consensus rearm after profit lock | Improves X3; still fails qualification | X4 improved net return and drawdown over X3 in all six pairs. Matched-entry median return change was 0 bps; gains came from avoiding many negative unpaired X3 trades and lower fees. X4 Base/Stress chains were −3.16%/−4.49%, still negative; Base was slightly worse than cost-aware X1. |
| X5 cost/ATR loss guard | Candidate failed | X5 Base/Stress chains were −3.40%/−4.65%, worse than X4’s −3.16%/−4.49%. New unmatched trades remained net negative. The historical trigger-price estimate was only a screening diagnostic; native account results decide this failure. |
| Contrarian ROC direction | Candidate failed | Same cost/ATR/volatility admission logic with only direction inverted. All six windows were worse than trend X1. Base chain −5.35% versus −3.07%; Stress −7.28% versus −4.69%. |
| Native execution checks | Pass for tested development runs | X4, X5, and contrarian ROC each completed three Base/Stress windows. Every decision was M2.2 verified; no liquidation was observed. Results remain scenario-only. |

## Reproducible artifacts

- X1/X2 case and exit economics: `.artifacts/route-a-exit-economics-task3/x1_x2_exit_economics.json` and `per_trade_exit_metrics.jsonl`.
- X3 comparison: `.artifacts/route-a-x3-profit-protect-task3/paired_x1_x3_results.json`.
- X4 comparison and native runs: `.artifacts/route-a-x4-opposite-rearm-task3/paired_x1_x3_x4_results.json` and `native/`.
- X5 comparison and native runs: `.artifacts/route-a-x5-cost-atrloss-task3/paired_x4_x5_results.json` and `native/`.
- Cost-aware ROC X1/X2/X6 paired results: .artifacts/route-a-roc-x6-atr-rearm-task3/paired_x1_x2_x6_results.json; X6 native runs are in native-x6/.
- X6 entry cohort and fixed-horizon diagnostics: .artifacts/route-a-roc-entry-economics-task3/entry_cohorts.md and entry_cohorts.json.
- Contrarian comparison and native runs: `.artifacts/route-a-roc-fade-task3/paired_trend_vs_fade.json` and `native/`.
- EMA reversal paired results: `.artifacts/route-a-ema-reversal-task1/paired_base_stress_summary.json`.

The 39 targeted lifecycle and ROC tests passed. Ruff passed on the modified teacher/runner and comparison scripts. MyPy passed on the modified exit, ROC teacher, and registry sources, plus the comparison scripts. The X6 focused MyPy check passed on three touched sources with imported-module following skipped. An initial dependency-following run surfaced three local runner annotation findings and callback/untyped-import issues in qa/run_route_a_train_student_screen_20261020.py; the local annotations were corrected afterward, but the dependency-following run was not repeated.

## Limits

All windows above were already consumed development windows. They can select or reject candidates, but cannot establish new-sample generalization. No teacher currently meets the economic bar for student training as a profitability demonstration. Formal acceptance remains open.

## Incremental X2 correctness fix — 2026-10-27

Code review found that `BaseTeacher1m.advise` returned immediately on an X2 ATR exit before storing the current 1m indicator snapshot. A subsequent flat-state entry decision could therefore compare the next bar with a pre-exit bar and emit a stale cross. The decision path now records the current indicator snapshot before returning the ATR exit and cancels any competing pending reversal. A focused regression covers the exit-minute cross followed by a still-below-EMA flat bar; it remains flat.

The lifecycle test file passes (23 tests); Ruff and MyPy pass on the modified teacher and test file. The read-only preregistration preflight passes after binding the updated teacher hash in the source erratum. The correction does not change candidate thresholds, source scope, costs, or acceptance rules. Paired native results were pending when this checkpoint paragraph was first written; they are now recorded below.

### Paired native confirmation on consumed S06 windows

To check that the observed stale-cross mechanism is not confined to one path, the corrected code was paired against an isolated overlay restoring the exact recorded pre-fix `base.py` hash. The two runs per window used identical source snapshot, feature recipe, scaler, account, Base costs, and teacher recipe; the overlay package differs only in `teachers_1m/base.py`. Historical S06 output also reproduces the pre-fix first-window decisions and case economics exactly, aside from run-specific case IDs.

| S06 window | Pre-fix net / max DD | Corrected net / max DD | Net change | Completed cases, pre-fix → corrected | Fees, pre-fix → corrected | Unpaired completed-trade PnL, pre-fix → corrected |
|---|---:|---:|---:|---:|---:|---:|
| `inner-screen-1` | −3.875% / 4.013% | −3.478% / 3.567% | +0.396 pp | 1,340 → 1,187 | $333.16 → $295.13 | −$58.67 → −$19.06 |
| `inner-screen-2` | −4.625% / 4.638% | −4.021% / 4.035% | +0.603 pp | 1,355 → 1,193 | $336.24 → $296.03 | −$95.83 → −$35.50 |

Every decision passed M2.2 (41,757 and 44,637 rows), both sides remained loss-making, liquidations were zero, and case cashflows reconciled exactly to account equity. In the two windows, 190/295 and 212/370 changed entry signals followed the same-minute ATR exit; shared completed entries had identical holding minutes and net return differences below `1e-8` bps. The account differences came from changed later re-entry paths: unpaired completed-trade PnL improved by $39.60 and $60.34 respectively, while these unpaired trade sets still lost money. This supports the code correction's mechanism, not profitability or acceptance. Both corrected runs remain negative, consumed-data D1 diagnostics.

Reproducible native comparison reports: `.artifacts/route-a-x2-snapshot-fix-task3/comparison.json` and `.artifacts/route-a-x2-snapshot-fix-task3/comparison-inner-screen-2.json`. The generalized analyzer and control-overlay builder pass Ruff and MyPy.

The same two pairs were then run under the registered Stress costs:

| S06 window | Pre-fix net / max DD | Corrected net / max DD | Net change | Fills | Fees |
|---|---:|---:|---:|---:|---:|
| `inner-screen-1` | −9.883% / 9.906% | −8.795% / 8.809% | +1.088 pp | 3,445 → 2,701 | $666.33 → $590.25 |
| `inner-screen-2` | −10.583% / 10.594% | −9.302% / 9.313% | +1.281 pp | 3,720 → 2,943 | $672.47 → $592.07 |

All Stress decisions also passed M2.2, with no liquidations and exact case/account reconciliation. The full-policy net and drawdown improve, but remain strongly negative. In the post-hoc common-entry subset, holding times are identical and median net-return change is zero; execution-cost differences mean individual common trades are not identical in Stress. The corrected runs have fewer severely negative unpaired cases, but those remaining unpaired sets also lose money. Stress reports: `.artifacts/route-a-x2-snapshot-fix-task3/comparison-stress-inner-screen-1.json` and `comparison-stress-inner-screen-2.json`.

The shared-base fix was paired with exact pre-fix controls for ROC, Donchian, and Z-score in both scenarios on `inner-screen-1`:

| Teacher | Scenario | Pre-fix net / max DD | Corrected net / max DD | Net change | Fills, pre-fix → corrected |
|---|---|---:|---:|---:|---:|
| ROC | Base | −4.316% / 4.362% | −4.102% / 4.144% | +0.214 pp | 2,980 → 2,878 |
| ROC | Stress | −10.970% / 10.970% | −10.540% / 10.540% | +0.430 pp | 4,318 → 3,895 |
| Donchian | Base | −4.331% / 4.417% | −4.267% / 4.331% | +0.064 pp | 2,905 → 2,857 |
| Donchian | Stress | −10.853% / 10.866% | −10.691% / 10.699% | +0.163 pp | 4,129 → 3,998 |
| Z-score | Base | −2.053% / 2.099% | −2.114% / 2.161% | −0.061 pp | 1,160 → 1,182 |
| Z-score | Stress | −4.649% / 4.680% | −4.759% / 4.790% | −0.110 pp | 1,160 → 1,182 |

Every corrected run passed M2.2 on all 41,757 decisions, had zero liquidations, and reconciled cases to account equity. The shared correction is not a universal economic improvement: it helps ROC and Donchian slightly on this window but worsens Z-score. All six corrected outcomes remain negative. Reports use `comparison-{roc,donchian,zscore}-x2-{base,stress}-inner-screen-1.json` under `.artifacts/route-a-x2-snapshot-fix-task3/`.

Cashflow attribution separates estimated pre-slippage price PnL from fees, slippage attribution, and funding. On inner-screen-1, estimated pre-cost price PnL is only +$7.82 for EMA, +$20.27 for ROC, +$1.27 for Donchian, and −$34.29 for Z-score. Base fees plus slippage are $355–$431 for EMA/ROC/Donchian and $177 for Z-score; Stress costs are $886–$1,075 for EMA/ROC/Donchian and $441 for Z-score. On EMA inner-screen-2, estimated pre-cost price PnL is −$47.01 Base and −$42.22 Stress. The evidence points to near-zero or inconsistent signal edge combined with transaction costs overwhelming it; Z-score also has negative gross price performance. Funding is negligible in these windows. Full decomposition is recorded in `06_NATIVE_RESEARCH_RESULTS.md` and each comparison JSON.

## Incremental X7 evaluation — 2026-10-27

An unregistered X7 ROC entry-age filter was evaluated natively against X1 on all six consumed S06 window/scenario pairs. It improved account return and drawdown in each pair, reducing the compounded Base/Stress loss from −3.074%/−4.687% to −1.996%/−3.818%. It remains loss-making in every pair and therefore does not meet acceptance.

Matched common entries had median net-return delta 0 bps in all six pairs. X7-only entries had worse average per-case results than X1-only entries in five of six pairs. The reduced account loss is chiefly evidence of reduced trade/cost exposure and changed path dependence, not improved signal quality. X7 remains D1-only and is not promoted as a qualified teaching strategy. The additional Z-score, Donchian and Bollinger entry-cohort splits also failed to show a stable after-cost gate across Base and Stress.

Artifacts: `.artifacts/route-a-roc-ema-age-x7-task3/paired_x1_x7_results.md`, `paired_x1_x7_results.json`, and `native-x7/`; expanded point-in-time cohorts: `.artifacts/route-a-roc-entry-economics-task3/entry_cohorts.md` and `entry_cohorts.json`. All six native pairs passed M2.2, had zero liquidations, and reconciled case/account cashflows. The focused Ubuntu lifecycle and ROC regression run completed with 42 passing tests after the latest source edits.
The follow-up ATR-normalized ROC consensus-strength cohorts did not reveal a defensible entry threshold. The large 2–4 and 4–8 signal/ATR buckets remained negative in Base and Stress; the one favorable-looking Base horizon bucket (1–2) had only 21 cases and negative realized native PnL, while Stress had one case. The extreme-strength buckets were also too sparse and negative under Stress. This rejects the current screen; it does not establish that all possible signals are exhausted.
## Incremental X8 evaluation — 2026-10-27

The matched-entry X1/X2 attribution showed X2 often improved low-MFE trade outcomes but gave back substantially more on cases whose X1 MFE exceeded four times costs. X8 kept the X1 indicator exit active throughout and added a 3×ATR exit until completed-bar MFE reached 4× scenario costs, then disabled the ATR exit. ATR exits still required flat/no-pending status and two opposite ROC consensus bars before re-entry. No fixed holding cap was added.

All six native X8 runs passed full M2.2, zero-liquidation, exact environment-identity and case/account reconciliation checks. X8 improved on X2 and X6 compounded returns, but remained slightly worse than X1: Base −3.160% vs −3.074%; Stress −4.708% vs −4.687%. It lost substantially more than X1 on W1 Base and W2 Stress, while gains came from other windows. LONG and SHORT remain negative in five of six pairs. X8 is D1-only and is not promoted.

Focused Ubuntu lifecycle and ROC regression suite now passes 47 tests. Ruff passed on X8 sources, tests, runner, and analysis scripts; focused MyPy passed on the exit, registry, and native runner. The native result and complete per-side economics are in `.artifacts/route-a-roc-x2-handoff-x8-task4/paired_x1_x2_x6_x8_results.md` and `.json`; runs are in `native-x8/`.

## Incremental ROC entry screen — 2026-10-27

The causal one-minute ROC20/ROC60 slope check found that the dominant both-rising cohort still loses after native costs: 1,031 Base and 622 Stress completed cases, with mean native net/case of −$0.291/−$0.646 and estimated 60-minute after-cost continuation of −9.19/−23.05 bps. The analyzer now also measures 120/240-minute continuation and all 56 two/three-factor combinations of seven directional confirmations. None had at least 20 cases in every window/scenario cell and positive estimated after-cost continuation across all six cells at any tested horizon. This rejects these post-hoc screens; it does not prove all entry mechanisms are exhausted.

No candidate is promoted and no account result is inferred from cohort selection. Source and outputs: `.artifacts/route-a-roc-entry-economics-task3/analyze_entry_cohorts.py`, `entry_cohorts.json`, and `entry_cohorts.md`. The analyzer reran under Ubuntu 24.04/Python 3.14.7, joined 2,039 cases with zero errors, and passed Ruff.

## Incremental fixed-direction native controls — 2026-10-27

Ran 12 same-account/same-risk/same-cost native controls (fixed 0.3 LONG and SHORT across three S06 windows and Base/Stress). All 12 pairs matched source snapshot, scaler, $1,000 risk budget, 1× leverage, cost scenario, configured exposure target, and M2.2; zero liquidations. Each fixed control held one open position to the block end, so its single case is terminal-censored and account return includes native mark-to-market.

ROC/X1 was negative in all six cells. Fixed LONG beat it in all six and fixed SHORT in five, but the favorable fixed side reversed with the window; neither is a stable ex-ante strategy. X1 gross price PnL ranged from −$19.67 to +$29.56 while fees plus slippage were $61.23–$228.79, yielding losses in every cell. This supports a combined failure of inconsistent signal quality and transaction costs, not a teacher qualification. Reports: `.artifacts/route-a-fixed-controls-task4/paired_fixed_side_vs_x1.md` and `.json`; native reports and scripts are under `.artifacts/route-a-fixed-controls-task4/` and `qa/run_route_a_fixed_side_controls_s06.py` / `qa/summarize_route_a_fixed_side_controls_s06.py`.

### EMA reverse-after-flat control comparison

The same fixed controls also beat EMA reverse-after-flat in all six paired cells. Its native losses were −5.12% to −14.43%, with $240–$1,444 fees plus slippage; fixed controls were below $1 per block, while the better direction reversed with the window. This confirms the reversal candidate's turnover/cost harm, not a profitable directional rule. Identity and M2.2 report: `.artifacts/route-a-fixed-controls-task4/paired_fixed_side_vs_ema_reversal.md` and `.json`.

## Incremental ROC cost-noise surplus cohorts — 2026-10-27

The X6 entry-cohort report now measures signed ROC edge beyond the registered cost plus ATR/realized-volatility buffer. On 2,039 completed cases with zero join errors, the `<0.25×` and `0.25–0.5×` surplus groups are adequately populated in all six window/scenario cells and remain negative in native net per case and at the pooled 60/120/240m after-cost horizons. Higher-surplus groups are sparse or still negative; `≥2×` has at least 20 cases in only one cell, with pooled 120/240m continuation negative. This does not establish an actionable threshold, and no filter is promoted. The estimates are post-hoc close-return diagnostics, not native counterfactual account paths.

Analyzer: `.artifacts/route-a-roc-entry-economics-task3/analyze_entry_cohorts.py`; report: `entry_cohorts.md` / `entry_cohorts.json` (schema v4). It reran under Ubuntu 24.04/Python 3.14.7, joined 2,039 cases with zero errors, and passed Ruff plus focused MyPy (`--follow-imports=silent`). The source check initially caught three loop-variable type reuse errors; those variables were renamed and both checks now pass.

## Fresh-Train runner scope correction — 2026-10-27

The registered execution now selects seven configurations: EMA exit-only and reverse-after-flat, legacy ROC, cost-aware/hysteresis ROC X1 and X7, Donchian, and Z-score. X7 is the existing cost-aware ROC with the frozen 21–60 completed-minute EMA-cross-age entry gate and unchanged X1 exit. It is included as a development-selected candidate because it improved account return and drawdown in all six consumed S06 Base/Stress pairs, while remaining negative in every pair. T05–T08 remain excluded from native evaluation per the active goal. The Base/Stress, two-seed, three-control matrix and per-candidate acceptance rule are unchanged; the report separately identifies in-scope screen completion and full nine-row preregistered-matrix completion. The cost-aware ROC round-trip cost still binds to the registered scenario without retuning. The report now includes paired X7-versus-X1 account return, drawdown, fees, slippage, and funding deltas for each window/scenario/seed. No scored source or outcome was read. The read-only preflight passes with one of five staged imports present (warmup only) and `market_payload_read: false`; the window-0 import manifest is absent and its five-day window ends 2026-10-30. No partial score or student training was run.

The stale `exits.py`, `registry.py`, and native-runner hashes are rebound in source errata SHA-256 `faacdc7cf5acb02c1165bcb4d99fcc31afe667a34aee8a5eed8c60774935368f`. Runner and test: `qa/run_route_a_teacher_student_train_screen_20261026.py`, `tests/test_route_a_teacher_train_screen_runner.py`.

### Fresh-Train X7 execution candidate — 2026-10-27

Added cost-aware ROC X7 to the not-yet-scored fresh-Train execution scope and bound the frozen 21–60-minute EMA-age gate plus scenario-specific round-trip costs. X7 is paired with cost-aware ROC X1 in the same native windows, Base/Stress scenarios, and seeds. The existing preregistered matrix, controls, dates, acceptance rule, and T05–T08 exclusions are unchanged. This is candidate selection from consumed S06 only; it does not claim independent validation or profitability. The full window-0 import is still absent, so no fresh score was produced. Focused implementation checks: 11 runner tests and Ruff pass under Ubuntu 24.04/Python 3.14.7; runner MyPy passes, and the touched test file's MyPy check also passes after replacing a broadly typed keyword dictionary with explicit typed arguments. After the exact window-0 import manifest appears, run `wsl.exe --distribution Ubuntu-24.04 --exec /bin/bash -lc 'cd "/mnt/d/Program Files/NautilusTrader_RL_ES_IL_Route_A" && .venv-ubuntu/bin/python qa/run_route_a_teacher_student_train_screen_20261026.py --window-index 0'`.
## Incremental EMA delayed-close lifecycle check — 2026-10-27

Extended the native close-ack delay regression from two to six pending 1m bars. It still holds while the close order is unresolved, accepts the verified full fill at minute 10, and emits the one-time reverse entry only at minute 11 after rechecking EMA direction. The targeted Ubuntu 24.04 test passed (1 selected); Ruff passed. No market data or economic outcome was read.


## EMA reversal mirror and invalidation regression - 2026-10-27

Added a short-to-long mirror of the delayed native close-ack lifecycle case: the close remains pending across six 1m bars, a fully filled native close is acknowledged at minute 10, and the long reversal is emitted at minute 11. A same-minute replay stays flat. A native RISK_MAX_DRAWDOWN planner veto after the close fill clears the reversal candidate, and a later bar does not replay it. Added parameterized invalidation checks for both reversal directions; if EMA direction has flipped by the next legal recheck, the candidate is cleared without opening.

The complete lifecycle file passes under .venv-ubuntu (30 tests); Ruff passes on the test file, and focused MyPy passes on base.py, ema.py, and the lifecycle test. This change adds regression coverage only; it does not alter trading behavior or historical outcomes. Existing 2024 native EMA comparisons remain DEVELOPMENT_DIAGNOSTIC_ONLY; no market source files were read in this step.


## Historical native EMA reversal outcome - 2026-10-27

The existing .artifacts/route-a-ema-reversal-task1/paired_base_stress_summary.json contains old X1 EXIT_ONLY_V1 versus REVERSE_AFTER_FLAT_V1 native pairs for February, March, and April 2024 under Base and Stress costs. The reversal version emits both sides, correcting the old one-sided behavior in the affected blocks, but its net return and maximum drawdown are worse in all six cells. Net returns (old -> reversal) were February Base -2.33% -> -5.12%, Stress -6.70% -> -12.91%; March Base -3.31% -> -5.83%, Stress -8.21% -> -14.43%; April Base -0.72% -> -1.56%, Stress -2.77% -> -5.16%. Fees also increased in every pair. These are consumed Train development diagnostics classified SCENARIO_ONLY; the behavior correction is not an economic qualification or profitability claim.


## Frozen screen recipe-identity integrity repair — 2026-10-27

A source-independent audit found that resume and summary validation keyed native-run records by teacher family but did not compare the serialized teacher_recipe. A stale or misplaced run could therefore be attributed to the ROC X1/X7 or EMA reversal candidate named by its directory. The runner now checks the exact recipe on resume and in summaries, rejects non-registered scenario/seed slots, and derives active-screen completion from the exact expected key set. The original preregistered matrix remains reported separately.

The focused runner suite passes (13 tests); Ruff and focused runner MyPy pass. The preregistration preflight passes with the registered row still recorded, one of five staged imports present, and market_payload_read: false. Source-errata SHA-256 is 7db874aef4eef349b9eb19b27614454c06646a22e7f77b959fa3067b25bdd1b0.

This repair changes execution-evidence validation only. Teacher recipes, thresholds, source dates and roles, seeds, controls, scenario costs, and acceptance rules are unchanged. No market payload or economic outcome was read, no partial window was scored, and student training remains unstarted.


## ROC episode-reset latch correction - 2026-10-27

A lifecycle review found that the cost/ATR profit-lock rearm latch was intentionally preserved by on_discontinuity() across missing-bar gaps, but BaseTeacher1m.reset() invokes that same hook. A reused ROC teacher could therefore carry a prior episode's opposite-consensus requirement into a new episode and suppress valid entries. RocCostHysteresisTeacher.reset() now clears the episode-scoped latch, exit side, and opposite-run counter after the shared reset, while ordinary within-episode gaps continue to preserve the latch. The new regression covers both paths and confirms same-regime entry is available again after a new episode warms up.

The focused ROC suite passes (22 tests); Ruff and focused MyPy pass. The active ROC implementation is now bound in the existing runtime_code_sha256 map (module SHA-256 85c17057c0ab9e0dca98f18c7dfed021ffda1bdd135c5d8bd8ed9e8dc0d94f23). Read-only preflight passes with registry state recorded, one of five staged imports present, and market_payload_read false. Source errata SHA-256 is ea9a78ff5c2c5f43a2f59b13e5f0aecdcbfd2bbe58b5d98d0405b48957d0d900.

This is a lifecycle correctness repair; thresholds, candidates, exit recipe, window roles, source dates, seeds, costs, controls, and acceptance criteria are unchanged. No new market payload or scored outcome was read.

## Source and fold identity isolation correction - 2026-10-27

BaseTeacher1m previously reset only when episode_id changed. When source_id or fold_id changed inside the same episode, EMA could compare snapshots from different data identities; ROC rolling returns/consensus, pullback history, pending reversals, and ATR exit state could also cross source or fold boundaries. The base teacher now resets all rolling teacher and exit state on an episode, source, or fold identity change, then anchors the current identities before processing the next bar.

Focused regressions switch from a bearish source-A bar to the first bullish source-B bar, and from one fold to another within the same episode. Both confirm that the new identity does not create a cross-boundary EMA entry; a later bar in the new identity produces its own valid short cross. The ROC regression confirms that source changes clear volatility and consensus history. The lifecycle and ROC focused pytest suites pass (55 tests); Ruff and focused MyPy pass. The updated base.py hash is bound in the existing source erratum, and the read-only preregistration preflight passes. No market payload or native economic outcome was read; recipes, thresholds, window roles, costs, seeds, controls, and acceptance rules are unchanged.

## X3/X8 partial-fill ATR basis correction - 2026-10-27

X3 and X8 re-arm state when the native average entry price changes after an add-on/partial fill. The old logic initialized the new basis from the completed bar high/low, although that extreme may have occurred before the fill. X3 could therefore activate a profit lock and emit an exit on a gain the updated position never experienced; X8 could incorrectly hand control to the indicator exit. The revised logic starts the new basis at the completed close for that bar, then resumes normal high/low tracking on later bars. X5 inherits the corrected X3 path.

Long and short X3 regressions reproduce the pre-fill-extreme false exit; the X8 regression reproduces the premature handoff. The lifecycle and ROC focused suites pass (59 tests); Ruff and focused MyPy pass. The updated exits.py hash is bound in the existing source erratum, and the read-only preflight passes. No native economic result or market payload was read; frozen recipe thresholds, fees, dates, roles, seeds, controls, and qualification rules are unchanged.

## X2 first-observed position-bar ATR basis correction — 2026-10-27

A source-only lifecycle audit found that X2 initialized its first observed open-position trail from the full completed bar high/low. A native fill can occur after that extreme, letting a pre-fill spike move the trail past subsequent prices and cause a false exit. X2 now initializes the first observed favorable extreme from the completed close; later bars retain normal high/low trailing. Long and short regressions reproduce the false-exit paths. This is a source correctness repair; thresholds, recipes, costs, source scope, window roles, seeds, controls, and qualification rules are unchanged. The source erratum binds the updated exits.py hash. No market payload or economic result was read. Existing historical X2 outcomes remain development diagnostics and were not rerun.

Validation for this correction: the focused lifecycle and ROC hysteresis suites pass (71 tests); Ruff passes on exits.py and the lifecycle test; focused MyPy reports no issues in 2 source files. The preregistration preflight passes with the registry recorded, one of five staged imports present, and market_payload_read false. Window 0 remains unscored until all frozen source imports are present.

## X3/X8 first-observed position-bar ATR basis correction — 2026-10-27

A follow-up source audit found that X3/X8 still initialized from the full bar high/low when an open position was first observed. X3 could then activate a cost lock using a pre-fill excursion; X8 could falsely satisfy its four-times-cost handoff and disable the ATR stop. Both policies now initialize that first observed basis from the completed close, consistent with X2 and the existing average-price-change correction. Long/short regressions cover premature X3 activation and X8 handoff. No recipe, threshold, data role, cost, control, or acceptance rule changed. No market payload or economic outcome was read.

Final verification for the X2/X3/X8 first-observed-fill corrections: the focused lifecycle and ROC hysteresis suites pass (71 tests), Ruff passes, and focused MyPy reports no issues in 2 source files. The frozen preregistration preflight passes with one of five imports present and market_payload_read false. No native window was scored.

## X2/X6/X8 first-bar adverse ATR stop check — 2026-10-27

The first-observed-bar close-only initialization safely excluded pre-fill extremes but could postpone an X2/X8 stop even when that close was already more than 3×ATR adverse to the native average entry. X2/X6 share AtrTrailingExit and initialize the favorable extreme as max(entry, close) for longs or min(entry, close) for shorts, then check that close against the stop immediately; X8 applies the same rule in its handoff path. This retains the pre-fill-extreme correction and preserves first-bar loss protection. Long/short regressions cover both policies; no source market payload or economic result was read.

## X3 first-bar close-based activation correction — 2026-10-27

The initial X3 fix excluded pre-fill high/low but returned before considering a qualifying completed close. X3 now initializes from native average entry and close, excludes the full-bar extreme on first observation or basis change, and evaluates the existing cost activation threshold on that close immediately. X4/X5 inherit the shared path. Long/short tests verify close-based activation without crediting pre-fill spikes. No thresholds, costs, or recipes changed; no market data or economic outcome was read.
## X2 动态退出输入完整性修复（2026-10-27）

只读代码审查发现，ATR 退出器在 OHLC/ATR 特征缺失时会返回无退出信号，而通用教师仍可能仅凭 EMA/ROC 等入场字段开仓。现在所有启用 ATR 退出的教师在账户空仓时遇到必需退出字段缺失会保持空仓并记录 ATR_EXIT_NOT_READY；已有持仓遇到这些字段缺失则显式抛出数据完整性错误，避免静默继续而遗漏冻结退出规则。X1 无 ATR 退出依赖，行为不变。

生命周期与 ROC hysteresis 定向套件共 76 项通过；Ruff 和聚焦 MyPy（base.py 与生命周期测试）通过；git diff --check 通过。现有源勘误已绑定 base.py 新哈希，source errata SHA-256 为 d5cc6c5642d2a00113b1060e234d50d21dfe1b542a68ac7d39ff67a8f61c8002。只读预检通过：注册仍为 recorded，4 个教师窗口、4 个学生窗口，5 个阶段导入仅 1 个存在，market_payload_read=false。未读取行情载荷或生成经济结果；窗口 0 未评分，配方、阈值、成本、窗口角色、种子与验收门槛均未改变。