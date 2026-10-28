# F04 — Frozen anti-noise candidate definitions

Status: `C1_CORE_IN_COMBINED_D1_REPLAY_UNQUALIFIED; C2_REDUNDANT_WITH_X1; C3_REQUIRES_FRESH_TRAIN_FIT`.

The only motivation is the already consumed T02 ROC/X1/Base diagnostic: 8,310 cases, about 2,484 USDT fees plus slippage, about 94 USDT inferred price gross, and short average holds. These figures are development evidence, not fresh validation. Preserve T02 baseline outputs unchanged.

## Candidate IDs

1. `ROC2_CONFIRM_REARM2_V1`: require ROC20 and ROC60 to agree in direction for two consecutive READY 1m bars before entry. After the current X1 consensus-break exit, require two consecutive non-consensus bars before rearming. Native hard-risk exits are never delayed.
2. `ROC2_CONFIRM_ATR_MARGIN025_V1`: candidate 1 plus `min(abs(ROC20), abs(ROC60)) / (ATR14 / close) >= 0.25` on the entry bar. This is a signal-strength filter only; it is not an expected-profit estimator and does not replace an edge-versus-cost gate.
3. `ROC2_STRESS_GROSS_EDGE_LB30_V1`: candidate 1 plus a Train-fit-only, later-Train-calibrated lower confidence bound for price gross edge of at least 30 bps, the frozen Stress round-trip cost, before native account replay. This is not evaluated on D1 and is not an automatic qualification rule; full policy results still must beat risk-matched controls on independent blocks.

All candidates retain the existing entry size, X1 exit, market-order policy, and Base/Stress costs. The first two have no fitted parameters. The 0.25 margin is a frozen engineering diagnostic value, not optimized on the observed paths. Candidate 3 cannot be instantiated without the new legal Train fit/calibration split required by F05.

The frozen candidate IDs are `ROC2_CONFIRM_REARM2_V1` and `ROC2_CONFIRM_ATR_MARGIN025_V1`.

## Native result and candidate overlap

The native cost-aware X1 recipe includes two-bar ROC consensus confirmation and two-bar neutral rearming, plus an entry rule requiring the weaker signed ROC edge to exceed round-trip cost plus `max(0.25 * ATR14_bps, 0.5 * realized_volatility_bps)`. Its paired S06 D1 result against the original T02 baseline reduced the compounded loss from −22.14% to −3.07% Base and from −49.51% to −4.69% Stress, but every candidate block remained negative. This compares the combined cost-aware/hysteresis recipe; it does not isolate the causal contribution of C1 alone. It is a substantial churn/cost reduction, not a qualified teacher or fresh validation. Results: `.artifacts/route-a-roc-hysteresis-task2/paired_native_results.json`.

`ROC2_CONFIRM_REARM2_V1` is therefore already represented by the native X1 policy and has failed the positive-return criterion on consumed D1. `ROC2_CONFIRM_ATR_MARGIN025_V1` adds no entries or vetoes to that policy: its 0.25 ATR edge floor is implied by the stricter existing `round-trip cost + 0.25 * ATR14` gate. Do not spend a duplicate native run on C2. Candidate 3 remains unevaluated; fitting/calibrating it on the already-consumed windows cannot be represented as fresh validation.

## Development-only diagnostic boundary

On consumed S07 decision rows, it is permissible to report original eligible-bar denominator, candidate-qualified entry events, vetoed events, and transition delay. It is not valid to translate reduced signal counts into candidate account returns or an assumed trade count: exits and later re-entry paths would change. A full net economic comparison requires a legal fresh source and native replay. Until then no candidate is selected or frozen as the teacher.

The exact raw T02 crossing and order funnel is in `02_COST_AND_CHURN_REPORT.md`; no new candidate performance run has been represented as complete.

## D1 signal-count diagnostic (not candidate selection)

Applied the frozen rules to the existing T02 ROC/X1/Base decision rows for the three consumed blocks. This reads local decision CSV/indicator snapshots only; no market archive, new account replay, or candidate return was produced. A disagreement/neutral counter resets on either consensus side; a candidate needs two consecutive agreed bars and two non-consensus bars to rearm. Candidate 2 also requires normalized ROC strength >= 0.25 at confirmation.

| Block | READY denominator | Baseline entry signals | Candidate 1 entries | Candidate 2 entries | C2 margin vetoes |
|---|---:|---:|---:|---:|---:|
| inner-screen-1 | 41757 | 3258 | 1571 | 1434 | 215 |
| inner-screen-2 | 44637 | 3466 | 1750 | 1603 | 230 |
| inner-screen-3 | 20158 | 1586 | 751 | 693 | 96 |

Across blocks: 106552 READY bars, 8310 baseline entry events, 4072 C1 events, 3730 C2 events. These event-count changes do not predict account costs or outcomes; full-policy economics remain unevaluated. Detailed counts are saved in the accompanying JSON file.
