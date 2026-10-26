# Teacher decision

**Decision: `BLOCKED_BY_SOURCE`; `NO_ELIGIBLE_ALPHA_TEACHER`.**

- Do not promote or train the formal student.
- Do not relax the frozen probability, expected-net, Q10, support, cost, or G1/G2 thresholds.
- The prior E1/E2 screen remains a consumed Train-inner diagnostic, not OOS. It had 138 candidates and zero frozen-gate selections; the stored report contains first-rejection counts but lacks the event-level score/gate-bit data required for the full R3 tables.
- The original monthly funding source physically overlapped protected Test/Holdout intervals. The old screen must not be rerun from that source.
- The October public-data attempt is not eligible for research use: the access ledger has no development row for its preregistered interval, and its funding API request timed out. The request-boundary defect is now fixed and unit-tested, but the fix does not retroactively authorize or unconsume the existing raw files.
- Account-specific historical fee/margin authority and an eligible `OBSERVED_NATIVE_CASE` have not been verified. Scenario-only or model-counterfactual outcomes cannot satisfy G2.

## A0–A7

| Gate | Status | Evidence / remaining condition |
|---|---|---|
| A0 Ubuntu | PASS | Ruff all project Python paths; mypy 225 source files; targeted Ubuntu tests pass. Full all-project pytest is not claimed. |
| A1 Source | BLOCKED_BY_SOURCE | Physical overlap guards and bounded funding response validation are tested; there is no eligible, preregistered, ledger-recorded fresh Train source available for a real run. |
| A2 Causal | PASS | Synthetic prefix-causality and admission-order regressions in the portfolio replay test group. |
| A3 Portfolio | PARTIAL / NOT_RUN | Synthetic account ledger covers long/short, costs, funding, MTM and censoring; required M2.2/native reconciliation remains unrun. |
| A4 Cohort | BLOCKED_BY_SOURCE | Diagnostic code is connected, but no legal data run can produce the missing event-level scores, 16-combination table, or full F0–F5 outcome report. |
| A5 Dynamic | PASS | Shared dynamic exit/label logic and long-hold/censor regressions pass; no fixed 240-minute forced exit is introduced. |
| A6 Alpha | BLOCKED_BY_SOURCE | No eligible fresh Train-fit/calibration/screen experiment with paired risk-matched evidence. Existing fit and screen outcomes are not substituted. |
| A7 G2 | BLOCKED_BY_SOURCE | No verifier-accepted economic authority or native educational case; status stays `SCENARIO_ONLY` where applicable. |

## Next actionable condition

Before any real market replay, establish a continuous, unconsumed Train-fit/calibration/screen interval, freeze its complete source/cost/exit recipe, register the exact development access before source reads, and verify that every physically decoded source (including funding) avoids protected Test/Holdout. Separately obtain auditable economic evidence; do not infer account-specific fees or margin from public candles.
