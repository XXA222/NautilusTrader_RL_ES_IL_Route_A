# 11 — X1/X2 shared-entry matching correction (2026-10-28)

## Scope and source boundary

This correction re-aggregates the existing 53,968-row S06 Train per-trade output at `.artifacts/route-a-exit-economics-task3/per_trade_exit_metrics.jsonl`. It does not read a new market source, access Test/Holdout, or rerun a native experiment. The full-policy account results and their `SCENARIO_ONLY` classification are unchanged.

## Corrected match rule

Within each teacher, block, and scenario, a shared entry requires the same `entry_fill_ts_ns`, side, and entry notional after rounding to six decimal places (1e-6 USDT). `entry_decision_ts_ns` is not part of the identity because the row-level pair ledger defines an entry by its fill timestamp. Duplicate fill-time/side keys now fail closed. Realized exit deltas include only pairs where both cases are `ORDINARY_COMPLETE`; terminal-censored cases are counted and excluded from those deltas.

## Reconciled counts

- Fill-time/side overlaps before notional validation: 16,092.
- Rejected for different rounded entry notionals: 784.
- Shared entries with matching rounded notionals across all statuses: 15,308.
- Ordinary-complete pairs included in realized exit comparisons: 15,299.
- Matching-notional pairs excluded from realized exit deltas due to censoring: 9 (five X1 complete/X2 censored; four censored on both sides).

The 15,308 all-status count now agrees with the row-level CSV/JSON ledger. The aggregate report separately exposes the complete-only count used for realized exit statistics. The loss-attribution report uses the same fill-time, side, and rounded-notional rule and reconciles its 24 groups against that complete-only count.

## Interpretation

This fixes a pairing/reporting inconsistency; it does not change any policy path or turn the post-hoc common-entry subset into a causal counterfactual. Every S06 account remains `SCENARIO_ONLY`; the native block returns remain negative. No teacher, student, profitability, or project-acceptance criterion passed as a result of this correction.

## Verification

- Targeted Ubuntu tests: 6 passed.
- Ruff on both analysis scripts and the new matching tests: passed.
- Targeted MyPy on both analysis scripts and the new matching tests: passed.
- Rebuilt the X1/X2 aggregate and loss-attribution reports from the existing S06 Train outputs; 24 matched groups reconciled.
- Aggregate JSON SHA-256: `fadaf19b4f8bc2b41bbcbf22b69bfab3a5e07dcf550196e46156508b78f6a29f`.
- Per-trade input SHA-256: `5e7f5b2561f91b8922b00af81afaf8f715c4c662af7f9331e0e59083de45249c`.
- Loss-attribution JSON SHA-256: `2037a22cd4299c3ef91e27eb2ca48749769cf59668600690f1076d49e367f217`.

## Next eligible economic run

The current metadata-only preregistration preflight passes, but only 1 of 5 staged imports is present and no market payload was read. The first registered scoring window extends through 2026-10-30, so it is incomplete on this update date. Once the complete window and its staged imports are available, rerun the metadata preflight and freeze/check the experiment recipe before scoring that authorized Train matrix. Do not score the partial window.
