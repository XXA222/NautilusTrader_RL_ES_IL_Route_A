# T0–T3 development evidence, 2026-10-29

Classification: D1_DEVELOPMENT_DIAGNOSTIC_ONLY / SCENARIO_ONLY. This record is not G1/G2, independent OOS, or proof of real profitability.

## T0

RunIdentityV1 binds the native screen's protocol/errata/code, feature, scaler, warmup and scored source, instrument rules, execution template, scenario costs, candidate recipe, seed and window. Resume and summary use the same verifier; missing identity is LEGACY_INCOMPATIBLE. Existing native reports are not rewritten. The pre-score source erratum binds the modified runner/native/helper files; no fresh scored runs existed when this binding was updated.

Ubuntu focused runner tests: 42 passed. Ruff passed; focused MyPy passed for identity and screen runner. Metadata-only preregistration preflight passed, 2/5 staged imports present, market_payload_read=false. The incomplete raw-only window is not a completed import.

## T1

qa/analyze_route_a_entry_opportunities.py consumes the retained D1 native decision CSVs and per-trade exit metrics. It requires decision/trade case joins and exact teacher/exit/scenario slots, and rejects cashflow mismatch. It reports close-to-close 5/15/60/120 minute directional moves only when all intervening bars are present; otherwise the observation is censored. MFE and horizon moves are not executable returns.

Strict-join v2: .artifacts/route-a-entry-opportunity-audit-20261029-v2/summary.json and summary.md; 48 runs. Base: 26,984 candidate rows, 26,972 completed trade cases, 12 censored; gross pre-slippage price attribution +237.65 USDT, fees 6,699.17, slippage attribution 1,362.46, funding -0.70, net -7,824.67. Stress: same counts, gross +230.98, fees 13,345.01, slippage 6,694.27, funding -0.96, net -19,809.25. These are aggregates of reset accounts and overlapping strategy/scenario paths, not independent trade counts or a compounding account curve. They describe costs and observed prices without establishing causal predictive edge.

## T2

candidate_event_union.py groups causal same-snapshot/minute/side proposals and keeps teacher reasons; conflicting snapshots, duplicate teacher proposals, or mixed source/fold/schema fail. conditional_outcome_labels.py requires the replay to bind exit, execution and scenario; counterfactual labels remain MODEL_COUNTERFACTUAL, never native fills, and provide censor/purge handling. The adapter consumes existing 1m TeacherSignal without HTF direction veto.

## T3 initial synthetic-only baseline

entry_meta_policy.py implements a small logistic and net-return model trained on earlier Train-fit and calibrated on later Train-calibration, with source/fold/schema/recipe and label-clock checks. It emits NO_TRADE when a position or order already exists, outside fit support, or expected net value is not above the fixed margin. It is not connected to the production native teacher or qualified using historical/fresh market data. No G1/G2 criteria changed.

Ubuntu combined T0/T1/T2/T3 and existing opportunity label/model tests: 60 passed. The candidate availability boundary received one additional regression after this run. Ruff and focused MyPy passed for the new files. No student, RL or GRU was trained.

## Remaining

The 2026-10-25–10-30 teacher screen is BLOCKED_BY_SOURCE until the full registered market and funding import is available, and the window has ended. After completing the registered import and its read-only preflight, execute:

    .venv-ubuntu/bin/python qa/run_route_a_teacher_student_train_screen_20261026.py --window-index 0

T3 still needs integrated, frozen-recipe Train-fit/calibration data and independently registered screen validation. No teacher or student qualification is established.

## Source-controlled D1 summaries and inference integrity

The strict-join summary.json and summary.md are also preserved alongside this report as 18_entry_opportunity_summary.json and 18_entry_opportunity_summary.md; they contain the same D1 diagnostic-only results.

BC inference now rejects nonfinite or malformed intent logits, rejects invalid softmax, and requires reset on source/fold/schema change before committing episode state. These are engineering protections, not evidence of trained student profitability.

Candidate group IDs also bind the full observed market, timestamps, account and pending order state; substituting a different observation under the same event ID is rejected. Focused synthetic regressions cover the identity boundary.
