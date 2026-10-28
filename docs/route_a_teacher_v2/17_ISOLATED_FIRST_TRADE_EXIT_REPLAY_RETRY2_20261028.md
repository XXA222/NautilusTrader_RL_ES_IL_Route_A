# Frozen diagnostic: isolated first-trade X1/X2 native replay

Status: FROZEN_BEFORE_RETRY2_RUN; no retry results are included in this file.

## Question

For the same first naturally generated and natively filled entry, how do the realized cash flows differ when only the indicator exit changes from X1 to X2, with later re-entries suppressed?

## Frozen inputs

- Protocol and source: route-a-1m-multiteacher-train-diagnostic-20261026-v1, source snapshot data/binance_ethusdt_20231230_20240415/market.parquet, using only its registered Train diagnostic roles. The protocol declares all_results_are_development_diagnostics: true, no student training, and no protected monthly archive overlap.
- Teachers: T01_EMA_9_21, T02_ROC_20_60, T03_DONCHIAN_20, T04_ZSCORE_60.
- Blocks: the three frozen inner-screen-1, inner-screen-2, and inner-screen-3 windows; no other dates.
- Exit variants: X1 versus X2. Entry recipe, instrument, feature stream, initialization, starting account, native account/risk rules, and data remain the existing preregistered values.
- Scenarios: Base (5 bps taker fee and 1 bp slippage per leg) and Stress (10 bps taker fee and 5 bps slippage per leg).
- Seed: 20261026 in both exits.
- Matrix: 4 teachers x 3 blocks x 2 scenarios = 24 paired slots; two native runs per slot.

## Frozen isolation and matching rule

Each run starts from the same flat account state. The teacher may act normally until its first filled entry has fully closed under the native engine. After that close is confirmed flat with no pending orders, the wrapper returns FLAT on subsequent decisions. The Gym rollout still continues through the complete registered block, preserving every 1-minute barrier and existing M2.2 verification; no later trade can affect the first trade's cash flow.

Pair the X1 and X2 cases only when both are ordinary-complete and their first entry fill timestamp, side, and entry notional (rounded to 1e-6 USDT) match. Chronological first fill is selected without consulting outcomes. Missing, censored, liquidated, or mismatched cases remain explicit unpaired rows; no replacement entry is selected.

## Frozen outputs and interpretation

Record entry/exit identities, gross cash PnL (realized_pnl + fees - funding), fees, slippage attribution, funding, net realized PnL, MFE, MAE, holding minutes, and maximum closed-bar peak-to-current gross-PnL giveback reconstructed from native account equity and barrier-level fee/funding records. Verify the close-bar cash identity against the completed native trade case. Keep slippage as attribution because it is already reflected in native fill prices.

All rows must retain SCENARIO_ONLY and DEVELOPMENT_DIAGNOSTIC_ONLY. This consumed Train diagnostic does not estimate a population causal effect, generalization, G1/G2, or profitability. No parameter is tuned from these outcomes. Results, if produced, go in a separate report; this frozen specification remains unchanged.


## Pre-run implementation binding (frozen before native replay)

The old S06 protocol's feature_recipe path in the worktree has since changed. The exact preregistered bytes were recovered from the existing immutable snapshot at .artifacts/route-a-x2-snapshot-fix-task3/frozen/route_a_1m_teacher_library_v1.json; SHA-256 is 27f03e698a58dde7aa2c09cf161b83a491a20e0de167608f482d6073aa67818e. The old native preregistration verified successfully with this byte-identical override. It also verified the original source manifest, completion marker, rules, and execution template without reading market rows.

This diagnostic evaluates the current working-tree teacher and native replay implementations below. It is a new consumed-Train development diagnostic, not a reproduction of the historical S06 account runs. These implementation bytes are frozen for this matrix; any code change before completion requires a new output path and a revised pre-run binding.

| Component | SHA-256 |
|---|---|
| qa/route_a_isolated_exit_replay.py | 4668a0c88f6d8191802d1c6970a150cde9b3d3a721aba7b3ce6a306544825df5 |
| qa/run_route_a_1m_teacher_native.py | 7c05ec8fd5b37693b72f3139e94db3801480e99cac6e1312469bdcf8f153e3d8 |
| qa/run_route_a_train_student_screen_20261020.py | 238c1e5aecdff77569b4f7dcc56d250995300a400b60b438e1edd52c4e767049 |
| packages/data/src/rles_data/features.py | 43acda5260799f3692ae8ca03ca3ec5f8b133fd6796cd4b59968a21f4db99c1f |
| packages/route_a_imitation/src/rles_route_a/indicators_1m/streaming.py | e6f8e842c7b4fd1b62ffb23895675c7b181123871e71de0982b35786f8e9b3ad |
| packages/route_a_imitation/src/rles_route_a/indicators_1m/contracts.py | d0df863035d26e7e5b39e1144d1f24c03fa561de02a60076d057296f5495ac01 |
| packages/route_a_imitation/src/rles_route_a/teachers_1m/base.py | d70bb785ae38904f46e0f8eb5ce74de6fe7c83a53f73f13c6bac9a5fd53f3159 |
| packages/route_a_imitation/src/rles_route_a/teachers_1m/ema.py | 1c7531e1030346fbc4297265b25e17eddbc57435a6622f52167be150fd488a9d |
| packages/route_a_imitation/src/rles_route_a/teachers_1m/momentum.py | 44184aee2c11840a152b0bf2f7c507f49b600f9e5b1e6ea691258ab3610611a5 |
| packages/route_a_imitation/src/rles_route_a/teachers_1m/donchian.py | d8c3a16e839354fbc1e285602e483dc6f9c963aed1095a02abc758e1ff16edcc |
| packages/route_a_imitation/src/rles_route_a/teachers_1m/zscore.py | e44853a58da117fd46c81f98d0e2f5137d54d8a0b397f33a415c3c889cc17a0c |
| packages/route_a_imitation/src/rles_route_a/teachers_1m/exits.py | c824bfadbbdbc68cc5f28b8366d2903bd0827118efb1126403a9997ce2dfaefa |
| packages/route_a_imitation/src/rles_route_a/teachers_1m/registry.py | 87e497970e98b3dd0f26004b5e57aa57d3c9e42bf767821e84edea4893294317 |
| packages/route_a_imitation/src/rles_route_a/teachers_1m/contracts.py | b3ec7ee4d5b9431e8266914bff892336debad0457b8865fbc8b10ec57fa41b03 |
| packages/route_a_imitation/src/rles_route_a/teachers/protocol.py | f28039fbe3b11298a6c49234ba235498b31533e91f058a1e2a75022597181c0f |
| packages/route_a_imitation/src/rles_route_a/integration/l2_adapter.py | 273db45bec7ef2bcc64cfef29600f663bd772508bdc36657b63098f592dc1368 |
| packages/route_a_imitation/src/rles_route_a/integration/gym_rollout.py | 24cc7cc2ebc949ff0d16f391632629f6864171c17ea718fe4a2f0ee471e8975c |
| packages/route_a_imitation/src/rles_route_a/evaluation/teacher_case_report_1m.py | 72b2133b38bfe1b9d8af8ba9d8372d3ca1b0895ae9406a1d31e472c186d78940 |
| packages/route_a_imitation/src/rles_route_a/evaluation/teacher_funnel_1m.py | e32749af77a5a237172c1c9c1a06f7beec0218229abdc9abf4240de98bef4002 |

The output may be interpreted only as SCENARIO_ONLY and DEVELOPMENT_DIAGNOSTIC_ONLY.

## Retry 2 eligibility and pre-run binding

Retry 2 keeps the parent question and every economic input unchanged. Its only implementation-binding change is the isolated replay wrapper SHA-256 in the frozen component table. The new source-independent guard rejects a direct sign flip before verified flat closure. The focused wrapper test suite (seven tests), Ruff, and focused MyPy pass before this freeze; no market row was read for those checks.

The active project objective explicitly permits consumed Train data for development diagnostics when outcomes remain marked DEVELOPMENT_DIAGNOSTIC_ONLY and are not represented as OOS, profitability, G1, or G2. This run is the task-three same-entry X1/X2 exit cashflow diagnostic; it does not tune or select a teacher. The frozen S06 source remains train_inner and declares every result diagnostic-only and no student training.

Metadata-only authorization checks passed before this freeze: the current preregistration SHA-256 is 96a5f9f8c5eaee9c65830a4bf9e2e44a79b0b6bede721b4813522a859070c036; 220 development/recorded access rows matched the exact registered market snapshot marker and covered source scope; the native protected-monthly-source guard passed. These checks read preregistration, import metadata, the research ledger, and source identity receipts, but no market parquet rows. No Test or Holdout market data was opened.

The earlier first attempt remains preserved with its preparation-only scaler file. Retry 1 was cancelled before market read. Retry 2 must use the previously absent output path .artifacts/route-a-exit-isolated-first-trade-retry2-20261028. Any failure must be preserved at that path and must not be overwritten or resumed without a new freeze.

All outcomes must remain SCENARIO_ONLY and D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2. The report cannot establish a population causal effect, generalization, G1/G2, or profitability. No student training is authorized by this diagnostic.

