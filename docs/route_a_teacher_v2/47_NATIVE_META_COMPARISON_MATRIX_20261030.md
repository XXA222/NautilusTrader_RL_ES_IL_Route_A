# Native Meta comparison matrix fix — 2026-10-30

## Finding

The frozen preregistration lists `ORIGINAL_8_X1`, but the study runner dispatches only the eight individual teachers, Meta, FLAT, fixed long, and fixed short. The legacy control factory also has no `ORIGINAL_8_X1` implementation. The aggregator previously removed the control from its expected matrix without reporting the planned-versus-active scope difference, so its 48-cell active matrix could appear to be the full preregistered study.

## Change

The comparison report now distinguishes the full preregistered plan (52 cells) from the executable active matrix (48 cells). A shared control-scope validator drives the runner and comparator: it runs supported controls from the frozen config, records `ORIGINAL_8_X1` as a preregistered control outside the active matrix, keeps its planned rows as `MISSING`, and rejects unknown or duplicate control IDs before source loading. Reports expose `status`, `active_status`, and `preregistered_controls_outside_active_matrix`. They do not invent an aggregate teacher or treat active-matrix completion as full preregistration completion. Missing Meta cells also remain explicit.

The native comparison also verifies the per-cell provenance receipts before aggregation. It rejects mixed receipt/no-receipt matrices and rejects protocol, market-source, seed, strategy, scenario, or time-window mismatches. The result states `run_receipt_status`; pure synthetic/legacy matrix inputs without receipts are marked `UNVERIFIED_NO_REGISTERED_RECEIPTS` and cannot be mistaken for a registered source run.

Each Meta-to-baseline comparison now reports per-window deltas for completed trades, gross PnL before costs, fees, slippage, funding, net PnL, account costs, inferred gross account PnL, win rate, average winning/losing/overall trade PnL, and average holding time. It also reports LONG and SHORT count, net PnL, and win-rate deltas. If either run lacks a denominator or a metric, the delta is `null` instead of implying a zero change.

## Verification

- Ubuntu 24.04 / WSL2 / Python 3.14.7: comparison and runner regression tests verify 52 planned cells, 48 active cells, explicit aggregate-control omission, and separate active/full completion states.
- Native engine, native Train-to-screen synthetic end-to-end, runner, matrix, control-scope, temporal-split, mixed-receipt rejection, and economic-delta tests: **27 passed**. Ruff passed for changed Python files; `git diff --check` passed.
- The live study was not run. Its registered source remains future dated and the last metadata-only preflight reported zero completed screen windows and no market payload read.

## Evidence boundary

This fixes comparison-runner integration and reporting defects. It is not an economic result and does not qualify any teacher.
