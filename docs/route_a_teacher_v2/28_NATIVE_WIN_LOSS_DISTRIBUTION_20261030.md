# Native win/loss distribution diagnostics — 2026-10-30

## Purpose

The 2026-10-28 audit requires economic reporting that does not substitute win rate for expected net return. Extend the native economics summary with separate completed-trade counts and average PnL for winners and losers, while keeping breakeven cases explicit.

## Change

`teacher_research.meta_evaluation.summarize_native_economics` now reports:

- `winning_trades`, `losing_trades`, and `breakeven_trades`;
- `average_winning_trade_pnl` and `average_losing_trade_pnl` (null when that class has no observations).

These statistics use only `ORDINARY_COMPLETE` native cases. Existing aggregate net PnL, win rate, and average PnL remain unchanged; right-censored and other non-complete cases remain outside the completed-trade distribution.

## Verification

- `pytest packages/route_a_imitation/tests/test_meta_evaluation.py tests/test_route_a_meta_native_engine.py` — **8 passed**.
- `ruff check` on the changed implementation and test files — **passed**.
- `git diff --check` — **passed**.

The synthetic mixed fixture verifies one winner (+8 USDT), one loser (-12 USDT), and one breakeven trade are reported in their respective groups. This is an accounting/reporting check only; it provides no market-profitability evidence. Current pre-registered source checks still show zero completed screen windows and no market payload read.

## Audit alignment and limits

This closes one reporting gap in the audit's per-trade economic diagnostics. It does not resolve the higher-priority run-identity/resume concern (T0), create a candidate opportunity audit (T1), establish candidate/label contracts (T2), or qualify a profitable teacher. Historical negative D1 results remain intact; no new market data was read.
