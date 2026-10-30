# Native case recipe reconciliation — 2026-10-30

## Change

Before replaying an exact selected Meta candidate against a native trade case,
the reconciliation adapter now verifies that the case carries the requested
cost scenario, the X1 exit family, and `SCENARIO_ONLY` economic status. A
scenario, exit, or evidence-status mismatch is retained as `NOT_COMPARABLE`
with an explicit reason and never reaches the price replay callback.

This prevents a Stress case or a different exit/evidence class from being
silently compared with a Base X1 replay.

## Verification

- Ubuntu 24.04 / Python 3.14.7: native Meta funnel, native engine, and training
  integration tests — **5 passed**.
- Expanded candidate, label, policy, native funnel, native rollout, engine,
  training integration, and study-runner suite — **109 passed**.
- Ruff and `git diff --check`: **passed**.

The tests use synthetic native receipts. No market payload was read and no new
economic result was produced.
