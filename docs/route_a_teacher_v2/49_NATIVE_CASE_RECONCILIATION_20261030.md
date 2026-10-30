# Native candidate-case reconciliation — 2026-10-30

## Finding

The Meta reconciliation adapter previously fell back to `(decision timestamp, side)` when a native case lacked the selected candidate event ID, and it silently omitted open/terminal-censored cases from the reconciliation denominator. That could label an ambiguous pair as comparable or hide incomplete native cases from the reconciliation summary.

## Change

Completed native cases now require an exact selected candidate event ID and exact teacher-source tuple before replay or economic comparison. Cases without a candidate identity are explicitly `UNVERIFIED_CANDIDATE_IDENTITY`. Open and terminal-censored native cases remain in the output as `NATIVE_CASE_INCOMPLETE` with their native status. The summary reports total cases, ordinary completed cases, comparable cases, and disposition counts; duplicate native case IDs fail closed.

## Verification

- Regression cases verify exact candidate matching, missing-identity rejection, and retention of open-censored cases.
- Native Meta funnel and integration regressions: **34 passed**. Ruff and `git diff --check` passed.

## Evidence boundary

This tightens pairing and denominator reporting. It does not create native market data or establish a profitable teacher.
