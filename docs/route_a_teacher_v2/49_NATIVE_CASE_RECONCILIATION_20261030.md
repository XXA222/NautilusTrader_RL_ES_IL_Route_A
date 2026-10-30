# Native candidate-case reconciliation — 2026-10-30

## Finding

The Meta reconciliation adapter previously fell back to `(decision timestamp, side)` when a native case lacked the selected candidate event ID, and it silently omitted open/terminal-censored cases from the reconciliation denominator. That could label an ambiguous pair as comparable or hide incomplete native cases from the reconciliation summary.

## Change

Every native case must first match an exact selected candidate event ID, decision clock, direction, and teacher-source tuple. Missing IDs are `UNVERIFIED_CANDIDATE_IDENTITY`; unmatched IDs remain `UNMATCHED_CANDIDATE`, including for censored cases. Only after identity matches are open and terminal-censored cases recorded as `NATIVE_CASE_INCOMPLETE` with their native status. The summary reports total cases, ordinary completed cases, comparable cases, and disposition counts; duplicate native case IDs fail closed.

## Verification

- Regression cases verify exact candidate matching, missing-identity rejection, and retention of open-censored cases.
- Native Meta funnel and integration regressions: **34 passed**. Ruff and `git diff --check` passed.
- Follow-up after applying candidate identity checks to censored cases: native funnel and end-to-end integration regressions **5 passed**; Ruff and `git diff --check` passed.

## Evidence boundary

This tightens pairing and denominator reporting. It does not create native market data or establish a profitable teacher.
