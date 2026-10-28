# 14 - Native Gym teacher feedback guard (2026-10-28)

## Change

Added synthetic provider regressions for ordinary Gym rollout and the screening path `rollout_with_m22_trace`. Accepted paths assert event order: the configured per-barrier native order-fact verifier runs first, the L6 no-external-cashflow verifier runs second, and only then does `observe_native_action` receive the exact advice, trace, and L3 order facts associated with the barrier. The M2.2 fixture stops immediately after `decision_observer`, so it verifies per-step callback ordering only.

The rejection cases also assert that no teacher callback occurs when either per-barrier verifier rejects the facts. No production teacher or execution code, candidate recipe, threshold, data source, scoring output, or economic gate changed.

## Verification

- Ubuntu full-file test: `.venv-ubuntu/bin/python -m pytest packages/route_a_imitation/tests/test_v03_native_gym_rollout.py -q` - 14 cases passed.
- Ruff on the test file - passed.
- Focused MyPy on the test file with imported modules skipped - passed.
- `git diff --check` on the test file - passed.

## Evidence boundary

The fixture uses synthetic Gym information. It verifies callback ordering against the configured per-barrier verifiers only. It does not verify native Nautilus history, the separate post-rollout retained-snapshot M2.2 reconciliation, native economic authority, profitability, G1/G2, or project acceptance. It reads no market payload and creates no economic result.

A fresh metadata-only preregistration preflight on 2026-10-27 returned `PREFLIGHT_PASS_STAGED_SOURCES_PARTIAL_METADATA_NOT_READ` with preregistration SHA-256 `e7f795fce9f8539e58085f6bd85449f96edb53324cebcbb6380dec47c8d38aca`. One of five staged imports is present and `market_payload_read` is false. No window was scored and no student training was started.
