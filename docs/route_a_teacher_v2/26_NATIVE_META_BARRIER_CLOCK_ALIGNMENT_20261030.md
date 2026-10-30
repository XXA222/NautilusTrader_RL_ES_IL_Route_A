# Meta native barrier clock alignment (2026-10-30)

Status: `ENGINEERING_TRACE_INVARIANT_VERIFIED; MARKET_ECONOMIC_SCREEN_PENDING_SOURCE`.

## Correction

`run_registered_meta_rollout` previously required candidate decisions and native receipts to have the same count as retained M2.2 barriers, but did not compare their decision clocks. A reordered or shifted trace with the same number of rows could therefore pass this completeness check.

The runner now requires exact timestamp-by-timestamp equality among candidate observation clocks, native receipt trace clocks, and the retained M2.2 barrier clocks before it writes the complete Meta trace. Missing barrier trace timestamps also fail closed.

## Validation

Ubuntu 24.04 / WSL2 / Python 3.14.7:

```text
.venv/bin/python -m pytest packages/route_a_imitation/tests/test_native_meta_rollout.py packages/route_a_imitation/tests/test_meta_native_gym.py tests/test_route_a_meta_native_engine.py tests/test_route_a_meta_native_training_e2e.py
5 passed
```

The runner regression injects a shifted barrier clock while preserving the barrier count and verifies rejection. The remaining cases exercise the existing Gym lifecycle, native order/fill receipts, native engine, and synthetic training end-to-end path.

No market payload was read and no profitability evidence was produced by these synthetic/integration checks.
