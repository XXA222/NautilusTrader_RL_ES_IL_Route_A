# Limit-entry coverage boundaries (2026-10-30)

Status: `ENGINEERING_FIX_VERIFIED; ACTIVE_META_PREREG_REMAINS_MARKET_ENTRY`.

## Correction

The research-only limit-touch replay previously allowed two inconsistent cases:

- A limit could be declared filled from a later OHLC bar even when the interval immediately after the candidate decision had no bar coverage.
- A bar beginning exactly at the decision timestamp was ignored, even though its interval contains only post-decision prices when the decision is made at the prior bar close.

The replay now treats a bar starting at the decision boundary as eligible, excludes a bar that straddles the decision from fill inference, and right-censors at the start of any uncovered interval. It validates the entire supplied bar sequence before early returns, so an early censor cannot hide malformed or overlapping later bars. A touched boundary bar is still not counted as a fill if OHLC cannot order touch and cancellation, and non-filled states remain distinct (`UNFILLED`, `CANCELLED`, `MISSED`, `RIGHT_CENSORED`).

## Validation

Ubuntu 24.04 / WSL2 / Python 3.14.7:

```text
.venv/bin/python -m pytest packages/route_a_imitation/tests/test_limit_entry.py packages/route_a_imitation/tests/test_x1_group_replay.py
9 passed

.venv/bin/python -m pytest packages/route_a_imitation/tests/test_opportunity_labels.py packages/route_a_imitation/tests/test_train_meta.py packages/route_a_imitation/tests/test_native_meta_rollout.py packages/route_a_imitation/tests/test_native_meta_funnel.py tests/test_route_a_meta_native_engine.py tests/test_route_a_meta_native_training_e2e.py
15 passed
```

## Scope

This is a causal-label correctness fix and synthetic/native-engine regression evidence. It does not establish profitability or new market-data results. The frozen `route-a-meta-x1-train-20261201-20261219-v1` preregistration still specifies `next_open_market`; its bytes and SHA are unchanged. Limit-touch replay is available to explicitly configured research callers, but is not represented as the active preregistered Meta execution assumption.
