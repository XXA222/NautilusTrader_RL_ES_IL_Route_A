# Limit-entry coverage boundaries (2026-10-30)

Status: `ENGINEERING_FIX_VERIFIED; ACTIVE_META_PREREG_REMAINS_MARKET_ENTRY`.

## Correction

The research-only limit-touch replay previously allowed two inconsistent cases:

- A limit could be declared filled from a later OHLC bar even when the interval immediately after the candidate decision had no bar coverage.
- A bar beginning exactly at the decision timestamp was ignored, even though its interval contains only post-decision prices when the decision is made at the prior bar close.

The replay now treats a bar starting at the decision boundary as eligible, excludes a bar that straddles the decision from fill inference, and right-censors at the start of any uncovered interval. It validates the entire supplied bar sequence before early returns, so an early censor cannot hide malformed or overlapping later bars. A touched boundary bar is still not counted as a fill if OHLC cannot order touch and cancellation, and non-filled states remain distinct (`UNFILLED`, `CANCELLED`, `MISSED`, `RIGHT_CENSORED`).

If the final eligible bar touches the limit but the assumed fill timestamp (bar end) equals the registered right-open block boundary, the outcome is now `RIGHT_CENSORED`, not `MISSED`: the touch is known, but the configured fill timestamp is outside the permitted block.

## Validation

Ubuntu 24.04 / WSL2 / Python 3.14.7:

```text
.venv/bin/python -m pytest packages/route_a_imitation/tests/test_limit_entry.py packages/route_a_imitation/tests/test_x1_group_replay.py packages/route_a_imitation/tests/test_opportunity_labels.py packages/route_a_imitation/tests/test_group_outcome_economics.py packages/route_a_imitation/tests/test_train_meta.py packages/route_a_imitation/tests/test_native_meta_rollout.py packages/route_a_imitation/tests/test_native_meta_funnel.py tests/test_route_a_meta_native_engine.py tests/test_route_a_meta_native_training_e2e.py
31 passed

ruff check packages/route_a_imitation/src/rles_route_a/teacher_research/limit_entry.py packages/route_a_imitation/tests/test_limit_entry.py packages/route_a_imitation/tests/test_x1_group_replay.py
All checks passed
```

The retained source row for `native-case-047b9e12-31d2-4dfc-931f-e32b441a5c14` was also re-read from `.artifacts/route-a-exit-economics-task3/per_trade_exit_metrics.jsonl` and passed `reconcile_native_trade`: gross `-0.174400000 USDT`, fees `0.249083520 USDT`, funding `0`, net `-0.423483520 USDT` (`-16.995716953` entry-notional bps). The result exactly matches the retained reconciliation artifact. This is an actual archived native case from consumed Train diagnostics (`SCENARIO_ONLY`), not independent profitability evidence.

## Scope

This is a causal-label correctness fix and synthetic/native-engine regression evidence. It does not establish profitability or new market-data results. The frozen `route-a-meta-x1-train-20261201-20261219-v1` preregistration still specifies `next_open_market`; its bytes and SHA are unchanged. Limit-touch replay is available to explicitly configured research callers, but is not represented as the active preregistered Meta execution assumption.
