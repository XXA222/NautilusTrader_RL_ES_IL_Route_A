# A-01 causal teacher set

`rles_il_core.experts` exposes a common causal `ExpertPolicy` interface with independent, deterministic teachers. Every teacher returns a bounded target exposure; only the native Gym environment can turn it into an order or fill.

| Teacher | ID | Inputs and frozen behavior |
|---|---|---|
| E1 trend/hold | `trend_v1` | Uses visible 15m/60m returns and 60m trend efficiency; enters at ±0.5, exits through flat before reversing, with the configured cooldown. Requires `causal_mtf_v2` or later. |
| E2 mean reversion | `reversion_v1` | Uses a 60-point rolling window of completed `close_1m` values; z entry 1.5, exit 0.3, target ±0.4, a trend continuation veto, and 240-minute timeout. Any missing minute clears the window and returns flat. Requires the identity-distinct `causal_mtf_v4` profile. |
| E3 cost/observe | `cost_guard_v1` | Wraps E1 with a caller-frozen `volatility_60m` high-volatility veto and target deadband. The threshold is an engineering input; it must be selected from Train-inner only. |
| Flat | `flat_v1` | Always returns zero; mechanism and cost baseline. |
| Fixed long/short | `fixed_long_v1`, `fixed_short_v1` | Uses a frozen nonzero exposure and entry decision timestamp, then holds through the finite episode and native settlement. It is a risk exposure baseline, not a profitable expert label. |

Multi-teacher data must be collected as separate complete native account episodes from the same market windows and merged with `expert_id` retained. Do not choose the best teacher at each timestamp or treat duplicate teacher windows as independent market minutes. `causal_mtf_v4` adds only the latest completed one-minute close; its feature/schema/transition hashes differ from earlier profiles, so existing actors cannot load across that change.

The rules and default thresholds are reproducible engineering candidates, not validated market edges. The A-00 rule/fee/margin evidence gate remains in force.
