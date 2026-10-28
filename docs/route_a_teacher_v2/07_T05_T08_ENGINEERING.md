# F07 — Second-batch teacher engineering

Status: `UNIT_ENGINEERING_PASS`; fresh-source/native economic evaluation not performed.

Implemented and registered T05–T08 using the existing `PointInTimeObservation`, `ExpertAdvice`, `BaseTeacher1m`, and ATR X2 exit interfaces. Higher timeframes remain feature-only; position sizing and native risk/execution are unchanged.

| ID | Entry | X1 exit |
|---|---|---|
| T05 MACD12/26/9 | DIF crosses DEA and histogram improves in the same direction | DIF/DEA reverse cross |
| T06 RSI14 | RSI crosses back above 30 after an oversold bar, or below 70 after an overbought bar | RSI returns to the neutral 50 threshold |
| T07 Bollinger20/2 | Close re-enters the band after the prior close was outside | Mean line or opposite outer-band risk exit |
| T08 EMA trend pullback | EMA9/21 trend, prior five-close pullback depth 0.3–2.0 ATR, and close breaks the recent three-close recovery window | Trend reversal or Donchian10 channel breach |

The pullback teacher clears its rolling price state on stale/missing/gapped bars and episode reset. All teacher state is derived from completed 1m snapshots. New implementations live in `teachers_1m/{macd,rsi,bollinger,pullback}.py`; recipes and factories are in `registry.py`, and exports in `teachers_1m/__init__.py`. The library JSON now records the T05 histogram requirement and T07 risk-exit identity.

Synthetic signal/lifecycle tests cover registry/config identity, long/short entries and exits, recovery rules, X2 construction, and the pullback gap reset. No S07 or new-source outcomes were used to select a teacher. Real-market native smoke is not run while F05 remains blocked.

Ubuntu command:

```bash
.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider \\
  packages/route_a_imitation/tests/test_teachers_1m_signals.py \\
  packages/route_a_imitation/tests/test_teachers_1m_lifecycle.py \\
  packages/route_a_imitation/tests/test_teachers_1m_second_batch.py \\
  packages/route_a_imitation/tests/test_indicators_1m.py \\
  packages/route_a_imitation/tests/test_teacher_1m_funnel.py \\
  packages/route_a_imitation/tests/test_portfolio_replay.py \\
  packages/route_a_imitation/tests/test_v03_l3_gym_adapter.py \\
  packages/route_a_imitation/tests/test_v03_native_gym_rollout.py
```

Observed: **105 passed**. `mypy packages`: **249 source files, no issues**. Ruff on teacher modules and relevant tests: passed.
