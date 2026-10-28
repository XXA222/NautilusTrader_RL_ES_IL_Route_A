# F01 — EMA/X1 reversal lifecycle

Status: `ENGINEERING_PASS`; the new reversal mode has not been evaluated on fresh Train economics.

## Change

`T01_EMA_9_21` now supports an opt-in `REVERSE_AFTER_FLAT_V1` lifecycle while preserving the frozen default `EXIT_ONLY_V1`. The candidate is created only from an X1 EMA reverse-cross exit. It records direction, event bar, episode/source identity, and native close-ack state. The candidate remains armed while the close or pending orders remain unresolved; after verified flat/no-pending state, EMA direction is checked on the next valid 1m close. There is no fixed bar-count expiry.

The provider feeds each native action and its L3/M2.2 order facts back to the teacher. A reversal is eligible only after the matching close request has a full FILLED status, fill event, evidence/status receipts, no remaining pending order, and observed flat exposure. The teacher waits until a later valid 1m close, rechecks EMA direction, then emits a normal entry advice. Partial fills wait. Rejection, risk override, changed exposure, a missing/stale/gapped bar, or reset clears the candidate.

Implementation: `teachers_1m/{base,contracts,ema,registry}.py` and `integration/gym_rollout.py`; lifecycle coverage is in `tests/test_teachers_1m_lifecycle.py`.

## Verification

Ubuntu 24.04 / Python 3.14.7:

```bash
.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider \\
  packages/route_a_imitation/tests/test_teachers_1m_lifecycle.py \\
  packages/route_a_imitation/tests/test_v03_l3_gym_adapter.py \\
  packages/route_a_imitation/tests/test_v03_native_gym_rollout.py
```

Observed result: **31 passed**. `mypy packages`: **244 source files, no issues**. Ruff on changed Python files and `git diff --check`: passed.

No new market data or economic result was generated. The frozen S07 report was not rewritten.
