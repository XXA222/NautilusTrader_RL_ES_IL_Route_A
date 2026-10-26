# S05 — 1m decision funnel, native facts, and trade cases

**Status:** engineering contract and regression acceptance complete. No market-data
rollout or economic claim was produced in S05; source authorization and dry source
validation remain S06 prerequisites.

## Implemented contract

- `evaluation/teacher_funnel_1m.py` holds separate pre-decision, planner, and
  post-step records. The optional `TeacherFunnelResearchHook1m` in the existing
  `NativeGymTeacherProvider` is a sidecar only: it captures raw signal and
  eligibility before `env.step`, then receives L3/L6 facts afterward. Nothing
  preloads a future fill.
- The sidecar commits only after the provider's existing same-run M2.2 decision
  trace and retained native order snapshots reconcile. Submitted and filled order
  identities must match the audited barriers; case IDs from the hook must match
  the decision lifecycle IDs. Any mismatch raises
  `INCONSISTENT_NATIVE_ACCOUNTING` before artifact commit.
- Funnels are grouped by teacher, X1/X2 exit, Base/Stress scenario, Train block,
  and LONG/SHORT/FLAT. A fill reported on a later barrier remains associated with
  its original submitted order ID. Veto categories are counted independently;
  a flat entry is eligible only when the 1m bar is ready and neither risk nor
  order-state veto is present.
- Planner status and reason are required. An unfilled submitted order must still
  be pending at the barrier or have a separately captured native reject/cancel
  reason. Otherwise the provider fails closed; a generic `SUBMIT` reason is not
  substituted for an unexplained zero-fill result.
- Fee and funding amounts are checked against that barrier's native commission
  events and `native_funding` L6 records. Currency mismatches are rejected rather
  than summed. Account equity must match the native barrier. Slippage, MFE/MAE,
  and position age remain explicit fields and must be populated by the native
  research adapter; the helper does not synthesize them from future prices.
- `evaluation/teacher_case_report_1m.py` preserves ordinary completion, open
  right-censoring, and terminal censoring separately. A terminal force-flat is
  never an ordinary X1/X2 exit. Completed-case profit factor excludes censored
  cases; native equity and cost attribution include the recorded case rows once.
- The writer emits `teacher-funnel.json`, pinned-schema `decision-rows.csv`,
  `native-trades.jsonl`, and `case-summary.json`. The CSV is the controlled
  columnar equivalent requested for environments without a pinned Parquet
  runtime. It retains direction, planner/fill outcomes, vetoes, equity, costs,
  funding, MFE/MAE, duration, censor status, and economic classification. The
  destination must not already exist. `SCENARIO_ONLY` remains visible in the
  artifacts and is never promoted to observed economics.

## Exact-clock correction discovered during S05 integration

An epoch Unix-nanosecond close timestamp cannot safely travel as a float market
feature. `PointInTimeObservation` now carries `exact_market_clocks_ns`, the
snapshot projector writes the 1m close clock there, and X1/X2 read it through
`market_clock_ns`. Legacy float clocks are accepted only while they remain
within the exact-integer range (`2**53`). Native 1m funnel validation requires
the exact `latest_1m_close_ts_ns`; if the S06 host context cannot provide that
integer metadata, S06 must block rather than round it.

## Ubuntu 24.04 WSL evidence

```text
.venv-ubuntu/bin/python -m pytest -rA -p no:cacheprovider \
  packages/route_a_imitation/tests/test_indicators_1m.py \
  packages/route_a_imitation/tests/test_teachers_1m_signals.py \
  packages/route_a_imitation/tests/test_teachers_1m_lifecycle.py \
  packages/route_a_imitation/tests/test_teacher_1m_funnel.py \
  packages/route_a_imitation/tests/test_v03_native_gym_rollout.py \
  packages/route_a_imitation/tests/test_v03_host_projection.py \
  packages/route_a_imitation/tests/test_v03_host_context.py \
  packages/route_a_imitation/tests/test_v03_transition_builder.py \
  packages/route_a_imitation/tests/test_contracts_teachers.py
exit 0 — 116 passed in 18.15s

.venv-ubuntu/bin/python -m mypy packages
exit 0 — Success: no issues found in 244 source files

.venv-ubuntu/bin/python -m ruff check packages qa tests
exit 0 — All checks passed!
```

The tests include synthetic-only writer round trips, exact epoch-ns X1/X2
clocks, delayed-fill attribution, side/block separation, native fee/funding
reconciliation, unexplained zero-fill rejection, terminal/open censoring, and
existing host projection/order lifecycle regressions. They do not read Test or
Holdout funding archives and do not qualify economic performance.

## Stop/acceptance review

- Filled/submitted ID mismatch or unmatched native case: fail closed before
  output commit — covered by negative tests.
- Zero-fill decision with no veto, pending native order, or explicit native
  rejection/cancellation reason: fail closed — covered by negative test.
- Duplicate or fictitious case, mixed teacher/exit/scenario/block/direction,
  terminal flatten counted as ordinary completion, or incomparable currencies:
  rejected by DTO/group/report validation.
- Real `teacher-funnel.json`, decision CSV, native trades, and case summary are
  intentionally deferred until S06 validates a newly registered legal Train
  source. No synthetic artifact is presented as a market result.
- **Conclusion:** S05 engineering acceptance passes; no native economic
  qualification is claimed. Proceed to S06 source dry-validation only.
