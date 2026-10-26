# Ubuntu QA record

Environment: WSL Ubuntu-24.04, project `.venv-ubuntu`, Python 3.14.7. Canonical root and `.canonical-project-root` were verified before the runs.

## Current continuation validation

| Command / scope | Result |
|---|---|
| `.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider packages/route_a_imitation/tests/test_opportunity_diagnostics.py packages/route_a_imitation/tests/test_gate_funnel.py packages/route_a_imitation/tests/test_portfolio_replay.py` | Exit 0; 30 tests (30 progress markers); includes paired Base-prediction/Stress-outcome audit and censor coverage. |
| `.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider tests/test_binance_archive_range.py` | 24 passed. |
| `.venv-ubuntu/bin/ruff check packages qa tests` | All checks passed. |
| `.venv-ubuntu/bin/mypy packages` | Success: no issues in 225 source files. |
| `.venv-ubuntu/bin/python -m py_compile qa/run_route_a_teacher_e1_e2_g1_20261025.py` | Exit 0. |
| `git diff --check` | Pass at final review. |

Independent pytest groups and static checks were launched concurrently. An initial archive command also named a nonexistent `tests/test_binance_archive.py`; it ran no tests and was corrected to the actual `test_binance_archive_range.py` file above.

The new paired-outcome test first failed because its expected tuple used input order while the audit intentionally emits event-ID order. The assertion was corrected to the deterministic order; the final 30-test group exited 0.

## Not established by these checks

- The historical runner’s integrated diagnostic-output path was syntax/lint checked, but not run against the contaminated source.
- No full all-project pytest run was performed in this continuation. Prior long-suite attempts recorded elsewhere as incomplete remain `NOT_COMPLETED`, not PASS.
- These synthetic/engineering checks do not prove G1/G2, native reconciliation, account-specific economic authority, or teacher qualification.
