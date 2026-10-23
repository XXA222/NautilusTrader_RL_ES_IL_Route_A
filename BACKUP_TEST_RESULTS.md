# Route A backup test results

Run date: 2026-10-22 (Asia/Shanghai)

Source snapshot: current working tree of `D:\Program Files\NautilusTrader_RL_ES_IL_Route_A`, captured without `.git` history and with Codex remote attachments excluded.

Command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='D:\Program Files\NautilusTrader_RL_ES_IL_Route_A'
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

Result: **737 passed, 5 skipped, 1 failed, 56 warnings**; 743 tests collected; 570.99 seconds.

The five skips were CUDA-dependent checks because a CUDA learner was unavailable. The single failure was `tests/test_historical_factory.py::test_market_audit_reports_only_registered_train_coverage_without_granting_missing_rules`. It is reproducible in isolation. `audit_market()` rejects the test fixture because its environment is missing the now-required `rules_path` (`packages/orchestrator/src/rles_orchestrator/application.py:451`). This report records the current result as-is; it does not imply a green test suite.

An older checked-in test record from 2026-10-16 reported 442 passed and 4 skipped; it does not describe this snapshot.
