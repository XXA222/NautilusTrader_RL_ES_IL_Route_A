# Backup validation — 2026-10-24

Canonical source root:

`D:\Program Files\NautilusTrader_RL_ES_IL_Route_A`

This report records local validation of the exact working-tree snapshot prepared
for the independent GitHub repository
`XXA222/NautilusTrader_RL_ES_IL_Route_A`.

## Results

### Root test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
```

Result: **738 passed, 5 skipped, 56 warnings** in 397.89 seconds.

The five skips were CUDA-only tests on a host without an available CUDA learner.
Warnings were dependency/runtime warnings from Torch, EvoTorch, and NumPy; no
test failed.

### Route A package test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider packages/route_a_imitation/tests
```

Result: **205 passed, 2 skipped** (207 tests collected).

### Static checks

```powershell
.\.venv\Scripts\python.exe -m ruff check packages qa tests
```

Result: **All checks passed**.

## Scope and limits

These results prove the listed local code and contract checks for this snapshot.
They do not claim profitable trading, native economic certification, or CUDA
coverage. GitHub Actions remains the authoritative clean-runner confirmation for
the published commit.
