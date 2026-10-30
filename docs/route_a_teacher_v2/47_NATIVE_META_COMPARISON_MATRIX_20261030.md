# Native Meta comparison matrix fix — 2026-10-30

## Finding

The registered study runner executes the `ORIGINAL_8_X1` control, but `compare_preregistered_native_runs()` removed that configured control when constructing its expected matrix. A complete study therefore supplied a cell that the aggregator rejected as unexpected at the final comparison step.

## Change

The comparison matrix now preserves every preregistered control, including `ORIGINAL_8_X1`. It remains a separate aggregate control alongside each of the eight individual teachers, Meta, FLAT, and fixed-direction controls. A missing Meta cell remains `MISSING`/`INCOMPLETE`; the change does not fill or qualify results.

## Verification

- Ubuntu 24.04 / WSL2 / Python 3.14.7: `test_meta_evaluation.py` plus `test_route_a_meta_study_20261029.py`: **12 passed**.
- Ruff passed for the changed evaluation module and its tests; `git diff --check` passed.
- The live study was not run. Its registered source remains future dated and the last metadata-only preflight reported zero completed screen windows and no market payload read.

## Evidence boundary

This fixes a comparison-runner integration defect. It is not an economic result and does not qualify any teacher.
