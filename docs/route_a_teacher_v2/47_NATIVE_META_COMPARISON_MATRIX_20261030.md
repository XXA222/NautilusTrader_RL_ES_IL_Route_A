# Native Meta comparison matrix fix — 2026-10-30

## Finding

The frozen preregistration lists `ORIGINAL_8_X1`, but the study runner dispatches only the eight individual teachers, Meta, FLAT, fixed long, and fixed short. The legacy control factory also has no `ORIGINAL_8_X1` implementation. The aggregator previously removed the control from its expected matrix without reporting the planned-versus-active scope difference, so its 48-cell active matrix could appear to be the full preregistered study.

## Change

The comparison report now distinguishes the full preregistered plan (52 cells) from the executable active matrix (48 cells). It records `ORIGINAL_8_X1` as a preregistered but unexecuted control, keeps its planned rows as `MISSING`, and reports both `status` and `active_status`. It does not invent an aggregate teacher or treat active-matrix completion as full preregistration completion. Missing Meta cells also remain explicit.

## Verification

- Ubuntu 24.04 / WSL2 / Python 3.14.7: comparison and runner regression tests verify 52 planned cells, 48 active cells, explicit aggregate-control omission, and separate active/full completion states.
- Ruff passed for the changed evaluation module and its tests; `git diff --check` passed.
- The live study was not run. Its registered source remains future dated and the last metadata-only preflight reported zero completed screen windows and no market payload read.

## Evidence boundary

This fixes a comparison-runner integration defect. It is not an economic result and does not qualify any teacher.
