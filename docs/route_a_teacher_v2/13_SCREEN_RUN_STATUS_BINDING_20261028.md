# 13 — Native screen status binding (2026-10-28)

## Change

The native teacher-screen runner now rejects both resumed and newly aggregated `native-run.json` artifacts unless they carry the expected `economic_status: SCENARIO_ONLY` and `qualification: D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2`. The aggregate report also records the same qualification string at its top level. This closes a receipt-integrity gap; it does not change candidate recipes, windows, costs, seeds, controls, pass criteria, or when student training may start.

## Evidence and verification

- Added four negative cases covering mismatched economic status and qualification on resume and summary aggregation.
- Targeted Ubuntu runner tests: 17 passed.
- Ruff: passed.
- Targeted MyPy on the runner and test: passed with imported QA/runtime modules skipped.
- Read-only source preflight: `PREFLIGHT_PASS_STAGED_SOURCES_PARTIAL_METADATA_NOT_READ`; registry remains `recorded`, one of five staged imports exists, and `market_payload_read` is false.
- Native scoring was not run; the first scored window is not fully imported.

The runner hash is bound in the existing source erratum as `07a6aa9d4eabddf71bf43f48921310e791d97c06767e2a488887193bb21527ce`; the resulting source-erratum SHA-256 is `bb72b06db2ab02242c1c859d561f9cd0486ebad966b6e783dd94ca219f4b07b5`. Parent preregistration SHA-256 remains `e7f795fce9f8539e58085f6bd85449f96edb53324cebcbb6380dec47c8d38aca`.

## Next native step

After the complete registered window-0 source is present, rerun `qa/validate_route_a_teacher_student_prereg_20261026.py`; only a passing preflight authorizes `.venv-ubuntu/bin/python qa/run_route_a_teacher_student_train_screen_20261026.py --window-index 0`. Do not score partial bars.
