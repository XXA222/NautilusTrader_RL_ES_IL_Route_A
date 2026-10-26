# Route A teacher remediation change log

This record is for the frozen Ubuntu remediation baseline only. It does not replace preregistrations or qualify a teacher.

## Implemented

- Corrected strict package typing without weakening mypy or converting censored values to zero.
- Reworked the historical screen portfolio path so candidate admission is causal; future complete outcomes are used only after admission for accounting. Added explicit block censoring, terminal mark-only accounting, fees, slippage, funding, and empty-block records.
- Added four-gate funnel counts, ordered attrition, all 16 gate combinations, block/family/side summaries, and explicit rejection reasons.
- Added prediction-only score/support diagnostics, a separate post-prediction screen outcome audit, and deterministic F0–F5 diagnostic-only candidate sets.
- Connected those diagnostics to the historical runner. The runner now serializes per-candidate predictions, fit-extrema support evidence, Base/Stress screen outcome audits, paired F0–F5 outcome distributions, and account replay metrics. Every ablation is marked `DIAGNOSTIC_REUSE_NOT_OOS`.
- Tightened the Binance funding REST fallback to request the preregistered inclusive date span and reject any response row outside the exact requested millisecond bounds. Existing physical-archive Test/Holdout guards remain fail-closed.

## Still incomplete

- The previous E1/E2 report stores 138 candidates and first-rejection counts, but not per-candidate scores/four-gate bitmasks. Its contaminated monthly-funding source cannot be reopened to reconstruct those fields.
- The new runner diagnostic payload has not been generated from real inputs. The October raw source attempt has no corresponding development access record and its funding request timed out; it is not treated as eligible fresh Train.
- The plan's separate D-06 `run_route_a_teacher_research_v2_ubuntu.py` fresh-authorization runner is still outstanding; integration into the legacy E1/E2 script is not a substitute for that deliverable.
- No new legal Train experiment, M2.2/native reconciliation, qualified Alpha Teacher, or formal student promotion has been completed.

See [UBUNTU_QA.md](UBUNTU_QA.md), [TEACHER_DECISION.md](TEACHER_DECISION.md), and [EXECUTION_STATUS.md](EXECUTION_STATUS.md).
