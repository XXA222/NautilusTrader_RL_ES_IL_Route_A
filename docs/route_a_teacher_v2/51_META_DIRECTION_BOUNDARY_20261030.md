# Meta direction-to-execution boundary — 2026-10-30

## Finding

`FrozenEntryMetaPolicy.score` maps candidate side to LONG/SHORT, but `MetaGatedTeacher` previously treated either directional score as eligible without independently checking that it matched the candidate. The normal frozen policy does not produce this mismatch, but a malformed adapter or corrupted score could still pass the decision boundary.

## Change

The native adapter now rejects a directional score that disagrees with the candidate side as `MODEL_DIRECTION_MISMATCH`. Such a candidate cannot be selected, and the one-position host receives no opposing entry action from that score.

## Verification

- Added a short candidate with a deliberately mismatched LONG score; Meta stays FLAT and records `REJECTED_MODEL_DIRECTION_MISMATCH`.
- Meta gated teacher, native funnel, and native integration regressions: **24 passed**. Ruff and `git diff --check` passed.

## Evidence boundary

This verifies the candidate/score/intent boundary under a synthetic mismatch. It is not a real market result or teacher qualification.
