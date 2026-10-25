# A-08: L6 WFO, ES selection, and final Holdout gate

The repository already has an L6 `run_wfo()` application over the frozen L2 fold manifest and existing R1/E1/E2 stage machinery. The A-08 path separates candidate training, fold Validation, selected-candidate retraining, fold Test, and the one-shot final Holdout. It reserves and receipts formal Validation/Test/Holdout reads through the project research-access registry and audit ledger. A cost-stress Validation gate can reject candidates before any Test query. If no candidate passes, the report records rejection and leaves Holdout locked.

The WFO application reopens a completed protocol only after validating its saved report, audit chain, Holdout receipt, and per-fold Test receipts. Interrupted Test/report work resumes without re-reading completed Test. Once the Holdout reservation has opened and evaluation may have begun without a complete receipt, resume refuses to replay the market interval. These properties are covered by the synthetic WFO suite.

## Current verification

- `tests/test_research_wfo.py`: **35 passed** on 2026-10-18.
- Coverage includes frozen fold/config validation, missing liquidation-fact rejection, stress-Validation rejection before Test, no-feasible-candidate behavior with Holdout locked, persisted Test resume, and one-shot Holdout open/close interruption handling.
- No real Test/Holdout market window was read for this check.

## Gate

The local rule candidates remain scenario-only and do not establish full-period ETHUSDT exchange filters, account fee tiers, historical margin/liquidation mapping, or a verified fill model. Therefore A-08 implementation mechanics are available for synthetic and explicitly scenario-labeled engineering, but no historical strategy selection, return claim, or fresh Holdout evaluation is authorized by the evidence currently recorded in [A-00](A00_inventory.md) and [the historical-rules blocker](../HISTORICAL_RULES_BLOCKER_20261016.md). A final economic WFO result requires those source facts plus a validated fresh-data access manifest; changing dates, snapshots, or output paths does not make consumed intervals independent.
