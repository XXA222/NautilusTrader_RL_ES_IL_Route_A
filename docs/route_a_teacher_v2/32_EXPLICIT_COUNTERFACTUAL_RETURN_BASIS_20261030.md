# Explicit counterfactual return basis — 2026-10-30

## Change

The counterfactual label path now binds its declared `return_basis_id` to the
formula used for gross directional returns:

- `log` → `LOG_REFERENCE_ENTRY_NOTIONAL_BPS`
- `simple` → `LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS`

`OpportunityOutcome` rejects ambiguous basis identifiers, and grouped outcomes
do the same. This prevents labels computed with one formula from being grouped
or fitted under an unrelated caller-supplied string. Native actual-fill returns
remain separately identified as `LINEAR_ACTUAL_FILL_ENTRY_NOTIONAL_BPS` by the
cash-flow reconciliation adapter.

Historical test fixtures using informal basis strings were updated to the
explicit identifiers. A mismatch regression check verifies rejection before
candidate outcomes can enter fitting.

## Verification

The related label, causal replay, model, diagnostics, gate-funnel, portfolio
replay, native case reconciliation, entry-policy, policy-I/O, and native
training E2E set completed with **85 passed** on Ubuntu 24.04 / Python 3.14.7.

The native-case reconciliation test uses a retained actual native trade case;
its fills, fees, funding, realized cash flow, and reference-price discrepancy
continue to reconcile as distinct evidence and return bases.

## Evidence boundary

This change fixes economic-unit identity and prevents mislabeled training rows.
It does not establish a profitable model or independent Train performance.
