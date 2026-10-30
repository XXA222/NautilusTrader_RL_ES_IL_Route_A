# Isolated exit replay entry identity — 2026-10-30

## Change

The X1/X2 first-trade pairing signature now includes the actual native entry
fill price, rounded to 1e-8, in addition to fill time, side, and entry notional
rounded to 1e-6. The report emits the price in `entry_identity` so a paired row
shows the economic entry basis used by the comparison.

This prevents two fills at the same time, direction, and notional but at
different prices from being treated as the same native entry. A mismatch stays
unpaired as `FIRST_ENTRY_FILL_IDENTITY_MISMATCH`; it cannot contribute to an
X2-minus-X1 exit delta.

## Verification

- Ubuntu 24.04 / Python 3.14.7: `tests/test_route_a_isolated_exit_replay.py` —
  **8 passed**.
- Ruff on the touched runner and test: **passed**.
- `git diff --check`: **passed**.

All checks used synthetic fixtures. No market payload was read and no native
economic replay was run. This is an identity-integrity fix, not profitability
evidence.
