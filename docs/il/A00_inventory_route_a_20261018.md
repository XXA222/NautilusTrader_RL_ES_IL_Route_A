# A-00 data and rule inventory

Generated UTC: 2026-10-18T14:07:19.631336+00:00

- Git HEAD: `c24d1bc4a47584f91a4c4db93bf77117d4e8341c`
- Worktree changes: 153
- Historical economics eligible: **False**
- Gate reason: The existing audit marks historical ETHUSDT rules, account fees and margin history scenario-only or unverified.

## Local data directories

- `binance_ethusdt_20231230_20240415`: 332 files, 26139479 bytes
- `binance_ethusdt_20240101`: 9 files, 243430 bytes
- `binance_ethusdt_20240101_quotes_1s`: 5 files, 95146468 bytes
- `binance_ethusdt_20240415_20240731`: 659 files, 25984832 bytes
- `binance_ethusdt_20240731_20241116`: 667 files, 26238016 bytes
- `binance_ethusdt_20240731_20241201`: 56 files, 1253882 bytes
- `binance_ethusdt_20240812_20241201`: 685 files, 26967765 bytes
- `binance_ethusdt_wfo_2024_apr_aug_twofold`: 3 files, 13842928 bytes
- `raw`: 12 files, 125272 bytes

## Rule-file candidates

- `tests\fixtures\real_ethusdt_20240101\SCENARIO_RULE_ASSUMPTIONS.json`
- `tests\fixtures\real_ethusdt_20240101\scenario_rules.json`

## Existing audit evidence

- docs/HISTORICAL_RULES_BLOCKER_20261016.md: present
- docs/AUDIT_REPORT_20261016.md: present
- docs/audit_evidence/20261016/persistent_artifacts_v2.json: present

## Registered research intervals

- Registry available: True
- Access records: 214
- `development:recorded`: 209
- `holdout:reserved`: 2
- `test:reserved`: 3
- Exact UTC-nanosecond intervals, purposes and states are in `A00_inventory_route_a_20261018.json`.
- Scope limit: The registry is append-only local governance, not a complete record of every historical data read. The 2026-10-16 inventory imported 67 known records from 10 WFO databases and 22 reports; older undocumented reads cannot be reconstructed from files alone.

## Archived market/rule evidence

- Manifest available: True
- Hash verified files: 14
- Missing or mismatched files: 0

## Decision

Proceed only with synthetic mechanism validation and clearly labeled Train-inner engineering diagnostics. Do not claim historical economic performance or select models on Test/Holdout. Current exchange rules are not substituted for historical effective rules.
