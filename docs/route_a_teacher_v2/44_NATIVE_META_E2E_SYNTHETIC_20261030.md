# Native Meta end-to-end synthetic verification — 2026-10-30

Status: `NATIVE_ENGINE_INTEGRATION_SYNTHETICALLY_VERIFIED; ECONOMIC_QUALIFICATION_NOT_ESTABLISHED`.

## Scope exercised

The tests ran the Meta teacher through the existing Nautilus/Gym account, risk, order, fill, exit, and equity path. The full synthetic study additionally collected the eight original 1m teacher signals at each native barrier, retained candidate decisions, produced frozen-X1 labels, trained and serialized Logistic + Ridge policies on temporally separated Train-fit and calibration segments, then ran a two-window Base/Stress comparison across Meta, all eight original teachers, FLAT, fixed long, and fixed short controls.

The synthetic native comparison exercises the 48-cell active matrix (Meta, eight individual teachers, FLAT, fixed long and fixed short) and retains `NOT_ESTABLISHED` qualification. The frozen preregistration also lists an `ORIGINAL_8_X1` aggregate control that the current runner does not implement; the separate planned-matrix report must therefore show 52 planned cells versus 48 active cells and mark the full plan incomplete. Tests also exercise the registered importer and study coordinator with generated exchange-format bars and funding data; these are synthetic fixtures, not downloaded or independent Train evidence.

## Verification

Ubuntu 24.04 / WSL2 / Python 3.14.7:

```text
pytest tests/test_route_a_meta_native_engine.py \
       tests/test_route_a_meta_native_training_e2e.py \
       packages/route_a_imitation/tests/test_meta_native_gym.py -q
4 passed in 42.48s
```

The native engine test confirms candidate-to-native case IDs and teacher attribution, all eight signal streams, candidate outcome/replay reconciliation, and an intentionally incomplete short-window matrix. The synthetic training study confirms all 48 active cells, serialized fit/calibration artifacts, and `NOT_ESTABLISHED` qualification; it does not establish completion of the 52-cell preregistered plan.

## Evidence boundary

This is integration and accounting evidence only. Synthetic prices cannot establish trading skill. The registered independent protocol remains future-dated at `2026-12-01` through `2026-12-19 UTC`; the latest read-only preflight reported `REGISTERED_FUTURE_SOURCE`, zero completed screen windows, and `market_payload_read: false`. Real Train fitting, calibration, and native Base/Stress economics must be evaluated after the authorized source is complete and the registered preflight permits it.
