from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from run_route_a_1m_teacher_native import (
    ROOT_DEFAULT_PREREG,
    _fill_costs,
    _window_bounds,
    verify_preregistration,
)
from rles_nautilus_adapter.session import _native_orders_for_account


def test_s06_preregistration_binds_source_and_recipe_without_market_decode():
    doc, digest = verify_preregistration(ROOT_DEFAULT_PREREG)
    assert digest == ROOT_DEFAULT_PREREG.with_suffix(".json.sha256").read_text().strip()
    assert doc["source"]["schema_version"] == "0.4.0"
    assert doc["source"]["physical_source_months"] == [
        "2023-12", "2024-01", "2024-02", "2024-03", "2024-04",
    ]
    assert doc["qualification"] == "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2"


def test_s06_preregistration_rejects_mutated_bytes(tmp_path: Path):
    copied = tmp_path / ROOT_DEFAULT_PREREG.name
    copied.write_bytes(ROOT_DEFAULT_PREREG.read_bytes() + b" ")
    copied.with_suffix(copied.suffix + ".sha256").write_text(
        ROOT_DEFAULT_PREREG.with_suffix(".json.sha256").read_text(),
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_preregistration(copied)


def test_s07_native_order_lookup_keeps_account_scope_and_recovers_exact_policy_id():
    from nautilus_trader.model import ClientOrderId, InstrumentId

    instrument = InstrumentId.from_str("ETHUSDT-PERP.SIM")
    account_id = "SIM-001"
    policy_id = "RLES-" + "a" * 64

    class NativeOrder:
        def __init__(self, client_order_id: str, instrument_id: object):
            self.client_order_id = ClientOrderId(client_order_id)
            self.instrument_id = instrument_id

    account_order = NativeOrder("ACCOUNT-ORDER", instrument)
    policy_order = NativeOrder(policy_id, instrument)
    unrelated_order = NativeOrder("UNRELATED-ORDER", instrument)

    class Cache:
        def orders(self, **scope):
            assert scope == {"instrument_id": instrument, "account_id": account_id}
            return (account_order,)

        def order(self, client_order_id):
            return {
                str(policy_order.client_order_id): policy_order,
                str(unrelated_order.client_order_id): unrelated_order,
            }.get(str(client_order_id))

    rows = _native_orders_for_account(
        Cache(), scope={"instrument_id": instrument, "account_id": account_id},
        instrument_id=instrument, policy_order_ids=frozenset({policy_id}),
    )
    assert rows == (account_order, policy_order)


def test_s06_slippage_and_fee_come_only_from_native_fill_and_reference():
    fees, slip, notional, fills = _fill_costs(
        ({
            "event_id": "f-1", "client_order_id": "o-1", "order_side": "BUY",
            "last_qty": "2", "last_px": "101", "commission": "0.10 USDT",
        },),
        {"o-1": Decimal("100")},
    )
    assert (fees, slip, notional) == (
        Decimal("0.10"),
        Decimal("2"),
        Decimal("202"),
    )
    assert len(fills) == 1


def test_s06_slippage_rejects_price_improvement():
    with pytest.raises(ValueError, match="price improvement"):
        _fill_costs(
            ({
                "event_id": "f-2", "client_order_id": "o-2", "order_side": "BUY",
                "last_qty": "1", "last_px": "99", "commission": "0 USDT",
            },),
            {"o-2": Decimal("100")},
        )


def test_s07_window_resolver_uses_only_registered_train_inner_blocks():
    doc, _digest = verify_preregistration(ROOT_DEFAULT_PREREG)
    assert _window_bounds(doc, "inner-screen-1") == (
        1706745600000000000, 1709251200000000000,
        "inner-screen-20240201_20240301",
    )
    assert _window_bounds(doc, "inner-screen-2") == (
        1709251200000000000, 1711929600000000000,
        "inner-screen-20240301_20240401",
    )
    assert _window_bounds(doc, "inner-screen-3") == (
        1711929600000000000, 1713139260000000000,
        "inner-screen-20240401_20240415",
    )
    with pytest.raises(ValueError, match="not a registered S07 window"):
        _window_bounds(doc, "2024-05-test-holdout")
