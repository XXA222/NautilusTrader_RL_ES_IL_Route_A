from hashlib import sha256
import json

import pytest

from rles_route_a.integration.host_gym_projection import l3_order_facts_from_gym_info


def _receipt(before, after):
    encoded = json.dumps({"before": before, "after": after}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


def test_l3_adapter_binds_submission_fills_and_pending_statuses():
    before = ()
    after = ({"client_order_id": "RLES-new", "status": "FILLED", "side": "BUY",
              "quantity": "1", "filled_qty": "1"},)
    info = {
        "execution_plan_action": "SUBMIT",
        "execution_client_order_id": "RLES-new",
        "execution_related_order_ids": (),
        "native_pending_order_ids_before": (),
        "native_pending_order_ids_after": (),
        "native_policy_fills": ({
            "event_id": "fill-1", "client_order_id": "RLES-new", "trade_id": "fill-1",
            "ts_event": 10, "last_qty": "1", "last_px": "100",
            "commission": "0.1 USDT",
        },),
        "native_order_statuses_before": before,
        "native_order_statuses_after": after,
        "native_order_status_receipt": _receipt(before, after),
    }

    facts = l3_order_facts_from_gym_info(info)

    assert facts.submitted_order_ids == ("RLES-new",)
    assert facts.filled_order_ids == ("RLES-new",)
    assert facts.native_order_statuses_after == after
    assert facts.native_evidence_receipt == facts.native_order_status_receipt


def test_l3_adapter_rejects_missing_or_tampered_native_order_receipt():
    info = {
        "execution_plan_action": "NOOP", "execution_client_order_id": None,
        "execution_related_order_ids": (), "native_pending_order_ids_before": (),
        "native_pending_order_ids_after": (), "native_policy_fills": (),
        "native_order_statuses_before": (), "native_order_statuses_after": (),
        "native_order_status_receipt": "0" * 64,
    }
    with pytest.raises(ValueError, match="receipt mismatch"):
        l3_order_facts_from_gym_info(info)


def test_repeated_native_order_status_rows_are_shared_read_only_facts():
    status = {"client_order_id": "RLES-open", "status": "ACCEPTED", "side": "BUY",
              "quantity": "1", "filled_qty": "0"}
    before = (status,)
    after = (status,)
    info = {
        "execution_plan_action": "NOOP", "execution_client_order_id": None,
        "execution_related_order_ids": (),
        "native_pending_order_ids_before": ("RLES-open",),
        "native_pending_order_ids_after": ("RLES-open",),
        "native_policy_fills": (),
        "native_order_statuses_before": before,
        "native_order_statuses_after": after,
        "native_order_status_receipt": _receipt(before, after),
    }

    first = l3_order_facts_from_gym_info(info)
    second = l3_order_facts_from_gym_info(info)

    assert first.native_order_statuses_after is second.native_order_statuses_after
    assert first.native_order_statuses_after[0] is second.native_order_statuses_after[0]
    with pytest.raises(TypeError):
        first.native_order_statuses_after[0]["status"] = "FILLED"  # type: ignore[index]
