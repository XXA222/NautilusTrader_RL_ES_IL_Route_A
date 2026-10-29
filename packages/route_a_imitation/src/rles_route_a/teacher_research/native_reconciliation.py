"""Cash-flow reconciliation for retained, completed native trade receipts."""
from __future__ import annotations

from math import isfinite
from typing import Mapping


def reconcile_native_trade(row: Mapping[str, object], *, tolerance: float = 1e-7) -> dict[str, float | str]:
    """Fill prices already include slippage; compare the actual native cash flow."""
    if row.get("status") != "ORDINARY_COMPLETE" or not row.get("trade_case_id"):
        raise ValueError("an identified ordinary completed native case is required")
    def number(name: str) -> float:
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
            raise ValueError(f"invalid native {name}")
        return float(value)
    side = number("side")
    entry = number("entry_fill_price")
    exit_price = number("exit_fill_price")
    notional = number("entry_notional")
    fees = number("fees")
    funding = number("funding")
    realized = number("realized_pnl")
    reported_gross = number("gross_cash_pnl_before_fees_and_funding")
    if side not in (-1, 1) or min(entry, exit_price, notional) <= 0 or fees < 0:
        raise ValueError("invalid native fill economics")
    gross = side * (exit_price / entry - 1) * notional
    net = gross - fees + funding
    if abs(gross - reported_gross) > tolerance or abs(net - realized) > tolerance:
        raise ValueError("native cash flow does not reconcile")
    return {
        "trade_case_id": str(row["trade_case_id"]),
        "return_basis_id": "LINEAR_ACTUAL_FILL_ENTRY_NOTIONAL_BPS",
        "gross_cash": gross, "fees_cash": fees, "funding_cash": funding,
        "net_cash": net, "net_bps": net / notional * 10000,
    }
