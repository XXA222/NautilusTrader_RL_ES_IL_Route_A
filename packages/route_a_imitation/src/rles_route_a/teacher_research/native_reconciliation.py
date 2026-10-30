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



def reconcile_counterfactual_to_native(*, replay, decision_ts_ns: int, side: int,
                                       native_case: Mapping[str, object],
                                       max_clock_difference_ns: int = 1_000_000_000) -> dict[str, float | str]:
    """Compare a linear reference-price label against the same native completed case.

    The research path and native fills may use different reference prices.
    The returned cash decomposition preserves that difference explicitly.
    """
    from .opportunity_labels import CounterfactualTradeReplay

    if not isinstance(replay, CounterfactualTradeReplay) or replay.exit_ts_ns is None:
        raise ValueError("completed counterfactual replay required")
    if (type(decision_ts_ns) is not int or type(side) is not int or side not in (-1, 1)
            or type(max_clock_difference_ns) is not int or max_clock_difference_ns < 0
            or native_case.get("entry_decision_ts_ns") != decision_ts_ns
            or native_case.get("side") != side
            or type(native_case.get("entry_fill_ts_ns")) is not int
            or type(native_case.get("exit_fill_ts_ns")) is not int
            or replay.entry_ts_ns <= decision_ts_ns
            or abs(replay.entry_ts_ns - native_case["entry_fill_ts_ns"]) > max_clock_difference_ns
            or abs(replay.exit_ts_ns - native_case["exit_fill_ts_ns"]) > max_clock_difference_ns):
        raise ValueError("research replay and native case do not identify the same trade and clocks")
    native = reconcile_native_trade(native_case)
    assert replay.exit_price is not None
    native_entry = float(native_case["entry_fill_price"])
    native_notional = float(native_case["entry_notional"])
    native_slippage = native_case.get("slippage_cost")
    if (isinstance(native_slippage, bool) or not isinstance(native_slippage, (int, float))
            or not isfinite(native_slippage) or native_slippage < 0):
        raise ValueError("native case requires finite nonnegative reported slippage")
    quantity = native_notional / native_entry
    modeled_notional = quantity * replay.entry_price
    gross = side * quantity * (replay.exit_price - replay.entry_price)
    fee = modeled_notional * (replay.entry_fee_bps + replay.exit_fee_bps) / 10000
    slip = modeled_notional * (replay.entry_slippage_bps + replay.exit_slippage_bps) / 10000
    funding = modeled_notional * replay.funding_cashflow_bps / 10000
    model_net = gross - fee - slip + funding
    native_net = float(native["net_cash"])
    price_and_native_slippage = float(native["gross_cash"]) - gross
    fees_difference = fee - float(native["fees_cash"])
    funding_difference = float(native["funding_cash"]) - funding
    decomposition = price_and_native_slippage + fees_difference + funding_difference + slip
    if abs((native_net - model_net) - decomposition) > 1e-8:
        raise ValueError("native/model return difference does not reconcile")
    return {
        "trade_case_id": str(native["trade_case_id"]),
        "model_evidence_type": "MODEL_COUNTERFACTUAL",
        "native_economic_status": str(native_case.get("economic_status", "")),
        "reference_basis": "LINEAR_REFERENCE_ENTRY_NOTIONAL_BPS",
        "native_basis": str(native["return_basis_id"]),
        "quantity": quantity, "modeled_reference_entry": replay.entry_price,
        "modeled_reference_exit": replay.exit_price,
        "modeled_net_cash": model_net, "modeled_net_bps": model_net / modeled_notional * 10000,
        "native_net_cash": native_net, "native_net_bps": float(native["net_bps"]),
        "native_reported_slippage_cash": float(native_slippage),
        "native_slippage_treatment": "included_in_actual_fill_prices_not_subtracted_again",
        "native_minus_modeled_cash": native_net - model_net,
        "native_fill_vs_reference_gross_cash": price_and_native_slippage,
        "modeled_fee_minus_native_fee_cash": fees_difference,
        "native_minus_modeled_funding_cash": funding_difference,
        "modeled_slippage_cash": slip,
    }


def reconcile_selected_native_cases(decisions, cases, replay, *, execution_recipe, scenario):
    """Compare completed cases and retain censored/unmatched native case dispositions."""
    from dataclasses import asdict
    cases = tuple(cases)
    if len({getattr(case, "trade_case_id", None) for case in cases}) != len(cases):
        raise ValueError("native trade-case identities must be unique")
    selected_by_id = {d.group.event_id: d.group for d in decisions if d.selected}
    results = []
    for case in cases:
        row = {"trade_case_id": case.trade_case_id}
        candidate_id = getattr(case, "candidate_event_id", None)
        if candidate_id is None:
            results.append({**row, "status": "UNVERIFIED_CANDIDATE_IDENTITY",
                            "reason": "native Meta case lacks the selected candidate event ID"})
            continue
        group = selected_by_id.get(candidate_id)
        if group is None:
            results.append({**row, "status": "UNMATCHED_CANDIDATE"})
            continue
        if (group.observation.decision_ts_ns != case.entry_decision_ts_ns
                or group.side != case.side
                or case.candidate_teacher_ids != tuple(name for name, _ in group.teachers)):
            results.append({**row, "status": "NOT_COMPARABLE",
                            "reason": "native case candidate id disagrees with entry clock or side"})
            continue
        row["event_id"] = group.event_id
        if case.status != "ORDINARY_COMPLETE":
            results.append({**row, "status": "NATIVE_CASE_INCOMPLETE",
                            "native_status": case.status})
            continue
        native = asdict(case)
        native["gross_cash_pnl_before_fees_and_funding"] = (
            case.side * (case.exit_fill_price / case.entry_fill_price - 1) * case.entry_notional)
        try:
            fact = replay(group, "X1", execution_recipe, scenario).trade
            comparison = reconcile_counterfactual_to_native(replay=fact,
                decision_ts_ns=case.entry_decision_ts_ns, side=case.side, native_case=native)
        except ValueError as exc:
            results.append({**row, "status": "NOT_COMPARABLE", "reason": str(exc)})
        else:
            results.append({**row, "status": "COMPARABLE", "comparison": comparison})
    from collections import Counter
    return {"native_cases": len(cases),
            "ordinary_complete_cases": sum(case.status == "ORDINARY_COMPLETE" for case in cases),
            "comparable": sum(item["status"] == "COMPARABLE" for item in results),
            "case_disposition_counts": dict(sorted(Counter(
                item["status"] for item in results).items())),
            "cases": results}
