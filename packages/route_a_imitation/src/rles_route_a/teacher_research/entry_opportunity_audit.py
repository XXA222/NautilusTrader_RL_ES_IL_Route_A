"""Descriptive D1 candidate and trade economics from retained native artifacts.

Close-to-close horizon returns are observations, not simulated fills or OOS evidence.
"""
from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping

MINUTE_NS = 60_000_000_000
HORIZONS = (5, 15, 60, 120)


def audit_entry_opportunities(
    decisions: Iterable[Mapping[str, Any]],
    trades: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize one teacher/exit/window/scenario with explicit censoring."""
    groups: dict[int, dict[str, Any]] = {}
    for side in (-1, 1):
        groups[side] = {
            "candidate_count": 0, "completed_trades": 0, "censored_trades": 0,
            "net_cash": 0.0, "gross_price_cash": 0.0, "fees": 0.0,
            "slippage_attribution": 0.0, "funding": 0.0,
            "mfe_bps": [], "mae_bps": [],
            "horizons": {str(h): {"observed": 0, "censored": 0,
                                  "sum_directional_bps": 0.0} for h in HORIZONS},
        }
    prices: dict[int, float] = {}
    candidates: list[tuple[int, int, float]] = []
    prior = -1
    case_ids: set[str] = set()
    source = fold = None
    for row in decisions:
        if row["economic_status"] != "SCENARIO_ONLY":
            raise ValueError("D1 scenario-only decisions required")
        if source is None:
            source, fold = row["source_identity"], row["fold_id"]
        if not source or not fold or (row["source_identity"], row["fold_id"]) != (source, fold):
            raise ValueError("decision source/fold changed")
        ts = int(row["decision_ts_ns"])
        if ts <= prior:
            raise ValueError("decision clock must increase")
        prior = ts
        available = int(row["latest_1m_available_at_ns"])
        close_ts = int(row["latest_1m_close_ts_ns"])
        if available > ts or close_ts > available:
            raise ValueError("decision uses unavailable bar")
        indicators = json.loads(row["indicator_values_json"])
        price = float(indicators["bar_close_1m"])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("invalid bar close")
        prices[ts] = price
        case_id = row.get("case_id")
        if case_id:
            case_ids.add(str(case_id))
        signal = row["raw_signal"]
        if signal in ("LONG_ENTRY", "SHORT_ENTRY"):
            side = 1 if signal == "LONG_ENTRY" else -1
            groups[side]["candidate_count"] += 1
            candidates.append((ts, side, price))
    for ts, side, price in candidates:
        for h in HORIZONS:
            slot = groups[side]["horizons"][str(h)]
            future = ts + h * MINUTE_NS
            # Missing intermediate bars or a block boundary are explicitly censored.
            if any(ts + i * MINUTE_NS not in prices for i in range(1, h + 1)):
                slot["censored"] += 1
            else:
                slot["observed"] += 1
                slot["sum_directional_bps"] += side * (prices[future] / price - 1) * 10000
    seen: set[str] = set()
    for row in trades:
        case_id = str(row["trade_case_id"])
        if not case_id or case_id in seen or case_id not in case_ids:
            raise ValueError("missing, duplicate, or unjoined trade case")
        seen.add(case_id)
        side = int(row["side"])
        if side not in groups:
            raise ValueError("trade side must be long or short")
        group = groups[side]
        if row["status"] == "TERMINAL_CENSORED":
            group["censored_trades"] += 1
            continue
        if row["status"] != "ORDINARY_COMPLETE":
            raise ValueError("unknown native trade case status")
        group["completed_trades"] += 1
        for key in ("realized_pnl", "gross_cash_pnl_before_fees_and_funding",
                    "fees", "slippage_cost", "funding", "mfe_bps", "mae_bps"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"nonfinite trade fact {key}")
        net, gross, fee, funding = (float(row[k]) for k in (
            "realized_pnl", "gross_cash_pnl_before_fees_and_funding", "fees", "funding"))
        if not math.isclose(net, gross - fee + funding, abs_tol=1e-6):
            raise ValueError("native case cashflow does not reconcile")
        group["net_cash"] += net
        # Slippage is already reflected in fill-price gross; add back for attribution.
        group["gross_price_cash"] += gross + float(row["slippage_cost"])
        group["fees"] += fee
        group["slippage_attribution"] += float(row["slippage_cost"])
        group["funding"] += funding
        group["mfe_bps"].append(float(row["mfe_bps"]))
        group["mae_bps"].append(float(row["mae_bps"]))
    if seen != case_ids:
        raise ValueError("decision cases lack matching native trade facts")
    for group in groups.values():
        for key in ("mfe_bps", "mae_bps"):
            values = group.pop(key)
            group["mean_" + key] = sum(values) / len(values) if values else None
        for slot in group["horizons"].values():
            total = slot.pop("sum_directional_bps")
            slot["mean_directional_bps"] = total / slot["observed"] if slot["observed"] else None
    return {"classification": "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_SCENARIO_ONLY",
            "source_identity": source, "fold_id": fold,
            "long": groups[1], "short": groups[-1],
            "interpretation": "Close-to-close horizons and MFE/MAE are not executable profits or OOS evidence."}
