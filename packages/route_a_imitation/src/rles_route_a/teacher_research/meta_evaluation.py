"""Native account and candidate-quality diagnostics, with distinct evidence types."""
from __future__ import annotations

from collections import Counter
from math import fsum, isfinite

from rles_route_a.evaluation.teacher_case_report_1m import NativeTradeCase1m
from .conditional_outcome_labels import GroupOutcome
from .meta_gated_teacher import CandidateDecision


def summarize_native_economics(cases: tuple[NativeTradeCase1m, ...],
                               equity_curve: tuple[float, ...]) -> dict[str, object]:
    """Summarize one native strategy/window/scenario; censored cases stay separate."""
    if (not equity_curve or any(isinstance(v, bool) or not isfinite(v) or v <= 0
                                for v in equity_curve)):
        raise ValueError("positive finite native equity curve required")
    if any(not isinstance(case, NativeTradeCase1m) for case in cases):
        raise TypeError("native trade cases required")
    keys = {(c.teacher_id, c.exit_policy_id, c.scenario_id, c.block_id, c.equity_currency)
            for c in cases}
    if len(keys) > 1:
        raise ValueError("native summary cannot mix strategies, scenarios, windows or currencies")
    complete = tuple(c for c in cases if c.status == "ORDINARY_COMPLETE")
    peak = equity_curve[0]
    drawdown = 0.
    for equity in equity_curve:
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)
    gross_fill = fsum(c.side * (c.exit_fill_price / c.entry_fill_price - 1) * c.entry_notional
                      for c in complete)
    fees = fsum(c.fees for c in complete)
    funding = fsum(c.funding for c in complete)
    slip = fsum(c.slippage_cost for c in complete)
    net = fsum(c.realized_pnl for c in complete)
    if abs((gross_fill - fees + funding) - net) > 1e-6 * max(1, len(complete)):
        raise ValueError("completed native trade cash flows do not reconcile")
    def side_stats(side):
        selected = [c for c in complete if c.side == side]
        return {"count": len(selected), "net_pnl": fsum(c.realized_pnl for c in selected),
                "win_rate": sum(c.realized_pnl > 0 for c in selected) / len(selected)
                            if selected else None}
    return {
        "evidence_type": "NATIVE_SCENARIO_ACCOUNT",
        "ordinary_completed_trades": len(complete),
        "open_censored": sum(c.status == "OPEN_CENSORED" for c in cases),
        "terminal_censored": sum(c.status == "TERMINAL_CENSORED" for c in cases),
        "completed_trade_gross_before_costs": gross_fill + slip,
        "completed_trade_fees": fees, "completed_trade_slippage": slip,
        "completed_trade_funding": funding, "completed_trade_net_pnl": net,
        "account_equity_change": equity_curve[-1] - equity_curve[0],
        "max_drawdown": drawdown,
        "win_rate": sum(c.realized_pnl > 0 for c in complete) / len(complete) if complete else None,
        "average_trade_pnl": net / len(complete) if complete else None,
        "average_holding_minutes": fsum(c.holding_minutes for c in complete) / len(complete)
                                   if complete else None,
        "long": side_stats(1), "short": side_stats(-1),
    }


def summarize_candidate_quality(decisions: tuple[CandidateDecision, ...],
                                outcomes: tuple[GroupOutcome, ...]) -> dict[str, object]:
    """Compare complete modeled outcomes only; never call them native trades."""
    by_id = {outcome.event_id: outcome for outcome in outcomes}
    if len(by_id) != len(outcomes):
        raise ValueError("duplicate modeled outcome")
    ids = [decision.group.event_id for decision in decisions]
    if len(set(ids)) != len(ids) or set(ids) != set(by_id):
        raise ValueError("candidate outcomes must cover each decision exactly once")
    if any(o.evidence_type != "MODEL_COUNTERFACTUAL" or o.economic_status != "SCENARIO_ONLY"
           for o in outcomes):
        raise ValueError("candidate quality requires scenario-only modeled outcomes")
    def segment(selected):
        labels = [by_id[d.group.event_id] for d in decisions if d.selected == selected]
        values = [o.net_return_bps for o in labels if o.outcome_status == "COMPLETE"
                  and o.net_return_bps is not None]
        return {"candidates": len(labels), "complete": len(values),
                "without_outcome": len(labels) - len(values),
                "mean_net_bps": fsum(values) / len(values) if values else None,
                "win_rate": sum(v > 0 for v in values) / len(values) if values else None}
    return {"evidence_type": "MODEL_COUNTERFACTUAL", "accepted": segment(True),
            "rejected": segment(False),
            "dispositions": dict(sorted(Counter(d.disposition for d in decisions).items()))}
