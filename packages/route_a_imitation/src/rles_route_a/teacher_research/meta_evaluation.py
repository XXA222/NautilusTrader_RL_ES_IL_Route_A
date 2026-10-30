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
    winners = tuple(c.realized_pnl for c in complete if c.realized_pnl > 0)
    losers = tuple(c.realized_pnl for c in complete if c.realized_pnl < 0)
    breakeven = sum(c.realized_pnl == 0 for c in complete)
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
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "breakeven_trades": breakeven,
        "average_winning_trade_pnl": fsum(winners) / len(winners) if winners else None,
        "average_losing_trade_pnl": fsum(losers) / len(losers) if losers else None,
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
    recipes = {(o.exit_policy_id, o.execution_recipe, o.cost_scenario_id, o.return_basis_id)
               for o in outcomes}
    if len(recipes) > 1:
        raise ValueError("candidate quality cannot mix exit, execution, scenario or return basis")
    for decision in decisions:
        outcome = by_id[decision.group.event_id]
        obs = decision.group.observation
        if (outcome.decision_ts_ns, outcome.source_identity, outcome.fold_id) != (
                obs.decision_ts_ns, obs.source_id, obs.fold_id):
            raise ValueError("modeled outcome does not match its causal candidate")
        if outcome.outcome_status == "COMPLETE":
            if (outcome.net_return_bps is None or not isfinite(outcome.net_return_bps)
                    or outcome.label_end_ts_ns is None or outcome.censor_ts_ns is not None):
                raise ValueError("complete outcome is missing valid economic facts")
        elif outcome.net_return_bps is not None:
            raise ValueError("incomplete outcome cannot supply a net return")
    def model_accepted(decision):
        return (decision.score is not None
                and decision.score.action in ("LONG", "SHORT"))

    if any(decision.selected and (not model_accepted(decision)
                                  or decision.rejection_override is not None)
           for decision in decisions):
        raise ValueError("native-selected candidates must pass the model gate and native constraints")

    def segment(items):
        labels = [by_id[d.group.event_id] for d in items]
        values = [o.net_return_bps for o in labels if o.outcome_status == "COMPLETE"
                  and o.net_return_bps is not None]
        return {"candidates": len(labels), "complete": len(values),
                "without_outcome": len(labels) - len(values),
                "mean_net_bps": fsum(values) / len(values) if values else None,
                "win_rate": sum(v > 0 for v in values) / len(values) if values else None}
    paired = [(d.score, by_id[d.group.event_id].net_return_bps) for d in decisions
              if d.score is not None and by_id[d.group.event_id].outcome_status == "COMPLETE"]
    for score, actual in paired:
        if (not isfinite(score.expected_net_bps) or not isfinite(score.probability_positive)
                or not 0 <= score.probability_positive <= 1):
            raise ValueError("invalid candidate prediction")
    n = len(paired)
    bins = []
    for index in range(5):
        lower, upper = index / 5, (index + 1) / 5
        members = [(score.probability_positive, value) for score, value in paired
                   if lower <= score.probability_positive < upper
                   or (index == 4 and score.probability_positive == 1)]
        count = len(members)
        bins.append({"lower_inclusive": lower, "upper": upper, "upper_inclusive": index == 4,
                     "count": count,
                     "mean_predicted_probability": fsum(p for p, _ in members) / count if count else None,
                     "observed_positive_fraction": sum(v > 0 for _, v in members) / count if count else None})
    prediction = {
        "paired_complete_scored": n,
        "unscored_candidates": sum(d.score is None for d in decisions),
        "scored_without_complete_outcome": sum(
            d.score is not None and by_id[d.group.event_id].outcome_status != "COMPLETE" for d in decisions),
        "mean_absolute_error_bps": fsum(abs(s.expected_net_bps - y) for s, y in paired) / n if n else None,
        "mean_prediction_error_bps": fsum(s.expected_net_bps - y for s, y in paired) / n if n else None,
        "brier_score": fsum((s.probability_positive - (y > 0)) ** 2 for s, y in paired) / n if n else None,
        "probability_bins": bins,
        "basis": "complete scored MODEL_COUNTERFACTUAL outcomes; overlapping candidates are dependent",
    }
    return {"evidence_type": "MODEL_COUNTERFACTUAL",
            "label_recipe": next(iter(recipes)) if recipes else None,
            "prediction_quality": prediction,
            "accepted": segment(d for d in decisions if model_accepted(d)),
            "rejected": segment(d for d in decisions if not model_accepted(d)),
            "accepted_not_native_selected": segment(
                d for d in decisions if model_accepted(d) and not d.selected),
            "accepted_blocked_by_override": segment(
                d for d in decisions if model_accepted(d) and d.rejection_override is not None),
            "native_selected": segment(d for d in decisions if d.selected),
            "acceptance_basis": "entry model gate before single-position native arbitration",
            "dispositions": dict(sorted(Counter(d.disposition for d in decisions).items())),
            "quality_by_disposition": {reason: segment(d for d in decisions if d.disposition == reason)
                                       for reason in sorted({d.disposition for d in decisions})}}


def paired_cost_candidate_quality(decisions, base_outcomes, stress_outcomes):
    """Compare the same selected candidate paths under two modeled cost schedules."""
    base = {item.event_id: item for item in base_outcomes}
    stress = {item.event_id: item for item in stress_outcomes}
    selected = tuple(item for item in decisions if item.selected)
    if (len(base) != len(base_outcomes) or len(stress) != len(stress_outcomes)
            or set(base) != set(stress) or set(base) != {d.group.event_id for d in selected}):
        raise ValueError("paired costs must cover exactly the selected candidates")
    rows = []
    for decision in selected:
        a, b = base[decision.group.event_id], stress[decision.group.event_id]
        if (a.cost_scenario_id, b.cost_scenario_id) != ("base", "stress"):
            raise ValueError("paired costs require base and stress labels")
        if (a.evidence_type != "MODEL_COUNTERFACTUAL" or b.evidence_type != "MODEL_COUNTERFACTUAL"
                or a.outcome_status != b.outcome_status or a.label_end_ts_ns != b.label_end_ts_ns
                or a.return_basis_id != b.return_basis_id):
            raise ValueError("paired cost labels must share the same modeled path")
        rows.append({"event_id": a.event_id, "status": a.outcome_status,
                     "base_net_bps": a.net_return_bps, "stress_net_bps": b.net_return_bps})
    complete = [row for row in rows if row["status"] == "COMPLETE"]
    return {"evidence_type": "MODEL_COUNTERFACTUAL", "selected_candidates": len(rows),
            "paired_complete": len(complete), "without_complete_outcome": len(rows)-len(complete),
            "base_mean_net_bps": fsum(row["base_net_bps"] for row in complete)/len(complete) if complete else None,
            "stress_mean_net_bps": fsum(row["stress_net_bps"] for row in complete)/len(complete) if complete else None,
            "positive_under_both": sum(row["base_net_bps"] > 0 and row["stress_net_bps"] > 0 for row in complete),
            "outcomes": rows}



def compare_preregistered_native_runs(config, runs) -> dict[str, object]:
    """Build a complete strategy/scenario/window matrix from retained native runs.

    A missing row is reported as missing; modeled labels never fill a native cell.
    Run objects contain strategy_id, scenario, window, cases, equity_curve and
    native_order_verification from the verified host rollout.
    """
    from dataclasses import replace

    original = tuple(config["teachers"])
    controls = tuple(item for item in config["controls"] if item != "ORIGINAL_8_X1")
    strategies = (*original, "META_X1", *controls)
    scenarios = tuple(config["cost_scenarios"])
    windows = tuple(tuple(window) for window in config["intervals"]["native_screen"])
    expected = {(strategy, scenario, window) for strategy in strategies
                for scenario in scenarios for window in windows}
    supplied = {}
    for run in runs:
        key = (run["strategy_id"], run["scenario"], tuple(run["window"]))
        if key not in expected or key in supplied:
            raise ValueError("unexpected or duplicate native matrix cell")
        verification = run["native_order_verification"]
        if (not isinstance(verification, dict)
                or verification.get("authority") != "retained_m22_session_snapshots"
                or type(verification.get("barriers")) is not int
                or verification["barriers"] <= 0):
            raise ValueError("native matrix cell lacks complete M2.2 authority")
        # Cases are attributed to their original owners; normalize only the
        # strategy grouping while retaining the original cases in the run.
        cases = tuple(replace(case, teacher_id=run["strategy_id"]) for case in run["cases"])
        if any(case.scenario_id != run["scenario"] for case in cases):
            raise ValueError("native trade case scenario differs from its matrix cell")
        metrics = summarize_native_economics(cases, tuple(run["equity_curve"]))
        account = run.get("account_metrics")
        if account is not None:
            if not isinstance(account, dict):
                raise ValueError("native account metrics must be a mapping")
            try:
                fee, slip, funding = (account[name] for name in
                                      ("total_fee", "total_slippage_cost", "total_funding"))
            except KeyError as exc:
                raise ValueError("native account costs are incomplete") from exc
            if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
                    for v in (fee, slip, funding)) or fee < 0 or slip < 0):
                raise ValueError("native account costs are invalid")
            metrics.update({"account_fees": fee, "account_slippage": slip,
                "account_funding": funding,
                "account_gross_before_costs_inferred": metrics["account_equity_change"] + fee + slip - funding})
        if run.get("candidate_summary") is not None:
            metrics["candidate_summary"] = run["candidate_summary"]
        supplied[key] = metrics
    rows = []
    for strategy, scenario, window in sorted(expected):
        metrics = supplied.get((strategy, scenario, window))
        rows.append({"strategy_id": strategy, "scenario": scenario, "window": window,
                     "status": "AVAILABLE" if metrics is not None else "MISSING",
                     "metrics": metrics})
    comparisons = []
    for scenario in scenarios:
        for window in windows:
            meta = supplied.get(("META_X1", scenario, window))
            if meta is None:
                continue
            for other in (*original, *controls):
                baseline = supplied.get((other, scenario, window))
                if baseline is None:
                    continue
                comparisons.append({"scenario": scenario, "window": window, "baseline": other,
                    "net_equity_improvement": meta["account_equity_change"] - baseline["account_equity_change"],
                    "drawdown_change": meta["max_drawdown"] - baseline["max_drawdown"]})
    window_consistency = []
    for scenario in scenarios:
        for baseline in (*original, *controls):
            pairs = [row for row in comparisons if row["scenario"] == scenario and row["baseline"] == baseline]
            complete = len(pairs) == len(windows)
            window_consistency.append({"scenario": scenario, "baseline": baseline,
                "expected_windows": len(windows), "available_windows": len(pairs),
                "positive_improvement_windows": sum(row["net_equity_improvement"] > 0 for row in pairs),
                "all_windows_improved": (all(row["net_equity_improvement"] > 0 for row in pairs)
                                         if complete and windows else None),
                "status": "COMPLETE" if complete and windows else "INCOMPLETE"})
    cost_resilience = []
    if {"base", "stress"}.issubset(scenarios):
        for window in windows:
            base = supplied.get(("META_X1", "base", window))
            stress = supplied.get(("META_X1", "stress", window))
            complete = base is not None and stress is not None
            traded = complete and all(x["ordinary_completed_trades"] > 0 for x in (base, stress))
            cost_resilience.append({"window": window,
                "status": "INCOMPLETE" if not complete else "AVAILABLE" if traded else "NO_COMPLETED_TRADE",
                "base_net_equity_change": base["account_equity_change"] if base else None,
                "stress_net_equity_change": stress["account_equity_change"] if stress else None,
                "base_completed_trades": base["ordinary_completed_trades"] if base else None,
                "stress_completed_trades": stress["ordinary_completed_trades"] if stress else None,
                "both_positive_after_costs": (base["account_equity_change"] > 0
                                              and stress["account_equity_change"] > 0)
                                             if traded else None,
                "interpretation": "scenario accounts can select different candidates; not a paired-trade test"})
    return {
        "evidence_type": "SCENARIO_ONLY_NATIVE",
        "expected_cells": len(expected), "available_cells": len(supplied),
        "status": "COMPLETE" if len(supplied) == len(expected) else "INCOMPLETE",
        "rows": rows, "comparisons": comparisons, "window_consistency": window_consistency,
        "cost_resilience": cost_resilience,
        "qualification": "NOT_ESTABLISHED",
    }
