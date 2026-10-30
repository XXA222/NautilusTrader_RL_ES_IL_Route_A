"""Native account and candidate-quality diagnostics, with distinct evidence types."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from math import fsum, isfinite
import re

from rles_route_a.evaluation.teacher_case_report_1m import NativeTradeCase1m
from .conditional_outcome_labels import GroupOutcome
from .meta_gated_teacher import CandidateDecision
from .opportunity_models import overlap_uniqueness_weights, summarize_prediction_calibration


ACTIVE_NATIVE_CONTROL_IDS = ("FLAT", "FIXED_LONG_0.3", "FIXED_SHORT_0.3")
KNOWN_UNIMPLEMENTED_CONTROL_IDS = ("ORIGINAL_8_X1",)


def registered_control_scope(config) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Classify frozen controls as executable or explicitly planned-only."""
    controls = config.get("controls")
    if (not isinstance(controls, (list, tuple))
            or any(not isinstance(item, str) or not item for item in controls)
            or len(controls) != len(set(controls))):
        raise ValueError("preregistered controls must be unique non-empty IDs")
    supported = set(ACTIVE_NATIVE_CONTROL_IDS) | set(KNOWN_UNIMPLEMENTED_CONTROL_IDS)
    unknown = tuple(item for item in controls if item not in supported)
    if unknown:
        raise ValueError(f"unrecognized preregistered control IDs: {', '.join(unknown)}")
    active = tuple(item for item in controls if item in ACTIVE_NATIVE_CONTROL_IDS)
    planned_only = tuple(item for item in controls if item in KNOWN_UNIMPLEMENTED_CONTROL_IDS)
    return active, planned_only


def _screen_window_ns(window):
    if not isinstance(window, (tuple, list)) or len(window) != 2:
        raise ValueError("native matrix window must have two UTC timestamps")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    result = []
    for value in window:
        if not isinstance(value, str):
            raise ValueError("native matrix window timestamps must be strings")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("native matrix window timestamps must include a timezone")
        delta = parsed.astimezone(timezone.utc) - epoch
        result.append((delta.days * 86400 + delta.seconds) * 1_000_000_000
                      + delta.microseconds * 1_000)
    if result[0] >= result[1]:
        raise ValueError("native matrix window must have positive duration")
    return result


def _verify_comparison_run_receipts(config, runs):
    receipts = [run.get("registered_screen_split") for run in runs]
    present = [receipt is not None for receipt in receipts]
    if any(present) and not all(present):
        raise ValueError("native comparison cannot mix runs with and without registered receipts")
    if not any(present):
        return "UNVERIFIED_NO_REGISTERED_RECEIPTS"
    shared = None
    digest_fields = ("preregistration_sha256", "source_marker_sha256",
                     "market_snapshot_sha256", "import_manifest_sha256")
    for run, receipt in zip(runs, receipts):
        if not isinstance(receipt, dict):
            raise ValueError("registered native run receipt must be an object")
        if (receipt.get("status") != "PASS_REGISTERED_TEMPORAL_SEPARATION"
                or receipt.get("independent_trials_claimed") is not False
                or receipt.get("protocol_id") != config.get("protocol_id")
                or receipt.get("strategy_id") != run["strategy_id"]
                or receipt.get("scenario") != run["scenario"]
                or receipt.get("screen_interval_ns") != _screen_window_ns(run["window"])
                or type(receipt.get("seed")) is not int or receipt["seed"] < 0):
            raise ValueError("registered native run receipt does not match its matrix cell")
        for name in digest_fields:
            if not isinstance(receipt.get(name), str) or not re.fullmatch(
                    r"[0-9a-f]{64}", receipt[name]):
                raise ValueError(f"registered native run receipt has invalid {name}")
        if run["strategy_id"] == "META_X1":
            available_after = receipt.get("model_available_after_ns")
            if (type(available_after) is not int
                    or available_after > receipt["screen_interval_ns"][0]):
                raise ValueError("Meta model receipt is not available before the screen window")
        identity = (receipt["protocol_id"], *(receipt[name] for name in digest_fields),
                    receipt["seed"])
        if shared is None:
            shared = identity
        elif identity != shared:
            raise ValueError("native comparison mixes preregistrations, sources, or seeds")
    return "VERIFIED_SHARED_PROTOCOL_SOURCE_AND_SEED"


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
        items = tuple(items)
        labels = [by_id[d.group.event_id] for d in items]
        status_counts = Counter(label.outcome_status for label in labels)
        incomplete_status_counts = Counter(label.outcome_status for label in labels
                                           if label.outcome_status != "COMPLETE")
        values = [o.net_return_bps for o in labels if o.outcome_status == "COMPLETE"
                  and o.net_return_bps is not None]
        decomposed = [o for o in labels if o.outcome_status == "COMPLETE"
                      and all(value is not None for value in (
                          o.gross_return_bps, o.fees_bps, o.slippage_bps,
                          o.funding_cashflow_bps))]
        positive = [value for value in values if value > 0]
        negative = [value for value in values if value < 0]
        breakeven = sum(value == 0 for value in values)
        return {"candidates": len(labels), "complete": len(values),
                "without_outcome": len(labels) - len(values),
                "complete_coverage": len(values) / len(labels) if labels else None,
                "decomposed_complete_outcomes": len(decomposed),
                "decomposition_coverage_of_complete": (
                    len(decomposed) / len(values) if values else None
                ),
                "mean_gross_return_bps": (
                    fsum(o.gross_return_bps for o in decomposed) / len(decomposed)
                    if decomposed else None
                ),
                "mean_fees_bps": (
                    fsum(o.fees_bps for o in decomposed) / len(decomposed)
                    if decomposed else None
                ),
                "mean_slippage_bps": (
                    fsum(o.slippage_bps for o in decomposed) / len(decomposed)
                    if decomposed else None
                ),
                "mean_funding_cashflow_bps": (
                    fsum(o.funding_cashflow_bps for o in decomposed) / len(decomposed)
                    if decomposed else None
                ),
                "outcome_status_counts": dict(sorted(status_counts.items())),
                "incomplete_outcome_status_counts": dict(sorted(incomplete_status_counts.items())),
                "positive_complete_outcomes": len(positive),
                "negative_complete_outcomes": len(negative),
                "breakeven_complete_outcomes": breakeven,
                "average_positive_net_bps": fsum(positive) / len(positive) if positive else None,
                "average_negative_net_bps": fsum(negative) / len(negative) if negative else None,
                "mean_net_bps": fsum(values) / len(values) if values else None,
                "win_rate": sum(v > 0 for v in values) / len(values) if values else None}

    def acceptance_slices(items):
        rows = tuple(items)
        return {"all": segment(rows),
                "accepted": segment(d for d in rows if model_accepted(d)),
                "rejected": segment(d for d in rows if not model_accepted(d)),
                "disposition_counts": dict(sorted(Counter(d.disposition for d in rows).items())),
                "rejection_disposition_counts": dict(sorted(Counter(
                    d.disposition for d in rows if d.disposition.startswith("REJECTED_")
                ).items()))}

    quality_by_side = {
        "long": acceptance_slices(d for d in decisions if d.group.side == 1),
        "short": acceptance_slices(d for d in decisions if d.group.side == -1),
    }
    teacher_ids = sorted({teacher_id for d in decisions for teacher_id, _ in d.group.teachers})
    quality_by_teacher = {
        teacher_id: acceptance_slices(
            d for d in decisions if teacher_id in {source for source, _ in d.group.teachers}
        )
        for teacher_id in teacher_ids
    }
    consensus_counts = sorted({len(d.group.teachers) for d in decisions})
    quality_by_consensus = {
        str(count): acceptance_slices(d for d in decisions if len(d.group.teachers) == count)
        for count in consensus_counts
    }
    def prediction_report(rows):
        rows = tuple(rows)
        scored = tuple(d for d in rows if d.score is not None)
        complete = tuple(d for d in scored if by_id[d.group.event_id].outcome_status == "COMPLETE")
        for decision in scored:
            score = decision.score
            assert score is not None
            if (not isfinite(score.expected_net_bps) or not isfinite(score.probability_positive)
                    or not 0 <= score.probability_positive <= 1):
                raise ValueError("invalid candidate prediction")
        labels = tuple(by_id[d.group.event_id] for d in complete)
        status_counts = Counter(by_id[d.group.event_id].outcome_status for d in scored)
        result = {
            "candidate_count": len(rows),
            "scored_candidates": len(scored),
            "unscored_candidates": len(rows) - len(scored),
            "complete_scored_candidates": len(complete),
            "complete_scored_coverage": len(complete) / len(rows) if rows else None,
            "scored_without_complete_outcome": len(scored) - len(complete),
            "scored_outcome_status_counts": dict(sorted(status_counts.items())),
            # This pure summarizer validates candidate/outcome pairing, but it
            # cannot establish that the source window is independent of model
            # fitting or threshold selection. The caller must attach that
            # evidence from the frozen run identity and source ledger.
            "independence_status": "UNVERIFIED_BY_GENERIC_SUMMARIZER",
            "thresholds_adjusted": False,
            "basis": "complete scored MODEL_COUNTERFACTUAL outcomes; overlapping candidates are dependent",
        }
        if not complete:
            result.update({
                "paired_complete_scored": 0,
                "mean_absolute_error_bps": None,
                "mean_prediction_error_bps": None,
                "brier_score": None,
                "probability_bins": [{"lower_inclusive": i / 5, "upper": (i + 1) / 5,
                    "upper_inclusive": i == 4, "count": 0,
                    "mean_predicted_probability": None, "observed_positive_fraction": None}
                    for i in range(5)],
                "net_return_bins": [],
                "probability_calibration_bins": [],
            })
            return result
        scores = tuple(d.score for d in complete)
        assert all(score is not None for score in scores)
        actual = tuple(label.net_return_bps for label in labels)
        if any(value is None for value in actual):
            raise ValueError("complete screen outcome lacks a net return")
        probabilities = tuple(score.probability_positive for score in scores)
        positives = tuple(value > 0 for value in actual)
        weighted = summarize_prediction_calibration(
            tuple(score.expected_net_bps for score in scores), actual,
            probabilities, positives,
            sample_weights=overlap_uniqueness_weights(labels),
        )
        n = len(complete)
        bins = []
        for index in range(5):
            lower, upper = index / 5, (index + 1) / 5
            members = [(p, y) for p, y in zip(probabilities, actual)
                       if lower <= p < upper or (index == 4 and p == 1)]
            count = len(members)
            bins.append({"lower_inclusive": lower, "upper": upper, "upper_inclusive": index == 4,
                         "count": count,
                         "mean_predicted_probability": fsum(p for p, _ in members) / count if count else None,
                         "observed_positive_fraction": sum(y > 0 for _, y in members) / count if count else None})
        result.update({
            "paired_complete_scored": n,
            "mean_absolute_error_bps": fsum(abs(s.expected_net_bps - y) for s, y in zip(scores, actual)) / n,
            "mean_prediction_error_bps": fsum(s.expected_net_bps - y for s, y in zip(scores, actual)) / n,
            "brier_score": fsum((p - y) ** 2 for p, y in zip(probabilities, positives)) / n,
            "probability_bins": bins,
            "net_return_bins": weighted["net_return_bins"],
            "probability_calibration_bins": weighted["probability_bins"],
            "uniqueness_weighted_metrics": {key: weighted[key] for key in (
                "effective_weighted_samples", "mean_predicted_net_bps", "mean_observed_net_bps",
                "mean_net_prediction_error_bps", "net_prediction_mae_bps",
                "net_prediction_rmse_bps", "mean_predicted_probability_positive",
                "observed_positive_rate", "probability_brier_score", "probability_log_loss")},
        })
        return result

    account_executable = tuple(d for d in decisions if not d.group.observation.pending_order_ids
                               and abs(d.group.observation.account_value("net_exposure")) <= 1e-8)
    account_blocked = tuple(d for d in decisions if d not in account_executable)
    prediction = prediction_report(decisions)
    prediction_by_account_state = {
        "executable_flat": prediction_report(account_executable),
        "position_or_pending_order_blocked": prediction_report(account_blocked),
    }
    return {"evidence_type": "MODEL_COUNTERFACTUAL",
            "label_recipe": next(iter(recipes)) if recipes else None,
            "prediction_quality": prediction,
            "prediction_quality_by_account_state": prediction_by_account_state,
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
                                       for reason in sorted({d.disposition for d in decisions})},
            "quality_by_side": quality_by_side,
            "quality_by_teacher": quality_by_teacher,
            "quality_by_consensus_count": quality_by_consensus,
            "slice_interpretation": {
                "teacher_slices_overlap": True,
                "consensus_slices_overlap": False,
                "basis": "complete MODEL_COUNTERFACTUAL outcomes; slices are descriptive, not independent trials",
            }}


def _reconcile_native_case_cashflows(cases, account, account_equity_change: float):
    """Fail closed if retained native cases disagree with the account ledger."""
    case_totals = {
        "net_pnl": fsum(case.realized_pnl for case in cases),
        "fees": fsum(case.fees for case in cases),
        "slippage": fsum(case.slippage_cost for case in cases),
        "funding": fsum(case.funding for case in cases),
    }
    account_totals = {
        "net_pnl": account_equity_change,
        "fees": float(account["total_fee"]),
        "slippage": float(account["total_slippage_cost"]),
        "funding": float(account["total_funding"]),
    }
    differences = {name: account_totals[name] - case_totals[name] for name in case_totals}
    tolerance = max(1e-6, 1e-9 * max(1., *(abs(x) for x in (*case_totals.values(),
                                                              *account_totals.values()))))
    if any(abs(value) > tolerance for value in differences.values()):
        raise ValueError("native trade-case cash flows do not reconcile with account snapshots")
    return {"status": "RECONCILED", "tolerance_cash": tolerance,
            "case_totals": case_totals, "account_totals": account_totals,
            "account_minus_cases": differences,
            "case_population": "all retained cases including terminal censoring"}


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
                     "base_gross_bps": a.gross_return_bps,
                     "base_fees_bps": a.fees_bps,
                     "base_slippage_bps": a.slippage_bps,
                     "base_funding_bps": a.funding_cashflow_bps,
                     "base_net_bps": a.net_return_bps,
                     "stress_gross_bps": b.gross_return_bps,
                     "stress_fees_bps": b.fees_bps,
                     "stress_slippage_bps": b.slippage_bps,
                     "stress_funding_bps": b.funding_cashflow_bps,
                     "stress_net_bps": b.net_return_bps})
    complete = [row for row in rows if row["status"] == "COMPLETE"]
    return {"evidence_type": "MODEL_COUNTERFACTUAL", "selected_candidates": len(rows),
            "paired_complete": len(complete), "without_complete_outcome": len(rows)-len(complete),
            "base_mean_net_bps": fsum(row["base_net_bps"] for row in complete)/len(complete) if complete else None,
            "stress_mean_net_bps": fsum(row["stress_net_bps"] for row in complete)/len(complete) if complete else None,
            "base_mean_gross_bps": fsum(row["base_gross_bps"] for row in complete)/len(complete)
                                   if complete and all(row["base_gross_bps"] is not None for row in complete) else None,
            "base_mean_fees_bps": fsum(row["base_fees_bps"] for row in complete)/len(complete)
                                  if complete and all(row["base_fees_bps"] is not None for row in complete) else None,
            "base_mean_slippage_bps": fsum(row["base_slippage_bps"] for row in complete)/len(complete)
                                      if complete and all(row["base_slippage_bps"] is not None for row in complete) else None,
            "base_mean_funding_bps": fsum(row["base_funding_bps"] for row in complete)/len(complete)
                                     if complete and all(row["base_funding_bps"] is not None for row in complete) else None,
            "stress_mean_gross_bps": fsum(row["stress_gross_bps"] for row in complete)/len(complete)
                                     if complete and all(row["stress_gross_bps"] is not None for row in complete) else None,
            "stress_mean_fees_bps": fsum(row["stress_fees_bps"] for row in complete)/len(complete)
                                    if complete and all(row["stress_fees_bps"] is not None for row in complete) else None,
            "stress_mean_slippage_bps": fsum(row["stress_slippage_bps"] for row in complete)/len(complete)
                                        if complete and all(row["stress_slippage_bps"] is not None for row in complete) else None,
            "stress_mean_funding_bps": fsum(row["stress_funding_bps"] for row in complete)/len(complete)
                                       if complete and all(row["stress_funding_bps"] is not None for row in complete) else None,
            "positive_under_both": sum(row["base_net_bps"] > 0 and row["stress_net_bps"] > 0 for row in complete),
            "outcomes": rows}



def compare_preregistered_native_runs(config, runs) -> dict[str, object]:
    """Build a complete strategy/scenario/window matrix from retained native runs.

    A missing row is reported as missing; modeled labels never fill a native cell.
    Run objects contain strategy_id, scenario, window, cases, equity_curve and
    native_order_verification from the verified host rollout.
    """
    from dataclasses import replace

    runs = tuple(runs)
    original = tuple(config["teachers"])
    active_controls, planned_only_controls = registered_control_scope(config)
    preregistered_controls = tuple(config["controls"])
    # The current runner has no implementation for this preregistered
    # aggregate teacher control. Keep it in the planned matrix as MISSING,
    # while separately reporting the executable active matrix.
    controls = active_controls
    strategies = (*original, "META_X1", *controls)
    planned_strategies = (*original, "META_X1", *preregistered_controls)
    scenarios = tuple(config["cost_scenarios"])
    windows = tuple(tuple(window) for window in config["intervals"]["native_screen"])
    active_expected = {(strategy, scenario, window) for strategy in strategies
                       for scenario in scenarios for window in windows}
    expected = {(strategy, scenario, window) for strategy in planned_strategies
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
        account = run.get("account_metrics")
        if not isinstance(account, dict):
            raise ValueError("native matrix cell requires complete native account costs")
        try:
            fee, slip, funding = (account[name] for name in
                                  ("total_fee", "total_slippage_cost", "total_funding"))
        except KeyError as exc:
            raise ValueError("native account costs are incomplete") from exc
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
                for v in (fee, slip, funding)) or fee < 0 or slip < 0):
            raise ValueError("native account costs are invalid")
        metrics = summarize_native_economics(cases, tuple(run["equity_curve"]))
        metrics.update({"account_fees": fee, "account_slippage": slip,
            "account_funding": funding,
            "account_gross_before_costs_inferred": metrics["account_equity_change"] + fee + slip - funding})
        metrics["native_case_cashflow_reconciliation"] = _reconcile_native_case_cashflows(
            cases, account, metrics["account_equity_change"])
        if run.get("candidate_summary") is not None:
            metrics["candidate_summary"] = run["candidate_summary"]
        supplied[key] = metrics
    run_receipt_status = _verify_comparison_run_receipts(config, runs)
    rows = []
    for strategy, scenario, window in sorted(expected):
        metrics = supplied.get((strategy, scenario, window))
        rows.append({"strategy_id": strategy, "scenario": scenario, "window": window,
                     "status": "AVAILABLE" if metrics is not None else "MISSING",
                     "metrics": metrics})
    comparisons = []
    comparison_metrics = {
        "completed_trade_count_delta": "ordinary_completed_trades",
        "completed_trade_gross_before_costs_delta": "completed_trade_gross_before_costs",
        "completed_trade_fees_delta": "completed_trade_fees",
        "completed_trade_slippage_delta": "completed_trade_slippage",
        "completed_trade_funding_delta": "completed_trade_funding",
        "completed_trade_net_pnl_delta": "completed_trade_net_pnl",
        "account_fees_delta": "account_fees",
        "account_slippage_delta": "account_slippage",
        "account_funding_delta": "account_funding",
        "account_gross_before_costs_delta": "account_gross_before_costs_inferred",
        "win_rate_delta": "win_rate",
        "average_winning_trade_pnl_delta": "average_winning_trade_pnl",
        "average_losing_trade_pnl_delta": "average_losing_trade_pnl",
        "average_trade_pnl_delta": "average_trade_pnl",
        "average_holding_minutes_delta": "average_holding_minutes",
    }

    def metric_delta(meta_metrics, baseline_metrics, metric):
        meta_value, baseline_value = meta_metrics.get(metric), baseline_metrics.get(metric)
        if (isinstance(meta_value, bool) or isinstance(baseline_value, bool)
                or not isinstance(meta_value, (int, float))
                or not isinstance(baseline_value, (int, float))):
            return None
        return meta_value - baseline_value

    for scenario in scenarios:
        for window in windows:
            meta = supplied.get(("META_X1", scenario, window))
            if meta is None:
                continue
            for other in (*original, *preregistered_controls):
                baseline = supplied.get((other, scenario, window))
                if baseline is None:
                    continue
                comparison = {"scenario": scenario, "window": window, "baseline": other,
                    "net_equity_improvement": meta["account_equity_change"] - baseline["account_equity_change"],
                    "drawdown_change": meta["max_drawdown"] - baseline["max_drawdown"]}
                comparison.update({key: metric_delta(meta, baseline, metric)
                                   for key, metric in comparison_metrics.items()})
                for side in ("long", "short"):
                    for metric in ("count", "net_pnl", "win_rate"):
                        comparison[f"{side}_{metric}_delta"] = metric_delta(
                            meta[side], baseline[side], metric)
                comparisons.append(comparison)
    window_consistency = []
    for scenario in scenarios:
        for baseline in (*original, *preregistered_controls):
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
    active_available = sum(key in supplied for key in active_expected)
    planned_status = "COMPLETE" if len(supplied) == len(expected) else "INCOMPLETE"
    active_status = "COMPLETE" if active_available == len(active_expected) else "INCOMPLETE"
    return {
        "evidence_type": "SCENARIO_ONLY_NATIVE",
        "expected_cells": len(expected), "available_cells": len(supplied),
        "active_expected_cells": len(active_expected),
        "active_available_cells": active_available,
        "active_status": active_status,
        "run_receipt_status": run_receipt_status,
        "status": planned_status,
        "active_controls": list(controls),
        "preregistered_controls": list(preregistered_controls),
        "preregistered_controls_outside_active_matrix": list(planned_only_controls),
        "preregistered_controls_not_executed": [
            control for control in preregistered_controls
            if not any(key[0] == control for key in supplied)
        ],
        "rows": rows, "comparisons": comparisons, "window_consistency": window_consistency,
        "cost_resilience": cost_resilience,
        "qualification": "NOT_ESTABLISHED",
    }
