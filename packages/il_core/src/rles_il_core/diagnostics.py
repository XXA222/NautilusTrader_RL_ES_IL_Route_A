"""Action support and reachability diagnostics that never score outcomes to filter data."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
import statistics
from typing import Iterable

import numpy as np

from .dataset import EpisodeTrajectory, validate_trajectory


def behavior_support_report(trajectories: Iterable[EpisodeTrajectory], *, bins: int = 21) -> dict:
    if type(bins) is not int or bins < 3:
        raise ValueError("at least three target-exposure bins required")
    actions: dict[str, list[float]] = defaultdict(list)
    gaps: dict[str, list[float]] = defaultdict(list)
    interventions: dict[str, int] = defaultdict(int)
    count: dict[str, int] = defaultdict(int)
    for episode in trajectories:
        validate_trajectory(episode)
        for row in episode.transitions:
            actions[episode.expert_id].append(row.action)
            executed = row.execution.get("executed_action")
            if executed is not None:
                gaps[episode.expert_id].append(abs(row.action - float(executed)))
            interventions[episode.expert_id] += int(
                float(row.execution.get("risk_guard_intervention") or 0.0) > 0
            )
            count[episode.expert_id] += 1
    edges = np.linspace(-1.0, 1.0, bins + 1)
    report = {}
    for expert_id, values in sorted(actions.items()):
        array = np.asarray(values, dtype=np.float64)
        hist, _ = np.histogram(array, bins=edges)
        gap = np.asarray(gaps[expert_id], dtype=np.float64)
        report[expert_id] = {
            "transitions": count[expert_id],
            "target_min": float(array.min()), "target_max": float(array.max()),
            "zero_target_fraction": float(np.mean(array == 0.0)),
            "occupied_target_bins": int(np.count_nonzero(hist)),
            "target_histogram": hist.tolist(),
            "risk_intervention_rate": interventions[expert_id] / count[expert_id],
            "raw_executed_gap_mean": float(gap.mean()) if gap.size else None,
            "raw_executed_gap_p95": float(np.quantile(gap, 0.95)) if gap.size else None,
        }
    return {"schema": "rles.il.support.v1", "bin_edges": edges.tolist(), "experts": report}


def native_rollout_report(trajectories: Iterable[EpisodeTrajectory]) -> dict:
    """Summarize independent native replay behavior and account facts.

    Returns descriptive metrics only. Equity and ledger values come from the native
    account snapshots/fills already captured by the collector; this is not a second
    execution or accounting model.
    """
    episodes = tuple(trajectories)
    if not episodes:
        raise ValueError("at least one complete native rollout is required")
    grouped: dict[str, list[EpisodeTrajectory]] = defaultdict(list)
    for episode in episodes:
        validate_trajectory(episode)
        if episode.episode_metadata.get("split") != "train_inner":
            raise ValueError("native rollout report accepts Train-inner episodes only")
        grouped[episode.expert_id].append(episode)

    experts: dict[str, dict] = {}
    for expert_id, group in sorted(grouped.items()):
        transitions = [row for episode in group for row in episode.transitions]
        actions = [float(row.action) for row in transitions]
        exposures: list[float] = []
        submissions = risk_interventions = 0
        target_flips = 0
        native_fill_count = 0
        zero_fill_submit_count = 0
        fees: dict[str, Decimal] = defaultdict(Decimal)
        funding: dict[str, Decimal] = defaultdict(Decimal)
        returns_by_episode: list[float] = []
        drawdowns_by_episode: list[float] = []
        holding_minutes: list[float] = []

        for episode in group:
            # Event and order IDs are unique within one native account replay.
            # A reset account may legitimately reuse the same canonical IDs.
            fill_facts: dict[str, dict] = {}
            submit_order_ids: set[str] = set()
            try:
                initial_equity = float(episode.episode_metadata["initial_equity"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("native rollout must retain initial native equity") from exc
            if not np.isfinite(initial_equity) or initial_equity <= 0:
                raise ValueError("initial native equity must be finite and positive")
            episode_equity = [initial_equity]
            exposure_rows: list[tuple[int, float]] = []
            previous_sign = 0
            for row in episode.transitions:
                execution = row.execution
                if "native_net_exposure" not in execution or "native_exposure_ts_ns" not in execution:
                    raise ValueError("native post-action exposure facts are required for holding diagnostics")
                exposure = float(execution["native_net_exposure"])
                exposure_ts = execution["native_exposure_ts_ns"]
                if not np.isfinite(exposure) or type(exposure_ts) is not int:
                    raise ValueError("native exposure facts must be finite and timestamped")
                exposure_rows.append((exposure_ts, exposure))
                exposures.append(exposure)

                sign = int(np.sign(row.action))
                if sign and previous_sign and sign != previous_sign:
                    target_flips += 1
                if sign:
                    previous_sign = sign
                submissions += int(execution.get("execution_plan_action") == "SUBMIT")
                if execution.get("execution_plan_action") == "SUBMIT":
                    client_order_id = execution.get("execution_client_order_id")
                    if not isinstance(client_order_id, str) or not client_order_id:
                        raise ValueError("native SUBMIT decision must retain its client order identity")
                    submit_order_ids.add(client_order_id)
                risk_interventions += int(float(execution.get("risk_guard_intervention") or 0.0) > 0)

                try:
                    equity = float(row.native_equity_after)
                except (TypeError, ValueError) as exc:
                    raise ValueError("native account equity must be numeric") from exc
                if not np.isfinite(equity) or equity <= 0:
                    raise ValueError("native account equity must be finite and positive")
                episode_equity.append(equity)

                native_fills = list(row.native_fills)
                if row.terminal_settlement is not None:
                    native_fills.extend(row.terminal_settlement.get("fills", ()))
                for fill in native_fills:
                    event_id = fill.get("event_id")
                    commission = fill.get("commission")
                    if not isinstance(event_id, str) or not event_id or not isinstance(commission, str):
                        raise ValueError("native fill identity and currency-qualified commission required")
                    previous = fill_facts.get(event_id)
                    if previous is not None:
                        if previous != dict(fill):
                            raise ValueError("native fill identity has conflicting facts")
                        continue
                    fill_facts[event_id] = dict(fill)
                    parts = commission.replace(",", "").split()
                    if len(parts) != 2:
                        raise ValueError("native commission must include amount and currency")
                    try:
                        amount = Decimal(parts[0])
                    except InvalidOperation as exc:
                        raise ValueError("native commission amount is invalid") from exc
                    if not amount.is_finite() or amount < 0:
                        raise ValueError("native commission must be finite and nonnegative")
                    fees[parts[1]] += amount
                native_funding = list(row.native_funding)
                if row.terminal_settlement is not None:
                    native_funding.extend(row.terminal_settlement.get("funding", ()))
                for adjustment in native_funding:
                    value = adjustment.get("pnl_change")
                    if not isinstance(value, str) or not isinstance(adjustment.get("currency"), str):
                        raise ValueError("native funding currency and PnL change required")
                    try:
                        amount = Decimal(value)
                    except InvalidOperation as exc:
                        raise ValueError("native funding amount is invalid") from exc
                    if not amount.is_finite():
                        raise ValueError("native funding amount must be finite")
                    funding[adjustment["currency"]] += amount

            peak = episode_equity[0]
            drawdown = 0.0
            for equity in episode_equity[1:]:
                peak = max(peak, equity)
                drawdown = max(drawdown, 1.0 - equity / peak)
            returns_by_episode.append(episode_equity[-1] / episode_equity[0] - 1.0)
            drawdowns_by_episode.append(drawdown)
            current_holding = 0.0
            for (start_ts, exposure), (end_ts, _next_exposure) in zip(
                exposure_rows, exposure_rows[1:], strict=False,
            ):
                if end_ts <= start_ts:
                    raise ValueError("native exposure timestamps must advance within an episode")
                if abs(exposure) > 1e-12:
                    current_holding += (end_ts - start_ts) / 60_000_000_000
                elif current_holding:
                    holding_minutes.append(current_holding)
                    current_holding = 0.0
            if current_holding:
                holding_minutes.append(current_holding)
            native_fill_count += len(fill_facts)
            filled_order_ids = {str(fact.get("client_order_id")) for fact in fill_facts.values()}
            zero_fill_submit_count += len(submit_order_ids - filled_order_ids)

        if not exposures or not returns_by_episode:
            raise ValueError("native rollout has no account or exposure samples")
        # Adjacent exposure intervals from the same episode form one continuous hold.
        # Episode boundaries are never joined, because each group is processed separately.
        duration_distribution = sorted(holding_minutes)
        nonzero_actions = sum(abs(value) > 1e-12 for value in actions)
        experts[expert_id] = {
            "episodes": len(group),
            "transitions": len(transitions),
            "nonzero_action_fraction": nonzero_actions / len(actions),
            "average_holding_minutes": statistics.fmean(duration_distribution) if duration_distribution else 0.0,
            "median_holding_minutes": statistics.median(duration_distribution) if duration_distribution else 0.0,
            "holding_duration_distribution_minutes": duration_distribution,
            "target_flip_count": target_flips,
            "submit_count": submissions,
            "submit_rate_per_transition": submissions / len(transitions),
            "native_fill_count": native_fill_count,
            "native_fill_rate_per_transition": native_fill_count / len(transitions),
            "zero_fill_submit_count": zero_fill_submit_count,
            "zero_position_fraction": sum(abs(value) <= 1e-12 for value in exposures) / len(exposures),
            "native_net_return": returns_by_episode[0] if len(returns_by_episode) == 1 else None,
            "native_net_return_by_episode": returns_by_episode,
            "native_max_drawdown": max(drawdowns_by_episode),
            "native_max_drawdown_by_episode": drawdowns_by_episode,
            "native_commissions_by_currency": {key: str(value) for key, value in sorted(fees.items())},
            "native_funding_pnl_by_currency": {key: str(value) for key, value in sorted(funding.items())},
            "risk_intervention_rate": risk_interventions / len(transitions),
            "risk_intervention_count": risk_interventions,
            "economic_claim_eligible": False,
        }
    return {"schema": "rles.il.native_rollout.v1", "accounting_source": "native_account_snapshots_and_ledger_facts",
            "interpretation": "descriptive_native_replay_only; no economic eligibility implied", "experts": experts}
