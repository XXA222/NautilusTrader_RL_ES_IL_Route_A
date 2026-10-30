"""Run the frozen Route A E1/E2/G1 Train-inner study on registered Binance data.

This is a causal 5-minute candidate screen with fixed 240-minute counterfactual
labels. It is intentionally not a Nautilus native execution/economic authority.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from rles_data.core import MINUTE_NS, parse_ns
from rles_data.features import make_feature_row
from rles_orchestrator.research_access import read_registered_train_window
from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.evaluation import CandidateBlockResult, evaluate_teacher_research_gate_v2
from rles_route_a.teacher_research.candidate_events import (
    CausalOpportunityGenerator,
    OpportunityEvent,
)
from rles_route_a.teacher_research.gate_funnel import summarize_frozen_gate_funnel
from rles_route_a.teacher_research.opportunity_diagnostics import (
    audit_screen_ablation_outcomes_after_prediction,
    audit_screen_outcomes_after_prediction,
    build_diagnostic_ablation_sets,
    build_opportunity_prediction_report,
)
from rles_route_a.teacher_research.opportunity_labels import (
    CounterfactualTradeReplay,
    label_counterfactual_candidates,
)
from rles_route_a.teacher_research.opportunity_models import OpportunityOutcome, fit_opportunity_model
from rles_route_a.teacher_research.portfolio_replay import replay_single_position_portfolio
from rles_gym_bridge.historical import historical_feature_spec


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs/route_a_teacher_e1_e2_g1_train_prereg_20261025.json"
MARKET = ROOT / "data/routea_teacher_train_20260812_20261002/market.parquet"
OUT = ROOT / ".artifacts/route-a-teacher-e1-e2-g1-20261025-run02"
INSTRUMENT = "ETHUSDT-PERP.SIM"
PROFILE = "causal_mtf_v7"
FEATURE_NAMES = (
    "closed_return_1h", "closed_return_1d", "breakout_1h",
    "zscore_1m", "trend_efficiency_15m",
)
MARKET_NAMES = {
    "trend_1h": "closed_return_1h",
    "trend_1d": "closed_return_1d",
    "breakout_1h": "breakout_1h",
    "zscore_1m": "zscore_1m",
    "efficiency_15m": "trend_efficiency_15m",
}
CADENCE_NS = 5 * MINUTE_NS
HORIZON_NS = 240 * MINUTE_NS
FOLD_ID = "route-a-teacher-train-inner-2026q3-v1"
EXIT_POLICY_ID = "fixed-240m-label-only-v1"
RETURN_BASIS_ID = "entry-notional-log-return-bps-v1"
OUTCOME_METHOD = "next-open-240m-ohlc-funding-counterfactual-v1"


def _sha(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()).hexdigest()


def _block_table(prereg: dict) -> dict[str, tuple[str, int, int, str]]:
    result: dict[str, tuple[str, int, int, str]] = {}
    for split, field in (("train_fit", "train_fit_blocks"),
                         ("train_calibration", "train_calibration_blocks"),
                         ("train_inner_screen", "screen_blocks")):
        for index, (start_text, end_text) in enumerate(prereg["windows"][field], 1):
            block_id = f"{split}-block-{index:02d}"
            result[block_id] = (split, parse_ns(start_text), parse_ns(end_text), block_id)
    return result


def _fit_block_coverage(prereg: dict, outcomes, *, calibration_start_ns: int) -> dict:
    """Expose preregistered versus actually usable fit blocks per outcome.

    This is diagnostic only: it never changes which blocks or outcomes are
    selected. A missing/empty block is visible instead of being hidden behind
    a single aggregate model.fit_independent_blocks count.
    """
    fit_blocks = {
        block_id: (start, end)
        for block_id, (split, start, end, _name) in _block_table(prereg).items()
        if split == "train_fit"
    }
    counts = {
        block_id: {"candidates": 0, "complete": 0, "censored": 0, "purged_complete": 0}
        for block_id in fit_blocks
    }
    for outcome in outcomes:
        block_id = outcome.independent_block_id
        if block_id not in counts:
            raise RuntimeError(f"fit outcome references an unregistered block: {block_id}")
        row = counts[block_id]
        row["candidates"] += 1
        if outcome.net_return_bps is None:
            row["censored"] += 1
        else:
            row["complete"] += 1
            if (outcome.label_end_ts_ns is not None
                    and outcome.label_end_ts_ns < calibration_start_ns):
                row["purged_complete"] += 1
    effective = sum(row["purged_complete"] > 0 for row in counts.values())
    return {
        "configured_block_count": len(fit_blocks),
        "effective_block_count": effective,
        "all_preregistered_blocks_effective": effective == len(fit_blocks),
        "by_block": counts,
    }


def _feature_events(store, marker_sha: str, prereg: dict, *,
                    decision_end_ns: int | None = None,
                    candidate_generator: CausalOpportunityGenerator | None = None):
    spec = historical_feature_spec(
        {"mark": 2 * 60 * MINUTE_NS, "index": 2 * 60 * MINUTE_NS,
         "funding_update": 24 * 60 * MINUTE_NS},
        profile=PROFILE,
    )
    field_index = {field.name: index for index, field in enumerate(spec.fields)}
    indices = tuple(field_index[name] for name in FEATURE_NAMES)
    bars = sorted(
        (event for event in store.events
         if event.kind == "ohlcv_1m" and event.payload.get("instrument") == INSTRUMENT),
        key=lambda event: event.ts_event_ns,
    )
    bar_times = [event.ts_event_ns for event in bars]
    opens = [float(event.payload["open"]) for event in bars]
    closes = [float(event.payload["close"]) for event in bars]
    funding = sorted(
        (event.ts_event_ns, float(event.payload["rate"]))
        for event in store.events if event.kind == "funding_settlement"
        and event.payload.get("instrument") == INSTRUMENT
    )
    funding_times = [row[0] for row in funding]
    block_table = _block_table(prereg)
    windows = prereg["windows"]
    fit_start = parse_ns(windows["train_fit"][0])
    screen_end = parse_ns(windows["train_inner_screen"][1])
    if decision_end_ns is not None:
        if type(decision_end_ns) is not int or decision_end_ns <= fit_start:
            raise ValueError("candidate scan end must be after Train-fit start")
        screen_end = min(screen_end, decision_end_ns)
    generator = candidate_generator or CausalOpportunityGenerator(
        breakout_threshold=prereg["candidate_generation"]["breakout_threshold"],
        pullback_z_threshold=prereg["candidate_generation"]["pullback_z_threshold"],
        minimum_efficiency=prereg["candidate_generation"]["minimum_efficiency"],
        rule_version=prereg["candidate_generation"]["rule_version"],
    )
    events: list[OpportunityEvent] = []
    features: dict[str, tuple[float, ...]] = {}
    block_ids: dict[str, str] = {}
    block_intervals: dict[str, tuple[int, int]] = {}
    for bar_number, bar in enumerate(bars):
        decision_ns = int(bar.available_at_ns)
        if decision_ns < fit_start or decision_ns >= screen_end:
            continue
        if bar.ts_event_ns % CADENCE_NS:
            continue
        row = make_feature_row(store, spec, decision_ns, instrument=INSTRUMENT)
        values = tuple(float(row.values[index]) for index in indices)
        feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
        market = {market_name: values[feature_index[field_name]]
                  for market_name, field_name in MARKET_NAMES.items()}
        available = {key: row.available_at_ns for key in market}
        observation = PointInTimeObservation(
            episode_id="route-a-teacher-e1-e2-train-inner",
            decision_ts_ns=decision_ns,
            market=market,
            feature_available_at_ns=available,
            account={},
            fold_id=FOLD_ID,
            source_id=marker_sha,
            snapshot_id=store.snapshot_hash,
            feature_spec_hash=spec.hash,
        )
        candidates = generator.candidates(observation)
        if not candidates:
            continue
        split_block = next((value for value in block_table.values()
                            if value[1] <= decision_ns < value[2]), None)
        if split_block is None:
            raise RuntimeError(f"candidate lies outside preregistered blocks: {decision_ns}")
        split, block_start, block_end, block_id = split_block
        for event in candidates:
            events.append(event)
            features[event.event_id] = values
            block_ids[event.event_id] = block_id
            block_intervals[event.event_id] = (block_start, block_end)
        if bar_number % 10_000 == 0:
            print(f"candidate_scan_bar={bar_number} candidates={len(events)}", flush=True)
    events.sort(key=lambda event: (event.decision_ts_ns, event.event_id))
    return (spec, tuple(events), features, block_ids, block_intervals,
            bars, bar_times, opens, closes, funding, funding_times)


def _replay_factory(*, bars, bar_times, opens, closes, funding, funding_times,
                    scenario: str, costs: dict, block_ends: dict[str, int]):
    fee = float(costs[scenario]["taker_fee_bps_per_leg"])
    slip = float(costs[scenario]["slippage_bps_per_leg"])
    open_times = [value - MINUTE_NS for value in bar_times]

    def replay(event: OpportunityEvent, exit_policy: str, execution_recipe: str,
               cost_scenario: str) -> CounterfactualTradeReplay:
        # Fill at the first 1m open strictly after the feature's availability.
        entry_index = bisect_right(open_times, event.decision_ts_ns)
        if entry_index >= len(bars):
            raise RuntimeError("no post-decision 1m open exists for candidate")
        entry_ts = bar_times[entry_index] - MINUTE_NS
        exit_ts = entry_ts + HORIZON_NS
        block_end = block_ends[event.event_id]
        if entry_ts >= block_end:
            # A candidate too close to a block boundary has no in-block fill.
            # Do not inspect its post-boundary entry candle; the adapter will
            # classify it as right-censored using the candidate's block end.
            event_bar_index = bisect_right(bar_times, event.decision_ts_ns) - 1
            if event_bar_index < 0:
                raise RuntimeError("candidate has no in-block reference candle")
            return CounterfactualTradeReplay(
                event_id=event.event_id, entry_ts_ns=entry_ts,
                entry_price=closes[event_bar_index], exit_ts_ns=None, exit_price=None,
                entry_fee_bps=fee, exit_fee_bps=fee,
                entry_slippage_bps=slip, exit_slippage_bps=slip,
                funding_cashflow_bps=0.0, censor_ts_ns=entry_ts + 1,
                simulation_receipt_id=_sha([event.event_id, scenario, "boundary-entry-censored"]),
            )
        if exit_ts >= block_end:
            # Preserve the valid in-block entry, but never look up the exit
            # candle or later funding when its fixed horizon crosses this
            # independent block boundary.
            return CounterfactualTradeReplay(
                event_id=event.event_id, entry_ts_ns=entry_ts,
                entry_price=opens[entry_index], exit_ts_ns=None, exit_price=None,
                entry_fee_bps=fee, exit_fee_bps=fee,
                entry_slippage_bps=slip, exit_slippage_bps=slip,
                funding_cashflow_bps=0.0, censor_ts_ns=block_end,
                simulation_receipt_id=_sha([event.event_id, scenario, "boundary-exit-censored"]),
            )
        exit_index = bisect_left(bar_times, exit_ts)
        if exit_index >= len(bars):
            censor = max(entry_ts + 1, min(block_end, bar_times[-1]))
            return CounterfactualTradeReplay(
                event_id=event.event_id, entry_ts_ns=entry_ts,
                entry_price=opens[entry_index], exit_ts_ns=None, exit_price=None,
                entry_fee_bps=fee, exit_fee_bps=fee,
                entry_slippage_bps=slip, exit_slippage_bps=slip,
                funding_cashflow_bps=0.0, censor_ts_ns=censor,
                simulation_receipt_id=_sha([event.event_id, scenario, "right-censored"]),
            )
        exit_available = int(bars[exit_index].available_at_ns)
        left = bisect_right(funding_times, entry_ts)
        right = bisect_right(funding_times, exit_available)
        funding_rates = [rate for _, rate in funding[left:right]]
        funding_cashflow = -event.side * math.fsum(funding_rates) * 10_000.0
        receipt = _sha([
            event.event_id, scenario, entry_ts, exit_available,
            opens[entry_index].hex(), closes[exit_index].hex(), funding_rates,
            exit_policy, execution_recipe,
        ])
        return CounterfactualTradeReplay(
            event_id=event.event_id, entry_ts_ns=entry_ts,
            entry_price=opens[entry_index], exit_ts_ns=exit_available,
            exit_price=closes[exit_index], entry_fee_bps=fee,
            exit_fee_bps=fee, entry_slippage_bps=slip,
            exit_slippage_bps=slip, funding_cashflow_bps=funding_cashflow,
            censor_ts_ns=None, simulation_receipt_id=receipt,
        )

    return replay


def _portfolio(chosen_ids: set[str], block_events: tuple[OpportunityEvent, ...],
               outcome_by_id: dict[str, OpportunityOutcome],
               replay_by_id: dict[str, CounterfactualTradeReplay],
               *, bars, bar_times, open_times, closes, funding, funding_times,
               block_end_ns: int):
    """Replay frozen selections in event order without outcome-based entry filtering.

    This remains a historical fixed-exit diagnostic, not the dynamic runtime
    authority. A censored position stays open to the block boundary and is
    terminal-marked only; it is not converted into a synthetic forced close.
    """
    event_by_id = {event.event_id: event for event in block_events}
    terminal_marks = {}
    for event_id in chosen_ids:
        replay = replay_by_id[event_id]
        if replay.exit_ts_ns is None and replay.entry_ts_ns < block_end_ns:
            mark_index = bisect_left(bar_times, block_end_ns) - 1
            if mark_index >= 0:
                terminal_marks[event_id] = (bar_times[mark_index], closes[mark_index])
    ledger = replay_single_position_portfolio(
        chosen_ids,
        block_events,
        replay_by_id,
        outcome_by_id,
        terminal_marks,
        block_end_ns=block_end_ns,
        gross_return_method="log",
    )
    accepted = list(ledger.admission.accepted_ids)
    busy = set(ledger.admission.busy_ids)
    returns = [trade.net_return_bps for trade in ledger.trades]

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for event_id in accepted:
        event = event_by_id[event_id]
        replay = replay_by_id[event_id]
        if replay.exit_ts_ns is None:
            continue
        entry_index = bisect_left(open_times, replay.entry_ts_ns)
        exit_index = bisect_left(bar_times, replay.exit_ts_ns - 1_000_000)
        entry_cost = replay.entry_fee_bps + replay.entry_slippage_bps
        exit_cost = replay.exit_fee_bps + replay.exit_slippage_bps
        start = bisect_right(funding_times, replay.entry_ts_ns)
        end = bisect_right(funding_times, replay.exit_ts_ns - 1_000_000)
        settled = funding[start:end]
        settled_index = 0
        paid = 0.0
        starting_equity = equity
        for index in range(entry_index, exit_index + 1):
            at = bar_times[index]
            while settled_index < len(settled) and settled[settled_index][0] <= at:
                paid += -event.side * settled[settled_index][1] * 10_000.0
                settled_index += 1
            marked = starting_equity + event.side * math.log(
                closes[index] / replay.entry_price
            ) * 10_000.0 + paid - entry_cost
            if index == exit_index:
                marked -= exit_cost
                net_return = outcome_by_id[event_id].net_return_bps
                if net_return is None:
                    raise ValueError("realized drawdown path has no complete outcome")
                equity = starting_equity + float(net_return)
            peak = max(peak, marked)
            max_dd = max(max_dd, peak - marked)
    if ledger.open_censored_count:
        terminal_equity = equity + ledger.terminal_mark_only_bps
        max_dd = max(max_dd, peak - terminal_equity)
    tail = max(0.0, -float(np.quantile(returns, 0.10))) if returns else 0.0
    risk_adj = (float(np.mean(returns) - 0.5 * np.std(returns, ddof=0))
                if returns else 0.0)
    return {
        "accepted": tuple(accepted), "busy": busy,
        "no_in_block_entry": set(ledger.admission.no_in_block_entry_ids),
        "returns": returns,
        "net": ledger.total_net_bps,
        "realized_net": ledger.realized_net_bps,
        "terminal_mark_only": ledger.terminal_mark_only_bps,
        "open_censored": ledger.open_censored_count, "risk_adjusted": risk_adj,
        "tail_loss": tail, "max_drawdown": max_dd,
        "turnover": len(accepted) * 20_000.0,
        "fees": ledger.fees_bps + ledger.slippage_bps,
        "funding": ledger.funding_bps,
        "filled_orders": ledger.filled_orders,
        "exposure_minutes": ledger.exposure_minutes,
    }


def _log_loss(probabilities, outcomes) -> float:
    if not outcomes:
        return 0.6931471805599453
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    y = np.asarray(outcomes, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def main() -> int:
    prereg_bytes = PREREG.read_bytes()
    prereg = json.loads(prereg_bytes)
    prereg_id = sha256(prereg_bytes).hexdigest()
    if PREREG.with_suffix(PREREG.suffix + ".sha256").read_text().strip() != prereg_id:
        raise RuntimeError("frozen preregistration SHA-256 mismatch")
    if prereg.get("schema") != "rles.route-a.teacher-e1-e2-g1-train-prereg.v1":
        raise RuntimeError("unexpected preregistration schema")
    if OUT.exists():
        raise FileExistsError(f"immutable study output already exists: {OUT}")
    marker_path = Path(str(MARKET) + ".COMPLETED.json")
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    marker_sha = sha256(marker_bytes).hexdigest()
    if marker.get("schema_version") != "0.4.1":
        raise RuntimeError("study requires a row-group-scoped 0.4.1 Train snapshot")
    start_ns = parse_ns("2026-08-12T00:00:00Z")
    end_ns = parse_ns("2026-10-01T00:01:00Z")
    environment = {
        "instrument": INSTRUMENT,
        "market_path": str(MARKET.resolve()),
        "start_ns": start_ns,
        "end_ns": end_ns,
        "curated_marker_sha256": marker_sha,
        "curated_read_mode": "row_group_scoped_v1",
    }
    config = {"output": str(OUT.resolve()), "environment": environment}
    store, read_receipt = read_registered_train_window(
        config, {"start_ns": start_ns, "end_ns": end_ns},
        marker_sha256=marker_sha, label="train-route-a-teacher-e1-e2-g1-run-20261025",
        return_receipt=True,
    )
    if read_receipt.registered_before_read is not True:
        raise RuntimeError("Train market read did not carry a prior project reservation")
    print(f"source_read_ok events={len(store.events)} snapshot={store.snapshot_hash}", flush=True)

    (spec, events, features, block_ids, block_intervals, bars, bar_times,
     opens, closes, funding, funding_times) = _feature_events(store, marker_sha, prereg)
    if not events:
        raise RuntimeError("frozen candidate recipe generated no candidate events")
    split_events = {
        split: tuple(event for event in events
                     if any(value[0] == split and value[1] <= event.decision_ts_ns < value[2]
                            for value in _block_table(prereg).values()))
        for split in ("train_fit", "train_calibration", "train_inner_screen")
    }
    if any(not split_events[name] for name in split_events):
        raise RuntimeError("candidate universe is empty for a preregistered split")
    print("candidate_counts " + json.dumps({k: len(v) for k, v in split_events.items()}), flush=True)
    open_times = [value - MINUTE_NS for value in bar_times]

    costs = prereg["label_recipe"]
    cost_map = {
        "base": {"taker_fee_bps_per_leg": costs["base_cost_bps_per_leg"]["taker_fee"],
                 "slippage_bps_per_leg": costs["base_cost_bps_per_leg"]["slippage"]},
        "stress": {"taker_fee_bps_per_leg": costs["stress_cost_bps_per_leg"]["taker_fee"],
                   "slippage_bps_per_leg": costs["stress_cost_bps_per_leg"]["slippage"]},
    }
    block_end_by_id = {block_id: end for block_id, (_split, _start, end, _name)
                       in _block_table(prereg).items()}
    end_by_event = {event.event_id: block_end_by_id[block_ids[event.event_id]]
                    for event in events}
    outputs: dict[tuple[str, str], object] = {}
    replays: dict[tuple[str, str], dict[str, CounterfactualTradeReplay]] = {}
    models = {}
    estimates: dict[tuple[str, str], dict[str, object]] = {}
    for scenario in ("base", "stress"):
        replay = _replay_factory(
            bars=bars, bar_times=bar_times, opens=opens, closes=closes,
            funding=funding, funding_times=funding_times, scenario=scenario,
            costs=cost_map, block_ends=end_by_event,
        )
        replays_by_split = {}
        for split in ("train_fit", "train_calibration", "train_inner_screen"):
            rows = split_events[split]
            cache: dict[str, CounterfactualTradeReplay] = {}

            def replay_and_cache(event, exit_policy, execution_recipe, cost_id):
                value = replay(event, exit_policy, execution_recipe, cost_id)
                cache[event.event_id] = value
                return value

            batch = label_counterfactual_candidates(
                rows, replay_and_cache,
                {event.event_id: features[event.event_id] for event in rows},
                {event.event_id: block_ids[event.event_id] for event in rows},
                {event.event_id: block_intervals[event.event_id] for event in rows},
                feature_names=FEATURE_NAMES,
                split=split,
                exit_policy_id=EXIT_POLICY_ID,
                execution_recipe=prereg["label_recipe"]["execution_recipe"],
                cost_scenario_id=scenario,
                return_basis_id=RETURN_BASIS_ID,
                outcome_method=OUTCOME_METHOD,
            )
            outputs[(scenario, split)] = batch
            replays[(scenario, split)] = cache
            replays_by_split[split] = batch
        fit_batch = replays_by_split["train_fit"]
        cal_batch = replays_by_split["train_calibration"]
        feature_subset = {event.event_id: features[event.event_id]
                          for event in (*fit_batch.candidate_events,
                                        *cal_batch.candidate_events)}
        event_subset = {event.event_id: event
                        for event in (*fit_batch.candidate_events,
                                      *cal_batch.candidate_events)}
        model = fit_opportunity_model(
            fit_batch.outcomes, cal_batch.outcomes, feature_subset,
            candidate_events=event_subset,
            feature_names=FEATURE_NAMES,
        )
        models[scenario] = model
        screen_batch = replays_by_split["train_inner_screen"]
        estimates[scenario] = {
            event.event_id: model.predict(event, features[event.event_id])
            for event in screen_batch.candidate_events
        }
        print(f"model_fit scenario={scenario} fit={model.fit_sample_count} "
              f"cal={model.calibration_sample_count} calibration_id={model.calibration_id}",
              flush=True)

    selection_rule = prereg["decision_rule"]["screen_selection"]
    screen_events = split_events["train_inner_screen"]
    base_estimates = estimates["base"]
    screen_block_by_id = {event.event_id: block_ids[event.event_id] for event in screen_events}
    screen_split_by_id = {event.event_id: "train_inner_screen" for event in screen_events}
    screen_feature_rows = {
        event.event_id: {
            **dict(zip(FEATURE_NAMES, features[event.event_id], strict=True)),
            "side": float(event.side),
            "family_code": float(event.family == "pullback"),
        }
        for event in screen_events
    }
    prediction_diagnostics = {}
    for scenario in ("base", "stress"):
        model = models[scenario]
        support_names = (*FEATURE_NAMES, "side", "family_code")
        fit_extrema = {
            name: (float(model.fit_min[index]), float(model.fit_max[index]))
            for index, name in enumerate(support_names)
        }
        prediction_diagnostics[scenario] = build_opportunity_prediction_report(
            screen_events, estimates[scenario], screen_block_by_id, screen_split_by_id,
            feature_rows=screen_feature_rows, fit_extrema=fit_extrema,
        )
    gate_funnel = summarize_frozen_gate_funnel(
        base_estimates,
        {
            event.event_id: (block_ids[event.event_id], event.family, event.side)
            for event in screen_events
        },
    )
    proposed = set(gate_funnel.selected_ids)
    reject_reasons = dict(gate_funnel.rejection_reasons)
    diagnostic_ablation_sets = build_diagnostic_ablation_sets(
        prediction_diagnostics["base"],
        negative_control_count=len(proposed),
        frozen_seed=int(prereg_id[:16], 16),
    )
    if set(diagnostic_ablation_sets[0].selected_ids) != proposed:
        raise RuntimeError("prediction diagnostics and frozen gate funnel disagree on F0")
    screen_outcomes_by_scenario = {
        scenario: {
            row.event_id: row for row in outputs[(scenario, "train_inner_screen")].outcomes
        }
        for scenario in ("base", "stress")
    }
    screen_outcome_audits = {
        scenario: audit_screen_outcomes_after_prediction(
            prediction_diagnostics[scenario], screen_outcomes_by_scenario[scenario],
            calibration_residual_p10_offset_bps=(
                models[scenario].calibration_residual_p10_offset_bps
            ),
        )
        for scenario in ("base", "stress")
    }
    ablation_outcome_audits = {
        scenario: audit_screen_ablation_outcomes_after_prediction(
            prediction_diagnostics["base"], screen_outcomes_by_scenario[scenario],
            diagnostic_ablation_sets,
            expected_outcome_cost_scenario_id=scenario,
        )
        for scenario in ("base", "stress")
    }
    ablation_ids_by_name = {
        row.ablation_id: set(row.selected_ids) for row in diagnostic_ablation_sets
    }

    block_table = _block_table(prereg)
    screen_batch_by_scenario = {s: outputs[(s, "train_inner_screen")]
                                for s in ("base", "stress")}
    gate_rows = []
    diagnostic_ablation_account_rows = []
    empty_screen_blocks = []
    screen_block_ids = [value[3] for value in block_table.values()
                        if value[0] == "train_inner_screen"]
    for block_id in screen_block_ids:
        block_events = tuple(event for event in screen_events if block_ids[event.event_id] == block_id)
        outcome_by_scenario = {
            scenario: {row.event_id: row
                       for row in screen_batch_by_scenario[scenario].outcomes
                       if row.independent_block_id == block_id}
            for scenario in ("base", "stress")
        }
        if not block_events:
            block_start, block_end = next(
                (start, end) for split, start, end, found_id in block_table.values()
                if split == "train_inner_screen" and found_id == block_id
            )
            empty_screen_blocks.append({
                "independent_block_id": block_id,
                "block_start_ts_ns": block_start,
                "block_end_ts_ns": block_end,
                "candidate_count": 0,
                "gate_row_status": "OMITTED_NO_CANDIDATE_DENOMINATOR",
            })
            continue
        block_start, block_end = block_intervals[block_events[0].event_id]
        proposed_block = {event.event_id for event in block_events} & proposed
        # A deterministic random-ranked negative control has exactly the same
        # intended count as the frozen model in each time block.
        block_event_ids = {event.event_id for event in block_events}
        control_ids = ablation_ids_by_name["F5"] & block_event_ids
        for scenario in ("base", "stress"):
            rows_for_scenario = outcome_by_scenario[scenario]
            replay_map = replays[(scenario, "train_inner_screen")]
            candidate_portfolio = _portfolio(
                proposed_block, block_events, rows_for_scenario, replay_map,
                bars=bars, bar_times=bar_times,
                open_times=open_times, closes=closes, funding=funding,
                funding_times=funding_times, block_end_ns=block_end,
            )
            baseline_ids = {event.event_id for event in block_events}
            baseline_portfolio = _portfolio(
                baseline_ids, block_events, rows_for_scenario, replay_map,
                bars=bars, bar_times=bar_times,
                open_times=open_times, closes=closes, funding=funding,
                funding_times=funding_times, block_end_ns=block_end,
            )
            control_portfolio = _portfolio(
                control_ids, block_events, rows_for_scenario, replay_map,
                bars=bars, bar_times=bar_times,
                open_times=open_times, closes=closes, funding=funding,
                funding_times=funding_times, block_end_ns=block_end,
            )
            portfolios_by_ablation = {
                "F0": candidate_portfolio,
                "F4": baseline_portfolio,
                "F5": control_portfolio,
            }
            for ablation in diagnostic_ablation_sets:
                if ablation.ablation_id in portfolios_by_ablation:
                    continue
                selected_block_ids = set(ablation.selected_ids) & block_event_ids
                portfolios_by_ablation[ablation.ablation_id] = _portfolio(
                    selected_block_ids, block_events, rows_for_scenario, replay_map,
                    bars=bars, bar_times=bar_times,
                    open_times=open_times, closes=closes, funding=funding,
                    funding_times=funding_times, block_end_ns=block_end,
                )
            for ablation in diagnostic_ablation_sets:
                portfolio = portfolios_by_ablation[ablation.ablation_id]
                diagnostic_ablation_account_rows.append({
                    "ablation_id": ablation.ablation_id,
                    "evidence_status": ablation.evidence_status,
                    "cost_scenario_id": scenario,
                    "independent_block_id": block_id,
                    "candidate_count": len(ablation_ids_by_name[ablation.ablation_id]
                                            & block_event_ids),
                    "selected_candidate_ids": tuple(sorted(
                        ablation_ids_by_name[ablation.ablation_id] & block_event_ids,
                    )),
                    "accepted_candidate_ids": tuple(sorted(portfolio["accepted"])),
                    "realized_net_bps": portfolio["realized_net"],
                    "terminal_mark_only_bps": portfolio["terminal_mark_only"],
                    "net_bps": portfolio["net"],
                    "fees_and_slippage_bps": portfolio["fees"],
                    "funding_bps": portfolio["funding"],
                    "filled_orders": portfolio["filled_orders"],
                    "open_censored_count": portfolio["open_censored"],
                    "exposure_minutes": portfolio["exposure_minutes"],
                    "turnover_bps": portfolio["turnover"],
                    "tail_loss_bps": portfolio["tail_loss"],
                    "max_drawdown_bps": portfolio["max_drawdown"],
                    "risk_adjusted_excess_bps": portfolio["risk_adjusted"],
                })
            actual_selected = set(candidate_portfolio["accepted"])
            for event in block_events:
                if event.event_id not in actual_selected and event.event_id in proposed_block:
                    if event.event_id in candidate_portfolio["no_in_block_entry"]:
                        reject_reasons[event.event_id] = "NO_IN_BLOCK_ENTRY"
                    elif event.event_id in candidate_portfolio["busy"]:
                        reject_reasons[event.event_id] = "POSITION_BUSY"
                    else:
                        reject_reasons[event.event_id] = "PORTFOLIO_REPLAY_VETO"
            model = models[scenario]
            predictions, labels = [], []
            fit_outcomes = outputs[(scenario, "train_fit")].outcomes
            fit_complete = [row for row in fit_outcomes if row.net_return_bps is not None]
            base_rate = (sum(row.net_return_bps > 0 for row in fit_complete) / len(fit_complete)
                         if fit_complete else 0.5)
            baseline_probabilities = []
            for event in block_events:
                outcome = rows_for_scenario[event.event_id]
                if outcome.net_return_bps is None:
                    continue
                estimate = estimates[scenario][event.event_id]
                predictions.append(estimate.probability_positive)
                baseline_probabilities.append(base_rate)
                labels.append(float(outcome.net_return_bps > 0))
            reason_rows = tuple(
                (event.event_id, None if event.event_id in actual_selected
                 else reject_reasons.get(event.event_id, "FROZEN_MODEL_VETO"))
                for event in block_events
            )
            gate_rows.append(CandidateBlockResult(
                preregistration_id=prereg_id,
                trial_id="route-a-teacher-e1-e2-g1-trial-v1",
                independent_block_id=block_id,
                block_start_ts_ns=block_start,
                block_end_ts_ns=block_end,
                split="train_inner_screen",
                cost_scenario_id=scenario,
                candidate_universe_ids=tuple(event.event_id for event in block_events),
                selected_candidate_ids=tuple(sorted(actual_selected)),
                candidate_outcomes_bps=tuple(
                    (event.event_id, rows_for_scenario[event.event_id].net_return_bps)
                    for event in block_events
                ),
                candidate_rejection_reasons=reason_rows,
                baseline_net_return_bps=baseline_portfolio["net"],
                candidate_net_return_bps=candidate_portfolio["net"],
                negative_control_net_return_bps=control_portfolio["net"],
                baseline_risk_adjusted_excess_bps=baseline_portfolio["risk_adjusted"],
                candidate_risk_adjusted_excess_bps=candidate_portfolio["risk_adjusted"],
                baseline_tail_loss_bps=baseline_portfolio["tail_loss"],
                candidate_tail_loss_bps=candidate_portfolio["tail_loss"],
                baseline_max_drawdown_bps=baseline_portfolio["max_drawdown"],
                candidate_max_drawdown_bps=candidate_portfolio["max_drawdown"],
                baseline_calibration_log_loss=_log_loss(baseline_probabilities, labels),
                candidate_calibration_log_loss=_log_loss(predictions, labels),
                baseline_turnover_bps=baseline_portfolio["turnover"],
                candidate_turnover_bps=candidate_portfolio["turnover"],
                candidate_fees_bps=candidate_portfolio["fees"],
                candidate_funding_bps=candidate_portfolio["funding"],
                candidate_filled_orders=candidate_portfolio["filled_orders"],
                candidate_exposure_minutes=candidate_portfolio["exposure_minutes"],
                risk_budget_id="single-exposure-unit-notional-v1",
                exit_policy_id=EXIT_POLICY_ID,
                execution_recipe=prereg["label_recipe"]["execution_recipe"],
            ))

    result = evaluate_teacher_research_gate_v2(
        gate_rows, bootstrap_replicates=10_000, bootstrap_seed=20261025,
    )
    report = {
        "schema": "rles.route-a.teacher-e1-e2-g1-run.v1",
        "preregistration_sha256": prereg_id,
        "source": {
            "path": MARKET.relative_to(ROOT).as_posix(),
            "marker_sha256": marker_sha,
            "snapshot_hash": store.snapshot_hash,
            "registered_before_read": read_receipt.registered_before_read,
            "source_event_count": len(store.events),
            "feature_profile": PROFILE,
            "feature_spec_sha256": spec.hash,
        },
        "scope": prereg["scope"],
        "candidate_counts": {name: len(rows) for name, rows in split_events.items()},
        "label_counts": {
            f"{scenario}:{split}": {
                "candidates": len(outputs[(scenario, split)].outcomes),
                "complete": outputs[(scenario, split)].complete_count,
                "censored": outputs[(scenario, split)].censored_count,
            }
            for scenario in ("base", "stress")
            for split in ("train_fit", "train_calibration", "train_inner_screen")
        },
        "fit_block_coverage": _fit_block_coverage(
            prereg,
            outputs[("base", "train_fit")].outcomes,
            calibration_start_ns=parse_ns(prereg["windows"]["train_calibration"][0]),
        ),
        "models": {
            scenario: {
                "fit_count": model.fit_sample_count,
                "calibration_count": model.calibration_sample_count,
                "fit_blocks": model.fit_independent_blocks,
                "calibration_blocks": model.calibration_independent_blocks,
                "calibration_id": model.calibration_id,
                "calibration_residual_p10_offset_bps": (
                    model.calibration_residual_p10_offset_bps
                ),
                "economic_status": model.economic_status,
                "evidence_type": model.evidence_type,
            }
            for scenario, model in models.items()
        },
        "frozen_selection_rule": selection_rule,
        "proposed_model_candidates": len(proposed),
        "screen_rejection_reason_counts": {
            reason: sum(value == reason for value in reject_reasons.values())
            for reason in sorted(set(reject_reasons.values()))
        },
        "gate_funnel": asdict(gate_funnel),
        "diagnostic_status": "DIAGNOSTIC_REUSE_NOT_OOS",
        "prediction_diagnostics": {
            scenario: asdict(prediction_diagnostics[scenario])
            for scenario in ("base", "stress")
        },
        "screen_outcome_audits": {
            scenario: asdict(screen_outcome_audits[scenario])
            for scenario in ("base", "stress")
        },
        "diagnostic_ablation_sets": [asdict(row) for row in diagnostic_ablation_sets],
        "diagnostic_ablation_outcome_audits": {
            scenario: [asdict(row) for row in ablation_outcome_audits[scenario]]
            for scenario in ("base", "stress")
        },
        "diagnostic_ablation_account_rows": diagnostic_ablation_account_rows,
        "g1": asdict(result),
        "g1_block_rows": [asdict(row) for row in gate_rows],
        "empty_screen_blocks": empty_screen_blocks,
        "economic_status": "SCENARIO_ONLY",
        "qualification": "diagnostic_only_no_alpha_teacher_promotion",
        "limitations": prereg["limitations"],
    }
    OUT.mkdir(parents=True)
    report_path = OUT / "teacher-e1-e2-g1-report.json"
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    report_path.write_text(payload, encoding="utf-8")
    report_sha = sha256(report_path.read_bytes()).hexdigest()
    report_path.with_suffix(report_path.suffix + ".sha256").write_text(report_sha, encoding="ascii")
    summary = {
        "report": str(report_path), "report_sha256": report_sha,
        "candidate_counts": report["candidate_counts"],
        "label_counts": report["label_counts"],
        "proposed_model_candidates": len(proposed),
        "g1_status": result.status,
        "g1_reasons": result.reason_codes,
        "mean_worst_scenario_increment_bps": result.mean_worst_scenario_increment_bps,
        "increment_ci95_bps": result.increment_ci95_bps,
        "negative_control_delta_ci95_bps": result.negative_control_delta_ci95_bps,
    }
    print(json.dumps(summary, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
