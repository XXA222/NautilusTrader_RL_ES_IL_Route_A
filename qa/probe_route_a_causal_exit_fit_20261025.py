"""Train-fit mechanism probe: label candidates with the runtime P0 ordinary exit.

This deliberately neither refits a model nor reopens the consumed screen.
It is not native replay or an Alpha Teacher gate. Four fit blocks, candidate
thresholds and Base/Stress costs are inherited unchanged from the prior recipe.
"""
from __future__ import annotations

from collections import Counter
import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import statistics

from rles_data.core import MINUTE_NS, parse_ns
from rles_data.features import make_feature_row
from rles_orchestrator.research_access import read_registered_train_window
from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.teacher_research.candidate_events import CausalOpportunityGenerator
from rles_route_a.teacher_research.causal_exit import CausalExitSchedule, TrendInvalidationExit
from rles_route_a.teacher_research.causal_exit_replay import (
    attach_funding_mark_proxies, make_causal_exit_replay,
)
from rles_route_a.teacher_research.opportunity_labels import label_counterfactual_candidates


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".artifacts/route-a-causal-exit-fit-20261025-run02"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-confirmation", action="store_true")
    args = parser.parse_args()
    output = (ROOT / ".artifacts/route-a-causal-exit-recovery-fit-20261025"
              if args.recovery_confirmation else OUT)
    return run(output, recovery_confirmation=args.recovery_confirmation)


def run(OUT, *, recovery_confirmation=False):
    if OUT.exists():
        raise FileExistsError(f"immutable probe already exists: {OUT}")
    path = ROOT / "qa/run_route_a_teacher_e1_e2_g1_20261025.py"
    module_spec = importlib.util.spec_from_file_location("frozen_teacher_runner", path)
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    prereg_bytes = runner.PREREG.read_bytes()
    prereg_sha = sha256(prereg_bytes).hexdigest()
    if runner.PREREG.with_suffix(".json.sha256").read_text().strip() != prereg_sha:
        raise RuntimeError("prior frozen recipe checksum mismatch")
    prereg = json.loads(prereg_bytes)
    generator = None
    if recovery_confirmation:
        generator = CausalOpportunityGenerator(
            rule_version="route_a_candidates_v3_recovery",
            require_recovery_confirmation=True,
        )
        runner.FEATURE_NAMES += ("return_5m", "return_15m")
        runner.MARKET_NAMES.update({"return_5m": "return_5m", "return_15m": "return_15m"})
    marker_bytes = Path(str(runner.MARKET) + ".COMPLETED.json").read_bytes()
    marker_sha = sha256(marker_bytes).hexdigest()
    marker = json.loads(marker_bytes)
    groups = marker["row_groups"][:6]
    scope_start = min(min(g["min_ts_event_ns"], g["min_available_at_ns"]) for g in groups)
    scope_end = 1 + max(max(g["max_ts_event_ns"], g["max_available_at_ns"]) for g in groups)
    fit_start, fit_end = map(parse_ns, prereg["windows"]["train_fit"])
    if scope_end >= parse_ns(prereg["windows"]["train_inner_screen"][0]):
        raise RuntimeError("fit probe scope reaches screen")
    rule = TrendInvalidationExit()  # same unchanged P0 ordinary exit threshold
    recipe = {
        "prior_preregistration_sha256": prereg_sha,
        "candidate_generator": asdict(generator or CausalOpportunityGenerator()),
        "exit_policy_id": rule.policy_id,
        "purpose": "TRAIN_FIT_MECHANISM_DIAGNOSTIC_NOT_MODEL_SELECTION_OR_QUALIFICATION",
        "execution_recipe": "next-open-constant-qty-entry-notional-costs-v1",
        "costs": {"base": [5., 1.], "stress": [10., 5.]},
        "fit_interval": [fit_start, fit_end], "read_scope": [scope_start, scope_end],
        "scope_note": "whole Parquet row groups include unscored calibration prefix",
        "hold_horizon": None, "exit_cadence_minutes": 1,
        "hard_risk": "not modeled: ordinary-exit candidate labels only",
        "funding": "constant quantity times last closed mark (<1m old) proxy; (entry, exit]",
        "prior_attempt": "run01 failed before outcomes: exact mark timestamp join rejected 1ms settlement jitter",
        "return_basis": "entry-notional-simple-bps-v1",
        "screen_read": False, "test_holdout_read": False,
    }
    OUT.mkdir(parents=True)
    frozen = json.dumps(recipe, sort_keys=True, indent=2).encode()
    (OUT / "recipe.json").write_bytes(frozen)  # freeze before loading outcomes
    recipe_sha = sha256(frozen).hexdigest()
    (OUT / "recipe.json.sha256").write_text(recipe_sha)
    config = {"output": str(OUT), "environment": {
        "instrument": runner.INSTRUMENT, "market_path": str(runner.MARKET),
        "start_ns": scope_start, "end_ns": scope_end,
        "curated_marker_sha256": marker_sha, "curated_read_mode": "row_group_scoped_v1",
    }}
    store, receipt = read_registered_train_window(
        config, {"start_ns": scope_start, "end_ns": scope_end}, marker_sha256=marker_sha,
        label="train-route-a-causal-exit-fit-20261025", return_receipt=True,
    )
    if not receipt.registered_before_read:
        raise RuntimeError("missing prior Train access reservation")
    (spec, events, features, block_ids, intervals, bars, *_) = runner._feature_events(
        store, marker_sha, prereg, decision_end_ns=fit_end,
        candidate_generator=generator,
    )
    index = {f.name: i for i, f in enumerate(spec.fields)}

    def observations():
        count = 0
        for bar in bars:
            ts = int(bar.available_at_ns)
            if not fit_start <= ts < fit_end:
                continue
            row = make_feature_row(store, spec, ts, instrument=runner.INSTRUMENT)
            market = {"trend_1h": float(row.values[index["closed_return_1h"]]),
                      "trend_1d": float(row.values[index["closed_return_1d"]])}
            yield PointInTimeObservation(
                episode_id="fit-exit-probe", decision_ts_ns=ts, market=market,
                feature_available_at_ns={key: row.available_at_ns for key in market},
                account={}, fold_id=runner.FOLD_ID, source_id=marker_sha,
                snapshot_id=store.snapshot_hash, feature_spec_hash=spec.hash,
            )
            count += 1
            if count % 5000 == 0:
                print(f"causal_exit_barriers={count}", flush=True)

    schedule = CausalExitSchedule.from_observations(
        observations(), rule=rule, start_ns=fit_start, end_ns=fit_end, cadence_ns=MINUTE_NS,
    )
    opens = tuple((b.ts_event_ns - MINUTE_NS, float(b.payload["open"])) for b in bars
                  if fit_start <= b.ts_event_ns - MINUTE_NS < fit_end)
    marks = {e.ts_event_ns: float(e.payload["price"]) for e in store.events
             if e.kind == "mark" and e.payload.get("instrument") == runner.INSTRUMENT
             and fit_start <= e.ts_event_ns < fit_end}
    rates = tuple(sorted((e.ts_event_ns, float(e.payload["rate"]))
                    for e in store.events if e.kind == "funding_settlement"
                    and e.payload.get("instrument") == runner.INSTRUMENT
                    and fit_start <= e.ts_event_ns < fit_end))
    funding = attach_funding_mark_proxies(rates, tuple(sorted(marks.items())))
    results = {}
    for scenario, (fee, slip) in recipe["costs"].items():
        replay = make_causal_exit_replay(
            schedule=schedule, opens=opens, funding=funding,
            block_ends={key: value[1] for key, value in intervals.items()},
            fee_bps=fee, slippage_bps=slip,
            execution_recipe=recipe["execution_recipe"], scenario=scenario,
        )
        batch = label_counterfactual_candidates(
            events, replay, features, block_ids, intervals, feature_names=runner.FEATURE_NAMES,
            split="train_fit", exit_policy_id=rule.policy_id,
            execution_recipe=recipe["execution_recipe"], cost_scenario_id=scenario,
            return_basis_id=recipe["return_basis"], gross_return_method="simple",
            outcome_method="p0-ordinary-causal-exit-next-open-v1",
        )
        by_block = {}
        for block, (split, _, _, _) in runner._block_table(prereg).items():
            if split != "train_fit":
                continue
            rows = [o for o in batch.outcomes if o.independent_block_id == block]
            values = [o.net_return_bps for o in rows if o.net_return_bps is not None]
            by_block[block] = {
                "candidates": len(rows), "complete": len(values), "censored": len(rows) - len(values),
                "mean_net_bps": statistics.mean(values) if values else None,
                "positive": sum(v > 0 for v in values),
                "complete_label_duration_over_240m": sum(
                    o.label_end_ts_ns is not None and o.label_end_ts_ns - o.decision_ts_ns > 240 * MINUTE_NS
                    for o in rows),
            }
        results[scenario] = {"blocks": by_block, "complete": batch.complete_count,
                             "censored": batch.censored_count}
        (OUT / f"{scenario}-outcomes.json").write_text(json.dumps(
            [asdict(o) for o in batch.outcomes], sort_keys=True, indent=2, allow_nan=False))
    report = {
        "recipe_sha256": recipe_sha, "snapshot_hash": store.snapshot_hash,
        "marker_sha256": marker_sha, "registered_before_read": receipt.registered_before_read,
        "candidates": len(events), "families": dict(Counter(e.family for e in events)),
        "results": results, "qualification": "NOT_EVALUATED_TRAIN_FIT_ONLY_SCENARIO_ONLY",
        "limitations": ["overlapping candidate returns are not account returns or independent samples",
                        "native risk overrides and execution equivalence remain unvalidated",
                        "funding settlement mark is a minute-close proxy, not exchange cashflow authority"],
    }
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode()
    (OUT / "report.json").write_bytes(encoded)
    (OUT / "report.json.sha256").write_text(sha256(encoded).hexdigest())
    print(encoded.decode(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
