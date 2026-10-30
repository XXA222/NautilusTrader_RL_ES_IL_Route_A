from __future__ import annotations

import argparse
import csv
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

import run_route_a_1m_teacher_native as native  # noqa: E402
from rles_route_a.teachers.protocol import advice  # noqa: E402
from rles_route_a.teachers_1m.contracts import TeacherSignal  # noqa: E402

PREREG = ROOT / "configs/route_a_1m_teacher_native_prereg_20261026.json"
FEATURE_SNAPSHOT = (
    ROOT / ".artifacts/route-a-x2-snapshot-fix-task3/frozen/"
    "route_a_1m_teacher_library_v1.json"
)
DEFAULT_OUTPUT = ROOT / ".artifacts/route-a-exit-isolated-first-trade-20261028"
TEACHER_IDS = (
    "T01_EMA_9_21",
    "T02_ROC_20_60",
    "T03_DONCHIAN_20",
    "T04_ZSCORE_60",
)
WINDOWS = ("inner-screen-1", "inner-screen-2", "inner-screen-3")
SCENARIOS = ("base", "stress")
EXITS = ("X1", "X2")
SEED = 20261026
EPSILON = 1e-8


class FirstRoundTripTeacher:
    """Forward the frozen teacher until its first native round trip is complete."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._seen_open = False
        self._open_side: int | None = None
        self._complete = False
        self.suppressed_decision_count = 0
        self._last_signal = getattr(delegate, "last_signal", None)

    @property
    def expert_id(self) -> str:
        return self._delegate.expert_id

    @property
    def recipe(self) -> Any:
        return self._delegate.recipe

    @property
    def last_signal(self) -> Any:
        return self._last_signal

    @property
    def completed(self) -> bool:
        return self._complete

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def reset(self, episode_id: str = "") -> None:
        self._seen_open = False
        self._open_side = None
        self._complete = False
        self.suppressed_decision_count = 0
        reset = getattr(self._delegate, "reset", None)
        if callable(reset):
            reset(episode_id)
        self._last_signal = getattr(self._delegate, "last_signal", None)

    def advise(self, observation: Any) -> Any:
        if self._complete:
            exposure = observation.account_value("net_exposure")
            if abs(exposure) > EPSILON:
                raise RuntimeError("isolated replay reopened after first trade completion")
            self.suppressed_decision_count += 1
            self._last_signal = TeacherSignal(
                "NONE", 0, "ISOLATED_FIRST_TRADE_COMPLETE",
            )
            return advice(
                observation, self.expert_id, "FLAT", 0.0,
                "ISOLATED_FIRST_TRADE_COMPLETE",
            )
        result = self._delegate.advise(observation)
        self._last_signal = getattr(self._delegate, "last_signal", None)
        return result

    def observe_native_action(
        self, action: Any, trace: Any, native_order_facts: Any,
    ) -> None:
        observe = getattr(self._delegate, "observe_native_action", None)
        if callable(observe):
            observe(action, trace, native_order_facts)
        after = float(trace.actual_exposure_after)
        pending = set(getattr(trace, "pending_order_ids_after", ()))
        pending.update(getattr(native_order_facts, "pending_order_ids_after", ()))
        if not math.isfinite(after):
            raise ValueError("native exposure must be finite")
        if abs(after) > EPSILON:
            if self._open_side is not None and (after > 0) != (self._open_side > 0):
                raise RuntimeError('native exposure reversed before first-round-trip close proof')
            self._seen_open = True
            if self._open_side is None:
                self._open_side = 1 if after > 0 else -1
            return
        if not self._seen_open or self._open_side is None or pending:
            return

        close_side = "SELL" if self._open_side > 0 else "BUY"
        filled_ids = set(getattr(trace, "filled_order_ids", ()))
        filled_ids.intersection_update(getattr(native_order_facts, "filled_order_ids", ()))
        fill_event_ids = {
            row.get("client_order_id")
            for row in getattr(native_order_facts, "native_policy_fills", ())
            if row.get("order_side") == close_side and row.get("event_id")
        }
        fully_filled_ids: set[str] = set()
        for row in getattr(native_order_facts, "native_order_statuses_after", ()):
            if (row.get("client_order_id") not in filled_ids
                    or str(row.get("status", "")).upper() != "FILLED"
                    or str(row.get("side", "")).upper() != close_side):
                continue
            try:
                quantity = Decimal(str(row.get("quantity", "0")))
                filled_quantity = Decimal(str(row.get("filled_qty", "0")))
            except (InvalidOperation, TypeError, ValueError):
                continue
            if quantity > 0 and filled_quantity == quantity:
                fully_filled_ids.add(row["client_order_id"])
        if filled_ids & fill_event_ids & fully_filled_ids:
            self._complete = True


def _read_cases(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "teacher-research-1m" / "native-trades.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"native teacher case output missing: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_decisions(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "teacher-research-1m" / "decision-rows.csv"
    if not path.is_file():
        raise FileNotFoundError(f"native decision rows missing: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _path_metrics(case: dict[str, Any], decisions: list[dict[str, str]]) -> dict[str, Any]:
    entry_ns = int(case["entry_fill_ts_ns"])
    case_id = case["trade_case_id"]
    started = False
    fees = 0.0
    funding = 0.0
    gross_path: list[float] = []
    closed_case_row = False
    equity_start = float(case["equity_start"])

    for row in decisions:
        try:
            execution_ns = int(row["execution_ts_ns"])
            exposure_before = float(row["position_before"])
            exposure_after = float(row["native_exposure_after"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("native decision row lacks exposure/account clocks") from exc
        if not started:
            if (execution_ns < entry_ns
                    or (abs(exposure_before) <= EPSILON and abs(exposure_after) <= EPSILON)):
                continue
            started = True
        fees += float(row["fee"] or 0.0)
        funding += float(row["funding"] or 0.0)
        equity = float(row["account_equity"])
        gross_path.append(equity - equity_start + fees - funding)
        if (row.get("case_id") == case_id
                and row.get("case_status") == case["status"]):
            closed_case_row = True
            break

    net = float(case["realized_pnl"])
    total_fees = float(case["fees"])
    total_funding = float(case["funding"])
    gross = net + total_fees - total_funding
    if case["status"] == "ORDINARY_COMPLETE":
        if not started or not closed_case_row or not gross_path:
            raise ValueError("completed case lacks a full native account-equity path")
        if not math.isclose(fees, total_fees, rel_tol=1e-7, abs_tol=0.01):
            raise ValueError("barrier fees do not reconcile to the completed trade case")
        if not math.isclose(funding, total_funding, rel_tol=1e-7, abs_tol=0.01):
            raise ValueError("barrier funding does not reconcile to the completed trade case")
        if not math.isclose(gross_path[-1], gross, rel_tol=1e-7, abs_tol=0.05):
            raise ValueError("close-bar gross cash PnL does not reconcile to the trade case")

    peak = 0.0
    max_giveback = 0.0
    for value in gross_path:
        peak = max(peak, value)
        max_giveback = max(max_giveback, peak - value)
    notional = float(case["entry_notional"])
    mfe_usdt = notional * float(case["mfe_bps"]) / 10_000.0
    return {
        "gross_cash_pnl": gross,
        "net_realized_pnl": net,
        "fees": total_fees,
        "slippage_attribution": float(case["slippage_cost"]),
        "funding": total_funding,
        "mfe_bps": float(case["mfe_bps"]),
        "mae_bps": float(case["mae_bps"]),
        "holding_minutes": float(case["holding_minutes"]),
        "max_closed_bar_peak_giveback_usdt": max_giveback if gross_path else None,
        "gross_profit_capture_ratio": gross / mfe_usdt if mfe_usdt > 0 else None,
        "net_profit_capture_ratio": net / mfe_usdt if mfe_usdt > 0 else None,
        "cash_identity_verified": (
            bool(closed_case_row) and bool(gross_path)
            and math.isclose(gross_path[-1], gross, rel_tol=1e-7, abs_tol=0.05)
        ),
        "native_decision_rows_in_trade": len(gross_path),
    }


def _load_run(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "native-run.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("economic_status") != "SCENARIO_ONLY"
            or report.get("qualification") != "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2"):
        raise ValueError("native run does not preserve its development-only qualification")
    if report.get("account", {}).get("liquidations") != 0:
        raise ValueError("isolated native run recorded a liquidation")
    cases = _read_cases(run_dir)
    if len(cases) != report.get("decisions", {}).get("native_trade_cases"):
        raise ValueError("native trade case count does not match the S05 run report")
    if len(cases) > 1:
        raise ValueError("first-trade wrapper allowed a second native trade case")
    case = cases[0] if cases else None
    diagnostics = _path_metrics(case, _read_decisions(run_dir)) if case else None
    return {
        "report": {
            "teacher": report["teacher"],
            "exit_variant": report["teacher_recipe"]["exit_variant"],
            "scenario": report["execution"]["scenario"],
            "seed": report["execution"]["seed"],
            "block_id": report["block_id"],
            "decisions": report["decisions"],
            "account": report["account"],
            "qualification": report["qualification"],
            "economic_status": report["economic_status"],
            "source": report["source"],
        },
        "case": case,
        "trade_diagnostics": diagnostics,
    }


def _case_identity(case: dict[str, Any]) -> tuple[int, int, float, float]:
    return (
        int(case["entry_fill_ts_ns"]),
        int(case["side"]),
        round(float(case["entry_notional"]), 6),
        round(float(case["entry_fill_price"]), 8),
    )


def _pair_slot(slot: dict[str, Any], x1: dict[str, Any], x2: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        **slot,
        "pairing_status": "UNPAIRED",
        "x1": x1,
        "x2": x2,
    }
    case1, case2 = x1["case"], x2["case"]
    if case1 is None or case2 is None:
        row["unpaired_reason"] = "NO_FIRST_NATIVE_TRADE_CASE"
        return row
    if case1["status"] != "ORDINARY_COMPLETE" or case2["status"] != "ORDINARY_COMPLETE":
        row["unpaired_reason"] = "FIRST_CASE_NOT_ORDINARY_COMPLETE"
        return row
    if _case_identity(case1) != _case_identity(case2):
        row["unpaired_reason"] = "FIRST_ENTRY_FILL_IDENTITY_MISMATCH"
        return row
    keys = (
        "gross_cash_pnl",
        "net_realized_pnl",
        "fees",
        "slippage_attribution",
        "funding",
        "mfe_bps",
        "mae_bps",
        "holding_minutes",
        "max_closed_bar_peak_giveback_usdt",
    )
    row["pairing_status"] = "PAIRED_SAME_NATIVE_ENTRY"
    row["entry_identity"] = {
        "entry_fill_ts_ns": case1["entry_fill_ts_ns"],
        "side": case1["side"],
        "entry_notional_rounded_1e6": round(float(case1["entry_notional"]), 6),
        "entry_fill_price_rounded_1e8": round(float(case1["entry_fill_price"]), 8),
    }
    row["x2_minus_x1"] = {
        key: x2["trade_diagnostics"][key] - x1["trade_diagnostics"][key]
        for key in keys
        if x1["trade_diagnostics"][key] is not None
        and x2["trade_diagnostics"][key] is not None
    }
    return row


def _validate_frozen_inputs(doc: dict[str, Any], feature_sha: str) -> None:
    expected_costs = {
        "base": {"taker_fee_bps_per_leg": 5, "slippage_bps_per_leg": 1},
        "stress": {"taker_fee_bps_per_leg": 10, "slippage_bps_per_leg": 5},
    }
    if doc.get("protocol_id") != "route-a-1m-multiteacher-train-diagnostic-20261026-v1":
        raise ValueError("wrong development diagnostic protocol")
    if doc.get("qualification") != "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2":
        raise ValueError("diagnostic preregistration has an unexpected qualification")
    if doc["source"].get("market_path") != "data/binance_ethusdt_20231230_20240415/market.parquet":
        raise ValueError("diagnostic source path differs from the frozen Train source")
    if doc["source"].get("physical_monthly_test_holdout_overlap_allowed") is not False:
        raise ValueError("source protocol does not prohibit protected-month overlap")
    if doc["roles"].get("all_results_are_development_diagnostics") is not True:
        raise ValueError("source protocol does not bind diagnostic-only outcomes")
    if doc["roles"].get("no_student_training") is not True:
        raise ValueError("source protocol unexpectedly permits student training")
    if doc["cost_scenarios"] != expected_costs:
        raise ValueError("Base/Stress costs differ from the frozen diagnostic recipe")
    if feature_sha != doc["feature_recipe"]["sha256"]:
        raise ValueError("feature recipe snapshot does not match the preregistration")


def run_matrix(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"immutable isolated-exit output already exists: {output}")
    if not FEATURE_SNAPSHOT.is_file():
        raise FileNotFoundError(f"frozen feature recipe snapshot missing: {FEATURE_SNAPSHOT}")
    doc, prereg_sha = native.verify_preregistration(
        PREREG, feature_recipe_override=FEATURE_SNAPSHOT,
    )
    feature_sha = native._sha256(FEATURE_SNAPSHOT)
    _validate_frozen_inputs(doc, feature_sha)
    native.FEATURE_RECIPE = FEATURE_SNAPSHOT
    output.mkdir(parents=True)
    paired_rows: list[dict[str, Any]] = []

    for teacher_id in TEACHER_IDS:
        for window_id in WINDOWS:
            for scenario in SCENARIOS:
                slot = {
                    "teacher": teacher_id,
                    "window_id": window_id,
                    "scenario": scenario,
                    "seed": SEED,
                }
                runs: dict[str, dict[str, Any]] = {}
                for exit_variant in EXITS:
                    run_dir = output / teacher_id / window_id / scenario / exit_variant
                    run_dir.mkdir(parents=True)
                    teacher = native.build_teacher_1m(
                        teacher_id,
                        exit_variant=cast(native.NativeExitVariant, exit_variant),
                        exit_round_trip_cost_bps=None,
                    )
                    isolated_teacher = FirstRoundTripTeacher(teacher)
                    report = native._run_native_window(
                        doc,
                        prereg_sha,
                        run_dir,
                        teacher_id=teacher_id,
                        exit_variant=cast(native.NativeExitVariant, exit_variant),
                        scenario=scenario,
                        window_id=window_id,
                        seed=SEED,
                        teacher_override=isolated_teacher,
                    )
                    if (report["decisions"]["m22_verified"] != report["decisions"]["requested"]
                            or report["account"]["liquidations"] != 0):
                        raise RuntimeError("native replay failed its existing M2.2/no-liquidation proof")
                    run = _load_run(run_dir)
                    runs[exit_variant] = run
                    (run_dir / "isolation-receipt.json").write_text(
                        json.dumps({
                            "schema": "rles.route-a-isolated-first-trade.v1",
                            "completed_first_round_trip": isolated_teacher.completed,
                            "suppressed_post_exit_decisions": isolated_teacher.suppressed_decision_count,
                            "native_case_count": len(_read_cases(run_dir)),
                            "economic_status": "SCENARIO_ONLY",
                            "qualification": "DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2",
                        }, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                paired_rows.append(_pair_slot(slot, runs["X1"], runs["X2"]))
                print(json.dumps({
                    "completed_slot": slot,
                    "pairing_status": paired_rows[-1]["pairing_status"],
                }, sort_keys=True), flush=True)

    summary = {
        "schema": "rles.route-a-isolated-first-trade-exit-report.v1",
        "protocol_id": doc["protocol_id"],
        "preregistration_sha256": prereg_sha,
        "feature_recipe_sha256": feature_sha,
        "source_snapshot_hash": doc["source"]["snapshot_hash"],
        "data_role": "CONSUMED_TRAIN_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "matrix": {
            "teachers": list(TEACHER_IDS),
            "windows": list(WINDOWS),
            "scenarios": list(SCENARIOS),
            "exits": list(EXITS),
            "seed": SEED,
            "paired_slots_expected": len(TEACHER_IDS) * len(WINDOWS) * len(SCENARIOS),
            "paired_slots_observed": len(paired_rows),
        },
        "pairing_counts": {
            status: sum(row["pairing_status"] == status for row in paired_rows)
            for status in ("PAIRED_SAME_NATIVE_ENTRY", "UNPAIRED")
        },
        "results": paired_rows,
        "economic_status": "SCENARIO_ONLY",
        "qualification": "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2",
        "interpretation_limit": (
            "First-trade matched development diagnostic only; it does not establish "
            "population causal effects, generalization, or profitability."
        ),
    }
    destination = output / "isolated-exit-report.json"
    destination.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(destination),
        "pairing_counts": summary["pairing_counts"],
        "qualification": summary["qualification"],
    }, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    run_matrix(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
