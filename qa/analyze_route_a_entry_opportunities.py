"""Analyze retained D1 native decision/trade artifacts; no market input."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

from rles_route_a.teacher_research.entry_opportunity_audit import audit_entry_opportunities

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / ".artifacts/research_1m_4teachers_v1/native/teachers"
TRADES = ROOT / ".artifacts/route-a-exit-economics-task3/per_trade_exit_metrics.jsonl"
OUTPUT = ROOT / ".artifacts/route-a-entry-opportunity-audit-20261029-v2"
TEACHERS = ("T01_EMA_9_21", "T02_ROC_20_60", "T03_DONCHIAN_20", "T04_ZSCORE_60")
EXITS = ("X1", "X2")
BLOCKS = ("inner-screen-1", "inner-screen-2", "inner-screen-3")
SCENARIOS = ("base", "stress")


def analyze() -> dict[str, Any]:
    trade_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    expected = {(teacher, variant, block, scenario)
                for teacher in TEACHERS for variant in EXITS
                for block in BLOCKS for scenario in SCENARIOS}
    with TRADES.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["teacher_id"], row["exit_variant"], row["block"], row["scenario"])
            if key not in expected or row["economic_status"] != "SCENARIO_ONLY":
                raise ValueError("trade outside the recorded D1 matrix")
            trade_groups[key].append(row)
    rows = []
    for teacher, variant, block, scenario in sorted(expected):
        key = (teacher, variant, block, scenario)
        path = NATIVE / teacher / variant / block / scenario / "teacher-research-1m/decision-rows.csv"
        if not path.is_file() or key not in trade_groups:
            raise FileNotFoundError(f"incomplete retained native D1 run: {key}")
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            def validated():
                for row in reader:
                    if (row["teacher_id"] != teacher or row["exit_policy_id"] != variant
                            or row["scenario_id"] != scenario):
                        raise ValueError(f"decision identity differs from D1 slot: {key}")
                    yield row
            result = audit_entry_opportunities(validated(), trade_groups[key])
        rows.append({"teacher": teacher, "exit": variant, "block": block,
                     "scenario": scenario, **result})
    return {
        "schema": "rles.route-a.entry-opportunity-audit.v1",
        "classification": "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_SCENARIO_ONLY",
        "market_payload_read": False,
        "scope": "previously consumed native decision and trade artifacts",
        "warning": ("Descriptive diagnosis only; horizons and MFE are not executable "
                    "profit. Results do not prove predictive edge or independent OOS."),
        "runs": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# D1 entry opportunity audit",
        "",
        "DEVELOPMENT_DIAGNOSTIC_ONLY / SCENARIO_ONLY; consumed Train data.",
        "Close-to-close horizon movements and MFE/MAE are not executable profits.",
        "These descriptive results do not establish predictive advantage, causal cost attribution or OOS.",
        "",
        "| Teacher | Exit | Block | Scenario | Side | Candidates | Closed | Censored | Gross price USDT | Net USDT | Fees | Slippage | Funding | MFE bp | MAE bp | 5m bp (n/c) | 15m bp (n/c) | 60m bp (n/c) | 120m bp (n/c) |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    def num(value: float | None) -> str:
        return "NA" if value is None else f"{value:.3f}"
    for row in report["runs"]:
        for side in ("long", "short"):
            g = row[side]
            horizons = []
            for h in (5, 15, 60, 120):
                item = g["horizons"][str(h)]
                horizons.append(f"{num(item['mean_directional_bps'])} ({item['observed']}/{item['censored']})")
            cols = [row["teacher"], row["exit"], row["block"], row["scenario"], side,
                    str(g["candidate_count"]), str(g["completed_trades"]), str(g["censored_trades"]),
                    num(g["gross_price_cash"]), num(g["net_cash"]), num(g["fees"]),
                    num(g["slippage_attribution"]), num(g["funding"]),
                    num(g["mean_mfe_bps"]), num(g["mean_mae_bps"]), *horizons]
            lines.append("| " + " | ".join(cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("immutable D1 analysis output already exists")
    report = analyze()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT / "summary.json"),
                      "run_count": len(report["runs"]),
                      "classification": report["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
