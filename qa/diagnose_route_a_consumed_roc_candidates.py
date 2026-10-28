"""Signal-count-only diagnostic for frozen ROC filters on consumed S07 rows.

This script never reads market archives and never replays an account. Its output
is development diagnostics, not candidate selection or economic evidence.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / ".artifacts" / "research_1m_4teachers_v1" / "native" / "teachers"
    / "T02_ROC_20_60" / "X1"
)
OUTPUT = ROOT / "docs" / "route_a_teacher_v2"
CANDIDATES = {
    "ROC2_CONFIRM_REARM2_V1": False,
    "ROC2_CONFIRM_ATR_MARGIN025_V1": True,
}


def _consensus(values: dict[str, Any]) -> int:
    roc20 = float(values["roc_20"])
    roc60 = float(values["roc_60"])
    if roc20 > 0 and roc60 > 0:
        return 1
    if roc20 < 0 and roc60 < 0:
        return -1
    return 0


def _candidate_count(
    rows: list[tuple[int, int, float]], *, require_margin: bool
) -> dict[str, Any]:
    armed = True
    run_side = 0
    run_length = 0
    neutral_length = 0
    events: list[int] = []
    margin_vetoes = 0

    for _timestamp, side, strength in rows:
        if side == 0:
            run_side = 0
            run_length = 0
            neutral_length += 1
            if neutral_length >= 2:
                armed = True
            continue

        neutral_length = 0
        if side == run_side:
            run_length += 1
        else:
            run_side = side
            run_length = 1

        if run_length == 2 and armed:
            if require_margin and strength < 0.25:
                margin_vetoes += 1
            else:
                events.append(side)
                armed = False

    return {
        "events": len(events),
        "long": sum(side > 0 for side in events),
        "short": sum(side < 0 for side in events),
        "mean_confirmation_delay_bars": 1.0 if events else None,
        "margin_vetoes": margin_vetoes,
    }


def run() -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for path in sorted(INPUT.glob("inner-screen-*/base/teacher-research-1m/decision-rows.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            raw_rows = list(csv.DictReader(stream))
        signals = Counter(row["raw_signal"] for row in raw_rows if row["bar_status"] == "READY")
        valid: list[tuple[int, int, float]] = []
        for row in raw_rows:
            if row["bar_status"] != "READY":
                continue
            values = json.loads(row["indicator_values_json"])
            side = _consensus(values)
            atr_ratio = float(values["atr_14"]) / float(values["bar_close_1m"])
            strength = (
                min(abs(float(values["roc_20"])), abs(float(values["roc_60"]))) / atr_ratio
                if atr_ratio > 0 else 0.0
            )
            valid.append((int(row["decision_ts_ns"]), side, strength))

        block_name = next(part for part in path.parts if part.startswith("inner-screen-"))
        blocks.append({
            "block": block_name,
            "ready_bars": len(valid),
            "baseline_entries": signals["LONG_ENTRY"] + signals["SHORT_ENTRY"],
            "baseline_long": signals["LONG_ENTRY"],
            "baseline_short": signals["SHORT_ENTRY"],
            "candidates": {
                candidate_id: _candidate_count(valid, require_margin=margin)
                for candidate_id, margin in CANDIDATES.items()
            },
        })

    if len(blocks) != 3:
        raise RuntimeError(f"expected 3 consumed Base blocks, found {len(blocks)}")
    totals: dict[str, Any] = {
        "ready_bars": sum(block["ready_bars"] for block in blocks),
        "baseline_entries": sum(block["baseline_entries"] for block in blocks),
        "candidates": {},
    }
    for candidate_id in CANDIDATES:
        totals["candidates"][candidate_id] = {
            key: sum(block["candidates"][candidate_id][key] for block in blocks)
            for key in ("events", "long", "short", "margin_vetoes")
        }
    result = {
        "scope": "consumed D1 T02 ROC/X1/Base signal-count diagnostic only",
        "economic_status": "D1_DIAGNOSTIC_ONLY",
        "market_archive_read": False,
        "account_replay_run": False,
        "blocks": blocks,
        "totals": totals,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "04_candidate_signal_diagnostics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    csv_path = OUTPUT / "04_candidate_signal_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "block", "ready_bars", "baseline_entries",
            "candidate_id", "events", "long", "short",
            "mean_confirmation_delay_bars", "margin_vetoes",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for block in blocks:
            for candidate_id in CANDIDATES:
                counts = block["candidates"][candidate_id]
                writer.writerow({
                    "block": block["block"],
                    "ready_bars": block["ready_bars"],
                    "baseline_entries": block["baseline_entries"],
                    "candidate_id": candidate_id,
                    **counts,
                })
    return result


if __name__ == "__main__":
    print(json.dumps(run()["totals"], indent=2, sort_keys=True))
