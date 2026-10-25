"""Audit frozen candidate coverage on Train-fit only, without reading screen.

The curated snapshot stores multi-day Parquet row groups, so this scope ends at
the last whole group before Train-inner screen. Candidate generation is clipped
to the preregistered Train-fit end; calibration rows are never scored here.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

from rles_data.core import parse_ns
from rles_orchestrator.research_access import read_registered_train_window


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "qa/run_route_a_teacher_e1_e2_g1_20261025.py"
OUTPUT = ROOT / ".artifacts/route-a-teacher-fit-block-coverage-audit-20261025"
ROW_GROUPS = (0, 1, 2, 3, 4, 5)


def _load_runner():
    spec = importlib.util.spec_from_file_location("route_a_teacher_fit_audit_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Route A frozen research runner cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable fit-block audit already exists: {OUTPUT}")
    runner = _load_runner()
    prereg_bytes = runner.PREREG.read_bytes()
    prereg = json.loads(prereg_bytes)
    prereg_sha = sha256(prereg_bytes).hexdigest()
    if runner.PREREG.with_suffix(runner.PREREG.suffix + ".sha256").read_text().strip() != prereg_sha:
        raise RuntimeError("frozen E1/E2/G1 preregistration checksum mismatch")
    marker_path = Path(str(runner.MARKET) + ".COMPLETED.json")
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    marker_sha = sha256(marker_bytes).hexdigest()
    if (marker.get("schema_version") != "0.4.1"
            or marker.get("snapshot_hash")
            != "9f07c4e0db1922b176c0f1186b511de8c9b41130ea1cbb3a69443234af708fab"):
        raise RuntimeError("fit audit source differs from the frozen Train snapshot")
    row_groups = marker.get("row_groups")
    if not isinstance(row_groups, list) or len(row_groups) <= max(ROW_GROUPS):
        raise RuntimeError("frozen snapshot lacks the required row-group metadata")
    selected = [row_groups[index] for index in ROW_GROUPS]
    scope_start = min(
        min(row["min_ts_event_ns"], row["min_available_at_ns"])
        for row in selected
    )
    scope_end = 1 + max(
        max(row["max_ts_event_ns"], row["max_available_at_ns"])
        for row in selected
    )
    screen_start = parse_ns(prereg["windows"]["train_inner_screen"][0])
    fit_end = parse_ns(prereg["windows"]["train_fit"][1])
    if scope_end >= screen_start:
        raise RuntimeError("fit-only diagnostic scope reaches the screen window")
    config = {
        "output": str(OUTPUT.resolve()),
        "environment": {
            "instrument": runner.INSTRUMENT,
            "market_path": str(runner.MARKET.resolve()),
            "start_ns": scope_start,
            "end_ns": scope_end,
            "curated_marker_sha256": marker_sha,
            "curated_read_mode": "row_group_scoped_v1",
        },
    }
    store, receipt = read_registered_train_window(
        config,
        {"start_ns": scope_start, "end_ns": scope_end},
        marker_sha256=marker_sha,
        label="train-route-a-fit-block-coverage-audit-20261025",
        return_receipt=True,
    )
    if receipt.registered_before_read is not True:
        raise RuntimeError("fit-block audit lacks a prior Train reservation")
    _, events, _, block_ids, _, *_ = runner._feature_events(
        store, marker_sha, prereg, decision_end_ns=fit_end,
    )
    configured_blocks = {
        block_id: interval
        for block_id, (split, *interval) in runner._block_table(prereg).items()
        if split == "train_fit"
    }
    counts = Counter(
        block_ids[event.event_id]
        for event in events
        if block_ids[event.event_id] in configured_blocks
    )
    result = {
        "schema": "rles.route-a.teacher-fit-block-coverage-audit.v1",
        "preregistration_sha256": prereg_sha,
        "source": {
            "snapshot_hash": store.snapshot_hash,
            "marker_sha256": marker_sha,
            "registered_before_read": receipt.registered_before_read,
            "row_groups": list(ROW_GROUPS),
            "scope_start_ns": scope_start,
            "scope_end_ns": scope_end,
            "scope_ends_before_screen": scope_end < screen_start,
        },
        "candidate_scan_end_ns": fit_end,
        "fit_block_candidate_counts": {
            block_id: counts[block_id] for block_id in configured_blocks
        },
        "candidate_total": sum(counts.values()),
        "interpretation": "ZERO_CANDIDATES_IN_BLOCK_IS_NOT_A_LABEL_OR_PURGE_LOSS",
        "screen_or_holdout_read": False,
    }
    if not result["source"]["scope_ends_before_screen"]:
        raise RuntimeError("fit-only diagnostic unexpectedly touches screen scope")
    OUTPUT.mkdir(parents=True)
    report_path = OUTPUT / "fit-block-coverage.json"
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    report_path.write_bytes(encoded)
    digest = sha256(encoded).hexdigest()
    report_path.with_suffix(report_path.suffix + ".sha256").write_text(digest, encoding="ascii")
    print(json.dumps({"report": str(report_path), "sha256": digest,
                      "fit_block_candidate_counts": result["fit_block_candidate_counts"],
                      "screen_or_holdout_read": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
