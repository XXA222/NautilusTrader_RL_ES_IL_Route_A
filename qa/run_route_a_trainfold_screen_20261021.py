"""Screen Route A using only the previously frozen 2024 WFO Train segment.

The source snapshot is rebuilt from checksum-verified daily Binance archives
for 2023-12-30 through 2024-06-14, then clipped at the frozen Train score end.
Data before 2024-04-16 is causal history and feature warm-up only. Fit and
screen windows are bounded by the frozen WFO fold's Train score interval;
Validation, Test, and Holdout rows are absent.
"""

from __future__ import annotations

import json
from hashlib import sha256
import os
from pathlib import Path
import re

import run_route_a_multiepisode_student_screen_20261020 as screen
import run_route_a_train_student_screen_20261020 as runner


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "routea_train_20231230_20240614"
WFO_REPORT = Path(
    r"D:\Program Files\NautilusTrader_RL_ES\.tmp\real-ethusdt-scenario"
    r"\wfo-fresh-apr-jul-v1-run2\wfo-report.json"
)
FIT_DAYS = (
    "2024-04-16", "2024-04-20", "2024-04-24", "2024-04-28",
    "2024-05-02", "2024-05-06", "2024-05-10", "2024-05-14",
)
SCREEN_WINDOWS = (
    ("jun03_h00", "2024-06-03T00:00:00+00:00", "2024-06-03T12:00:00+00:00"),
    ("jun10_h00", "2024-06-10T00:00:00+00:00", "2024-06-10T12:00:00+00:00"),
)
RUN_ID = os.environ.get("ROUTE_A_EXPERIMENT_RUN", "run59")


def configure() -> None:
    if not re.fullmatch(r"run[0-9]{2,3}", RUN_ID):
        raise ValueError("Train-fold run ID must be runNN or runNNN")
    if not WFO_REPORT.is_file():
        raise FileNotFoundError(f"frozen WFO Train manifest missing: {WFO_REPORT}")
    report = json.loads(WFO_REPORT.read_text(encoding="utf-8"))
    manifests = report.get("fold_manifests")
    if not isinstance(manifests, list) or len(manifests) != 1:
        raise RuntimeError("expected the audited single-fold Apr-Jul WFO manifest")
    train = manifests[0].get("train")
    if not isinstance(train, dict):
        raise RuntimeError("frozen WFO manifest has no Train interval")
    score_start = train.get("score_start_ns")
    score_end = train.get("score_end_ns")
    if (score_start != runner.ns("2024-04-16T00:00:00+00:00")
            or score_end != runner.ns("2024-06-15T00:00:00+00:00")):
        raise RuntimeError("frozen WFO Train interval differs from the reviewed boundary")

    marker = SOURCE_DIR / "market.parquet.COMPLETED.json"
    imported = SOURCE_DIR / "import.json"
    if not marker.is_file() or not imported.is_file():
        raise FileNotFoundError("Train-only curated source is incomplete")
    import_record = json.loads(imported.read_text(encoding="utf-8"))
    if (import_record.get("schema") != "rles.route-a-train-snapshot.v1"
            or import_record.get("start_ns", score_start) >= score_start
            or import_record.get("end_ns") != score_end):
        raise PermissionError("curated Train source crosses the frozen Train fold boundary")
    for day in FIT_DAYS:
        start = runner.ns(f"{day}T00:00:00+00:00")
        end = runner.ns(f"{day}T23:59:00+00:00")
        if start < score_start or end >= score_end:
            raise PermissionError(f"fit date lies outside the frozen Train score interval: {day}")
    for window_id, start_text, end_text in SCREEN_WINDOWS:
        start, end = runner.ns(start_text), runner.ns(end_text)
        if start < score_start or end > score_end:
            raise PermissionError(f"screen window lies outside the frozen Train interval: {window_id}")

    runner.DATA_DIR = SOURCE_DIR
    runner.MARKET = SOURCE_DIR / "market.parquet"
    runner.MARKER = marker
    runner.IMPORT = imported
    screen.EXPERIMENT_RUN = RUN_ID
    screen.TEACHER_MODE = "stop_cooldown_5m_v1"
    screen.STOP_LOSS_FRACTION = 0.001
    screen.STOP_COOLDOWN_NS = 300_000_000_000
    screen.INTENT_CLASS_WEIGHTING = "none"
    screen.INTENT_SAMPLING = "sqrt_balanced_intent_v1"
    screen.FIT_DAYS = FIT_DAYS
    screen.SCREEN_WINDOWS = SCREEN_WINDOWS
    screen.FIT_EPISODE_STEPS = 350
    screen.EPISODE_STEPS = 700
    screen.DAGGER_ROLLOUTS = True
    screen.DAGGER_SEED = int(os.environ.get("ROUTE_A_DAGGER_SEED", "101"))
    screen.OUT = ROOT / ".artifacts" / f"route-a-multiepisode-student-screen-20261021-{RUN_ID}"


if __name__ == "__main__":
    configure()
    result = screen.main()
    if result != 0:
        raise SystemExit(result)
    wfo = json.loads(WFO_REPORT.read_text(encoding="utf-8"))
    train = wfo["fold_manifests"][0]["train"]
    output = screen.OUT / "screen-report.json"
    report = json.loads(output.read_text(encoding="utf-8"))
    report["trainfold_provenance"] = {
        "wfo_protocol_id": wfo["protocol_id"],
        "wfo_report_sha256": sha256(WFO_REPORT.read_bytes()).hexdigest(),
        "fold_id": "60-15-15-0000",
        "warmup_start_ns": train["warmup_start_ns"],
        "score_start_ns": train["score_start_ns"],
        "score_end_ns_exclusive": train["score_end_ns"],
        "curated_import_sha256": sha256(runner.IMPORT.read_bytes()).hexdigest(),
        "curated_marker_sha256": sha256(runner.MARKER.read_bytes()).hexdigest(),
        "fit_days": list(FIT_DAYS),
        "screen_window_ids": [row[0] for row in SCREEN_WINDOWS],
        "split_policy": "Train-only; pre-Apr16 source rows provide causal history and warm-up",
    }
    runner.write_verified(output, report)
