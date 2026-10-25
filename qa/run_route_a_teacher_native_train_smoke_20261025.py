"""Run the causal P0 teacher through a registered 35-hour native Train replay.

This is an engineering/accounting smoke with public historical market data and
scenario instrument/cost rules. It is never an economic qualification claim.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

import run_route_a_train_student_screen_20261020 as harness  # noqa: E402
from rles_orchestrator import application  # noqa: E402
from rles_route_a.teachers.causal_position import CausalPositionTeacher  # noqa: E402
from rles_route_a.teachers.recovery import RecoveryTeacher  # noqa: E402
from rles_route_a.teachers.selector import FrozenTeacherSelector  # noqa: E402
from rles_route_a.teachers.wait import WaitTeacher  # noqa: E402


MARKET = ROOT / "data/routea_teacher_train_20260812_20261002/market.parquet"
RULES = ROOT / ".artifacts/audit-development-20261016/data/scenario-rules.json"
CONFIG_TEMPLATE = ROOT / "configs/ethusdt_20240101_real.json"
OUTPUT = ROOT / ".artifacts/route-a-teacher-native-train-smoke-20261025-run03"
START = "2026-08-14T00:00:00+00:00"
END = "2026-08-15T12:00:00+00:00"
EPISODE_STEPS = 2100  # 35h; leaves a 1m host/factory terminal margin.
SEED = 20261025


def _ns(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("smoke bounds must carry a timezone")
    delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable native smoke output already exists: {OUTPUT}")
    marker_path = Path(str(MARKET) + ".COMPLETED.json")
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    marker_sha = sha256(marker_bytes).hexdigest()
    if marker.get("schema_version") != "0.4.1":
        raise RuntimeError("native Train smoke requires the curated 0.4.1 source")
    row_groups = marker.get("row_groups")
    if not isinstance(row_groups, list) or not row_groups:
        raise RuntimeError("native Train smoke requires row-group scope metadata")
    scope_start = min(
        min(row["min_ts_event_ns"], row["min_available_at_ns"])
        for row in row_groups
    )
    scope_end = 1 + max(
        max(row["max_ts_event_ns"], row["max_available_at_ns"])
        for row in row_groups
    )

    harness.MARKET = MARKET
    harness.MARKER = marker_path
    harness.IMPORT = MARKET.parent / "import.json"
    harness.RULES = RULES
    harness.CONFIG = CONFIG_TEMPLATE
    harness.OUT_BASE = OUTPUT
    harness.FEATURE_PROFILE = "causal_mtf_v7"
    harness.FOLD_ID = "route-a-teacher-native-smoke-2026q3-v1"
    config, factory = harness.make_config(
        label="causal-position-p0-35h-base",
        marker_sha=marker_sha,
        scope_start=scope_start,
        scope_end=scope_end,
        start_ns=_ns(START),
        end_ns=_ns(END),
        mode="base",
        fit_scaler=True,
    )
    teacher = FrozenTeacherSelector(
        primary=CausalPositionTeacher(),
        recovery=RecoveryTeacher(),
        wait=WaitTeacher(),
    )
    provider, grant = harness.make_provider(
        factory, teacher, marker_sha, "causal-position-p0-35h-base",
    )
    barriers, native_report = provider.rollout_with_m22_trace(
        factory=factory,
        host_rollout=application.rollout,
        grant=grant,
        seed=SEED,
        decision_policy=None,
        rollout_id="causal-position-p0-35h-base",
        episode_steps=EPISODE_STEPS,
        reset_options={"rles_episode_index": SEED},
    )
    metrics = harness.metric_row(
        barriers, dict(native_report), expected_source=marker_sha,
    )
    report_metrics = native_report.get("metrics")
    if not isinstance(report_metrics, dict):
        raise RuntimeError("native host report omitted account metrics")
    verification = native_report["route_a_native_order_verification"]
    max_position_age_ns = max((
        barrier.observation.account_value("position_duration_ns")
        for barrier in barriers
        if abs(barrier.observation.account_value("net_exposure")) > 1e-8
    ), default=0.0)
    funding = float(report_metrics.get("total_funding", 0.0))
    result = {
        "schema": "rles.route-a.teacher-native-train-smoke.v1",
        "source": {
            "snapshot_hash": marker.get("snapshot_hash"),
            "marker_sha256": marker_sha,
            "registered_train_scope": [scope_start, scope_end],
        },
        "window": {"start_utc": START, "end_utc": END,
                   "requested_barriers": EPISODE_STEPS,
                   "observed_barriers": len(barriers)},
        "teacher": "causal_position_p0_v2",
        "execution": {
            "economic_status": metrics["summary"].economic_status,
            "m22_authority": verification["authority"],
            "m22_verified_barriers": verification["barriers"],
            "filled_order_refs": metrics["filled_orders"],
            "liquidations": metrics["liquidations"],
            "max_native_position_age_seconds": max_position_age_ns / 1e9,
        },
        "account_metrics": {
            key: report_metrics.get(key)
            for key in ("net_return", "max_drawdown", "total_fee",
                        "total_slippage_cost", "total_funding", "turnover_ratio")
        },
        "funding_crossed_and_accounted": funding != 0.0,
        "qualification": "ENGINEERING_PASS_SCENARIO_ONLY_NOT_ALPHA_OR_ECONOMIC_AUTHORITY",
        "limitations": [
            "public Binance market archive plus scenario-only instrument/cost rules",
            "not historical account fees, margin rules, or actual exchange fills",
            "does not establish profitability or Alpha Teacher eligibility",
        ],
    }
    if (len(barriers) != EPISODE_STEPS
            or verification["barriers"] != len(barriers)
            or metrics["summary"].economic_status != "SCENARIO_ONLY"
            or metrics["liquidations"] != 0
            or not funding):
        raise RuntimeError("native Train smoke did not satisfy its declared checks")
    report_path = OUTPUT / "native-smoke.json"
    digest = harness.write_verified(report_path, result)
    print(json.dumps({"report": str(report_path), "sha256": digest,
                      "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
