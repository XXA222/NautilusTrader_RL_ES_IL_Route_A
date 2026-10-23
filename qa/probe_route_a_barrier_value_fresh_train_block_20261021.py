"""Run a single frozen barrier-value candidate on unused Train-inner dates."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs" / "route_a_barrier_value_may28_jun03_prereg_20261021.json"
PREREG_SHA = PREREG.with_suffix(PREREG.suffix + ".sha256")
RUN_ID = "run04"


def main() -> int:
    prereg_bytes = PREREG.read_bytes()
    prereg_sha = sha256(prereg_bytes).hexdigest()
    if PREREG_SHA.read_text(encoding="ascii").strip() != prereg_sha:
        raise RuntimeError("frozen Train-inner preregistration SHA mismatch")
    prereg = json.loads(prereg_bytes)
    if prereg.get("schema") != "rles.route-a.barrier-value.train-inner-prereg.v1":
        raise RuntimeError("unexpected Train-inner preregistration schema")

    os.environ["ROUTE_A_BARRIER_VALUE_RUN"] = RUN_ID
    os.environ["ROUTE_A_VALUE_EVAL_STEP_MINUTES"] = "5"
    os.environ["ROUTE_A_VALUE_FEATURE_END_UTC"] = prereg["evaluation"]["end_utc_exclusive"]
    import probe_route_a_barrier_value_teacher_20261021 as experiment
    import probe_route_a_temporal_event_classifier_teacher_20261021 as temporal

    fit = prereg["fit"]
    windows = prereg["evaluation"]["subperiods_utc"]
    temporal.FOLDS = tuple(
        (
            f"subperiod{index:02d}",
            fit["start_utc"],
            fit["end_utc_exclusive"],
            start,
            end,
        )
        for index, (start, end) in enumerate(windows, start=1)
    )
    result = experiment.main()
    if result != 0:
        return result

    report_path = experiment.OUT / "barrier-value-teacher-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if len(report.get("folds", [])) != len(windows):
        raise RuntimeError("experiment report does not cover every preregistered subperiod")
    primary = float(prereg["evaluation"]["primary_minimum_predicted_net_value_bps"])
    rows = []
    for fold in report["folds"]:
        candidates = [
            row for row in fold["evaluation"]["thresholds"]
            if float(row["minimum_predicted_net_value_bps"]) == primary
        ]
        if len(candidates) != 1:
            raise RuntimeError("primary threshold is missing or duplicated in a fold")
        rows.append(candidates[0])

    total = sum(int(row["trade_count"]) for row in rows)
    stress_sum = sum(
        float(row["net_mean_bps"]) * int(row["trade_count"])
        for row in rows if row["trade_count"]
    )
    stress_mean = stress_sum / total if total else None
    delta = (
        float(prereg["path_and_cost_assumptions"]["stress_round_trip_cost_bps"])
        - float(prereg["path_and_cost_assumptions"]["base_same_trade_cost_sensitivity_bps"])
    )
    subperiod_means = [row["net_mean_bps"] for row in rows]
    screen_pass = bool(
        total >= int(prereg["evaluation"]["minimum_trade_count_for_screen"])
        and stress_mean is not None
        and stress_mean > 0.0
        and all(value is not None and value > 0.0 for value in subperiod_means)
    )
    report["frozen_preregistration"] = {
        "path": PREREG.relative_to(ROOT).as_posix(),
        "sha256": prereg_sha,
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_threshold_bps": primary,
        "screen_rule": prereg["screen_rule"],
        "screen_pass": screen_pass,
        "screen_evaluation": {
            "trade_count": total,
            "minimum_trade_count": int(prereg["evaluation"]["minimum_trade_count_for_screen"]),
            "stress_net_mean_bps": stress_mean,
            "base_same_trade_sensitivity_net_mean_bps": (
                stress_mean + delta if stress_mean is not None else None
            ),
            "subperiod_stress_net_means_bps": subperiod_means,
            "subperiods_each_positive": all(
                value is not None and value > 0.0 for value in subperiod_means
            ),
        },
        "base_sensitivity_cost_delta_bps_per_trade": delta,
    }
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_sha = sha256(report_path.read_bytes()).hexdigest()
    report_path.with_suffix(report_path.suffix + ".sha256").write_text(
        report_sha, encoding="ascii"
    )
    print(json.dumps({
        "report": str(report_path),
        "report_sha256": report_sha,
        "preregistration_sha256": prereg_sha,
        **report["frozen_preregistration"]["screen_evaluation"],
        "screen_pass": screen_pass,
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
