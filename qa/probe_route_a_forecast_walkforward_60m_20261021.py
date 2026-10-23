"""Run the frozen 60-minute non-overlap Route A Ridge validation matrix."""

from __future__ import annotations

import os

import probe_route_a_forecast_walkforward_20261021 as probe
import run_route_a_train_student_screen_20261020 as runner


RUN_ID = os.environ.get("ROUTE_A_WALKFORWARD_60M_RUN", "run04")
probe.OUT = runner.ROOT / f".artifacts/route-a-forecast-walkforward-20261021-{RUN_ID}"
probe.HORIZON_MINUTES = 60
probe.SAMPLE_MINUTES = 60

if __name__ == "__main__":
    raise SystemExit(probe.main())

