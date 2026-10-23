"""Repeat the frozen long-horizon BC screen with intent-aware inference."""

from __future__ import annotations

import run_route_a_long_horizon_student_screen_20261020 as screen
from rles_route_a.config import BCConfig as _BCConfig


screen.OUT = screen.runner.ROOT / ".artifacts" / "route-a-long-horizon-student-screen-20261020-run29"
screen.FIT = ("2024-02-01T00:00:00+00:00", "2024-02-10T00:00:00+00:00")
screen.FIT_STEPS = 12_000


def _intent_aware_config(**kwargs):
    return _BCConfig(
        **kwargs,
        intent_weight=1.0,
        inference_mode="intent_aware_v1",
        intent_class_weighting="inverse_frequency_v1",
    )


screen.BCConfig = _intent_aware_config


if __name__ == "__main__":
    raise SystemExit(screen.main())
