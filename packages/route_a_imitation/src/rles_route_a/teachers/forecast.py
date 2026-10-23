"""Frozen causal forecast teacher with an explicit net-cost entry hurdle."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite

from rles_route_a.contracts import ExpertAdvice, PointInTimeObservation
from .protocol import advice, classify_target


@dataclass(frozen=True)
class RidgeForecastTeacher:
    """Apply a Train-fitted ridge return forecast to the current causal features.

    Coefficients and scaling statistics are immutable recipe inputs. The model
    predicts forward log-return basis points; ``entry_hurdle_bps`` must include
    the conservative round-trip scenario cost before an opening command is
    issued. This object never reads market data or future labels.
    """

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    coefficients: tuple[float, ...]
    entry_hurdle_bps: float = 30.0
    exit_hurdle_bps: float = 15.0
    size: float = 0.2
    forecast_horizon_minutes: int = 60
    decision_interval_minutes: int = 0
    expert_id: str = "ridge_forecast_net_cost_v1"

    def __post_init__(self) -> None:
        count = len(self.feature_names)
        values = (*self.means, *self.stds, *self.coefficients,
                  self.entry_hurdle_bps, self.exit_hurdle_bps, self.size)
        if (count == 0 or len(set(self.feature_names)) != count
                or any(not isinstance(name, str) or not name.strip() for name in self.feature_names)
                or len(self.means) != count or len(self.stds) != count
                or len(self.coefficients) != count + 1
                or not all(isfinite(value) for value in values)
                or any(value <= 0 for value in self.stds)
                or self.entry_hurdle_bps <= 0
                or not 0 <= self.exit_hurdle_bps < self.entry_hurdle_bps
                or not 0 < self.size <= 1
                or type(self.forecast_horizon_minutes) is not int
                or self.forecast_horizon_minutes <= 0
                or type(self.decision_interval_minutes) is not int
                or self.decision_interval_minutes < 0
                or not isinstance(self.expert_id, str) or not self.expert_id.strip()):
            raise ValueError("invalid frozen ridge forecast teacher recipe")

    def predict_bps(self, observation: PointInTimeObservation) -> float:
        standardized = tuple(
            (observation.market_value(name) - mean) / std
            for name, mean, std in zip(self.feature_names, self.means, self.stds)
        )
        prediction = self.coefficients[0] + fsum(
            coefficient * value
            for coefficient, value in zip(self.coefficients[1:], standardized)
        )
        if not isfinite(prediction):
            raise ValueError("non-finite forecast teacher prediction")
        return prediction

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        current = observation.account_value("net_exposure")
        if self.decision_interval_minutes:
            decision_minute = observation.decision_ts_ns // 60_000_000_000
            if decision_minute % self.decision_interval_minutes:
                return advice(
                    observation, self.expert_id,
                    classify_target(current, current), current,
                    "FORECAST_WAIT_FOR_DECISION_INTERVAL",
                )
        prediction = self.predict_bps(observation)
        if prediction >= self.entry_hurdle_bps:
            target, reason = self.size, "FORECAST_ABOVE_COST_HURDLE_LONG"
        elif prediction <= -self.entry_hurdle_bps:
            target, reason = -self.size, "FORECAST_ABOVE_COST_HURDLE_SHORT"
        elif (current > 1e-8 and prediction >= self.exit_hurdle_bps):
            target, reason = current, "FORECAST_HOLDS_LONG"
        elif (current < -1e-8 and prediction <= -self.exit_hurdle_bps):
            target, reason = current, "FORECAST_HOLDS_SHORT"
        elif abs(current) > 1e-8:
            target, reason = 0.0, "FORECAST_BELOW_EXIT_HURDLE"
        else:
            target, reason = 0.0, "FORECAST_BELOW_ENTRY_HURDLE"
        return advice(observation, self.expert_id, classify_target(current, target), target, reason)
