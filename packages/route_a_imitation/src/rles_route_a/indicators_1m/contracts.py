"""Immutable causal contracts for Route A one-minute indicators."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Literal


MINUTE_NS = 60_000_000_000
IndicatorStatus = Literal["READY", "NOT_READY"]


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class CausalBar:
    """One closed UTC-aligned OHLCV bar with an explicit availability clock."""

    open_time_ns: int
    close_time_ns: int
    available_at_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe_minutes: int = 1

    def __post_init__(self) -> None:
        if type(self.timeframe_minutes) is not int or self.timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be a positive integer")
        for name in ("open_time_ns", "close_time_ns", "available_at_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        span = self.timeframe_minutes * MINUTE_NS
        if self.close_time_ns - self.open_time_ns != span:
            raise ValueError("bar clock does not match its timeframe")
        if self.close_time_ns % span != 0:
            raise ValueError("bar close must be UTC epoch aligned")
        if self.available_at_ns < self.close_time_ns:
            raise ValueError("bar cannot be available before it closes")
        open_price = _finite(self.open, "open")
        high = _finite(self.high, "high")
        low = _finite(self.low, "low")
        close = _finite(self.close, "close")
        volume = _finite(self.volume, "volume")
        if min(open_price, high, low, close) <= 0 or volume < 0:
            raise ValueError("bar prices must be positive and volume non-negative")
        if high < max(open_price, close) or low > min(open_price, close) or high < low:
            raise ValueError("inconsistent OHLC range")


@dataclass(frozen=True, slots=True)
class IndicatorRecipe:
    """Frozen numerical and availability semantics for one indicator family."""

    recipe_id: str
    name: str
    parameters: tuple[tuple[str, float | int | str], ...]
    warmup_bars: int
    input_field_names: tuple[str, ...]
    output_field_names: tuple[str, ...]
    numeric_convention: str
    availability_semantics: str

    def __post_init__(self) -> None:
        if not self.recipe_id or not self.name:
            raise ValueError("indicator recipe id and name are required")
        if type(self.warmup_bars) is not int or self.warmup_bars <= 0:
            raise ValueError("indicator warmup must be positive")
        if (not self.input_field_names or not self.output_field_names
                or len(set(self.output_field_names)) != len(self.output_field_names)):
            raise ValueError("indicator input/output fields must be explicit and unique")
        if not self.numeric_convention or not self.availability_semantics:
            raise ValueError("indicator numerical and availability semantics are required")
        parameter_names = tuple(name for name, _value in self.parameters)
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("duplicate indicator parameter")


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Finite values visible after one completed one-minute bar."""

    close_time_ns: int
    available_at_ns: int
    decision_ts_ns: int
    sequence: int
    values: Mapping[str, float]
    recipe_status: Mapping[str, IndicatorStatus]

    def __post_init__(self) -> None:
        for name in ("close_time_ns", "available_at_ns", "decision_ts_ns", "sequence"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"invalid snapshot {name}")
        if not self.close_time_ns <= self.available_at_ns <= self.decision_ts_ns:
            raise ValueError("indicator snapshot violates availability ordering")
        if any(not isinstance(name, str) or not name for name in self.values):
            raise ValueError("invalid indicator field name")
        normalized_values = {
            name: _finite(value, f"indicator.{name}") for name, value in self.values.items()
        }
        normalized_status: dict[str, IndicatorStatus] = {}
        for name, status in self.recipe_status.items():
            if not isinstance(name, str) or not name or status not in ("READY", "NOT_READY"):
                raise ValueError("invalid indicator recipe status")
            normalized_status[name] = status
        object.__setattr__(self, "values", MappingProxyType(normalized_values))
        object.__setattr__(self, "recipe_status", MappingProxyType(normalized_status))

    def value(self, name: str) -> float:
        if name not in self.values:
            raise KeyError(f"indicator is not ready: {name}")
        return self.values[name]

