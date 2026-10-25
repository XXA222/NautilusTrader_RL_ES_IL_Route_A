"""Deterministic causal teachers. Parameters are engineering candidates, not alpha claims."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class FrozenVisibleContext:
    """Raw L2 features and clock copied from one completed, visible barrier."""

    decision_ts_ns: int
    market_features: Mapping[str, float]
    account_features: Mapping[str, float] = field(default_factory=dict)
    pending_order_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.decision_ts_ns) is not int or self.decision_ts_ns <= 0:
            raise ValueError("positive decision timestamp required")
        values, account = dict(self.market_features), dict(self.account_features)
        if (any(type(k) is not str or not k or not np.isfinite(v) for k, v in values.items())
                or any(type(k) is not str or not k or not np.isfinite(v)
                       for k, v in account.items())):
            raise ValueError("finite named visible market/account features required")
        if (type(self.pending_order_ids) is not tuple
                or len(self.pending_order_ids) != len(set(self.pending_order_ids))
                or any(type(value) is not str or not value.strip()
                       for value in self.pending_order_ids)):
            raise ValueError("unique native pending order identities required")
        object.__setattr__(self, "market_features", MappingProxyType(values))
        object.__setattr__(self, "account_features", MappingProxyType(account))
        object.__setattr__(self, "pending_order_ids", tuple(self.pending_order_ids))

    def account_value(self, name: str) -> float:
        if name not in self.account_features:
            raise ValueError(f"required native account feature is missing: {name}")
        value = float(self.account_features[name])
        if not np.isfinite(value):
            raise ValueError(f"native account feature is non-finite: {name}")
        return value


class ExpertPolicy(Protocol):
    expert_id: str

    def reset(self, episode_metadata: Mapping[str, object]) -> None: ...

    def act(self, *, observation: np.ndarray, context: FrozenVisibleContext) -> float: ...


@dataclass(frozen=True, slots=True)
class E1Config:
    enter_return_15m: float = 0.0
    enter_return_60m: float = 0.0
    enter_efficiency_60m: float = 0.35
    exit_efficiency_60m: float = 0.10
    target_exposure: float = 0.5
    cooldown_steps: int = 15

    def __post_init__(self) -> None:
        if (not all(np.isfinite(v) for v in (
            self.enter_return_15m, self.enter_return_60m, self.enter_efficiency_60m,
            self.exit_efficiency_60m, self.target_exposure,
        )) or not 0 < self.target_exposure <= 1 or self.cooldown_steps < 0):
            raise ValueError("invalid E1 frozen parameters")


class E1TrendExpert:
    """Trend/hold teacher using only the frozen causal 15m/60m feature view."""

    expert_id = "trend_v1"

    def __init__(self, config: E1Config = E1Config()) -> None:
        self.config = config
        self._position = 0
        self._last_change = -10**18
        self._steps = 0

    def reset(self, episode_metadata: Mapping[str, object]) -> None:
        if not isinstance(episode_metadata, Mapping):
            raise TypeError("episode metadata mapping required")
        self._position, self._steps = 0, 0
        self._last_change = -10**18

    def act(self, *, observation: np.ndarray, context: FrozenVisibleContext) -> float:
        if not isinstance(context, FrozenVisibleContext):
            raise TypeError("frozen visible context required")
        if np.asarray(observation).ndim != 1 or not np.isfinite(observation).all():
            raise ValueError("finite one-dimensional observation required")
        feature = context.market_features
        required = {"return_15m", "return_60m", "trend_efficiency_60m"}
        if not required <= set(feature):
            raise ValueError("E1 requires causal_mtf_v2 raw features")
        ret15, ret60, efficiency = (feature[k] for k in (
            "return_15m", "return_60m", "trend_efficiency_60m",
        ))
        signal = 1 if ret15 > self.config.enter_return_15m and ret60 > self.config.enter_return_60m else (
            -1 if ret15 < -self.config.enter_return_15m and ret60 < -self.config.enter_return_60m else 0
        )
        target = self._position
        if self._position == 0:
            if signal and efficiency >= self.config.enter_efficiency_60m:
                target = signal
        elif signal == -self._position:
            target = 0  # one flat barrier before considering a reversal
        elif signal == 0 or efficiency < self.config.exit_efficiency_60m:
            target = 0
        if (target != self._position and target != 0
                and self._steps - self._last_change < self.config.cooldown_steps):
            target = self._position
        if target != self._position:
            self._position, self._last_change = target, self._steps
        self._steps += 1
        return float(self._position * self.config.target_exposure)


class FlatExpert:
    """Constant-zero target baseline; useful for plumbing and cost diagnostics."""

    expert_id = "flat_v1"

    def reset(self, episode_metadata: Mapping[str, object]) -> None:
        if not isinstance(episode_metadata, Mapping):
            raise TypeError("episode metadata mapping required")

    def act(self, *, observation: np.ndarray, context: FrozenVisibleContext) -> float:
        if np.asarray(observation).ndim != 1 or not np.isfinite(observation).all():
            raise ValueError("finite one-dimensional observation required")
        return 0.0


@dataclass(frozen=True, slots=True)
class E2Config:
    entry_z: float = 1.5
    exit_z: float = 0.3
    target_exposure: float = 0.4
    max_holding_minutes: int = 240
    lookback_minutes: int = 60

    def __post_init__(self) -> None:
        if (not np.isfinite(self.entry_z) or not np.isfinite(self.exit_z)
                or not np.isfinite(self.target_exposure) or not self.entry_z > self.exit_z >= 0
                or not 0 < self.target_exposure <= 1 or type(self.max_holding_minutes) is not int
                or self.max_holding_minutes <= 0 or type(self.lookback_minutes) is not int
                or self.lookback_minutes < 2):
            raise ValueError("invalid E2 frozen reversion parameters")


class E2ReversionExpert:
    """Causal rolling-close mean reversion with trend veto and a fixed time exit."""

    expert_id = "reversion_v1"
    _minute_ns = 60_000_000_000

    def __init__(self, config: E2Config = E2Config()) -> None:
        self.config = config
        self._closes: list[float] = []
        self._last_ts_ns: int | None = None
        self._position = 0
        self._entry_ts_ns: int | None = None

    def reset(self, episode_metadata: Mapping[str, object]) -> None:
        if not isinstance(episode_metadata, Mapping):
            raise TypeError("episode metadata mapping required")
        self._closes.clear()
        self._last_ts_ns = None
        self._position = 0
        self._entry_ts_ns = None

    def act(self, *, observation: np.ndarray, context: FrozenVisibleContext) -> float:
        if np.asarray(observation).ndim != 1 or not np.isfinite(observation).all():
            raise ValueError("finite one-dimensional observation required")
        features = context.market_features
        required = {"close_1m", "return_15m", "return_60m"}
        if not required <= set(features):
            raise ValueError("E2 requires causal_mtf_v4 close_1m and trend features")
        close = float(features["close_1m"])
        if not np.isfinite(close) or close <= 0:
            raise ValueError("E2 requires a finite positive closed 1m price")
        if self._last_ts_ns is not None and context.decision_ts_ns - self._last_ts_ns != self._minute_ns:
            self._closes.clear()
            self._position = 0
            self._entry_ts_ns = None
        self._last_ts_ns = context.decision_ts_ns
        self._closes.append(close)
        self._closes = self._closes[-self.config.lookback_minutes:]
        if len(self._closes) < self.config.lookback_minutes:
            self._position = 0
            self._entry_ts_ns = None
            return 0.0
        values = np.asarray(self._closes, dtype=np.float64)
        deviation = float(values.std(ddof=0))
        if deviation == 0.0 or not np.isfinite(deviation):
            self._position = 0
            self._entry_ts_ns = None
            return 0.0
        z_score = (close - float(values.mean())) / deviation
        ret15, ret60 = float(features["return_15m"]), float(features["return_60m"])
        if self._position:
            if (abs(z_score) <= self.config.exit_z or self._entry_ts_ns is None
                    or context.decision_ts_ns - self._entry_ts_ns
                    >= self.config.max_holding_minutes * self._minute_ns):
                self._position = 0
                self._entry_ts_ns = None
        elif z_score >= self.config.entry_z and not (ret15 > 0 and ret60 > 0):
            self._position = -1
            self._entry_ts_ns = context.decision_ts_ns
        elif z_score <= -self.config.entry_z and not (ret15 < 0 and ret60 < 0):
            self._position = 1
            self._entry_ts_ns = context.decision_ts_ns
        return float(self._position * self.config.target_exposure)


@dataclass(frozen=True, slots=True)
class E3Config:
    max_volatility_60m: float
    deadband: float = 0.05
    trend: E1Config = E1Config()

    def __post_init__(self) -> None:
        if (not np.isfinite(self.max_volatility_60m) or self.max_volatility_60m <= 0
                or not np.isfinite(self.deadband) or not 0 <= self.deadband < 1):
            raise ValueError("invalid E3 volatility threshold or target deadband")


class E3CostAwareExpert:
    """Volatility veto around E1 to supply cost/observation behavior examples."""

    expert_id = "cost_guard_v1"

    def __init__(self, config: E3Config) -> None:
        self.config = config
        self._trend = E1TrendExpert(config.trend)
        self._last_action = 0.0

    def reset(self, episode_metadata: Mapping[str, object]) -> None:
        self._trend.reset(episode_metadata)
        self._last_action = 0.0

    def act(self, *, observation: np.ndarray, context: FrozenVisibleContext) -> float:
        if "volatility_60m" not in context.market_features:
            raise ValueError("E3 requires causal_mtf_v2 volatility_60m")
        volatility = float(context.market_features["volatility_60m"])
        if not np.isfinite(volatility) or volatility < 0:
            raise ValueError("E3 volatility must be finite and nonnegative")
        target = (0.0 if volatility >= self.config.max_volatility_60m else
                  self._trend.act(observation=observation, context=context))
        if abs(target - self._last_action) < self.config.deadband:
            target = self._last_action
        self._last_action = target
        return float(target)


class RecoveryExpert:
    """T3 native account risk overlay with drawdown, stop-loss and size exits."""

    expert_id = "recovery_v1"

    def __init__(self, *, max_abs_exposure: float = 0.8,
                 stop_loss_fraction: float = 0.04, max_drawdown: float = 0.15) -> None:
        values = (max_abs_exposure, stop_loss_fraction, max_drawdown)
        if (not all(np.isfinite(value) for value in values)
                or not 0 < max_abs_exposure <= 1
                or not 0 < stop_loss_fraction < 1
                or not 0 < max_drawdown < 1):
            raise ValueError("invalid T3 native risk limits")
        self.max_abs_exposure = float(max_abs_exposure)
        self.stop_loss_fraction = float(stop_loss_fraction)
        self.max_drawdown = float(max_drawdown)

    def reset(self, episode_metadata: Mapping[str, object]) -> None:
        if not isinstance(episode_metadata, Mapping):
            raise TypeError("episode metadata mapping required")

    def act(self, *, observation: np.ndarray, context: FrozenVisibleContext) -> float:
        if np.asarray(observation).ndim != 1 or not np.isfinite(observation).all():
            raise ValueError("finite one-dimensional observation required")
        current = context.account_value("net_exposure")
        drawdown = context.account_value("drawdown")
        unrealized = context.account_value("unrealized_pnl_pct")
        if abs(current) > 1.0 + 1e-9 or not 0 <= drawdown < 1.0:
            raise ValueError("native account exposure or drawdown is outside its bounded domain")
        if drawdown >= self.max_drawdown or (abs(current) > 1e-8
                                               and unrealized <= -self.stop_loss_fraction):
            return 0.0
        if abs(current) > self.max_abs_exposure:
            return float(np.copysign(self.max_abs_exposure, current))
        return current


class FixedExposureExpert:
    """Frozen fixed long/short benchmark entered at a predeclared decision time."""

    def __init__(self, *, target_exposure: float, entry_ts_ns: int) -> None:
        if (not np.isfinite(target_exposure) or target_exposure == 0 or abs(target_exposure) > 1
                or type(entry_ts_ns) is not int or entry_ts_ns <= 0):
            raise ValueError("fixed exposure baseline needs a bounded nonzero target and entry clock")
        self.target_exposure = float(target_exposure)
        self.entry_ts_ns = entry_ts_ns
        self.expert_id = "fixed_long_v1" if target_exposure > 0 else "fixed_short_v1"

    def reset(self, episode_metadata: Mapping[str, object]) -> None:
        if not isinstance(episode_metadata, Mapping):
            raise TypeError("episode metadata mapping required")

    def act(self, *, observation: np.ndarray, context: FrozenVisibleContext) -> float:
        if np.asarray(observation).ndim != 1 or not np.isfinite(observation).all():
            raise ValueError("finite one-dimensional observation required")
        return self.target_exposure if context.decision_ts_ns >= self.entry_ts_ns else 0.0
