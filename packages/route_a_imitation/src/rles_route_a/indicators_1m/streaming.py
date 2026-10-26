"""Single-pass causal 1m indicator calculation with bounded state."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import fsum, sqrt

from .contracts import CausalBar, IndicatorSnapshot, IndicatorStatus, MINUTE_NS
from .recipes import ROUTE_A_1M_INDICATOR_RECIPES_V1


@dataclass(slots=True)
class _Ema:
    period: int
    _seed: list[float] = field(default_factory=list)
    value: float | None = None

    def update(self, value: float) -> float | None:
        if self.value is None:
            self._seed.append(value)
            if len(self._seed) < self.period:
                return None
            self.value = fsum(self._seed) / self.period
            self._seed.clear()
            return self.value
        alpha = 2.0 / (self.period + 1.0)
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


@dataclass(slots=True)
class StreamingIndicators1m:
    """Advance every frozen indicator exactly once per contiguous closed bar."""

    _bars: deque[CausalBar] = field(default_factory=lambda: deque(maxlen=61))
    _ema9: _Ema = field(default_factory=lambda: _Ema(9))
    _ema21: _Ema = field(default_factory=lambda: _Ema(21))
    _ema12: _Ema = field(default_factory=lambda: _Ema(12))
    _ema26: _Ema = field(default_factory=lambda: _Ema(26))
    _macd_signal: _Ema = field(default_factory=lambda: _Ema(9))
    _rsi_prev_close: float | None = None
    _rsi_seed_gains: list[float] = field(default_factory=list)
    _rsi_seed_losses: list[float] = field(default_factory=list)
    _rsi_avg_gain: float | None = None
    _rsi_avg_loss: float | None = None
    _atr_prev_close: float | None = None
    _atr_seed: list[float] = field(default_factory=list)
    _atr_value: float | None = None
    _sequence: int = 0
    _last_bar: CausalBar | None = None
    _last_snapshot: IndicatorSnapshot | None = None

    def update(self, bar: CausalBar, *, decision_ts_ns: int) -> IndicatorSnapshot:
        if not isinstance(bar, CausalBar) or bar.timeframe_minutes != 1:
            raise TypeError("StreamingIndicators1m requires a CausalBar with timeframe=1")
        if type(decision_ts_ns) is not int or decision_ts_ns <= 0:
            raise ValueError("decision clock must be a positive integer")
        if bar.available_at_ns > decision_ts_ns:
            raise ValueError("1m bar is not yet available at the decision clock")
        if self._last_bar is not None:
            if bar.close_time_ns == self._last_bar.close_time_ns:
                if bar == self._last_bar and self._last_snapshot is not None:
                    if decision_ts_ns != self._last_snapshot.decision_ts_ns:
                        raise ValueError("duplicate bar replay changed its decision clock")
                    return self._last_snapshot
                raise ValueError("duplicate 1m close timestamp has conflicting facts")
            if bar.close_time_ns < self._last_bar.close_time_ns:
                raise ValueError("out-of-order 1m bar")
            if bar.close_time_ns - self._last_bar.close_time_ns != MINUTE_NS:
                raise ValueError("1m history contains a gap")

        prior_bars = tuple(self._bars)
        values: dict[str, float] = {
            "bar_open_1m": float(bar.open),
            "bar_high_1m": float(bar.high),
            "bar_low_1m": float(bar.low),
            "bar_close_1m": float(bar.close),
            "bar_volume_1m": float(bar.volume),
        }

        ema9 = self._ema9.update(bar.close)
        ema21 = self._ema21.update(bar.close)
        ema12 = self._ema12.update(bar.close)
        ema26 = self._ema26.update(bar.close)
        if ema9 is not None:
            values["ema_9"] = ema9
        if ema21 is not None:
            values["ema_21"] = ema21

        if len(prior_bars) >= 20:
            values["roc_20"] = bar.close / prior_bars[-20].close - 1.0
        if len(prior_bars) >= 60:
            values["roc_60"] = bar.close / prior_bars[-60].close - 1.0

        if len(prior_bars) >= 10:
            values["donchian_upper_10"] = max(row.high for row in prior_bars[-10:])
            values["donchian_lower_10"] = min(row.low for row in prior_bars[-10:])
        if len(prior_bars) >= 20:
            values["donchian_upper_20"] = max(row.high for row in prior_bars[-20:])
            values["donchian_lower_20"] = min(row.low for row in prior_bars[-20:])

        atr = self._update_atr(bar)
        if atr is not None:
            values["atr_14"] = atr

        rsi = self._update_rsi(bar.close)
        if rsi is not None:
            values["rsi_14"] = rsi

        if ema12 is not None and ema26 is not None:
            dif = ema12 - ema26
            dea = self._macd_signal.update(dif)
            if dea is not None:
                values["macd_dif"] = dif
                values["macd_dea"] = dea
                values["macd_hist"] = dif - dea

        self._bars.append(bar)
        closes = tuple(row.close for row in self._bars)
        if len(closes) >= 20:
            bollinger = closes[-20:]
            mean20, std20 = _mean_std(bollinger)
            values["bollinger_mid_20"] = mean20
            values["bollinger_upper_20_2"] = mean20 + 2.0 * std20
            values["bollinger_lower_20_2"] = mean20 - 2.0 * std20
        if len(closes) >= 60:
            z_window = closes[-60:]
            mean60, std60 = _mean_std(z_window)
            if std60 > 0.0:
                values["zscore_60"] = (bar.close - mean60) / std60

        status: dict[str, IndicatorStatus] = {
            "route_a.ema_9_21.v1": _ready(values, "ema_9", "ema_21"),
            "route_a.roc_20_60.v1": _ready(values, "roc_20", "roc_60"),
            "route_a.donchian_20_10.v1": _ready(
                values,
                "donchian_upper_20",
                "donchian_lower_20",
                "donchian_upper_10",
                "donchian_lower_10",
            ),
            "route_a.zscore_60.v1": _ready(values, "zscore_60"),
            "route_a.atr_14.v1": _ready(values, "atr_14"),
            "route_a.macd_12_26_9.v1": _ready(
                values, "macd_dif", "macd_dea", "macd_hist"
            ),
            "route_a.rsi_14.v1": _ready(values, "rsi_14"),
            "route_a.bollinger_20_2.v1": _ready(
                values, "bollinger_mid_20", "bollinger_upper_20_2", "bollinger_lower_20_2"
            ),
        }
        expected_ids = {recipe.recipe_id for recipe in ROUTE_A_1M_INDICATOR_RECIPES_V1}
        if set(status) != expected_ids:
            raise RuntimeError("indicator status registry differs from frozen recipes")
        self._sequence += 1
        snapshot = IndicatorSnapshot(
            close_time_ns=bar.close_time_ns,
            available_at_ns=bar.available_at_ns,
            decision_ts_ns=decision_ts_ns,
            sequence=self._sequence,
            values=values,
            recipe_status=status,
        )
        self._last_bar = bar
        self._last_snapshot = snapshot
        return snapshot

    def _update_atr(self, bar: CausalBar) -> float | None:
        if self._atr_prev_close is None:
            true_range = bar.high - bar.low
        else:
            true_range = max(
                bar.high - bar.low,
                abs(bar.high - self._atr_prev_close),
                abs(bar.low - self._atr_prev_close),
            )
        self._atr_prev_close = bar.close
        if self._atr_value is None:
            self._atr_seed.append(true_range)
            if len(self._atr_seed) < 14:
                return None
            self._atr_value = fsum(self._atr_seed) / 14.0
            self._atr_seed.clear()
        else:
            self._atr_value = (13.0 * self._atr_value + true_range) / 14.0
        return self._atr_value

    def _update_rsi(self, close: float) -> float | None:
        previous = self._rsi_prev_close
        self._rsi_prev_close = close
        if previous is None:
            return None
        change = close - previous
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        if self._rsi_avg_gain is None or self._rsi_avg_loss is None:
            self._rsi_seed_gains.append(gain)
            self._rsi_seed_losses.append(loss)
            if len(self._rsi_seed_gains) < 14:
                return None
            self._rsi_avg_gain = fsum(self._rsi_seed_gains) / 14.0
            self._rsi_avg_loss = fsum(self._rsi_seed_losses) / 14.0
            self._rsi_seed_gains.clear()
            self._rsi_seed_losses.clear()
        else:
            self._rsi_avg_gain = (13.0 * self._rsi_avg_gain + gain) / 14.0
            self._rsi_avg_loss = (13.0 * self._rsi_avg_loss + loss) / 14.0
        return _rsi(self._rsi_avg_gain, self._rsi_avg_loss)


def aggregate_completed_bars(
    bars: tuple[CausalBar, ...], *, timeframe_minutes: int, decision_ts_ns: int
) -> tuple[CausalBar, ...]:
    """Aggregate only complete, contiguous UTC buckets available by the decision clock."""
    if type(timeframe_minutes) is not int or timeframe_minutes <= 1:
        raise ValueError("higher timeframe must be an integer greater than one minute")
    if type(decision_ts_ns) is not int or decision_ts_ns <= 0:
        raise ValueError("decision clock must be positive")
    previous_close: int | None = None
    for bar in bars:
        if not isinstance(bar, CausalBar) or bar.timeframe_minutes != 1:
            raise TypeError("higher-timeframe aggregation requires 1m CausalBar inputs")
        if previous_close is not None:
            if bar.close_time_ns <= previous_close:
                raise ValueError("duplicate or out-of-order 1m aggregation input")
            if bar.close_time_ns - previous_close != MINUTE_NS:
                raise ValueError("higher-timeframe aggregation crosses a missing minute")
        previous_close = bar.close_time_ns

    span = timeframe_minutes * MINUTE_NS
    groups: dict[int, list[CausalBar]] = {}
    for bar in bars:
        bucket_close = ((bar.close_time_ns - 1) // span + 1) * span
        groups.setdefault(bucket_close, []).append(bar)
    result = []
    for bucket_close, group in sorted(groups.items()):
        if bucket_close > decision_ts_ns or len(group) != timeframe_minutes:
            continue
        if group[0].open_time_ns != bucket_close - span or group[-1].close_time_ns != bucket_close:
            continue
        available_at = max(row.available_at_ns for row in group)
        if available_at > decision_ts_ns:
            continue
        result.append(CausalBar(
            open_time_ns=bucket_close - span,
            close_time_ns=bucket_close,
            available_at_ns=available_at,
            open=group[0].open,
            high=max(row.high for row in group),
            low=min(row.low for row in group),
            close=group[-1].close,
            volume=fsum(row.volume for row in group),
            timeframe_minutes=timeframe_minutes,
        ))
    return tuple(result)


def replay_indicator_prefix(
    bars: tuple[CausalBar, ...], *, decision_delay_ns: int = 0
) -> tuple[IndicatorSnapshot, ...]:
    """Deterministically rebuild indicator state from an authorized earlier prefix."""
    if type(decision_delay_ns) is not int or decision_delay_ns < 0:
        raise ValueError("decision delay must be a non-negative integer")
    engine = StreamingIndicators1m()
    return tuple(
        engine.update(
            bar,
            decision_ts_ns=max(bar.available_at_ns, bar.close_time_ns + decision_delay_ns),
        )
        for bar in bars
    )


def _mean_std(values: tuple[float, ...]) -> tuple[float, float]:
    mean = fsum(values) / len(values)
    variance = fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, sqrt(max(variance, 0.0))


def _rsi(avg_gain: float, avg_loss: float) -> float:
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    relative_strength = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def _ready(values: dict[str, float], *names: str) -> IndicatorStatus:
    return "READY" if all(name in values for name in names) else "NOT_READY"

