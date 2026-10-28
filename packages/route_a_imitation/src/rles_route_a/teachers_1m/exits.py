"""Independent X2 ATR trailing exit; native execution remains L3-owned."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from rles_route_a.contracts import PointInTimeObservation

from .contracts import TeacherSignal


@dataclass(slots=True)
class AtrTrailingExit:
    atr_multiplier: float = 3.0
    policy_id: str = "X2_ATR_TRAIL_3_V1"
    side: int = 0
    favorable_extreme: float | None = None
    trail: float | None = None
    last_close_time_ns: int | None = None
    combine_indicator_exit: bool = False

    required_fields = (
        "bar_high_1m",
        "bar_low_1m",
        "bar_close_1m",
        "atr_14",
    )

    def __post_init__(self) -> None:
        if (isinstance(self.atr_multiplier, bool)
                or not isinstance(self.atr_multiplier, (int, float))
                or not isfinite(self.atr_multiplier)
                or self.atr_multiplier <= 0):
            raise ValueError("ATR multiplier must be positive and finite")
        if not self.policy_id:
            raise ValueError("ATR exit policy id is required")

    def reset(self) -> None:
        self.side = 0
        self.favorable_extreme = None
        self.trail = None
        self.last_close_time_ns = None

    def evaluate(
        self, observation: PointInTimeObservation, current_exposure: float
    ) -> TeacherSignal | None:
        if abs(current_exposure) <= 1e-8:
            self.reset()
            return None
        if any(name not in observation.market for name in self.required_fields):
            return None
        side = 1 if current_exposure > 0 else -1
        close_time_ns = observation.market_clock_ns("latest_1m_close_ts_ns")
        high = observation.market_value("bar_high_1m")
        low = observation.market_value("bar_low_1m")
        close = observation.market_value("bar_close_1m")
        atr = observation.market_value("atr_14")
        if atr <= 0:
            raise ValueError("ATR exit requires a positive mature ATR")
        entry_price = observation.account_value("position_avg_px_open")
        if entry_price <= 0:
            raise ValueError("ATR trailing exit requires the native average open price")
        if self.side != side:
            self.side = side
            self.last_close_time_ns = close_time_ns
            # Anchor to native basis and close; the bar high/low may predate the fill.
            self.favorable_extreme = (
                max(entry_price, close) if side > 0 else min(entry_price, close)
            )
            self.trail = (
                self.favorable_extreme - self.atr_multiplier * atr
                if side > 0
                else self.favorable_extreme + self.atr_multiplier * atr
            )
            if side > 0 and close < self.trail:
                return TeacherSignal("EXIT_LONG", 0, "ATR_TRAIL_EXIT_LONG")
            if side < 0 and close > self.trail:
                return TeacherSignal("EXIT_SHORT", 0, "ATR_TRAIL_EXIT_SHORT")
            return None
        if self.last_close_time_ns is not None and close_time_ns <= self.last_close_time_ns:
            raise ValueError("ATR exit must advance on a new completed 1m bar")
        self.last_close_time_ns = close_time_ns
        assert self.favorable_extreme is not None and self.trail is not None
        if side > 0:
            self.favorable_extreme = max(self.favorable_extreme, high)
            self.trail = max(
                self.trail,
                self.favorable_extreme - self.atr_multiplier * atr,
            )
            if close < self.trail:
                return TeacherSignal("EXIT_LONG", 0, "ATR_TRAIL_EXIT_LONG")
        else:
            self.favorable_extreme = min(self.favorable_extreme, low)
            self.trail = min(
                self.trail,
                self.favorable_extreme + self.atr_multiplier * atr,
            )
            if close > self.trail:
                return TeacherSignal("EXIT_SHORT", 0, "ATR_TRAIL_EXIT_SHORT")
        return None


@dataclass(slots=True)
class AtrTrailHandoffToIndicatorExit(AtrTrailingExit):
    """Use X2 as an added stop until a cost-scaled excursion; X1 stays active throughout."""

    round_trip_cost_bps: float = 12.0
    handoff_cost_multiple: float = 4.0
    policy_id: str = "X8_ATR_TRAIL_HANDOFF_AT_4X_COST_V1"
    combine_indicator_exit: bool = True
    entry_price: float | None = None
    handed_off: bool = False

    def __post_init__(self) -> None:
        AtrTrailingExit.__post_init__(self)
        for name, value in (
            ("round-trip cost", self.round_trip_cost_bps),
            ("ATR-to-indicator handoff multiple", self.handoff_cost_multiple),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")

    def reset(self) -> None:
        AtrTrailingExit.reset(self)
        self.entry_price = None
        self.handed_off = False

    def evaluate(
        self, observation: PointInTimeObservation, current_exposure: float
    ) -> TeacherSignal | None:
        if abs(current_exposure) <= 1e-8:
            self.reset()
            return None
        if any(name not in observation.market for name in self.required_fields):
            return None
        side = 1 if current_exposure > 0 else -1
        close_time_ns = observation.market_clock_ns("latest_1m_close_ts_ns")
        high = observation.market_value("bar_high_1m")
        low = observation.market_value("bar_low_1m")
        close = observation.market_value("bar_close_1m")
        atr = observation.market_value("atr_14")
        entry_price = observation.account_value("position_avg_px_open")
        if atr <= 0:
            raise ValueError("ATR exit requires a positive mature ATR")
        if entry_price <= 0:
            raise ValueError("ATR-to-indicator handoff requires the native average open price")

        side_changed = self.side != side
        basis_changed = (
            self.entry_price is None
            or abs(entry_price - self.entry_price) > max(1e-8, entry_price * 1e-8)
        )
        if side_changed or basis_changed:
            self.side = side
            self.entry_price = entry_price
            self.last_close_time_ns = close_time_ns
            # Use native basis and close; the bar high/low may predate this position basis.
            self.favorable_extreme = (
                max(entry_price, close) if side > 0 else min(entry_price, close)
            )
            self.trail = (
                self.favorable_extreme - self.atr_multiplier * atr
                if side > 0
                else self.favorable_extreme + self.atr_multiplier * atr
            )
            self.handed_off = self._favorable_bps(side, self.entry_price) >= (
                self.handoff_cost_multiple * self.round_trip_cost_bps
            )
            if not self.handed_off and side > 0 and close < self.trail:
                return TeacherSignal("EXIT_LONG", 0, "ATR_TRAIL_EXIT_LONG")
            if not self.handed_off and side < 0 and close > self.trail:
                return TeacherSignal("EXIT_SHORT", 0, "ATR_TRAIL_EXIT_SHORT")
            return None

        if self.last_close_time_ns is not None and close_time_ns <= self.last_close_time_ns:
            raise ValueError("ATR exit must advance on a new completed 1m bar")
        self.last_close_time_ns = close_time_ns
        assert self.entry_price is not None
        assert self.favorable_extreme is not None and self.trail is not None
        if side > 0:
            self.favorable_extreme = max(self.favorable_extreme, high)
            self.trail = max(self.trail, self.favorable_extreme - self.atr_multiplier * atr)
            if not self.handed_off and close < self.trail:
                return TeacherSignal("EXIT_LONG", 0, "ATR_TRAIL_EXIT_LONG")
        else:
            self.favorable_extreme = min(self.favorable_extreme, low)
            self.trail = min(self.trail, self.favorable_extreme + self.atr_multiplier * atr)
            if not self.handed_off and close > self.trail:
                return TeacherSignal("EXIT_SHORT", 0, "ATR_TRAIL_EXIT_SHORT")
        if self._favorable_bps(side, self.entry_price) >= (
            self.handoff_cost_multiple * self.round_trip_cost_bps
        ):
            self.handed_off = True
        return None

    def _favorable_bps(self, side: int, entry_price: float) -> float:
        assert self.favorable_extreme is not None
        if side > 0:
            return (self.favorable_extreme / entry_price - 1.0) * 10_000
        return (1.0 - self.favorable_extreme / entry_price) * 10_000

@dataclass(slots=True)
class CostAwareAtrProfitProtectExit(AtrTrailingExit):
    """X3 overlay that locks a cost-scaled gain and retains the teacher exit."""

    round_trip_cost_bps: float = 12.0
    activation_cost_multiple: float = 2.0
    lock_cost_multiple: float = 1.5
    policy_id: str = "X3_ATR_COST_PROFIT_PROTECT_V1"
    combine_indicator_exit: bool = True
    entry_price: float | None = None
    activated: bool = False

    def __post_init__(self) -> None:
        AtrTrailingExit.__post_init__(self)
        for name, value in (
            ("round-trip cost", self.round_trip_cost_bps),
            ("activation cost multiple", self.activation_cost_multiple),
            ("profit-lock cost multiple", self.lock_cost_multiple),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if self.activation_cost_multiple <= self.lock_cost_multiple:
            raise ValueError("activation multiple must exceed the profit-lock multiple")

    def reset(self) -> None:
        AtrTrailingExit.reset(self)
        self.entry_price = None
        self.activated = False

    def evaluate(
        self, observation: PointInTimeObservation, current_exposure: float
    ) -> TeacherSignal | None:
        if abs(current_exposure) <= 1e-8:
            self.reset()
            return None
        if any(name not in observation.market for name in self.required_fields):
            return None
        side = 1 if current_exposure > 0 else -1
        close_time_ns = observation.market_clock_ns("latest_1m_close_ts_ns")
        high = observation.market_value("bar_high_1m")
        low = observation.market_value("bar_low_1m")
        close = observation.market_value("bar_close_1m")
        atr = observation.market_value("atr_14")
        entry_price = observation.account_value("position_avg_px_open")
        if atr <= 0:
            raise ValueError("ATR exit requires a positive mature ATR")
        if entry_price <= 0:
            raise ValueError("profit protection requires the native average open price")
        side_changed = self.side != side
        if side_changed:
            self.side = side
            self.entry_price = entry_price
            self.last_close_time_ns = close_time_ns
            # Entry and close are causal; the full-bar extreme may predate the fill.
            self.favorable_extreme = (
                max(entry_price, close) if side > 0 else min(entry_price, close)
            )
            self.trail = None
            self.activated = False
            observed_extreme = close
        else:
            if (self.last_close_time_ns is not None
                    and close_time_ns <= self.last_close_time_ns):
                raise ValueError("ATR exit must advance on a new completed 1m bar")
            self.last_close_time_ns = close_time_ns
            assert self.entry_price is not None
            assert self.favorable_extreme is not None
            basis_changed = (
                abs(entry_price - self.entry_price)
                > max(1e-8, self.entry_price * 1e-8)
            )
            if basis_changed:
                # The new basis and current close are observed; full-bar extremes may predate the fill.
                self.entry_price = entry_price
                self.favorable_extreme = (
                    max(entry_price, close) if side > 0 else min(entry_price, close)
                )
                self.trail = None
                self.activated = False
                observed_extreme = close
            else:
                observed_extreme = high if side > 0 else low
        assert self.entry_price is not None and self.favorable_extreme is not None
        if side > 0:
            self.favorable_extreme = max(self.favorable_extreme, observed_extreme)
            favorable_bps = (self.favorable_extreme / self.entry_price - 1.0) * 10_000
        else:
            self.favorable_extreme = min(self.favorable_extreme, observed_extreme)
            favorable_bps = (1.0 - self.favorable_extreme / self.entry_price) * 10_000
        if not self.activated:
            trigger_bps = self.activation_cost_multiple * self.round_trip_cost_bps
            if favorable_bps < trigger_bps:
                return None
            self.activated = True
        lock_bps = self.lock_cost_multiple * self.round_trip_cost_bps
        if side > 0:
            cost_lock = self.entry_price * (1.0 + lock_bps / 10_000)
            atr_trail = self.favorable_extreme - self.atr_multiplier * atr
            candidate = max(cost_lock, atr_trail)
            self.trail = candidate if self.trail is None else max(self.trail, candidate)
            if close < self.trail:
                return TeacherSignal("EXIT_LONG", 0, "COST_AWARE_ATR_PROFIT_LOCK_LONG")
        else:
            cost_lock = self.entry_price * (1.0 - lock_bps / 10_000)
            atr_trail = self.favorable_extreme + self.atr_multiplier * atr
            candidate = min(cost_lock, atr_trail)
            self.trail = candidate if self.trail is None else min(self.trail, candidate)
            if close > self.trail:
                return TeacherSignal("EXIT_SHORT", 0, "COST_AWARE_ATR_PROFIT_LOCK_SHORT")
        return None


@dataclass(slots=True)
class CostAwareAtrProfitAndLossGuardExit(CostAwareAtrProfitProtectExit):
    """X5 adds a cost/ATR adverse stop before the X3 profit lock activates."""

    loss_guard_cost_multiple: float = 2.0
    loss_guard_atr_multiple: float = 2.0
    policy_id: str = "X5_ATR_COST_PROFIT_AND_LOSS_GUARD_V1"

    def __post_init__(self) -> None:
        CostAwareAtrProfitProtectExit.__post_init__(self)
        for name, value in (
            ("loss-guard cost multiple", self.loss_guard_cost_multiple),
            ("loss-guard ATR multiple", self.loss_guard_atr_multiple),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")

    def evaluate(
        self, observation: PointInTimeObservation, current_exposure: float
    ) -> TeacherSignal | None:
        if abs(current_exposure) > 1e-8 and all(
            name in observation.market for name in self.required_fields
        ):
            side = 1 if current_exposure > 0 else -1
            close = observation.market_value("bar_close_1m")
            atr = observation.market_value("atr_14")
            entry_price = observation.account_value("position_avg_px_open")
            basis_changed = (
                self.entry_price is None
                or abs(entry_price - self.entry_price)
                > max(1e-8, entry_price * 1e-8)
            )
            if (not self.activated or basis_changed) and close > 0 and atr > 0 and entry_price > 0:
                adverse_bps = side * (close / entry_price - 1.0) * 10_000
                atr_bps = atr / close * 10_000
                guard_bps = max(
                    self.loss_guard_cost_multiple * self.round_trip_cost_bps,
                    self.loss_guard_atr_multiple * atr_bps,
                )
                if adverse_bps <= -guard_bps:
                    direction = "LONG" if side > 0 else "SHORT"
                    return TeacherSignal(
                        "EXIT_LONG" if side > 0 else "EXIT_SHORT",
                        0,
                        f"COST_AWARE_ATR_LOSS_GUARD_{direction}",
                    )
        return CostAwareAtrProfitProtectExit.evaluate(
            self, observation, current_exposure,
        )

