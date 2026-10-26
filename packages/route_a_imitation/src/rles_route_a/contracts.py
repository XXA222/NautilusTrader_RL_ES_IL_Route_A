"""Lightweight Route-A-only read-only views; never replace L1/L2/L3 native DTOs."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Mapping, Protocol, Sequence
from types import MappingProxyType

Intent = Literal["FLAT", "OPEN_LONG", "OPEN_SHORT", "HOLD", "ADD", "REDUCE", "EXIT"]
INTENTS: tuple[Intent, ...] = ("FLAT", "OPEN_LONG", "OPEN_SHORT", "HOLD", "ADD", "REDUCE", "EXIT")
ACTION_SCHEMA = "single_exposure_v1"


class FrozenMap(Mapping[str, float]):
    """Copy-on-create, read-only AND spawn-pickleable numeric key/value mapping.

    A bare MappingProxyType isn't pickleable under Windows spawned DataLoader
    workers. This wrapper explicitly serializes only frozen primitive contents.
    """
    __slots__ = ("_data",)
    _data: Mapping[str, float]

    def __init__(self, mapping: Mapping[str, float]):
        object.__setattr__(self, "_data", MappingProxyType(dict(mapping)))

    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __reduce__(self):
        return (FrozenMap, (dict(self._data),))

    def __setattr__(self, key, value):
        raise AttributeError("frozen observation mapping cannot be reassigned")


def finite_number(x: float, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x):
        raise ValueError(f"{name} must be a finite number")
    return float(x)


@dataclass(frozen=True, slots=True)
class PointInTimeObservation:
    """A point-in-time read-only projection supplied by an authorized L2/L3 bridge.

    feature_available_at is REQUIRED, including for 1h/1d bars. No market-data
    loading is performed by this package. Account/order fields are L3 facts.
    """

    episode_id: str
    decision_ts_ns: int
    market: Mapping[str, float]
    feature_available_at_ns: Mapping[str, int]
    account: Mapping[str, float]
    pending_order_ids: tuple[str, ...] = ()
    fold_id: str = ""
    source_id: str = ""
    snapshot_id: str = ""
    feature_spec_hash: str = ""
    exact_market_clocks_ns: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.episode_id or type(self.decision_ts_ns) is not int or self.decision_ts_ns <= 0:
            raise ValueError("episode id and positive ns clock required")
        for field_name in ("source_id", "snapshot_id", "feature_spec_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or (value and not value.strip()):
                raise ValueError(f"invalid observation {field_name}")
        for field_name in ("snapshot_id", "feature_spec_hash"):
            value = getattr(self, field_name)
            if value and (len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 identity")
        if set(self.market) != set(self.feature_available_at_ns):
            raise ValueError("feature timestamps must exactly cover all market features")
        if not self.market:
            raise ValueError("market features absent")
        if any(not isinstance(k, str) or not k or not k.strip() for k in self.market):
            raise ValueError("invalid market feature keys")
        if any(not isinstance(k, str) or not k or not k.strip() for k in self.account):
            raise ValueError("invalid native account field names")
        if (not isinstance(self.exact_market_clocks_ns, tuple)
                or any(not isinstance(row, tuple) or len(row) != 2 for row in self.exact_market_clocks_ns)):
            raise ValueError("exact market clocks must be immutable (name, ns) pairs")
        clock_names: set[str] = set()
        for name, timestamp in self.exact_market_clocks_ns:
            if (not isinstance(name, str) or not name.strip() or name in clock_names
                    or type(timestamp) is not int or timestamp <= 0
                    or timestamp > self.decision_ts_ns):
                raise ValueError("invalid or future exact market clock")
            clock_names.add(name)
        for k, v in self.market.items():
            finite_number(v, k)
            at = self.feature_available_at_ns[k]
            if type(at) is not int or not 0 <= at <= self.decision_ts_ns:
                raise ValueError(f"future or invalid market feature: {k}")
        for k, v in self.account.items():
            finite_number(v, f"account.{k}")
        if len(self.pending_order_ids) != len(set(self.pending_order_ids)):
            raise ValueError("duplicate pending native order ids")
        if any(not isinstance(v, str) or not v.strip() for v in self.pending_order_ids):
            raise ValueError("invalid pending native order")
        # Dataclass frozen=True does not freeze mutable mapping inputs. Snapshot
        # them after checking causal availability to prevent time-of-check/time-of-use
        # leakage or hidden state changes during teacher/model inference.
        object.__setattr__(self, "market", FrozenMap(self.market))
        object.__setattr__(self, "feature_available_at_ns",
                           FrozenMap(self.feature_available_at_ns))
        object.__setattr__(self, "account", FrozenMap(self.account))
        object.__setattr__(self, "pending_order_ids", tuple(self.pending_order_ids))
        object.__setattr__(self, "exact_market_clocks_ns", tuple(self.exact_market_clocks_ns))

    def account_value(self, name: str) -> float:
        if name not in self.account:
            raise KeyError(f"required native account field is missing: {name}")
        return finite_number(self.account[name], name)

    def market_value(self, name: str) -> float:
        if name not in self.market:
            raise KeyError(f"required causal feature is missing: {name}")
        return finite_number(self.market[name], name)

    def exact_market_clock(self, name: str) -> int:
        """Return an exact integer market clock, never a float-encoded ns value."""
        for clock_name, timestamp in self.exact_market_clocks_ns:
            if clock_name == name:
                return timestamp
        raise KeyError(f"required exact market clock is missing: {name}")

    def market_clock_ns(self, name: str) -> int:
        """Read an exact clock, allowing legacy float values only in the safe range."""
        try:
            return self.exact_market_clock(name)
        except KeyError:
            pass
        if name not in self.market:
            raise KeyError(f"required causal market clock is missing: {name}")
        value = self.market_value(name)
        if abs(value) > 2**53 or not value.is_integer():
            raise ValueError("epoch-ns market clock requires exact integer metadata")
        return int(value)


@dataclass(frozen=True, slots=True)
class ExpertAdvice:
    expert_id: str
    decision_ts_ns: int
    intent: Intent
    requested_exposure: float
    reason_codes: tuple[str, ...]
    horizon_hint_bars: int | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.expert_id or self.intent not in INTENTS:
            raise ValueError("invalid expert id or intent")
        if type(self.decision_ts_ns) is not int or self.decision_ts_ns <= 0:
            raise ValueError("invalid expert decision time")
        a = finite_number(self.requested_exposure, "teacher exposure")
        if not -1 <= a <= 1:
            raise ValueError("teacher exposure outside current NETTING contract")
        if self.confidence is not None and not 0 <= finite_number(self.confidence, "confidence") <= 1:
            raise ValueError("uncalibrated or invalid confidence")
        if self.horizon_hint_bars is not None and (type(self.horizon_hint_bars) is not int or self.horizon_hint_bars < 0):
            raise ValueError("negative horizon")
        if not self.reason_codes or any(type(code) is not str or not code.strip()
                                        for code in self.reason_codes):
            raise ValueError("auditable reason code required")


@dataclass(frozen=True, slots=True)
class NativeActionTrace:
    """Subset of L3 native trace. Observations and PnL remain owned by L3/L6."""

    decision_ts_ns: int
    requested_action: float
    risk_adjusted_target: float
    planner_target: float
    actual_exposure_after: float
    native_equity_before: float
    native_equity_after: float
    planner_action: str
    submitted_order_ids: tuple[str, ...] = ()
    filled_order_ids: tuple[str, ...] = ()
    related_existing_order_ids: tuple[str, ...] = ()
    pending_order_ids_after: tuple[str, ...] = ()
    fee_currency: str | None = None
    scenario_only: bool = True
    soft_risk_target: float | None = None
    execution_decision_id: str = ""
    execution_plan_id: str = ""
    execution_idempotency_key: str = ""
    planner_reason: str = ""
    execution_barrier_ts_ns: int = 0

    def __post_init__(self) -> None:
        if type(self.decision_ts_ns) is not int or self.decision_ts_ns <= 0:
            raise ValueError("invalid trace timestamp")
        for field_name in ("requested_action", "risk_adjusted_target", "planner_target", "actual_exposure_after", "native_equity_before", "native_equity_after"):
            finite_number(getattr(self, field_name), field_name)
        if self.soft_risk_target is not None:
            finite_number(self.soft_risk_target, "soft_risk_target")
            if not -1 <= self.soft_risk_target <= 1:
                raise ValueError("soft risk target not representable by action schema")
        for field_name in ("execution_decision_id", "execution_plan_id",
                           "execution_idempotency_key", "planner_reason"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or (value and not value.strip()):
                raise ValueError(f"invalid {field_name}")
        if (type(self.execution_barrier_ts_ns) is not int or self.execution_barrier_ts_ns < 0
                or (self.execution_barrier_ts_ns
                    and self.execution_barrier_ts_ns <= self.decision_ts_ns)):
            raise ValueError("execution barrier clock must follow the Route A decision clock")
        if not -1 <= self.requested_action <= 1:
            raise ValueError("requested action not representable by action schema")
        if self.native_equity_before <= 0 or self.native_equity_after <= 0:
            raise ValueError("cannot compute log reward for non-positive equity")
        if self.planner_action not in {"NOOP", "WAIT", "CANCEL", "SUBMIT"}:
            raise ValueError("unknown planner action")
        # A fill may belong to an order submitted in an EARLIER barrier.
        valid_ids = set(self.submitted_order_ids) | set(self.related_existing_order_ids)
        for field_name in ("submitted_order_ids", "filled_order_ids", "related_existing_order_ids",
                           "pending_order_ids_after"):
            ids = getattr(self, field_name)
            if len(ids) != len(set(ids)) or any(type(v) is not str or not v.strip() for v in ids):
                raise ValueError(f"duplicate or invalid native order identity in {field_name}")
        if set(self.filled_order_ids) - valid_ids:
            raise ValueError("filled ids require new or pre-existing native order provenance")


def assert_executable_exit(advice: ExpertAdvice, trace: NativeActionTrace, *,
                           before_exposure: float, tolerance: float = 1e-6) -> None:
    """Fail closed: requested FLAT/EXIT is not necessarily a native close.

    CANCEL/WAIT are legitimate intermediate states but are NOT proof of exit.
    Call only after native order chain settled and latest position was re-read.
    """
    if advice.intent not in ("EXIT", "FLAT"):
        raise ValueError("exit check requires EXIT or FLAT intent")
    if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
            or not isfinite(tolerance) or tolerance <= 0
            or not isinstance(before_exposure, (int, float))
            or isinstance(before_exposure, bool) or not isfinite(before_exposure)):
        raise ValueError("finite native exposure and positive tolerance required")
    if abs(advice.requested_exposure) > tolerance or abs(trace.requested_action) > tolerance:
        raise AssertionError("exit label did not request zero")
    if abs(trace.risk_adjusted_target) > tolerance or abs(trace.planner_target) > tolerance:
        raise AssertionError("risk or planner did not preserve the flat target")
    if trace.planner_action in {"WAIT", "CANCEL"}:
        raise AssertionError("open cancel/ack chain: exit not yet complete")
    if abs(trace.actual_exposure_after) > tolerance:
        raise AssertionError("exit was not actually filled / position remains open")
    if (abs(before_exposure) > tolerance
            and not set(trace.submitted_order_ids).intersection(trace.filled_order_ids)):
        raise AssertionError("cannot claim successful native exit without fill proof")


class ExpertTeacher(Protocol):
    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice: ...


class RouteAPolicy(Protocol):
    def reset(self, episode_id: str) -> None: ...
    def predict_exposure(self, observation: PointInTimeObservation) -> float: ...


class NativeRolloutProvider(Protocol):
    """Must delegate to the host project's registered L3 factory + L6 access."""

    def rollout(self, policy: RouteAPolicy, *, fold_id: str, split: str,
                seed: int) -> Sequence[NativeActionTrace]: ...
