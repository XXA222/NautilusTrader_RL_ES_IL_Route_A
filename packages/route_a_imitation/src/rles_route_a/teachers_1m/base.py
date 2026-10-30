"""Common causal state machine for independent one-minute teachers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from rles_route_a.contracts import (
    ExpertAdvice, NativeActionTrace, PointInTimeObservation,
)
from rles_route_a.indicators_1m import MINUTE_NS
from rles_route_a.teachers.protocol import advice

from .contracts import PendingReversal1m, TeacherRecipe1m, TeacherSignal
from .exits import AtrTrailingExit


class BaseTeacher1m(ABC):
    """Evaluate one completed 1m snapshot once; account execution remains native."""

    required_fields: tuple[str, ...] = ()

    def __init__(
        self, recipe: TeacherRecipe1m, *, atr_exit: AtrTrailingExit | None = None
    ) -> None:
        if recipe.htf_mode != "FEATURE_ONLY":
            raise ValueError("the frozen first-pass teacher requires htf_mode=FEATURE_ONLY")
        self.recipe = recipe
        self._episode_id = ""
        self._source_id: str | None = None
        self._fold_id: str | None = None
        self._feature_spec_hash: str | None = None
        self._last_close_time_ns: int | None = None
        self._previous: dict[str, float] = {}
        self._last_signal = TeacherSignal("NONE", 0, "NOT_EVALUATED")
        self._pending_reversal: PendingReversal1m | None = None
        self._atr_exit = atr_exit

    @property
    def expert_id(self) -> str:
        return self.recipe.teacher_id

    @property
    def last_signal(self) -> TeacherSignal:
        return self._last_signal

    def reset(self, episode_id: str = "") -> None:
        if not isinstance(episode_id, str):
            raise TypeError("episode id must be a string")
        self._episode_id = episode_id
        self._source_id = None
        self._fold_id = None
        self._feature_spec_hash = None
        self._last_close_time_ns = None
        self._previous = {}
        self._last_signal = TeacherSignal("NONE", 0, "NOT_EVALUATED")
        self._pending_reversal = None
        self.on_discontinuity()
        if self._atr_exit is not None:
            self._atr_exit.reset()

    def on_discontinuity(self) -> None:
        """Clear teacher-specific rolling state after a reset, gap, or stale bar."""

    def advise(self, observation: PointInTimeObservation) -> ExpertAdvice:
        if observation.episode_id != self._episode_id:
            self.reset(observation.episode_id)
        if (
            (self._source_id is not None and observation.source_id != self._source_id)
            or (self._fold_id is not None and observation.fold_id != self._fold_id)
            or (self._feature_spec_hash is not None
                and observation.feature_spec_hash != self._feature_spec_hash)
        ):
            # Rolling indicators and pending exits cannot cross market or feature identities.
            self.reset(observation.episode_id)
        self._source_id = observation.source_id
        self._fold_id = observation.fold_id
        self._feature_spec_hash = observation.feature_spec_hash
        current_exposure = observation.account_value("net_exposure")
        try:
            close_time_ns = observation.market_clock_ns("latest_1m_close_ts_ns")
        except KeyError:
            if "latest_1m_close_ts_ns" not in observation.market:
                self._previous = {}
                self._pending_reversal = None
                self.on_discontinuity()
                self._last_signal = TeacherSignal("NONE", 0, "NO_FRESH_CAUSAL_BAR")
                return self._quiet(observation, current_exposure, "NO_FRESH_CAUSAL_BAR")
        if close_time_ns % MINUTE_NS != 0 or close_time_ns > observation.decision_ts_ns:
            raise ValueError("invalid latest completed 1m close clock")
        if observation.decision_ts_ns - close_time_ns >= MINUTE_NS:
            self._previous = {}
            self._pending_reversal = None
            self.on_discontinuity()
            self._last_signal = TeacherSignal("NONE", 0, "NO_FRESH_CAUSAL_BAR")
            return self._quiet(observation, current_exposure, "NO_FRESH_CAUSAL_BAR")
        if self._last_close_time_ns is not None:
            if close_time_ns < self._last_close_time_ns:
                raise ValueError("teacher received an out-of-order completed 1m bar")
            if close_time_ns == self._last_close_time_ns:
                self._last_signal = TeacherSignal("NONE", 0, "NO_NEW_BAR")
                return self._quiet(observation, current_exposure, "NO_NEW_BAR")
            if close_time_ns - self._last_close_time_ns != MINUTE_NS:
                self._last_close_time_ns = close_time_ns
                self._previous = {}
                self._pending_reversal = None
                self.on_discontinuity()
                if self._atr_exit is not None:
                    missing_exit_fields = tuple(
                        name for name in self._atr_exit.required_fields
                        if name not in observation.market
                    )
                    if abs(current_exposure) <= 1e-8:
                        self._atr_exit.reset()
                    elif missing_exit_fields:
                        raise ValueError(
                            "ATR exit inputs missing while a position is open: "
                            + ", ".join(missing_exit_fields)
                        )
                    else:
                        atr_signal = self._atr_exit.evaluate(
                            observation, current_exposure,
                        )
                        if atr_signal is not None:
                            self._last_signal = atr_signal
                            return advice(
                                observation,
                                self.expert_id,
                                "EXIT",
                                0.0,
                                "ATR_TRAILING_EXIT",
                                atr_signal.reason_code,
                            )
                self._last_signal = TeacherSignal("NONE", 0, "MISSING_BAR")
                return self._quiet(observation, current_exposure, "MISSING_BAR")
        self._last_close_time_ns = close_time_ns

        atr_exit = self._atr_exit
        missing_exit_fields = (
            tuple(
                name
                for name in atr_exit.required_fields
                if name not in observation.market
            )
            if atr_exit is not None
            else ()
        )
        if missing_exit_fields and abs(current_exposure) > 1e-8:
            raise ValueError(
                "ATR exit inputs missing while a position is open: "
                + ", ".join(missing_exit_fields)
            )

        atr_signal = (
            self._atr_exit.evaluate(observation, current_exposure)
            if self._atr_exit is not None
            else None
        )

        if any(name not in observation.market for name in self.required_fields):
            self._previous = {}
            self._pending_reversal = None
            self.on_discontinuity()
            if atr_signal is not None:
                self._pending_reversal = None
                self._last_signal = atr_signal
                return advice(
                    observation,
                    self.expert_id,
                    "EXIT",
                    0.0,
                    "ATR_TRAILING_EXIT",
                    atr_signal.reason_code,
                )
            self._last_signal = TeacherSignal("NONE", 0, "INDICATOR_NOT_READY")
            return self._quiet(observation, current_exposure, "INDICATOR_NOT_READY")
        current = {name: observation.market_value(name) for name in self.required_fields}
        if not self._previous:
            self._previous = current
            if atr_signal is not None:
                self._pending_reversal = None
                self._last_signal = atr_signal
                return advice(
                    observation,
                    self.expert_id,
                    "EXIT",
                    0.0,
                    "ATR_TRAILING_EXIT",
                    atr_signal.reason_code,
                )
            self._last_signal = TeacherSignal("NONE", 0, "INDICATOR_PREVIOUS_NOT_READY")
            return self._quiet(observation, current_exposure, "INDICATOR_PREVIOUS_NOT_READY")

        entry = self.entry_signal(current, self._previous)
        exit_signal = (
            None
            if self._atr_exit is not None and not self._atr_exit.combine_indicator_exit
            else self.exit_signal(current_exposure, current, self._previous)
        )
        self._previous = current
        if atr_signal is not None:
            self._pending_reversal = None
            self._last_signal = atr_signal
            return advice(
                observation,
                self.expert_id,
                "EXIT",
                0.0,
                "ATR_TRAILING_EXIT",
                atr_signal.reason_code,
            )
        if missing_exit_fields:
            self._pending_reversal = None
            self._last_signal = TeacherSignal("NONE", 0, "ATR_EXIT_NOT_READY")
            return self._quiet(observation, current_exposure, "ATR_EXIT_NOT_READY")
        pending_advice = self._pending_reversal_advice(
            observation, current, current_exposure, close_time_ns,
        )
        if pending_advice is not None:
            return pending_advice
        if abs(current_exposure) > 1e-8:
            if exit_signal is not None:
                reversal_side = self.reversal_side_for_exit(exit_signal)
                if reversal_side is not None:
                    self._pending_reversal = PendingReversal1m(
                        side=reversal_side,
                        event_close_ts_ns=close_time_ns,
                        episode_id=observation.episode_id,
                        source_id=observation.source_id,
                        reason_code=exit_signal.reason_code,
                    )
                self._last_signal = exit_signal
                return advice(
                    observation,
                    self.expert_id,
                    "EXIT",
                    0.0,
                    "INDICATOR_EXIT",
                    exit_signal.reason_code,
                )
            self._last_signal = TeacherSignal("NONE", 0, "POSITION_HOLD")
            return advice(
                observation,
                self.expert_id,
                "HOLD",
                current_exposure,
                "POSITION_HOLD",
            )

        self._last_signal = entry
        if entry.side > 0:
            return advice(
                observation,
                self.expert_id,
                "OPEN_LONG",
                self.recipe.target_exposure,
                entry.reason_code,
            )
        if entry.side < 0:
            return advice(
                observation,
                self.expert_id,
                "OPEN_SHORT",
                -self.recipe.target_exposure,
                entry.reason_code,
            )
        return advice(observation, self.expert_id, "FLAT", 0.0, entry.reason_code)

    def reversal_side_for_exit(
        self, exit_signal: TeacherSignal
    ) -> Literal[-1, 1] | None:
        return None

    def reversal_entry_signal(
        self, side: Literal[-1, 1], current: Mapping[str, float]
    ) -> TeacherSignal | None:
        return None

    def _pending_reversal_advice(
        self,
        observation: PointInTimeObservation,
        current: Mapping[str, float],
        current_exposure: float,
        close_time_ns: int,
    ) -> ExpertAdvice | None:
        candidate = self._pending_reversal
        if candidate is None:
            return None
        if (candidate.episode_id != observation.episode_id
                or candidate.source_id != observation.source_id):
            self._pending_reversal = None
            return None
        if candidate.close_confirmed_bar_close_ts_ns is not None:
            if abs(current_exposure) > 1e-8:
                self._pending_reversal = None
                self._last_signal = TeacherSignal("NONE", 0, "REVERSAL_NOT_FLAT")
                return self._quiet(observation, current_exposure, "REVERSAL_NOT_FLAT")
            if observation.pending_order_ids:
                self._last_signal = TeacherSignal("NONE", 0, "REVERSAL_PENDING_ORDERS")
                return self._quiet(observation, current_exposure, "REVERSAL_PENDING_ORDERS")
            if close_time_ns <= candidate.close_confirmed_bar_close_ts_ns:
                self._last_signal = TeacherSignal("NONE", 0, "WAIT_NEXT_REVERSAL_BAR")
                return self._quiet(
                    observation, current_exposure, "WAIT_NEXT_REVERSAL_BAR",
                )
            signal = self.reversal_entry_signal(candidate.side, current)
            self._pending_reversal = None
            if signal is None:
                self._last_signal = TeacherSignal("NONE", 0, "EMA_REVERSAL_EXPIRED")
                return self._quiet(observation, current_exposure, "EMA_REVERSAL_EXPIRED")
            self._last_signal = signal
            if signal.side > 0:
                return advice(
                    observation, self.expert_id, "OPEN_LONG",
                    self.recipe.target_exposure, signal.reason_code,
                )
            return advice(
                observation, self.expert_id, "OPEN_SHORT",
                -self.recipe.target_exposure, signal.reason_code,
            )
        if abs(current_exposure) <= 1e-8:
            self._last_signal = TeacherSignal("NONE", 0, "WAIT_NATIVE_CLOSE_ACK")
            return advice(observation, self.expert_id, "FLAT", 0.0, "WAIT_NATIVE_CLOSE_ACK")
        if current_exposure * candidate.side < 0:
            signal = TeacherSignal(
                "EXIT_LONG" if current_exposure > 0 else "EXIT_SHORT",
                0,
                "EMA_REVERSE_CLOSE_RETRY",
            )
            self._last_signal = signal
            return advice(
                observation, self.expert_id, "EXIT", 0.0, signal.reason_code,
            )
        self._pending_reversal = None
        return self._quiet(observation, current_exposure, "REVERSAL_POSITION_CHANGED")

    def observe_native_action(
        self,
        action: ExpertAdvice,
        trace: NativeActionTrace,
        native_order_facts: Any,
    ) -> None:
        # Use only the L3/M2.2 evidence attached to the native barrier.
        candidate = self._pending_reversal
        if candidate is None:
            return
        risk_reasons = ("RISK", "MAX_DRAWDOWN", "LIQUIDATION", "FORCE_FLAT")
        hard_risk = (
            any(any(token in reason.upper() for token in risk_reasons)
                for reason in action.reason_codes)
            or any(token in trace.planner_reason.upper() for token in risk_reasons)
        )
        if action.expert_id != self.expert_id or hard_risk:
            self._pending_reversal = None
            return
        reversal_close = (
            action.intent == "EXIT"
            and any(reason.startswith("EMA_REVERSE_")
                    or reason == "EMA_REVERSE_CROSS"
                    for reason in action.reason_codes)
        )
        if reversal_close:
            candidate.close_request_seen = True
            candidate.close_order_ids.update(trace.submitted_order_ids)
        elif action.intent not in ("HOLD", "FLAT") and "PENDING_ACK" not in action.reason_codes:
            self._pending_reversal = None
            return

        statuses = getattr(native_order_facts, "native_order_statuses_after", ())
        rejected = {"REJECTED", "DENIED", "CANCELED", "CANCELLED", "EXPIRED"}
        known_ids = candidate.close_order_ids
        rejected_ids = {
            row.get("client_order_id")
            for row in statuses
            if row.get("client_order_id") in known_ids
            and str(row.get("status", "")).upper() in rejected
        }
        # A rejected close does not invalidate the observed EMA reversal. Keep
        # the candidate so the next legal bar can retry closing the old side;
        # discard only the failed order identity so it cannot poison a later
        # successful close proof.
        candidate.close_order_ids.difference_update(rejected_ids)
        if abs(trace.actual_exposure_after) > 1e-8:
            return
        fills = getattr(native_order_facts, "native_policy_fills", ())
        status_receipt = getattr(native_order_facts, "native_order_status_receipt", "")
        evidence_receipt = getattr(native_order_facts, "native_evidence_receipt", "")
        filled_ids = set(getattr(native_order_facts, "filled_order_ids", ()))
        filled_ids.update(trace.filled_order_ids)
        fill_event_ids = {
            row.get("client_order_id") for row in fills
            if row.get("client_order_id") in known_ids and row.get("event_id")
        }
        expected_side = "SELL" if candidate.side < 0 else "BUY"
        full_status_ids: set[str] = set()
        for row in statuses:
            order_id = row.get("client_order_id")
            if order_id not in known_ids or str(row.get("status", "")).upper() != "FILLED":
                continue
            if str(row.get("side", "")).upper() != expected_side:
                continue
            try:
                quantity = Decimal(str(row.get("quantity", "0")))
                filled_quantity = Decimal(str(row.get("filled_qty", "0")))
            except InvalidOperation:
                continue
            if quantity > 0 and filled_quantity == quantity:
                full_status_ids.add(order_id)
        completed_close_ids = known_ids & filled_ids & fill_event_ids & full_status_ids
        pending_after = set(getattr(native_order_facts, "pending_order_ids_after", ()))
        pending_after.update(trace.pending_order_ids_after)
        proof = (
            candidate.close_request_seen
            and bool(completed_close_ids)
            and not pending_after
            and bool(evidence_receipt)
            and bool(status_receipt)
            and self._last_close_time_ns is not None
            and self._last_close_time_ns >= candidate.event_close_ts_ns
        )
        if proof:
            candidate.close_confirmed_bar_close_ts_ns = self._last_close_time_ns

    def _quiet(
        self, observation: PointInTimeObservation, current_exposure: float, reason: str
    ) -> ExpertAdvice:
        if abs(current_exposure) > 1e-8:
            return advice(
                observation, self.expert_id, "HOLD", current_exposure, reason
            )
        return advice(observation, self.expert_id, "FLAT", 0.0, reason)

    @abstractmethod
    def entry_signal(
        self, current: Mapping[str, float], previous: Mapping[str, float]
    ) -> TeacherSignal:
        raise NotImplementedError

    @abstractmethod
    def exit_signal(
        self,
        current_exposure: float,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> TeacherSignal | None:
        raise NotImplementedError

