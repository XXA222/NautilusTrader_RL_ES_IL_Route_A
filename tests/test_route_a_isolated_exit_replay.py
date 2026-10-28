from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

from route_a_isolated_exit_replay import (  # noqa: E402
    FirstRoundTripTeacher,
    _case_identity,
    _pair_slot,
    _path_metrics,
)
from rles_route_a.teachers.protocol import advice  # noqa: E402


class _Observation:
    def __init__(self, exposure: float, decision_ts_ns: int) -> None:
        self.exposure = exposure
        self.decision_ts_ns = decision_ts_ns
        self.pending_order_ids = ()

    def account_value(self, name: str) -> float:
        assert name == "net_exposure"
        return self.exposure


class _Teacher:
    expert_id = "fake_teacher"
    recipe = SimpleNamespace(warmup_bars=0)
    last_signal = None

    def __init__(self) -> None:
        self.advise_calls = 0

    def reset(self, _episode_id: str = "") -> None:
        pass

    def advise(self, observation: _Observation):
        self.advise_calls += 1
        exposure = observation.account_value("net_exposure")
        if abs(exposure) < 1e-8:
            return advice(observation, self.expert_id, "OPEN_LONG", 0.3, "FAKE_ENTRY")
        return advice(observation, self.expert_id, "HOLD", 0.3, "FAKE_HOLD")


def test_first_round_trip_wrapper_suppresses_only_after_confirmed_close():
    delegate = _Teacher()
    teacher = FirstRoundTripTeacher(delegate)
    first = _Observation(0.0, 60)
    open_advice = teacher.advise(first)
    teacher.observe_native_action(
        open_advice,
        SimpleNamespace(actual_exposure_after=0.3, filled_order_ids=(),
                        pending_order_ids_after=()),
        SimpleNamespace(filled_order_ids=(), native_policy_fills=(),
                        native_order_statuses_after=(), pending_order_ids_after=()),
    )
    hold = teacher.advise(_Observation(0.3, 120))
    teacher.observe_native_action(
        hold,
        SimpleNamespace(actual_exposure_after=0.0, filled_order_ids=("close-id",),
                        pending_order_ids_after=()),
        SimpleNamespace(
            filled_order_ids=("close-id",),
            native_policy_fills=({"client_order_id": "close-id", "order_side": "SELL",
                                  "event_id": "fill-close"},),
            native_order_statuses_after=({"client_order_id": "close-id", "status": "FILLED",
                                         "side": "SELL", "quantity": "0.3",
                                         "filled_qty": "0.3"},),
            pending_order_ids_after=(),
        ),
    )

    flat = teacher.advise(_Observation(0.0, 180))

    assert teacher.completed is True
    assert flat.intent == "FLAT"
    assert flat.requested_exposure == 0.0
    assert teacher.suppressed_decision_count == 1
    assert delegate.advise_calls == 2


def test_first_round_trip_rejects_direct_exposure_reversal():
    delegate = _Teacher()
    teacher = FirstRoundTripTeacher(delegate)
    entry = teacher.advise(_Observation(0.0, 60))
    teacher.observe_native_action(
        entry,
        SimpleNamespace(actual_exposure_after=0.3, filled_order_ids=(),
                        pending_order_ids_after=()),
        SimpleNamespace(filled_order_ids=(), native_policy_fills=(),
                        native_order_statuses_after=(), pending_order_ids_after=()),
    )
    reversal = teacher.advise(_Observation(0.3, 120))

    with pytest.raises(RuntimeError, match='exposure reversed before first-round-trip close proof'):
        teacher.observe_native_action(
            reversal,
            SimpleNamespace(actual_exposure_after=-0.3, filled_order_ids=(),
                            pending_order_ids_after=()),
            SimpleNamespace(filled_order_ids=(), native_policy_fills=(),
                            native_order_statuses_after=(), pending_order_ids_after=()),
        )


def test_first_round_trip_wrapper_waits_for_pending_close_ack():
    delegate = _Teacher()
    teacher = FirstRoundTripTeacher(delegate)
    entry = teacher.advise(_Observation(0.0, 60))
    teacher.observe_native_action(
        entry,
        SimpleNamespace(actual_exposure_after=0.3, filled_order_ids=(),
                        pending_order_ids_after=()),
        SimpleNamespace(filled_order_ids=(), native_policy_fills=(),
                        native_order_statuses_after=(), pending_order_ids_after=()),
    )
    close = advice(_Observation(0.3, 120), delegate.expert_id, "EXIT", 0.0, "CLOSE")
    teacher.observe_native_action(
        close,
        SimpleNamespace(actual_exposure_after=0.0, filled_order_ids=("close-id",),
                        pending_order_ids_after=("close-pending",)),
        SimpleNamespace(
            filled_order_ids=("close-id",),
            native_policy_fills=({"client_order_id": "close-id", "order_side": "SELL",
                                  "event_id": "fill-close"},),
            native_order_statuses_after=({"client_order_id": "close-id", "status": "FILLED",
                                         "side": "SELL", "quantity": "0.3",
                                         "filled_qty": "0.3"},),
            pending_order_ids_after=("close-pending",),
        ),
    )

    assert teacher.completed is False
    teacher.advise(_Observation(0.0, 180))
    assert delegate.advise_calls == 2


def test_first_round_trip_requires_native_close_fill_proof():
    delegate = _Teacher()
    teacher = FirstRoundTripTeacher(delegate)
    entry = teacher.advise(_Observation(0.0, 60))
    teacher.observe_native_action(
        entry,
        SimpleNamespace(actual_exposure_after=0.3, filled_order_ids=(),
                        pending_order_ids_after=()),
        SimpleNamespace(filled_order_ids=(), native_policy_fills=(),
                        native_order_statuses_after=(), pending_order_ids_after=()),
    )
    close = advice(_Observation(0.3, 120), delegate.expert_id, "EXIT", 0.0, "CLOSE")
    teacher.observe_native_action(
        close,
        SimpleNamespace(actual_exposure_after=0.0, filled_order_ids=(),
                        pending_order_ids_after=()),
        SimpleNamespace(filled_order_ids=(), native_policy_fills=(),
                        native_order_statuses_after=(), pending_order_ids_after=()),
    )

    assert teacher.completed is False
    teacher.advise(_Observation(0.0, 180))
    assert delegate.advise_calls == 2


def test_trade_path_metrics_reconcile_cash_and_peak_giveback():
    case = {
        "trade_case_id": "case-1",
        "status": "ORDINARY_COMPLETE",
        "entry_fill_ts_ns": 100,
        "exit_fill_ts_ns": 300,
        "entry_notional": 1000.0,
        "equity_start": 1000.0,
        "realized_pnl": 6.0,
        "fees": 4.0,
        "funding": 0.0,
        "slippage_cost": 1.0,
        "mfe_bps": 100.0,
        "mae_bps": 10.0,
        "holding_minutes": 2.0,
    }
    decisions = [
        {"execution_ts_ns": "100", "position_before": "0", "native_exposure_after": "0.3",
         "fee": "2", "funding": "0", "account_equity": "998",
         "case_id": "", "case_status": "NONE"},
        {"execution_ts_ns": "200", "position_before": "0.3", "native_exposure_after": "0.3",
         "fee": "0", "funding": "0", "account_equity": "1012",
         "case_id": "", "case_status": "NONE"},
        {"execution_ts_ns": "300", "position_before": "0.3", "native_exposure_after": "0",
         "fee": "2", "funding": "0", "account_equity": "1006",
         "case_id": "case-1", "case_status": "ORDINARY_COMPLETE"},
    ]

    metrics = _path_metrics(case, decisions)

    assert metrics["cash_identity_verified"] is True
    assert metrics["gross_cash_pnl"] == 10.0
    assert metrics["net_realized_pnl"] == 6.0
    assert metrics["max_closed_bar_peak_giveback_usdt"] == 4.0
    assert metrics["gross_profit_capture_ratio"] == 1.0


def test_case_pairing_requires_same_timestamp_side_and_rounded_notional():
    case = {"entry_fill_ts_ns": 100, "side": 1, "entry_notional": 100.0000004}
    assert _case_identity(case) == (100, 1, 100.0)

    incomplete = {"case": None, "trade_diagnostics": None}
    paired = _pair_slot({"teacher": "T01"}, incomplete, incomplete)
    assert paired["pairing_status"] == "UNPAIRED"
    assert paired["unpaired_reason"] == "NO_FIRST_NATIVE_TRADE_CASE"


def test_case_pairing_accepts_same_native_entry_and_reports_cash_delta():
    case1 = {
        "status": "ORDINARY_COMPLETE",
        "entry_fill_ts_ns": 100,
        "side": 1,
        "entry_notional": 100.0,
    }
    case2 = dict(case1)
    metrics1 = {
        "gross_cash_pnl": 4.0, "net_realized_pnl": 2.0, "fees": 2.0,
        "slippage_attribution": 0.5, "funding": 0.0, "mfe_bps": 80.0,
        "mae_bps": 20.0, "holding_minutes": 3.0,
        "max_closed_bar_peak_giveback_usdt": 1.0,
    }
    metrics2 = {**metrics1, "gross_cash_pnl": 5.0, "net_realized_pnl": 3.0}
    row = _pair_slot(
        {"teacher": "T01"},
        {"case": case1, "trade_diagnostics": metrics1},
        {"case": case2, "trade_diagnostics": metrics2},
    )

    assert row["pairing_status"] == "PAIRED_SAME_NATIVE_ENTRY"
    assert row["x2_minus_x1"]["gross_cash_pnl"] == 1.0
    assert row["x2_minus_x1"]["net_realized_pnl"] == 1.0
