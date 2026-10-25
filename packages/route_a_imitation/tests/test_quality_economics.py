import math
import pytest
from rles_route_a.contracts import NativeActionTrace
from rles_route_a.teachers.quality import teacher_quality
from rles_route_a.integration.exit_lifecycle import verify_native_flat_lifecycle
from rles_route_a.integration.native_economics import NativeRewardReceipt
from rles_route_a.evaluation.diagnostics import summarize_actions


def trace(ts, req, actual, action, *, submitted=(), filled=(), existing=(), pending=()):
    return NativeActionTrace(ts, req, req, req, actual, 1000, 999, action,
                             submitted_order_ids=submitted, filled_order_ids=filled,
                             related_existing_order_ids=existing,
                             pending_order_ids_after=pending)


def test_teacher_quality_rejects_deadband_only_alpha():
    rows = [trace(1, .01, 0., "NOOP"), trace(2, -.01, 0., "NOOP")]
    q = teacher_quality("trend", rows, alpha_teacher=True)
    assert q.engine_trace_status == "FAIL"
    q2 = teacher_quality("wait", rows, alpha_teacher=False)
    assert q2.engine_trace_status == "ENGINEERING_PASS"
    assert q2.economic_status == "SCENARIO_ONLY"


def test_teacher_quality_uses_frozen_reachability_tolerance_for_commands():
    rows = [trace(1, .01, 0., "NOOP"), trace(2, .03, 0., "NOOP")]
    result = teacher_quality("trend", rows, alpha_teacher=True, tolerance=.02)
    assert result.nonzero_command_count == 1
    assert result.engine_trace_status == "FAIL"


def test_teacher_quality_counts_only_completed_fills_from_current_submit():
    prior_order_fill = trace(
        1, .5, .5, "SUBMIT", submitted=("new-order",),
        filled=("old-order",), existing=("old-order",),
    )
    q_prior = teacher_quality("trend", [prior_order_fill], alpha_teacher=True)
    assert q_prior.engine_trace_status == "FAIL"
    assert q_prior.native_fill_count == 0

    partial = trace(
        1, .5, .2, "SUBMIT", submitted=("new-order",),
        filled=("new-order",), pending=("new-order",),
    )
    q_partial = teacher_quality("trend", [partial], alpha_teacher=True)
    assert q_partial.engine_trace_status == "FAIL"
    assert q_partial.native_fill_count == 0

    complete = trace(
        1, .5, .5, "SUBMIT", submitted=("new-order",), filled=("new-order",),
    )
    q_complete = teacher_quality("trend", [complete], alpha_teacher=True)
    assert q_complete.engine_trace_status == "ENGINEERING_PASS"
    assert q_complete.native_fill_count == 1


def test_native_fills_preexisting_orders():
    t = trace(2, 0., 0., "WAIT", existing=("old_order",), filled=("old_order",))
    assert t.filled_order_ids == ("old_order",)
    with pytest.raises(ValueError, match="provenance"):
        trace(2, 0., 0., "WAIT", filled=("unknown",))


def test_flat_lifecycle_wait_cancel_ack_close():
    rows = [trace(1, 0., .4, "CANCEL", existing=("entry",), pending=("entry",)),
            trace(2, 0., .4, "WAIT", pending=()),
            trace(3, 0., 0., "SUBMIT", submitted=("close",), filled=("close",))]
    assert verify_native_flat_lifecycle(rows, initial_abs_exposure=.4)
    assert not verify_native_flat_lifecycle(rows[:2], initial_abs_exposure=.4)


def test_flat_lifecycle_accepts_delayed_fill_of_its_submitted_close():
    rows = [trace(1, 0., .4, "SUBMIT", submitted=("close",), pending=("close",)),
            trace(2, 0., 0., "WAIT", filled=("close",), existing=("close",))]
    assert verify_native_flat_lifecycle(rows, initial_abs_exposure=.4)


def test_flat_lifecycle_rejects_changed_risk_or_planner_target():
    rows = [trace(1, 0., 0., "SUBMIT", submitted=("close",), filled=("close",))]
    rows[0] = NativeActionTrace(
        decision_ts_ns=1, requested_action=0., risk_adjusted_target=.2,
        planner_target=0., actual_exposure_after=0., native_equity_before=1000.,
        native_equity_after=999., planner_action="SUBMIT",
        submitted_order_ids=("close",), filled_order_ids=("close",),
    )
    with pytest.raises(ValueError, match="risk or planner"):
        verify_native_flat_lifecycle(rows, initial_abs_exposure=.4)


def test_flat_lifecycle_requires_finite_initial_exposure_and_tolerance():
    rows = [trace(1, 0., 0., "SUBMIT", submitted=("close",), filled=("close",))]
    with pytest.raises(ValueError, match="nonflat initial"):
        verify_native_flat_lifecycle(rows, initial_abs_exposure=float("nan"))
    with pytest.raises(ValueError, match="tolerance"):
        verify_native_flat_lifecycle(rows, initial_abs_exposure=.4, tolerance=float("inf"))


def test_native_reward_conservation():
    r = NativeRewardReceipt(1000., 999., math.log(999/1000), False)
    r.validate()
    with pytest.raises(ValueError, match="conservation"):
        NativeRewardReceipt(1000., 900., 0.0, False).validate()
    with pytest.raises(ValueError, match="cashflow"):
        NativeRewardReceipt(1000., 1100., 0.0, True).validate()
    NativeRewardReceipt(1000., 1100., 0.0, True, "l6-receipt-1").validate()


def test_native_action_summary_flags_flat_and_flipping_policies():
    flat = summarize_actions([trace(1, 0., 0., "NOOP"), trace(2, 0., 0., "NOOP")])
    assert flat["request_std"] == 0.
    assert flat["request_flat_fraction"] == 1.
    assert flat["actual_flat_fraction"] == 1.
    assert flat["request_direct_reversal_fraction"] == 0.

    flipping = summarize_actions([
        trace(1, .4, .2, "SUBMIT"), trace(2, -.4, -.2, "SUBMIT"),
    ])
    assert flipping["request_std"] == pytest.approx(.4)
    assert flipping["request_flat_fraction"] == 0.
    assert flipping["request_direct_reversal_fraction"] == 1.
    assert flipping["actual_exposure_std"] == pytest.approx(.2)
    with pytest.raises(ValueError, match="threshold"):
        summarize_actions([trace(1, 0., 0., "NOOP")], meaningful_delta=0.)
