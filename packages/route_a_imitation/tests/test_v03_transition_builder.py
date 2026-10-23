from math import log
from dataclasses import replace
import pytest

from rles_route_a.contracts import PointInTimeObservation, ExpertAdvice
from rles_route_a.integration.native_transition_builder import (
    EncodedNativeState,native_nonterminal_transition,native_terminal_transition,
)
from rles_route_a.integration.host_gym_projection import (
    HostGymProjection,L3OrderFacts,TerminalL6Receipt,project_nonterminal_teacher_barrier,
)
from rles_route_a.integration.native_economics import NativeRewardReceipt
from rles_route_a.datasets.daggerset import DAggerSet,AggregatedLabel
from rles_route_a.datasets.sequence_dataset import LabeledStep
from rles_route_a.teachers.wait import WaitTeacher

STEP=60_000_000_000


def frame(timestamp,pos=0.,episode="ep",fold="f"):
    return PointInTimeObservation(episode,timestamp,{"return_1m":.2},
                                  {"return_1m":timestamp-1},{"net_exposure":pos},fold_id=fold)


def barrier(o,after,planner_target=.5):
    a=ExpertAdvice("teacher",o.decision_ts_ns,"OPEN_LONG",.5,("TRAIN_INNER",))
    info=dict(decision_ts_ns=after.decision_ts_ns,
              terminal_settled=False,execution_plan_action="SUBMIT",
              execution_client_order_id="RLES-entry",execution_decision_id="decision-1",
              execution_plan_id="plan-1",execution_idempotency_key="id1",
              execution_plan_reason="target-changed",
              evidence_hash="hash1",policy_equity_currency="USDT",requested_action=.5,
              risk_adjusted_target=planner_target,quantity_rounded_target_exposure=planner_target,
              policy_executed_action_pre_terminal=after.account_value("net_exposure"),
              policy_equity_before="1000",policy_equity_after="999",
              policy_equity_change="-1",r_econ=log(999/1000),
              native_pending_order_ids_before=o.pending_order_ids,
              native_pending_order_ids_after=())
    evidence=L3OrderFacts(("RLES-entry",),("RLES-entry",),native_evidence_receipt="l3")
    return project_nonterminal_teacher_barrier(observation=o,advice=a,
        gym_info=info,l3_orders=evidence,execution_recipe="exec",source_identity="source",
        no_external_cashflow_receipt="synthetic-L6-no-flow")


def encoded(o,vector=(.4,.5),source="source"):
    return EncodedNativeState(o,vector,"encoder-1",source)


def test_native_transition_trains_behavior_command_and_retains_planner_mapping():
    a,b=frame(3*STEP,0.),frame(4*STEP,.3)
    p=barrier(a,b,planner_target=.25)
    t=native_nonterminal_transition(encoded(a),encoded(b),p,
                                    decision_period_ns=STEP,sample_id="s1")
    assert t.requested_action==.5
    assert t.executed_target==t.action_trace.planner_target
    assert t.executed_target==pytest.approx(.25)
    assert t.next_state==(.4,.5)
    assert t.native_log_reward==pytest.approx(log(.999))
    assert t.discount_steps==1 and t.bootstrap_allowed
    assert t.native_execution_recipe=="exec" and t.fold_id=="f"
    assert t.action_trace == p.teacher_barrier.trace
    assert t.action_trace.planner_action == "SUBMIT"
    assert t.action_trace.submitted_order_ids == ("RLES-entry",)

    from rles_route_a.datasets.transitions import OfflineBatch
    assert OfflineBatch((t,)).action[0, 0].item() == pytest.approx(.5)
    with pytest.raises(ValueError, match="differs from native planner target"):
        from dataclasses import replace
        replace(t, executed_target=t.requested_action)


def test_native_transition_rejects_clock_and_account_disagreement():
    a,b=frame(3*STEP,0.),frame(4*STEP,.3)
    p=barrier(a,b)
    with pytest.raises(ValueError,match="post-advance clock"):
        native_nonterminal_transition(encoded(a),encoded(frame(5*STEP,.3)),p,
                                      decision_period_ns=STEP,sample_id="s1")
    with pytest.raises(ValueError,match="executed exposure"):
        native_nonterminal_transition(encoded(a),encoded(frame(4*STEP,.1)),p,
                                      decision_period_ns=STEP,sample_id="s1")
    with pytest.raises(PermissionError,match="source/encoder"):
        native_nonterminal_transition(encoded(a),encoded(b,source="other"),p,
                                      decision_period_ns=STEP,sample_id="s1")
    with pytest.raises(PermissionError,match="Fold/episode"):
        native_nonterminal_transition(encoded(a),encoded(frame(4*STEP,.3,episode="other")),p,
                                      decision_period_ns=STEP,sample_id="s1")
    altered_trace=replace(p.teacher_barrier.trace,pending_order_ids_after=("pending-1",))
    altered=replace(p,teacher_barrier=replace(p.teacher_barrier,trace=altered_trace))
    with pytest.raises(ValueError,match="pending orders disagree"):
        native_nonterminal_transition(encoded(a),encoded(b),altered,
                                      decision_period_ns=STEP,sample_id="pending-mismatch")


def test_native_terminal_transition_requires_l6_and_true_settled_post_state():
    before=frame(3*STEP,0.)
    policy_after=frame(4*STEP,.3)
    projected=barrier(before,policy_after)
    receipt=TerminalL6Receipt(
        receipt_id="terminal-l6",source_identity="source",settlement_sha256="a"*64,
        policy_barrier_ts_ns=before.decision_ts_ns,
        settlement_barrier_ts_ns=policy_after.decision_ts_ns,
        equity_before=1000.,policy_equity_after=999.,settled_equity_after=990.,
        currency="USDT",gym_recorded_r_econ=log(.999),recorded_r_econ=log(.99),
    )
    barrier_with_l6=replace(
        projected.teacher_barrier,
        reward=NativeRewardReceipt(1000.,990.,log(.99),False),
        terminal_l6_receipt=receipt.receipt_id,
    )
    native=HostGymProjection(barrier_with_l6,policy_after.decision_ts_ns,"USDT","l3")
    post=frame(4*STEP,0.)
    settled=EncodedNativeState(post,(.4,.5),"encoder-1","source",receipt.receipt_id)
    row=native_terminal_transition(
        encoded(before),settled,native,receipt,decision_period_ns=STEP,
        sample_id="terminal-sample",terminated=True,
    )
    assert row.next_state==encoded(post).vector
    assert row.native_log_reward==pytest.approx(log(.99))
    assert row.terminated and not row.bootstrap_allowed
    assert row.action_trace==barrier_with_l6.trace
    with pytest.raises(ValueError,match="settled flat account"):
        native_terminal_transition(
            encoded(before),EncodedNativeState(policy_after,(.4,.5),"encoder-1","source",receipt.receipt_id),
            native,receipt,decision_period_ns=STEP,
            sample_id="not-flat",terminated=True,
        )
    with pytest.raises(ValueError,match="cannot retain pending"):
        unsettled = EncodedNativeState(
            replace(post,pending_order_ids=("pending-after-settlement",)),
            (.4,.5),"encoder-1","source",receipt.receipt_id,
        )
        native_terminal_transition(
            encoded(before),unsettled,native,receipt,decision_period_ns=STEP,
            sample_id="pending-terminal",terminated=True,
        )
    with pytest.raises(ValueError,match="L6 receipt"):
        native_terminal_transition(
            encoded(before),settled,native,replace(receipt,receipt_id="other"),
            decision_period_ns=STEP,sample_id="wrong-receipt",terminated=True,
        )
    with pytest.raises(ValueError,match="L6 receipt"):
        native_terminal_transition(
            encoded(before),encoded(post),native,receipt,decision_period_ns=STEP,
            sample_id="unbound-post-state",terminated=True,
        )


def test_dagger_does_not_allow_recipe_drift_between_rounds():
    o=frame(3*STEP)
    base=LabeledStep(o,WaitTeacher().advise(o),0.,source_identity="source")
    d=DAggerSet((base,))
    d.append(AggregatedLabel(1,"student",base,"frozen-L3-1"))
    another=frame(4*STEP)
    sample=LabeledStep(another,WaitTeacher().advise(another),0.,source_identity="source")
    with pytest.raises(ValueError,match="mix projection"):
        d.append(AggregatedLabel(2,"student",sample,"changed-L3"))
    assert d.collected==1
