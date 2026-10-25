"""Synthetic host-shape fixtures only; no claim of actual Nautilus integration."""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import json
from math import log
from types import SimpleNamespace
import pytest

from rles_route_a.contracts import NativeActionTrace, ExpertAdvice
from rles_route_a.datasets.sequence_dataset import EpisodeSequenceDataset
from rles_route_a.integration.host_gym_projection import (
    L3OrderFacts, TerminalL6Receipt, project_nonterminal_teacher_barrier,
    project_terminal_teacher_barrier, terminal_settlement_sha256,
)
from rles_route_a.integration.native_rollout import TeacherFact, native_teacher_step
from rles_route_a.integration.trace_audit import audit_teacher_barriers
from rles_route_a.integration.gym_rollout import NativeGymTeacherProvider
from rles_route_a.integration.exit_lifecycle import verify_native_flat_lifecycle
from rles_route_a.integration.m22_trace import (
    verify_m22_decision_trace,
    verify_m22_order_facts,
)
from rles_route_a.evaluation.teacher_gate import NativePair, screen_teacher
from rles_route_a.teachers.wait import WaitTeacher


def info(ts, req=0.5, action="SUBMIT"):
    return dict(decision_ts_ns=ts+60, terminal_settled=False,
        execution_plan_action=action,
        execution_client_order_id="RLES-order1" if action=="SUBMIT" else None,
        execution_decision_id=f"decision-{ts}", execution_plan_id=f"plan-{ts}",
        execution_idempotency_key="idem-1", execution_plan_reason="target-changed",
        evidence_hash="native-proof-1",
        policy_equity_currency="USDT",
        requested_action=req, risk_soft_target=req * .8, risk_adjusted_target=req,
        quantity_rounded_target_exposure=req,
        policy_executed_action_pre_terminal=req,
        policy_equity_before="1000", policy_equity_after="999.9",
        policy_equity_change="-0.1", r_econ=log(999.9/1000),
        execution_cancel_order_ids=(), execution_related_order_ids=(),
        native_pending_order_ids_before=(), native_pending_order_ids_after=())


def advice(o, req=.5):
    return ExpertAdvice("trend", o.decision_ts_ns,
        "OPEN_LONG" if req else "EXIT", req, ("SYNTHETIC_NATIVE",))


def projection(o, gym=None, orders=None, req=.5, scenario_only=True):
    facts = orders or L3OrderFacts(("RLES-order1",), ("RLES-order1",),
                                   native_evidence_receipt="l3-proof")
    gym_info = gym or info(o.decision_ts_ns, req=req)
    if gym is None:
        gym_info["native_pending_order_ids_before"] = o.pending_order_ids
        gym_info["native_pending_order_ids_after"] = facts.pending_order_ids_after
    return project_nonterminal_teacher_barrier(
        observation=o, advice=advice(o, req), gym_info=gym_info,
        l3_orders=facts,
        execution_recipe="frozen-exec", source_identity="frozen-train",
        no_external_cashflow_receipt="l6-no-flow-synthetic",scenario_only=scenario_only)


def test_gym_pre_and_post_decision_clocks_are_distinct(observed):
    obs=observed(ts=2_000_000_000,pos=0.)
    projected=projection(obs)
    assert projected.teacher_barrier.trace.decision_ts_ns == 2_000_000_000
    assert projected.teacher_barrier.trace.soft_risk_target == pytest.approx(.4)
    assert projected.advance_ts_ns == 2_000_000_060
    labels, summary=audit_teacher_barriers((projected.teacher_barrier,))
    assert labels[0].action_status == "FILLED"
    assert labels[0].economic_case_status == "OPEN_CASE_CENSORED"
    assert labels[0].case_outcome_bps is None
    assert labels[0].case_end_ts_ns is None
    assert labels[0].case_censor_ts_ns == projected.teacher_barrier.trace.execution_barrier_ts_ns
    assert labels[0].supervision_weight == 0
    assert labels[0].intent_supervision_weight == 0
    assert summary.economic_status == "SCENARIO_ONLY"
    assert summary.executable_label_count == 0
    assert summary.unique_filled_order_refs == 1
    assert summary.episode_equity_segments == ((obs.episode_id, 1000., 999.9),)


def test_native_economic_class_requires_separate_authority(observed):
    obs = observed(ts=2_000_000_000, pos=0.)
    raw = info(obs.decision_ts_ns)
    facts = L3OrderFacts(("RLES-order1",), ("RLES-order1",),
                         native_evidence_receipt="l3-proof")
    common = dict(
        env_factory=lambda: None, teacher=WaitTeacher(), fold_id=obs.fold_id,
        execution_recipe="frozen-exec", source_identity="frozen-train",
        native_order_facts_verifier=lambda *_: True,
        no_external_cashflow_receipt=lambda _info: "l6-no-flow",
        no_external_cashflow_verifier=lambda *_: True,
        episode_id=lambda *_: obs.episode_id,
    )
    scenario_provider = NativeGymTeacherProvider(**common)
    scenario = scenario_provider._project_step(
        obs, advice(obs), raw, facts, terminated=False, truncated=False,
        trajectory_id="scenario-run",
    )
    assert scenario.trace.scenario_only is True
    assert scenario.native_economic_evidence_authority is None

    observed_provider = NativeGymTeacherProvider(
        **common, native_economic_evidence_verifier=lambda *_: True,
    )
    native = observed_provider._project_step(
        obs, advice(obs), raw, facts, terminated=False, truncated=False,
        trajectory_id="native-run",
    )
    assert native.trace.scenario_only is False
    assert native.native_economic_evidence_authority == (
        "test_v03_host_projection.test_native_economic_class_requires_separate_authority.<locals>.<lambda>"
    )

    rejected_provider = NativeGymTeacherProvider(
        **common, native_economic_evidence_verifier=lambda *_: False,
    )
    with pytest.raises(PermissionError, match="observed market/execution"):
        rejected_provider._project_step(
            obs, advice(obs), raw, facts, terminated=False, truncated=False,
            trajectory_id="rejected-run",
        )


def test_positive_native_step_pnl_does_not_close_open_trade_case(observed):
    obs = observed(ts=2_000_000_000, pos=0.)
    gym = info(obs.decision_ts_ns, req=.5)
    gym["policy_equity_after"] = "1000.1"
    gym["policy_equity_change"] = "0.1"
    gym["r_econ"] = log(1000.1 / 1000.)
    barrier = projection(obs, gym=gym, scenario_only=False).teacher_barrier
    labels, summary = audit_teacher_barriers((barrier,))
    case = labels[0]
    assert case.economic_case_status == "OPEN_CASE_CENSORED"
    assert case.case_outcome_bps is None
    assert case.case_end_ts_ns is None
    assert case.case_censor_ts_ns == barrier.trace.execution_barrier_ts_ns
    assert len(case.case_fact_sha256) == 64
    assert case.supervision_weight == 0
    assert case.intent_supervision_weight == 0
    assert summary.executable_label_count == 0


def test_negative_native_step_pnl_does_not_close_open_trade_case(observed):
    obs = observed(ts=2_000_000_000, pos=0.)
    barrier = projection(obs, scenario_only=False).teacher_barrier
    labels, summary = audit_teacher_barriers((barrier,))
    case = labels[0]
    assert case.economic_case_status == "OPEN_CASE_CENSORED"
    assert case.case_outcome_bps is None
    assert case.case_end_ts_ns is None
    assert case.case_censor_ts_ns == barrier.trace.execution_barrier_ts_ns
    assert len(case.case_fact_sha256) == 64
    assert case.supervision_weight == 0
    assert case.intent_supervision_weight == 0
    assert summary.executable_label_count == 0


def test_completed_reconciled_fee_inclusive_trade_becomes_teaching_case(observed):
    rows = []
    decisions = (
        (0.0, "SUBMIT", .5, 1000.0, 1000.1),
        (.5, "NOOP", .5, 1000.1, 1000.3),
        (.5, "SUBMIT", 0.0, 1000.3, 1000.2),
        (0.0, "SUBMIT", .5, 1000.2, 1000.4),
        (.5, "NOOP", .5, 1000.4, 1000.5),
        (.5, "NOOP", .5, 1000.5, 1000.7),
        (.5, "SUBMIT", 0.0, 1000.7, 1000.6),
    )
    for offset, (position, action, req, before, after) in enumerate(decisions):
        obs = observed(ts=2_000_000_000 + offset * 60_000_000_000,
                       pos=position, episode="round-trip")
        gym = info(obs.decision_ts_ns, req=req, action=action)
        gym["policy_equity_before"] = str(before)
        gym["policy_equity_after"] = str(after)
        gym["policy_equity_change"] = str(after - before)
        gym["r_econ"] = log(after / before)
        order_ref = f"RLES-order{offset + 1}"
        if action == "SUBMIT":
            gym["execution_client_order_id"] = order_ref
        facts = (L3OrderFacts((order_ref,), (order_ref,),
                              native_evidence_receipt=f"l3-{offset}")
                 if action == "SUBMIT" else
                 L3OrderFacts(native_evidence_receipt=f"l3-{offset}"))
        projected = projection(obs, gym=gym, orders=facts, req=req,
                               scenario_only=False).teacher_barrier
        rows.append(replace(
            projected,
            native_order_lifecycle_verified=True,
            native_economic_evidence_authority="test.host_l6_m2_2_native_economics",
        ))

    labels, summary = audit_teacher_barriers(tuple(rows))

    assert [row.economic_case_status for row in labels] == [
        "OBSERVED_NATIVE_TRADE_POSITIVE",
        "OBSERVED_NATIVE_TRADE_POSITIVE",
        "OBSERVED_NATIVE_TRADE_POSITIVE",
        "OBSERVED_NATIVE_TRADE_POSITIVE",
        "OBSERVED_NATIVE_TRADE_POSITIVE",
        "OBSERVED_NATIVE_TRADE_POSITIVE",
        "OBSERVED_NATIVE_TRADE_POSITIVE",
    ]
    # Keep the native per-decision label strength intact. Case balance belongs in
    # the sampler, since BC renormalizes supervision weights within each batch.
    assert [row.supervision_weight for row in labels] == pytest.approx([1.] * 7)
    assert [row.intent_supervision_weight for row in labels] == pytest.approx([1.] * 7)
    assert sum(row.supervision_weight for row in labels[:3]) == pytest.approx(3)
    assert sum(row.supervision_weight for row in labels[3:]) == pytest.approx(4)
    assert all(row.case_outcome_bps == pytest.approx(log(1000.2 / 1000.) * 10_000)
               for row in labels[:3])
    assert all(row.case_outcome_bps == pytest.approx(log(1000.6 / 1000.2) * 10_000)
               for row in labels[3:])
    assert all(row.case_end_ts_ns == rows[2].trace.execution_barrier_ts_ns
               for row in labels[:3])
    assert all(row.case_end_ts_ns == rows[6].trace.execution_barrier_ts_ns
               for row in labels[3:])
    assert len({row.case_fact_sha256 for row in labels[:3]}) == 1
    assert len({row.case_fact_sha256 for row in labels[3:]}) == 1
    assert labels[0].case_fact_sha256 != labels[3].case_fact_sha256
    assert summary.economic_status == "OBSERVED_NATIVE_TRADE_CASES"
    assert summary.executable_label_count == 7
    assert summary.completed_native_trade_case_count == 2
    assert summary.positive_native_trade_case_count == 2
    assert summary.negative_native_trade_case_count == 0
    assert summary.neutral_native_trade_case_count == 0

    unverified, unverified_summary = audit_teacher_barriers(tuple(
        replace(row, native_order_lifecycle_verified=False) for row in rows
    ))
    assert all(row.economic_case_status.startswith("OBSERVED_NATIVE_STEP_")
               for row in unverified)
    assert unverified_summary.executable_label_count == 0
    assert unverified_summary.completed_native_trade_case_count == 0


def test_open_native_trade_at_episode_end_is_censored_not_zero(observed):
    rows = []
    decisions = (
        (0.0, "SUBMIT", .5, 1000.0, 1000.1),
        (.5, "NOOP", .5, 1000.1, 1000.3),
    )
    for offset, (position, action, req, before, after) in enumerate(decisions):
        obs = observed(ts=8_000_000_000 + offset * 60_000_000_000,
                       pos=position, episode="open-at-fit-end")
        gym = info(obs.decision_ts_ns, req=req, action=action)
        gym["policy_equity_before"] = str(before)
        gym["policy_equity_after"] = str(after)
        gym["policy_equity_change"] = str(after - before)
        gym["r_econ"] = log(after / before)
        order_ref = f"RLES-open-{offset}"
        if action == "SUBMIT":
            gym["execution_client_order_id"] = order_ref
            facts = L3OrderFacts((order_ref,), (order_ref,),
                                 native_evidence_receipt=f"l3-open-{offset}")
        else:
            facts = L3OrderFacts(native_evidence_receipt=f"l3-open-{offset}")
        barrier = projection(
            obs, gym=gym, orders=facts, req=req, scenario_only=False,
        ).teacher_barrier
        rows.append(replace(
            barrier,
            native_order_lifecycle_verified=True,
            native_economic_evidence_authority="test.host_l6_m2_2_native_economics",
        ))

    labels, summary = audit_teacher_barriers(tuple(rows))

    assert [row.economic_case_status for row in labels] == [
        "OPEN_CASE_CENSORED", "OPEN_CASE_CENSORED",
    ]
    assert all(row.case_outcome_bps is None for row in labels)
    assert all(row.case_end_ts_ns is None for row in labels)
    assert all(row.case_censor_ts_ns == rows[-1].trace.execution_barrier_ts_ns
               for row in labels)
    assert len({row.case_fact_sha256 for row in labels}) == 1
    assert summary.completed_native_trade_case_count == 0
    assert summary.open_case_censored_count == 1

    dataset = EpisodeSequenceDataset(
        labels,
        feature_names=("market.trend_1h", "account.net_exposure"),
        context_bars=2,
    )
    last = dataset[1]
    assert last["case_outcome_bps"].item() == 0.0  # masked placeholder only
    assert last["case_assessed"].item() is False
    assert last["case_outcome_mask"].item() is False
    assert last["case_censored"].item() is True
    assert last["case_censor_ts_ns"].item() == rows[-1].trace.execution_barrier_ts_ns

    with pytest.raises(ValueError, match="never an outcome"):
        replace(labels[0], case_outcome_bps=0.0)

    unverified, unverified_summary = audit_teacher_barriers(tuple(
        replace(row, native_order_lifecycle_verified=False) for row in rows
    ))
    assert all(row.economic_case_status == "OPEN_CASE_CENSORED" for row in unverified)
    assert all(row.case_outcome_bps is None for row in unverified)
    assert unverified_summary.completed_native_trade_case_count == 0
    assert unverified_summary.open_case_censored_count == 1


def test_scenario_terminal_force_flat_keeps_open_case_censored(observed):
    obs = observed(ts=2_000_000_000, pos=0., episode="scenario-terminal")
    entry = projection(obs, scenario_only=True).teacher_barrier
    terminal = replace(
        entry,
        terminal_l6_receipt="l6-scenario-terminal",
        terminal_settlement_ts_ns=entry.trace.execution_barrier_ts_ns + 1,
        native_order_lifecycle_verified=True,
    )

    labels, summary = audit_teacher_barriers((terminal,))

    assert labels[0].economic_case_status == "OPEN_CASE_CENSORED"
    assert labels[0].case_outcome_bps is None
    assert labels[0].case_end_ts_ns is None
    assert labels[0].case_censor_ts_ns == entry.trace.execution_barrier_ts_ns
    assert summary.open_case_censored_count == 1
    assert summary.completed_native_trade_case_count == 0
    assert summary.executable_label_count == 0


def test_completed_native_loss_teaches_flat_only_at_entry(observed):
    rows = []
    decisions = (
        (0.0, "SUBMIT", .5, 1000.0, 999.9),
        (.5, "NOOP", .5, 999.9, 999.8),
        (.5, "SUBMIT", 0.0, 999.8, 999.7),
    )
    for offset, (position, action, req, before, after) in enumerate(decisions):
        obs = observed(ts=2_000_000_000 + offset * 60_000_000_000,
                       pos=position, episode="losing-round-trip")
        gym = info(obs.decision_ts_ns, req=req, action=action)
        gym["policy_equity_before"] = str(before)
        gym["policy_equity_after"] = str(after)
        gym["policy_equity_change"] = str(after - before)
        gym["r_econ"] = log(after / before)
        order_ref = f"RLES-loss-order{offset + 1}"
        if action == "SUBMIT":
            gym["execution_client_order_id"] = order_ref
        facts = (L3OrderFacts((order_ref,), (order_ref,),
                              native_evidence_receipt=f"l3-loss-{offset}")
                 if action == "SUBMIT" else
                 L3OrderFacts(native_evidence_receipt=f"l3-loss-{offset}"))
        projected = projection(obs, gym=gym, orders=facts, req=req,
                               scenario_only=False).teacher_barrier
        rows.append(replace(
            projected,
            native_order_lifecycle_verified=True,
            native_economic_evidence_authority="test.host_l6_m2_2_native_economics",
        ))

    labels, summary = audit_teacher_barriers(tuple(rows))

    assert summary.completed_native_trade_case_count == 1
    assert summary.negative_native_trade_case_count == 1
    assert summary.positive_native_trade_case_count == 0
    assert summary.negative_entry_avoidance_label_count == 1
    assert [row.economic_case_status for row in labels] == [
        "OBSERVED_NATIVE_TRADE_NEGATIVE",
    ] * 3
    assert labels[0].advice.intent == "FLAT"
    assert labels[0].advice.requested_exposure == 0.0
    assert labels[0].label_origin == "INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE"
    assert "negative_native_trade_entry_avoidance" in labels[0].advice.reason_codes
    assert labels[0].intent_supervision_weight == 1.0
    assert labels[0].supervision_weight == 0.0
    assert all(row.intent_supervision_weight == 0.0 for row in labels[1:])
    assert all(row.supervision_weight == 0.0 for row in labels)


def test_policy_barrier_pending_orders_match_teacher_context_and_l3_facts(observed):
    obs = observed(ts=2_000_000_000, pending=("open-1",))
    pre_mismatch = info(obs.decision_ts_ns)
    with pytest.raises(ValueError, match="pre-barrier pending orders differ"):
        projection(obs, gym=pre_mismatch)

    empty_context = observed(ts=2_000_000_000)
    after_mismatch = info(empty_context.decision_ts_ns)
    after_mismatch["native_pending_order_ids_after"] = ("open-1",)
    with pytest.raises(ValueError, match="post-barrier pending orders differ"):
        projection(empty_context, gym=after_mismatch)

    malformed = info(empty_context.decision_ts_ns)
    malformed["native_pending_order_ids_before"] = ("open-1", "open-1")
    with pytest.raises(ValueError, match="invalid native Gym pre-barrier"):
        projection(empty_context, gym=malformed)


def test_m22_execution_decision_trace_joins_route_a_native_barrier(observed):
    o = observed(ts=2_000_000_000, pos=0.)
    native_barrier = projection(o).teacher_barrier
    trace = native_barrier.trace
    m22_row = {
        "decision_id": trace.execution_decision_id,
        "decision_ts_ns": trace.execution_barrier_ts_ns,
        "plan_id": trace.execution_plan_id,
        "idempotency_key": trace.execution_idempotency_key,
        "action": trace.planner_action,
        "reason": trace.planner_reason,
        "client_order_id": trace.submitted_order_ids[0],
        "related_order_ids": list(trace.related_existing_order_ids),
        "actor_requested_target": trace.requested_action,
        "soft_target": trace.soft_risk_target,
        "hard_risk_target": trace.risk_adjusted_target,
        "quantity_rounded_target": trace.planner_target,
        "executed_exposure": trace.actual_exposure_after,
        "policy_equity_before": trace.native_equity_before,
        "policy_equity_after": trace.native_equity_after,
        "policy_equity_change": trace.native_equity_after - trace.native_equity_before,
        "equity_currency": trace.fee_currency,
    }

    policy_equity = {trace.execution_decision_id: {
        "policy_equity_before": trace.native_equity_before,
        "policy_equity_after": trace.native_equity_after,
        "policy_equity_change": trace.native_equity_after - trace.native_equity_before,
    }}
    checked = verify_m22_decision_trace(
        (native_barrier,), (m22_row,), policy_barrier_equity_by_id=policy_equity,
    )

    assert checked.decisions == 1 and checked.submits == 1
    assert checked.related_order_references == 0
    with pytest.raises(ValueError, match="actor_requested_target"):
        verify_m22_decision_trace(
            (native_barrier,), ({**m22_row, "actor_requested_target": .25},),
            policy_barrier_equity_by_id=policy_equity,
        )
    with pytest.raises(ValueError, match="policy_equity_after"):
        verify_m22_decision_trace(
            (native_barrier,), ({**m22_row, "policy_equity_after": 987.},),
            policy_barrier_equity_by_id=policy_equity,
        )
    with pytest.raises(ValueError, match="exact match"):
        verify_m22_decision_trace(
            (native_barrier,), (m22_row, {**m22_row, "decision_id": "extra-decision"}),
            policy_barrier_equity_by_id=policy_equity,
        )


def test_m22_native_snapshots_reconcile_order_status_and_policy_fills(observed):
    o = observed(ts=2_000_000_000, pos=0.)
    status_after = ({
        "client_order_id": "RLES-order1", "status": "FILLED", "side": "BUY",
        "quantity": "1", "filled_qty": "1",
    },)
    status_receipt = sha256(json.dumps(
        {"before": (), "after": status_after}, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()
    fill = {
        "event_id": "fill-1", "client_order_id": "RLES-order1", "ts_event": o.decision_ts_ns + 30,
        "order_side": "BUY",
        "last_qty": "1", "last_px": "100", "commission": "0.1 USDT",
    }
    facts = L3OrderFacts(
        submitted_order_ids=("RLES-order1",), filled_order_ids=("RLES-order1",),
        native_evidence_receipt="host-m22-order-receipt",
        native_order_statuses_after=status_after,
        native_order_status_receipt=status_receipt,
        native_policy_fills=(fill,),
    )
    barrier = projection(o, orders=facts).teacher_barrier
    before = SimpleNamespace(
        decision_ts_ns=o.decision_ts_ns, all_orders=(), orders=(), fills=(),
    )
    after = SimpleNamespace(
        decision_ts_ns=o.decision_ts_ns + 60,
        all_orders=({
            "client_order_id": "RLES-order1", "status": "FILLED", "side": "BUY",
            "quantity": "1", "filled_qty": "1",
        },),
        orders=(), fills=(fill,),
    )

    verified = verify_m22_order_facts((barrier,), (facts,), (before, after))
    assert verified.barriers == 1
    assert verified.order_status_facts == 1
    assert verified.policy_fill_events == 1

    with pytest.raises(ValueError, match="provenance"):
        L3OrderFacts(
            submitted_order_ids=("RLES-order1",),
            native_evidence_receipt="bad-order-source",
            native_policy_fills=({**fill, "client_order_id": "unrelated-order"},),
        )

    mismatched = SimpleNamespace(
        decision_ts_ns=after.decision_ts_ns,
        all_orders=({**after.all_orders[0], "status": "CANCELED"},),
        orders=(), fills=(fill,),
    )
    with pytest.raises(ValueError, match="post-barrier order states"):
        verify_m22_order_facts((barrier,), (facts,), (before, mismatched))

    for invalid_field, invalid_value in (
            ("status", None), ("side", "LONG"), ("quantity", "0"),
            ("filled_qty", "1.1")):
        invalid = SimpleNamespace(
            decision_ts_ns=after.decision_ts_ns,
            all_orders=({**after.all_orders[0], invalid_field: invalid_value},),
            orders=(), fills=(fill,),
        )
        with pytest.raises(ValueError, match="M2.2 post-barrier"):
            verify_m22_order_facts((barrier,), (facts,), (before, invalid))

    for invalid_field, invalid_value in (
            ("order_side", "LONG"), ("last_qty", "0"),
            ("last_px", "NaN"), ("commission", "0.1")):
        invalid = SimpleNamespace(
            decision_ts_ns=after.decision_ts_ns,
            all_orders=after.all_orders, orders=(),
            fills=({**fill, invalid_field: invalid_value},),
        )
        with pytest.raises(ValueError, match="native fill"):
            verify_m22_order_facts((barrier,), (facts,), (before, invalid))


def test_terminal_projector_binds_l6_settlement_without_relabeling_forced_close(observed):
    from math import log

    obs = observed(ts=2_000_000_000)
    gym = info(obs.decision_ts_ns)
    policy_ts = gym["decision_ts_ns"]
    settlement_ts = policy_ts + 60
    settlement = {
        "policy_barrier_ts_ns": policy_ts,
        "settlement_barrier_ts_ns": settlement_ts,
        "equity_before": "999.9", "equity_after": "999.8",
        "equity_delta": "-0.1", "equity_currency": "USDT",
        "pre_terminal_net_exposure": 0.5,
        "post_terminal_open_position_count": 0,
        "orders": [],
        "fills": [{"client_order_id": "terminal-force-flat", "event_id": "close-1"}],
        "funding": [],
    }
    gym.update({
        "terminal_settled": True, "terminal_value_semantics": "finite_settlement_no_bootstrap_v1",
        "settlement_after_ns": settlement_ts, "terminal_settlement": settlement,
        "r_econ": log(999.8 / 1000),
    })
    facts = L3OrderFacts(("RLES-order1",), ("RLES-order1",),
                         native_evidence_receipt="policy-order-proof")
    receipt = TerminalL6Receipt(
        receipt_id="l6-terminal-1", source_identity="frozen-train",
        settlement_sha256=terminal_settlement_sha256(settlement),
        policy_barrier_ts_ns=policy_ts, settlement_barrier_ts_ns=settlement_ts,
        equity_before=1000., policy_equity_after=999.9, settled_equity_after=999.8,
        currency="USDT", gym_recorded_r_econ=gym["r_econ"],
        recorded_r_econ=gym["r_econ"],
    )

    projected = project_terminal_teacher_barrier(
        observation=obs, advice=advice(obs), gym_info=gym, l3_orders=facts,
        terminal_receipt=receipt, execution_recipe="frozen-exec",
        source_identity="frozen-train",
    )

    barrier = projected.teacher_barrier
    assert barrier.reward.next_equity == pytest.approx(999.8)
    assert barrier.trace.native_equity_after == pytest.approx(999.8)
    assert barrier.trace.filled_order_ids == ("RLES-order1",)
    assert barrier.terminal_l6_receipt == "l6-terminal-1"
    barrier.reward.validate()
    labels, summary = audit_teacher_barriers((barrier,))
    assert labels[0].action_status == "FILLED"
    assert summary.native_equity_end == pytest.approx(999.8)

    flow_receipt = replace(
        receipt, receipt_id="l6-terminal-flow-adjusted", recorded_r_econ=0.01,
        has_external_cashflow=True, cashflow_adjustment_receipt="l6-flow-adjustment-1",
    )
    flow_projected = project_terminal_teacher_barrier(
        observation=obs, advice=advice(obs), gym_info=gym, l3_orders=facts,
        terminal_receipt=flow_receipt, execution_recipe="frozen-exec",
        source_identity="frozen-train",
    )
    assert flow_projected.teacher_barrier.reward.recorded_r_econ == pytest.approx(0.01)
    flow_projected.teacher_barrier.reward.validate()


def test_terminal_projector_rejects_tampered_settlement(observed):
    obs = observed(ts=2_000_000_000)
    gym = info(obs.decision_ts_ns)
    gym.update({"terminal_settled": True, "terminal_settlement": {"equity_after": "1"}})
    receipt = TerminalL6Receipt(
        receipt_id="l6-terminal-1", source_identity="frozen-train",
        settlement_sha256="0" * 64, policy_barrier_ts_ns=gym["decision_ts_ns"],
        settlement_barrier_ts_ns=gym["decision_ts_ns"], equity_before=1000.,
        policy_equity_after=999.9, settled_equity_after=1., currency="USDT",
        gym_recorded_r_econ=0., recorded_r_econ=0.,
    )
    with pytest.raises(ValueError, match="does not bind Gym settlement"):
        project_terminal_teacher_barrier(
            observation=obs, advice=advice(obs), gym_info=gym,
            l3_orders=L3OrderFacts(("RLES-order1",), native_evidence_receipt="l3"),
            terminal_receipt=receipt, execution_recipe="frozen-exec",
            source_identity="frozen-train",
        )


def test_submit_cannot_be_inferred_to_have_filled(observed):
    obs=observed(ts=2_000_000_000)
    empty=L3OrderFacts(("RLES-order1",), native_evidence_receipt="l3-proof")
    p=projection(obs, orders=empty)
    label=native_teacher_step(TeacherFact(obs,advice(obs),p.teacher_barrier.trace))
    assert label.action_status=="UNFILLED" and label.supervision_weight==0


def test_previous_order_fill_does_not_validate_current_order(observed):
    o=observed(ts=3_000_000_000)
    facts=L3OrderFacts(("RLES-order1",), ("old",), ("old",), native_evidence_receipt="l3")
    p=projection(o,orders=facts)
    label=native_teacher_step(TeacherFact(o, advice(o), p.teacher_barrier.trace))
    assert label.action_status == "UNFILLED" and label.supervision_weight==0


def test_partial_fill_and_pending_do_not_become_full_labels(observed):
    o=observed(ts=3_000_000_000)
    partial=L3OrderFacts(("RLES-order1",),("RLES-order1",),(),("RLES-order1",),"l3")
    label=native_teacher_step(TeacherFact(o, advice(o),projection(o,orders=partial).teacher_barrier.trace))
    assert label.supervision_weight == 0
    i=info(o.decision_ts_ns)
    i["policy_executed_action_pre_terminal"]=0.2
    label=native_teacher_step(TeacherFact(o, advice(o),projection(o,gym=i).teacher_barrier.trace))
    assert label.supervision_weight == 0


@pytest.mark.parametrize("field,value,pattern",[
    ("terminal_settled",True,"terminal"),
    ("decision_ts_ns",2_000_000_000,"clock"),
    ("risk_adjusted_target",None,"risk_adjusted"),
    ("quantity_rounded_target_exposure",None,"quantity_rounded"),
    ("r_econ",0.2,"conservation"),
    ("requested_action",-0.5,"policy action"),
    ("policy_equity_change","-8","not conserved"),
    ("execution_client_order_id","fake","SUBMIT"),
    ("policy_executed_action_pre_terminal",float("nan"),"nonfinite"),
    ("execution_plan_action","FILLED","planner action"),
])
def test_invalid_host_info_fails_closed(observed,field,value,pattern):
    o=observed(ts=2_000_000_000)
    raw=info(o.decision_ts_ns)
    raw[field]=value
    with pytest.raises(ValueError,match=pattern):
        projection(o,gym=raw)


def test_order_facts_cannot_be_minted_from_submit_alone(observed):
    o=observed(ts=2_000_000_000)
    with pytest.raises(ValueError,match="receipt"):
        L3OrderFacts(("RLES-order1",),("RLES-order1",))
    with pytest.raises(ValueError,match="provenance"):
        L3OrderFacts(("RLES-order1",),("other",),native_evidence_receipt="receipt")
    with pytest.raises(ValueError,match="matching L3"):
        projection(o,orders=L3OrderFacts(native_evidence_receipt="receipt"))


def test_cancel_native_provenance_and_noop_blocked(observed):
    o=observed(ts=2_000_000_000,pos=.5,pending=("old-order",))
    raw=info(o.decision_ts_ns,req=0.,action="CANCEL")
    raw["native_pending_order_ids_before"]=("old-order",)
    raw["policy_executed_action_pre_terminal"]=.5
    raw["execution_cancel_order_ids"]=("old-order",)
    with pytest.raises(ValueError,match="cancel IDs"):
        projection(o,gym=raw,orders=L3OrderFacts(native_evidence_receipt="l3"),req=0.)
    p=projection(o,gym=raw,orders=L3OrderFacts(related_existing_order_ids=("old-order",),
                                                    native_evidence_receipt="l3"),req=0.)
    sample=native_teacher_step(TeacherFact(o,advice(o,0.),p.teacher_barrier.trace))
    assert sample.supervision_weight==0 and sample.action_status=="CANCEL"


def test_episodes_cannot_be_interleaved_or_compounded(observed):
    a=observed(ts=2_000_000_000,episode="epA")
    b=observed(ts=3_000_000_000,episode="epB")
    c=observed(ts=4_000_000_000,episode="epA")
    pa,pb,pc=(projection(x).teacher_barrier for x in (a,b,c))
    labels, summary=audit_teacher_barriers((pa,pb))
    assert len(summary.episode_equity_segments)==2
    with pytest.raises(ValueError,match="interleaved"):
        audit_teacher_barriers((pa,pb,pc))


def test_flat_lifecycle_rejects_entry_fill_during_wait():
    def trace(t,action,actual,sub=(),fills=(),old=(),pending=()):
        return NativeActionTrace(t,0.,0.,0.,actual,100.,100.,action,sub,fills,old,pending)
    rows=[trace(1,"CANCEL",.5,old=("open",),pending=("open",)),
          trace(2,"WAIT",0.,fills=("open",),old=("open",))]
    assert not verify_native_flat_lifecycle(rows,initial_abs_exposure=.5)
    assert verify_native_flat_lifecycle(rows+[trace(3,"SUBMIT",0.,("close",),("close",))],
                                        initial_abs_exposure=.5)


def pair(window,scenario,*,seed=31,tret=.03,bret=.01,fills=4,liquidations=0,matched=True):
    return NativePair(window,scenario,seed,"teacher","flat","registered-source",
                      tret,bret,.05,.02,fills,liquidations,matched,True,True)


def test_paired_teacher_gates_are_conservative():
    records=[pair(w,s,seed=seed) for w in ("w1","w2")
             for s in ("base","stress") for seed in (31,47)]
    result=screen_teacher(records)
    assert result.status=="TRAIN_INNER_SCREEN_PASS"
    assert result.economic_status=="SCENARIO_ONLY"
    assert result.window_count==2
    assert "UNVERIFIED_NONOVERLAPPING_WINDOWS" in screen_teacher([
        replace(records[0],nonoverlap_verified=False)]+records[1:]).reasons
    assert "MISSING_REQUIRED_COST_SCENARIOS" in screen_teacher([
        r for r in records if r.scenario == "base"]).reasons
    assert "INSUFFICIENT_RANDOM_SEEDS" in screen_teacher([
        r for r in records if r.seed == 31]).reasons
    assert screen_teacher([replace(records[0],teacher_net_return=-.1)]+records[1:]).status=="TRAIN_INNER_SCREEN_REJECT"
    assert "NONPOSITIVE_WORST_PAIR_EXCESS" in screen_teacher([
        replace(records[0],teacher_net_return=-.1)]+records[1:]).reasons
    assert "NATIVE_LIQUIDATION" in screen_teacher([
        replace(records[0],teacher_liquidations=1)]+records[1:]).reasons
    assert "UNMATCHED_NATIVE_BARRIERS" in screen_teacher([
        replace(records[0],matched_native_barriers=False)]+records[1:]).reasons
    with pytest.raises(ValueError,match="incomplete"):
        screen_teacher(records[:-1])
    with pytest.raises(ValueError,match="duplicate"):
        screen_teacher(records+[records[-1]])
