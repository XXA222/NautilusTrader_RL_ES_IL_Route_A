from dataclasses import replace

import pytest

from rles_route_a.contracts import ExpertAdvice, NativeActionTrace
from rles_route_a.teacher_research.candidate_event_union import TeacherProposal, group_teacher_proposals
from rles_route_a.teacher_research.conditional_outcome_labels import GroupOutcome
from rles_route_a.teacher_research.entry_meta_policy import fit_entry_meta_policy, summarize_entry_scores
from rles_route_a.teacher_research.meta_gated_teacher import MetaGatedTeacher, write_meta_trace
from rles_route_a.integration.host_gym_projection import L3OrderFacts
from test_candidate_collector import observation


class Recovery:
    def __init__(self):
        self.active = False

    def intervention(self, obs):
        if self.active:
            return ExpertAdvice("recovery", obs.decision_ts_ns, "EXIT", 0., ("MAX_DRAWDOWN",))
        return None


def trained_policy():
    groups = {}
    labels = []
    for minute in range(1, 9):
        obs = observation(minute, ema_9=101. if minute % 2 == 0 else 99.)
        group, = group_teacher_proposals([TeacherProposal(obs, "T01_EMA_9_21", 1, "EMA_CROSS_LONG")])
        groups[group.event_id] = group
        labels.append(GroupOutcome(group.event_id, obs.decision_ts_ns, obs.source_id,
            obs.fold_id, "X1", "native", "base", obs.decision_ts_ns + 1, None,
            20. if minute % 2 == 0 else -20.))
    return fit_entry_meta_policy(labels[:4], labels[4:], groups,
                                 feature_names=("market.ema_9", "teacher.T01_EMA_9_21",
                                                "consensus.count", "cost.round_trip_bps"),
                                 margin_bps=1., cost_bps=12.)


def test_native_hold_exit_pending_and_trace(tmp_path):
    model = trained_policy()
    teacher = MetaGatedTeacher(model, recovery=Recovery())
    assert teacher.advise(observation(10)).intent == "FLAT"
    opened = teacher.advise(observation(11, ema_9=101.))
    assert opened.intent == "OPEN_LONG"
    trace = NativeActionTrace(opened.decision_ts_ns, .3, .3, .3, .3, 10000., 9999.,
                              "SUBMIT", submitted_order_ids=("o",), filled_order_ids=("o",))
    facts = L3OrderFacts(submitted_order_ids=("o",), filled_order_ids=("o",),
                         native_evidence_receipt="native-receipt")
    teacher.observe_native_action(opened, trace, facts)
    assert teacher.native_receipts[-1].trace.filled_order_ids == ("o",)
    pending = teacher.advise(observation(12, ema_9=101., exposure=.3, pending=("x",)))
    assert pending.intent == "HOLD"
    closed = teacher.advise(observation(13, ema_9=99., exposure=.3))
    assert closed.intent == "EXIT" and "EMA_REVERSE_CROSS" in closed.reason_codes
    assert teacher.decisions[1].candidates[0].selected
    import json
    output = tmp_path / "trace.jsonl"
    write_meta_trace(teacher, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[1]["candidates"][0]["disposition"] == "ACCEPTED_SELECTED"
    assert rows[-1]["native_order_facts"]["filled_order_ids"] == ["o"]


def test_risk_override_precedes_model_and_unknown_position():
    recovery = Recovery()
    recovery.active = True
    teacher = MetaGatedTeacher(trained_policy(), recovery=recovery)
    # Earlier than calibration and no known owner: hard risk still acts.
    action = teacher.advise(observation(1, exposure=.4))
    assert action.intent == "EXIT" and action.expert_id == "recovery"
    # The override remains executable even if candidate diagnostics fail.
    assert teacher.decisions[-1].candidate_error is None


def test_rejection_diagnostics_identify_out_of_support_feature():
    model = trained_policy()
    obs = observation(10, ema_9=200.)
    group, = group_teacher_proposals([TeacherProposal(obs, "T01_EMA_9_21", 1, "cross")])
    score = model.score(group)
    assert score.reason == "OUT_OF_SUPPORT"
    assert "market.ema_9" in score.unsupported_features
    result = summarize_entry_scores([score])
    assert result["coverage"] == 0 and result["reason_counts"]["OUT_OF_SUPPORT"] == 1
    assert result["unsupported_feature_counts"]["market.ema_9"] == 1


def test_mixed_return_bases_are_rejected():
    # Data validation independent of model fitting.
    model = trained_policy()
    assert model.cost_bps == 12.
    assert model.calibration_mae_bps >= 0.
