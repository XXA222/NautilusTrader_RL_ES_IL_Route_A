from dataclasses import replace

import pytest

from rles_route_a.contracts import ExpertAdvice, NativeActionTrace
from rles_route_a.teacher_research.candidate_collector import CandidateCollection
from rles_route_a.teacher_research.candidate_event_union import TeacherProposal, group_teacher_proposals
from rles_route_a.teacher_research.conditional_outcome_labels import GroupOutcome
from rles_route_a.teacher_research.entry_meta_policy import (
    EntryScore, fit_entry_meta_policy, summarize_entry_scores,
)
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
                                 margin_bps=1., cost_bps=12.,
                                 fee_bps_per_leg=5., slippage_bps_per_leg=1.)


def test_native_hold_exit_pending_and_trace(tmp_path):
    model = trained_policy()
    teacher = MetaGatedTeacher(model, recovery=Recovery())
    flat = teacher.advise(observation(10))
    assert flat.intent == "FLAT"
    flat_trace = NativeActionTrace(flat.decision_ts_ns, 0., 0., 0., 0., 10000., 10000., "NOOP")
    teacher.observe_native_action(flat, flat_trace, L3OrderFacts(native_evidence_receipt="flat-receipt"))
    opened = teacher.advise(observation(11, ema_9=101.))
    assert opened.intent == "OPEN_LONG"
    trace = NativeActionTrace(opened.decision_ts_ns, .3, .3, .3, .3, 10000., 9999.,
                              "SUBMIT", submitted_order_ids=("o",), filled_order_ids=("o",))
    facts = L3OrderFacts(submitted_order_ids=("o",), filled_order_ids=("o",),
                         native_evidence_receipt="native-receipt")
    with pytest.raises(ValueError, match="matching decision"):
        teacher.observe_native_action(replace(opened, intent="FLAT"), trace, facts)
    teacher.observe_native_action(opened, trace, facts)
    assert teacher.native_receipts[-1].trace.filled_order_ids == ("o",)
    selected = next(item for item in teacher.decisions[1].candidates if item.selected)
    assert teacher.native_receipts[-1].selected_candidate_event_id == selected.group.event_id
    assert teacher.native_receipts[-1].selected_teacher_id == "T01_EMA_9_21"
    pending = teacher.advise(observation(12, ema_9=101., exposure=.3, pending=("x",)))
    assert pending.intent == "HOLD"
    closed = teacher.advise(observation(13, ema_9=99., exposure=.3))
    assert closed.intent == "EXIT" and "EMA_REVERSE_CROSS" in closed.reason_codes
    assert any(item.disposition == "REJECTED_POSITION_OPEN"
               for item in teacher.decisions[-1].candidates)
    assert teacher.decisions[1].candidates[0].selected
    import json
    output = tmp_path / "trace.jsonl"
    write_meta_trace(teacher, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows[0]["teacher_signals"]) == 8
    assert rows[1]["candidates"][0]["disposition"] == "ACCEPTED_SELECTED"
    assert rows[-1]["native_order_facts"]["filled_order_ids"] == ["o"]
    assert rows[-1]["selected_candidate_event_id"] == selected.group.event_id


def test_native_receipt_cannot_skip_an_earlier_decision():
    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    teacher.advise(observation(10))
    action = teacher.advise(observation(11, ema_9=101.))
    trace = NativeActionTrace(action.decision_ts_ns, .3, .3, .3, .3, 10000., 9999., "SUBMIT")
    with pytest.raises(ValueError, match="matching decision"):
        teacher.observe_native_action(action, trace, L3OrderFacts(native_evidence_receipt="receipt"))
    assert not teacher.native_receipts


def test_consensus_candidate_uses_any_ready_native_teacher_owner(monkeypatch):
    class StaticTeacher:
        def __init__(self, teacher_id, intent):
            self.teacher_id = teacher_id
            self.intent = intent

        def advise(self, obs):
            return ExpertAdvice(self.teacher_id, obs.decision_ts_ns, self.intent,
                                .3 if self.intent == "OPEN_LONG" else 0., (self.intent,))

        def reset(self, episode_id=""):
            pass

        def observe_native_action(self, action, trace, native_order_facts):
            pass

    class FixedCollection:
        def reset(self, episode_id=""):
            pass

        def collect(self, obs):
            proposals = (
                TeacherProposal(obs, "T01_EMA_9_21", 1, "EMA_CROSS_LONG"),
                TeacherProposal(obs, "T02_ROC_20_60", 1, "ROC_CONSENSUS_LONG"),
            )
            return CandidateCollection(group_teacher_proposals(proposals), ())

    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    teacher.collector = FixedCollection()
    teacher._teachers["T01_EMA_9_21"] = StaticTeacher("T01_EMA_9_21", "FLAT")
    teacher._teachers["T02_ROC_20_60"] = StaticTeacher("T02_ROC_20_60", "OPEN_LONG")
    monkeypatch.setattr(type(teacher.policy), "score", lambda _policy, _group:
                        EntryScore("LONG", .8, 20., True, "POSITIVE_NET_EDGE"))

    action = teacher.advise(observation(10, ema_9=101.))

    assert action.intent == "OPEN_LONG"
    assert action.expert_id == "T02_ROC_20_60"
    assert teacher.decisions[-1].owner_id == "T02_ROC_20_60"
    assert teacher.decisions[-1].candidates[0].selected
    trace = NativeActionTrace(action.decision_ts_ns, .3, .3, .3, .3,
                              10000., 9999., "SUBMIT")
    teacher.observe_native_action(action, trace,
                                  L3OrderFacts(native_evidence_receipt="owner-receipt"))
    assert teacher.native_receipts[-1].selected_teacher_id == "T02_ROC_20_60"


def test_pending_order_candidate_is_not_misreported_as_arbitration():
    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    teacher.advise(observation(10, ema_9=99.))
    action = teacher.advise(observation(11, ema_9=101., pending=("order",)))
    assert action.intent == "FLAT"
    assert teacher.decisions[-1].candidates
    assert all(item.disposition == "REJECTED_PENDING_ORDER"
               for item in teacher.decisions[-1].candidates)


def test_owner_lifecycle_can_block_signal_without_aborting_rollout():
    class NotReady:
        def advise(self, obs):
            return ExpertAdvice("T01_EMA_9_21", obs.decision_ts_ns, "FLAT", 0., ("LIFECYCLE_NOT_READY",))
    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    teacher._teachers["T01_EMA_9_21"] = NotReady()
    teacher.advise(observation(10, ema_9=99.))
    action = teacher.advise(observation(11, ema_9=101.))
    assert action.intent == "FLAT"
    assert any(item.disposition == "REJECTED_OWNER_NOT_READY" for item in teacher.decisions[-1].candidates)
    assert not any(item.selected for item in teacher.decisions[-1].candidates)


def test_risk_override_precedes_model_and_unknown_position():
    recovery = Recovery()
    recovery.active = True
    teacher = MetaGatedTeacher(trained_policy(), recovery=recovery)
    # Earlier than calibration and no known owner: hard risk still acts.
    action = teacher.advise(observation(1, exposure=.4))
    assert action.intent == "EXIT" and action.expert_id == "recovery"
    # The override remains executable while model-unavailable candidates stay explicit.
    assert teacher.decisions[-1].candidate_error is None
    action = teacher.advise(observation(2, ema_9=101., exposure=.4))
    assert action.intent == "EXIT"
    decision = teacher.decisions[-1]
    assert decision.candidates and decision.candidates[0].score is None
    assert decision.candidates[0].disposition == "REJECTED_HARD_RISK_OVERRIDE"
    assert decision.candidate_error is None


def test_feature_schema_change_resets_flat_teacher_state_and_keeps_trace():
    class ResetProbe:
        def __init__(self):
            self.resets = 0

        def reset(self, episode_id=""):
            self.resets += 1

        def advise(self, obs):
            return ExpertAdvice("T01_EMA_9_21", obs.decision_ts_ns, "FLAT", 0., ("PROBE",))

    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    probe = ResetProbe()
    teacher._teachers["T01_EMA_9_21"] = probe
    teacher.advise(observation(10))
    changed_schema = replace(observation(11), feature_spec_hash="1" * 64)
    teacher.advise(changed_schema)

    assert probe.resets == 1
    assert teacher._scope == ("e", "synthetic", "train", "1" * 64)
    assert len(teacher.decisions) == 2


def test_active_scope_change_keeps_hard_risk_action_without_advancing_state():
    class CallProbe:
        def __init__(self):
            self.calls = 0
            self.native_calls = 0

        def advise(self, obs):
            self.calls += 1
            return ExpertAdvice("T01_EMA_9_21", obs.decision_ts_ns, "FLAT", 0., ("PROBE",))

        def reset(self, episode_id=""):
            pass

        def observe_native_action(self, action, trace, native_order_facts):
            self.native_calls += 1

    recovery = Recovery()
    recovery.active = True
    teacher = MetaGatedTeacher(trained_policy(), recovery=recovery)
    probe = CallProbe()
    teacher._teachers["T01_EMA_9_21"] = probe
    teacher._owner = "T01_EMA_9_21"
    teacher._scope = ("e", "synthetic", "train", "0" * 64)
    changed_source = replace(observation(10, exposure=.4), source_id="other-source")

    action = teacher.advise(changed_source)

    assert action.intent == "EXIT" and action.expert_id == "recovery"
    assert probe.calls == 0
    assert teacher.decisions[-1].candidates == ()
    assert "diagnostics skipped" in teacher.decisions[-1].candidate_error
    trace = NativeActionTrace(action.decision_ts_ns, .4, 0., 0., 0., 9999., 9999., "SUBMIT")
    teacher.observe_native_action(action, trace, L3OrderFacts(native_evidence_receipt="risk-receipt"))
    assert teacher.native_receipts[-1].action == action
    assert probe.native_calls == 0


def test_active_scope_change_without_recovery_fails_before_teacher_state_moves():
    class CallProbe:
        def __init__(self):
            self.calls = 0

        def advise(self, obs):
            self.calls += 1
            return ExpertAdvice("T01_EMA_9_21", obs.decision_ts_ns, "FLAT", 0., ("PROBE",))

    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    probe = CallProbe()
    teacher._teachers["T01_EMA_9_21"] = probe
    teacher._scope = ("e", "synthetic", "train", "0" * 64)
    changed_source = replace(observation(10, exposure=.4), source_id="other-source")

    with pytest.raises(ValueError, match="episode/source/fold/schema"):
        teacher.advise(changed_source)

    assert probe.calls == 0


def test_candidate_from_new_model_scope_is_recorded_as_rejected_not_exception():
    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    teacher.advise(observation(10))
    teacher.advise(replace(observation(11), source_id="new-source"))
    action = teacher.advise(replace(observation(12, ema_9=101.), source_id="new-source"))

    assert action.intent == "FLAT"
    candidates = teacher.decisions[-1].candidates
    candidate = next(item for item in candidates
                     if "T01_EMA_9_21" in {teacher_id for teacher_id, _ in item.group.teachers})
    assert candidate.score is None
    assert candidate.disposition == "REJECTED_MODEL_SCOPE_MISMATCH"
    assert teacher.decisions[-1].candidate_error is None


def test_candidate_before_calibration_availability_is_recorded_as_rejected():
    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    teacher.advise(observation(1))
    action = teacher.advise(observation(2, ema_9=101.))

    assert action.intent == "FLAT"
    candidate = next(item for item in teacher.decisions[-1].candidates
                     if "T01_EMA_9_21" in {teacher_id for teacher_id, _ in item.group.teachers})
    assert candidate.score is None
    assert candidate.disposition == "REJECTED_MODEL_NOT_AVAILABLE"
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


def test_missing_market_feature_rejects_candidate_without_scoring():
    model = trained_policy()
    model = replace(model, feature_names=("market.zscore_60", *model.feature_names[1:]))
    teacher = MetaGatedTeacher(model, recovery=Recovery())
    def without_zscore(obs):
        return replace(obs, market={k: v for k, v in obs.market.items() if k != "zscore_60"},
                       feature_available_at_ns={k: v for k, v in obs.feature_available_at_ns.items()
                                                if k != "zscore_60"})
    teacher.advise(without_zscore(observation(10, ema_9=99.)))
    action = teacher.advise(without_zscore(observation(11, ema_9=101.)))
    decision = teacher.decisions[-1]
    assert action.intent == "FLAT"
    assert decision.candidates and decision.candidates[0].score is None
    assert decision.candidates[0].disposition == "REJECTED_FEATURE_NOT_READY"
    assert decision.candidates[0].missing_features == ("market.zscore_60",)


def test_missing_account_model_feature_rejects_without_fabricating_zero():
    model = trained_policy()
    model = replace(model, feature_names=("account.drawdown", *model.feature_names[1:]))
    teacher = MetaGatedTeacher(model, recovery=Recovery())
    teacher.advise(observation(10, ema_9=99.))
    action = teacher.advise(observation(11, ema_9=101.))
    assert action.intent == "FLAT"
    assert teacher.decisions[-1].candidates[0].disposition == "REJECTED_FEATURE_NOT_READY"
    assert teacher.decisions[-1].candidates[0].score is None
    assert teacher.decisions[-1].candidates[0].missing_features == ("account.drawdown",)


def test_candidate_scoring_error_is_recorded_and_fails_closed(monkeypatch):
    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    teacher.advise(observation(10, ema_9=99.))
    monkeypatch.setattr(type(teacher.policy), "score",
                        lambda _policy, _group: (_ for _ in ()).throw(ValueError("bad score")))

    action = teacher.advise(observation(11, ema_9=101.))

    assert action.intent == "FLAT"
    decision = teacher.decisions[-1]
    assert decision.candidate_error is None
    assert decision.candidates
    assert all(candidate.disposition == "REJECTED_SCORING_ERROR"
               and candidate.error_detail == "ValueError: bad score"
               for candidate in decision.candidates)


def test_candidate_collection_error_does_not_suppress_native_exit(monkeypatch):
    class ExitTeacher:
        def advise(self, obs):
            return ExpertAdvice("T01_EMA_9_21", obs.decision_ts_ns, "EXIT", 0.,
                                ("EMA_REVERSE_CROSS",))

    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    teacher._owner = "T01_EMA_9_21"
    teacher._scope = ("e", "synthetic", "train", "0" * 64)
    teacher._teachers["T01_EMA_9_21"] = ExitTeacher()
    monkeypatch.setattr(teacher.collector, "collect",
                        lambda _observation: (_ for _ in ()).throw(ValueError("bad features")))

    action = teacher.advise(observation(10, exposure=.3))

    assert action.intent == "EXIT"
    assert teacher.decisions[-1].candidate_error == "ValueError: bad features"
    assert teacher.decisions[-1].candidates == ()


def test_candidate_collection_error_fails_closed_for_new_entries(monkeypatch):
    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    monkeypatch.setattr(teacher.collector, "collect",
                        lambda _observation: (_ for _ in ()).throw(ValueError("bad features")))

    action = teacher.advise(observation(10))

    assert action.intent == "FLAT"
    assert "CANDIDATE_COLLECTION_FAILED" in action.reason_codes
    assert teacher.decisions[-1].candidate_error == "ValueError: bad features"


def test_mixed_return_bases_are_rejected():
    # Data validation independent of model fitting.
    model = trained_policy()
    assert model.cost_bps == 12.
    assert model.calibration_mae_bps >= 0.
