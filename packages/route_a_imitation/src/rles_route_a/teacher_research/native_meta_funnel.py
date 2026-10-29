"""Adapt Meta signals to the existing native teacher-case research hook."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from rles_route_a.evaluation.teacher_case_report_1m import NativeTradeCase1m
from rles_route_a.teachers_1m.contracts import TeacherSignal
from .meta_gated_teacher import MetaGatedTeacher


class MetaNativeFunnelAdapter:
    """Delegate native fills, fees, funding and case construction to the host hook."""

    def __init__(self, *, teacher: MetaGatedTeacher, native_hook):
        required = ("capture_pre_decision", "capture_post_decision", "build_native_cases", "commit")
        if not isinstance(teacher, MetaGatedTeacher) or any(
                not callable(getattr(native_hook, name, None)) for name in required):
            raise TypeError("Meta teacher and complete native case hook are required")
        self.teacher = teacher
        self.native_hook = native_hook

    def capture_pre_decision(self, observation, action, visible_context, teacher):
        if teacher is not self.teacher or not teacher.decisions:
            raise ValueError("native funnel has no corresponding Meta decision")
        selected = [x for x in teacher.decisions[-1].candidates if x.selected]
        reason = action.reason_codes[-1]
        if action.intent == "OPEN_LONG":
            raw, side = "LONG_ENTRY", 1
        elif action.intent == "OPEN_SHORT":
            raw, side = "SHORT_ENTRY", -1
        elif action.intent == "EXIT":
            raw = "EXIT_LONG" if observation.account_value("net_exposure") > 0 else "EXIT_SHORT"
            side = 0
        else:
            raw, side = "NONE", 0
        if raw in ("LONG_ENTRY", "SHORT_ENTRY"):
            if len(selected) != 1 or selected[0].group.side != side:
                raise ValueError("Meta native entry must bind its single selected candidate")
            reason = selected[0].group.teachers[0][1]
        facade = SimpleNamespace(last_signal=TeacherSignal(raw, side, reason),
                                 required_fields=(),
                                 recipe=SimpleNamespace(indicator_recipe_id="route_a.meta_gated_x1.v1"))
        return self.native_hook.capture_pre_decision(observation, action, visible_context, facade)

    def capture_post_decision(self, pre, action, gym_info, barrier, order_facts):
        return self.native_hook.capture_post_decision(pre, action, gym_info, barrier, order_facts)

    def build_native_cases(self, records):
        cases = self.native_hook.build_native_cases(records)
        if not isinstance(cases, tuple) or any(not isinstance(case, NativeTradeCase1m) for case in cases):
            raise TypeError("Meta native case hook must return native trade cases")
        selected_by_decision = {}
        for decision in self.teacher.decisions:
            selected = tuple(item.group for item in decision.candidates if item.selected)
            if len(selected) > 1:
                raise ValueError("Meta native decision selected multiple entry candidates")
            if selected:
                if decision.observation.decision_ts_ns in selected_by_decision:
                    raise ValueError("Meta candidate decisions have duplicate clocks")
                selected_by_decision[decision.observation.decision_ts_ns] = selected[0]
        enriched = []
        for case in cases:
            group = selected_by_decision.get(case.entry_decision_ts_ns)
            if group is None or group.side != case.side:
                raise ValueError("native Meta trade case has no matching selected candidate")
            teacher_ids = tuple(name for name, _reason in group.teachers)
            if (case.candidate_event_id is not None
                    and (case.candidate_event_id != group.event_id
                         or case.candidate_teacher_ids != teacher_ids)):
                raise ValueError("native case candidate provenance conflicts with Meta selection")
            enriched.append(replace(case, candidate_event_id=group.event_id,
                                    candidate_teacher_ids=teacher_ids))
        return tuple(enriched)

    def commit(self, records, cases):
        return self.native_hook.commit(records, cases)
