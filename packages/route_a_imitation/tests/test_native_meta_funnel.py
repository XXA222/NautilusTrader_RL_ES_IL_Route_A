import pytest

from rles_route_a.teacher_research.meta_gated_teacher import MetaGatedTeacher
from rles_route_a.teacher_research.native_meta_funnel import MetaNativeFunnelAdapter
from test_candidate_collector import observation
from test_meta_gated_teacher import Recovery, trained_policy


class NativeHook:
    def capture_pre_decision(self, obs, action, visible, teacher):
        return teacher.last_signal, teacher.recipe.indicator_recipe_id
    def capture_post_decision(self, *args):
        return args
    def build_native_cases(self, records):
        return records
    def commit(self, records, cases):
        return records, cases


def test_meta_funnel_projects_selected_signal_to_native_case_hook():
    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    hook = MetaNativeFunnelAdapter(teacher=teacher, native_hook=NativeHook())
    first = observation(10)
    action = teacher.advise(first)
    signal, recipe = hook.capture_pre_decision(first, action, {}, teacher)
    assert signal.raw_signal == "NONE" and recipe == "route_a.meta_gated_x1.v1"
    second = observation(11, ema_9=101.)
    action = teacher.advise(second)
    signal, _ = hook.capture_pre_decision(second, action, {}, teacher)
    assert signal.raw_signal == "LONG_ENTRY" and signal.reason_code == "EMA_CROSS_LONG"
    third = observation(12, ema_9=99., exposure=.3)
    action = teacher.advise(third)
    signal, _ = hook.capture_pre_decision(third, action, {}, teacher)
    assert signal.raw_signal == "EXIT_LONG" and signal.reason_code == "EMA_REVERSE_CROSS"
    with pytest.raises(ValueError, match="corresponding"):
        hook.capture_pre_decision(second, action, {}, MetaGatedTeacher(trained_policy(), recovery=Recovery()))
