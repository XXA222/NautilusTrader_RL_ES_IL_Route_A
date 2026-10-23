from dataclasses import replace

import pytest

from rles_route_a.evaluation.teacher_gate import (
    NativePair,
    StudentPair,
    screen_student,
    screen_teacher,
    verify_student_screen,
)


def _teacher_screen():
    pairs = tuple(
        NativePair(
            window_id=window,
            scenario=scenario,
            seed=seed,
            teacher_recipe="teacher-v1",
            baseline_recipe="flat-v1",
            source_identity="train-source-identity",
            teacher_net_return=0.02,
            baseline_net_return=0.0,
            teacher_max_drawdown=0.10,
            baseline_max_drawdown=0.05,
            teacher_filled_orders=2,
            teacher_liquidations=0,
            matched_native_barriers=True,
            scenario_only=True,
            nonoverlap_verified=True,
        )
        for window in ("window-1", "window-2")
        for scenario in ("base", "stress")
        for seed in (11, 29)
    )
    return screen_teacher(pairs)


def _student_pairs(*, student_return=0.03):
    teacher_gate = _teacher_screen()
    return tuple(
        StudentPair(
            window_id=window,
            scenario=scenario,
            seed=seed,
            teacher_recipe="teacher-v1",
            student_recipe="student-v1",
            source_identity="train-source-identity",
            teacher_net_return=0.02,
            student_net_return=student_return,
            teacher_max_drawdown=0.10,
            student_max_drawdown=0.12,
            student_filled_orders=2,
            student_liquidations=0,
            matched_native_barriers=True,
            teacher_gate_sha256=teacher_gate.receipt_sha256,
            actor_sha256="a" * 64,
            observation_schema_sha256="b" * 64,
            action_schema_sha256="c" * 64,
            execution_recipe_sha256="d" * 64,
            teacher_evidence_sha256="e" * 64,
            student_evidence_sha256="f" * 64,
            scenario_only=True,
            nonoverlap_verified=True,
        )
        for window in ("window-1", "window-2")
        for scenario in ("base", "stress")
        for seed in (11, 29)
    )


def test_student_gate_requires_positive_net_and_beats_qualified_teacher_every_pair():
    teacher = _teacher_screen()
    result = screen_student(_student_pairs(), teacher_screen=teacher)
    assert teacher.status == "TRAIN_INNER_SCREEN_PASS"
    assert result.status == "TRAIN_INNER_SCREEN_PASS"
    assert result.economic_status == "SCENARIO_ONLY"
    assert result.pair_count == 8
    assert result.window_count == 2
    assert result.worst_excess_return == pytest.approx(0.01)
    assert result.mean_excess_return == pytest.approx(0.01)
    assert result.actor_sha256 == "a" * 64
    assert result.teacher_gate_sha256 == teacher.receipt_sha256
    assert not result.reasons
    assert verify_student_screen(result)
    assert not verify_student_screen(replace(result, paired_records=result.paired_records[:-1]))


def test_student_gate_rejects_nonpositive_or_underperforming_pair():
    teacher = _teacher_screen()
    pairs = list(_student_pairs())
    pairs[0] = replace(pairs[0], student_net_return=-0.001)
    result = screen_student(tuple(pairs), teacher_screen=teacher)
    assert result.status == "TRAIN_INNER_SCREEN_REJECT"
    assert "STUDENT_NONPOSITIVE_NET_RETURN" in result.reasons
    assert "STUDENT_DOES_NOT_BEAT_TEACHER_IN_EVERY_PAIR" in result.reasons


def test_student_gate_rejects_mixed_frozen_actor_or_wrong_teacher_receipt():
    teacher = _teacher_screen()
    mixed = list(_student_pairs())
    mixed[-1] = replace(mixed[-1], actor_sha256="b" * 64)
    with pytest.raises(ValueError, match="mixes teacher, actor, source"):
        screen_student(tuple(mixed), teacher_screen=teacher)

    stale = list(_student_pairs())
    stale[-1] = replace(stale[-1], teacher_gate_sha256="b" * 64)
    with pytest.raises(ValueError, match="mixes teacher, actor, source"):
        screen_student(tuple(stale), teacher_screen=teacher)


def test_student_gate_rejects_teacher_metrics_outside_the_qualified_receipt():
    teacher = _teacher_screen()
    pairs = list(_student_pairs())
    pairs[0] = replace(pairs[0], teacher_net_return=0.021)
    with pytest.raises(ValueError, match="teacher metrics differ"):
        screen_student(tuple(pairs), teacher_screen=teacher)


def test_student_gate_rejects_incomplete_matrix_and_unqualified_teacher():
    teacher = _teacher_screen()
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        screen_student(_student_pairs()[:-1], teacher_screen=teacher)

    failed_teacher_pairs = tuple(
        replace(pair, teacher_net_return=-0.01) for pair in _teacher_screen_pairs()
    )
    failed_teacher = screen_teacher(failed_teacher_pairs)
    assert failed_teacher.status == "TRAIN_INNER_SCREEN_REJECT"
    with pytest.raises(ValueError, match="passed teacher gate"):
        screen_student(_student_pairs(), teacher_screen=failed_teacher)


def _teacher_screen_pairs():
    return tuple(
        NativePair(
            window_id=window,
            scenario=scenario,
            seed=seed,
            teacher_recipe="teacher-v1",
            baseline_recipe="flat-v1",
            source_identity="train-source-identity",
            teacher_net_return=-0.01,
            baseline_net_return=-0.10,
            teacher_max_drawdown=0.10,
            baseline_max_drawdown=0.20,
            teacher_filled_orders=2,
            teacher_liquidations=0,
            matched_native_barriers=True,
            scenario_only=True,
            nonoverlap_verified=True,
        )
        for window in ("window-1", "window-2")
        for scenario in ("base", "stress")
        for seed in (11, 29)
    )
