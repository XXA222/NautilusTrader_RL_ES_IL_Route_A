"""Paired native TRAIN-INNER teacher screening; not an independent alpha claim.

The host L6 supplies actual fee-inclusive account returns and evaluation
identity. This package neither selects already-consumed OOS windows nor opens
market files. We only compare matching scenarios/seed/windows and report clear
pass/fail reasons. Scenario-only economics can never be promoted to certified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Sequence, cast


@dataclass(frozen=True, slots=True)
class NativePair:
    """One pre-registered replay pair, native net returns in same units."""

    window_id: str
    scenario: str
    seed: int
    teacher_recipe: str
    baseline_recipe: str
    source_identity: str
    teacher_net_return: float
    baseline_net_return: float
    teacher_max_drawdown: float
    baseline_max_drawdown: float
    teacher_filled_orders: int
    teacher_liquidations: int
    matched_native_barriers: bool
    scenario_only: bool = True
    # Set only after L6 verifies the actual window intervals, not their names.
    nonoverlap_verified: bool = False

    def __post_init__(self) -> None:
        if not all(
            isinstance(x, str) and x.strip()
            for x in (
                self.window_id,
                self.scenario,
                self.teacher_recipe,
                self.baseline_recipe,
                self.source_identity,
            )
        ):
            raise ValueError("native pair identity missing")
        if self.teacher_recipe == self.baseline_recipe:
            raise ValueError("teacher must be compared with a distinct baseline")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("invalid preregistered seed")
        if (
            any(
                not isinstance(x, (int, float)) or isinstance(x, bool) or not isfinite(x)
                for x in (
                    self.teacher_net_return,
                    self.baseline_net_return,
                    self.teacher_max_drawdown,
                    self.baseline_max_drawdown,
                )
            )
            or not 0 <= self.teacher_max_drawdown <= 1
            or not 0 <= self.baseline_max_drawdown <= 1
        ):
            raise ValueError("invalid native return/drawdown")
        if any(
            type(x) is not int or x < 0
            for x in (self.teacher_filled_orders, self.teacher_liquidations)
        ):
            raise ValueError("invalid native order/risk counts")
        if (
            type(self.matched_native_barriers) is not bool
            or type(self.scenario_only) is not bool
            or type(self.nonoverlap_verified) is not bool
        ):
            raise ValueError("native proof/scenario flags required")


@dataclass(frozen=True, slots=True)
class TeacherScreen:
    status: str
    economic_status: str
    pair_count: int
    window_count: int
    worst_excess_return: float
    mean_excess_return: float
    reasons: tuple[str, ...]
    teacher_recipe: str = ""
    baseline_recipe: str = ""
    source_identity: str = ""
    window_ids: tuple[str, ...] = ()
    scenarios: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    receipt_sha256: str = ""
    paired_records: tuple[NativePair, ...] = ()
    thresholds: tuple[tuple[str, object], ...] = ()


def screen_teacher(
    pairs: Sequence[NativePair],
    *,
    min_windows: int = 2,
    min_seeds: int = 2,
    min_fill_orders: int = 1,
    max_drawdown: float = 0.30,
    require_all_scenarios_positive: bool = True,
    require_positive_teacher_net: bool = True,
    expected_scenarios: tuple[str, ...] = ("base", "stress"),
) -> TeacherScreen:
    """Conservative diagnostics only; NEVER consumes or certifies Validation/Test.

    All pairs must share one teacher/baseline/source and form a complete
    window×scenario×seed matrix. Every window requires independent *host*
    evidence for nonoverlap. IDs alone do not prove temporal independence.
    """
    if not pairs:
        raise ValueError("empty native pair matrix")
    if (
        type(min_windows) is not int
        or min_windows < 2
        or type(min_seeds) is not int
        or min_seeds < 2
        or type(min_fill_orders) is not int
        or min_fill_orders < 0
    ):
        raise ValueError("invalid teacher sample thresholds")
    if (
        not isinstance(max_drawdown, (int, float))
        or not isfinite(max_drawdown)
        or not 0 <= max_drawdown <= 1
    ):
        raise ValueError("invalid teacher drawdown gate")
    teacher_id = {(p.teacher_recipe, p.baseline_recipe, p.source_identity) for p in pairs}
    if len(teacher_id) != 1:
        raise ValueError("native screen mixes incompatible recipes or market snapshots")
    if not expected_scenarios or len(set(expected_scenarios)) != len(expected_scenarios):
        raise ValueError("nonempty unique required scenarios must be frozen")
    windows = {p.window_id for p in pairs}
    scenarios = {p.scenario for p in pairs}
    seeds = {p.seed for p in pairs}
    indexed = {(p.window_id, p.scenario, p.seed) for p in pairs}
    if len(indexed) != len(pairs) or len(indexed) != len(windows) * len(scenarios) * len(seeds):
        raise ValueError("incomplete or duplicate paired scenario/seed/window matrix")
    reasons = []
    if len(windows) < min_windows:
        reasons.append("INSUFFICIENT_TRAIN_INNER_WINDOWS")
    if len(seeds) < min_seeds:
        reasons.append("INSUFFICIENT_RANDOM_SEEDS")
    if scenarios != set(expected_scenarios):
        reasons.append("MISSING_REQUIRED_COST_SCENARIOS")
    if any(not p.nonoverlap_verified for p in pairs):
        reasons.append("UNVERIFIED_NONOVERLAPPING_WINDOWS")
    if any(not p.matched_native_barriers for p in pairs):
        reasons.append("UNMATCHED_NATIVE_BARRIERS")
    if any(p.teacher_liquidations > 0 for p in pairs):
        reasons.append("NATIVE_LIQUIDATION")
    if any(p.teacher_max_drawdown > max_drawdown for p in pairs):
        reasons.append("MAX_DRAWDOWN_EXCEEDED")
    if sum(p.teacher_filled_orders for p in pairs) < min_fill_orders:
        reasons.append("TEACHER_HAS_NO_NATIVE_FILLS")
    excess = [p.teacher_net_return - p.baseline_net_return for p in pairs]
    if require_positive_teacher_net and min(p.teacher_net_return for p in pairs) <= 0:
        reasons.append("TEACHER_NONPOSITIVE_NET_RETURN")
    if require_all_scenarios_positive and min(excess) <= 0:
        reasons.append("NONPOSITIVE_WORST_PAIR_EXCESS")
    elif sum(excess) / len(excess) <= 0:
        reasons.append("NONPOSITIVE_MEAN_PAIR_EXCESS")
    screen_status: str = "TRAIN_INNER_SCREEN_PASS" if not reasons else "TRAIN_INNER_SCREEN_REJECT"
    receipt_payload = {
        "schema": "rles.route_a.teacher_screen.v1",
        "pairs": [
            asdict(pair)
            for pair in sorted(
                pairs,
                key=lambda row: (row.window_id, row.scenario, row.seed),
            )
        ],
        "thresholds": {
            "min_windows": min_windows,
            "min_seeds": min_seeds,
            "min_fill_orders": min_fill_orders,
            "max_drawdown": max_drawdown,
            "require_all_scenarios_positive": require_all_scenarios_positive,
            "require_positive_teacher_net": require_positive_teacher_net,
            "expected_scenarios": list(expected_scenarios),
        },
        "status": screen_status,
        "reasons": reasons,
    }
    receipt_sha256 = sha256(
        json.dumps(
            receipt_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return TeacherScreen(
        status=screen_status,
        economic_status="SCENARIO_ONLY"
        if any(p.scenario_only for p in pairs)
        else "HOST_NATIVE_ECONOMICS_UNVERIFIED",
        pair_count=len(pairs),
        window_count=len(windows),
        worst_excess_return=min(excess),
        mean_excess_return=sum(excess) / len(excess),
        reasons=tuple(reasons),
        teacher_recipe=next(iter(teacher_id))[0],
        baseline_recipe=next(iter(teacher_id))[1],
        source_identity=next(iter(teacher_id))[2],
        window_ids=tuple(sorted(windows)),
        scenarios=tuple(sorted(scenarios)),
        seeds=tuple(sorted(seeds)),
        receipt_sha256=receipt_sha256,
        paired_records=tuple(
            sorted(
                pairs,
                key=lambda row: (row.window_id, row.scenario, row.seed),
            )
        ),
        thresholds=(
            ("min_windows", min_windows),
            ("min_seeds", min_seeds),
            ("min_fill_orders", min_fill_orders),
            ("max_drawdown", max_drawdown),
            ("require_all_scenarios_positive", require_all_scenarios_positive),
            ("require_positive_teacher_net", require_positive_teacher_net),
            ("expected_scenarios", expected_scenarios),
        ),
    )


@dataclass(frozen=True, slots=True)
class StudentPair:
    """One same-window native comparison of a qualified teacher and frozen student."""

    window_id: str
    scenario: str
    seed: int
    teacher_recipe: str
    student_recipe: str
    source_identity: str
    teacher_net_return: float
    student_net_return: float
    teacher_max_drawdown: float
    student_max_drawdown: float
    student_filled_orders: int
    student_liquidations: int
    matched_native_barriers: bool
    teacher_gate_sha256: str
    actor_sha256: str
    observation_schema_sha256: str
    action_schema_sha256: str
    execution_recipe_sha256: str
    teacher_evidence_sha256: str
    student_evidence_sha256: str
    scenario_only: bool = True
    nonoverlap_verified: bool = False

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.window_id,
                self.scenario,
                self.teacher_recipe,
                self.student_recipe,
                self.source_identity,
            )
        ):
            raise ValueError("student pair identity missing")
        if self.teacher_recipe == self.student_recipe:
            raise ValueError("student must be compared with a distinct teacher")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("invalid preregistered student seed")
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value)
            for value in (
                self.teacher_net_return,
                self.student_net_return,
                self.teacher_max_drawdown,
                self.student_max_drawdown,
            )
        ):
            raise ValueError("invalid student pair return/drawdown")
        if any(
            not 0 <= value <= 1
            for value in (
                self.teacher_max_drawdown,
                self.student_max_drawdown,
            )
        ):
            raise ValueError("student pair drawdown must be in [0, 1]")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.student_filled_orders,
                self.student_liquidations,
            )
        ):
            raise ValueError("invalid student order/risk counts")
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for value in (
                self.teacher_gate_sha256,
                self.actor_sha256,
                self.observation_schema_sha256,
                self.action_schema_sha256,
                self.execution_recipe_sha256,
                self.teacher_evidence_sha256,
                self.student_evidence_sha256,
            )
        ):
            raise ValueError(
                "teacher, actor, domain, execution, and replay SHA-256 identities are required"
            )
        if (
            type(self.matched_native_barriers) is not bool
            or type(self.scenario_only) is not bool
            or type(self.nonoverlap_verified) is not bool
        ):
            raise ValueError("native proof/scenario flags required")


@dataclass(frozen=True, slots=True)
class StudentScreen:
    """Train-inner paired result; a pass is never a certified economic claim."""

    status: str
    economic_status: str
    pair_count: int
    window_count: int
    worst_excess_return: float
    mean_excess_return: float
    teacher_recipe: str
    student_recipe: str
    source_identity: str
    actor_sha256: str
    observation_schema_sha256: str
    action_schema_sha256: str
    execution_recipe_sha256: str
    teacher_gate_sha256: str
    receipt_sha256: str
    reasons: tuple[str, ...]
    paired_records: tuple[StudentPair, ...] = ()
    teacher_records: tuple[NativePair, ...] = ()
    thresholds: tuple[tuple[str, object], ...] = ()
    teacher_thresholds: tuple[tuple[str, object], ...] = ()


def screen_student(
    pairs: Sequence[StudentPair],
    *,
    teacher_screen: TeacherScreen,
    min_windows: int = 2,
    min_seeds: int = 2,
    min_filled_orders: int = 1,
    max_drawdown: float = 0.30,
    expected_scenarios: tuple[str, ...] = ("base", "stress"),
) -> StudentScreen:
    """Require a frozen student to profit and beat its qualified teacher in every pair.

    This is a conservative Train-inner screening rule. It deliberately keeps
    scenario-only status when the underlying rules, fees, or margin facts are
    assumptions; it cannot qualify a teacher or certify historical economics.
    """
    if not pairs:
        raise ValueError("empty student pair matrix")
    if not isinstance(teacher_screen, TeacherScreen):
        raise TypeError("a verified teacher screen receipt is required")
    if (
        type(min_windows) is not int
        or min_windows < 2
        or type(min_seeds) is not int
        or min_seeds < 2
        or type(min_filled_orders) is not int
        or min_filled_orders < 0
    ):
        raise ValueError("invalid student sample thresholds")
    if (
        not isinstance(max_drawdown, (int, float))
        or isinstance(max_drawdown, bool)
        or not isfinite(max_drawdown)
        or not 0 <= max_drawdown <= 1
    ):
        raise ValueError("invalid student drawdown gate")
    if not expected_scenarios or len(set(expected_scenarios)) != len(expected_scenarios):
        raise ValueError("nonempty unique student cost scenarios required")
    identities = {
        (
            p.teacher_recipe,
            p.student_recipe,
            p.source_identity,
            p.teacher_gate_sha256,
            p.actor_sha256,
            p.observation_schema_sha256,
            p.action_schema_sha256,
            p.execution_recipe_sha256,
        )
        for p in pairs
    }
    if len(identities) != 1:
        raise ValueError("student comparison mixes teacher, actor, source, or gate identities")
    (
        teacher_recipe,
        _student_recipe,
        source_identity,
        gate_sha,
        actor_sha,
        _observation_schema_sha,
        _action_schema_sha,
        _execution_recipe_sha,
    ) = next(iter(identities))
    if (
        teacher_screen.status != "TRAIN_INNER_SCREEN_PASS"
        or teacher_screen.teacher_recipe != teacher_recipe
        or teacher_screen.source_identity != source_identity
        or teacher_screen.receipt_sha256 != gate_sha
        or not teacher_screen.paired_records
    ):
        raise ValueError("student comparison is not bound to its passed teacher gate")
    windows = {pair.window_id for pair in pairs}
    scenarios = {pair.scenario for pair in pairs}
    seeds = {pair.seed for pair in pairs}
    indexed = {(pair.window_id, pair.scenario, pair.seed) for pair in pairs}
    if len(indexed) != len(pairs) or len(indexed) != len(windows) * len(scenarios) * len(seeds):
        raise ValueError("incomplete or duplicate paired student scenario matrix")
    teacher_indexed = {
        (window, scenario, seed)
        for window in teacher_screen.window_ids
        for scenario in teacher_screen.scenarios
        for seed in teacher_screen.seeds
    }
    qualified_pairs = {
        (pair.window_id, pair.scenario, pair.seed): pair for pair in teacher_screen.paired_records
    }
    for pair in pairs:
        qualified = qualified_pairs.get((pair.window_id, pair.scenario, pair.seed))
        if (
            qualified is None
            or pair.teacher_net_return != qualified.teacher_net_return
            or pair.teacher_max_drawdown != qualified.teacher_max_drawdown
        ):
            raise ValueError(
                "student replay teacher metrics differ from its qualified gate receipt"
            )
    reasons: list[str] = []
    if teacher_screen.status != "TRAIN_INNER_SCREEN_PASS":
        reasons.append("TEACHER_GATE_NOT_PASSED")
    if (
        windows != set(teacher_screen.window_ids)
        or scenarios != set(teacher_screen.scenarios)
        or seeds != set(teacher_screen.seeds)
        or indexed != teacher_indexed
    ):
        reasons.append("STUDENT_MATRIX_DIFFERS_FROM_QUALIFIED_TEACHER_MATRIX")
    if len(windows) < min_windows:
        reasons.append("INSUFFICIENT_TRAIN_INNER_WINDOWS")
    if len(seeds) < min_seeds:
        reasons.append("INSUFFICIENT_RANDOM_SEEDS")
    if scenarios != set(expected_scenarios):
        reasons.append("MISSING_REQUIRED_COST_SCENARIOS")
    if any(not pair.nonoverlap_verified for pair in pairs):
        reasons.append("UNVERIFIED_NONOVERLAPPING_WINDOWS")
    if any(not pair.matched_native_barriers for pair in pairs):
        reasons.append("UNMATCHED_NATIVE_BARRIERS")
    if any(pair.teacher_net_return <= 0 for pair in pairs):
        reasons.append("TEACHER_NONPOSITIVE_NET_RETURN")
    if any(pair.student_net_return <= 0 for pair in pairs):
        reasons.append("STUDENT_NONPOSITIVE_NET_RETURN")
    excess = [pair.student_net_return - pair.teacher_net_return for pair in pairs]
    if min(excess) <= 0:
        reasons.append("STUDENT_DOES_NOT_BEAT_TEACHER_IN_EVERY_PAIR")
    if any(pair.student_liquidations > 0 for pair in pairs):
        reasons.append("STUDENT_NATIVE_LIQUIDATION")
    if any(
        pair.teacher_max_drawdown > max_drawdown or pair.student_max_drawdown > max_drawdown
        for pair in pairs
    ):
        reasons.append("MAX_DRAWDOWN_EXCEEDED")
    if sum(pair.student_filled_orders for pair in pairs) < min_filled_orders:
        reasons.append("STUDENT_HAS_NO_NATIVE_FILLS")
    student_status = "TRAIN_INNER_SCREEN_PASS" if not reasons else "TRAIN_INNER_SCREEN_REJECT"
    student_receipt_payload = {
        "schema": "rles.route_a.student_screen.v1",
        "pairs": [
            asdict(pair)
            for pair in sorted(
                pairs,
                key=lambda row: (row.window_id, row.scenario, row.seed),
            )
        ],
        "teacher_gate_sha256": gate_sha,
        "status": student_status,
        "reasons": reasons,
        "thresholds": {
            "min_windows": min_windows,
            "min_seeds": min_seeds,
            "min_filled_orders": min_filled_orders,
            "max_drawdown": max_drawdown,
            "expected_scenarios": list(expected_scenarios),
        },
    }
    student_receipt_sha256 = sha256(
        json.dumps(
            student_receipt_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return StudentScreen(
        status=student_status,
        economic_status="SCENARIO_ONLY"
        if any(pair.scenario_only for pair in pairs)
        or teacher_screen.economic_status == "SCENARIO_ONLY"
        else "HOST_NATIVE_ECONOMICS_UNVERIFIED",
        pair_count=len(pairs),
        window_count=len(windows),
        worst_excess_return=min(excess),
        mean_excess_return=sum(excess) / len(excess),
        teacher_recipe=teacher_recipe,
        student_recipe=next(iter(identities))[1],
        source_identity=source_identity,
        actor_sha256=actor_sha,
        observation_schema_sha256=next(iter(identities))[5],
        action_schema_sha256=next(iter(identities))[6],
        execution_recipe_sha256=next(iter(identities))[7],
        teacher_gate_sha256=gate_sha,
        receipt_sha256=student_receipt_sha256,
        reasons=tuple(reasons),
        paired_records=tuple(
            sorted(
                pairs,
                key=lambda row: (row.window_id, row.scenario, row.seed),
            )
        ),
        teacher_records=teacher_screen.paired_records,
        thresholds=(
            ("min_windows", min_windows),
            ("min_seeds", min_seeds),
            ("min_filled_orders", min_filled_orders),
            ("max_drawdown", max_drawdown),
            ("expected_scenarios", expected_scenarios),
        ),
        teacher_thresholds=teacher_screen.thresholds,
    )


def verify_student_screen(screen: StudentScreen) -> bool:
    """Recompute both gates from the serialized pair matrices and frozen thresholds."""
    if (
        not isinstance(screen, StudentScreen)
        or not screen.paired_records
        or not screen.teacher_records
    ):
        return False
    try:
        teacher_thresholds = dict(screen.teacher_thresholds)
        teacher = screen_teacher(
            screen.teacher_records,
            min_windows=cast(int, teacher_thresholds["min_windows"]),
            min_seeds=cast(int, teacher_thresholds["min_seeds"]),
            min_fill_orders=cast(int, teacher_thresholds["min_fill_orders"]),
            max_drawdown=cast(float, teacher_thresholds["max_drawdown"]),
            require_all_scenarios_positive=cast(
                bool, teacher_thresholds["require_all_scenarios_positive"]
            ),
            require_positive_teacher_net=cast(
                bool, teacher_thresholds["require_positive_teacher_net"]
            ),
            expected_scenarios=cast(tuple[str, ...], teacher_thresholds["expected_scenarios"]),
        )
        if teacher.receipt_sha256 != screen.teacher_gate_sha256:
            return False
        thresholds = dict(screen.thresholds)
        recomputed = screen_student(
            screen.paired_records,
            teacher_screen=teacher,
            min_windows=cast(int, thresholds["min_windows"]),
            min_seeds=cast(int, thresholds["min_seeds"]),
            min_filled_orders=cast(int, thresholds["min_filled_orders"]),
            max_drawdown=cast(float, thresholds["max_drawdown"]),
            expected_scenarios=cast(tuple[str, ...], thresholds["expected_scenarios"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return recomputed == screen
