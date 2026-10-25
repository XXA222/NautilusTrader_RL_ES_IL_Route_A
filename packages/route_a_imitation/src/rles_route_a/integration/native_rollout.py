"""L3-only native fact validation; does not execute or fake fills."""
from __future__ import annotations
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Callable, Iterable, Protocol
import math
from rles_route_a.contracts import ExpertAdvice, NativeActionTrace
from rles_route_a.datasets.sequence_dataset import LabeledStep
from rles_route_a.contracts import PointInTimeObservation
from .native_economics import NativeRewardReceipt


@dataclass(frozen=True)
class TeacherFact:
    observation: PointInTimeObservation
    advice: ExpertAdvice
    native: NativeActionTrace


class NativeTeacherFact(Protocol):
    @property
    def observation(self) -> PointInTimeObservation: ...

    @property
    def advice(self) -> ExpertAdvice: ...

    @property
    def native(self) -> NativeActionTrace: ...


def native_teacher_step(fact: NativeTeacherFact, *, tolerance: float = 1e-6) -> LabeledStep:
    """Keep native exposure as evidence; BC learns the teacher's requested action.

    A blocked or unfilled command must not be labelled as if a fill occurred.
    For multi-step cancel/ACK, feed subsequent native barriers as separate facts.
    """
    o, advice, trace = fact.observation, fact.advice, fact.native
    if o.decision_ts_ns != advice.decision_ts_ns or o.decision_ts_ns != trace.decision_ts_ns:
        raise ValueError("teacher/native time mismatch")
    if not math.isclose(advice.requested_exposure, trace.requested_action, abs_tol=tolerance):
        raise ValueError("teacher command does not match native requested action")
    status = trace.planner_action
    economic_case_status = "UNASSESSED"
    case_outcome_bps = None
    case_end_ts_ns = None
    case_fact_sha256 = ""
    reward = getattr(fact, "reward", None)
    source_identity = getattr(fact, "source_identity", "")
    if reward is not None:
        if not isinstance(reward, NativeRewardReceipt):
            raise TypeError("teacher economic case requires an L6 native reward receipt")
        reward.validate()
        if (abs(reward.prior_equity - trace.native_equity_before) > tolerance
                or abs(reward.next_equity - trace.native_equity_after) > tolerance):
            raise ValueError("teacher case reward receipt differs from native equity trace")
        candidate_end_ts_ns = trace.execution_barrier_ts_ns
        if (type(candidate_end_ts_ns) is int
                and candidate_end_ts_ns > o.decision_ts_ns
                and isinstance(source_identity, str)
                and source_identity.strip()):
            case_end_ts_ns = candidate_end_ts_ns
            case_outcome_bps = reward.recorded_r_econ * 10_000.0
            if trace.scenario_only:
                economic_case_status = "SCENARIO_ONLY"
            elif case_outcome_bps > 0:
                economic_case_status = "OBSERVED_NATIVE_STEP_POSITIVE"
            elif case_outcome_bps < 0:
                economic_case_status = "OBSERVED_NATIVE_STEP_NEGATIVE"
            else:
                economic_case_status = "OBSERVED_NATIVE_STEP_NEUTRAL"
            case_payload = {
                "source_identity": source_identity,
                "episode_id": o.episode_id,
                "decision_ts_ns": o.decision_ts_ns,
                "case_end_ts_ns": case_end_ts_ns,
                "execution_decision_id": trace.execution_decision_id,
                "execution_plan_id": trace.execution_plan_id,
                "execution_recipe": getattr(fact, "execution_recipe", ""),
                "native_equity_before": trace.native_equity_before,
                "native_equity_after": trace.native_equity_after,
                "recorded_r_econ": reward.recorded_r_econ,
                "has_external_cashflow": reward.has_external_cashflow,
                "host_cashflow_adjustment_receipt": reward.host_cashflow_adjustment_receipt,
            }
            case_fact_sha256 = sha256(json.dumps(
                case_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")).hexdigest()
    # Native SUBMIT does NOT imply a fill. A fill from an older, unrelated
    # pending order does not prove that this decision's requested trade worked.
    # Actual exposure must reach the *planner* target without residual pending
    # orders before legacy executed-only exposure supervision treats a position
    # as execution evidence. New command-learning recipes can still learn the
    # separately validated requested action without claiming that it filled.
    current_fill_proven = bool(set(trace.filled_order_ids) & set(trace.submitted_order_ids))
    settled_target = (abs(trace.actual_exposure_after - trace.planner_target) <= tolerance
                      and not trace.pending_order_ids_after)
    if status == "SUBMIT":
        status = "FILLED" if current_fill_proven and settled_target else "UNFILLED"
    if status in {"CANCEL", "WAIT", "UNFILLED"}:
        exposure = trace.actual_exposure_after
        weight = 0.0
    elif status == "NOOP" and (
            trace.pending_order_ids_after
            or (abs(trace.actual_exposure_after - trace.planner_target) > tolerance
                and trace.planner_reason not in {
                    "deadband", "native account already flat",
                    "quantity rounded to no change",
                })):
        # A no-op is executable supervision only after all native orders settle
        # and the host planner confirms it is an intentional settled no-op. The
        # deadband decision itself proves the planner accepted the current
        # exposure; small mark moves need not equal the target bit-for-bit.
        exposure = trace.actual_exposure_after
        weight = 0.0
    else:
        exposure = trace.actual_exposure_after
        weight = 1.0
    # This L6 receipt settles one adjacent native barrier, not a specific trade's
    # full entry-to-exit lifecycle. Retain it as diagnostics, but do not train a
    # behavior clone from step PnL until a separately bound case receipt exists.
    if reward is not None:
        weight = 0.0
    return LabeledStep(o, advice, exposure, action_status=status,
                       split="train", supervision_weight=weight,
                       intent_supervision_weight=weight if reward is not None else 1.0,
                       economic_case_status=economic_case_status,
                       case_outcome_bps=case_outcome_bps,
                       case_end_ts_ns=case_end_ts_ns,
                       case_fact_sha256=case_fact_sha256)


def teacher_native_rollout(provider: Callable[..., Iterable[NativeTeacherFact]], *,
                           fold_id: str, grant: object) -> tuple[LabeledStep, ...]:
    """Provider must use ProjectResearchRegistry+scoped source BEFORE market IO."""
    from .scope import HostTrainGrant
    if not isinstance(grant, HostTrainGrant) or grant.fold_id != fold_id:
        raise PermissionError("host-provided registered Train grant required before rollout")
    samples = []
    for fact in provider(fold_id=fold_id, grant=grant):
        source_identity = getattr(fact, "source_identity", grant.source_identity)
        if source_identity != grant.source_identity:
            raise PermissionError("native teacher fact source differs from registered Train grant")
        trajectory_id = getattr(fact, "trajectory_id", "")
        if not isinstance(trajectory_id, str) or (trajectory_id and not trajectory_id.strip()):
            raise ValueError("native teacher trajectory identity is malformed")
        samples.append(replace(
            native_teacher_step(fact), source_identity=grant.source_identity,
            trajectory_id=trajectory_id,
        ))
    return tuple(samples)
