"""Synthetic host contract test; no real exchange or market data is accessed."""
from hashlib import sha256
import json

from rles_route_a.integration.gym_rollout import NativeGymTeacherProvider
from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.teacher_research.meta_gated_teacher import MetaGatedTeacher
from test_candidate_collector import observation
from test_meta_gated_teacher import Recovery, trained_policy
from test_v03_host_context import _feature


def test_meta_teacher_receives_verified_gym_barrier():
    obs = observation(10)
    class Env:
        def reset(self, *, seed):
            return None, {}
        def close(self):
            pass
        def step(self, action):
            receipt = sha256(json.dumps({"before": (), "after": ()}, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            info = {
                "decision_ts_ns": obs.decision_ts_ns + 60,
                "terminal_settled": False, "execution_plan_action": "NOOP",
                "execution_client_order_id": None, "execution_related_order_ids": (),
                "native_pending_order_ids_before": (), "native_pending_order_ids_after": (),
                "native_policy_fills": (), "native_order_statuses_before": (),
                "native_order_statuses_after": (), "native_order_status_receipt": receipt,
                "execution_decision_id": "decision-1", "execution_plan_id": "plan-1",
                "execution_idempotency_key": "idempotency-1", "execution_plan_reason": "flat",
                "evidence_hash": "e"*64, "policy_equity_currency": "USDT",
                "requested_action": action[0], "risk_soft_target": action[0],
                "risk_adjusted_target": action[0], "quantity_rounded_target_exposure": action[0],
                "policy_executed_action_pre_terminal": action[0],
                "policy_equity_before": "10000", "policy_equity_after": "10000",
                "policy_equity_change": "0", "r_econ": 0., "execution_cancel_order_ids": (),
            }
            return None, 0., False, False, info

    def visible(_env):
        return {"decision_ts_ns": obs.decision_ts_ns, "source_id": obs.source_id,
                "snapshot_id": obs.snapshot_id, "feature_spec_hash": obs.feature_spec_hash,
                "market": {name: _feature(value, obs.decision_ts_ns)
                           for name, value in obs.market.items()},
                "account": dict(obs.account), "pending_order_ids": (),
                "risk": {}, "execution": {}, "time": {}}

    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    provider = NativeGymTeacherProvider(
        env_factory=Env, teacher=teacher, fold_id="train", source_identity="synthetic",
        execution_recipe="native", market_feature_map={k: k for k in obs.market},
        visible_context=visible, episode_id=lambda _env, _info: "e",
        native_order_facts_verifier=lambda *_: True,
        no_external_cashflow_receipt=lambda _: "no-flow",
        no_external_cashflow_verifier=lambda *_: True)
    grant = HostTrainGrant("train", "owner", "reservation", "synthetic", "train", True)
    rollout = provider.rollout(fold_id="train", grant=grant, seed=7)
    barrier = next(rollout)
    rollout.close()
    assert barrier.trace.requested_action == 0.
    assert len(teacher.decisions) == len(teacher.native_receipts) == 1
    assert teacher.native_receipts[0].trace is barrier.trace
    assert teacher.native_receipts[0].selected_candidate_event_id is None



def test_meta_teacher_multistep_native_order_and_fill_receipts():
    class Env:
        def __init__(self):
            self.minute = 10
            self.exposure = 0.
        def reset(self, *, seed):
            return None, {}
        def close(self):
            pass
        def step(self, action):
            now = observation(self.minute).decision_ts_ns
            target = action[0]
            submitted = target != self.exposure
            order_id = f"synthetic-order-{self.minute}" if submitted else None
            status = ({"client_order_id": order_id, "status": "FILLED",
                       "side": "BUY" if target > self.exposure else "SELL",
                       "quantity": "1", "filled_qty": "1"},) if submitted else ()
            receipt = sha256(json.dumps({"before": (), "after": status}, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            fills = ({"event_id": f"synthetic-fill-{self.minute}", "client_order_id": order_id,
                      "ts_event": now + 30, "last_qty": "1", "last_px": "100",
                      "commission": "0.1 USDT"},) if submitted else ()
            info = {
                "decision_ts_ns": now + 60, "terminal_settled": False,
                "execution_plan_action": "SUBMIT" if submitted else "NOOP",
                "execution_client_order_id": order_id, "execution_related_order_ids": (),
                "native_pending_order_ids_before": (), "native_pending_order_ids_after": (),
                "native_policy_fills": fills, "native_order_statuses_before": (),
                "native_order_statuses_after": status, "native_order_status_receipt": receipt,
                "execution_decision_id": f"decision-{self.minute}",
                "execution_plan_id": f"plan-{self.minute}",
                "execution_idempotency_key": f"idempotency-{self.minute}",
                "execution_plan_reason": "synthetic", "evidence_hash": "e"*64,
                "policy_equity_currency": "USDT", "requested_action": target,
                "risk_soft_target": target, "risk_adjusted_target": target,
                "quantity_rounded_target_exposure": target,
                "policy_executed_action_pre_terminal": target,
                "policy_equity_before": "10000", "policy_equity_after": "10000",
                "policy_equity_change": "0", "r_econ": 0., "execution_cancel_order_ids": (),
            }
            self.exposure = target
            self.minute += 1
            return None, 0., False, False, info

    def visible(env):
        obs = observation(env.minute, ema_9=101. if env.minute in (11, 12) else 99.,
                          exposure=env.exposure)
        return {"decision_ts_ns": obs.decision_ts_ns, "source_id": obs.source_id,
                "snapshot_id": obs.snapshot_id, "feature_spec_hash": obs.feature_spec_hash,
                "market": {name: _feature(value, obs.decision_ts_ns)
                           for name, value in obs.market.items()},
                "account": dict(obs.account), "pending_order_ids": (),
                "risk": {}, "execution": {}, "time": {}}

    teacher = MetaGatedTeacher(trained_policy(), recovery=Recovery())
    initial = observation(10)
    provider = NativeGymTeacherProvider(
        env_factory=Env, teacher=teacher, fold_id="train", source_identity="synthetic",
        execution_recipe="native", market_feature_map={k: k for k in initial.market},
        visible_context=visible, episode_id=lambda _env, _info: "e",
        native_order_facts_verifier=lambda *_: True,
        no_external_cashflow_receipt=lambda _: "synthetic-no-flow",
        no_external_cashflow_verifier=lambda *_: True)
    grant = HostTrainGrant("train", "owner", "reservation", "synthetic", "train", True)
    rollout = provider.rollout(fold_id="train", grant=grant, seed=7)
    barriers = tuple(next(rollout) for _ in range(4))
    rollout.close()
    assert [r.action.intent for r in teacher.native_receipts] == ["FLAT", "OPEN_LONG", "HOLD", "EXIT"]
    assert teacher.native_receipts[1].trace.filled_order_ids == ("synthetic-order-11",)
    assert teacher.native_receipts[3].trace.filled_order_ids == ("synthetic-order-13",)
    selected = next(item for item in teacher.decisions[1].candidates if item.selected)
    assert teacher.native_receipts[1].selected_candidate_event_id == selected.group.event_id
    assert teacher.native_receipts[1].selected_teacher_id == "T01_EMA_9_21"
    assert teacher.native_receipts[3].selected_candidate_event_id is None
    assert len(barriers) == 4 and len(teacher.decisions) == 4
