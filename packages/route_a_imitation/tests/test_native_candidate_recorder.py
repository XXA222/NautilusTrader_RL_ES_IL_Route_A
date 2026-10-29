from dataclasses import replace

import pytest

from rles_route_a.contracts import ExpertAdvice, NativeActionTrace
from rles_route_a.teacher_research.native_candidate_recorder import NativeCandidateRecorder, write_candidate_records
from test_candidate_collector import observation


class Primary:
    def __init__(self):
        self.actions = []
    def advise(self, obs):
        return ExpertAdvice("existing", obs.decision_ts_ns, "HOLD",
                            obs.account["net_exposure"], ("PRESERVE_POSITION",))
    def observe_native_action(self, *args):
        self.actions.append(args)


def test_native_recorder_keeps_candidates_without_changing_position_or_receipt(tmp_path):
    primary = Primary()
    recorder = NativeCandidateRecorder(primary)
    recorder.advise(observation(1, exposure=.3))
    obs = observation(2, exposure=.3, pending=("native-order",), ema_9=101.)
    action = recorder.advise(obs)
    assert action.expert_id == "existing" and action.requested_exposure == .3
    record = recorder.require_complete()[-1]
    group, = record.collection.groups
    assert group.observation is obs and group.side == 1
    trace = NativeActionTrace(obs.decision_ts_ns, .3, .3, .3, .3, 10000., 10000., "NOOP")
    recorder.observe_native_action(action, trace, "native-facts")
    assert primary.actions == [(action, trace, "native-facts")]
    import json
    output = tmp_path / "candidates.jsonl"
    write_candidate_records(recorder, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[-1]["pending_order_ids"] == ["native-order"]
    assert rows[-1]["candidates"][0]["event_id"] == group.event_id
    with pytest.raises(FileExistsError):
        write_candidate_records(recorder, output)


def test_collection_failure_does_not_block_existing_risk_and_fails_research():
    primary = Primary()
    recorder = NativeCandidateRecorder(primary)
    obs = replace(observation(1, exposure=.3), snapshot_id="")
    assert recorder.advise(obs).reason_codes == ("PRESERVE_POSITION",)
    with pytest.raises(ValueError, match="failed"):
        recorder.require_complete()



def test_candidate_runner_requires_complete_matching_native_barriers():
    from types import SimpleNamespace
    from rles_route_a.integration.gym_rollout import NativeGymTeacherProvider
    from rles_route_a.integration.scope import HostTrainGrant
    from rles_route_a.teacher_research.native_candidate_recorder import run_native_candidate_collection

    recorder = NativeCandidateRecorder(Primary())
    obs = observation(1)
    recorder.advise(obs)
    grant = HostTrainGrant("train", "owner", "reservation", "synthetic", "train", True)
    provider = NativeGymTeacherProvider(env_factory=lambda: None, teacher=recorder,
        fold_id="train", source_identity="synthetic", execution_recipe="transition",
        native_order_facts_verifier=lambda *_: True, no_external_cashflow_receipt=lambda *_: "r",
        no_external_cashflow_verifier=lambda *_: True, episode_id=lambda *_: "e")
    trace = SimpleNamespace(decision_ts_ns=obs.decision_ts_ns)
    provider.rollout_with_m22_trace = lambda **kwargs: ((SimpleNamespace(trace=trace),),
        {"route_a_native_order_verification":{"barriers":1,"authority":"retained_m22_session_snapshots"}})
    args = dict(provider=provider, recorder=recorder,
        factory=SimpleNamespace(transition_hash="transition"), host_rollout=lambda: None,
        grant=grant, verify_grant=lambda value: value == grant, seed=7, rollout_id="synthetic",
        episode_steps=1)
    barriers, _, records = run_native_candidate_collection(**args)
    assert len(barriers) == len(records) == 1
    trace.decision_ts_ns += 1
    with pytest.raises(ValueError, match="matching M2.2"):
        run_native_candidate_collection(**args)
