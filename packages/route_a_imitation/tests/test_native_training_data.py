from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rles_route_a.integration.scope import HostTrainGrant
from rles_route_a.teacher_research.native_candidate_recorder import NativeCandidateRecorder
from rles_route_a.teacher_research.native_training_data import training_batch_from_native
from test_candidate_collector import observation
from test_native_candidate_recorder import Primary

M = 60_000_000_000


def epoch(text):
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()) * 1_000_000_000


def timed_obs(ts, fast):
    base = observation(1, ema_9=fast)
    market = {**dict(base.market), "latest_1m_close_ts_ns": float(ts)}
    return replace(base, decision_ts_ns=ts, market=market,
                   feature_available_at_ns={name: ts for name in market},
                   exact_market_clocks_ns=(("latest_1m_close_ts_ns", ts),),
                   source_id="a"*64, fold_id="train", snapshot_id=f"{ts:064x}")


def test_batch_uses_verified_receipt_and_does_not_read_screen_payload(monkeypatch):
    from rles_route_a.teacher_research import native_training_data as module
    root = Path(__file__).resolve().parents[3]
    prereg = json.loads((root / "configs/route_a_meta_x1_train_prereg_20261029.json").read_text())
    fit = epoch(prereg["intervals"]["train_fit"][0])
    cal = epoch(prereg["intervals"]["train_calibration"][0])
    screen = epoch(prereg["intervals"]["native_screen"][0][0])
    recorder = NativeCandidateRecorder(Primary())
    for ts, fast in ((fit,99.), (fit+M,101.), (cal,99.), (cal+M,99.), (cal+2*M,101.)):
        recorder.advise(timed_obs(ts, fast))
    owner = prereg["protocol_id"]
    read = SimpleNamespace(split="train", registered_before_read=True, owner=owner,
        reservation_id="reservation", source_identity={"marker_sha256": "a"*64})
    grant = HostTrainGrant.from_registered_read(read, fold_id="train")
    class ScreenEvent:
        ts_event_ns = screen
        kind = "mark"
        @property
        def payload(self):
            raise AssertionError("screen payload must not be read")
    class EarlySettlement:
        ts_event_ns = epoch(prereg["intervals"]["warmup"][0])
        kind = "funding_settlement"
        @property
        def payload(self):
            raise AssertionError("pre-decision settlement is irrelevant")
    mark = SimpleNamespace(ts_event_ns=fit, kind="mark",
                           payload={"instrument":"ETH", "price":"100"})
    rate = SimpleNamespace(ts_event_ns=fit+1, kind="funding_settlement",
                           payload={"instrument":"ETH", "rate":"0.0001"})
    factory = SimpleNamespace(application_registered_train_read=read,
        application_registered_train_read_verifier=lambda receipt: receipt is read,
        instrument="ETH", store=SimpleNamespace(events=(EarlySettlement(),mark,rate,ScreenEvent())))
    bars = (SimpleNamespace(open_time_ns=fit, open=100.),
            SimpleNamespace(open_time_ns=fit+M, open=101.))
    monkeypatch.setattr(module, "causal_1m_indicator_inputs", lambda *args, **kwargs: bars)
    batch = training_batch_from_native(recorder=recorder, factory=factory, grant=grant,
                                      verify_grant=lambda g: g == grant, prereg=prereg)
    assert len(batch.fit_groups) == len(batch.calibration_groups) == 1
    assert batch.opens == ((fit,100.),(fit+M,101.))
    assert batch.funding == ((fit+1, .0001, 100.),)
    with pytest.raises(PermissionError, match="receipt"):
        training_batch_from_native(recorder=recorder,
            factory=SimpleNamespace(**{**vars(factory), "application_registered_train_read_verifier": lambda _: False}),
            grant=grant, verify_grant=lambda _: True, prereg=prereg)



def test_rejected_training_retains_candidates_and_all_labels(tmp_path):
    from rles_route_a.teacher_research.conditional_outcome_labels import GroupOutcome
    from rles_route_a.teacher_research.native_training_data import save_rejected_training_artifacts
    from rles_route_a.teacher_research.train_meta import MetaTrainingError
    recorder = NativeCandidateRecorder(Primary())
    recorder.advise(observation(1))
    recorder.advise(observation(2, ema_9=101.))
    group, = recorder.records[-1].collection.groups
    obs = group.observation
    outcome = GroupOutcome(group.event_id, obs.decision_ts_ns, obs.source_id, obs.fold_id,
        "X1", "recipe", "base", None, obs.decision_ts_ns + 1, None,
        outcome_status="RIGHT_CENSORED")
    failure = MetaTrainingError("too few", fit_outcomes=(outcome,),
                                calibration_outcomes=(), diagnostics={"fit_candidates": 1})
    path = tmp_path / "rejected"
    save_rejected_training_artifacts(failure=failure, recorder=recorder, output_dir=path)
    assert (path / "REJECTED").is_file() and not (path / "policy.json").exists()
    assert json.loads((path / "diagnostics.json").read_text())["status"] == "TRAIN_REJECTED"
    assert json.loads((path / "outcomes.jsonl").read_text())["outcome_status"] == "RIGHT_CENSORED"
    assert len((path / "candidate-barriers.jsonl").read_text().splitlines()) == 2



def test_synthetic_native_candidate_to_frozen_model_pipeline(tmp_path, monkeypatch):
    """Synthetic host facts only: checks candidate, X1 labels, fit and persistence."""
    from hashlib import sha256
    import random
    from datetime import timedelta, timezone
    from rles_execution.spec import ResolvedExecutionConfig
    from rles_route_a.teacher_research import native_training_data as module
    from rles_route_a.teacher_research.policy_io import load_frozen_entry_policy
    from rles_route_a.teacher_research.native_training_data import train_and_save_from_native

    root = Path(__file__).resolve().parents[3]
    prereg = json.loads((root / "configs/route_a_meta_x1_train_prereg_20261029.json").read_text())
    start = datetime(2026, 12, 1, tzinfo=timezone.utc)
    def stamp(minutes):
        return (start + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")
    prereg["intervals"] = {"warmup":[stamp(0),stamp(10)],
        "train_fit":[stamp(10),stamp(28)], "train_calibration":[stamp(28),stamp(50)],
        "native_screen":[[stamp(50),stamp(60)]]}
    prereg["model"].update({"minimum_complete_fit":4,"minimum_complete_calibration":4,
        "features":["market.ema_9", "teacher.T01_EMA_9_21",
                    "consensus.count", "cost.round_trip_bps"]})
    config = tmp_path / "synthetic-prereg.json"
    raw = json.dumps(prereg).encode()
    config.write_bytes(raw)
    config.with_suffix(".json.sha256").write_text(sha256(raw).hexdigest())
    source_start = epoch(stamp(0))
    recorder = NativeCandidateRecorder(Primary())
    for minute in range(50):
        close = source_start + minute*M
        base = timed_obs(close, 101. if minute % 4 < 2 else 99.)
        obs = replace(base, decision_ts_ns=close+1_000_000,
            feature_available_at_ns={name: close+1_000_000 for name in base.market})
        recorder.advise(obs)
    owner = prereg["protocol_id"]
    read = SimpleNamespace(split="train", registered_before_read=True, owner=owner,
        reservation_id="reservation", source_identity={"marker_sha256":"a"*64})
    grant = HostTrainGrant.from_registered_read(read, fold_id="train")
    execution_doc = json.loads((root / "configs/ethusdt_20240101_real.json").read_text())
    execution = ResolvedExecutionConfig.from_dict(execution_doc["environment"]["execution"])
    factory = SimpleNamespace(application_registered_train_read=read,
        application_registered_train_read_verifier=lambda receipt: receipt is read,
        instrument="ETH", store=SimpleNamespace(events=()),
        features=SimpleNamespace(hash="0"*64), mode="base", execution=execution,
        route_a_execution_recipe_hash="stable-recipe",
        rule=SimpleNamespace(payload={"taker_fee":"0.0005"}))
    rng = random.Random(42)
    bars = tuple(SimpleNamespace(open_time_ns=source_start+minute*M,
                                 open=100.+rng.uniform(-5,5)) for minute in range(50))
    monkeypatch.setattr(module, "causal_1m_indicator_inputs", lambda *a, **kw: bars)
    result = train_and_save_from_native(prereg_path=config, recorder=recorder,
        collection_factory=factory, collection_grant=grant,
        training_factory=factory, training_grant=grant,
        verify_grant=lambda value: value == grant, scenario="base", output_dir=tmp_path/"fit")
    assert result.diagnostics["fit_complete_used"] >= 4
    assert result.diagnostics["calibration_complete_used"] >= 4
    assert result.policy.label_recipe == ("X1", "stable-recipe", "base")
    assert load_frozen_entry_policy(tmp_path/"fit"/"policy.json").fit_count == result.policy.fit_count
    assert (tmp_path/"fit"/"COMPLETE").read_text().strip() == "MODEL_COUNTERFACTUAL_TRAIN_ONLY"
    assert len((tmp_path/"fit"/"candidate-barriers.jsonl").read_text().splitlines()) == 50
