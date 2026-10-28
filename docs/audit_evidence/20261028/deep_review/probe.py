import json, runpy
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import torch
torch.set_num_threads(1)
from rles_route_a.config import BCConfig
from rles_route_a.models.bc import BCGRUPolicy
from rles_route_a.export.inference import BCInferencePolicy
from rles_route_a.contracts import PointInTimeObservation
names=("market.x","account.net_exposure")
cfg=BCConfig(feature_names=names,context_bars=3,hidden_size=8,layers=1,inference_mode="intent_aware_v1",open_intent_min_confidence=0.9)
model=BCGRUPolicy(names,8,1)
with torch.no_grad():
    model.exposure_head.weight.zero_()
    model.exposure_head.bias.fill_(0.5)
    model.intent_head.weight.zero_()
    model.intent_head.bias.zero_()
    model.intent_head.bias[1]=float("nan")
obs=PointInTimeObservation(episode_id="same",decision_ts_ns=100,market={"x":1.0},feature_available_at_ns={"x":99},account={"net_exposure":0.0},fold_id="fold-A",source_id="source-A",feature_spec_hash="a"*64)
policy=BCInferencePolicy(model,cfg)
policy.reset("same")
print("nonfinite_intent_result",policy.predict_exposure(obs),"clock",policy._last_ts)
with torch.no_grad(): model.intent_head.bias.zero_()
next_obs=replace(obs,decision_ts_ns=200,fold_id="fold-B",source_id="source-B",feature_spec_hash="b"*64)
print("changed_identity_result",policy.predict_exposure(next_obs),"retained_history",len(policy._series))
fixture=runpy.run_path("tests/test_route_a_teacher_train_screen_runner.py")
runner=__import__("qa.run_route_a_teacher_student_train_screen_20261026",fromlist=["_completed_run"])
report=fixture["_native_run"](candidate="T01_EMA_9_21",scenario="base",seed=20261026,net_return=.1,window_start=100,window_end=200,marker_sha="a"*64)
report["preregistration_sha256"]="OLD_PROTOCOL"
report["verified_feature_recipe_sha256"]="OLD_FEATURES"
report["execution"]["scenario_costs"]={"taker_fee_bps_per_leg":0}
report["runtime_code_sha256"]={"teacher.py":"OLD_CODE"}
expected={"protocol_id":report["protocol_id"],"teacher":report["teacher"],"teacher_recipe":report["teacher_recipe"],"marker_sha256":"a"*64,"window_ns":[100,200],"scenario":"base","seed":20261026,"preregistration_sha256":"NEW_PROTOCOL","runtime_code_sha256":{"teacher.py":"NEW_CODE"}}
with patch.object(Path,"exists",return_value=True),patch.object(Path,"is_file",return_value=True),patch.object(runner,"_read_json",return_value=report):
    print("stale_code_and_cost_resume_accepted",runner._completed_run(Path("synthetic"),expected) is report)
print("wheel_guard_root_example",Path("/project/.venv/lib/python3.14/site-packages/rles_data/binance_archive.py").parents[4])
