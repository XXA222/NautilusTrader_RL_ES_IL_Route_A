"""Write compact trace/label metadata, not copied L3 native account ledgers.

For large numeric feature payloads the existing L2 Arrow/Parquet storage owns IO.
This writer contains only references and teacher attribution metadata.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence
from rles_route_a.datasets.sequence_dataset import LabeledStep


def write_label_index(steps: Sequence[LabeledStep], destination: str | Path, *,
                      source_identity: str, host_reservation_id: str) -> Path:
    if not steps or not source_identity or not host_reservation_id:
        raise ValueError("registered source and nonempty labels required")
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for s in steps:
            rec = {"fold_id": s.observation.fold_id, "episode_id": s.observation.episode_id,
                   "decision_ts_ns": s.observation.decision_ts_ns,
                   "teacher_id": s.advice.expert_id, "intent": s.advice.intent,
                   "teacher_requested_exposure": s.advice.requested_exposure,
                   "realizable_label": s.executable_exposure,
                   "reason_codes": s.advice.reason_codes, "native_status": s.action_status,
                   "economic_case_status": s.economic_case_status,
                   "case_outcome_bps": s.case_outcome_bps,
                   "case_end_ts_ns": s.case_end_ts_ns,
                   "case_fact_sha256": s.case_fact_sha256,
                   "source_identity": source_identity, "host_reservation_id": host_reservation_id,
                   "split": s.split}
            f.write(json.dumps(rec, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n")
    return path
