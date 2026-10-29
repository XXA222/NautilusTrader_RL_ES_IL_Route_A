"""Prepare research labels from already verified native Train barriers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from rles_data.features import causal_1m_indicator_inputs
from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.integration.scope import HostTrainGrant
from .candidate_event_union import CandidateGroup
from .causal_exit_replay import attach_funding_mark_proxies
from .native_candidate_recorder import NativeCandidateRecorder


@dataclass(frozen=True, slots=True)
class NativeTrainingBatch:
    observations: tuple[PointInTimeObservation, ...]
    fit_groups: tuple[CandidateGroup, ...]
    calibration_groups: tuple[CandidateGroup, ...]
    opens: tuple[tuple[int, float], ...]
    funding: tuple[tuple[int, float, float], ...]


def _ns(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()) * 1_000_000_000


def training_batch_from_native(
    *, recorder: NativeCandidateRecorder, factory: Any, grant: HostTrainGrant,
    verify_grant: Callable[[HostTrainGrant], bool], prereg: dict,
) -> NativeTrainingBatch:
    """Use only host-authorized warmup/fit/calibration observations and prices.

    Funding marks are explicit minute-mark proxies, not authoritative account
    funding receipts. No native-screen outcome enters this batch.
    """
    if (not isinstance(recorder, NativeCandidateRecorder) or not isinstance(grant, HostTrainGrant)
            or not callable(verify_grant) or verify_grant(grant) is not True):
        raise PermissionError("verified native Train candidate collection required")
    read = getattr(factory, "application_registered_train_read", None)
    verifier = getattr(factory, "application_registered_train_read_verifier", None)
    if (not callable(verifier) or verifier(read) is not True
            or HostTrainGrant.from_registered_read(read, fold_id=grant.fold_id) != grant):
        raise PermissionError("candidate batch lacks the same verified host Train receipt")
    if grant.owner != prereg["protocol_id"]:
        raise PermissionError("Train grant owner differs from the independent preregistration")
    rows = recorder.require_complete()
    warmup_start = _ns(prereg["intervals"]["warmup"][0])
    fit_start, fit_end = map(_ns, prereg["intervals"]["train_fit"])
    cal_start, cal_end = map(_ns, prereg["intervals"]["train_calibration"])
    observations = tuple(row.observation for row in rows)
    for obs in observations:
        if (not warmup_start <= obs.decision_ts_ns < cal_end
                or (obs.source_id, obs.fold_id) != (grant.source_identity, grant.fold_id)):
            raise ValueError("native candidate observation falls outside registered model-fit scope")
    fit_groups = tuple(group for row in rows for group in row.collection.groups
                       if fit_start <= group.observation.decision_ts_ns < fit_end)
    calibration_groups = tuple(group for row in rows for group in row.collection.groups
                               if cal_start <= group.observation.decision_ts_ns < cal_end)
    opens, funding = extract_model_price_path(factory, start_ns=warmup_start, end_ns=cal_end,
                                              funding_after_ns=observations[0].decision_ts_ns)
    return NativeTrainingBatch(observations, fit_groups, calibration_groups, opens, funding)




def extract_model_price_path(factory, *, start_ns: int, end_ns: int, funding_after_ns: int | None = None):
    """Extract only the authorized label interval from the existing native store."""
    if type(start_ns) is not int or type(end_ns) is not int or not 0 < start_ns < end_ns:
        raise ValueError("valid model price-path interval required")
    if funding_after_ns is not None and (type(funding_after_ns) is not int
                                         or not start_ns <= funding_after_ns < end_ns):
        raise ValueError("funding cutoff must be inside the model interval")
    store = getattr(factory, "store", None)
    instrument = getattr(factory, "instrument", None)
    if store is None or not isinstance(instrument, str) or not instrument:
        raise ValueError("native factory lacks the already authorized source store")
    bars = causal_1m_indicator_inputs(store, end_ns - 1, instrument=instrument,
                                       start_event_ns=start_ns, require_latest=False)
    opens = tuple((bar.open_time_ns, float(bar.open)) for bar in bars
                  if start_ns <= bar.open_time_ns < end_ns)
    rates, marks = [], {}
    for event in store.events:
        if not start_ns <= event.ts_event_ns < end_ns or event.kind not in ("funding_settlement", "mark"):
            continue
        # All modeled entries occur strictly after the first decision, so an
        # earlier settlement cannot contribute to any candidate's funding.
        if (event.kind == "funding_settlement" and funding_after_ns is not None
                and event.ts_event_ns <= funding_after_ns):
            continue
        if event.payload.get("instrument") != instrument:
            continue
        if event.kind == "funding_settlement":
            rates.append((event.ts_event_ns, float(event.payload["rate"])))
        elif event.kind == "mark":
            value = float(event.payload["price"])
            if event.ts_event_ns in marks and marks[event.ts_event_ns] != value:
                raise ValueError("conflicting funding mark prices")
            marks[event.ts_event_ns] = value
    rates = tuple(sorted(rates))
    funding = attach_funding_mark_proxies(rates, tuple(sorted(marks.items()))) if rates else ()
    return opens, funding


def save_training_artifacts(*, result, recorder: NativeCandidateRecorder, output_dir) -> None:
    """Save immutable candidate rows, every modeled label, diagnostics and policy."""
    from dataclasses import asdict
    import json
    from pathlib import Path
    from .native_candidate_recorder import write_candidate_records
    from .policy_io import save_frozen_entry_policy
    from .train_meta import MetaTrainingResult

    if not isinstance(result, MetaTrainingResult):
        raise TypeError("trained Meta result required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_candidate_records(recorder, output / "candidate-barriers.jsonl")
    with (output / "outcomes.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for split, labels in (("train_fit", result.fit_outcomes),
                              ("train_calibration", result.calibration_outcomes)):
            for label in labels:
                stream.write(json.dumps({"split": split, **asdict(label)}, sort_keys=True,
                                         allow_nan=False, separators=(",", ":")) + "\n")
    (output / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    save_frozen_entry_policy(result.policy, output / "policy.json")
    (output / "COMPLETE").write_text("MODEL_COUNTERFACTUAL_TRAIN_ONLY\n", encoding="ascii")



def save_rejected_training_artifacts(*, failure, recorder: NativeCandidateRecorder, output_dir) -> None:
    """Retain candidate/label coverage and the reason when a model cannot fit."""
    from dataclasses import asdict
    import json
    from pathlib import Path
    from .native_candidate_recorder import write_candidate_records
    from .train_meta import MetaTrainingError

    if not isinstance(failure, MetaTrainingError):
        raise TypeError("diagnosed Meta training failure required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_candidate_records(recorder, output / "candidate-barriers.jsonl")
    with (output / "outcomes.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for split, labels in (("train_fit", failure.fit_outcomes),
                              ("train_calibration", failure.calibration_outcomes)):
            for label in labels:
                stream.write(json.dumps({"split": split, **asdict(label)}, sort_keys=True,
                                         allow_nan=False, separators=(",", ":")) + "\n")
    (output / "diagnostics.json").write_text(
        json.dumps({**failure.diagnostics, "status": "TRAIN_REJECTED", "reason": str(failure)},
                   sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "REJECTED").write_text("MODEL_NOT_AVAILABLE\n", encoding="ascii")



def train_and_save_from_native(
    *, prereg_path, recorder: NativeCandidateRecorder, collection_factory, collection_grant,
    training_factory, training_grant, verify_grant, scenario: str, output_dir,
):
    """Collect the already verified Train barriers, train the frozen model and save evidence."""
    from hashlib import sha256
    import json
    from pathlib import Path
    from .train_meta import MetaTrainingError, train_meta_from_preregistered_x1

    path = Path(prereg_path)
    raw = path.read_bytes()
    if path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip() != sha256(raw).hexdigest():
        raise ValueError("independent Meta preregistration hash mismatch")
    prereg = json.loads(raw)
    if (training_grant.source_identity, training_grant.fold_id, training_grant.owner) != (
            collection_grant.source_identity, collection_grant.fold_id, collection_grant.owner):
        raise PermissionError("candidate collection and model-training grants differ")
    batch = training_batch_from_native(recorder=recorder, factory=collection_factory,
        grant=collection_grant, verify_grant=verify_grant, prereg=prereg)
    from .native_candidate_recorder import summarize_teacher_signal_coverage
    signal_coverage = {split: summarize_teacher_signal_coverage(recorder.require_complete(),
        start_ns=_ns(prereg["intervals"][split][0]), end_ns=_ns(prereg["intervals"][split][1]))
        for split in ("train_fit", "train_calibration")}
    try:
        result = train_meta_from_preregistered_x1(prereg_path=path, grant=training_grant,
            verify_grant=verify_grant, fit_groups=batch.fit_groups,
            calibration_groups=batch.calibration_groups, observations=batch.observations,
            opens=batch.opens, funding=batch.funding, scenario=scenario, native_factory=training_factory)
    except MetaTrainingError as failure:
        failure.diagnostics["teacher_signal_coverage"] = signal_coverage
        save_rejected_training_artifacts(failure=failure, recorder=recorder, output_dir=output_dir)
        raise
    result.diagnostics["teacher_signal_coverage"] = signal_coverage
    save_training_artifacts(result=result, recorder=recorder, output_dir=output_dir)
    return result
