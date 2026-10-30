from dataclasses import replace
from hashlib import sha256

import pytest

from rles_route_a.teacher_research import (
    OpportunityEvent,
    OpportunityOutcome,
    fit_opportunity_model,
    feature_vector_sha256,
)

FEATURE_SPEC_HASH = sha256(b"feature-spec-v1").hexdigest()
RULE_VERSION = "candidate-v2"


def _snapshot_id(event_id):
    return sha256(f"snapshot-{event_id}".encode()).hexdigest()


def _outcome(
    index, *, split, decision, net, features=(0.0, 0.0),
    feature_names=("edge", "activity"), end=None, candidate_side=1,
    candidate_family="breakout", **overrides,
):
    event_id = sha256(f"event-{index}".encode()).hexdigest()
    values = dict(
        event_id=event_id,
        candidate_family=candidate_family,
        candidate_side=candidate_side,
        decision_ts_ns=decision,
        feature_available_at_ns=decision - 1,
        rule_version=RULE_VERSION,
        snapshot_id=_snapshot_id(event_id),
        feature_spec_hash=FEATURE_SPEC_HASH,
        feature_vector_sha256=feature_vector_sha256(
            event_id, FEATURE_SPEC_HASH, feature_names, features,
        ),
        label_end_ts_ns=decision + 10 if end is None and net is not None else end,
        net_return_bps=net,
        fold_id="fold-a",
        source_identity="registered-train-source",
        split=split,
        independent_block_id=f"block-{index // 4}",
        independent_block_start_ts_ns=max(0, decision - (index % 4) * 100),
        independent_block_end_ts_ns=max(0, decision - (index % 4) * 100) + 400,
        exit_policy_id="fixed-p0-v1",
        execution_recipe="market-like-v1",
        cost_scenario_id="base-12bp",
        return_basis_id="LOG_REFERENCE_ENTRY_NOTIONAL_BPS",
        simulation_receipt_id=f"sim-receipt-{index}",
        outcome_method="shared-flat-entry-and-fixed-exit-policy",
        evidence_type="MODEL_COUNTERFACTUAL",
        economic_status="SCENARIO_ONLY",
    )
    values.update(overrides)
    return OpportunityOutcome(**values)


def _candidate_event(row):
    return OpportunityEvent(
        event_id=row.event_id,
        decision_ts_ns=row.decision_ts_ns,
        feature_available_at_ns=row.feature_available_at_ns,
        side=row.candidate_side,
        family=row.candidate_family,
        fold_id=row.fold_id,
        source_identity=row.source_identity,
        rule_version=row.rule_version,
        snapshot_id=row.snapshot_id,
        feature_spec_hash=row.feature_spec_hash,
    )


def test_outcome_rejects_ambiguous_return_basis_identity():
    row = _outcome(0, split="train_fit", decision=1_000, net=1.0)
    with pytest.raises(ValueError, match="explicit supported reference basis"):
        replace(row, return_basis_id="entry-notional-bps-v1")


def test_logistic_value_model_uses_later_calibration_and_fit_only_support():
    fit = []
    calibration = []
    features = {}
    for i in range(12):
        decision = 1_000 + i * 100
        x = float(i % 2)
        side = 1 if x else -1
        net = 8.0 if side > 0 else -5.0
        feature_values = (x, float(i // 4))
        row = _outcome(
            i, split="train_fit", decision=decision, net=net,
            features=feature_values, candidate_side=side,
        )
        fit.append(row)
        features[row.event_id] = feature_values
    for i in range(12, 24):
        decision = 10_000 + i * 100
        x = float(i % 2)
        side = 1 if x else -1
        net = 7.0 if side > 0 else -4.0
        feature_values = (x, float(i // 4))
        row = _outcome(
            i, split="train_calibration", decision=decision, net=net,
            features=feature_values, candidate_side=side,
        )
        calibration.append(row)
        features[row.event_id] = feature_values

    model = fit_opportunity_model(
        fit, calibration, features,
        candidate_events={row.event_id: _candidate_event(row) for row in (*fit, *calibration)},
        feature_names=("edge", "activity"),
    )
    long_event = OpportunityEvent(
        event_id=sha256(b"prediction-long").hexdigest(),
        decision_ts_ns=99_000,
        feature_available_at_ns=98_999,
        side=1,
        family="breakout",
        fold_id="fold-a",
        source_identity="registered-train-source",
        rule_version=RULE_VERSION,
        snapshot_id=sha256(b"snapshot-long").hexdigest(),
        feature_spec_hash=FEATURE_SPEC_HASH,
    )
    short_event = OpportunityEvent(
        event_id=sha256(b"prediction-short").hexdigest(),
        decision_ts_ns=99_000,
        feature_available_at_ns=98_999,
        side=-1,
        family="breakout",
        fold_id="fold-a",
        source_identity="registered-train-source",
        rule_version=RULE_VERSION,
        snapshot_id=sha256(b"snapshot-short").hexdigest(),
        feature_spec_hash=FEATURE_SPEC_HASH,
    )
    long_estimate = model.predict(long_event, (1.0, 2.0))
    short_estimate = model.predict(short_event, (0.0, 2.0))

    assert long_estimate.probability_positive > short_estimate.probability_positive
    assert long_estimate.expected_net_increment_bps > short_estimate.expected_net_increment_bps
    assert long_estimate.global_residual_floor_proxy_bps < long_estimate.expected_net_increment_bps
    assert long_estimate.in_support
    assert not model.predict(long_event, (2.0, 2.0)).in_support
    assert long_estimate.economic_status == "SCENARIO_ONLY"
    assert long_estimate.evidence_type == "MODEL_COUNTERFACTUAL"
    assert len(long_estimate.fold_calibration_id) == 64
    assert model.fit_sample_count == 12
    assert model.calibration_sample_count == 12
    assert model.fit_effective_label_count == pytest.approx(12.)
    assert model.calibration_effective_label_count == pytest.approx(12.)
    assert model.fit_weighting_id == "AVERAGE_INVERSE_LABEL_CONCURRENCY_V1"
    assert model.fit_independent_blocks == 3
    assert model.calibration_independent_blocks == 3
    assert not model.mean.flags.writeable
    with pytest.raises(ValueError, match="source/fold/schema"):
        model.predict(replace(long_event, rule_version="different-rule"), (1.0, 2.0))


def test_fit_labels_crossing_calibration_start_are_purged():
    fit = [
        _outcome(i, split="train_fit", decision=100 + i * 100,
                 net=10 if i % 2 else -10,
                 features=(float(i % 2), float(i // 4)),
                 feature_names=("x", "regime_proxy"),
                 candidate_side=1 if i % 2 else -1,
                 end=20_000 if i == 0 else 110 + i * 100)
        for i in range(12)
    ]
    calibration = [
        _outcome(i + 20, split="train_calibration", decision=10_000 + i * 100,
                 net=5 if i % 2 else -5,
                 features=(float(i % 2), float(i // 4)),
                 feature_names=("x", "regime_proxy"),
                 candidate_side=1 if i % 2 else -1)
        for i in range(8)
    ]
    features = {
        **{row.event_id: (float(i % 2), float(i // 4)) for i, row in enumerate(fit)},
        **{row.event_id: (float(i % 2), float(i // 4))
           for i, row in enumerate(calibration)},
    }

    model = fit_opportunity_model(
        fit, calibration, features,
        candidate_events={row.event_id: _candidate_event(row) for row in (*fit, *calibration)},
        feature_names=("x", "regime_proxy"),
    )
    assert len(model.calibration_id) == 64


def test_model_rejects_overlapping_fit_or_calibration_blocks():
    fit = [
        _outcome(
            i, split="train_fit", decision=100 + i * 100,
            net=10 if i % 2 else -10,
            features=(0.0, 0.0),
            candidate_side=1 if i % 2 else -1,
        )
        for i in range(12)
    ]
    calibration = [
        _outcome(
            i + 20, split="train_calibration", decision=10_000 + i * 100,
            net=5 if i % 2 else -5,
            features=(0.0, 0.0),
            candidate_side=1 if i % 2 else -1,
        )
        for i in range(12)
    ]
    # Make adjacent fit block intervals overlap while retaining each block's
    # consistent identity. Rebuild the second block rows as one common range.
    fit[4:8] = [
        replace(row, independent_block_start_ts_ns=400,
                independent_block_end_ts_ns=900)
        for row in fit[4:8]
    ]
    rows = (*fit, *calibration)
    with pytest.raises(ValueError, match="Train-fit independent time blocks overlap"):
        fit_opportunity_model(
            fit, calibration,
            {row.event_id: (0.0, 0.0) for row in rows},
            candidate_events={row.event_id: _candidate_event(row) for row in rows},
            feature_names=("edge", "activity"),
        )


def test_model_rejects_non_train_roles_hindsight_and_identity_mismatch():
    fit = [
        _outcome(i, split="train_fit", decision=100 + i * 100,
                 net=5 if i % 2 else -5, features=(float(i % 2),),
                 feature_names=("edge",), candidate_side=1 if i % 2 else -1)
        for i in range(8)
    ]
    calibration = [
        _outcome(i + 10, split="train_calibration", decision=10_000 + i * 100,
                 net=5 if i % 2 else -5, features=(float(i % 2),),
                 feature_names=("edge",), candidate_side=1 if i % 2 else -1)
        for i in range(8)
    ]
    features = {
        row.event_id: (float(i % 2),)
        for i, row in enumerate((*fit, *calibration))
    }

    with pytest.raises(PermissionError, match="Validation/Test/Holdout"):
        replace(fit[0], split="validation")
    with pytest.raises(ValueError, match="only full-universe"):
        replace(fit[0], evidence_type="HINDSIGHT_ORACLE")
    with pytest.raises(ValueError, match="SCENARIO_ONLY"):
        replace(fit[0], economic_status="ECONOMIC_AUTHORITY_VERIFIED")
    changed = [*calibration]
    changed[0] = replace(changed[0], exit_policy_id="different-exit-v1")
    with pytest.raises(ValueError, match="exit_policy_id"):
        fit_opportunity_model(
            fit, changed, features,
            candidate_events={row.event_id: _candidate_event(row) for row in (*fit, *changed)},
            feature_names=("edge",),
        )
    candidate_events = {
        row.event_id: _candidate_event(row) for row in (*fit, *calibration)
    }
    first = fit[0]
    candidate_events[first.event_id] = replace(
        candidate_events[first.event_id], family="pullback",
    )
    with pytest.raises(ValueError, match="provenance differs"):
        fit_opportunity_model(
            fit, calibration, features, candidate_events=candidate_events,
            feature_names=("edge",),
        )


def test_censored_outcomes_are_not_zero_and_nonfinite_features_fail():
    with pytest.raises(ValueError, match="censored outcomes"):
        _outcome(99, split="train_fit", decision=100, net=None, end=None)
    fit = [
        _outcome(i, split="train_fit", decision=100 + i * 100,
                 net=5 if i % 2 else -5, features=(float(i % 2),),
                 feature_names=("edge",), candidate_side=1 if i % 2 else -1)
        for i in range(8)
    ]
    calibration = [
        _outcome(i + 10, split="train_calibration", decision=10_000 + i * 100,
                 net=5 if i % 2 else -5, features=(float(i % 2),),
                 feature_names=("edge",), candidate_side=1 if i % 2 else -1)
        for i in range(8)
    ]
    features = {
        row.event_id: (float(i % 2),)
        for i, row in enumerate((*fit, *calibration))
    }
    features[fit[0].event_id] = (float("nan"),)
    with pytest.raises(ValueError, match="non-finite"):
        fit_opportunity_model(
            fit, calibration, features,
            candidate_events={row.event_id: _candidate_event(row) for row in (*fit, *calibration)},
            feature_names=("edge",),
        )
    features[fit[0].event_id] = (17.0,)
    with pytest.raises(ValueError, match="label-bound snapshot"):
        fit_opportunity_model(
            fit, calibration, features,
            candidate_events={row.event_id: _candidate_event(row) for row in (*fit, *calibration)},
            feature_names=("edge",),
        )

    censored = _outcome(
        99, split="train_fit", decision=100, net=None,
        features=(0.0,), feature_names=("edge",), censor_ts_ns=200,
    )
    censored_fit = (*fit, censored)
    censored_features = {
        **{row.event_id: (float(i % 2),) for i, row in enumerate(fit)},
        **{row.event_id: (float(i % 2),) for i, row in enumerate(calibration)},
        censored.event_id: (1.0,),
    }
    censored_events = {
        row.event_id: _candidate_event(row) for row in (*censored_fit, *calibration)
    }
    with pytest.raises(ValueError, match="label-bound snapshot"):
        fit_opportunity_model(
            censored_fit, calibration, censored_features,
            candidate_events=censored_events, feature_names=("edge",),
        )
