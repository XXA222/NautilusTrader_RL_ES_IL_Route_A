from dataclasses import replace

import pytest

from rles_route_a.teacher_research.candidate_events import CausalOpportunityGenerator
from rles_route_a.teachers.causal_position import CausalPositionTeacher


def _candidate(observed, side, *, recovering=True):
    obs = observed(hourly=side * .01, daily=side * .02)
    market = {**obs.market, "zscore_1m": -side * 2., "efficiency_15m": -side * .5,
              "return_15m": -side * .005,
              "return_5m": side * .002 * (1 if recovering else -1)}
    return replace(obs, market=market, feature_available_at_ns={k: obs.decision_ts_ns - 1 for k in market})


def _generator():
    return CausalOpportunityGenerator(
        rule_version="route_a_candidates_v3_recovery", require_recovery_confirmation=True,
    )


@pytest.mark.parametrize("side", [-1, 1])
def test_pullback_needs_observed_recovery_and_is_direction_symmetric(observed, side):
    generator = _generator()
    assert generator.candidates(_candidate(observed, side, recovering=False)) == ()
    obs = _candidate(observed, side)
    events = generator.candidates(obs)
    assert len(events) == 1
    assert events[0].side == side
    assert events[0].family == "pullback"
    teacher = CausalPositionTeacher(generator=generator, expert_id="p0_recovery_research_v3")
    assert teacher.advise(obs).requested_exposure == side * .3


def test_recovery_missing_or_future_feature_fails_closed(observed):
    generator = _generator()
    obs = _candidate(observed, 1)
    with pytest.raises((KeyError, ValueError)):
        generator.candidates(observed())
    with pytest.raises(ValueError):
        replace(obs, feature_available_at_ns={**obs.feature_available_at_ns,
                                             "return_5m": obs.decision_ts_ns + 1})
    changed = replace(obs, feature_available_at_ns={**obs.feature_available_at_ns,
                                                   "return_5m": obs.decision_ts_ns})
    assert generator.candidates(changed)[0].feature_available_at_ns == obs.decision_ts_ns


def test_new_mechanism_cannot_masquerade_as_frozen_v2():
    with pytest.raises(ValueError, match="versioned"):
        CausalOpportunityGenerator(require_recovery_confirmation=True)


def test_prior_recipe_and_new_mechanism_have_distinct_event_ids(observed):
    # The short pullback previously passed even while price kept rising.
    obs = _candidate(observed, -1)
    old = CausalOpportunityGenerator().candidates(obs)
    new = _generator().candidates(obs)
    assert len(old) == len(new) == 1
    assert old[0].event_id != new[0].event_id


def test_entry_cadence_matches_labels_but_does_not_delay_exit(observed):
    teacher = CausalPositionTeacher(generator=_generator(), entry_cadence_minutes=5)
    obs = _candidate(observed, 1)
    def at(minute):
        ts = minute * 60_000_000_000 + 1_000_000
        return replace(obs, decision_ts_ns=ts,
                       feature_available_at_ns={key: ts for key in obs.market})
    assert teacher.advise(at(10)).intent == "OPEN_LONG"
    assert teacher.advise(at(11)).reason_codes == ("ENTRY_CADENCE_WAIT",)
    held = at(11)
    held = replace(held, account={**held.account, "net_exposure": .3},
                   market={**held.market, "trend_1h": -.01, "trend_1d": -.01})
    assert teacher.advise(held).intent == "EXIT"
