from types import SimpleNamespace

import pytest

from rles_route_a.integration.live_dagger_projection import LiveNautilusGymProjection


@pytest.mark.parametrize(
    ("disposition", "expected"),
    (("SUBMIT", "PROJECTED"), ("NOOP", "NOOP"), ("WAIT", "WAIT"), ("CANCEL", "CANCEL")),
)
def test_live_projection_preserves_supported_host_dispositions(observed, disposition, expected):
    observation = observed()
    recipe = "a" * 64
    env = SimpleNamespace(
        rles_project_target=lambda requested: {
            "decision_ts_ns": observation.decision_ts_ns,
            "execution_recipe": recipe,
            "requested_exposure": requested,
            "target": 0.25,
            "disposition": disposition,
        }
    )
    projection = LiveNautilusGymProjection(env, execution_recipe=recipe)

    result = projection.project_native_target(observation, 0.5)

    assert result.target == pytest.approx(0.25)
    assert result.host_execution_recipe == recipe
    assert result.disposition == expected


@pytest.mark.parametrize("disposition", (None, 7, "UNKNOWN"))
def test_live_projection_rejects_missing_or_unknown_disposition(observed, disposition):
    observation = observed()
    recipe = "b" * 64
    env = SimpleNamespace(
        rles_project_target=lambda requested: {
            "decision_ts_ns": observation.decision_ts_ns,
            "execution_recipe": recipe,
            "requested_exposure": requested,
            "target": 0.25,
            "disposition": disposition,
        }
    )
    projection = LiveNautilusGymProjection(env, execution_recipe=recipe)

    with pytest.raises(ValueError, match="supported disposition"):
        projection.project_native_target(observation, 0.5)
