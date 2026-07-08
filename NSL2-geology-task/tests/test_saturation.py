from __future__ import annotations

from src.task.saturation import (
    evaluate_saturation_conditions,
    success_floor_tripped,
    survey_bursts,
    survey_plateau_tripped,
)
from src.typing.trajectory import EpisodeTrajectory


def _episode(
    idx: int,
    *,
    success: bool = False,
    interweave: bool = False,
    bootstrap: bool = False,
) -> EpisodeTrajectory:
    breakdown: dict[str, object] = {}
    if interweave:
        breakdown["interweave_bootstrap"] = True
    if bootstrap:
        breakdown["bootstrap_active"] = True
    return EpisodeTrajectory(
        episode_id=f"ep-{idx}",
        generation_id=0,
        episode_index=idx,
        prompt_responses=[],
        trajectory={},
        score=1.0 if success else 0.0,
        episode_runtime_success=True,
        success=success,
        llm_turns_count=1,
        container_variation="variation",
        started_at="2026-07-07T00:00:00",
        completed_at="2026-07-07T00:00:01",
        duration_seconds=1.0,
        task_breakdown=breakdown,
    )


def test_success_floor_trips_below_threshold_after_warmup() -> None:
    episodes = [_episode(i, success=(i == 0)) for i in range(13)]

    assert success_floor_tripped(
        episodes,
        min_episodes=10,
        success_rate_floor=0.10,
    )


def test_success_floor_respects_warmup() -> None:
    episodes = [_episode(i, success=False) for i in range(9)]

    assert not success_floor_tripped(
        episodes,
        min_episodes=10,
        success_rate_floor=0.10,
    )


def test_success_floor_no_trip_at_or_above_threshold() -> None:
    episodes = [_episode(i, success=(i == 0)) for i in range(10)]

    assert not success_floor_tripped(
        episodes,
        min_episodes=10,
        success_rate_floor=0.10,
    )


def test_success_floor_scoped_to_epoch() -> None:
    prior_epoch = [_episode(i, success=True) for i in range(40)]
    current_epoch = [_episode(i + 40, success=False) for i in range(50)]

    assert not success_floor_tripped(
        prior_epoch + current_epoch,
        min_episodes=50,
        success_rate_floor=0.10,
    )
    assert success_floor_tripped(
        current_epoch,
        min_episodes=50,
        success_rate_floor=0.10,
    )


def test_survey_burst_grouping() -> None:
    episodes = [
        _episode(0),
        _episode(1, interweave=True),
        _episode(2, interweave=True),
        _episode(3),
        _episode(4, interweave=True),
    ]

    assert survey_bursts(episodes) == [(1, 2), (4, 4)]


def test_two_consecutive_zero_success_bursts_trips() -> None:
    episodes = [
        _episode(0, interweave=True),
        _episode(1),
        _episode(2),
        _episode(3, interweave=True),
    ]

    assert survey_plateau_tripped(
        episodes,
        plateau_bursts=2,
        interweave_survey_remaining=0,
    )


def test_burst_with_a_success_resets() -> None:
    episodes = [
        _episode(0, interweave=True),
        _episode(1),
        _episode(2, success=True),
        _episode(3, interweave=True),
    ]

    assert not survey_plateau_tripped(
        episodes,
        plateau_bursts=2,
        interweave_survey_remaining=0,
    )


def test_open_burst_not_counted() -> None:
    episodes = [
        _episode(0, interweave=True),
        _episode(1),
        _episode(2, interweave=True),
    ]

    assert not survey_plateau_tripped(
        episodes,
        plateau_bursts=2,
        interweave_survey_remaining=3,
    )


def test_span_includes_intervening_crossbreed_successes() -> None:
    episodes = [
        _episode(0, interweave=True),
        _episode(1),
        _episode(2, success=True),
        _episode(3),
        _episode(4, interweave=True),
    ]

    assert not survey_plateau_tripped(
        episodes,
        plateau_bursts=2,
        interweave_survey_remaining=0,
    )


def test_bootstrap_survey_excluded() -> None:
    episodes = [
        _episode(0, interweave=True, bootstrap=True),
        _episode(1),
        _episode(2, interweave=True),
    ]

    assert survey_bursts(episodes) == [(2, 2)]


def test_c2_requires_two_bursts() -> None:
    episodes = [_episode(0), _episode(1, interweave=True), _episode(2)]

    assert not survey_plateau_tripped(
        episodes,
        plateau_bursts=2,
        interweave_survey_remaining=0,
    )


def test_disabled_never_trips() -> None:
    episodes = [_episode(i, success=False, interweave=i in {0, 2}) for i in range(50)]

    result = evaluate_saturation_conditions(
        episodes,
        enabled=False,
        min_episodes=10,
        success_rate_floor=0.10,
        plateau_bursts=2,
        interweave_survey_remaining=0,
    )

    assert not result.tripped
