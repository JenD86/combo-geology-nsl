from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.typing.trajectory import EpisodeTrajectory


@dataclass(frozen=True)
class SaturationCheck:
    tripped: bool
    reason: str | None = None
    c1_tripped: bool = False
    c2_tripped: bool = False


def _is_interweave_survey_episode(episode: EpisodeTrajectory) -> bool:
    breakdown = episode.task_breakdown or {}
    return bool(breakdown.get("interweave_bootstrap")) and not bool(
        breakdown.get("bootstrap_active")
    )


def success_floor_tripped(
    episodes: Sequence[EpisodeTrajectory],
    *,
    min_episodes: int,
    success_rate_floor: float,
) -> bool:
    if len(episodes) < max(0, int(min_episodes)):
        return False
    if not episodes:
        return False
    successes = sum(1 for episode in episodes if episode.success)
    return (successes / len(episodes)) < float(success_rate_floor)


def survey_bursts(episodes: Sequence[EpisodeTrajectory]) -> list[tuple[int, int]]:
    bursts: list[tuple[int, int]] = []
    idx = 0
    while idx < len(episodes):
        if not _is_interweave_survey_episode(episodes[idx]):
            idx += 1
            continue
        start = idx
        while idx + 1 < len(episodes) and _is_interweave_survey_episode(
            episodes[idx + 1]
        ):
            idx += 1
        bursts.append((start, idx))
        idx += 1
    return bursts


def survey_plateau_tripped(
    episodes: Sequence[EpisodeTrajectory],
    *,
    plateau_bursts: int,
    interweave_survey_remaining: int,
) -> bool:
    bursts = survey_bursts(episodes)
    if interweave_survey_remaining > 0 and bursts:
        bursts = bursts[:-1]
    required = max(2, int(plateau_bursts))
    if len(bursts) < required:
        return False
    selected = bursts[-required:]
    start = selected[0][0]
    end = selected[-1][1]
    return not any(episode.success for episode in episodes[start : end + 1])


def evaluate_saturation_conditions(
    episodes: Sequence[EpisodeTrajectory],
    *,
    enabled: bool,
    min_episodes: int,
    success_rate_floor: float,
    plateau_bursts: int,
    interweave_survey_remaining: int,
) -> SaturationCheck:
    if not enabled:
        return SaturationCheck(tripped=False)
    if len(episodes) < max(0, int(min_episodes)):
        return SaturationCheck(tripped=False)

    c1 = success_floor_tripped(
        episodes,
        min_episodes=min_episodes,
        success_rate_floor=success_rate_floor,
    )
    if c1:
        successes = sum(1 for episode in episodes if episode.success)
        rate = successes / len(episodes) if episodes else 0.0
        return SaturationCheck(
            tripped=True,
            reason=(
                f"C1 epoch success rate {rate:.3f} < "
                f"{float(success_rate_floor):.3f} (episodes={len(episodes)})"
            ),
            c1_tripped=True,
        )

    c2 = survey_plateau_tripped(
        episodes,
        plateau_bursts=plateau_bursts,
        interweave_survey_remaining=interweave_survey_remaining,
    )
    if c2:
        return SaturationCheck(
            tripped=True,
            reason="C2 consecutive zero-success interweave survey bursts",
            c2_tripped=True,
        )

    return SaturationCheck(tripped=False)
