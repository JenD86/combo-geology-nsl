from __future__ import annotations

import json
from pathlib import Path

from src.task.base import SaturationDecision
from src.typing.trajectory import EpisodeTrajectory, GenerationData
from tasks.feature_hypothesis_australia import FeatureHypothesisAustraliaTask


def _task(tmp_path: Path, **config: object) -> FeatureHypothesisAustraliaTask:
    return FeatureHypothesisAustraliaTask(
        {
            "store_dir": str(tmp_path / "store_root"),
            "kg_dir": str(tmp_path / "kg_root"),
            **config,
        }
    )


def _episode(
    idx: int,
    *,
    success: bool = False,
    row_count: int = 0,
    interweave: bool = False,
    admitted: bool = False,
) -> EpisodeTrajectory:
    breakdown = {"interweave_bootstrap": True} if interweave else {}
    if admitted:
        breakdown["kg_admission_passed"] = True
    return EpisodeTrajectory(
        episode_id=f"ep-{idx}",
        generation_id=0,
        episode_index=idx,
        prompt_responses=[{"prompt": "p", "raw_response": "r"} for _ in range(row_count)],
        raw_training_rows=[{"prompt": "p", "raw_response": "r"} for _ in range(row_count)],
        trajectory={},
        score=1.0 if success else 0.0,
        episode_runtime_success=True,
        success=success,
        llm_turns_count=1,
        container_variation="coe_fairbairn",
        started_at="2026-07-07T00:00:00",
        completed_at="2026-07-07T00:00:01",
        duration_seconds=1.0,
        task_breakdown=breakdown,
    )


def _generation(episodes: list[EpisodeTrajectory]) -> GenerationData:
    data = GenerationData(generation_id=0)
    for episode in episodes:
        data.add_episode(episode)
    return data


def test_task_parses_saturation_config(tmp_path: Path) -> None:
    task = _task(
        tmp_path,
        saturation_enabled=True,
        saturation_min_episodes="7",
        saturation_success_rate_floor="0.25",
        saturation_plateau_bursts="3",
        saturation_max_rollovers="4",
        saturation_epoch_progress_admissions="2",
        saturation_archive_keep_last="5",
    )

    assert task._saturation_enabled is True
    assert task._saturation_min_episodes == 7
    assert task._saturation_success_rate_floor == 0.25
    assert task._saturation_plateau_bursts == 3
    assert task._saturation_max_rollovers == 4
    assert task._saturation_epoch_progress_admissions == 2
    assert task._saturation_archive_keep_last == 5


def test_task_saturation_defaults_disabled(tmp_path: Path) -> None:
    task = _task(tmp_path)
    data = _generation([_episode(i, success=False) for i in range(50)])

    assert task.saturation_enabled() is False
    assert task.evaluate_saturation(data).decision is SaturationDecision.CONTINUE


def test_evaluate_saturation_detects_closed_interweave_plateau(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        saturation_enabled=True,
        saturation_min_episodes=1,
        saturation_success_rate_floor=0.0,
    )
    kg_dir = tmp_path / "kg_root" / "coe_fairbairn"
    (kg_dir / "interweave_state.json").write_text(
        json.dumps({"interweave_survey_remaining": 0}),
        encoding="utf-8",
    )
    data = _generation(
        [
            _episode(0, interweave=True),
            _episode(1),
            _episode(2),
            _episode(3, interweave=True),
        ]
    )

    outcome = task.evaluate_saturation(data)

    assert outcome.decision is SaturationDecision.ROLLOVER
    assert outcome.reason is not None
    assert "C2" in outcome.reason


def test_open_burst_uses_coe_fairbairn_interweave_state(tmp_path: Path) -> None:
    task = _task(
        tmp_path,
        saturation_enabled=True,
        saturation_min_episodes=1,
        saturation_success_rate_floor=0.0,
    )
    kg_dir = tmp_path / "kg_root" / "coe_fairbairn"
    (kg_dir / "interweave_state.json").write_text(
        json.dumps({"interweave_survey_remaining": 1}),
        encoding="utf-8",
    )
    data = _generation(
        [
            _episode(0, interweave=True),
            _episode(1),
            _episode(2),
            _episode(3, interweave=True),
        ]
    )

    outcome = task.evaluate_saturation(data)

    assert outcome.decision is SaturationDecision.CONTINUE


def test_evaluate_saturation_stop_after_max_rollovers(tmp_path: Path) -> None:
    task = _task(
        tmp_path,
        saturation_enabled=True,
        saturation_min_episodes=1,
        saturation_success_rate_floor=0.10,
        saturation_max_rollovers=1,
    )
    data = _generation([_episode(i, success=False) for i in range(2)])

    outcome = task.evaluate_saturation(data)

    assert outcome.decision is SaturationDecision.STOP
    assert outcome.reason == "saturation_exhausted"


def test_rollover_preserves_training_rows_and_archives_coe_fairbairn_kg(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, saturation_enabled=True)
    kg_dir = tmp_path / "kg_root" / "coe_fairbairn"
    store_dir = tmp_path / "store_root" / "coe_fairbairn"
    (kg_dir / "experiments.jsonl").write_text("{}\n", encoding="utf-8")
    (store_dir / "admitted" / "layers" / "layer.bin").write_text(
        "layer",
        encoding="utf-8",
    )
    data = _generation([_episode(0, success=True, row_count=3)])

    task.rollover_knowledge_graph(
        "C2 plateau",
        data,
        rolled_at="2026-07-07T00:00:00",
    )

    assert data.training_row_count == 3
    assert kg_dir.exists()
    assert store_dir.exists()
    assert not (kg_dir / "experiments.jsonl").exists()
    assert not (store_dir / "admitted" / "layers" / "layer.bin").exists()
    archives = list(task._saturation_archive_dir.glob("*"))
    assert len(archives) == 1
    assert (archives[0] / "knowledge_coe_fairbairn" / "experiments.jsonl").exists()
    assert (
        archives[0] / "store_coe_fairbairn" / "admitted" / "layers" / "layer.bin"
    ).exists()


def test_saturation_checkpoint_state_round_trips(tmp_path: Path) -> None:
    task = _task(tmp_path, saturation_enabled=True)
    data = _generation([_episode(i, success=False) for i in range(3)])

    task.rollover_knowledge_graph(
        "C1 old epoch",
        data,
        rolled_at="2026-07-07T00:00:00",
    )
    state = task.generation_checkpoint_state()
    resumed = _task(tmp_path / "resumed", saturation_enabled=True)

    resumed.load_generation_checkpoint_state(state)

    assert resumed._saturation_epoch_start_count == 3
    assert resumed._saturation_consecutive_unproductive_rollovers == 1
