"""Use LeRobot's native dataset APIs with a Hepha simulation backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hepha_lerobot.conditioning import (
    NEXT_TASK_PHASE,
    drawer_condition_feature,
    drawer_condition_values,
    next_task_phase_feature,
    task_phase_condition_feature,
    task_phase_condition_values,
    validate_task_phase,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import (
    build_dataset_frame,
    combine_feature_dicts,
    hw_to_dataset_features,
)

from lerobot.datasets import LeRobotDataset
from simulation import SimulationBackend


def create_dataset(
    *,
    backend: SimulationBackend,
    repo_id: str,
    root: Path,
    fps: int,
    use_videos: bool,
    image_writer_processes: int = 0,
    image_writer_threads: int = 2,
    video_encoding_batch_size: int = 1,
    streaming_encoding: bool = False,
    encoder_threads: int | None = 2,
) -> LeRobotDataset:
    """Create a LeRobotDataset using only upstream feature conversion APIs."""

    features = combine_feature_dicts(
        hw_to_dataset_features(
            backend.action_features, ACTION, use_video=use_videos
        ),
        hw_to_dataset_features(
            backend.observation_features, OBS_STR, use_video=use_videos
        ),
        drawer_condition_feature(),
        task_phase_condition_feature(),
        next_task_phase_feature(),
    )
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type=f"hepha_{backend.name}",
        features=features,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=video_encoding_batch_size,
        streaming_encoding=streaming_encoding,
        encoder_threads=encoder_threads,
    )


def add_robot_frame(
    dataset: LeRobotDataset,
    *,
    observation: dict,
    action: dict,
    task: str,
    drawer_index: int,
    current_task_phase: int,
    next_task_phase: int,
) -> None:
    current_task_phase = validate_task_phase(current_task_phase)
    next_task_phase = validate_task_phase(next_task_phase)
    observation = {
        **observation,
        **drawer_condition_values(drawer_index),
        **task_phase_condition_values(current_task_phase),
    }
    observation_frame = build_dataset_frame(
        dataset.features, observation, prefix=OBS_STR
    )
    action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
    dataset.add_frame(
        {
            **observation_frame,
            **action_frame,
            NEXT_TASK_PHASE: np.asarray([next_task_phase - 1], dtype=np.int64),
            "task": task,
        }
    )
