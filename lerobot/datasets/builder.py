"""Use LeRobot's native dataset APIs with a Hepha simulation backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hepha_lerobot.conditioning import (
    NEXT_TASK_PHASE,
    WORKSPACE_CONDITION_NAMES,
    drawer_condition_feature,
    drawer_condition_values,
    next_task_phase_feature,
    task_phase_condition_feature,
    task_phase_condition_values,
    validate_task_phase,
    workspace_condition_feature,
    workspace_condition_values,
)
from hepha_lerobot.workspaces import CNC_ACTION_NAMES
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
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
    include_task_phase: bool = True,
    include_workspace: bool = False,
    include_rewards: bool = False,
) -> LeRobotDataset:
    """Create a LeRobotDataset using only upstream feature conversion APIs."""

    action_features = backend.action_features
    if include_workspace:
        action_features = {
            name: feature_type
            for name, feature_type in action_features.items()
            if name not in CNC_ACTION_NAMES
        }
    feature_groups = [
        hw_to_dataset_features(action_features, ACTION, use_video=use_videos),
        hw_to_dataset_features(backend.observation_features, OBS_STR, use_video=use_videos),
        drawer_condition_feature(),
    ]
    if include_workspace:
        feature_groups.append(workspace_condition_feature())
    if include_task_phase:
        feature_groups.extend([task_phase_condition_feature(), next_task_phase_feature()])
    if include_rewards:
        feature_groups.append(
            {
                REWARD: {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["reward"],
                },
                DONE: {"dtype": "bool", "shape": (1,), "names": ["done"]},
            }
        )
    features = combine_feature_dicts(*feature_groups)
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
    current_task_phase: int | None,
    next_task_phase: int | None,
    workspace: str | None = None,
    reward: float | None = None,
    done: bool | None = None,
) -> None:
    include_task_phase = NEXT_TASK_PHASE in dataset.features
    environment_names = dataset.features.get("observation.environment_state", {}).get("names", [])
    include_workspace = all(name in environment_names for name in WORKSPACE_CONDITION_NAMES)
    observation = {
        **observation,
        **drawer_condition_values(drawer_index),
    }
    if include_workspace:
        if workspace is None:
            raise ValueError("Workspace-aware datasets require workspace A or B")
        observation.update(workspace_condition_values(workspace))
    if include_task_phase:
        if current_task_phase is None or next_task_phase is None:
            raise ValueError("Phase-aware datasets require current and next task phases")
        current_task_phase = validate_task_phase(current_task_phase)
        next_task_phase = validate_task_phase(next_task_phase)
        observation.update(task_phase_condition_values(current_task_phase))
    observation_frame = build_dataset_frame(dataset.features, observation, prefix=OBS_STR)
    action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
    frame = {**observation_frame, **action_frame, "task": task}
    if include_task_phase:
        frame[NEXT_TASK_PHASE] = np.asarray([next_task_phase - 1], dtype=np.int64)
    if REWARD in dataset.features:
        if reward is None or done is None:
            raise ValueError("Reward datasets require reward and done for every frame")
        frame[REWARD] = np.asarray([reward], dtype=np.float32)
        frame[DONE] = np.asarray([done], dtype=np.bool_)
    dataset.add_frame(frame)
