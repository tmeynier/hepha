from __future__ import annotations

import numpy as np

from simulation import SimulationConfig
from simulation.backends.mujoco import ACTUATOR_NAMES, MujocoBackend
from simulation.backends.mujoco.episode import (
    cube_position,
    initialize_task_episode,
)
from simulation.backends.mujoco.ik import MujocoIKController


def test_policy_and_recording_initialization_share_seeded_state() -> None:
    config = SimulationConfig(render=False, fps=30)
    with MujocoBackend(config) as rollout_backend:
        drawer_rng = initialize_task_episode(rollout_backend, seed=15)
        rollout_qpos = rollout_backend.data.qpos.copy()
        rollout_ctrl = rollout_backend.data.ctrl.copy()
        rollout_cube = cube_position(rollout_backend.model, rollout_backend.data)
        expected_drawer = int(drawer_rng.integers(1, 10))

    with MujocoBackend(config) as recording_backend:
        controller = MujocoIKController(recording_backend, seed=15)
        controller.reset(episode_seed=0)

        np.testing.assert_allclose(recording_backend.data.qpos, rollout_qpos)
        np.testing.assert_allclose(recording_backend.data.ctrl, rollout_ctrl)
        np.testing.assert_allclose(
            cube_position(recording_backend.model, recording_backend.data),
            rollout_cube,
        )
        assert controller.drawer_index == expected_drawer
        assert recording_backend.data.time == 0.0
        for finger in ("finger_l", "finger_r"):
            actuator_index = ACTUATOR_NAMES.index(finger)
            assert recording_backend.data.ctrl[
                recording_backend.actuator_ids[actuator_index]
            ] == 0.0
