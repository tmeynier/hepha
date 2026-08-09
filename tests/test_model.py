from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from hepha_lerobot.recording.controllers import available_controllers

from simulation import SimulationConfig, available_backends, create_backend
from simulation.backends.mujoco import ACTUATOR_NAMES, MujocoBackend
from simulation.backends.mujoco.ik import (
    CollisionAwareIK,
    MujocoIKController,
    hand_pose,
)


def test_mujoco_model_matches_control_contract() -> None:
    assert "mujoco" in available_backends()
    with create_backend("mujoco", SimulationConfig(render=False)) as backend:
        assert isinstance(backend, MujocoBackend)
        assert backend.model.nu == len(ACTUATOR_NAMES)
        assert backend.joint_positions().shape == (len(ACTUATOR_NAMES),)
        assert tuple(backend.action_features) == tuple(
            f"{name}.pos" for name in ACTUATOR_NAMES
        )
        assert backend.observation_features["head_camera"] == (256, 256, 3)


def test_collision_aware_ik_uses_the_full_side_chain(monkeypatch) -> None:
    def use_middle_of_bounds(_objective, bounds, **_kwargs):
        return SimpleNamespace(x=np.mean(np.asarray(bounds), axis=1))

    monkeypatch.setattr(
        "simulation.backends.mujoco.ik.differential_evolution",
        use_middle_of_bounds,
    )
    monkeypatch.setattr(
        "simulation.backends.mujoco.ik.minimize",
        lambda _objective, initial, **_kwargs: SimpleNamespace(x=initial),
    )

    with MujocoBackend(SimulationConfig(render=False)) as backend:
        target, rotation = hand_pose(backend.model, backend.data, "r")
        solution = CollisionAwareIK(backend).solve(
            side="r",
            target=target,
            target_rotation=rotation,
            seed=0,
        )

        assert len(solution.joint_names) == 8
        assert solution.joint_values.shape == (8,)
        assert solution.error_m < 1e-9


def test_default_ik_controller_starts_physical_task_with_closed_fingers() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4)
        controller.reset(episode_seed=7)
        first_action = controller.action(0.0)

        assert controller.phase == "pre_cube_view"
        assert 1 <= controller.drawer_index <= 9
        assert first_action.shape == (len(ACTUATOR_NAMES),)
        assert first_action[ACTUATOR_NAMES.index("finger_l")] == 0.0
        assert first_action[ACTUATOR_NAMES.index("finger_r")] == 0.0
        assert not controller.done


def test_mujoco_exposes_only_the_physical_ik_controller() -> None:
    assert available_controllers("mujoco") == ("ik",)
