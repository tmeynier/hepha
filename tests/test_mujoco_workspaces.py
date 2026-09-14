from __future__ import annotations

import mujoco
import numpy as np
from hepha_lerobot.workspaces import TaskPhase, Workspace, workspace_for_phase

from simulation import SimulationConfig
from simulation.backends.mujoco import MujocoBackend
from simulation.backends.mujoco.ik import (
    COMMON_JOINTS,
    WORKSPACE_TRANSIT_BIN_CLEARANCE_M,
    MujocoIKController,
    _controlled_joints,
    _minimum_fingertip_z,
    _storage_bin_top_z,
    drawer_handle_pose,
)
from simulation.backends.mujoco.workspaces import (
    CNC_X_JOINT,
    CNC_Y_JOINT,
    CNC_Z_JOINT,
    DRAWER_CENTER_CNC_Y_M,
    WORKSPACE_CNC_Y_M,
    WORKSPACE_CNC_Y_OFFSET_M,
    resolve_mujoco_workspace,
    set_workspace_qpos,
)


def test_task_phases_follow_a_b_a_b_a() -> None:
    assert [workspace_for_phase(phase) for phase in TaskPhase] == [
        Workspace.STORAGE,
        Workspace.DRAWERS,
        Workspace.STORAGE,
        Workspace.DRAWERS,
        Workspace.STORAGE,
    ]


def test_mujoco_workspace_targets_are_discrete_and_inside_joint_limits() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        a = resolve_mujoco_workspace(backend.model, Workspace.STORAGE)
        all_b = [
            resolve_mujoco_workspace(backend.model, Workspace.DRAWERS, drawer=drawer)
            for drawer in range(1, 10)
        ]

        assert set(a) == set(COMMON_JOINTS)
        assert len({tuple(target.values()) for target in all_b}) == 1
        for targets in [a, *all_b]:
            for name, value in targets.items():
                joint_id = backend.model.joint(name).id
                assert backend.model.jnt_range[joint_id, 0] <= value
                assert value <= backend.model.jnt_range[joint_id, 1]

        assert np.isclose(a[CNC_X_JOINT], backend.model.joint(CNC_X_JOINT).range[1])
        assert np.isclose(a[CNC_Y_JOINT], WORKSPACE_CNC_Y_M)
        assert np.isclose(a[CNC_Z_JOINT], backend.model.joint(CNC_Z_JOINT).range[1])

        for target in all_b:
            assert np.isclose(target[CNC_X_JOINT], backend.model.joint(CNC_X_JOINT).range[0])
            assert np.isclose(target[CNC_Y_JOINT], a[CNC_Y_JOINT])
            assert np.isclose(target[CNC_Z_JOINT], a[CNC_Z_JOINT])


def test_workspaces_center_world_x_on_the_middle_drawer() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        mujoco.mj_resetData(backend.model, backend.data)
        set_workspace_qpos(backend.model, backend.data, Workspace.STORAGE)
        mujoco.mj_forward(backend.model, backend.data)
        middle_handle, _ = drawer_handle_pose(backend.model, backend.data, 5)

        # CNC-Y travels along world X, so equal values here prove that the
        # carriage is laterally centered on the middle drawer column.
        assert WORKSPACE_CNC_Y_OFFSET_M == 0.0
        assert np.isclose(backend.data.body("head_link").xpos[0], middle_handle[0])
        assert np.isclose(
            WORKSPACE_CNC_Y_M,
            DRAWER_CENTER_CNC_Y_M + WORKSPACE_CNC_Y_OFFSET_M,
        )


def test_workspace_b_is_fixed_for_every_drawer() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        poses = []
        for drawer in range(1, 10):
            mujoco.mj_resetData(backend.model, backend.data)
            set_workspace_qpos(
                backend.model,
                backend.data,
                Workspace.DRAWERS,
                drawer=drawer,
            )
            mujoco.mj_forward(backend.model, backend.data)
            poses.append(backend.data.body("head_link").xpos.copy())

        for pose in poses[1:]:
            np.testing.assert_allclose(pose, poses[0], atol=1e-9)


def test_nominal_pose_clears_storage_bin_during_every_workspace_transit() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        controller = MujocoIKController(backend, seed=0, trajectory_randomization_scale=0.0)
        controller.reset(episode_seed=0)
        required_z = (
            _storage_bin_top_z(backend.model, backend.data)
            + WORKSPACE_TRANSIT_BIN_CLEARANCE_M
        )

        assert np.isclose(required_z, 0.10)
        assert _minimum_fingertip_z(backend.model, backend.data) >= required_z
        assert controller._workspace_transit_is_clear(Workspace.STORAGE)
        assert controller._workspace_transit_is_clear(Workspace.DRAWERS)

        planning_data = mujoco.MjData(backend.model)
        planning_data.qpos[:] = backend.data.qpos
        for joint_name, target in controller._transit_targets().items():
            planning_data.joint(joint_name).qpos[0] = target
        for side in ("l", "r"):
            planning_data.joint(f"hand_{side}_link_hand_{side}_finger_{side}_joint").qpos[0] = 1.0
        mujoco.mj_forward(backend.model, planning_data)
        assert _minimum_fingertip_z(backend.model, planning_data) >= required_z


def test_low_fingertip_aborts_before_workspace_motion(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        controller = MujocoIKController(
            backend,
            seed=0,
            trajectory_randomization_scale=0.0,
            early_failures=False,
        )
        controller.reset(episode_seed=0)
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._minimum_fingertip_z",
            lambda _model, _data: 0.05,
        )
        controller.phase = "move_to_b_for_open"

        controller.action(0.0)

        assert controller.done
        assert controller.motion is None
        assert "early failure after workspace transit" in controller.status


def test_ik_solver_default_chain_excludes_all_cnc_joints() -> None:
    assert set(_controlled_joints("l")).isdisjoint(COMMON_JOINTS)
    assert set(_controlled_joints("r")).isdisjoint(COMMON_JOINTS)


def test_controller_starts_at_exact_workspace_a() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=2)
        controller.reset(episode_seed=3)
        expected = resolve_mujoco_workspace(backend.model, Workspace.STORAGE)

        assert controller.recording_workspace == "A"
        for name, value in expected.items():
            assert np.isclose(backend.data.joint(name).qpos[0], value)
