"""Shared initialization for Hepha's MuJoCo cube-and-drawer episodes."""

from __future__ import annotations

import mujoco
import numpy as np
from hepha_lerobot.workspaces import Workspace

from .backend import ACTUATOR_NAMES, MujocoBackend
from .workspaces import set_workspace_qpos

CUBE_SPAWN_RADIUS_M = 0.10
FINGER_CLOSED = 0.0

# User-selected nominal posture. The right arm follows the MuJoCo model's
# mirror convention: shoulder, wrist, and hand change sign; forearm and arm do
# not. Fingers are controlled separately by each manipulation phase.
WORKING_ARM_POSE_RADIANS = {
    "shoulder_l": 0.220,
    "forearm_l": 0.346,
    "arm_l": -0.550,
    "wrist_l": -0.518,
    "hand_l": -1.570,
    "shoulder_r": -0.220,
    "forearm_r": 0.346,
    "arm_r": -0.550,
    "wrist_r": 0.518,
    "hand_r": 1.570,
}

IK_TARGET_MARKERS = (
    "hand_tip_marker",
    "drawer_target_marker",
    "above_drawer_target_marker",
    "drawer_close_target_marker",
)

IK_ACTIVE_FRAME_MARKERS = (
    "active_ik_source_marker",
    "active_ik_target_marker",
)

IK_DEBUG_MARKERS = (*IK_TARGET_MARKERS, *IK_ACTIVE_FRAME_MARKERS)


def _named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return object_id


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    return _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _set_working_arm_pose(backend: MujocoBackend) -> None:
    model, data = backend.model, backend.data
    for actuator_name, angle_radians in WORKING_ARM_POSE_RADIANS.items():
        action_index = ACTUATOR_NAMES.index(actuator_name)
        actuator_id = int(backend.actuator_ids[action_index])
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        data.qpos[int(model.jnt_qposadr[joint_id])] = angle_radians
        data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0


def randomize_cube(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
) -> None:
    """Sample the cube uniformly by area in the storage-bin spawn circle."""

    joint_id = _joint_id(model, "cube_link_free_joint")
    qpos_id = int(model.jnt_qposadr[joint_id])
    storage_center_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "storage_bin_center_marker")
    data.qpos[qpos_id : qpos_id + 3] = data.xpos[storage_center_id]
    radius = CUBE_SPAWN_RADIUS_M * np.sqrt(rng.uniform())
    angle = rng.uniform(-np.pi, np.pi)
    data.qpos[qpos_id] += radius * np.cos(angle)
    data.qpos[qpos_id + 1] += radius * np.sin(angle)
    yaw = rng.uniform(-np.pi, np.pi)
    data.qpos[qpos_id + 3 : qpos_id + 7] = (
        np.cos(yaw / 2),
        0.0,
        0.0,
        np.sin(yaw / 2),
    )
    dof_id = int(model.jnt_dofadr[joint_id])
    data.qvel[dof_id : dof_id + 6] = 0.0


def cube_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    cube_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom")
    return data.geom_xpos[cube_id].copy()


def cube_spawn_quadrant(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    storage_center_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "storage_bin_center_marker")
    storage_rotation = data.xmat[storage_center_id].reshape(3, 3)
    offset = storage_rotation.T @ (cube_position(model, data) - data.xpos[storage_center_id])
    vertical = "upper" if offset[1] >= 0.0 else "bottom"
    horizontal = "left" if offset[0] < 0.0 else "right"
    return f"{vertical}_{horizontal}"


def initialize_task_episode(
    backend: MujocoBackend,
    *,
    seed: int,
) -> np.random.Generator:
    """Reproduce the exact state from which IK demonstrations are recorded.

    The returned generator has already produced the cube position and yaw. The IK
    recorder deliberately continues with this same generator when choosing a
    drawer, preserving its historical seed-to-episode mapping.
    """

    backend.reset(seed=seed)
    model = backend.model
    data = backend.data

    # Every demonstration starts at the same discrete storage workspace A.
    mujoco.mj_resetData(model, data)
    set_workspace_qpos(model, data, Workspace.STORAGE)
    _set_working_arm_pose(backend)

    rng = np.random.default_rng(seed)
    mujoco.mj_forward(model, data)
    randomize_cube(model, data, rng)

    # Position controls must initially hold the MJCF joint state. Both grippers
    # are closed in the first recorded frame; the learned initialization motion
    # opens them before the first task IK in the demonstrations.
    for actuator_id in backend.actuator_ids:
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_id = int(model.jnt_qposadr[joint_id])
        data.ctrl[actuator_id] = np.clip(data.qpos[qpos_id], *model.actuator_ctrlrange[actuator_id])
    for side in ("l", "r"):
        joint_id = _joint_id(model, f"hand_{side}_link_hand_{side}_finger_{side}_joint")
        data.qpos[int(model.jnt_qposadr[joint_id])] = FINGER_CLOSED
        action_index = ACTUATOR_NAMES.index(f"finger_{side}")
        data.ctrl[backend.actuator_ids[action_index]] = FINGER_CLOSED

    # IK targets are debug-only annotations and must not be visible before a
    # controller computes them. Policy rollout never computes them at all.
    for marker_body in IK_DEBUG_MARKERS:
        body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, marker_body)
        model.body_pos[body_id] = (0.0, 0.0, -10.0)
        model.body_quat[body_id] = (1.0, 0.0, 0.0, 0.0)

    mujoco.mj_forward(model, data)
    mujoco.mj_step(
        model,
        data,
        nstep=max(1, round(0.5 / model.opt.timestep)),
    )
    # Settling the free cube can slightly deflect the position-controlled CNC.
    # Snap the robot back to its exact categorical workspace and working pose
    # before frame zero.
    set_workspace_qpos(model, data, Workspace.STORAGE)
    _set_working_arm_pose(backend)
    cnc_joints = (
        "base_link_base_cnc_x_joint",
        "cnc_x_link_cnc_x_cnc_y_joint",
        "cnc_y_link_cnc_y_head_joint",
    )
    for name in cnc_joints:
        joint_id = _joint_id(model, name)
        data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0
    mujoco.mj_forward(model, data)
    data.time = 0.0
    backend.sync_viewer()
    return rng
