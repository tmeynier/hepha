"""Shared initialization for Hepha's MuJoCo cube-and-drawer episodes."""

from __future__ import annotations

import mujoco
import numpy as np

from .backend import ACTUATOR_NAMES, MujocoBackend

CUBE_SPAWN_RADIUS_M = 0.10
FINGER_CLOSED = 0.0

IK_TARGET_MARKERS = (
    "hand_tip_marker",
    "drawer_target_marker",
    "above_drawer_target_marker",
    "drawer_close_target_marker",
)


def _named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return object_id


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    return _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def randomize_cube(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
) -> None:
    """Sample the cube uniformly by area in the storage-bin spawn circle."""

    joint_id = _joint_id(model, "cube_link_free_joint")
    qpos_id = int(model.jnt_qposadr[joint_id])
    storage_center_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "storage_bin_center_marker"
    )
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
    cube_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom"
    )
    return data.geom_xpos[cube_id].copy()


def cube_spawn_quadrant(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    storage_center_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "storage_bin_center_marker"
    )
    storage_rotation = data.xmat[storage_center_id].reshape(3, 3)
    offset = storage_rotation.T @ (
        cube_position(model, data) - data.xpos[storage_center_id]
    )
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

    # Demonstrations start from the MJCF pose, except that the longitudinal CNC
    # axis starts at its midpoint. This differs from the generic backend home pose.
    mujoco.mj_resetData(model, data)
    base_id = _joint_id(model, "base_link_base_cnc_x_joint")
    data.qpos[int(model.jnt_qposadr[base_id])] = np.mean(model.jnt_range[base_id])

    rng = np.random.default_rng(seed)
    mujoco.mj_forward(model, data)
    randomize_cube(model, data, rng)

    # Position controls must initially hold the MJCF joint state. Both grippers
    # are closed in the first recorded frame; the learned initialization motion
    # opens them before the first task IK in the demonstrations.
    for actuator_id in backend.actuator_ids:
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_id = int(model.jnt_qposadr[joint_id])
        data.ctrl[actuator_id] = np.clip(
            data.qpos[qpos_id], *model.actuator_ctrlrange[actuator_id]
        )
    for side in ("l", "r"):
        joint_id = _joint_id(model, f"hand_{side}_link_hand_{side}_finger_{side}_joint")
        data.qpos[int(model.jnt_qposadr[joint_id])] = FINGER_CLOSED
        action_index = ACTUATOR_NAMES.index(f"finger_{side}")
        data.ctrl[backend.actuator_ids[action_index]] = FINGER_CLOSED

    # IK targets are debug-only annotations and must not be visible before a
    # controller computes them. Policy rollout never computes them at all.
    for marker_body in IK_TARGET_MARKERS:
        body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, marker_body)
        model.body_pos[body_id] = (0.0, 0.0, -10.0)
        model.body_quat[body_id] = (1.0, 0.0, 0.0, 0.0)

    mujoco.mj_forward(model, data)
    mujoco.mj_step(
        model,
        data,
        nstep=max(1, round(0.5 / model.opt.timestep)),
    )
    data.time = 0.0
    backend.sync_viewer()
    return rng
