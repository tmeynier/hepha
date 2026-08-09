"""Physical scripted demonstration for the complete cube-and-drawer task.

The controller is intentionally limited to MuJoCo-specific task generation.  Dataset
serialization and episode management remain in :mod:`hepha_lerobot.recording`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
from scipy.optimize import differential_evolution, minimize

from .backend import ACTUATOR_NAMES, MujocoBackend

Side = Literal["l", "r"]

COMMON_JOINTS = (
    "base_link_base_cnc_x_joint",
    "cnc_x_link_cnc_x_cnc_y_joint",
    "cnc_y_link_cnc_y_head_joint",
)
SIDE_JOINTS: dict[Side, tuple[str, ...]] = {
    "l": (
        "head_link_head_shoulder_l_joint",
        "shoulder_l_link_shoulder_l_forearm_l_joint",
        "forearm_l_link_forearm_l_arm_l_joint",
        "arm_l_link_arm_l_wrist_l_joint",
        "wrist_l_link_wrist_l_hand_l_joint",
    ),
    "r": (
        "head_link_head_shoulder_r_joint",
        "shoulder_r_link_shoulder_r_forearm_r_joint",
        "forearm_r_link_forearm_r_arm_r_joint",
        "arm_r_link_arm_r_wrist_r_joint",
        "wrist_r_link_wrist_r_hand_r_joint",
    ),
}
FINGER_JOINTS: dict[Side, str] = {
    "l": "hand_l_link_hand_l_finger_l_joint",
    "r": "hand_r_link_hand_r_finger_r_joint",
}
ROBOT_JOINTS = (*COMMON_JOINTS, *SIDE_JOINTS["l"], *SIDE_JOINTS["r"])

FINGER_CLOSED = 0.0
FINGER_OPEN = 1.0
CUBE_GRASP = 0.28
DRAWER_GRASP = 0.10
CONTACT_MARGIN_M = 0.005
GRASP_CONTACT_TOLERANCE_M = 0.012
MAX_CUBE_FINGER_PENETRATION_M = 0.006


def _named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return object_id


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    return _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    return float(data.qpos[model.jnt_qposadr[_joint_id(model, name)]])


def _controlled_joints(side: Side) -> tuple[str, ...]:
    return (*COMMON_JOINTS, *SIDE_JOINTS[side])


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector.copy() if norm < 1e-12 else vector / norm


def _frame_from_xz(x_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    x_axis = _normalize(x_axis)
    z_axis = _normalize(z_axis - x_axis * float(z_axis @ x_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, _normalize(np.cross(x_axis, y_axis))))


def _frame_from_yz(y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    y_axis = _normalize(y_axis)
    z_axis = _normalize(z_axis - y_axis * float(z_axis @ y_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    return np.column_stack((x_axis, y_axis, _normalize(np.cross(x_axis, y_axis))))


def _rotation_about_x(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def _matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def _world_point(data: mujoco.MjData, body_id: int, local_point: np.ndarray) -> np.ndarray:
    return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local_point


def _fixed_finger_tip(model: mujoco.MjModel, side: Side) -> tuple[int, np.ndarray, np.ndarray]:
    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, f"hand_{side}_link")
    geom_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"hand_{side}_link_collision_box_03_geom"
    )
    rotation = _matrix_from_quaternion(model.geom_quat[geom_id])
    local_tip = model.geom_pos[geom_id] - rotation[:, 1] * model.geom_size[geom_id, 1]
    local_frame = _frame_from_yz(rotation[:, 2], -rotation[:, 1])
    return body_id, local_tip, local_frame


def _moving_finger_tip(model: mujoco.MjModel, side: Side) -> tuple[int, np.ndarray]:
    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, f"finger_{side}_link")
    geom_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"finger_{side}_link_collision_box_02_geom"
    )
    cap_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"finger_{side}_link_collision_box_03_geom"
    )
    rotation = _matrix_from_quaternion(model.geom_quat[geom_id])
    center = model.geom_pos[geom_id]
    axis = rotation[:, 1]
    endpoints = (
        center - axis * model.geom_size[geom_id, 1],
        center + axis * model.geom_size[geom_id, 1],
    )
    tip = min(endpoints, key=lambda point: float(np.linalg.norm(point - model.geom_pos[cap_id])))
    return body_id, tip.copy()


def hand_pose(
    model: mujoco.MjModel, data: mujoco.MjData, side: Side
) -> tuple[np.ndarray, np.ndarray]:
    """Return the midpoint and task frame between a hand's two fingertips."""

    fixed_body, fixed_tip, fixed_frame = _fixed_finger_tip(model, side)
    moving_body, moving_tip = _moving_finger_tip(model, side)
    midpoint = 0.5 * (
        _world_point(data, fixed_body, fixed_tip)
        + _world_point(data, moving_body, moving_tip)
    )
    rotation = (
        data.xmat[fixed_body].reshape(3, 3)
        @ fixed_frame
        @ _rotation_about_x(-90.0)
    )
    return midpoint, rotation


def _cube_grasp_target(
    model: mujoco.MjModel, data: mujoco.MjData, side: Side
) -> tuple[np.ndarray, np.ndarray]:
    cube_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom"
    )
    cube_center = data.geom_xpos[cube_id].copy()
    cube_rotation = data.geom_xmat[cube_id].reshape(3, 3)
    finger_body, finger_tip, _ = _fixed_finger_tip(model, side)
    direction = _world_point(data, finger_body, finger_tip) - cube_center
    direction[2] = 0.0
    if np.linalg.norm(direction) < 1e-9:
        direction = -cube_rotation[:, 1]
    normals = [sign * cube_rotation[:, axis] for axis in (0, 1) for sign in (-1.0, 1.0)]
    face_normal = max(normals, key=lambda normal: float(direction @ normal))
    return cube_center, _frame_from_xz(face_normal, cube_rotation[:, 2])


def drawer_handle_pose(
    model: mujoco.MjModel, data: mujoco.MjData, drawer_index: int
) -> tuple[np.ndarray, np.ndarray]:
    corners: list[np.ndarray] = []
    for box_index in range(5, 9):
        geom_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"drawer_{drawer_index}_link_collision_box_{box_index:02d}_geom",
        )
        center = data.geom_xpos[geom_id]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        for signs in np.ndindex(2, 2, 2):
            direction = np.array([1.0 if sign else -1.0 for sign in signs])
            corners.append(center + rotation @ (model.geom_size[geom_id] * direction))
    values = np.asarray(corners)
    center = 0.5 * (values.min(axis=0) + values.max(axis=0))
    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, f"drawer_{drawer_index}_link")
    return center, data.xmat[body_id].reshape(3, 3).copy()


def _cube_above_drawer_pose(
    model: mujoco.MjModel, data: mujoco.MjData, drawer_index: int
) -> tuple[np.ndarray, np.ndarray]:
    target, drawer_rotation = drawer_handle_pose(model, data, drawer_index)
    target[1] -= 0.06
    target[2] += 0.09
    drawer_x = drawer_rotation[:, 0].copy()
    drawer_x[2] = 0.0
    if np.linalg.norm(drawer_x) < 1e-9:
        drawer_x = np.array([1.0, 0.0, 0.0])
    return target, _frame_from_xz(drawer_x, np.array([0.0, 0.0, 1.0]))


def _hand_target_for_held_cube(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: Side,
    desired_cube_bottom: np.ndarray,
    desired_cube_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a desired cube pose to a hand pose using the measured physical grasp.

    This is target geometry only: it never changes, constrains, or attaches the cube.
    """

    cube_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom"
    )
    hand_position, hand_rotation = hand_pose(model, data, side)
    cube_position = data.geom_xpos[cube_id].copy()
    cube_rotation = data.geom_xmat[cube_id].reshape(3, 3).copy()
    relative_position = hand_rotation.T @ (cube_position - hand_position)
    relative_rotation = hand_rotation.T @ cube_rotation

    desired_cube_center = (
        desired_cube_bottom
        + model.geom_size[cube_id, 2] * desired_cube_rotation[:, 2]
    )
    desired_hand_rotation = desired_cube_rotation @ relative_rotation.T
    desired_hand_position = (
        desired_cube_center - desired_hand_rotation @ relative_position
    )
    return desired_hand_position, desired_hand_rotation


def _body_descends_from(model: mujoco.MjModel, body_id: int, ancestor: str) -> bool:
    while body_id > 0:
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) == ancestor:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _body_name_starts_with(model: mujoco.MjModel, body_id: int, prefix: str) -> bool:
    while body_id > 0:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.startswith(prefix):
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _contact_name(model: mujoco.MjModel, contact: mujoco.MjContact, endpoint: int) -> str:
    geom_id = int(contact.geom[endpoint])
    if geom_id >= 0:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or str(geom_id)
    flex_id = int(contact.flex[endpoint])
    if flex_id >= 0:
        return f"flex:{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, flex_id) or flex_id}"
    return "unknown"


def _all_contact_penalty(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    margin: float,
    *,
    allowed_drawer_hand: Side | None = None,
) -> tuple[float, list[str]]:
    penalty = 0.0
    messages: list[str] = []
    for contact in data.contact:
        if allowed_drawer_hand is not None and contact.geom1 >= 0 and contact.geom2 >= 0:
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            hand1 = _body_descends_from(
                model, body1, f"hand_{allowed_drawer_hand}_link"
            ) or _body_descends_from(
                model, body1, f"finger_{allowed_drawer_hand}_link"
            )
            hand2 = _body_descends_from(
                model, body2, f"hand_{allowed_drawer_hand}_link"
            ) or _body_descends_from(
                model, body2, f"finger_{allowed_drawer_hand}_link"
            )
            drawer1 = _body_name_starts_with(model, body1, "drawer_")
            drawer2 = _body_name_starts_with(model, body2, "drawer_")
            if (hand1 and drawer2) or (hand2 and drawer1):
                continue
        violation = margin - float(contact.dist)
        if violation > 0.0:
            penalty += violation * violation
            messages.append(
                f"{_contact_name(model, contact, 0)} <-> "
                f"{_contact_name(model, contact, 1)}: dist={contact.dist:.6f} m"
            )
    return penalty, messages


def _pair_contact_penalty(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: Side,
    margin: float,
    pair: Literal["drawer", "hand"],
) -> tuple[float, list[str]]:
    other: Side = "l" if side == "r" else "r"

    def is_hand(body_id: int, hand_side: Side) -> bool:
        return _body_descends_from(model, body_id, f"hand_{hand_side}_link") or _body_descends_from(
            model, body_id, f"finger_{hand_side}_link"
        )

    penalty = 0.0
    messages: list[str] = []
    for contact in data.contact:
        if contact.geom1 < 0 or contact.geom2 < 0:
            continue
        body1, body2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        moving1, moving2 = is_hand(body1, side), is_hand(body2, side)
        if pair == "drawer":
            paired1 = _body_name_starts_with(model, body1, "drawer_")
            paired2 = _body_name_starts_with(model, body2, "drawer_")
        else:
            paired1, paired2 = is_hand(body1, other), is_hand(body2, other)
        if not ((moving1 and paired2) or (moving2 and paired1)):
            continue
        violation = margin - float(contact.dist)
        if violation > 0.0:
            penalty += violation * violation
            messages.append(
                f"{_contact_name(model, contact, 0)} <-> "
                f"{_contact_name(model, contact, 1)}: dist={contact.dist:.6f} m"
            )
    return penalty, messages


@dataclass(frozen=True)
class IKSolution:
    joint_names: tuple[str, ...]
    joint_values: np.ndarray
    achieved_position: np.ndarray
    error_m: float
    collisions: tuple[str, ...]


class CollisionAwareIK:
    """Original global-plus-local Hepha solver, without viewer or recording code."""

    def __init__(
        self,
        backend: MujocoBackend,
        *,
        global_maxiter: int = 250,
        global_popsize: int = 20,
        local_maxiter: int = 1000,
    ) -> None:
        self.backend = backend
        self.model = backend.model
        self.global_maxiter = global_maxiter
        self.global_popsize = global_popsize
        self.local_maxiter = local_maxiter

    def solve(
        self,
        *,
        side: Side,
        target: np.ndarray,
        target_rotation: np.ndarray,
        seed: int,
        joint_names: tuple[str, ...] | None = None,
        finger_position: float = FINGER_OPEN,
        reset_drawers: bool = True,
        crossed_axes: bool = False,
        align_blue_axis: bool = False,
        position_weight: float = 1000.0,
        orientation_weight: float = 50.0,
        protect_drawers: bool = False,
        protect_other_hand: bool = False,
        allow_drawer_contact: bool = False,
    ) -> IKSolution:
        model = self.model
        data = mujoco.MjData(model)
        data.qpos[:] = self.backend.data.qpos
        data.qvel[:] = 0.0
        names = joint_names or _controlled_joints(side)
        joint_ids = np.array([_joint_id(model, name) for name in names], dtype=int)
        qpos_ids = model.jnt_qposadr[joint_ids].astype(int)
        bounds = [tuple(model.jnt_range[joint_id]) for joint_id in joint_ids]
        initial = data.qpos[qpos_ids].copy()
        finger_id = _joint_id(model, FINGER_JOINTS[side])
        finger_qpos_id = int(model.jnt_qposadr[finger_id])

        def prepare(candidate: np.ndarray) -> None:
            data.qpos[qpos_ids] = candidate
            data.qpos[finger_qpos_id] = np.clip(
                finger_position, *model.jnt_range[finger_id]
            )
            if reset_drawers:
                _close_drawers(model, data)
            mujoco.mj_forward(model, data)

        def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
            prepare(candidate)
            position, rotation = hand_pose(model, data, side)
            collision_cost, messages = _all_contact_penalty(
                model,
                data,
                CONTACT_MARGIN_M,
                allowed_drawer_hand=side if allow_drawer_contact else None,
            )
            if protect_drawers:
                cost, extra = _pair_contact_penalty(model, data, side, 0.020, "drawer")
                collision_cost += 100.0 * cost
                messages.extend(f"hand/drawer: {message}" for message in extra)
            if protect_other_hand:
                cost, extra = _pair_contact_penalty(model, data, side, 0.020, "hand")
                collision_cost += 100.0 * cost
                messages.extend(f"hand/hand: {message}" for message in extra)
            return position, rotation, collision_cost, messages

        spans = np.array([high - low for low, high in bounds])

        def objective(candidate: np.ndarray) -> float:
            position, rotation, collision_cost, _ = evaluate(candidate)
            position_error = position - target
            if crossed_axes:
                red_error = np.cross(rotation[:, 0], target_rotation[:, 2])
                blue_error = np.cross(rotation[:, 2], target_rotation[:, 0])
            else:
                red_error = np.cross(rotation[:, 0], target_rotation[:, 0])
                blue_error = (
                    np.cross(rotation[:, 2], target_rotation[:, 2])
                    if align_blue_axis
                    else np.zeros(3)
                )
            regularization = (candidate - initial) / spans
            return (
                position_weight * float(position_error @ position_error)
                + orientation_weight * float(red_error @ red_error)
                + orientation_weight * float(blue_error @ blue_error)
                + 50_000_000.0 * collision_cost
                + 0.002 * float(regularization @ regularization)
            )

        global_result = differential_evolution(
            objective,
            bounds,
            seed=seed,
            maxiter=self.global_maxiter,
            popsize=self.global_popsize,
            polish=False,
            tol=1e-7,
            atol=1e-9,
            updating="immediate",
        )
        local_result = minimize(
            objective,
            global_result.x,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": self.local_maxiter, "ftol": 1e-12},
        )
        position, _, _, collisions = evaluate(local_result.x)
        return IKSolution(
            joint_names=names,
            joint_values=local_result.x.copy(),
            achieved_position=position.copy(),
            error_m=float(np.linalg.norm(position - target)),
            collisions=tuple(collisions),
        )


def _close_drawers(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for drawer_index in range(1, 10):
        joint_id = _joint_id(model, f"base_link_base_drawer_{drawer_index}_joint")
        data.qpos[model.jnt_qposadr[joint_id]] = model.jnt_range[joint_id, 1]
        data.qvel[model.jnt_dofadr[joint_id]] = 0.0


def _set_finger_state(
    model: mujoco.MjModel, data: mujoco.MjData, side: Side, value: float
) -> None:
    joint_id = _joint_id(model, FINGER_JOINTS[side])
    data.qpos[model.jnt_qposadr[joint_id]] = np.clip(value, *model.jnt_range[joint_id])


def _randomize_cube(
    model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator
) -> None:
    joint_id = _joint_id(model, "cube_link_free_joint")
    qpos_id = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_id : qpos_id + 3] = model.qpos0[qpos_id : qpos_id + 3]
    data.qpos[qpos_id] += rng.uniform(-0.175, 0.175)
    data.qpos[qpos_id + 1] += rng.uniform(-0.125, 0.125)
    yaw = rng.uniform(-np.pi, np.pi)
    data.qpos[qpos_id + 3 : qpos_id + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    data.qvel[model.jnt_dofadr[joint_id] : model.jnt_dofadr[joint_id] + 6] = 0.0


def _min_geom_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    source_names: tuple[str, ...],
    target_name: str,
) -> float:
    target = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, target_name)
    return min(
        float(
            mujoco.mj_geomDistance(
                model,
                data,
                _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, source),
                target,
                0.1,
                None,
            )
        )
        for source in source_names
    )


def cube_grasp_ready(
    model: mujoco.MjModel, data: mujoco.MjData, side: Side
) -> tuple[bool, tuple[float, float]]:
    target = "cube_link_collision_box_01_geom"
    fixed = _min_geom_distance(
        model, data, (f"hand_{side}_link_collision_box_03_geom",), target
    )
    moving = _min_geom_distance(
        model,
        data,
        (
            f"finger_{side}_link_collision_box_02_geom",
            f"finger_{side}_link_collision_box_03_geom",
        ),
        target,
    )
    touching = fixed <= GRASP_CONTACT_TOLERANCE_M and moving <= GRASP_CONTACT_TOLERANCE_M
    valid_depth = (
        fixed >= -MAX_CUBE_FINGER_PENETRATION_M
        and moving >= -MAX_CUBE_FINGER_PENETRATION_M
    )
    return touching and valid_depth, (fixed, moving)


def _cube_fully_inside_drawer(
    model: mujoco.MjModel, data: mujoco.MjData, drawer_index: int
) -> bool:
    floor_id = _named_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"drawer_{drawer_index}_link_collision_box_02_geom",
    )
    cube_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom"
    )
    floor_rotation = data.geom_xmat[floor_id].reshape(3, 3)
    local_cube = floor_rotation.T @ (data.geom_xpos[cube_id] - data.geom_xpos[floor_id])
    horizontal_limit = model.geom_size[floor_id, :2] - model.geom_size[cube_id, :2] - 0.002
    inside_xy = bool(
        np.all(horizontal_limit > 0.0)
        and np.all(np.abs(local_cube[:2]) <= horizontal_limit)
    )
    floor_top = float(model.geom_size[floor_id, 2])
    cube_bottom = float(local_cube[2] - model.geom_size[cube_id, 2])
    cube_top = float(local_cube[2] + model.geom_size[cube_id, 2])
    return inside_xy and floor_top - 0.003 <= cube_bottom and cube_top <= floor_top + 0.055


@dataclass
class _Motion:
    start: np.ndarray
    target: np.ndarray
    frame_count: int
    after: str
    frame_index: int = 0

    @property
    def complete(self) -> bool:
        return self.frame_index >= self.frame_count

    def next_action(self) -> np.ndarray:
        denominator = max(1, self.frame_count - 1)
        alpha = min(1.0, self.frame_index / denominator)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        self.frame_index += 1
        return (1.0 - alpha) * self.start + alpha * self.target


class MujocoIKController:
    """Default physical demonstration: grasp cube, open drawer, insert, and close."""

    def __init__(
        self,
        backend: MujocoBackend,
        *,
        seed: int = 0,
        move_duration_s: float = 3.0,
        ik_global_maxiter: int = 250,
        ik_global_popsize: int = 20,
        ik_local_maxiter: int = 1000,
    ) -> None:
        if not isinstance(backend, MujocoBackend):
            raise TypeError("MujocoIKController requires the MuJoCo backend")
        self.backend = backend
        self.model = backend.model
        self.data = backend.data
        self.seed = seed
        self.move_duration_s = move_duration_s
        self.ik = CollisionAwareIK(
            backend,
            global_maxiter=ik_global_maxiter,
            global_popsize=ik_global_popsize,
            local_maxiter=ik_local_maxiter,
        )
        self.phase = "idle"
        self.motion: _Motion | None = None
        self.cube_hand: Side | None = None
        self.drawer_hand: Side | None = None
        self.drawer_index = 5
        self.initial_targets: dict[str, float] = {}
        self.episode_seed = seed
        self.done = False
        self.successful = False
        self.status = "not started"

    def reset(self, *, episode_seed: int) -> None:
        self.backend.reset(seed=episode_seed)
        # The original physical task was designed and tuned from the MJCF pose,
        # whereas the general-purpose viewer uses actuator-range midpoints.
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(self.seed + episode_seed)
        _close_drawers(self.model, self.data)
        _randomize_cube(self.model, self.data, rng)
        for actuator_id in self.backend.actuator_ids:
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qpos_id = int(self.model.jnt_qposadr[joint_id])
            self.data.ctrl[actuator_id] = np.clip(
                self.data.qpos[qpos_id], *self.model.actuator_ctrlrange[actuator_id]
            )
        for side in ("l", "r"):
            _set_finger_state(self.model, self.data, side, FINGER_CLOSED)
            action_index = ACTUATOR_NAMES.index(f"finger_{side}")
            self.data.ctrl[self.backend.actuator_ids[action_index]] = FINGER_CLOSED
        mujoco.mj_forward(self.model, self.data)
        self.initial_targets = {
            name: _joint_qpos(self.model, self.data, name) for name in ROBOT_JOINTS
        }
        self.drawer_index = int(rng.integers(1, 10))
        self.episode_seed = episode_seed
        self.cube_hand = None
        self.drawer_hand = None
        self.phase = "pre_cube_view"
        self.motion = None
        self.done = False
        self.successful = False
        self.status = f"targeting drawer {self.drawer_index}"

    def _current_action(self) -> np.ndarray:
        return self.data.ctrl[self.backend.actuator_ids].astype(float, copy=True)

    def _actuator_index_for_joint(self, joint_name: str) -> int:
        joint_id = _joint_id(self.model, joint_name)
        matches = np.flatnonzero(
            self.model.actuator_trnid[self.backend.actuator_ids, 0] == joint_id
        )
        if len(matches) != 1:
            raise RuntimeError(f"Expected one position actuator for joint {joint_name}")
        return int(matches[0])

    def _start_motion(
        self,
        targets: dict[str, float],
        *,
        after: str,
        duration_scale: float = 1.0,
        fingers: dict[Side, float] | None = None,
    ) -> None:
        start = self._current_action()
        target = start.copy()
        for joint_name, value in targets.items():
            target[self._actuator_index_for_joint(joint_name)] = value
        for side, value in (fingers or {}).items():
            target[ACTUATOR_NAMES.index(f"finger_{side}")] = value
        target = np.clip(target, self.backend.control_low, self.backend.control_high)
        frames = max(2, round(self.move_duration_s * duration_scale * self.backend.config.fps))
        self.motion = _Motion(start, target, frames, after)

    def _solve_to(
        self,
        *,
        side: Side,
        target: np.ndarray,
        rotation: np.ndarray,
        seed_offset: int,
        **kwargs: object,
    ) -> dict[str, float]:
        solution = self.ik.solve(
            side=side,
            target=target,
            target_rotation=rotation,
            seed=self.seed + self.episode_seed + seed_offset,
            **kwargs,
        )
        if solution.error_m > 0.005:
            print(
                f"Warning: {self.phase} IK error is {solution.error_m:.6f} m; "
                "the terminal physical task check will decide whether to keep the episode."
            )
        return dict(zip(solution.joint_names, solution.joint_values, strict=True))

    def _rest_targets(self, side: Side) -> dict[str, float]:
        targets = {name: self.initial_targets[name] for name in SIDE_JOINTS[side]}
        arm_joint = f"forearm_{side}_link_forearm_{side}_arm_{side}_joint"
        joint_id = _joint_id(self.model, arm_joint)
        targets[arm_joint] = float(
            self.model.jnt_range[joint_id, 1 if side == "l" else 0]
        )
        return targets

    def _dispatch_phase(self) -> None:
        base = "base_link_base_cnc_x_joint"
        lateral = "cnc_x_link_cnc_x_cnc_y_joint"
        base_id, lateral_id = _joint_id(self.model, base), _joint_id(self.model, lateral)

        if self.phase == "pre_cube_view":
            self._start_motion({base: float(self.model.jnt_range[base_id, 1])}, after="open")
        elif self.phase == "open":
            self._start_motion({}, after="stage", fingers={"l": FINGER_OPEN, "r": FINGER_OPEN})
        elif self.phase == "stage":
            lateral_center = float(np.mean(self.model.jnt_range[lateral_id]))
            self._start_motion(
                {base: float(self.model.jnt_range[base_id, 1]), lateral: lateral_center},
                after="cube_ik",
                fingers={"l": FINGER_OPEN, "r": FINGER_OPEN},
            )
        elif self.phase == "cube_ik":
            mujoco.mj_forward(self.model, self.data)
            cube_id = _named_id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom"
            )
            cube_position = self.data.geom_xpos[cube_id]
            distances = {
                side: float(
                    np.linalg.norm(hand_pose(self.model, self.data, side)[0] - cube_position)
                )
                for side in ("l", "r")
            }
            self.cube_hand = min(distances, key=distances.__getitem__)
            self.drawer_hand = "r" if self.cube_hand == "l" else "l"
            target, rotation = _cube_grasp_target(self.model, self.data, self.cube_hand)
            targets = self._solve_to(
                side=self.cube_hand,
                target=target,
                rotation=rotation,
                seed_offset=0,
                position_weight=100_000.0,
                orientation_weight=10.0,
                protect_other_hand=True,
            )
            self._start_motion(
                targets,
                after="close_cube",
                fingers={self.cube_hand: FINGER_OPEN, self.drawer_hand: FINGER_OPEN},
            )
        elif self.phase == "close_cube":
            assert self.cube_hand is not None
            self._start_motion(
                {}, after="cube_clearance", fingers={self.cube_hand: CUBE_GRASP}
            )
        elif self.phase == "cube_clearance":
            assert self.cube_hand is not None
            ready, gaps = cube_grasp_ready(self.model, self.data, self.cube_hand)
            if not ready:
                print(
                    "Warning: cube grasp contact was not confirmed "
                    f"(fixed={gaps[0]:.4f} m, moving={gaps[1]:.4f} m)."
                )
            target = min(
                _joint_qpos(self.model, self.data, base) + 0.06,
                self.model.jnt_range[base_id, 1],
            )
            self._start_motion(
                {base: float(target)},
                after="return_without_base",
                fingers={self.cube_hand: CUBE_GRASP},
            )
        elif self.phase == "return_without_base":
            assert self.cube_hand is not None
            targets = {name: value for name, value in self.initial_targets.items() if name != base}
            self._start_motion(
                targets, after="lift_cube", fingers={self.cube_hand: CUBE_GRASP}
            )
        elif self.phase == "lift_cube":
            assert self.cube_hand is not None
            arm = (
                f"forearm_{self.cube_hand}_link_forearm_{self.cube_hand}_"
                f"arm_{self.cube_hand}_joint"
            )
            wrist = f"arm_{self.cube_hand}_link_arm_{self.cube_hand}_wrist_{self.cube_hand}_joint"
            arm_id, wrist_id = _joint_id(self.model, arm), _joint_id(self.model, wrist)
            arm_target = _joint_qpos(self.model, self.data, arm) + (
                0.5 if self.cube_hand == "l" else -0.5
            )
            wrist_target = _joint_qpos(self.model, self.data, wrist) - 0.25
            targets = {
                arm: float(np.clip(arm_target, *self.model.jnt_range[arm_id])),
                wrist: float(np.clip(wrist_target, *self.model.jnt_range[wrist_id])),
            }
            self._start_motion(
                targets, after="drawer_ik", fingers={self.cube_hand: CUBE_GRASP}
            )
        elif self.phase == "drawer_ik":
            assert self.drawer_hand is not None
            target, rotation = drawer_handle_pose(self.model, self.data, self.drawer_index)
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=rotation,
                seed_offset=1,
                crossed_axes=True,
                position_weight=40_000.0,
                orientation_weight=10.0,
                finger_position=DRAWER_GRASP,
                protect_other_hand=True,
                allow_drawer_contact=True,
            )
            self._start_motion(
                targets, after="grasp_drawer", fingers={self.drawer_hand: FINGER_OPEN}
            )
        elif self.phase == "grasp_drawer":
            assert self.drawer_hand is not None
            self._start_motion({}, after="pull_drawer", fingers={self.drawer_hand: DRAWER_GRASP})
        elif self.phase == "pull_drawer":
            assert self.drawer_hand is not None
            drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
            drawer_id = _joint_id(self.model, drawer_joint)
            opening = max(
                0.0,
                _joint_qpos(self.model, self.data, drawer_joint)
                - float(self.model.jnt_range[drawer_id, 0]),
            )
            limit = max(
                float(self.model.jnt_range[base_id, 0]),
                float(self.model.jnt_range[base_id, 1]) - 0.05,
            )
            target = min(_joint_qpos(self.model, self.data, base) + opening, limit)
            self._start_motion(
                {base: target},
                after="release_open_drawer",
                fingers={self.drawer_hand: DRAWER_GRASP},
            )
        elif self.phase == "release_open_drawer":
            assert self.drawer_hand is not None
            self._start_motion(
                {},
                after="open_drawer_clearance",
                duration_scale=0.4,
                fingers={self.drawer_hand: FINGER_OPEN},
            )
        elif self.phase == "open_drawer_clearance":
            assert self.drawer_hand is not None
            target = min(
                _joint_qpos(self.model, self.data, base) + 0.05,
                self.model.jnt_range[base_id, 1],
            )
            self._start_motion(
                {base: float(target)},
                after="drawer_hand_rest",
                duration_scale=0.6,
                fingers={self.drawer_hand: FINGER_OPEN},
            )
        elif self.phase == "drawer_hand_rest":
            assert self.drawer_hand is not None and self.cube_hand is not None
            self._start_motion(
                self._rest_targets(self.drawer_hand),
                after="cube_above_drawer",
                fingers={self.drawer_hand: FINGER_CLOSED, self.cube_hand: CUBE_GRASP},
            )
        elif self.phase == "cube_above_drawer":
            assert self.cube_hand is not None
            cube_target, cube_rotation = _cube_above_drawer_pose(
                self.model, self.data, self.drawer_index
            )
            target, rotation = _hand_target_for_held_cube(
                self.model,
                self.data,
                self.cube_hand,
                cube_target,
                cube_rotation,
            )
            names = (base, lateral, *SIDE_JOINTS[self.cube_hand])
            targets = self._solve_to(
                side=self.cube_hand,
                target=target,
                rotation=rotation,
                seed_offset=2,
                joint_names=names,
                reset_drawers=False,
                align_blue_axis=True,
                position_weight=25_000.0,
                orientation_weight=5.0,
                finger_position=CUBE_GRASP,
                protect_drawers=True,
                protect_other_hand=True,
            )
            fingers: dict[Side, float] = {self.cube_hand: CUBE_GRASP}
            if self.drawer_hand is not None:
                fingers[self.drawer_hand] = FINGER_CLOSED
            self._start_motion(targets, after="drop_cube", fingers=fingers)
        elif self.phase == "drop_cube":
            assert self.cube_hand is not None
            self._start_motion(
                {},
                after="cube_hand_rest",
                duration_scale=0.6,
                fingers={self.cube_hand: FINGER_OPEN},
            )
        elif self.phase == "cube_hand_rest":
            assert self.cube_hand is not None
            self._start_motion(
                self._rest_targets(self.cube_hand),
                after="center_lateral",
                fingers={self.cube_hand: FINGER_OPEN},
            )
        elif self.phase == "center_lateral":
            self._start_motion(
                {lateral: float(np.mean(self.model.jnt_range[lateral_id]))},
                after="pre_close_clearance",
                fingers={"l": FINGER_CLOSED, "r": FINGER_CLOSED},
            )
        elif self.phase == "pre_close_clearance":
            target = min(
                _joint_qpos(self.model, self.data, base) + 0.06,
                self.model.jnt_range[base_id, 1],
            )
            self._start_motion(
                {base: float(target)},
                after="choose_closing_hand",
                fingers={"l": FINGER_CLOSED, "r": FINGER_CLOSED},
            )
        elif self.phase == "choose_closing_hand":
            target, _ = drawer_handle_pose(self.model, self.data, self.drawer_index)
            target[2] += 0.03
            distances = {
                side: float(np.linalg.norm(hand_pose(self.model, self.data, side)[0] - target))
                for side in ("l", "r")
            }
            self.drawer_hand = min(distances, key=distances.__getitem__)
            self._start_motion(
                {}, after="close_drawer_ik", fingers={self.drawer_hand: DRAWER_GRASP}
            )
        elif self.phase == "close_drawer_ik":
            assert self.drawer_hand is not None
            target, rotation = drawer_handle_pose(self.model, self.data, self.drawer_index)
            target[2] += 0.03
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=rotation,
                seed_offset=4,
                reset_drawers=False,
                crossed_axes=True,
                position_weight=40_000.0,
                orientation_weight=10.0,
                finger_position=DRAWER_GRASP,
                protect_other_hand=True,
                allow_drawer_contact=True,
            )
            self._start_motion(
                targets, after="push_drawer", fingers={self.drawer_hand: DRAWER_GRASP}
            )
        elif self.phase == "push_drawer":
            assert self.drawer_hand is not None
            drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
            drawer_id = _joint_id(self.model, drawer_joint)
            delta = _joint_qpos(self.model, self.data, drawer_joint) - float(
                self.model.jnt_range[drawer_id, 1]
            )
            target = np.clip(
                _joint_qpos(self.model, self.data, base) + delta - 0.03,
                *self.model.jnt_range[base_id],
            )
            self._start_motion(
                {base: float(target)},
                after="release_closed_drawer",
                fingers={self.drawer_hand: DRAWER_GRASP},
            )
        elif self.phase == "release_closed_drawer":
            assert self.drawer_hand is not None
            target = min(
                _joint_qpos(self.model, self.data, base) + 0.04,
                self.model.jnt_range[base_id, 1],
            )
            self._start_motion(
                {base: float(target)},
                after="final_return",
                duration_scale=0.6,
                fingers={self.drawer_hand: FINGER_CLOSED},
            )
        elif self.phase == "final_return":
            self._start_motion(
                self.initial_targets,
                after="finished",
                fingers={"l": FINGER_CLOSED, "r": FINGER_CLOSED},
            )
        elif self.phase == "finished":
            drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
            drawer_id = _joint_id(self.model, drawer_joint)
            opening = float(self.model.jnt_range[drawer_id, 1]) - _joint_qpos(
                self.model, self.data, drawer_joint
            )
            inside = _cube_fully_inside_drawer(self.model, self.data, self.drawer_index)
            self.successful = opening <= 0.003 and inside
            self.status = (
                f"cube inside closed drawer {self.drawer_index}"
                if self.successful
                else f"task failed: drawer opening={opening:.4f} m, cube_inside={inside}"
            )
            self.done = True
        else:
            raise RuntimeError(f"Unknown IK task phase: {self.phase}")

    def action(self, progress: float) -> np.ndarray:
        del progress
        if self.phase == "idle":
            raise RuntimeError("Controller must be reset before requesting an action")
        if self.motion is not None and self.motion.complete:
            self.phase = self.motion.after
            self.motion = None
        while self.motion is None and not self.done:
            self._dispatch_phase()
        return self._current_action() if self.done else self.motion.next_action()
