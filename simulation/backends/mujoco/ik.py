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
GRASP_CONTACT_TOLERANCE_M = 0.016
MAX_CUBE_FINGER_PENETRATION_M = 0.020
IK_POSITION_TOLERANCE_M = 0.010
IK_ORIENTATION_TOLERANCE_DEG = 30.0
MIN_DRAWER_OPENING_M = 0.064
MAX_DRAWER_CLOSED_OPENING_M = 0.003
CUBE_SPAWN_RADIUS_M = 0.10
CUBE_GRASP_APPROACH_HEIGHT_M = 0.10
DRAWER_APPROACH_DISTANCE_M = 0.07
DRAWER_PUSH_MARGIN_M = 0.03

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


def _closest_hand(
    model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray
) -> Side:
    distances = {
        side: float(np.linalg.norm(hand_pose(model, data, side)[0] - target))
        for side in ("l", "r")
    }
    return min(distances, key=distances.__getitem__)


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


def _drawer_handle_corners(
    model: mujoco.MjModel, data: mujoco.MjData, drawer_index: int
) -> np.ndarray:
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
    return np.asarray(corners)


def drawer_handle_pose(
    model: mujoco.MjModel, data: mujoco.MjData, drawer_index: int
) -> tuple[np.ndarray, np.ndarray]:
    values = _drawer_handle_corners(model, data, drawer_index)
    center = 0.5 * (values.min(axis=0) + values.max(axis=0))
    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, f"drawer_{drawer_index}_link")
    return center, data.xmat[body_id].reshape(3, 3).copy()


def _drawer_hand_target_pose(
    model: mujoco.MjModel, data: mujoco.MjData, drawer_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a live target 7 cm in front of the drawer-handle center."""

    handle_center, drawer_rotation = drawer_handle_pose(model, data, drawer_index)
    drawer_joint = _joint_id(model, f"base_link_base_drawer_{drawer_index}_joint")
    inward_axis = _normalize(data.xaxis[drawer_joint])
    target = handle_center - inward_axis * DRAWER_APPROACH_DISTANCE_M
    hand_rotation = _frame_from_xz(
        drawer_rotation[:, 2], drawer_rotation[:, 0]
    )
    return target, hand_rotation


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
    orientation_error_deg: float
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
        joint_bounds: dict[str, tuple[float, float]] | None = None,
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
        bounds = [
            (joint_bounds or {}).get(name, tuple(model.jnt_range[joint_id]))
            for name, joint_id in zip(names, joint_ids, strict=True)
        ]
        for name, joint_id, (low, high) in zip(
            names, joint_ids, bounds, strict=True
        ):
            model_low, model_high = model.jnt_range[joint_id]
            if low < model_low or high > model_high or low > high:
                raise ValueError(
                    f"Invalid IK bounds for {name}: {(low, high)} outside "
                    f"{tuple(model.jnt_range[joint_id])}"
                )
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
        position, rotation, _, collisions = evaluate(local_result.x)
        if crossed_axes:
            axis_pairs = (
                (rotation[:, 0], target_rotation[:, 2]),
                (rotation[:, 2], target_rotation[:, 0]),
            )
        elif align_blue_axis:
            axis_pairs = (
                (rotation[:, 0], target_rotation[:, 0]),
                (rotation[:, 2], target_rotation[:, 2]),
            )
        else:
            axis_pairs = ((rotation[:, 0], target_rotation[:, 0]),)
        orientation_error_deg = max(
            float(
                np.degrees(
                    np.arccos(
                        np.clip(abs(float(_normalize(actual) @ _normalize(desired))), 0.0, 1.0)
                    )
                )
            )
            for actual, desired in axis_pairs
        )
        return IKSolution(
            joint_names=names,
            joint_values=local_result.x.copy(),
            achieved_position=position.copy(),
            error_m=float(np.linalg.norm(position - target)),
            orientation_error_deg=orientation_error_deg,
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
    radius = CUBE_SPAWN_RADIUS_M * np.sqrt(rng.uniform())
    angle = rng.uniform(-np.pi, np.pi)
    data.qpos[qpos_id] += radius * np.cos(angle)
    data.qpos[qpos_id + 1] += radius * np.sin(angle)
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
    cube_rotation = data.geom_xmat[cube_id].reshape(3, 3)
    cube_corners = []
    for signs in np.ndindex(2, 2, 2):
        direction = np.array([1.0 if sign else -1.0 for sign in signs])
        world_corner = (
            data.geom_xpos[cube_id]
            + cube_rotation @ (model.geom_size[cube_id] * direction)
        )
        cube_corners.append(
            floor_rotation.T @ (world_corner - data.geom_xpos[floor_id])
        )
    local_corners = np.asarray(cube_corners)
    horizontal_limit = model.geom_size[floor_id, :2] - 0.002
    inside_xy = bool(np.all(np.abs(local_corners[:, :2]) <= horizontal_limit))
    floor_top = float(model.geom_size[floor_id, 2])
    cube_bottom = float(local_corners[:, 2].min())
    cube_top = float(local_corners[:, 2].max())
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
    """Drawer-first physical task with nearest-hand selection and cube handoff."""

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
        self.placement_hand: Side | None = None
        self.drawer_index = 5
        self.initial_targets: dict[str, float] = {}
        self.ik_targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.cube_approach_vertical_qpos: float | None = None
        self.episode_seed = seed
        self.done = False
        self.successful = False
        self.status = "not started"

    def reset(self, *, episode_seed: int) -> None:
        self.backend.reset(seed=episode_seed)
        # The original physical task was designed and tuned from the MJCF pose,
        # whereas the general-purpose viewer uses actuator-range midpoints.
        mujoco.mj_resetData(self.model, self.data)
        base_id = _joint_id(self.model, "base_link_base_cnc_x_joint")
        base_qpos_id = int(self.model.jnt_qposadr[base_id])
        self.data.qpos[base_qpos_id] = np.mean(self.model.jnt_range[base_id])
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
        self._hide_ik_targets()
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(
            self.model,
            self.data,
            nstep=max(1, round(0.5 / self.model.opt.timestep)),
        )
        self.data.time = 0.0
        self.initial_targets = {
            name: _joint_qpos(self.model, self.data, name) for name in ROBOT_JOINTS
        }
        self.drawer_index = int(rng.integers(1, 10))
        self._initialize_ik_targets()
        self.episode_seed = episode_seed
        self.phase = "pre_cube_view"
        self.motion = None
        self.cube_approach_vertical_qpos = None
        self.done = False
        self.successful = False
        handoff = self.cube_hand != self.placement_hand
        cube_id = _named_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom"
        )
        cube_position = self.data.geom_xpos[cube_id]
        self.status = (
            f"cube=({cube_position[0]:.4f}, {cube_position[1]:.4f}, "
            f"{cube_position[2]:.4f}), drawer={self.drawer_index}, "
            f"drawer hand={self.drawer_hand}; "
            f"cube hand={self.cube_hand}, handoff={handoff}"
        )

    def _set_ik_target(
        self, marker_body: str, position: np.ndarray, rotation: np.ndarray
    ) -> None:
        body_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, marker_body)
        self.model.body_pos[body_id] = position
        mujoco.mju_mat2Quat(
            self.model.body_quat[body_id], np.asarray(rotation, dtype=float).reshape(9)
        )

    def _refresh_drawer_targets(self) -> None:
        """Keep IK 2 and IK 4 at the live pre-handle approach pose."""

        target, rotation = _drawer_hand_target_pose(
            self.model, self.data, self.drawer_index
        )
        for target_name in ("drawer_open", "drawer_close"):
            self.ik_targets[target_name] = (target.copy(), rotation.copy())
        if self.backend.config.debug:
            self._set_ik_target("drawer_target_marker", target, rotation)
            self._set_ik_target("drawer_close_target_marker", target, rotation)
            mujoco.mj_forward(self.model, self.data)
            self.backend.sync_viewer()

    def _hide_ik_targets(self) -> None:
        for marker_body in IK_TARGET_MARKERS:
            body_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, marker_body)
            self.model.body_pos[body_id] = (0.0, 0.0, -10.0)
            self.model.body_quat[body_id] = (1.0, 0.0, 0.0, 0.0)

    def _initialize_ik_targets(self) -> None:
        """Create four IK frames; the two drawer frames are refreshed live."""

        cube_id = _named_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom"
        )
        cube_position = self.data.geom_xpos[cube_id].copy()
        self.cube_hand = _closest_hand(self.model, self.data, cube_position)
        handle_position, _ = drawer_handle_pose(
            self.model, self.data, self.drawer_index
        )
        self.drawer_hand = _closest_hand(self.model, self.data, handle_position)
        self.placement_hand = self.drawer_hand
        cube_target, grasp_rotation = _cube_grasp_target(
            self.model, self.data, self.cube_hand
        )
        cube_target = cube_target + np.array(
            [0.0, 0.0, CUBE_GRASP_APPROACH_HEIGHT_M]
        )

        opened_data = mujoco.MjData(self.model)
        opened_data.qpos[:] = self.data.qpos
        drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
        drawer_id = _joint_id(self.model, drawer_joint)
        opened_data.qpos[self.model.jnt_qposadr[drawer_id]] = self.model.jnt_range[
            drawer_id, 0
        ]
        mujoco.mj_forward(self.model, opened_data)

        cube_bottom, desired_cube_rotation = _cube_above_drawer_pose(
            self.model, opened_data, self.drawer_index
        )
        cube_rotation = self.data.geom_xmat[cube_id].reshape(3, 3).copy()
        _, placement_grasp_rotation = _cube_grasp_target(
            self.model, self.data, self.placement_hand
        )
        relative_rotation = placement_grasp_rotation.T @ cube_rotation
        cube_place_rotation = desired_cube_rotation @ relative_rotation.T
        cube_place_target = (
            cube_bottom
            + self.model.geom_size[cube_id, 2] * desired_cube_rotation[:, 2]
        )

        drawer_target, drawer_rotation = _drawer_hand_target_pose(
            self.model, self.data, self.drawer_index
        )
        self.ik_targets = {
            "cube_grasp": (cube_target, grasp_rotation),
            "drawer_open": (drawer_target.copy(), drawer_rotation.copy()),
            "cube_place": (cube_place_target, cube_place_rotation),
            "drawer_close": (drawer_target.copy(), drawer_rotation.copy()),
        }
        if self.backend.config.debug:
            for marker_body, (position, rotation) in zip(
                IK_TARGET_MARKERS, self.ik_targets.values(), strict=True
            ):
                self._set_ik_target(marker_body, position, rotation)
            mujoco.mj_forward(self.model, self.data)
            self.backend.sync_viewer()

    def _select_post_drawer_hands_and_targets(self) -> None:
        """Select hands from their live poses after the drawer has been opened."""

        cube_id = _named_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom"
        )
        cube_position = self.data.geom_xpos[cube_id].copy()
        handle_position, _ = drawer_handle_pose(
            self.model, self.data, self.drawer_index
        )
        self.cube_hand = _closest_hand(self.model, self.data, cube_position)
        self.placement_hand = _closest_hand(self.model, self.data, handle_position)

        cube_target, cube_grasp_rotation = _cube_grasp_target(
            self.model, self.data, self.cube_hand
        )
        cube_target = cube_target + np.array(
            [0.0, 0.0, CUBE_GRASP_APPROACH_HEIGHT_M]
        )
        cube_bottom, desired_cube_rotation = _cube_above_drawer_pose(
            self.model, self.data, self.drawer_index
        )
        _, placement_grasp_rotation = _cube_grasp_target(
            self.model, self.data, self.placement_hand
        )
        cube_rotation = self.data.geom_xmat[cube_id].reshape(3, 3).copy()
        relative_rotation = placement_grasp_rotation.T @ cube_rotation
        cube_place_rotation = desired_cube_rotation @ relative_rotation.T
        cube_place_target = (
            cube_bottom
            + self.model.geom_size[cube_id, 2] * desired_cube_rotation[:, 2]
        )
        self.ik_targets["cube_grasp"] = (cube_target, cube_grasp_rotation)
        self.ik_targets["cube_place"] = (cube_place_target, cube_place_rotation)

        if self.backend.config.debug:
            self._set_ik_target("hand_tip_marker", cube_target, cube_grasp_rotation)
            self._set_ik_target(
                "above_drawer_target_marker",
                cube_place_target,
                cube_place_rotation,
            )
            mujoco.mj_forward(self.model, self.data)
            self.backend.sync_viewer()

    def _fail(self, checkpoint: str, reason: str) -> None:
        self.motion = None
        self.successful = False
        self.done = True
        self.status = f"early failure after {checkpoint}: {reason}"
        print(self.status)

    def _drawer_opening(self) -> float:
        drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
        drawer_id = _joint_id(self.model, drawer_joint)
        return max(
            0.0,
            float(self.model.jnt_range[drawer_id, 1])
            - _joint_qpos(self.model, self.data, drawer_joint),
        )

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
    ) -> dict[str, float] | None:
        solution = self.ik.solve(
            side=side,
            target=target,
            target_rotation=rotation,
            seed=self.seed + self.episode_seed + seed_offset,
            **kwargs,
        )
        # TEMPORARILY DISABLED FOR DEBUGGING:
        # if (
        #     solution.error_m > IK_POSITION_TOLERANCE_M
        #     or solution.orientation_error_deg > IK_ORIENTATION_TOLERANCE_DEG
        # ):
        #     self._fail(
        #         self.phase,
        #         f"IK target not reached (position error={solution.error_m:.4f} m, "
        #         f"orientation error={solution.orientation_error_deg:.1f} deg; "
        #         f"limits={IK_POSITION_TOLERANCE_M:.3f} m/"
        #         f"{IK_ORIENTATION_TOLERANCE_DEG:.1f} deg)",
        #     )
        #     return None
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
        vertical = "cnc_y_link_cnc_y_head_joint"
        base_id = _joint_id(self.model, base)
        lateral_id = _joint_id(self.model, lateral)
        vertical_id = _joint_id(self.model, vertical)

        if self.phase == "pre_cube_view":
            self._start_motion({}, after="open")
        elif self.phase == "open":
            self._start_motion({}, after="stage", fingers={"l": FINGER_OPEN, "r": FINGER_OPEN})
        elif self.phase == "stage":
            lateral_center = float(np.mean(self.model.jnt_range[lateral_id]))
            self._start_motion(
                {lateral: lateral_center},
                after="drawer_ik",
                fingers={"l": FINGER_OPEN, "r": FINGER_OPEN},
            )
        elif self.phase == "cube_ik":
            self._select_post_drawer_hands_and_targets()
            assert (
                self.cube_hand is not None
                and self.drawer_hand is not None
                and self.placement_hand is not None
            )
            target, rotation = self.ik_targets["cube_grasp"]
            targets = self._solve_to(
                side=self.cube_hand,
                target=target,
                rotation=rotation,
                seed_offset=0,
                reset_drawers=False,
                align_blue_axis=True,
                position_weight=100_000.0,
                orientation_weight=10.0,
                protect_drawers=True,
                protect_other_hand=True,
                joint_bounds={
                    vertical: (
                        float(self.model.jnt_range[vertical_id, 0]),
                        float(self.model.jnt_range[vertical_id, 1])
                        - CUBE_GRASP_APPROACH_HEIGHT_M,
                    )
                },
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="descend_to_cube",
                fingers={self.cube_hand: FINGER_OPEN, self.drawer_hand: FINGER_OPEN},
            )
        elif self.phase == "descend_to_cube":
            assert self.cube_hand is not None
            cube_id = _named_id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "cube_link_collision_box_01_geom",
            )
            hand_position, _ = hand_pose(self.model, self.data, self.cube_hand)
            vertical_axis = _normalize(self.data.xaxis[vertical_id])
            vertical_qpos = _joint_qpos(self.model, self.data, vertical)
            self.cube_approach_vertical_qpos = vertical_qpos
            target = np.clip(
                vertical_qpos
                + float((self.data.geom_xpos[cube_id] - hand_position) @ vertical_axis),
                *self.model.jnt_range[vertical_id],
            )
            self._start_motion(
                {vertical: float(target)},
                after="close_cube",
                fingers={self.cube_hand: FINGER_OPEN},
            )
        elif self.phase == "close_cube":
            assert self.cube_hand is not None
            self._start_motion(
                {}, after="cube_clearance", fingers={self.cube_hand: CUBE_GRASP}
            )
        elif self.phase == "cube_clearance":
            assert self.cube_hand is not None
            # TEMPORARILY DISABLED FOR DEBUGGING:
            # ready, gaps = cube_grasp_ready(self.model, self.data, self.cube_hand)
            # if not ready:
            #     self._fail(
            #         "cube grasp",
            #         "both finger pads did not contact the cube within the permitted "
            #         f"gap/compression (fixed={gaps[0]:.4f} m, "
            #         f"moving={gaps[1]:.4f} m)",
            #     )
            #     return
            if self.cube_approach_vertical_qpos is None:
                raise RuntimeError("Cube approach height was not captured before grasping")
            self._start_motion(
                {vertical: self.cube_approach_vertical_qpos},
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
                targets,
                after=(
                    "cube_above_drawer"
                    if self.cube_hand == self.placement_hand
                    else "handoff_receiver_ik"
                ),
                fingers={self.cube_hand: CUBE_GRASP},
            )
        elif self.phase == "handoff_receiver_ik":
            assert self.cube_hand is not None and self.placement_hand is not None
            if self.cube_hand == self.placement_hand:
                self._start_motion(
                    {},
                    after="cube_above_drawer",
                    fingers={self.cube_hand: CUBE_GRASP},
                )
                return
            receiver = self.placement_hand
            target, rotation = _cube_grasp_target(
                self.model, self.data, receiver
            )
            targets = self._solve_to(
                side=receiver,
                target=target,
                rotation=rotation,
                seed_offset=2,
                joint_names=SIDE_JOINTS[receiver],
                reset_drawers=False,
                align_blue_axis=True,
                position_weight=100_000.0,
                orientation_weight=10.0,
                finger_position=FINGER_OPEN,
                protect_drawers=True,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="handoff_grasp",
                fingers={self.cube_hand: CUBE_GRASP, receiver: FINGER_OPEN},
            )
        elif self.phase == "handoff_grasp":
            assert self.cube_hand is not None and self.placement_hand is not None
            self._start_motion(
                {},
                after="handoff_release",
                fingers={
                    self.cube_hand: CUBE_GRASP,
                    self.placement_hand: CUBE_GRASP,
                },
            )
        elif self.phase == "handoff_release":
            assert self.cube_hand is not None and self.placement_hand is not None
            donor = self.cube_hand
            receiver = self.placement_hand
            self.cube_hand = receiver
            self._start_motion(
                {},
                after="cube_above_drawer",
                fingers={donor: FINGER_OPEN, receiver: CUBE_GRASP},
            )
        elif self.phase == "drawer_ik":
            assert self.drawer_hand is not None
            target, rotation = self.ik_targets["drawer_open"]
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=rotation,
                seed_offset=1,
                align_blue_axis=True,
                position_weight=40_000.0,
                orientation_weight=50.0,
                finger_position=FINGER_OPEN,
                protect_other_hand=True,
                joint_bounds={
                    base: (
                        float(self.model.jnt_range[base_id, 0])
                        + DRAWER_APPROACH_DISTANCE_M,
                        float(self.model.jnt_range[base_id, 1]),
                    )
                },
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="approach_drawer",
                duration_scale=1.5,
                fingers={self.drawer_hand: FINGER_OPEN},
            )
        elif self.phase == "approach_drawer":
            assert self.drawer_hand is not None
            handle_position, _ = drawer_handle_pose(
                self.model, self.data, self.drawer_index
            )
            hand_position, _ = hand_pose(self.model, self.data, self.drawer_hand)
            target = np.clip(
                _joint_qpos(self.model, self.data, base)
                + float(handle_position[1] - hand_position[1]),
                *self.model.jnt_range[base_id],
            )
            self._start_motion(
                {base: float(target)},
                after="grasp_drawer",
                fingers={self.drawer_hand: FINGER_OPEN},
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
            target = min(
                _joint_qpos(self.model, self.data, base) + opening + 0.03,
                limit,
            )
            self._start_motion(
                {base: target},
                after="release_open_drawer",
                fingers={self.drawer_hand: DRAWER_GRASP},
            )
        elif self.phase == "release_open_drawer":
            assert self.drawer_hand is not None
            # TEMPORARILY DISABLED FOR DEBUGGING:
            # opening = self._drawer_opening()
            # if opening < MIN_DRAWER_OPENING_M:
            #     self._fail(
            #         "drawer opening",
            #         f"drawer opened {opening:.4f} m; required at least "
            #         f"{MIN_DRAWER_OPENING_M:.4f} m",
            #     )
            #     return
            self._start_motion(
                {},
                after="open_drawer_clearance",
                duration_scale=0.4,
                fingers={self.drawer_hand: FINGER_OPEN},
            )
        elif self.phase == "open_drawer_clearance":
            assert self.drawer_hand is not None
            approach_position = self.ik_targets["drawer_open"][0]
            hand_position, _ = hand_pose(self.model, self.data, self.drawer_hand)
            target = np.clip(
                _joint_qpos(self.model, self.data, base)
                + float(approach_position[1] - hand_position[1]),
                *self.model.jnt_range[base_id],
            )
            self._start_motion(
                {base: float(target)},
                after="drawer_hand_rest",
                duration_scale=0.6,
                fingers={self.drawer_hand: FINGER_OPEN},
            )
        elif self.phase == "drawer_hand_rest":
            assert self.drawer_hand is not None and self.cube_hand is not None
            # TEMPORARILY DISABLED FOR DEBUGGING:
            # opening = self._drawer_opening()
            # if opening < MIN_DRAWER_OPENING_M:
            #     self._fail(
            #         "drawer release",
            #         f"drawer rebounded to {opening:.4f} m open; required at least "
            #         f"{MIN_DRAWER_OPENING_M:.4f} m",
            #     )
            #     return
            self._start_motion(
                self._rest_targets(self.drawer_hand),
                after="cube_ik",
                fingers={self.drawer_hand: FINGER_OPEN},
            )
        elif self.phase == "cube_above_drawer":
            assert self.cube_hand is not None
            target, rotation = self.ik_targets["cube_place"]
            targets = self._solve_to(
                side=self.cube_hand,
                target=target,
                rotation=rotation,
                seed_offset=3,
                reset_drawers=False,
                align_blue_axis=True,
                position_weight=25_000.0,
                orientation_weight=20.0,
                finger_position=CUBE_GRASP,
                protect_drawers=True,
                protect_other_hand=True,
            )
            if targets is None:
                return
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
            # TEMPORARILY DISABLED FOR DEBUGGING:
            # mujoco.mj_forward(self.model, self.data)
            # if not _cube_fully_inside_drawer(
            #     self.model, self.data, self.drawer_index
            # ):
            #     self._fail(
            #         "cube release",
            #         f"cube is not fully inside drawer {self.drawer_index}",
            #     )
            #     return
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
            target, _ = self.ik_targets["drawer_close"]
            self.drawer_hand = _closest_hand(self.model, self.data, target)
            self._start_motion(
                {}, after="close_drawer_ik", fingers={self.drawer_hand: FINGER_CLOSED}
            )
        elif self.phase == "close_drawer_ik":
            assert self.drawer_hand is not None
            target, rotation = self.ik_targets["drawer_close"]
            drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
            drawer_id = _joint_id(self.model, drawer_joint)
            drawer_travel = float(np.ptp(self.model.jnt_range[drawer_id]))
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=rotation,
                seed_offset=4,
                reset_drawers=False,
                align_blue_axis=True,
                position_weight=40_000.0,
                orientation_weight=10.0,
                finger_position=FINGER_CLOSED,
                protect_other_hand=True,
                joint_bounds={
                    base: (
                        float(self.model.jnt_range[base_id, 0])
                        + DRAWER_APPROACH_DISTANCE_M
                        + drawer_travel
                        + DRAWER_PUSH_MARGIN_M,
                        float(self.model.jnt_range[base_id, 1]),
                    )
                },
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="approach_drawer_for_closing",
                fingers={self.drawer_hand: FINGER_CLOSED},
            )
        elif self.phase == "approach_drawer_for_closing":
            assert self.drawer_hand is not None
            handle_position, _ = drawer_handle_pose(
                self.model, self.data, self.drawer_index
            )
            hand_position, _ = hand_pose(self.model, self.data, self.drawer_hand)
            target = np.clip(
                _joint_qpos(self.model, self.data, base)
                + float(handle_position[1] - hand_position[1]),
                *self.model.jnt_range[base_id],
            )
            self._start_motion(
                {base: float(target)},
                after="push_drawer",
                fingers={self.drawer_hand: FINGER_CLOSED},
            )
        elif self.phase == "push_drawer":
            assert self.drawer_hand is not None
            drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
            drawer_id = _joint_id(self.model, drawer_joint)
            delta = _joint_qpos(self.model, self.data, drawer_joint) - float(
                self.model.jnt_range[drawer_id, 1]
            )
            target = np.clip(
                _joint_qpos(self.model, self.data, base)
                + delta
                - DRAWER_PUSH_MARGIN_M,
                *self.model.jnt_range[base_id],
            )
            self._start_motion(
                {base: float(target)},
                after="release_closed_drawer",
                fingers={self.drawer_hand: FINGER_CLOSED},
            )
        elif self.phase == "release_closed_drawer":
            assert self.drawer_hand is not None
            # TEMPORARILY DISABLED FOR DEBUGGING:
            # opening = self._drawer_opening()
            # if opening > MAX_DRAWER_CLOSED_OPENING_M:
            #     self._fail(
            #         "drawer closing",
            #         f"drawer remains open by {opening:.4f} m; permitted at most "
            #         f"{MAX_DRAWER_CLOSED_OPENING_M:.4f} m",
            #     )
            #     return
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
            # TEMPORARILY DISABLED FOR DEBUGGING:
            # opening = self._drawer_opening()
            # inside = _cube_fully_inside_drawer(
            #     self.model, self.data, self.drawer_index
            # )
            # if opening > MAX_DRAWER_CLOSED_OPENING_M or not inside:
            #     self._fail(
            #         "drawer release",
            #         f"post-release drawer opening={opening:.4f} m, "
            #         f"cube_inside={inside}; required opening <= "
            #         f"{MAX_DRAWER_CLOSED_OPENING_M:.4f} m and cube_inside=True",
            #     )
            #     return
            self._start_motion(
                self.initial_targets,
                after="finished",
                fingers={"l": FINGER_CLOSED, "r": FINGER_CLOSED},
            )
        elif self.phase == "finished":
            opening = self._drawer_opening()
            inside = _cube_fully_inside_drawer(self.model, self.data, self.drawer_index)
            self.successful = opening <= MAX_DRAWER_CLOSED_OPENING_M and inside
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
        self._refresh_drawer_targets()
        if self.motion is not None and self.motion.complete:
            self.phase = self.motion.after
            self.motion = None
        while self.motion is None and not self.done:
            self._dispatch_phase()
        return self._current_action() if self.done else self.motion.next_action()
