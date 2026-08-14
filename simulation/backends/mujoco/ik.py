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
CUBE_GRASP = 0.36
DRAWER_GRASP = 0.05
IDLE_FOREARM_DRAWER_PULL_POSITION = -0.25
HANDOFF_DONOR_ARM_ELEVATION_RAD = 0.5
CONTACT_MARGIN_M = 0.005
MIN_DRAWER_OPENING_M = 0.020
MAX_DRAWER_CLOSED_OPENING_M = 0.010
CUBE_LIFT_CHECK_M = 0.020
CUBE_HAND_DISTANCE_M = 0.050
CUBE_SPAWN_RADIUS_M = 0.10
CUBE_GRASP_APPROACH_HEIGHT_M = 0.13
DRAWER_APPROACH_DISTANCE_M = 0.07
DRAWER_CLOSE_APPROACH_DISTANCE_M = 0.05
DRAWER_TARGET_Z_OFFSET_M = -0.01
DRAWER_PUSH_MARGIN_M = 0.03
CUBE_PLACE_HANDLE_Y_OFFSET_M = -0.05
CUBE_PLACE_HANDLE_Z_OFFSET_M = 0.07
TOP_ROW_CUBE_PLACE_CLEARANCE_M = 0.04

# Conservative episode-level trajectory augmentation. These are amplitudes, so
# a value such as 0.003 means a uniform sample in [-3 mm, 3 mm]. Safety checks,
# collision margins, and joint limits are intentionally not randomized.
CUBE_GRASP_DEPTH_NOISE_M = 0.001
CUBE_GRASP_LATERAL_NOISE_M = 0.003
CUBE_GRASP_HEIGHT_NOISE_M = 0.008
CUBE_GRASP_ORIENTATION_NOISE_DEG = 3.0
DRAWER_HANDLE_LATERAL_NOISE_M = 0.008
DRAWER_APPROACH_NOISE_M = 0.004
DRAWER_CONTACT_DEPTH_NOISE_M = 0.002
DRAWER_TARGET_Z_NOISE_M = 0.002
DRAWER_ORIENTATION_NOISE_DEG = 3.0
DRAWER_PULL_SHORTFALL_NOISE_M = 0.004
CUBE_PLACE_LATERAL_NOISE_M = 0.007
CUBE_PLACE_DEPTH_NOISE_M = 0.005
CUBE_PLACE_UPWARD_NOISE_M = 0.004
CUBE_PLACE_ORIENTATION_NOISE_DEG = 4.0
HANDOFF_POSITION_NOISE_M = 0.006
POST_GRASP_ARM_NOISE_RAD = 0.04
POST_GRASP_WRIST_NOISE_RAD = 0.03
HANDOFF_DONOR_ELEVATION_NOISE_RAD = 0.04
IDLE_FOREARM_NOISE_RAD = 0.03
INITIAL_VIEW_CENTER_NOISE_FRACTION = 0.01
PRE_CLOSE_CLEARANCE_NOISE_M = 0.005
DRAWER_PUSH_MARGIN_NOISE_M = 0.004
CUBE_GRASP_POSITION_NOISE = 0.01
DRAWER_GRASP_POSITION_NOISE = 0.005
MOTION_DURATION_NOISE_FRACTION = 0.15

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


def _rotation_from_rpy_degrees(angles: np.ndarray) -> np.ndarray:
    """Return a local roll-pitch-yaw perturbation matrix."""

    roll, pitch, yaw = np.deg2rad(np.asarray(angles, dtype=float))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    rotation_y = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rotation_z = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rotation_z @ rotation_y @ rotation_x


@dataclass(frozen=True)
class TrajectoryRandomization:
    """One fixed, reproducible trajectory perturbation profile per episode."""

    cube_grasp_local_offset: np.ndarray
    cube_grasp_rotation: np.ndarray
    cube_grasp_height_delta: float
    drawer_handle_lateral_offset: float
    drawer_open_approach_delta: float
    drawer_close_approach_delta: float
    drawer_contact_depth_delta: float
    drawer_target_z_delta: float
    drawer_rotation: np.ndarray
    drawer_pull_shortfall: float
    cube_place_local_offset: np.ndarray
    cube_place_rotation: np.ndarray
    handoff_world_offset: np.ndarray
    post_grasp_arm_delta: float
    post_grasp_wrist_delta: float
    handoff_donor_elevation_delta: float
    idle_forearm_delta: float
    initial_lateral_fraction_delta: float
    initial_vertical_fraction_delta: float
    pre_close_clearance_delta: float
    drawer_push_margin_delta: float
    cube_grasp_position: float
    drawer_grasp_position: float

    @classmethod
    def sample(
        cls,
        *,
        seed: int,
        episode_seed: int,
        scale: float = 1.0,
    ) -> TrajectoryRandomization:
        if scale < 0.0:
            raise ValueError("Trajectory randomization scale must be non-negative")
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, episode_seed, 0x48455048])
        )

        def symmetric(amplitude: float, size: int | None = None):
            return rng.uniform(-amplitude * scale, amplitude * scale, size=size)

        cube_angles = symmetric(CUBE_GRASP_ORIENTATION_NOISE_DEG, size=3)
        drawer_angles = symmetric(DRAWER_ORIENTATION_NOISE_DEG, size=3)
        placement_angles = symmetric(CUBE_PLACE_ORIENTATION_NOISE_DEG, size=3)
        return cls(
            cube_grasp_local_offset=np.array(
                [
                    symmetric(CUBE_GRASP_DEPTH_NOISE_M),
                    symmetric(CUBE_GRASP_LATERAL_NOISE_M),
                    0.0,
                ],
                dtype=float,
            ),
            cube_grasp_rotation=_rotation_from_rpy_degrees(cube_angles),
            cube_grasp_height_delta=float(symmetric(CUBE_GRASP_HEIGHT_NOISE_M)),
            drawer_handle_lateral_offset=float(
                symmetric(DRAWER_HANDLE_LATERAL_NOISE_M)
            ),
            drawer_open_approach_delta=float(symmetric(DRAWER_APPROACH_NOISE_M)),
            drawer_close_approach_delta=float(symmetric(DRAWER_APPROACH_NOISE_M)),
            drawer_contact_depth_delta=float(
                symmetric(DRAWER_CONTACT_DEPTH_NOISE_M)
            ),
            drawer_target_z_delta=float(symmetric(DRAWER_TARGET_Z_NOISE_M)),
            drawer_rotation=_rotation_from_rpy_degrees(drawer_angles),
            drawer_pull_shortfall=float(
                rng.uniform(0.0, DRAWER_PULL_SHORTFALL_NOISE_M * scale)
            ),
            cube_place_local_offset=np.array(
                [
                    symmetric(CUBE_PLACE_LATERAL_NOISE_M),
                    symmetric(CUBE_PLACE_DEPTH_NOISE_M),
                    rng.uniform(0.0, CUBE_PLACE_UPWARD_NOISE_M * scale),
                ],
                dtype=float,
            ),
            cube_place_rotation=_rotation_from_rpy_degrees(placement_angles),
            handoff_world_offset=np.asarray(
                symmetric(HANDOFF_POSITION_NOISE_M, size=3), dtype=float
            ),
            post_grasp_arm_delta=float(symmetric(POST_GRASP_ARM_NOISE_RAD)),
            post_grasp_wrist_delta=float(symmetric(POST_GRASP_WRIST_NOISE_RAD)),
            handoff_donor_elevation_delta=float(
                symmetric(HANDOFF_DONOR_ELEVATION_NOISE_RAD)
            ),
            idle_forearm_delta=float(symmetric(IDLE_FOREARM_NOISE_RAD)),
            initial_lateral_fraction_delta=float(
                symmetric(INITIAL_VIEW_CENTER_NOISE_FRACTION)
            ),
            initial_vertical_fraction_delta=float(
                symmetric(INITIAL_VIEW_CENTER_NOISE_FRACTION)
            ),
            pre_close_clearance_delta=float(
                symmetric(PRE_CLOSE_CLEARANCE_NOISE_M)
            ),
            drawer_push_margin_delta=float(
                symmetric(DRAWER_PUSH_MARGIN_NOISE_M)
            ),
            cube_grasp_position=float(
                np.clip(
                    CUBE_GRASP + symmetric(CUBE_GRASP_POSITION_NOISE),
                    FINGER_CLOSED,
                    FINGER_OPEN,
                )
            ),
            drawer_grasp_position=float(
                np.clip(
                    DRAWER_GRASP + symmetric(DRAWER_GRASP_POSITION_NOISE),
                    FINGER_CLOSED,
                    FINGER_OPEN,
                )
            ),
        )


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


def _drawer_is_center(drawer_index: int) -> bool:
    if not 1 <= drawer_index <= 9:
        raise ValueError(f"Drawer index must be in [1, 9], got {drawer_index}")
    return (drawer_index - 1) % 3 == 1


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

    handle_center, hand_rotation = _drawer_handle_target_pose(
        model, data, drawer_index
    )
    drawer_joint = _joint_id(model, f"base_link_base_drawer_{drawer_index}_joint")
    inward_axis = _normalize(data.xaxis[drawer_joint])
    target = handle_center - inward_axis * DRAWER_APPROACH_DISTANCE_M
    target[2] += DRAWER_TARGET_Z_OFFSET_M
    return target, hand_rotation


def _drawer_handle_target_pose(
    model: mujoco.MjModel, data: mujoco.MjData, drawer_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a live hand frame at the exact center of the drawer handle."""

    handle_center, drawer_rotation = drawer_handle_pose(model, data, drawer_index)
    hand_rotation = _frame_from_xz(
        drawer_rotation[:, 2], drawer_rotation[:, 0]
    )
    return handle_center, hand_rotation


def _drawer_close_target_pose(
    model: mujoco.MjModel, data: mujoco.MjData, drawer_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a live closing target 5 cm in front of the handle center."""

    handle_center, hand_rotation = _drawer_handle_target_pose(
        model, data, drawer_index
    )
    drawer_joint = _joint_id(model, f"base_link_base_drawer_{drawer_index}_joint")
    inward_axis = _normalize(data.xaxis[drawer_joint])
    return (
        handle_center - inward_axis * DRAWER_CLOSE_APPROACH_DISTANCE_M,
        hand_rotation,
    )


def _drawer_gripper_rotation(rotation: np.ndarray, side: Side) -> np.ndarray:
    """Orient the mirrored grippers so their fixed finger stays above the handle."""

    side_rotation = (
        np.diag((-1.0, -1.0, 1.0))
        if side == "r"
        else np.diag((1.0, -1.0, -1.0))
    )
    return rotation @ side_rotation


def _cube_above_drawer_pose(
    model: mujoco.MjModel, data: mujoco.MjData, drawer_index: int
) -> tuple[np.ndarray, np.ndarray]:
    target, drawer_rotation = drawer_handle_pose(model, data, drawer_index)
    target[1] += CUBE_PLACE_HANDLE_Y_OFFSET_M
    target[2] += CUBE_PLACE_HANDLE_Z_OFFSET_M
    if drawer_index <= 3:
        target[2] += TOP_ROW_CUBE_PLACE_CLEARANCE_M
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
        directed_axes: bool = False,
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
            if directed_axes:
                red_error = rotation[:, 0] - target_rotation[:, 0]
                blue_error = (
                    rotation[:, 2] - target_rotation[:, 2]
                    if align_blue_axis
                    else np.zeros(3)
                )
            elif crossed_axes:
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
        if directed_axes:
            axis_pairs = (
                (rotation[:, 0], target_rotation[:, 0]),
                (rotation[:, 2], target_rotation[:, 2]),
            ) if align_blue_axis else ((rotation[:, 0], target_rotation[:, 0]),)
        elif crossed_axes:
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
                        np.clip(
                            float(_normalize(actual) @ _normalize(desired))
                            if directed_axes
                            else abs(float(_normalize(actual) @ _normalize(desired))),
                            -1.0 if directed_axes else 0.0,
                            1.0,
                        )
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
    storage_center_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "storage_bin_center_marker"
    )
    data.qpos[qpos_id : qpos_id + 3] = data.xpos[storage_center_id]
    radius = CUBE_SPAWN_RADIUS_M * np.sqrt(rng.uniform())
    angle = rng.uniform(-np.pi, np.pi)
    data.qpos[qpos_id] += radius * np.cos(angle)
    data.qpos[qpos_id + 1] += radius * np.sin(angle)
    yaw = rng.uniform(-np.pi, np.pi)
    data.qpos[qpos_id + 3 : qpos_id + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    data.qvel[model.jnt_dofadr[joint_id] : model.jnt_dofadr[joint_id] + 6] = 0.0


def _cube_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    cube_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom"
    )
    return data.geom_xpos[cube_id].copy()


def _cube_near_hand(
    model: mujoco.MjModel, data: mujoco.MjData, side: Side
) -> tuple[bool, float]:
    distance = float(np.linalg.norm(_cube_position(model, data) - hand_pose(model, data, side)[0]))
    return distance <= CUBE_HAND_DISTANCE_M, distance


def _cube_center_inside_drawer(
    model: mujoco.MjModel, data: mujoco.MjData, drawer_index: int
) -> bool:
    floor_id = _named_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"drawer_{drawer_index}_link_collision_box_02_geom",
    )
    floor_rotation = data.geom_xmat[floor_id].reshape(3, 3)
    local_center = floor_rotation.T @ (
        _cube_position(model, data) - data.geom_xpos[floor_id]
    )
    inside_xy = bool(
        np.all(np.abs(local_center[:2]) <= model.geom_size[floor_id, :2])
    )
    floor_top = float(model.geom_size[floor_id, 2])
    inside_height = floor_top - 0.005 <= local_center[2] <= floor_top + 0.040
    return inside_xy and inside_height


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
        trajectory_randomization_scale: float = 1.0,
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
        if trajectory_randomization_scale < 0.0:
            raise ValueError("Trajectory randomization scale must be non-negative")
        self.trajectory_randomization_scale = trajectory_randomization_scale
        self.trajectory = TrajectoryRandomization.sample(
            seed=seed,
            episode_seed=0,
            scale=trajectory_randomization_scale,
        )
        self._motion_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0, 0x4D4F544E])
        )
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
        self.cube_initial_z: float | None = None
        self.drawer_close_cnc_start: float | None = None
        self.handoff_target: np.ndarray | None = None
        self.handoff_donor: Side | None = None
        self.episode_seed = seed
        self.done = False
        self.successful = False
        self.status = "not started"

    def reset(self, *, episode_seed: int) -> None:
        self.backend.reset(seed=self.seed + episode_seed)
        # The original physical task was designed and tuned from the MJCF pose,
        # whereas the general-purpose viewer uses actuator-range midpoints.
        mujoco.mj_resetData(self.model, self.data)
        base_id = _joint_id(self.model, "base_link_base_cnc_x_joint")
        base_qpos_id = int(self.model.jnt_qposadr[base_id])
        self.data.qpos[base_qpos_id] = np.mean(self.model.jnt_range[base_id])
        rng = np.random.default_rng(self.seed + episode_seed)
        mujoco.mj_forward(self.model, self.data)
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
        self.cube_initial_z = float(_cube_position(self.model, self.data)[2])
        self.drawer_index = int(rng.integers(1, 10))
        self.trajectory = TrajectoryRandomization.sample(
            seed=self.seed,
            episode_seed=episode_seed,
            scale=self.trajectory_randomization_scale,
        )
        self._motion_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, episode_seed, 0x4D4F544E])
        )
        self._initialize_ik_targets()
        self.episode_seed = episode_seed
        self.phase = "initialize_cnc"
        self.motion = None
        self.cube_approach_vertical_qpos = None
        self.drawer_close_cnc_start = None
        self.handoff_target = None
        self.handoff_donor = None
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
        """Keep the opening and closing approach frames live with the drawer."""

        opening_target, opening_rotation = _drawer_hand_target_pose(
            self.model, self.data, self.drawer_index
        )
        closing_target, closing_rotation = _drawer_close_target_pose(
            self.model, self.data, self.drawer_index
        )
        drawer_joint = _joint_id(
            self.model, f"base_link_base_drawer_{self.drawer_index}_joint"
        )
        inward_axis = _normalize(self.data.xaxis[drawer_joint])
        _, drawer_rotation = drawer_handle_pose(
            self.model, self.data, self.drawer_index
        )
        handle_axis = _normalize(drawer_rotation[:, 0])
        common_offset = (
            handle_axis * self.trajectory.drawer_handle_lateral_offset
            + np.array([0.0, 0.0, self.trajectory.drawer_target_z_delta])
        )
        opening_target += (
            common_offset
            - inward_axis * self.trajectory.drawer_open_approach_delta
        )
        closing_target += (
            common_offset
            - inward_axis * self.trajectory.drawer_close_approach_delta
        )
        opening_rotation = opening_rotation @ self.trajectory.drawer_rotation
        closing_rotation = closing_rotation @ self.trajectory.drawer_rotation
        self.ik_targets["drawer_open"] = (opening_target, opening_rotation)
        self.ik_targets["drawer_close"] = (closing_target, closing_rotation)
        if self.backend.config.debug:
            self._set_ik_target(
                "drawer_target_marker", opening_target, opening_rotation
            )
            self._set_ik_target(
                "drawer_close_target_marker", closing_target, closing_rotation
            )
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
        cube_target = (
            cube_target
            + grasp_rotation @ self.trajectory.cube_grasp_local_offset
            + np.array(
                [
                    0.0,
                    0.0,
                    CUBE_GRASP_APPROACH_HEIGHT_M
                    + self.trajectory.cube_grasp_height_delta,
                ]
            )
        )
        grasp_rotation = grasp_rotation @ self.trajectory.cube_grasp_rotation

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
        cube_bottom = (
            cube_bottom
            + desired_cube_rotation @ self.trajectory.cube_place_local_offset
        )
        desired_cube_rotation = (
            desired_cube_rotation @ self.trajectory.cube_place_rotation
        )
        cube_place_target = (
            cube_bottom
            + self.model.geom_size[cube_id, 2] * desired_cube_rotation[:, 2]
        )

        drawer_target, drawer_rotation = _drawer_hand_target_pose(
            self.model, self.data, self.drawer_index
        )
        drawer_close_target, drawer_close_rotation = _drawer_close_target_pose(
            self.model, self.data, self.drawer_index
        )
        self.ik_targets = {
            "cube_grasp": (cube_target, grasp_rotation),
            "drawer_open": (drawer_target.copy(), drawer_rotation.copy()),
            "cube_place": (cube_place_target, desired_cube_rotation),
            "drawer_close": (drawer_close_target, drawer_close_rotation),
        }
        self._refresh_drawer_targets()
        if self.backend.config.debug:
            for marker_body, (position, rotation) in zip(
                IK_TARGET_MARKERS, self.ik_targets.values(), strict=True
            ):
                self._set_ik_target(marker_body, position, rotation)
            mujoco.mj_forward(self.model, self.data)
            self.backend.sync_viewer()

    def _select_post_drawer_hands_and_targets(self) -> None:
        """Select live hands and refresh only the cube-grasp target.

        The cube-placement frame is created once during reset from the planned
        open-drawer pose and intentionally remains fixed for the whole episode.
        """

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
        cube_target = (
            cube_target
            + cube_grasp_rotation @ self.trajectory.cube_grasp_local_offset
            + np.array(
                [
                    0.0,
                    0.0,
                    CUBE_GRASP_APPROACH_HEIGHT_M
                    + self.trajectory.cube_grasp_height_delta,
                ]
            )
        )
        cube_grasp_rotation = (
            cube_grasp_rotation @ self.trajectory.cube_grasp_rotation
        )
        self.ik_targets["cube_grasp"] = (cube_target, cube_grasp_rotation)

        if self.backend.config.debug:
            self._set_ik_target("hand_tip_marker", cube_target, cube_grasp_rotation)
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
        duration_jitter = self._motion_rng.uniform(
            1.0 - MOTION_DURATION_NOISE_FRACTION * self.trajectory_randomization_scale,
            1.0 + MOTION_DURATION_NOISE_FRACTION * self.trajectory_randomization_scale,
        )
        frames = max(
            2,
            round(
                self.move_duration_s
                * duration_scale
                * duration_jitter
                * self.backend.config.fps
            ),
        )
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
        return dict(zip(solution.joint_names, solution.joint_values, strict=True))

    def _rest_targets(self, side: Side) -> dict[str, float]:
        """Return the exact horizontal arm pose captured during initialization."""

        return {name: self.initial_targets[name] for name in SIDE_JOINTS[side]}

    def _handoff_donor_rest_targets(self, side: Side) -> dict[str, float]:
        """Return a raised post-handoff pose without moving the shared CNC axes."""

        targets = self._rest_targets(side)
        arm_joint = SIDE_JOINTS[side][2]
        arm_id = _joint_id(self.model, arm_joint)
        elevation = (
            HANDOFF_DONOR_ARM_ELEVATION_RAD
            + self.trajectory.handoff_donor_elevation_delta
        ) * (1.0 if side == "l" else -1.0)
        targets[arm_joint] = float(
            np.clip(elevation, *self.model.jnt_range[arm_id])
        )
        return targets

    def _initial_view_targets(self) -> dict[str, float]:
        """Return the full pre-first-IK pose used to view the storage bin."""

        targets = self.initial_targets.copy()
        common_targets = (
            (COMMON_JOINTS[0], 1.0),
            (
                COMMON_JOINTS[1],
                0.5 + self.trajectory.initial_lateral_fraction_delta,
            ),
            (
                COMMON_JOINTS[2],
                0.5 + self.trajectory.initial_vertical_fraction_delta,
            ),
        )
        for joint_name, normalized_position in common_targets:
            joint_id = _joint_id(self.model, joint_name)
            low, high = self.model.jnt_range[joint_id]
            targets[joint_name] = float(
                low + normalized_position * (high - low)
            )
        return targets

    def _dispatch_phase(self) -> None:
        base = "base_link_base_cnc_x_joint"
        lateral = "cnc_x_link_cnc_x_cnc_y_joint"
        vertical = "cnc_y_link_cnc_y_head_joint"
        base_id = _joint_id(self.model, base)
        lateral_id = _joint_id(self.model, lateral)
        vertical_id = _joint_id(self.model, vertical)

        if self.phase == "initialize_cnc":
            self._start_motion(
                self._initial_view_targets(),
                after="open_before_first_ik",
                fingers={"l": FINGER_CLOSED, "r": FINGER_CLOSED},
            )
        elif self.phase == "open_before_first_ik":
            self._start_motion(
                self._initial_view_targets(),
                after="drawer_ik",
                duration_scale=0.4,
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
                        - CUBE_GRASP_APPROACH_HEIGHT_M
                        - self.trajectory.cube_grasp_height_delta,
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
                {},
                after="cube_clearance",
                fingers={self.cube_hand: self.trajectory.cube_grasp_position},
            )
        elif self.phase == "cube_clearance":
            assert self.cube_hand is not None
            if self.cube_approach_vertical_qpos is None:
                raise RuntimeError("Cube approach height was not captured before grasping")
            self._start_motion(
                {vertical: self.cube_approach_vertical_qpos},
                after="return_without_base",
                fingers={self.cube_hand: self.trajectory.cube_grasp_position},
            )
        elif self.phase == "return_without_base":
            assert self.cube_hand is not None
            if self.cube_initial_z is None:
                raise RuntimeError("Initial cube height was not captured")
            cube_z = float(_cube_position(self.model, self.data)[2])
            if cube_z < self.cube_initial_z + CUBE_LIFT_CHECK_M:
                self._fail(
                    "cube grasp",
                    f"cube was not lifted (initial z={self.cube_initial_z:.4f} m, "
                    f"current z={cube_z:.4f} m)",
                )
                return
            targets = {name: value for name, value in self.initial_targets.items() if name != base}
            self._start_motion(
                targets,
                after="lift_cube",
                fingers={self.cube_hand: self.trajectory.cube_grasp_position},
            )
        elif self.phase == "lift_cube":
            assert self.cube_hand is not None
            arm = (
                f"forearm_{self.cube_hand}_link_forearm_{self.cube_hand}_"
                f"arm_{self.cube_hand}_joint"
            )
            wrist = f"arm_{self.cube_hand}_link_arm_{self.cube_hand}_wrist_{self.cube_hand}_joint"
            arm_id, wrist_id = _joint_id(self.model, arm), _joint_id(self.model, wrist)
            arm_lift = 0.5 + self.trajectory.post_grasp_arm_delta
            arm_target = _joint_qpos(self.model, self.data, arm) + (
                arm_lift if self.cube_hand == "l" else -arm_lift
            )
            wrist_target = (
                _joint_qpos(self.model, self.data, wrist)
                - 0.25
                - self.trajectory.post_grasp_wrist_delta
            )
            targets = {
                arm: float(np.clip(arm_target, *self.model.jnt_range[arm_id])),
                wrist: float(np.clip(wrist_target, *self.model.jnt_range[wrist_id])),
            }
            self._start_motion(
                targets,
                after=(
                    "cube_above_drawer"
                    if self.cube_hand == self.placement_hand
                    else "handoff_donor_ik"
                ),
                fingers={self.cube_hand: self.trajectory.cube_grasp_position},
            )
        elif self.phase == "handoff_donor_ik":
            assert self.cube_hand is not None and self.placement_hand is not None
            donor = self.cube_hand
            receiver = self.placement_hand
            donor_position, donor_rotation = hand_pose(self.model, self.data, donor)
            receiver_position, _ = hand_pose(self.model, self.data, receiver)
            self.handoff_target = (
                0.5 * (donor_position + receiver_position)
                + self.trajectory.handoff_world_offset
            )
            targets = self._solve_to(
                side=donor,
                target=self.handoff_target,
                rotation=donor_rotation,
                seed_offset=2,
                joint_names=SIDE_JOINTS[donor],
                reset_drawers=False,
                align_blue_axis=True,
                directed_axes=True,
                position_weight=100_000.0,
                orientation_weight=10.0,
                finger_position=self.trajectory.cube_grasp_position,
                protect_drawers=True,
                protect_other_hand=True,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="handoff_receiver_ik",
                fingers={
                    donor: self.trajectory.cube_grasp_position,
                    receiver: FINGER_OPEN,
                },
            )
        elif self.phase == "handoff_receiver_ik":
            assert self.cube_hand is not None and self.placement_hand is not None
            if self.cube_hand == self.placement_hand:
                self._start_motion(
                    {},
                    after="cube_above_drawer",
                    fingers={self.cube_hand: self.trajectory.cube_grasp_position},
                )
                return
            receiver = self.placement_hand
            target, rotation = _cube_grasp_target(
                self.model, self.data, receiver
            )
            target = target + rotation @ self.trajectory.cube_grasp_local_offset
            rotation = rotation @ self.trajectory.cube_grasp_rotation
            targets = self._solve_to(
                side=receiver,
                target=target,
                rotation=rotation,
                seed_offset=3,
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
                fingers={
                    self.cube_hand: self.trajectory.cube_grasp_position,
                    receiver: FINGER_OPEN,
                },
            )
        elif self.phase == "handoff_grasp":
            assert self.cube_hand is not None and self.placement_hand is not None
            self._start_motion(
                {},
                after="handoff_verify",
                fingers={
                    self.cube_hand: self.trajectory.cube_grasp_position,
                    self.placement_hand: self.trajectory.cube_grasp_position,
                },
            )
        elif self.phase == "handoff_verify":
            assert self.cube_hand is not None and self.placement_hand is not None
            ready, distance = _cube_near_hand(
                self.model, self.data, self.placement_hand
            )
            if not ready:
                self._fail(
                    "cube handoff",
                    f"cube is {distance:.4f} m from the receiving hand; required "
                    f"at most {CUBE_HAND_DISTANCE_M:.4f} m",
                )
                return
            self._start_motion(
                {},
                after="handoff_release",
                fingers={
                    self.cube_hand: self.trajectory.cube_grasp_position,
                    self.placement_hand: self.trajectory.cube_grasp_position,
                },
            )
        elif self.phase == "handoff_release":
            assert self.cube_hand is not None and self.placement_hand is not None
            donor = self.cube_hand
            receiver = self.placement_hand
            self.handoff_donor = donor
            self.cube_hand = receiver
            self._start_motion(
                {},
                after="handoff_donor_rest",
                fingers={
                    donor: FINGER_OPEN,
                    receiver: self.trajectory.cube_grasp_position,
                },
            )
        elif self.phase == "handoff_donor_rest":
            assert (
                self.handoff_donor is not None
                and self.cube_hand is not None
                and self.handoff_donor != self.cube_hand
            )
            self._start_motion(
                self._handoff_donor_rest_targets(self.handoff_donor),
                after="cube_above_drawer",
                fingers={
                    self.handoff_donor: FINGER_CLOSED,
                    self.cube_hand: self.trajectory.cube_grasp_position,
                },
            )
        elif self.phase == "drawer_ik":
            assert self.drawer_hand is not None
            target, rotation = self.ik_targets["drawer_open"]
            gripper_rotation = _drawer_gripper_rotation(rotation, self.drawer_hand)
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=gripper_rotation,
                seed_offset=1,
                reset_drawers=False,
                align_blue_axis=True,
                directed_axes=True,
                position_weight=40_000.0,
                orientation_weight=50.0,
                finger_position=FINGER_OPEN,
                protect_other_hand=True,
                joint_bounds={
                    base: (
                        float(self.model.jnt_range[base_id, 0])
                        + DRAWER_APPROACH_DISTANCE_M
                        + self.trajectory.drawer_open_approach_delta,
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
                + float(handle_position[1] - hand_position[1])
                + self.trajectory.drawer_contact_depth_delta,
                *self.model.jnt_range[base_id],
            )
            self._start_motion(
                {base: float(target)},
                after="grasp_drawer",
                fingers={self.drawer_hand: FINGER_OPEN},
            )
        elif self.phase == "grasp_drawer":
            assert self.drawer_hand is not None
            self._start_motion(
                {},
                after="pull_drawer",
                fingers={self.drawer_hand: self.trajectory.drawer_grasp_position},
            )
        elif self.phase == "pull_drawer":
            assert self.drawer_hand is not None
            idle_hand: Side = "r" if self.drawer_hand == "l" else "l"
            idle_forearm = SIDE_JOINTS[idle_hand][1]
            self._start_motion(
                {
                    base: float(
                        self.model.jnt_range[base_id, 1]
                        - self.trajectory.drawer_pull_shortfall
                    ),
                    idle_forearm: (
                        IDLE_FOREARM_DRAWER_PULL_POSITION
                        + self.trajectory.idle_forearm_delta
                    ),
                },
                after="release_open_drawer",
                fingers={self.drawer_hand: self.trajectory.drawer_grasp_position},
            )
        elif self.phase == "release_open_drawer":
            assert self.drawer_hand is not None
            opening = self._drawer_opening()
            if opening < MIN_DRAWER_OPENING_M:
                self._fail(
                    "drawer opening",
                    f"drawer opened {opening:.4f} m; required at least "
                    f"{MIN_DRAWER_OPENING_M:.4f} m",
                )
                return
            self._start_motion(
                {},
                after="drawer_hand_rest",
                duration_scale=0.4,
                fingers={self.drawer_hand: FINGER_OPEN},
            )
        elif self.phase == "drawer_hand_rest":
            assert self.drawer_hand is not None and self.cube_hand is not None
            self._start_motion(
                self._initial_view_targets(),
                after="view_cube",
                fingers={"l": FINGER_OPEN, "r": FINGER_OPEN},
            )
        elif self.phase == "view_cube":
            self._start_motion(
                self._initial_view_targets(),
                after="cube_ik",
                duration_scale=0.5,
                fingers={"l": FINGER_OPEN, "r": FINGER_OPEN},
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
                directed_axes=True,
                position_weight=25_000.0,
                orientation_weight=20.0,
                finger_position=self.trajectory.cube_grasp_position,
                protect_drawers=True,
                protect_other_hand=True,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="drop_cube",
                fingers={self.cube_hand: self.trajectory.cube_grasp_position},
            )
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
            mujoco.mj_forward(self.model, self.data)
            if not _cube_center_inside_drawer(
                self.model, self.data, self.drawer_index
            ):
                self._fail(
                    "cube release",
                    f"cube is not fully inside drawer {self.drawer_index}",
                )
                return
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
                _joint_qpos(self.model, self.data, base)
                + 0.06
                + self.trajectory.pre_close_clearance_delta,
                self.model.jnt_range[base_id, 1],
            )
            self._start_motion(
                {base: float(target)},
                after="choose_closing_hand",
                fingers={"l": FINGER_CLOSED, "r": FINGER_CLOSED},
            )
        elif self.phase == "choose_closing_hand":
            target, _ = self.ik_targets["drawer_close"]
            if _drawer_is_center(self.drawer_index):
                closing_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [self.seed, self.episode_seed, self.drawer_index, 4]
                    )
                )
                self.drawer_hand = ("l", "r")[int(closing_rng.integers(0, 2))]
            else:
                self.drawer_hand = _closest_hand(self.model, self.data, target)
            self._start_motion(
                {}, after="close_drawer_ik", fingers={self.drawer_hand: FINGER_CLOSED}
            )
        elif self.phase == "close_drawer_ik":
            assert self.drawer_hand is not None
            target, rotation = self.ik_targets["drawer_close"]
            gripper_rotation = _drawer_gripper_rotation(rotation, self.drawer_hand)
            drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
            drawer_id = _joint_id(self.model, drawer_joint)
            drawer_travel = float(np.ptp(self.model.jnt_range[drawer_id]))
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=gripper_rotation,
                seed_offset=4,
                reset_drawers=False,
                align_blue_axis=True,
                directed_axes=True,
                position_weight=40_000.0,
                orientation_weight=10.0,
                finger_position=FINGER_CLOSED,
                protect_other_hand=True,
                joint_bounds={
                    base: (
                        float(self.model.jnt_range[base_id, 0])
                        + DRAWER_CLOSE_APPROACH_DISTANCE_M
                        + self.trajectory.drawer_close_approach_delta
                        + drawer_travel
                        + DRAWER_PUSH_MARGIN_M
                        + self.trajectory.drawer_push_margin_delta,
                        float(self.model.jnt_range[base_id, 1]),
                    )
                },
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="push_drawer",
                fingers={self.drawer_hand: FINGER_CLOSED},
            )
        elif self.phase == "push_drawer":
            assert self.drawer_hand is not None
            self.drawer_close_cnc_start = _joint_qpos(self.model, self.data, base)
            drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
            drawer_id = _joint_id(self.model, drawer_joint)
            delta = _joint_qpos(self.model, self.data, drawer_joint) - float(
                self.model.jnt_range[drawer_id, 1]
            )
            target = np.clip(
                self.drawer_close_cnc_start
                - DRAWER_CLOSE_APPROACH_DISTANCE_M
                - self.trajectory.drawer_close_approach_delta
                + delta
                - DRAWER_PUSH_MARGIN_M
                - self.trajectory.drawer_push_margin_delta,
                *self.model.jnt_range[base_id],
            )
            self._start_motion(
                {base: float(target)},
                after="release_closed_drawer",
                fingers={self.drawer_hand: FINGER_CLOSED},
            )
        elif self.phase == "release_closed_drawer":
            assert self.drawer_hand is not None
            opening = self._drawer_opening()
            if opening > MAX_DRAWER_CLOSED_OPENING_M:
                self._fail(
                    "drawer closing",
                    f"drawer remains open by {opening:.4f} m; permitted at most "
                    f"{MAX_DRAWER_CLOSED_OPENING_M:.4f} m",
                )
                return
            if self.drawer_close_cnc_start is None:
                raise RuntimeError("Drawer closing CNC-X start position was not saved")
            self._start_motion(
                {base: self.drawer_close_cnc_start},
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
            opening = self._drawer_opening()
            inside = _cube_center_inside_drawer(self.model, self.data, self.drawer_index)
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
