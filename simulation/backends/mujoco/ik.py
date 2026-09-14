"""Physical scripted demonstration for the complete cube-and-drawer task.

The controller is intentionally limited to MuJoCo-specific task generation.  Dataset
serialization and episode management remain in :mod:`hepha_lerobot.recording`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
from hepha_lerobot.workspaces import TaskPhase, Workspace
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation

from . import episode as episode_utils
from .backend import ACTUATOR_NAMES, MujocoBackend
from .workspaces import CNC_JOINTS, resolve_mujoco_workspace, set_workspace_qpos

CUBE_SPAWN_RADIUS_M = episode_utils.CUBE_SPAWN_RADIUS_M
IK_TARGET_MARKERS = episode_utils.IK_TARGET_MARKERS
_cube_position = episode_utils.cube_position
_cube_spawn_quadrant = episode_utils.cube_spawn_quadrant
_randomize_cube = episode_utils.randomize_cube
initialize_task_episode = episode_utils.initialize_task_episode

Side = Literal["l", "r"]

COMMON_JOINTS = CNC_JOINTS
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
# The finger actuator is angular. In the current gripper geometry, this leaves
# an approximately 32 mm inner gap around the 30 mm foam cube.
CUBE_GRASP = 0.185
CUBE_GRASP_SETTLE_DURATION_S = 0.30
CUBE_DRAWER_SETTLE_DURATION_S = 1.0
CUBE_DROP_ASSIST_FALL_SPEED_M_S = 0.10
CUBE_GRASP_LOCK_EQUALITIES: dict[Side, str] = {
    "l": "cube_grasp_lock_l",
    "r": "cube_grasp_lock_r",
}
CUBE_GRIPPER_CONTACT_PAIRS = (
    "cube_hand_l_pad",
    "cube_finger_l_shaft",
    "cube_finger_l_tip",
    "cube_hand_r_pad",
    "cube_finger_r_shaft",
    "cube_finger_r_tip",
)
CUBE_GRIPPER_GRASP_FRICTION = (1.5, 1.5, 0.03, 0.001, 0.001)
CUBE_GRIPPER_RELEASE_FRICTION = (0.0, 0.0, 0.0, 0.0, 0.0)
# In the rendered hand-frame convention, the positive signed offset places the
# task origin 30 mm toward the wrist/arm from the fingertip midpoint.
HAND_TASK_FRAME_INSET_M = 0.030
# The complete storage-bin pickup uses the physical midpoint between the two
# fingertips, matching the drawer-handle grasp convention.
STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M = 0.0
DRAWER_OPEN_HAND_FRAME_INSET_M = 0.0
HANDOFF_HAND_FRAME_INSET_M = 0.0
DRAWER_FINGER_OPEN = 0.4
DRAWER_GRASP = 0.05
IDLE_FOREARM_DRAWER_PULL_POSITION = -0.25
HANDOFF_DONOR_ARM_ELEVATION_RAD = 0.5
CONTACT_MARGIN_M = 0.005
MIN_DRAWER_OPENING_M = 0.020
MAX_DRAWER_CLOSED_OPENING_M = 0.010
CUBE_LIFT_CHECK_M = 0.020
CUBE_HAND_DISTANCE_M = 0.050
HANDOFF_OVERLAP_HOLD_DURATION_SCALE = 2.0
HANDOFF_RECEIVER_GRASP = CUBE_GRASP
HANDOFF_RECEIVER_WIDE_OPEN = float(np.pi / 2.0)
HANDOFF_RECEIVER_APPROACH_DISTANCE_M = 0.08
HANDOFF_DONOR_RETREAT_DISTANCE_M = 0.08
# Lift the fixed handoff point above the nominal shared-workspace midpoint so
# both arms and the carried cube clear the storage-bin walls.
HANDOFF_WORKSPACE_HEIGHT_OFFSET_M = 0.05
HANDOFF_RECEIVER_VERTICAL_OFFSET_M = 0.0
HANDOFF_HAND_CLEARANCE_M = 0.008
HANDOFF_RECEIVER_APPROACH_POSITION_WEIGHT = 2_000_000.0
HANDOFF_RECEIVER_APPROACH_RED_ORIENTATION_WEIGHT = 30_000.0
HANDOFF_RECEIVER_APPROACH_BLUE_ORIENTATION_WEIGHT = 30_000.0
HANDOFF_RECEIVER_TRANSLATION_CONTINUITY_WEIGHT = 600.0
# The second receiver IK must put the fingertip midpoint at the cube center;
# make Cartesian coincidence dominate orientation and posture regularization.
HANDOFF_RECEIVER_TRANSLATION_POSITION_WEIGHT = 20_000_000.0
HANDOFF_RECEIVER_TRANSLATION_RED_ORIENTATION_WEIGHT = 30_000.0
HANDOFF_RECEIVER_TRANSLATION_BLUE_ORIENTATION_WEIGHT = 3_000.0
HANDOFF_CONTACT_ABORT_FRAMES = 5
HANDOFF_CONTACT_ABORT_PENETRATION_M = 0.001
HANDOFF_CLEARANCE_PHASES = frozenset(
    {
        "handoff_grasp",
        "handoff_verify",
        "handoff_load_transfer",
        "handoff_release",
    }
)
CUBE_GRASP_APPROACH_HEIGHT_M = 0.13
WORKSPACE_TRANSIT_BIN_CLEARANCE_M = 0.01
WORKSPACE_TRANSIT_SHOULDER_LIFT_RADIANS = 0.30
DRAWER_APPROACH_DISTANCE_M = 0.07
DRAWER_CLOSE_APPROACH_DISTANCE_M = 0.05
DRAWER_TARGET_Z_OFFSET_M = -0.01
DRAWER_PUSH_MARGIN_M = 0.03
DRAWER_PULL_OVERSHOOT_M = 0.02
CUBE_PLACE_HANDLE_Y_OFFSET_M = -0.05
CUBE_PLACE_HANDLE_Z_OFFSET_M = 0.06
TOP_ROW_CUBE_PLACE_CLEARANCE_M = 0.04
DRAWER_IK_POSITION_WEIGHT = 50_000.0
DRAWER_OPEN_IK_POSITION_WEIGHT = 100_000.0
DRAWER_IK_ORIENTATION_WEIGHT = 100.0
CUBE_GRASP_IK_POSITION_WEIGHT = 1_000_000.0
CUBE_GRASP_IK_ORIENTATION_WEIGHT = 1_500.0
CUBE_PLACE_IK_ORIENTATION_WEIGHT = 600.0
CUBE_PLACE_RETREAT_DISTANCE_M = 0.10
CUBE_RETREAT_IK_POSITION_WEIGHT = 500_000.0
DRAWER_OPEN_RETREAT_EXTRA_DISTANCE_M = 0.02
DRAWER_CLOSE_RETREAT_VERTICAL_DISTANCE_M = 0.02
DRAWER_RETREAT_IK_POSITION_WEIGHT = 500_000.0
# Keep solutions near the fixed user-selected posture and discourage abrupt
# branch changes from the arm's current pose. Errors are normalized by each
# joint's full MuJoCo range before they are squared.
DEFAULT_IK_POSTURE_WEIGHT = 300.0
DEFAULT_IK_PREVIOUS_POSTURE_WEIGHT = 30.0
DEFAULT_IK_MAX_POSTURE_DEVIATION_RADIANS = float(np.deg2rad(75.0))
# Cartesian errors are measured in metres and would otherwise be numerically
# small beside orientation and normalized-posture costs. Apply this shared
# multiplier inside the solver so every IK phase prioritizes its target point.
IK_POSITION_WEIGHT_MULTIPLIER = 10.0

# Conservative episode-level trajectory augmentation. These are amplitudes, so
# a value such as 0.003 means a uniform sample in [-3 mm, 3 mm]. The cube-grasp
# height is the one-sided exception. Safety checks, collision margins, and joint
# limits are intentionally not randomized.
CUBE_GRASP_DEPTH_NOISE_M = 0.001
CUBE_GRASP_LATERAL_NOISE_M = 0.003
CUBE_GRASP_HEIGHT_NOISE_M = 0.010
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
HANDOFF_DONOR_ELEVATION_NOISE_RAD = 0.04
IDLE_FOREARM_NOISE_RAD = 0.03
INITIAL_VIEW_CENTER_NOISE_FRACTION = 0.01
PRE_CLOSE_CLEARANCE_NOISE_M = 0.005
DRAWER_PUSH_MARGIN_NOISE_M = 0.004
CUBE_GRASP_POSITION_NOISE = 0.01
DRAWER_GRASP_POSITION_NOISE = 0.005
MOTION_DURATION_NOISE_FRACTION = 0.15
PERTURBATION_PHASES = (
    "drawer_ik",
    "cube_ik",
    "cube_above_drawer",
    "close_drawer_ik",
)
PERTURBATION_TRIGGER_RANGE = (0.30, 0.70)
PERTURBATION_DURATION_S = 0.60
PERTURBATION_RECOVERY_DURATION_S = 1.20
PERTURBATION_GANTRY_RANGE = (0.010, 0.020)
PERTURBATION_ARM_RANGE = (0.040, 0.080)
TASK_PHASE_COUNT = len(TaskPhase)
TASK_PHASE_TRANSITION_WINDOW_FRAMES = 5
TASK_PHASE_TRANSITION_AFTER = {
    1: "move_to_b_for_open",
    2: "move_to_a_for_pick",
    3: "move_to_b_for_place",
    4: "move_to_a_final",
}


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
    """Task IK controls an arm only; the CNC is owned by the workspace coordinator."""
    return SIDE_JOINTS[side]


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


def _rotation_about_y(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))


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
        rng = np.random.default_rng(np.random.SeedSequence([seed, episode_seed, 0x48455048]))

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
            cube_grasp_height_delta=float(rng.uniform(0.0, CUBE_GRASP_HEIGHT_NOISE_M * scale)),
            drawer_handle_lateral_offset=float(symmetric(DRAWER_HANDLE_LATERAL_NOISE_M)),
            drawer_open_approach_delta=float(symmetric(DRAWER_APPROACH_NOISE_M)),
            drawer_close_approach_delta=float(symmetric(DRAWER_APPROACH_NOISE_M)),
            drawer_contact_depth_delta=float(symmetric(DRAWER_CONTACT_DEPTH_NOISE_M)),
            drawer_target_z_delta=float(symmetric(DRAWER_TARGET_Z_NOISE_M)),
            drawer_rotation=_rotation_from_rpy_degrees(drawer_angles),
            drawer_pull_shortfall=float(rng.uniform(0.0, DRAWER_PULL_SHORTFALL_NOISE_M * scale)),
            cube_place_local_offset=np.array(
                [
                    symmetric(CUBE_PLACE_LATERAL_NOISE_M),
                    symmetric(CUBE_PLACE_DEPTH_NOISE_M),
                    rng.uniform(0.0, CUBE_PLACE_UPWARD_NOISE_M * scale),
                ],
                dtype=float,
            ),
            cube_place_rotation=_rotation_from_rpy_degrees(placement_angles),
            handoff_donor_elevation_delta=float(symmetric(HANDOFF_DONOR_ELEVATION_NOISE_RAD)),
            idle_forearm_delta=float(symmetric(IDLE_FOREARM_NOISE_RAD)),
            initial_lateral_fraction_delta=float(symmetric(INITIAL_VIEW_CENTER_NOISE_FRACTION)),
            initial_vertical_fraction_delta=float(symmetric(INITIAL_VIEW_CENTER_NOISE_FRACTION)),
            pre_close_clearance_delta=float(symmetric(PRE_CLOSE_CLEARANCE_NOISE_M)),
            drawer_push_margin_delta=float(symmetric(DRAWER_PUSH_MARGIN_NOISE_M)),
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
    geom_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"hand_{side}_link_collision_box_03_geom")
    rotation = _matrix_from_quaternion(model.geom_quat[geom_id])
    local_tip = model.geom_pos[geom_id] - rotation[:, 1] * model.geom_size[geom_id, 1]
    local_frame = _frame_from_yz(rotation[:, 2], -rotation[:, 1])
    return body_id, local_tip, local_frame


def _moving_finger_tip(model: mujoco.MjModel, side: Side) -> tuple[int, np.ndarray]:
    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, f"finger_{side}_link")
    geom_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"finger_{side}_link_collision_box_02_geom"
    )
    cap_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"finger_{side}_link_collision_box_03_geom")
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
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: Side,
    *,
    task_frame_inset_m: float = HAND_TASK_FRAME_INSET_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical hand frame, 3 cm palmward from the fingertips."""

    fixed_body, fixed_tip, fixed_frame = _fixed_finger_tip(model, side)
    moving_body, moving_tip = _moving_finger_tip(model, side)
    fingertip_midpoint = 0.5 * (
        _world_point(data, fixed_body, fixed_tip) + _world_point(data, moving_body, moving_tip)
    )
    rotation = data.xmat[fixed_body].reshape(3, 3) @ fixed_frame @ _rotation_about_x(-90.0)
    task_position = fingertip_midpoint + task_frame_inset_m * rotation[:, 1]
    return task_position, rotation


def grasp_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: Side,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the palm-inset coordinate frame used for object manipulation."""

    return hand_pose(
        model,
        data,
        side,
        task_frame_inset_m=HAND_TASK_FRAME_INSET_M,
    )


def _minimum_fingertip_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Return the lowest world-Z coordinate among all four fingertips."""

    heights = []
    for side in ("l", "r"):
        fixed_body, fixed_tip, _ = _fixed_finger_tip(model, side)
        moving_body, moving_tip = _moving_finger_tip(model, side)
        heights.extend(
            (
                float(_world_point(data, fixed_body, fixed_tip)[2]),
                float(_world_point(data, moving_body, moving_tip)[2]),
            )
        )
    return min(heights)


def _storage_bin_top_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Read the storage-bin wall height from its collision boxes."""

    tops = []
    for box_index in range(1, 6):
        geom_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"storage_bin_link_collision_box_{box_index:02d}_geom",
        )
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        half_height = float(np.abs(rotation[2]) @ model.geom_size[geom_id])
        tops.append(float(data.geom_xpos[geom_id, 2]) + half_height)
    return max(tops)


def _closest_hand(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    *,
    task_frame_inset_m: float = HAND_TASK_FRAME_INSET_M,
) -> Side:
    """Select the hand using the same task-frame origin as the upcoming IK."""

    distances = {
        side: float(
            np.linalg.norm(
                hand_pose(
                    model,
                    data,
                    side,
                    task_frame_inset_m=task_frame_inset_m,
                )[0]
                - target
            )
        )
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
    cube_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom")
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


@dataclass(frozen=True)
class _HandoffReceiverPlan:
    """Cube-frame receiver geometry for a collision-separated handoff."""

    donor_face_axis: int
    receiver_face_axis: int
    receiver_face: np.ndarray
    approach_direction: np.ndarray
    approach_target: np.ndarray
    contact_target: np.ndarray
    approach_rotation: np.ndarray
    contact_rotation: np.ndarray


def _handoff_receiver_plan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    donor: Side,
    receiver: Side,
) -> _HandoffReceiverPlan:
    """Choose the cube-side axis perpendicular to the donor's grasp axis."""

    if donor == receiver:
        raise ValueError("Handoff donor and receiver must be different arms")
    cube_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom")
    cube_center = data.geom_xpos[cube_id].copy()
    cube_rotation = data.geom_xmat[cube_id].reshape(3, 3)
    _, donor_rotation = hand_pose(
        model,
        data,
        donor,
        task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
    )
    # The donor's red axis identifies the occupied pair among the cube's X/Y
    # side faces. The receiver must use the other, perpendicular side-face pair.
    donor_face_axis = max(
        (0, 1),
        key=lambda axis: abs(float(donor_rotation[:, 0] @ cube_rotation[:, axis])),
    )
    receiver_face_axis = 1 - donor_face_axis
    receiver_position, receiver_current_rotation = hand_pose(
        model,
        data,
        receiver,
        task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
    )
    cube_blue_axis = cube_rotation[:, 2]
    receiver_faces = (
        -cube_rotation[:, receiver_face_axis],
        cube_rotation[:, receiver_face_axis],
    )
    receiver_direction = receiver_position - cube_center
    receiver_face = max(
        receiver_faces,
        key=lambda face: float(receiver_direction @ face),
    )
    # Receiver red must be +/- cube blue, and receiver blue points along the
    # selected face normal. Pick the red-axis sign requiring the least rotation.
    contact_rotation = min(
        (
            _frame_from_xz(cube_blue_axis, receiver_face),
            _frame_from_xz(-cube_blue_axis, receiver_face),
        ),
        key=lambda candidate: Rotation.from_matrix(
            receiver_current_rotation.T @ candidate
        ).magnitude(),
    )
    # Translation into contact is deliberately horizontal.  Orientation uses
    # the true (possibly tilted) cube face above; motion uses only its XY
    # projection so the second IK is a straight inward Cartesian step.
    approach_direction = receiver_face.copy()
    approach_direction[2] = 0.0
    approach_direction = _normalize(approach_direction)
    # Both receiver IKs use this final grasp orientation.
    approach_rotation = contact_rotation.copy()
    # Keep the receiver slightly above the donor's grasp layer.  The outside
    # target differs from this contact target only along the horizontal
    # approach direction.
    contact_target = cube_center + HANDOFF_RECEIVER_VERTICAL_OFFSET_M * cube_blue_axis
    return _HandoffReceiverPlan(
        donor_face_axis=donor_face_axis,
        receiver_face_axis=receiver_face_axis,
        receiver_face=receiver_face.copy(),
        approach_direction=approach_direction.copy(),
        approach_target=(
            contact_target
            + HANDOFF_RECEIVER_APPROACH_DISTANCE_M * approach_direction
        ),
        contact_target=contact_target,
        approach_rotation=approach_rotation,
        contact_rotation=contact_rotation,
    )


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
    model: mujoco.MjModel,
    data: mujoco.MjData,
    drawer_index: int,
    side: Side,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a live target 7 cm in front of the drawer-handle center."""

    handle_center, drawer_rotation = drawer_handle_pose(model, data, drawer_index)
    hand_rotation = _drawer_gripper_rotation(drawer_rotation, side)
    drawer_joint = _joint_id(model, f"base_link_base_drawer_{drawer_index}_joint")
    inward_axis = _normalize(data.xaxis[drawer_joint])
    target = handle_center - inward_axis * DRAWER_APPROACH_DISTANCE_M
    target[2] += DRAWER_TARGET_Z_OFFSET_M
    return target, hand_rotation


def _drawer_close_target_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    drawer_index: int,
    side: Side,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a live closing target 5 cm in front of the handle center."""

    handle_center, drawer_rotation = drawer_handle_pose(model, data, drawer_index)
    hand_rotation = _drawer_gripper_rotation(drawer_rotation, side)
    drawer_joint = _joint_id(model, f"base_link_base_drawer_{drawer_index}_joint")
    inward_axis = _normalize(data.xaxis[drawer_joint])
    return (
        handle_center - inward_axis * DRAWER_CLOSE_APPROACH_DISTANCE_M,
        hand_rotation,
    )


def _closed_drawer_vertical_retreat(fingertip_z: float, handle_z: float) -> float:
    """Move vertically away from the closed handle before returning to rest."""

    return float(np.sign(fingertip_z - handle_z)) * (
        DRAWER_CLOSE_RETREAT_VERTICAL_DISTANCE_M
    )


def _drawer_gripper_rotation(drawer_rotation: np.ndarray, side: Side) -> np.ndarray:
    """Rotate the finger gap perpendicular to the horizontal drawer handle."""

    hand_angle = -90.0 if side == "l" else 90.0
    return drawer_rotation @ _rotation_about_y(hand_angle)


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


def _all_contact_penalty(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    margin: float,
) -> float:
    return sum(max(0.0, margin - float(contact.dist)) ** 2 for contact in data.contact)


def _pair_contact_penalty(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: Side,
    margin: float,
    pair: Literal["drawer", "hand"],
) -> float:
    other: Side = "l" if side == "r" else "r"

    def is_hand(body_id: int, hand_side: Side) -> bool:
        return _body_descends_from(model, body_id, f"hand_{hand_side}_link") or _body_descends_from(
            model, body_id, f"finger_{hand_side}_link"
        )

    penalty = 0.0
    for contact in data.contact:
        if contact.geom1 < 0 or contact.geom2 < 0:
            continue
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        moving1, moving2 = is_hand(body1, side), is_hand(body2, side)
        if pair == "drawer":
            paired1 = _body_name_starts_with(model, body1, "drawer_")
            paired2 = _body_name_starts_with(model, body2, "drawer_")
        else:
            paired1, paired2 = is_hand(body1, other), is_hand(body2, other)
        if (moving1 and paired2) or (moving2 and paired1):
            penalty += max(0.0, margin - float(contact.dist)) ** 2
    return penalty


@dataclass(frozen=True)
class IKSolution:
    joint_names: tuple[str, ...]
    joint_values: np.ndarray
    achieved_position: np.ndarray
    error_m: float
    orientation_error_deg: float
    red_orientation_error_deg: float = 0.0
    blue_orientation_error_deg: float = 0.0


class PositionOrientationIK:
    """Numerical arm IK with Cartesian tracking and a soft nominal-posture prior."""

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
        planning_qpos: np.ndarray | None = None,
        joint_names: tuple[str, ...] | None = None,
        joint_bounds: dict[str, tuple[float, float]] | None = None,
        finger_position: float = FINGER_OPEN,
        reset_drawers: bool = True,
        crossed_axes: bool = False,
        align_red_axis: bool = True,
        align_blue_axis: bool = False,
        directed_axes: bool = False,
        directed_red_axis: bool | None = None,
        directed_blue_axis: bool | None = None,
        position_weight: float = 1000.0,
        orientation_weight: float = 50.0,
        red_orientation_weight: float | None = None,
        blue_orientation_weight: float | None = None,
        include_collision_penalty: bool = True,
        hand_frame_inset_m: float = HAND_TASK_FRAME_INSET_M,
        posture_weight: float = DEFAULT_IK_POSTURE_WEIGHT,
        posture_reference: dict[str, float] | None = None,
        previous_posture_weight: float = DEFAULT_IK_PREVIOUS_POSTURE_WEIGHT,
        max_posture_deviation_radians: float | None = (DEFAULT_IK_MAX_POSTURE_DEVIATION_RADIANS),
        protect_drawers: bool = False,
        protect_other_hand: bool = False,
    ) -> IKSolution:
        model = self.model
        data = mujoco.MjData(model)
        if planning_qpos is None:
            data.qpos[:] = self.backend.data.qpos
        else:
            candidate_qpos = np.asarray(planning_qpos, dtype=float)
            if candidate_qpos.shape != data.qpos.shape:
                raise ValueError(
                    f"planning_qpos has shape {candidate_qpos.shape}; "
                    f"expected {data.qpos.shape}"
                )
            data.qpos[:] = candidate_qpos
        data.qvel[:] = 0.0
        names = joint_names or _controlled_joints(side)
        joint_ids = np.array([_joint_id(model, name) for name in names], dtype=int)
        qpos_ids = model.jnt_qposadr[joint_ids].astype(int)
        requested_bounds = [
            (joint_bounds or {}).get(name, tuple(model.jnt_range[joint_id]))
            for name, joint_id in zip(names, joint_ids, strict=True)
        ]
        for name, joint_id, (low, high) in zip(names, joint_ids, requested_bounds, strict=True):
            model_low, model_high = model.jnt_range[joint_id]
            if low < model_low or high > model_high or low > high:
                raise ValueError(
                    f"Invalid IK bounds for {name}: {(low, high)} outside "
                    f"{tuple(model.jnt_range[joint_id])}"
                )
        initial = data.qpos[qpos_ids].copy()
        reference = np.array(
            [
                initial[index]
                if posture_reference is None
                else posture_reference.get(name, initial[index])
                for index, name in enumerate(names)
            ]
        )
        if position_weight < 0.0:
            raise ValueError("IK position weight must be non-negative")
        if posture_weight < 0.0 or previous_posture_weight < 0.0:
            raise ValueError("IK posture weights must be non-negative")
        effective_position_weight = position_weight * IK_POSITION_WEIGHT_MULTIPLIER
        if max_posture_deviation_radians is not None:
            if max_posture_deviation_radians <= 0.0:
                raise ValueError("IK maximum posture deviation must be positive")
            bounds = [
                (
                    max(low, nominal - max_posture_deviation_radians),
                    min(high, nominal + max_posture_deviation_radians),
                )
                for (low, high), nominal in zip(requested_bounds, reference, strict=True)
            ]
            for name, (low, high) in zip(names, bounds, strict=True):
                if low > high:
                    raise ValueError(
                        f"IK posture window for {name} does not overlap its joint bounds"
                    )
        else:
            bounds = requested_bounds
        spans = np.array(
            [model.jnt_range[joint_id, 1] - model.jnt_range[joint_id, 0] for joint_id in joint_ids]
        )
        finger_id = _joint_id(model, FINGER_JOINTS[side])
        finger_qpos_id = int(model.jnt_qposadr[finger_id])
        red_is_directed = directed_axes if directed_red_axis is None else directed_red_axis
        blue_is_directed = directed_axes if directed_blue_axis is None else directed_blue_axis
        red_weight = (
            orientation_weight if red_orientation_weight is None else red_orientation_weight
        )
        blue_weight = (
            orientation_weight if blue_orientation_weight is None else blue_orientation_weight
        )

        def prepare(candidate: np.ndarray) -> None:
            data.qpos[qpos_ids] = candidate
            data.qpos[finger_qpos_id] = np.clip(finger_position, *model.jnt_range[finger_id])
            if reset_drawers:
                _close_drawers(model, data)
            mujoco.mj_forward(model, data)

        def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            prepare(candidate)
            return hand_pose(
                model,
                data,
                side,
                task_frame_inset_m=hand_frame_inset_m,
            )

        def objective(candidate: np.ndarray) -> float:
            position, rotation = evaluate(candidate)
            position_error = position - target
            if crossed_axes:
                red_error = np.cross(rotation[:, 0], target_rotation[:, 2])
                blue_error = np.cross(rotation[:, 2], target_rotation[:, 0])
            else:
                red_error = (
                    (
                        rotation[:, 0] - target_rotation[:, 0]
                        if red_is_directed
                        else np.cross(rotation[:, 0], target_rotation[:, 0])
                    )
                    if align_red_axis
                    else np.zeros(3)
                )
                blue_error = (
                    (
                        rotation[:, 2] - target_rotation[:, 2]
                        if blue_is_directed
                        else np.cross(rotation[:, 2], target_rotation[:, 2])
                    )
                    if align_blue_axis
                    else np.zeros(3)
                )
            pose_cost = (
                effective_position_weight * float(position_error @ position_error)
                + red_weight * float(red_error @ red_error)
                + blue_weight * float(blue_error @ blue_error)
            )
            normalized_posture_error = (candidate - reference) / spans
            posture_cost = posture_weight * float(
                normalized_posture_error @ normalized_posture_error
            )
            normalized_previous_error = (candidate - initial) / spans
            continuity_cost = previous_posture_weight * float(
                normalized_previous_error @ normalized_previous_error
            )
            if not include_collision_penalty:
                return pose_cost + posture_cost + continuity_cost
            collision_cost = _all_contact_penalty(model, data, CONTACT_MARGIN_M)
            if protect_drawers:
                collision_cost += 100.0 * _pair_contact_penalty(model, data, side, 0.020, "drawer")
            if protect_other_hand:
                collision_cost += 100.0 * _pair_contact_penalty(
                    model,
                    data,
                    side,
                    HANDOFF_HAND_CLEARANCE_M,
                    "hand",
                )
            return pose_cost + 50_000_000.0 * collision_cost + posture_cost + continuity_cost

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
        position, rotation = evaluate(local_result.x)
        if crossed_axes:
            axis_pairs = (
                (rotation[:, 0], target_rotation[:, 2], directed_axes),
                (rotation[:, 2], target_rotation[:, 0], directed_axes),
            )
        else:
            axis_pairs = tuple(
                pair
                for enabled, pair in (
                    (
                        align_red_axis,
                        (rotation[:, 0], target_rotation[:, 0], red_is_directed),
                    ),
                    (
                        align_blue_axis,
                        (rotation[:, 2], target_rotation[:, 2], blue_is_directed),
                    ),
                )
                if enabled
            )
        axis_errors = tuple(
            float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(_normalize(actual) @ _normalize(desired))
                            if is_directed
                            else abs(float(_normalize(actual) @ _normalize(desired))),
                            -1.0 if is_directed else 0.0,
                            1.0,
                        )
                    )
                )
            )
            for actual, desired, is_directed in axis_pairs
        )
        orientation_error_deg = max(axis_errors)
        red_orientation_error_deg = axis_errors[0] if align_red_axis else 0.0
        blue_orientation_error_deg = (
            axis_errors[-1] if align_blue_axis else 0.0
        )
        return IKSolution(
            joint_names=names,
            joint_values=local_result.x.copy(),
            achieved_position=position.copy(),
            error_m=float(np.linalg.norm(position - target)),
            orientation_error_deg=orientation_error_deg,
            red_orientation_error_deg=red_orientation_error_deg,
            blue_orientation_error_deg=blue_orientation_error_deg,
        )


def _close_drawers(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for drawer_index in range(1, 10):
        joint_id = _joint_id(model, f"base_link_base_drawer_{drawer_index}_joint")
        data.qpos[model.jnt_qposadr[joint_id]] = model.jnt_range[joint_id, 1]
        data.qvel[model.jnt_dofadr[joint_id]] = 0.0


def _set_finger_state(model: mujoco.MjModel, data: mujoco.MjData, side: Side, value: float) -> None:
    joint_id = _joint_id(model, FINGER_JOINTS[side])
    data.qpos[model.jnt_qposadr[joint_id]] = np.clip(value, *model.jnt_range[joint_id])


def _cube_near_hand(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: Side,
    *,
    task_frame_inset_m: float = HAND_TASK_FRAME_INSET_M,
) -> tuple[bool, float]:
    hand_position, _ = hand_pose(
        model,
        data,
        side,
        task_frame_inset_m=task_frame_inset_m,
    )
    distance = float(np.linalg.norm(_cube_position(model, data) - hand_position))
    return distance <= CUBE_HAND_DISTANCE_M, distance


def _cube_hand_contact_count(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: Side,
) -> int:
    """Count active contacts between the cube and either finger of one hand."""

    cube_geom = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom")
    cube_body = int(model.geom_bodyid[cube_geom])

    def belongs_to_hand(body_id: int) -> bool:
        return _body_descends_from(model, body_id, f"hand_{side}_link") or _body_descends_from(
            model,
            body_id,
            f"finger_{side}_link",
        )

    count = 0
    for contact in data.contact:
        if contact.geom1 < 0 or contact.geom2 < 0:
            continue
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if (body1 == cube_body and belongs_to_hand(body2)) or (
            body2 == cube_body and belongs_to_hand(body1)
        ):
            count += 1
    return count


def _cube_hand_has_bilateral_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: Side,
) -> bool:
    """Return whether the cube touches both the fixed and moving gripper jaws."""

    cube_geom = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom")
    cube_body = int(model.geom_bodyid[cube_geom])
    fixed_contact = False
    moving_contact = False
    for contact in data.contact:
        if contact.geom1 < 0 or contact.geom2 < 0:
            continue
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 == cube_body:
            hand_body = body2
        elif body2 == cube_body:
            hand_body = body1
        else:
            continue
        is_moving_finger = _body_descends_from(model, hand_body, f"finger_{side}_link")
        fixed_contact |= (
            _body_descends_from(model, hand_body, f"hand_{side}_link")
            and not is_moving_finger
        )
        moving_contact |= is_moving_finger
    return fixed_contact and moving_contact


def _set_cube_gripper_friction(model: mujoco.MjModel, *, grasping: bool) -> None:
    """Enable grasp friction or remove it immediately before cube release."""

    friction = (
        CUBE_GRIPPER_GRASP_FRICTION
        if grasping
        else CUBE_GRIPPER_RELEASE_FRICTION
    )
    for pair_name in CUBE_GRIPPER_CONTACT_PAIRS:
        pair_id = _named_id(model, mujoco.mjtObj.mjOBJ_PAIR, pair_name)
        model.pair_friction[pair_id] = friction


def _set_cube_grasp_lock(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: Side | None,
) -> None:
    """Attach the cube at its live relative pose, or disable both grasp locks."""

    equality_ids = {
        candidate: _named_id(
            model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            equality_name,
        )
        for candidate, equality_name in CUBE_GRASP_LOCK_EQUALITIES.items()
    }
    for equality_id in equality_ids.values():
        data.eq_active[equality_id] = False
    if side is None:
        mujoco.mj_forward(model, data)
        return

    hand_body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, f"hand_{side}_link")
    cube_body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_link")
    hand_rotation = data.xmat[hand_body_id].reshape(3, 3)
    cube_rotation = data.xmat[cube_body_id].reshape(3, 3)
    relative_position = hand_rotation.T @ (
        data.xpos[cube_body_id] - data.xpos[hand_body_id]
    )
    relative_rotation = hand_rotation.T @ cube_rotation
    relative_quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(relative_quaternion, relative_rotation.reshape(9))

    equality_id = equality_ids[side]
    model.eq_data[equality_id, 3:6] = relative_position
    model.eq_data[equality_id, 6:10] = relative_quaternion
    data.eq_active[equality_id] = True
    mujoco.mj_forward(model, data)


def _hand_hand_deep_contact_count(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """Count gripper contacts penetrating beyond the handoff tolerance."""

    def belongs_to_hand(body_id: int, side: Side) -> bool:
        return _body_descends_from(model, body_id, f"hand_{side}_link") or _body_descends_from(
            model,
            body_id,
            f"finger_{side}_link",
        )

    count = 0
    for contact in data.contact:
        if contact.geom1 < 0 or contact.geom2 < 0:
            continue
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        hands_are_paired = (
            belongs_to_hand(body1, "l") and belongs_to_hand(body2, "r")
        ) or (belongs_to_hand(body1, "r") and belongs_to_hand(body2, "l"))
        if hands_are_paired and contact.dist < -HANDOFF_CONTACT_ABORT_PENETRATION_M:
            count += 1
    return count


def _cube_center_inside_drawer(
    model: mujoco.MjModel, data: mujoco.MjData, drawer_index: int
) -> bool:
    floor_id = _named_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"drawer_{drawer_index}_link_collision_box_02_geom",
    )
    floor_rotation = data.geom_xmat[floor_id].reshape(3, 3)
    local_center = floor_rotation.T @ (_cube_position(model, data) - data.geom_xpos[floor_id])
    inside_xy = bool(np.all(np.abs(local_center[:2]) <= model.geom_size[floor_id, :2]))
    floor_top = float(model.geom_size[floor_id, 2])
    inside_height = floor_top - 0.005 <= local_center[2] <= floor_top + 0.040
    return inside_xy and inside_height


def _cube_center_inside_storage_bin(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    floor_id = _named_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "storage_bin_link_collision_box_05_geom",
    )
    cube_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom")
    floor_rotation = data.geom_xmat[floor_id].reshape(3, 3)
    local_center = floor_rotation.T @ (data.geom_xpos[cube_id] - data.geom_xpos[floor_id])
    cube_radius = model.geom_size[cube_id, :2]
    inside_xy = bool(
        np.all(np.abs(local_center[:2]) <= model.geom_size[floor_id, :2] - cube_radius)
    )
    floor_top = float(model.geom_size[floor_id, 2])
    inside_height = floor_top - 0.005 <= local_center[2] <= floor_top + 0.060
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
        early_failures: bool = True,
        cube_grasp_lock: bool = False,
        cube_drop_assist: bool = True,
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
        self.early_failures = early_failures
        self.cube_grasp_lock = cube_grasp_lock
        self.cube_drop_assist = cube_drop_assist
        self.trajectory = TrajectoryRandomization.sample(
            seed=seed,
            episode_seed=0,
            scale=trajectory_randomization_scale,
        )
        self._motion_rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 0x4D4F544E]))
        self.ik = PositionOrientationIK(
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
        self.handoff_receiver_plan: _HandoffReceiverPlan | None = None
        self.handoff_receiver_approach_targets: dict[str, float] | None = None
        self.handoff_donor_hold_targets: dict[str, float] | None = None
        self.handoff_donor: Side | None = None
        self.cube_grasp_lock_side: Side | None = None
        self.active_ik_side: Side | None = None
        self.active_ik_target: tuple[np.ndarray, np.ndarray] | None = None
        self.active_ik_hand_frame_inset_m = HAND_TASK_FRAME_INSET_M
        self._handoff_contact_frames = 0
        self._warned_handoff_contact = False
        self.perturbation_phase = PERTURBATION_PHASES[0]
        self.perturbation_trigger = PERTURBATION_TRIGGER_RANGE[0]
        self.perturbation_applied = False
        self.perturbation_expected_cube_held = False
        self.perturbation_resume_target: np.ndarray | None = None
        self.perturbation_resume_after: str | None = None
        self._recording_action = self._current_action()
        self.task_phase = 1
        self._recording_task_phase = 1
        self._recording_next_task_phase = 1
        self.workspace = Workspace.STORAGE
        self.cube_spawn_position = _cube_position(self.model, self.data)
        self.cube_quadrant = _cube_spawn_quadrant(self.model, self.data)
        self.episode_seed = seed
        self.done = False
        self.successful = False
        self.status = "not started"

    def reset(self, *, episode_seed: int) -> None:
        rng = initialize_task_episode(
            self.backend,
            seed=self.seed + episode_seed,
        )
        _set_cube_gripper_friction(self.model, grasping=True)
        _set_cube_grasp_lock(self.model, self.data, None)
        self.initial_targets = {
            name: _joint_qpos(self.model, self.data, name) for name in ROBOT_JOINTS
        }
        self.cube_spawn_position = _cube_position(self.model, self.data)
        self.cube_quadrant = _cube_spawn_quadrant(self.model, self.data)
        self.cube_initial_z = float(self.cube_spawn_position[2])
        self.drawer_index = int(rng.integers(1, 10))
        self.trajectory = TrajectoryRandomization.sample(
            seed=self.seed,
            episode_seed=episode_seed,
            scale=self.trajectory_randomization_scale,
        )
        self._motion_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, episode_seed, 0x4D4F544E])
        )
        perturbation_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, episode_seed, 0x50455254])
        )
        self.perturbation_phase = PERTURBATION_PHASES[
            int(perturbation_rng.integers(0, len(PERTURBATION_PHASES)))
        ]
        self.perturbation_trigger = float(perturbation_rng.uniform(*PERTURBATION_TRIGGER_RANGE))
        self._perturbation_rng = perturbation_rng
        self._initialize_ik_targets()
        self.episode_seed = episode_seed
        self.phase = "start_at_a"
        self.motion = None
        self.cube_approach_vertical_qpos = None
        self.drawer_close_cnc_start = None
        self.handoff_target = None
        self.handoff_receiver_plan = None
        self.handoff_receiver_approach_targets = None
        self.handoff_donor_hold_targets = None
        self.handoff_donor = None
        self.cube_grasp_lock_side = None
        self.active_ik_side = None
        self.active_ik_target = None
        self.active_ik_hand_frame_inset_m = HAND_TASK_FRAME_INSET_M
        self._handoff_contact_frames = 0
        self._warned_handoff_contact = False
        self.perturbation_applied = False
        self.perturbation_expected_cube_held = False
        self.perturbation_resume_target = None
        self.perturbation_resume_after = None
        self._recording_action = self._current_action()
        self.task_phase = 1
        self._recording_task_phase = 1
        self._recording_next_task_phase = 1
        self.workspace = Workspace.STORAGE
        self.done = False
        self.successful = False
        handoff = self.cube_hand != self.placement_hand
        cube_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom")
        cube_position = self.data.geom_xpos[cube_id]
        self.status = (
            f"cube=({cube_position[0]:.4f}, {cube_position[1]:.4f}, "
            f"{cube_position[2]:.4f}), drawer={self.drawer_index}, "
            f"drawer hand={self.drawer_hand}; "
            f"cube hand={self.cube_hand}, handoff={handoff}; "
            f"cube grasp lock={'enabled' if self.cube_grasp_lock else 'disabled'}; "
            f"cube drop assist={'enabled' if self.cube_drop_assist else 'disabled'}; "
            f"perturbation={self.perturbation_phase}@{self.perturbation_trigger:.2f}"
        )

    def _lock_cube_to_hand(self, side: Side) -> None:
        if not self.cube_grasp_lock:
            return
        _set_cube_grasp_lock(self.model, self.data, side)
        self.cube_grasp_lock_side = side
        if self.backend.config.debug:
            print(f"Cube grasp lock active: {side} hand")

    def _unlock_cube(self) -> None:
        if self.cube_grasp_lock_side is None:
            return
        _set_cube_grasp_lock(self.model, self.data, None)
        self.cube_grasp_lock_side = None
        if self.backend.config.debug:
            print("Cube grasp lock released")

    def _grasp_is_confirmed(self, *, ready: bool, contacts: int) -> bool:
        """Require proximity, then either physical contact or the optional lock."""

        return ready and (contacts > 0 or self.cube_grasp_lock)

    def _stabilize_cube_drawer_drop(self) -> None:
        """Suppress post-release bounce while leaving gravity free to lower the cube."""

        floor_id = _named_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"drawer_{self.drawer_index}_link_collision_box_02_geom",
        )
        cube_geom_id = _named_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "cube_link_collision_box_01_geom",
        )
        cube_joint_id = _joint_id(self.model, "cube_link_free_joint")
        qpos_id = int(self.model.jnt_qposadr[cube_joint_id])
        dof_id = int(self.model.jnt_dofadr[cube_joint_id])

        floor_rotation = self.data.geom_xmat[floor_id].reshape(3, 3)
        cube_rotation = self.data.geom_xmat[cube_geom_id].reshape(3, 3)
        cube_half_extents_in_drawer = (
            np.abs(floor_rotation.T @ cube_rotation)
            @ self.model.geom_size[cube_geom_id]
        )
        safe_half_width = np.maximum(
            self.model.geom_size[floor_id, :2]
            - cube_half_extents_in_drawer[:2]
            - CONTACT_MARGIN_M,
            0.0,
        )

        local_position = floor_rotation.T @ (
            self.data.qpos[qpos_id : qpos_id + 3]
            - self.data.geom_xpos[floor_id]
        )
        local_position[:2] = np.clip(
            local_position[:2],
            -safe_half_width,
            safe_half_width,
        )
        self.data.qpos[qpos_id : qpos_id + 3] = (
            self.data.geom_xpos[floor_id] + floor_rotation @ local_position
        )

        local_velocity = floor_rotation.T @ self.data.qvel[dof_id : dof_id + 3]
        local_velocity[:2] = 0.0
        landing_height = (
            self.model.geom_size[floor_id, 2]
            + cube_half_extents_in_drawer[2]
            + CONTACT_MARGIN_M
        )
        if local_position[2] > landing_height:
            # Keep a gentle minimum descent speed until the cube is close to
            # the floor, avoiding lip contacts that can suspend a light cube.
            local_velocity[2] = min(
                -CUBE_DROP_ASSIST_FALL_SPEED_M_S,
                float(local_velocity[2]),
            )
        else:
            # Near the floor, only cancel upward rebound and let contact plus
            # gravity determine the final resting height.
            local_velocity[2] = min(0.0, float(local_velocity[2]))
        self.data.qvel[dof_id : dof_id + 3] = floor_rotation @ local_velocity
        self.data.qvel[dof_id + 3 : dof_id + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _set_ik_target(self, marker_body: str, position: np.ndarray, rotation: np.ndarray) -> None:
        body_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, marker_body)
        self.model.body_pos[body_id] = position
        mujoco.mju_mat2Quat(
            self.model.body_quat[body_id], np.asarray(rotation, dtype=float).reshape(9)
        )

    def _update_active_ik_debug_frames(self) -> None:
        """Show the exact live hand task frame and most recent IK target frame."""

        if (
            not self.backend.config.debug
            or self.active_ik_side is None
            or self.active_ik_target is None
        ):
            return
        source_position, source_rotation = hand_pose(
            self.model,
            self.data,
            self.active_ik_side,
            task_frame_inset_m=self.active_ik_hand_frame_inset_m,
        )
        target_position, target_rotation = self.active_ik_target
        self._set_ik_target("active_ik_source_marker", source_position, source_rotation)
        self._set_ik_target("active_ik_target_marker", target_position, target_rotation)
        mujoco.mj_forward(self.model, self.data)

    def _activate_ik_debug_frames(
        self,
        *,
        side: Side,
        target: np.ndarray,
        rotation: np.ndarray,
        hand_frame_inset_m: float,
    ) -> None:
        if not self.backend.config.debug:
            return
        self.active_ik_side = side
        self.active_ik_target = (
            np.asarray(target, dtype=float).copy(),
            np.asarray(rotation, dtype=float).copy(),
        )
        self.active_ik_hand_frame_inset_m = hand_frame_inset_m
        self._update_active_ik_debug_frames()
        source_position, _ = hand_pose(
            self.model,
            self.data,
            side,
            task_frame_inset_m=hand_frame_inset_m,
        )
        print(
            f"IK frames [{self.phase}] hand={side}: "
            f"source=({source_position[0]:+.4f}, {source_position[1]:+.4f}, "
            f"{source_position[2]:+.4f}) m, "
            f"target=({target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}) m, "
            f"hand-frame inset={hand_frame_inset_m * 1000:.0f} mm"
        )

    def _refresh_drawer_targets(self) -> None:
        """Keep the opening and closing approach frames live with the drawer."""

        if self.drawer_hand is None:
            raise RuntimeError("Drawer hand must be selected before refreshing targets")
        opening_target, opening_rotation = _drawer_hand_target_pose(
            self.model, self.data, self.drawer_index, self.drawer_hand
        )
        closing_target, closing_rotation = _drawer_close_target_pose(
            self.model, self.data, self.drawer_index, self.drawer_hand
        )
        drawer_joint = _joint_id(self.model, f"base_link_base_drawer_{self.drawer_index}_joint")
        inward_axis = _normalize(self.data.xaxis[drawer_joint])
        _, drawer_rotation = drawer_handle_pose(self.model, self.data, self.drawer_index)
        handle_axis = _normalize(drawer_rotation[:, 0])
        common_offset = handle_axis * self.trajectory.drawer_handle_lateral_offset + np.array(
            [0.0, 0.0, self.trajectory.drawer_target_z_delta]
        )
        opening_target += common_offset - inward_axis * self.trajectory.drawer_open_approach_delta
        closing_target += common_offset - inward_axis * self.trajectory.drawer_close_approach_delta
        opening_rotation = opening_rotation @ self.trajectory.drawer_rotation
        closing_rotation = closing_rotation @ self.trajectory.drawer_rotation
        self.ik_targets["drawer_open"] = (opening_target, opening_rotation)
        self.ik_targets["drawer_close"] = (closing_target, closing_rotation)
        if self.backend.config.debug:
            self._set_ik_target("drawer_target_marker", opening_target, opening_rotation)
            self._set_ik_target("drawer_close_target_marker", closing_target, closing_rotation)
            mujoco.mj_forward(self.model, self.data)
            self.backend.sync_viewer()

    def _drawer_contact_target(
        self, *, drawer_qpos: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the handle contact pose at its live or requested opening."""

        if self.drawer_hand is None:
            raise RuntimeError("Drawer hand must be selected before planning contact")
        planning_data = mujoco.MjData(self.model)
        planning_data.qpos[:] = self.data.qpos
        drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
        drawer_id = _joint_id(self.model, drawer_joint)
        if drawer_qpos is not None:
            planning_data.qpos[int(self.model.jnt_qposadr[drawer_id])] = drawer_qpos
        mujoco.mj_forward(self.model, planning_data)
        target, rotation = drawer_handle_pose(self.model, planning_data, self.drawer_index)
        target = (
            target
            + rotation[:, 0] * self.trajectory.drawer_handle_lateral_offset
            + np.array([0.0, 0.0, self.trajectory.drawer_target_z_delta])
        )
        return target, _drawer_gripper_rotation(rotation, self.drawer_hand)

    def _initialize_ik_targets(self) -> None:
        """Create four IK frames; the two drawer frames are refreshed live."""

        cube_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom")
        cube_position = self.data.geom_xpos[cube_id].copy()
        self.cube_hand = _closest_hand(
            self.model,
            self.data,
            cube_position,
            task_frame_inset_m=STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M,
        )
        drawer_planning_data = mujoco.MjData(self.model)
        drawer_planning_data.qpos[:] = self.data.qpos
        set_workspace_qpos(
            self.model,
            drawer_planning_data,
            Workspace.DRAWERS,
            drawer=self.drawer_index,
        )
        mujoco.mj_forward(self.model, drawer_planning_data)
        handle_position, _ = drawer_handle_pose(self.model, drawer_planning_data, self.drawer_index)
        self.drawer_hand = _closest_hand(
            self.model,
            drawer_planning_data,
            handle_position,
            task_frame_inset_m=DRAWER_OPEN_HAND_FRAME_INSET_M,
        )
        cube_target, grasp_rotation = _cube_grasp_target(self.model, self.data, self.cube_hand)
        cube_target = (
            cube_target
            + grasp_rotation @ self.trajectory.cube_grasp_local_offset
            + np.array(
                [
                    0.0,
                    0.0,
                    CUBE_GRASP_APPROACH_HEIGHT_M + self.trajectory.cube_grasp_height_delta,
                ]
            )
        )
        grasp_rotation = grasp_rotation @ self.trajectory.cube_grasp_rotation

        opened_data = mujoco.MjData(self.model)
        opened_data.qpos[:] = drawer_planning_data.qpos
        drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
        drawer_id = _joint_id(self.model, drawer_joint)
        opened_data.qpos[self.model.jnt_qposadr[drawer_id]] = self.model.jnt_range[drawer_id, 0]
        mujoco.mj_forward(self.model, opened_data)
        opened_handle_position, _ = drawer_handle_pose(self.model, opened_data, self.drawer_index)
        self.placement_hand = (
            self.cube_hand
            if _drawer_is_center(self.drawer_index)
            else _closest_hand(self.model, opened_data, opened_handle_position)
        )

        cube_bottom, desired_cube_rotation = _cube_above_drawer_pose(
            self.model, opened_data, self.drawer_index
        )
        cube_bottom = cube_bottom + desired_cube_rotation @ self.trajectory.cube_place_local_offset
        desired_cube_rotation = desired_cube_rotation @ self.trajectory.cube_place_rotation
        cube_place_target = (
            cube_bottom + self.model.geom_size[cube_id, 2] * desired_cube_rotation[:, 2]
        )

        drawer_target, drawer_rotation = _drawer_hand_target_pose(
            self.model, self.data, self.drawer_index, self.drawer_hand
        )
        drawer_close_target, drawer_close_rotation = _drawer_close_target_pose(
            self.model, self.data, self.drawer_index, self.drawer_hand
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

        cube_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom")
        cube_position = self.data.geom_xpos[cube_id].copy()
        self.cube_hand = _closest_hand(
            self.model,
            self.data,
            cube_position,
            task_frame_inset_m=STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M,
        )

        placement_data = mujoco.MjData(self.model)
        placement_data.qpos[:] = self.data.qpos
        set_workspace_qpos(
            self.model,
            placement_data,
            Workspace.DRAWERS,
            drawer=self.drawer_index,
        )
        drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
        drawer_id = _joint_id(self.model, drawer_joint)
        placement_data.qpos[self.model.jnt_qposadr[drawer_id]] = self.model.jnt_range[drawer_id, 0]
        mujoco.mj_forward(self.model, placement_data)
        handle_position, _ = drawer_handle_pose(self.model, placement_data, self.drawer_index)
        self.placement_hand = (
            self.cube_hand
            if _drawer_is_center(self.drawer_index)
            else _closest_hand(self.model, placement_data, handle_position)
        )

        cube_target, cube_grasp_rotation = _cube_grasp_target(self.model, self.data, self.cube_hand)
        cube_target = (
            cube_target
            + cube_grasp_rotation @ self.trajectory.cube_grasp_local_offset
            + np.array(
                [
                    0.0,
                    0.0,
                    CUBE_GRASP_APPROACH_HEIGHT_M + self.trajectory.cube_grasp_height_delta,
                ]
            )
        )
        cube_grasp_rotation = cube_grasp_rotation @ self.trajectory.cube_grasp_rotation
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

    def _task_failure(self, checkpoint: str, reason: str) -> bool:
        """Abort a failed task check, or warn and continue when disabled."""

        if self.early_failures:
            self._fail(checkpoint, reason)
            return True
        self.status = f"ignored early failure after {checkpoint}: {reason}"
        print(self.status)
        return False

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

    @property
    def recording_action(self) -> dict[str, float]:
        """Expert label for the current observation, excluding injected disturbance."""

        return self.backend.action_dict(self._recording_action)

    @property
    def recording_task_phase(self) -> int:
        """One-based semantic phase used as the current frame's policy input."""

        return self._recording_task_phase

    @property
    def recording_next_task_phase(self) -> int:
        """One-based phase-transition target associated with the current frame."""

        return self._recording_next_task_phase

    @property
    def recording_workspace(self) -> str:
        """Categorical CNC workspace associated with the generated action."""

        return self.workspace.value

    @property
    def recording_metadata(self) -> dict[str, object]:
        """Episode-level fields used for recording coverage and success reports."""

        return {
            "drawer_index": self.drawer_index,
            "cube_position": self.cube_spawn_position.tolist(),
            "cube_quadrant": self.cube_quadrant,
        }

    def _start_raw_motion(
        self,
        target: np.ndarray,
        *,
        after: str,
        duration_s: float,
    ) -> None:
        start = self._current_action()
        clipped_target = np.clip(
            np.asarray(target, dtype=float),
            self.backend.control_low,
            self.backend.control_high,
        )
        frames = max(2, round(duration_s * self.backend.config.fps))
        self.motion = _Motion(start, clipped_target, frames, after)

    def _advance_task_phase(self, task_phase: int) -> None:
        """Advance the semantic task state without allowing backward transitions.

        A perturbation can make the cube fall back into the storage bin during
        placement. The expert then repeats its grasp routine, but the semantic
        phase remains at placement so that the recorded state machine is
        compatible with a forward-only System 2 controller.
        """

        if not 1 <= task_phase <= TASK_PHASE_COUNT:
            raise ValueError(f"Invalid semantic task phase: {task_phase}")
        self.task_phase = max(self.task_phase, task_phase)

    def _next_task_phase_target(self) -> int:
        """Return the transition target for the action generated on this frame.

        The final five frames before a semantic boundary target the following
        phase. This supplies enough consecutive transition labels for the future
        two-out-of-three phase vote; all other frames target the current phase.
        """

        next_phase = self.task_phase + 1
        expected_after = TASK_PHASE_TRANSITION_AFTER.get(self.task_phase)
        if expected_after is None or self.motion is None:
            return self.task_phase
        remaining_frames = self.motion.frame_count - self.motion.frame_index
        if (
            self.motion.after == expected_after
            and remaining_frames < TASK_PHASE_TRANSITION_WINDOW_FRAMES
        ):
            return next_phase
        return self.task_phase

    def _perturbed_target(self) -> np.ndarray:
        current = self._current_action()
        target = current.copy()
        for actuator_index in range(len(ACTUATOR_NAMES)):
            if actuator_index < len(COMMON_JOINTS) or ACTUATOR_NAMES[actuator_index].startswith(
                "finger_"
            ):
                continue
            low, high = PERTURBATION_ARM_RANGE
            magnitude = float(self._perturbation_rng.uniform(low, high))
            direction = -1.0 if self._perturbation_rng.random() < 0.5 else 1.0
            candidate = np.clip(
                current[actuator_index] + direction * magnitude,
                self.backend.control_low[actuator_index],
                self.backend.control_high[actuator_index],
            )
            if abs(candidate - current[actuator_index]) < 0.5 * magnitude:
                candidate = np.clip(
                    current[actuator_index] - direction * magnitude,
                    self.backend.control_low[actuator_index],
                    self.backend.control_high[actuator_index],
                )
            target[actuator_index] = candidate
        return target

    def _should_start_perturbation(self) -> bool:
        if (
            self.perturbation_applied
            or self.motion is None
            or self.phase != self.perturbation_phase
        ):
            return False
        denominator = max(1, self.motion.frame_count - 1)
        return self.motion.frame_index / denominator >= self.perturbation_trigger

    def _begin_perturbation(self) -> None:
        if self.motion is None:
            raise RuntimeError("Cannot perturb without an active expert motion")
        self.perturbation_applied = True
        self.perturbation_expected_cube_held = self.phase == "cube_above_drawer"
        self.perturbation_resume_target = self.motion.target.copy()
        self.perturbation_resume_after = self.motion.after
        self.phase = "apply_perturbation"
        self._start_raw_motion(
            self._perturbed_target(),
            after="recover_perturbation",
            duration_s=PERTURBATION_DURATION_S,
        )

    def _cube_is_still_held(self) -> bool:
        if self.cube_hand is None:
            return False
        near_hand, _ = _cube_near_hand(self.model, self.data, self.cube_hand)
        return near_hand

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
                self.move_duration_s * duration_scale * duration_jitter * self.backend.config.fps
            ),
        )
        self.motion = _Motion(start, target, frames, after)

    def _start_workspace_motion(
        self,
        workspace: Workspace,
        *,
        after: str,
        fingers: dict[Side, float] | None = None,
    ) -> None:
        """Move all CNC axes together to one allowed discrete workspace."""

        targets = self._workspace_targets(workspace)
        self.workspace = workspace
        self._start_motion(targets, after=after, fingers=fingers)

    def _workspace_targets(self, workspace: Workspace) -> dict[str, float]:
        return resolve_mujoco_workspace(
            self.model,
            workspace,
            drawer=self.drawer_index if workspace is Workspace.DRAWERS else None,
        )

    def _drawer_robotward_axis(self) -> np.ndarray:
        """Return the drawer travel direction toward the robot, projected horizontally."""

        drawer_joint = _joint_id(
            self.model,
            f"base_link_base_drawer_{self.drawer_index}_joint",
        )
        robotward_axis = -self.data.xaxis[drawer_joint].copy()
        robotward_axis[2] = 0.0
        return _normalize(robotward_axis)

    def _workspace_transit_is_clear(self, workspace: Workspace) -> bool:
        """Require all fingertips to clear the storage bin throughout CNC travel."""

        planning_data = mujoco.MjData(self.model)
        planning_data.qpos[:] = self.data.qpos
        set_workspace_qpos(
            self.model,
            planning_data,
            workspace,
            drawer=self.drawer_index if workspace is Workspace.DRAWERS else None,
        )
        mujoco.mj_forward(self.model, planning_data)

        bin_top = _storage_bin_top_z(self.model, self.data)
        required_z = bin_top + WORKSPACE_TRANSIT_BIN_CLEARANCE_M
        current_z = _minimum_fingertip_z(self.model, self.data)
        target_z = _minimum_fingertip_z(self.model, planning_data)
        minimum_z = min(current_z, target_z)
        if minimum_z + 1e-9 < required_z:
            self._fail(
                "workspace transit",
                f"lowest fingertip z={minimum_z:.3f} m; required at least "
                f"{required_z:.3f} m above the storage bin",
            )
            return False
        return True

    def _solve_to(
        self,
        *,
        side: Side,
        target: np.ndarray,
        rotation: np.ndarray,
        seed_offset: int,
        **kwargs: object,
    ) -> dict[str, float] | None:
        kwargs.setdefault("posture_reference", self._rest_targets(side))
        kwargs.setdefault("posture_weight", DEFAULT_IK_POSTURE_WEIGHT)
        kwargs.setdefault("previous_posture_weight", DEFAULT_IK_PREVIOUS_POSTURE_WEIGHT)
        kwargs.setdefault(
            "max_posture_deviation_radians",
            DEFAULT_IK_MAX_POSTURE_DEVIATION_RADIANS,
        )
        self._activate_ik_debug_frames(
            side=side,
            target=target,
            rotation=rotation,
            hand_frame_inset_m=float(
                kwargs.get("hand_frame_inset_m", HAND_TASK_FRAME_INSET_M)
            ),
        )
        solution = self.ik.solve(
            side=side,
            target=target,
            target_rotation=rotation,
            seed=self.seed + self.episode_seed + seed_offset,
            **kwargs,
        )
        return dict(zip(solution.joint_names, solution.joint_values, strict=True))

    def _check_handoff_hand_clearance(self) -> bool:
        """Abort only sustained hand contact, or warn when early failures are disabled."""

        if self.phase not in HANDOFF_CLEARANCE_PHASES:
            self._handoff_contact_frames = 0
            return False
        if _hand_hand_deep_contact_count(self.model, self.data) == 0:
            self._handoff_contact_frames = 0
            return False

        self._handoff_contact_frames += 1
        if self._handoff_contact_frames < HANDOFF_CONTACT_ABORT_FRAMES:
            return False
        reason = (
            f"left and right grippers remained in contact for "
            f"{self._handoff_contact_frames} frames during {self.phase}"
        )
        if self.early_failures:
            self._fail("cube handoff", reason)
            return True
        if not self._warned_handoff_contact:
            self.status = f"ignored early failure after cube handoff: {reason}"
            print(self.status)
            self._warned_handoff_contact = True
        return False

    def _flat_hand_posture(self, side: Side) -> dict[str, float]:
        """Keep an arm near rest while preferring a floor-parallel hand."""

        posture = self._rest_targets(side)
        posture[SIDE_JOINTS[side][-1]] = 0.0
        return posture

    def _solve_cube_grasp(
        self,
        *,
        side: Side,
        target: np.ndarray,
        rotation: np.ndarray,
        seed_offset: int,
        finger_position: float = FINGER_OPEN,
        joint_names: tuple[str, ...] | None = None,
        protect_other_hand: bool = True,
        hand_frame_inset_m: float = HAND_TASK_FRAME_INSET_M,
    ) -> dict[str, float]:
        """Solve a grasp while keeping the hand and cube blue axes parallel."""

        return self._solve_to(
            side=side,
            target=target,
            rotation=rotation,
            seed_offset=seed_offset,
            joint_names=joint_names,
            reset_drawers=False,
            align_red_axis=False,
            align_blue_axis=True,
            directed_axes=True,
            # Cartesian coincidence is primary.  The directed blue-axis term
            # shapes the grasp without being allowed to pull the fingertips
            # away from the cube target.
            position_weight=CUBE_GRASP_IK_POSITION_WEIGHT,
            orientation_weight=CUBE_GRASP_IK_ORIENTATION_WEIGHT,
            posture_reference=self._flat_hand_posture(side),
            protect_drawers=True,
            protect_other_hand=protect_other_hand,
            finger_position=finger_position,
            hand_frame_inset_m=hand_frame_inset_m,
        )

    def _solve_handoff_receiver(
        self,
        *,
        target: np.ndarray,
        rotation: np.ndarray,
        seed_offset: int,
        finger_position: float,
        planning_qpos: np.ndarray | None = None,
        posture_reference: dict[str, float] | None = None,
        previous_posture_weight: float = DEFAULT_IK_PREVIOUS_POSTURE_WEIGHT,
        position_weight: float = HANDOFF_RECEIVER_APPROACH_POSITION_WEIGHT,
        red_orientation_weight: float = (
            HANDOFF_RECEIVER_APPROACH_RED_ORIENTATION_WEIGHT
        ),
        blue_orientation_weight: float = (
            HANDOFF_RECEIVER_APPROACH_BLUE_ORIENTATION_WEIGHT
        ),
    ) -> dict[str, float] | None:
        """Solve the receiver using weighted pose objectives without error cutoffs."""

        assert self.placement_hand is not None
        return self._solve_to(
            side=self.placement_hand,
            target=target,
            rotation=rotation,
            seed_offset=seed_offset,
            joint_names=SIDE_JOINTS[self.placement_hand],
            planning_qpos=planning_qpos,
            reset_drawers=False,
            align_red_axis=True,
            align_blue_axis=True,
            directed_axes=True,
            directed_red_axis=False,
            position_weight=position_weight,
            red_orientation_weight=red_orientation_weight,
            blue_orientation_weight=blue_orientation_weight,
            posture_reference=(
                self._flat_hand_posture(self.placement_hand)
                if posture_reference is None
                else posture_reference
            ),
            previous_posture_weight=previous_posture_weight,
            protect_drawers=True,
            protect_other_hand=True,
            finger_position=finger_position,
            hand_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
        )

    def _cube_grasp_command(self) -> float:
        """Keep the receiving hand firmly closed after a completed handoff."""

        if self.handoff_donor is not None:
            return HANDOFF_RECEIVER_GRASP
        return self.trajectory.cube_grasp_position

    def _with_fixed_handoff_donor(
        self,
        receiver_targets: dict[str, float],
    ) -> dict[str, float]:
        """Merge receiver IK with the donor pose captured before its approach."""

        if self.handoff_donor_hold_targets is None:
            raise RuntimeError("Handoff donor pose was not captured")
        return {**self.handoff_donor_hold_targets, **receiver_targets}

    def _solve_cube_placement(
        self,
        *,
        target: np.ndarray,
        rotation: np.ndarray,
        seed_offset: int,
        finger_position: float,
        planning_qpos: np.ndarray | None = None,
        position_weight: float = 25_000.0,
    ) -> dict[str, float]:
        """Solve a flat-hand placement or withdrawal pose."""

        assert self.cube_hand is not None
        return self._solve_to(
            side=self.cube_hand,
            target=target,
            rotation=rotation,
            seed_offset=seed_offset,
            planning_qpos=planning_qpos,
            reset_drawers=False,
            align_red_axis=False,
            align_blue_axis=True,
            directed_axes=True,
            position_weight=position_weight,
            orientation_weight=CUBE_PLACE_IK_ORIENTATION_WEIGHT,
            posture_reference=self._flat_hand_posture(self.cube_hand),
            protect_drawers=True,
            protect_other_hand=True,
            finger_position=finger_position,
        )

    def _rest_targets(self, side: Side) -> dict[str, float]:
        """Return the exact horizontal arm pose captured during initialization."""

        return {name: self.initial_targets[name] for name in SIDE_JOINTS[side]}

    def _handoff_workspace_center(self) -> np.ndarray:
        """Return the raised center of the two arms' shared nominal workspace.

        The point is derived from the model at workspace A with both arms in
        their nominal working pose. It therefore does not move with either
        hand's live pose and uses the zero-inset fingertip-midpoint frames. Its
        X/Y coordinates remain at that geometric midpoint, while Z is raised
        to keep the handoff clear of the storage bin.
        """

        planning_data = mujoco.MjData(self.model)
        planning_data.qpos[:] = self.data.qpos
        set_workspace_qpos(self.model, planning_data, Workspace.STORAGE)
        for side in ("l", "r"):
            for name in SIDE_JOINTS[side]:
                joint_id = _joint_id(self.model, name)
                planning_data.qpos[int(self.model.jnt_qposadr[joint_id])] = (
                    self.initial_targets[name]
                )
            _set_finger_state(self.model, planning_data, side, FINGER_CLOSED)
        mujoco.mj_forward(self.model, planning_data)
        hand_centers = [
            hand_pose(
                self.model,
                planning_data,
                side,
                task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
            )[0]
            for side in ("l", "r")
        ]
        center = 0.5 * (hand_centers[0] + hand_centers[1])
        center[2] += HANDOFF_WORKSPACE_HEIGHT_OFFSET_M
        return center

    def _transit_targets(self) -> dict[str, float]:
        """Raise both shoulders enough for open fingertips to clear the storage bin."""

        targets = {
            name: self.initial_targets[name] for name in (*SIDE_JOINTS["l"], *SIDE_JOINTS["r"])
        }
        for side, direction in (("l", 1.0), ("r", -1.0)):
            shoulder = SIDE_JOINTS[side][0]
            shoulder_id = _joint_id(self.model, shoulder)
            targets[shoulder] = float(
                np.clip(
                    targets[shoulder] + direction * WORKSPACE_TRANSIT_SHOULDER_LIFT_RADIANS,
                    *self.model.jnt_range[shoulder_id],
                )
            )
        return targets

    def _working_hand_rotation(self, side: Side) -> np.ndarray:
        """Return this hand's natural orientation in the bent working pose."""

        planning_data = mujoco.MjData(self.model)
        planning_data.qpos[:] = self.data.qpos
        for name in SIDE_JOINTS[side]:
            joint_id = _joint_id(self.model, name)
            planning_data.qpos[int(self.model.jnt_qposadr[joint_id])] = self.initial_targets[name]
        set_workspace_qpos(
            self.model,
            planning_data,
            Workspace.DRAWERS,
            drawer=self.drawer_index,
        )
        mujoco.mj_forward(self.model, planning_data)
        return hand_pose(self.model, planning_data, side)[1]

    def _handoff_donor_rest_targets(self, side: Side) -> dict[str, float]:
        """Return a raised post-handoff pose without moving the shared CNC axes."""

        targets = self._rest_targets(side)
        arm_joint = SIDE_JOINTS[side][2]
        arm_id = _joint_id(self.model, arm_joint)
        elevation = (
            HANDOFF_DONOR_ARM_ELEVATION_RAD + self.trajectory.handoff_donor_elevation_delta
        ) * (1.0 if side == "l" else -1.0)
        targets[arm_joint] = float(np.clip(elevation, *self.model.jnt_range[arm_id]))
        return targets

    def _initial_view_targets(self) -> dict[str, float]:
        """Return the initial arm pose without changing workspace A."""

        return {name: self.initial_targets[name] for name in (*SIDE_JOINTS["l"], *SIDE_JOINTS["r"])}

    def _dispatch_phase(self) -> None:
        if self.phase == "recover_perturbation":
            if self.perturbation_resume_target is None:
                raise RuntimeError("Perturbation recovery target was not saved")
            self._start_raw_motion(
                self.perturbation_resume_target,
                after="assess_perturbation_recovery",
                duration_s=PERTURBATION_RECOVERY_DURATION_S,
            )
        elif self.phase == "assess_perturbation_recovery":
            if self.perturbation_resume_after is None:
                raise RuntimeError("Perturbation continuation phase was not saved")
            if self.perturbation_expected_cube_held and not self._cube_is_still_held():
                mujoco.mj_forward(self.model, self.data)
                if _cube_center_inside_storage_bin(self.model, self.data):
                    self._start_workspace_motion(
                        Workspace.STORAGE,
                        after="restart_after_perturbation_drop",
                        fingers={"l": FINGER_OPEN, "r": FINGER_OPEN},
                    )
                else:
                    self._start_workspace_motion(
                        Workspace.STORAGE,
                        after="abort_after_perturbation_drop",
                        fingers={"l": FINGER_OPEN, "r": FINGER_OPEN},
                    )
            else:
                self.phase = self.perturbation_resume_after
        elif self.phase == "restart_after_perturbation_drop":
            self.cube_initial_z = float(_cube_position(self.model, self.data)[2])
            self.phase = "cube_ik"
        elif self.phase == "abort_after_perturbation_drop":
            reason = "cube fell outside storage bin"
            if self._task_failure("perturbation", reason):
                return
            if self.perturbation_resume_after is None:
                raise RuntimeError("Perturbation continuation phase was not saved")
            self.phase = self.perturbation_resume_after
        elif self.phase == "start_at_a":
            self._start_motion(
                self._transit_targets(),
                after="move_to_b_for_open",
                fingers={"l": FINGER_CLOSED, "r": FINGER_CLOSED},
            )
        elif self.phase == "move_to_b_for_open":
            if not self._workspace_transit_is_clear(Workspace.DRAWERS):
                return
            self._advance_task_phase(TaskPhase.OPEN_DRAWER)
            targets = self._workspace_targets(Workspace.DRAWERS)
            self.workspace = Workspace.DRAWERS
            self._start_motion(
                {
                    COMMON_JOINTS[1]: targets[COMMON_JOINTS[1]],
                    COMMON_JOINTS[2]: targets[COMMON_JOINTS[2]],
                },
                after="enter_b_for_open",
                fingers={"l": FINGER_CLOSED, "r": FINGER_CLOSED},
            )
        elif self.phase == "enter_b_for_open":
            target = self._workspace_targets(Workspace.DRAWERS)[COMMON_JOINTS[0]]
            self._start_motion(
                {COMMON_JOINTS[0]: target},
                after="open_before_first_ik",
                fingers={"l": FINGER_CLOSED, "r": FINGER_CLOSED},
            )
        elif self.phase == "open_before_first_ik":
            self._start_motion(
                {},
                after="drawer_ik",
                duration_scale=0.4,
                fingers={"l": DRAWER_FINGER_OPEN, "r": DRAWER_FINGER_OPEN},
            )
        elif self.phase == "cube_ik":
            self._select_post_drawer_hands_and_targets()
            assert (
                self.cube_hand is not None
                and self.drawer_hand is not None
                and self.placement_hand is not None
            )
            target, rotation = self.ik_targets["cube_grasp"]
            targets = self._solve_cube_grasp(
                side=self.cube_hand,
                target=target,
                rotation=rotation,
                seed_offset=0,
                hand_frame_inset_m=STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M,
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
            target, rotation = _cube_grasp_target(self.model, self.data, self.cube_hand)
            target = target + rotation @ self.trajectory.cube_grasp_local_offset
            rotation = rotation @ self.trajectory.cube_grasp_rotation
            targets = self._solve_cube_grasp(
                side=self.cube_hand,
                target=target,
                rotation=rotation,
                seed_offset=5,
                finger_position=FINGER_OPEN,
                hand_frame_inset_m=STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="close_cube",
                fingers={self.cube_hand: FINGER_OPEN},
            )
        elif self.phase == "close_cube":
            assert self.cube_hand is not None
            self._start_motion(
                {},
                after="settle_cube_grasp",
                fingers={self.cube_hand: self.trajectory.cube_grasp_position},
            )
        elif self.phase == "settle_cube_grasp":
            assert self.cube_hand is not None
            self._start_motion(
                {},
                after="cube_clearance",
                duration_scale=CUBE_GRASP_SETTLE_DURATION_S / self.move_duration_s,
                fingers={self.cube_hand: self.trajectory.cube_grasp_position},
            )
        elif self.phase == "cube_clearance":
            assert self.cube_hand is not None
            ready, distance = _cube_near_hand(
                self.model,
                self.data,
                self.cube_hand,
                task_frame_inset_m=STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M,
            )
            contacts = _cube_hand_contact_count(self.model, self.data, self.cube_hand)
            bilateral_contact = _cube_hand_has_bilateral_contact(
                self.model,
                self.data,
                self.cube_hand,
            )
            grasp_confirmed = self._grasp_is_confirmed(
                ready=ready,
                contacts=contacts,
            )
            if not grasp_confirmed and self._task_failure(
                "cube grasp",
                f"grasp was not stable after closing "
                f"(distance={distance:.4f} m, contacts={contacts}, "
                f"bilateral={bilateral_contact})",
            ):
                return
            if ready:
                self._lock_cube_to_hand(self.cube_hand)
            target, rotation = self.ik_targets["cube_grasp"]
            targets = self._solve_cube_grasp(
                side=self.cube_hand,
                target=target,
                rotation=rotation,
                seed_offset=6,
                finger_position=self.trajectory.cube_grasp_position,
                hand_frame_inset_m=STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M,
            )
            if targets is None:
                return
            # Clear the idle arm during the cube lift itself.  This keeps both
            # hands above the bin for the following CNC motion without adding
            # a separate post-grasp transport-pose action.
            idle_hand: Side = "r" if self.cube_hand == "l" else "l"
            transit_targets = self._transit_targets()
            targets.update(
                {name: transit_targets[name] for name in SIDE_JOINTS[idle_hand]}
            )
            self._start_motion(
                targets,
                after="return_without_base",
                fingers={self.cube_hand: self.trajectory.cube_grasp_position},
            )
        elif self.phase == "return_without_base":
            assert self.cube_hand is not None
            if self.cube_initial_z is None:
                raise RuntimeError("Initial cube height was not captured")
            cube_z = float(_cube_position(self.model, self.data)[2])
            if cube_z < self.cube_initial_z + CUBE_LIFT_CHECK_M and self._task_failure(
                "cube grasp",
                f"cube was not lifted (initial z={self.cube_initial_z:.4f} m, "
                f"current z={cube_z:.4f} m)",
            ):
                return
            self.phase = (
                "cube_above_drawer"
                if self.cube_hand == self.placement_hand
                else "handoff_donor_ik"
            )
        elif self.phase == "handoff_donor_ik":
            assert self.cube_hand is not None and self.placement_hand is not None
            donor = self.cube_hand
            receiver = self.placement_hand
            _, donor_rotation = hand_pose(
                self.model,
                self.data,
                donor,
                task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
            )
            self.handoff_target = self._handoff_workspace_center()
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
                protect_drawers=True,
                protect_other_hand=True,
                finger_position=self.trajectory.cube_grasp_position,
                hand_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
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
            donor = self.cube_hand
            receiver = self.placement_hand
            # Freeze the donor exactly where it finished its handoff motion.
            # Every receiver motion below explicitly carries these targets.
            self.handoff_donor_hold_targets = {
                name: _joint_qpos(self.model, self.data, name)
                for name in SIDE_JOINTS[donor]
            }
            self.handoff_receiver_plan = _handoff_receiver_plan(
                self.model,
                self.data,
                donor,
                receiver,
            )
            if self.backend.config.debug:
                cube_id = _named_id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    "cube_link_collision_box_01_geom",
                )
                cube_rotation = self.data.geom_xmat[cube_id].reshape(3, 3)

                def cube_face_label(face: np.ndarray) -> str:
                    scores = cube_rotation[:, :2].T @ face
                    axis = int(np.argmax(np.abs(scores)))
                    sign = "+" if scores[axis] >= 0.0 else "-"
                    return f"{sign}{('X', 'Y')[axis]}"

                receiver_face = cube_face_label(self.handoff_receiver_plan.receiver_face)
                donor_axis = ("X", "Y")[self.handoff_receiver_plan.donor_face_axis]
                print(
                    f"Handoff cube axes: donor occupies +/-{donor_axis}; "
                    f"receiver red=+/-cube blue; "
                    f"blue={receiver_face}; approach face={receiver_face} "
                    f"(cube frame: red=X, green=Y, blue=Z)"
                )
            targets = self._solve_handoff_receiver(
                target=self.handoff_receiver_plan.approach_target,
                rotation=self.handoff_receiver_plan.approach_rotation,
                seed_offset=3,
                finger_position=HANDOFF_RECEIVER_WIDE_OPEN,
            )
            if targets is None:
                return
            self.handoff_receiver_approach_targets = targets.copy()
            self._start_motion(
                self._with_fixed_handoff_donor(targets),
                after="handoff_receiver_contact_ik",
                fingers={
                    donor: self.trajectory.cube_grasp_position,
                    receiver: HANDOFF_RECEIVER_WIDE_OPEN,
                },
            )
        elif self.phase == "handoff_receiver_contact_ik":
            assert self.cube_hand is not None and self.placement_hand is not None
            if self.handoff_receiver_plan is None:
                raise RuntimeError("Handoff receiver plan was not created")
            if self.handoff_receiver_approach_targets is None:
                raise RuntimeError("Handoff receiver approach was not solved")
            targets = self._solve_handoff_receiver(
                target=self.handoff_receiver_plan.contact_target,
                rotation=self.handoff_receiver_plan.contact_rotation,
                seed_offset=4,
                # Keep the collision geometry and physical receiver fingers
                # fully open throughout the straight move to cube contact.
                finger_position=HANDOFF_RECEIVER_WIDE_OPEN,
                posture_reference=self.handoff_receiver_approach_targets,
                previous_posture_weight=(
                    HANDOFF_RECEIVER_TRANSLATION_CONTINUITY_WEIGHT
                ),
                position_weight=HANDOFF_RECEIVER_TRANSLATION_POSITION_WEIGHT,
                red_orientation_weight=(
                    HANDOFF_RECEIVER_TRANSLATION_RED_ORIENTATION_WEIGHT
                ),
                blue_orientation_weight=(
                    HANDOFF_RECEIVER_TRANSLATION_BLUE_ORIENTATION_WEIGHT
                ),
            )
            if targets is None:
                return
            self._start_motion(
                self._with_fixed_handoff_donor(targets),
                after="handoff_grasp",
                fingers={
                    self.cube_hand: self.trajectory.cube_grasp_position,
                    self.placement_hand: HANDOFF_RECEIVER_WIDE_OPEN,
                },
            )
        elif self.phase == "handoff_grasp":
            assert self.cube_hand is not None and self.placement_hand is not None
            self._start_motion(
                self._with_fixed_handoff_donor({}),
                after="handoff_verify",
                fingers={
                    self.cube_hand: self.trajectory.cube_grasp_position,
                    self.placement_hand: HANDOFF_RECEIVER_GRASP,
                },
            )
        elif self.phase == "handoff_verify":
            assert self.cube_hand is not None and self.placement_hand is not None
            ready, distance = _cube_near_hand(
                self.model,
                self.data,
                self.placement_hand,
                task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
            )
            if not ready:
                # Handoff failure is mandatory: stopping here leaves the donor
                # closed instead of dropping a cube that the receiver missed.
                self._fail(
                    "cube handoff",
                    f"cube is {distance:.4f} m from the receiving hand; required "
                    f"at most {CUBE_HAND_DISTANCE_M:.4f} m",
                )
                return
            self._start_motion(
                self._with_fixed_handoff_donor({}),
                after="handoff_load_transfer",
                duration_scale=HANDOFF_OVERLAP_HOLD_DURATION_SCALE,
                fingers={
                    self.cube_hand: self.trajectory.cube_grasp_position,
                    self.placement_hand: HANDOFF_RECEIVER_GRASP,
                },
            )
        elif self.phase == "handoff_load_transfer":
            assert self.cube_hand is not None and self.placement_hand is not None
            receiver = self.placement_hand
            ready, distance = _cube_near_hand(
                self.model,
                self.data,
                receiver,
                task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
            )
            receiver_contacts = _cube_hand_contact_count(self.model, self.data, receiver)
            bilateral_contact = _cube_hand_has_bilateral_contact(
                self.model,
                self.data,
                receiver,
            )
            if not self._grasp_is_confirmed(
                ready=ready,
                contacts=receiver_contacts,
            ):
                self._fail(
                    "cube handoff",
                    f"receiver grasp was not stable after overlap hold "
                    f"(distance={distance:.4f} m, contacts={receiver_contacts}, "
                    f"bilateral={bilateral_contact})",
                )
                return
            self._start_motion(
                self._with_fixed_handoff_donor({}),
                after="handoff_release",
                fingers={
                    self.cube_hand: self.trajectory.cube_grasp_position,
                    receiver: HANDOFF_RECEIVER_GRASP,
                },
            )
        elif self.phase == "handoff_release":
            assert self.cube_hand is not None and self.placement_hand is not None
            donor = self.cube_hand
            receiver = self.placement_hand
            ready, distance = _cube_near_hand(
                self.model,
                self.data,
                receiver,
                task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
            )
            receiver_contacts = _cube_hand_contact_count(self.model, self.data, receiver)
            bilateral_contact = _cube_hand_has_bilateral_contact(
                self.model,
                self.data,
                receiver,
            )
            if not self._grasp_is_confirmed(
                ready=ready,
                contacts=receiver_contacts,
            ):
                # On failure, both hands retain their closed command and the
                # donor never enters its opening/retreat motion.
                self._fail(
                    "cube handoff",
                    f"receiver grasp was not stable after overlap hold "
                    f"(distance={distance:.4f} m, contacts={receiver_contacts}, "
                    f"bilateral={bilateral_contact})",
                )
                return
            self._lock_cube_to_hand(receiver)
            donor_position, donor_rotation = hand_pose(
                self.model,
                self.data,
                donor,
                task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
            )
            donor_retreat_axis = donor_position - _cube_position(self.model, self.data)
            donor_retreat_axis[2] = 0.0
            if np.linalg.norm(donor_retreat_axis) < 1e-9:
                if self.handoff_receiver_plan is None:
                    raise RuntimeError("Handoff receiver plan was not created")
                donor_retreat_axis = -self.handoff_receiver_plan.receiver_face.copy()
                donor_retreat_axis[2] = 0.0
            donor_retreat_axis = _normalize(donor_retreat_axis)
            targets = self._solve_to(
                side=donor,
                target=(
                    donor_position
                    + HANDOFF_DONOR_RETREAT_DISTANCE_M * donor_retreat_axis
                ),
                rotation=donor_rotation,
                seed_offset=6,
                joint_names=SIDE_JOINTS[donor],
                reset_drawers=False,
                align_red_axis=True,
                align_blue_axis=True,
                directed_axes=True,
                position_weight=CUBE_GRASP_IK_POSITION_WEIGHT,
                orientation_weight=CUBE_GRASP_IK_ORIENTATION_WEIGHT,
                finger_position=FINGER_OPEN,
                posture_reference=self._flat_hand_posture(donor),
                protect_drawers=True,
                protect_other_hand=True,
                hand_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
            )
            if targets is None:
                return
            receiver_hold_targets = {
                name: _joint_qpos(self.model, self.data, name)
                for name in SIDE_JOINTS[receiver]
            }
            self.handoff_donor = donor
            self.cube_hand = receiver
            self._start_motion(
                {**targets, **receiver_hold_targets},
                after="handoff_receiver_stability",
                fingers={
                    donor: FINGER_OPEN,
                    receiver: self._cube_grasp_command(),
                },
            )
        elif self.phase == "handoff_receiver_stability":
            assert self.cube_hand is not None
            ready, distance = _cube_near_hand(
                self.model,
                self.data,
                self.cube_hand,
                task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
            )
            receiver_contacts = _cube_hand_contact_count(
                self.model,
                self.data,
                self.cube_hand,
            )
            bilateral_contact = _cube_hand_has_bilateral_contact(
                self.model,
                self.data,
                self.cube_hand,
            )
            if not self._grasp_is_confirmed(
                ready=ready,
                contacts=receiver_contacts,
            ):
                self._fail(
                    "cube handoff",
                    f"receiver lost the cube after donor release "
                    f"(distance={distance:.4f} m, contacts={receiver_contacts}, "
                    f"bilateral={bilateral_contact})",
                )
                return
            self.phase = "handoff_donor_rest"
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
                    self.cube_hand: self._cube_grasp_command(),
                },
            )
        elif self.phase == "cube_above_drawer":
            assert self.cube_hand is not None
            target, rotation = self.ik_targets["cube_place"]

            # Solve the arm as it must look at B, but execute those arm-joint
            # targets while the live CNC is still at A. The subsequent CNC
            # translation carries an already-positioned hand into the open
            # drawer instead of articulating near its collision geometry.
            planning_data = mujoco.MjData(self.model)
            planning_data.qpos[:] = self.data.qpos
            set_workspace_qpos(
                self.model,
                planning_data,
                Workspace.DRAWERS,
                drawer=self.drawer_index,
            )
            targets = self._solve_cube_placement(
                target=target,
                rotation=rotation,
                seed_offset=3,
                planning_qpos=planning_data.qpos,
                finger_position=self._cube_grasp_command(),
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="move_to_b_for_place",
                fingers={self.cube_hand: self._cube_grasp_command()},
            )
        elif self.phase == "move_to_b_for_place":
            assert self.cube_hand is not None
            if not self._workspace_transit_is_clear(Workspace.DRAWERS):
                return
            self._advance_task_phase(TaskPhase.PLACE_AND_CLOSE)
            targets = self._workspace_targets(Workspace.DRAWERS)
            self.workspace = Workspace.DRAWERS
            self._start_motion(
                {
                    COMMON_JOINTS[1]: targets[COMMON_JOINTS[1]],
                    COMMON_JOINTS[2]: targets[COMMON_JOINTS[2]],
                },
                after="enter_b_for_place",
                fingers={self.cube_hand: self._cube_grasp_command()},
            )
        elif self.phase == "enter_b_for_place":
            assert self.cube_hand is not None
            target = self._workspace_targets(Workspace.DRAWERS)[COMMON_JOINTS[0]]
            self._start_motion(
                {COMMON_JOINTS[0]: target},
                after="drop_cube",
                fingers={self.cube_hand: self._cube_grasp_command()},
            )
        elif self.phase == "drawer_ik":
            assert self.drawer_hand is not None
            target, rotation = self.ik_targets["drawer_open"]
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=rotation,
                seed_offset=1,
                reset_drawers=False,
                # Only the finger-gap axis matters for a horizontal handle.
                align_blue_axis=False,
                directed_axes=True,
                position_weight=DRAWER_OPEN_IK_POSITION_WEIGHT,
                orientation_weight=DRAWER_IK_ORIENTATION_WEIGHT,
                include_collision_penalty=False,
                finger_position=DRAWER_FINGER_OPEN,
                hand_frame_inset_m=DRAWER_OPEN_HAND_FRAME_INSET_M,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="approach_drawer",
                duration_scale=1.5,
                fingers={self.drawer_hand: DRAWER_FINGER_OPEN},
            )
        elif self.phase == "approach_drawer":
            assert self.drawer_hand is not None
            target, rotation = self._drawer_contact_target()
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=rotation,
                seed_offset=7,
                reset_drawers=False,
                align_blue_axis=False,
                directed_axes=True,
                position_weight=DRAWER_OPEN_IK_POSITION_WEIGHT,
                orientation_weight=DRAWER_IK_ORIENTATION_WEIGHT,
                include_collision_penalty=False,
                finger_position=DRAWER_FINGER_OPEN,
                hand_frame_inset_m=DRAWER_OPEN_HAND_FRAME_INSET_M,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="grasp_drawer",
                fingers={self.drawer_hand: DRAWER_FINGER_OPEN},
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
            drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
            drawer_id = _joint_id(self.model, drawer_joint)
            target, rotation = self._drawer_contact_target(
                drawer_qpos=float(self.model.jnt_range[drawer_id, 0])
            )
            # Pull slightly beyond the handle pose at the drawer's open joint
            # limit. This margin compensates for IK error and contact
            # compliance so the physical drawer reaches its full opening.
            pull_overshoot = max(
                0.0,
                DRAWER_PULL_OVERSHOOT_M - self.trajectory.drawer_pull_shortfall,
            )
            target = target + pull_overshoot * self._drawer_robotward_axis()
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=rotation,
                seed_offset=8,
                reset_drawers=False,
                align_blue_axis=False,
                directed_axes=True,
                position_weight=DRAWER_OPEN_IK_POSITION_WEIGHT,
                orientation_weight=DRAWER_IK_ORIENTATION_WEIGHT,
                include_collision_penalty=False,
                finger_position=self.trajectory.drawer_grasp_position,
                hand_frame_inset_m=DRAWER_OPEN_HAND_FRAME_INSET_M,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="release_open_drawer",
                fingers={self.drawer_hand: self.trajectory.drawer_grasp_position},
            )
        elif self.phase == "release_open_drawer":
            assert self.drawer_hand is not None
            opening = self._drawer_opening()
            if opening < MIN_DRAWER_OPENING_M and self._task_failure(
                "drawer opening",
                f"drawer opened {opening:.4f} m; required at least {MIN_DRAWER_OPENING_M:.4f} m",
            ):
                return
            self._start_motion(
                {},
                after="retreat_open_drawer",
                duration_scale=0.4,
                fingers={self.drawer_hand: DRAWER_FINGER_OPEN},
            )
        elif self.phase == "retreat_open_drawer":
            assert self.drawer_hand is not None
            target, rotation = self.ik_targets["drawer_open"]
            target = (
                target
                + DRAWER_OPEN_RETREAT_EXTRA_DISTANCE_M * self._drawer_robotward_axis()
            )
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=rotation,
                seed_offset=11,
                reset_drawers=False,
                align_blue_axis=False,
                directed_axes=True,
                position_weight=DRAWER_OPEN_IK_POSITION_WEIGHT,
                orientation_weight=DRAWER_IK_ORIENTATION_WEIGHT,
                include_collision_penalty=False,
                finger_position=DRAWER_FINGER_OPEN,
                hand_frame_inset_m=DRAWER_OPEN_HAND_FRAME_INSET_M,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="move_to_a_for_pick",
                fingers={self.drawer_hand: DRAWER_FINGER_OPEN},
            )
        elif self.phase == "move_to_a_for_pick":
            if not self._workspace_transit_is_clear(Workspace.STORAGE):
                return
            self._advance_task_phase(TaskPhase.PICK_CUBE)
            targets = self._workspace_targets(Workspace.STORAGE)
            self.workspace = Workspace.STORAGE
            self._start_motion(
                {COMMON_JOINTS[0]: targets[COMMON_JOINTS[0]]},
                after="align_a_for_pick",
                fingers={"l": FINGER_OPEN, "r": FINGER_OPEN},
            )
        elif self.phase == "align_a_for_pick":
            targets = self._workspace_targets(Workspace.STORAGE)
            self._start_motion(
                {
                    COMMON_JOINTS[1]: targets[COMMON_JOINTS[1]],
                    COMMON_JOINTS[2]: targets[COMMON_JOINTS[2]],
                },
                after="cube_ik",
                fingers={"l": FINGER_OPEN, "r": FINGER_OPEN},
            )
        elif self.phase == "drop_cube":
            assert self.cube_hand is not None
            self._unlock_cube()
            _set_cube_gripper_friction(self.model, grasping=False)
            if self.cube_drop_assist and self.backend.config.debug:
                print(
                    f"Cube drawer-drop assist active: guiding into drawer "
                    f"{self.drawer_index}"
                )
            self._start_motion(
                {},
                after="settle_cube_in_drawer",
                duration_scale=0.6,
                fingers={self.cube_hand: FINGER_OPEN},
            )
        elif self.phase == "settle_cube_in_drawer":
            assert self.cube_hand is not None
            # Keep the open gripper stationary while gravity lets the cube
            # leave any marginal lip contact and settle on the drawer floor.
            self._start_motion(
                {},
                after="retreat_from_drawer",
                duration_scale=CUBE_DRAWER_SETTLE_DURATION_S / self.move_duration_s,
                fingers={self.cube_hand: FINGER_OPEN},
            )
        elif self.phase == "retreat_from_drawer":
            assert self.cube_hand is not None
            mujoco.mj_forward(self.model, self.data)
            if not _cube_center_inside_drawer(
                self.model, self.data, self.drawer_index
            ) and self._task_failure(
                "cube release",
                f"cube is not fully inside drawer {self.drawer_index}",
            ):
                return
            hand_position, hand_rotation = hand_pose(
                self.model,
                self.data,
                self.cube_hand,
                task_frame_inset_m=HAND_TASK_FRAME_INSET_M,
            )
            robotward_axis = self._drawer_robotward_axis()
            retreat_target = hand_position + CUBE_PLACE_RETREAT_DISTANCE_M * robotward_axis
            targets = self._solve_cube_placement(
                target=retreat_target,
                rotation=hand_rotation,
                seed_offset=11,
                finger_position=FINGER_OPEN,
                position_weight=CUBE_RETREAT_IK_POSITION_WEIGHT,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="choose_closing_hand",
                fingers={self.cube_hand: FINGER_OPEN},
            )
        elif self.phase == "choose_closing_hand":
            target, _ = self.ik_targets["drawer_close"]
            if _drawer_is_center(self.drawer_index):
                closing_rng = np.random.default_rng(
                    np.random.SeedSequence([self.seed, self.episode_seed, self.drawer_index, 4])
                )
                self.drawer_hand = ("l", "r")[int(closing_rng.integers(0, 2))]
            else:
                self.drawer_hand = _closest_hand(self.model, self.data, target)
            self._refresh_drawer_targets()
            self._start_motion(
                {}, after="close_drawer_ik", fingers={self.drawer_hand: FINGER_CLOSED}
            )
        elif self.phase == "close_drawer_ik":
            assert self.drawer_hand is not None
            target, _ = self.ik_targets["drawer_close"]
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=self._working_hand_rotation(self.drawer_hand),
                seed_offset=4,
                reset_drawers=False,
                align_blue_axis=False,
                directed_axes=True,
                position_weight=DRAWER_IK_POSITION_WEIGHT,
                orientation_weight=DRAWER_IK_ORIENTATION_WEIGHT,
                include_collision_penalty=False,
                finger_position=FINGER_CLOSED,
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
            drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
            drawer_id = _joint_id(self.model, drawer_joint)
            target, _ = self._drawer_contact_target()
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=self._working_hand_rotation(self.drawer_hand),
                seed_offset=9,
                reset_drawers=False,
                align_blue_axis=False,
                directed_axes=True,
                position_weight=DRAWER_IK_POSITION_WEIGHT,
                orientation_weight=DRAWER_IK_ORIENTATION_WEIGHT,
                include_collision_penalty=False,
                finger_position=FINGER_CLOSED,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="push_drawer_closed",
                fingers={self.drawer_hand: FINGER_CLOSED},
            )
        elif self.phase == "push_drawer_closed":
            assert self.drawer_hand is not None
            drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
            drawer_id = _joint_id(self.model, drawer_joint)
            target, _ = self._drawer_contact_target(
                drawer_qpos=float(self.model.jnt_range[drawer_id, 1])
            )
            targets = self._solve_to(
                side=self.drawer_hand,
                target=target,
                rotation=self._working_hand_rotation(self.drawer_hand),
                seed_offset=10,
                reset_drawers=False,
                align_blue_axis=False,
                directed_axes=True,
                position_weight=DRAWER_IK_POSITION_WEIGHT,
                orientation_weight=DRAWER_IK_ORIENTATION_WEIGHT,
                include_collision_penalty=False,
                finger_position=FINGER_CLOSED,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="release_closed_drawer",
                fingers={self.drawer_hand: FINGER_CLOSED},
            )
        elif self.phase == "release_closed_drawer":
            assert self.drawer_hand is not None
            opening = self._drawer_opening()
            if opening > MAX_DRAWER_CLOSED_OPENING_M and self._task_failure(
                "drawer closing",
                f"drawer remains open by {opening:.4f} m; permitted at most "
                f"{MAX_DRAWER_CLOSED_OPENING_M:.4f} m",
            ):
                return
            fingertip_position, fingertip_rotation = hand_pose(
                self.model,
                self.data,
                self.drawer_hand,
                task_frame_inset_m=0.0,
            )
            drawer_joint = f"base_link_base_drawer_{self.drawer_index}_joint"
            drawer_id = _joint_id(self.model, drawer_joint)
            closed_data = mujoco.MjData(self.model)
            closed_data.qpos[:] = self.data.qpos
            closed_data.qpos[int(self.model.jnt_qposadr[drawer_id])] = (
                self.model.jnt_range[drawer_id, 1]
            )
            mujoco.mj_forward(self.model, closed_data)
            closed_handle_position, _ = drawer_handle_pose(
                self.model,
                closed_data,
                self.drawer_index,
            )
            vertical_retreat = _closed_drawer_vertical_retreat(
                float(fingertip_position[2]),
                float(closed_handle_position[2]),
            )
            retreat_target = fingertip_position + np.array(
                [0.0, 0.0, vertical_retreat]
            )
            current_posture = {
                name: _joint_qpos(self.model, self.data, name)
                for name in SIDE_JOINTS[self.drawer_hand]
            }
            targets = self._solve_to(
                side=self.drawer_hand,
                target=retreat_target,
                rotation=fingertip_rotation,
                seed_offset=12,
                reset_drawers=False,
                align_blue_axis=True,
                directed_axes=True,
                position_weight=DRAWER_RETREAT_IK_POSITION_WEIGHT,
                orientation_weight=CUBE_PLACE_IK_ORIENTATION_WEIGHT,
                posture_reference=current_posture,
                protect_drawers=True,
                protect_other_hand=True,
                finger_position=FINGER_CLOSED,
                hand_frame_inset_m=0.0,
            )
            if targets is None:
                return
            self._start_motion(
                targets,
                after="closed_drawer_hand_rest",
                fingers={self.drawer_hand: FINGER_CLOSED},
            )
        elif self.phase == "closed_drawer_hand_rest":
            assert self.drawer_hand is not None
            self._start_motion(
                self._transit_targets(),
                after="move_to_a_final",
                duration_scale=0.6,
                fingers={self.drawer_hand: FINGER_CLOSED},
            )
        elif self.phase == "move_to_a_final":
            if not self._workspace_transit_is_clear(Workspace.STORAGE):
                return
            self._advance_task_phase(TaskPhase.RETURN_TO_STORAGE)
            targets = self._workspace_targets(Workspace.STORAGE)
            self.workspace = Workspace.STORAGE
            self._start_motion(
                {COMMON_JOINTS[0]: targets[COMMON_JOINTS[0]]},
                after="align_a_final",
                fingers={"l": FINGER_CLOSED, "r": FINGER_CLOSED},
            )
        elif self.phase == "align_a_final":
            targets = self._workspace_targets(Workspace.STORAGE)
            self._start_motion(
                {
                    COMMON_JOINTS[1]: targets[COMMON_JOINTS[1]],
                    COMMON_JOINTS[2]: targets[COMMON_JOINTS[2]],
                },
                after="final_return",
                fingers={"l": FINGER_CLOSED, "r": FINGER_CLOSED},
            )
        elif self.phase == "final_return":
            self._start_motion(
                self._initial_view_targets(),
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
        if not self.done:
            self._check_handoff_hand_clearance()
        self._refresh_drawer_targets()
        if self.motion is not None and self.motion.complete:
            self.phase = self.motion.after
            self.motion = None
        while self.motion is None and not self.done:
            self._dispatch_phase()
        if not self.done and self._should_start_perturbation():
            self._begin_perturbation()
        if self.cube_drop_assist and self.phase in {
            "drop_cube",
            "settle_cube_in_drawer",
        }:
            self._stabilize_cube_drawer_drop()
        executed_action = self._current_action() if self.done else self.motion.next_action()
        if self.phase == "apply_perturbation" and self.perturbation_resume_target is not None:
            self._recording_action = self.perturbation_resume_target.copy()
        else:
            self._recording_action = executed_action.copy()
        self._recording_task_phase = self.task_phase
        self._recording_next_task_phase = self._next_task_phase_target()
        self._update_active_ik_debug_frames()
        return executed_action
