"""Sparse task rewards and contact-onset penalties for MuJoCo AWR rollouts."""

from __future__ import annotations

from dataclasses import dataclass, fields

import mujoco
from hepha_lerobot.evaluation.task_sweep import (
    DRAWER_CLOSED_THRESHOLD_M,
    DRAWER_OPEN_THRESHOLD_M,
    _drawer_opening,
)

from simulation.backends.mujoco import MujocoBackend
from simulation.backends.mujoco.episode import cube_position
from simulation.backends.mujoco.ik import CUBE_LIFT_CHECK_M, _cube_center_inside_drawer


@dataclass
class RewardBreakdown:
    step_cost: float = -0.001
    drawer_opened: float = 0.0
    cube_grasped: float = 0.0
    cube_entered_drawer: float = 0.0
    final_success: float = 0.0
    external_collision: float = 0.0
    selected_drawer_collision: float = 0.0
    floor_collision: float = 0.0
    cube_fall: float = 0.0
    arm_collision: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(getattr(self, item.name) for item in fields(self)))


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, kind, object_id) or ""


def _arm_side(body_name: str) -> str | None:
    if body_name.endswith("_l_link") or "_l_finger" in body_name:
        return "l"
    if body_name.endswith("_r_link") or "_r_finger" in body_name:
        return "r"
    return None


def _is_robot(body_name: str) -> bool:
    return _arm_side(body_name) is not None or body_name in {
        "cnc_x_link",
        "cnc_y_link",
        "head_link",
    }


def _is_cube(body_name: str) -> bool:
    return body_name == "cube_link"


def _is_grasping_link(body_name: str) -> bool:
    return "hand_" in body_name or "finger" in body_name


class AWRRewardTracker:
    """Evaluate rewards after each action-induced physics transition.

    Collision costs are emitted once when a body-pair contact begins. Keeping a
    contact for ten frames therefore has the same cost as keeping it for one.
    """

    def __init__(
        self,
        backend: MujocoBackend,
        *,
        drawer_index: int,
        stable_grasp_frames: int = 5,
    ) -> None:
        self.backend = backend
        self.drawer_index = drawer_index
        self.stable_grasp_frames = stable_grasp_frames
        self.initial_cube_z = float(cube_position(backend.model, backend.data)[2])
        self.drawer_opened = False
        self.cube_grasped = False
        self.cube_entered_drawer = False
        self._stable_grasp_count = 0
        self._cube_airborne = False
        self._active_contacts = self._classified_contacts()

    def _classify_contact(self, geom1: int, geom2: int) -> tuple[str, int, int] | None:
        model = self.backend.model
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        body_name1 = _name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
        body_name2 = _name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
        geom_name1 = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
        geom_name2 = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2)

        side1, side2 = _arm_side(body_name1), _arm_side(body_name2)
        if side1 is not None and side2 is not None and side1 != side2:
            return ("arm_collision", min(body1, body2), max(body1, body2))

        robot_first = _is_robot(body_name1)
        robot_second = _is_robot(body_name2)
        if robot_first == robot_second:
            return None
        if robot_second:
            body1, body2 = body2, body1
            body_name1, body_name2 = body_name2, body_name1
            geom_name1, geom_name2 = geom_name2, geom_name1

        if geom_name2 == "floor":
            category = "floor_collision"
        elif body_name2 == f"drawer_{self.drawer_index}_link":
            handle_prefix = f"drawer_{self.drawer_index}_link_collision_box_"
            if geom_name2.startswith(handle_prefix) and geom_name2.removeprefix(
                handle_prefix
            ).startswith(("05", "06", "07", "08")):
                return None
            category = "selected_drawer_collision"
        elif _is_cube(body_name2) and _is_grasping_link(body_name1):
            return None
        else:
            category = "external_collision"
        return (category, min(body1, body2), max(body1, body2))

    def _classified_contacts(self) -> set[tuple[str, int, int]]:
        contacts: set[tuple[str, int, int]] = set()
        for index in range(self.backend.data.ncon):
            contact = self.backend.data.contact[index]
            classified = self._classify_contact(int(contact.geom1), int(contact.geom2))
            if classified is not None:
                contacts.add(classified)
        return contacts

    def _cube_touches_floor(self) -> bool:
        model, data = self.backend.model, self.backend.data
        for index in range(data.ncon):
            contact = data.contact[index]
            names = {
                _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
            }
            bodies = {
                _name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[int(contact.geom1)]),
                ),
                _name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[int(contact.geom2)]),
                ),
            }
            if "floor" in names and "cube_link" in bodies:
                return True
        return False

    def _cube_securely_grasped(self) -> bool:
        """Require cube contact on both opposing pads of either gripper."""
        model, data = self.backend.model, self.backend.data
        contacts: set[str] = set()
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = _name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1])
            )
            body2 = _name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2])
            )
            if body1 == "cube_link":
                contacts.add(_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2))
            elif body2 == "cube_link":
                contacts.add(_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1))
        for side in ("l", "r"):
            fixed_pad = any(name.startswith(f"hand_{side}_link_") for name in contacts)
            moving_pad = any(name.startswith(f"finger_{side}_link_") for name in contacts)
            if fixed_pad and moving_pad:
                return True
        return False

    def evaluate(self, *, final_step: bool = False) -> RewardBreakdown:
        breakdown = RewardBreakdown()
        opening = _drawer_opening(self.backend, self.drawer_index)
        if not self.drawer_opened and opening >= DRAWER_OPEN_THRESHOLD_M:
            self.drawer_opened = True
            breakdown.drawer_opened = 1.0

        lift = float(cube_position(self.backend.model, self.backend.data)[2]) - self.initial_cube_z
        self._stable_grasp_count = (
            self._stable_grasp_count + 1
            if lift >= CUBE_LIFT_CHECK_M and self._cube_securely_grasped()
            else 0
        )
        if not self.cube_grasped and self._stable_grasp_count >= self.stable_grasp_frames:
            self.cube_grasped = True
            breakdown.cube_grasped = 3.0

        inside = _cube_center_inside_drawer(
            self.backend.model, self.backend.data, self.drawer_index
        )
        if not self.cube_entered_drawer and inside:
            self.cube_entered_drawer = True
            breakdown.cube_entered_drawer = 2.0

        current_contacts = self._classified_contacts()
        for category, _, _ in current_contacts - self._active_contacts:
            setattr(
                breakdown,
                category,
                getattr(breakdown, category)
                + {
                    "external_collision": -2.0,
                    "selected_drawer_collision": -0.5,
                    "floor_collision": -1.0,
                    "arm_collision": -2.0,
                }[category],
            )
        self._active_contacts = current_contacts

        touches_floor = self._cube_touches_floor()
        if lift >= CUBE_LIFT_CHECK_M and not touches_floor:
            self._cube_airborne = True
        elif touches_floor and self._cube_airborne:
            breakdown.cube_fall = -3.0
            self._cube_airborne = False

        if final_step and inside and opening <= DRAWER_CLOSED_THRESHOLD_M:
            breakdown.final_success = 5.0
        return breakdown
