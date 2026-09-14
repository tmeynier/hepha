"""Shared physical task milestones for MuJoCo closed-loop evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from simulation.backends.mujoco import MujocoBackend

DRAWER_OPEN_THRESHOLD_M = 0.040
DRAWER_CLOSED_THRESHOLD_M = 0.005


@dataclass
class TaskMilestones:
    drawer_opened: bool = False
    cube_grasped: bool = False
    cube_in_drawer: bool = False
    drawer_closed_after_insertion: bool = False

    def update(
        self,
        *,
        drawer_opening_m: float,
        stable_cube_grasp: bool,
        cube_inside_drawer: bool,
    ) -> None:
        self.drawer_opened |= drawer_opening_m >= DRAWER_OPEN_THRESHOLD_M
        self.cube_grasped |= stable_cube_grasp
        self.cube_in_drawer |= cube_inside_drawer
        self.drawer_closed_after_insertion |= (
            self.drawer_opened
            and self.cube_in_drawer
            and drawer_opening_m <= DRAWER_CLOSED_THRESHOLD_M
        )


def drawer_opening(backend: MujocoBackend, drawer_index: int) -> float:
    """Return the selected drawer's distance from its closed joint limit."""

    joint_name = f"base_link_base_drawer_{drawer_index}_joint"
    joint_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )
    if joint_id < 0:
        raise RuntimeError(f"MuJoCo joint not found: {joint_name}")
    qpos_id = int(backend.model.jnt_qposadr[joint_id])
    return max(
        0.0,
        float(backend.model.jnt_range[joint_id, 1] - backend.data.qpos[qpos_id]),
    )
