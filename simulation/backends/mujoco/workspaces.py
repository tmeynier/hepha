"""Map shared discrete workspaces onto this MuJoCo model's CNC joints."""

from __future__ import annotations

import mujoco
from hepha_lerobot.workspaces import Workspace

CNC_X_JOINT = "base_link_base_cnc_x_joint"
CNC_Y_JOINT = "cnc_x_link_cnc_x_cnc_y_joint"
CNC_Z_JOINT = "cnc_y_link_cnc_y_head_joint"
CNC_JOINTS = (CNC_X_JOINT, CNC_Y_JOINT, CNC_Z_JOINT)
DRAWER_CENTER_CNC_Y_M = 0.04285938
WORKSPACE_CNC_Y_OFFSET_M = 0.0
WORKSPACE_CNC_Y_M = DRAWER_CENTER_CNC_Y_M + WORKSPACE_CNC_Y_OFFSET_M


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"MuJoCo joint not found: {name}")
    return joint_id


def _range(model: mujoco.MjModel, name: str) -> tuple[float, float]:
    minimum, maximum = model.jnt_range[_joint_id(model, name)]
    return float(minimum), float(maximum)


def resolve_mujoco_workspace(
    model: mujoco.MjModel,
    workspace: Workspace | str,
    *,
    drawer: int | None = None,
) -> dict[str, float]:
    """Resolve A or drawer-dependent B directly from the model's CNC limits."""
    workspace = Workspace(workspace)
    x_min, x_max = _range(model, CNC_X_JOINT)
    y_min, y_max = _range(model, CNC_Y_JOINT)
    z_min, z_max = _range(model, CNC_Z_JOINT)

    if workspace is Workspace.STORAGE:
        if drawer is not None:
            raise ValueError("A does not accept a drawer number.")
        # The model's joint axes are expressed in its local frame: CNC-X is the
        # depth travel between the storage bin and cabinet, while CNC-Y and Z
        # align the head with drawer columns and rows.  The centered CNC-Y
        # value puts the head directly opposite the middle drawer column.
        targets = {
            CNC_X_JOINT: x_max,
            CNC_Y_JOINT: WORKSPACE_CNC_Y_M,
            CNC_Z_JOINT: z_max,
        }
    else:
        if drawer is None:
            raise ValueError("B requires drawer 1 through 9.")
        targets = {
            # For now B stays at the same centered lateral and vertical pose as
            # A. Only the depth axis moves from the storage side to the drawers.
            CNC_X_JOINT: x_min,
            CNC_Y_JOINT: WORKSPACE_CNC_Y_M,
            CNC_Z_JOINT: z_max,
        }

    ranges = {
        CNC_X_JOINT: (x_min, x_max),
        CNC_Y_JOINT: (y_min, y_max),
        CNC_Z_JOINT: (z_min, z_max),
    }
    for name, target in targets.items():
        minimum, maximum = ranges[name]
        if not minimum <= target <= maximum:
            raise ValueError(
                f"Workspace {workspace} targets {name}={target:.3f}, outside "
                f"[{minimum:.3f}, {maximum:.3f}]."
            )
    return targets


def set_workspace_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    workspace: Workspace | str,
    *,
    drawer: int | None = None,
) -> dict[str, float]:
    targets = resolve_mujoco_workspace(model, workspace, drawer=drawer)
    for name, target in targets.items():
        joint_id = _joint_id(model, name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = target
    return targets
