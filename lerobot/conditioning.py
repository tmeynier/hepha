"""Task-conditioning helpers shared by recording and evaluation."""

from __future__ import annotations

import numpy as np
from lerobot.utils.constants import OBS_ENV_STATE

DRAWER_COUNT = 9
DRAWER_CONDITION_NAMES = tuple(
    f"requested_drawer_{drawer_index}" for drawer_index in range(1, DRAWER_COUNT + 1)
)
TASK_PHASE_COUNT = 5
TASK_PHASE_CONDITION_NAMES = tuple(
    f"current_task_phase_{phase}" for phase in range(1, TASK_PHASE_COUNT + 1)
)
NEXT_TASK_PHASE = "next_task_phase"
DEFAULT_TASK = (
    "Open drawer {drawer_index}, pick up the cube, place it inside that drawer, "
    "and close the drawer."
)


def validate_drawer_index(drawer_index: int) -> int:
    """Return a valid one-based drawer index."""

    drawer_index = int(drawer_index)
    if not 1 <= drawer_index <= DRAWER_COUNT:
        raise ValueError(
            f"Drawer index must be between 1 and {DRAWER_COUNT}, got {drawer_index}"
        )
    return drawer_index


def drawer_condition(drawer_index: int) -> np.ndarray:
    """Encode a requested drawer as a stable nine-way one-hot condition."""

    drawer_index = validate_drawer_index(drawer_index)
    condition = np.zeros(DRAWER_COUNT, dtype=np.float32)
    condition[drawer_index - 1] = 1.0
    return condition


def drawer_condition_values(drawer_index: int) -> dict[str, float]:
    """Return named scalar values consumed by LeRobot's frame builder."""

    return dict(
        zip(
            DRAWER_CONDITION_NAMES,
            drawer_condition(drawer_index).tolist(),
            strict=True,
        )
    )


def validate_task_phase(task_phase: int) -> int:
    """Return a valid one-based semantic task phase."""

    task_phase = int(task_phase)
    if not 1 <= task_phase <= TASK_PHASE_COUNT:
        raise ValueError(
            f"Task phase must be between 1 and {TASK_PHASE_COUNT}, got {task_phase}"
        )
    return task_phase


def task_phase_condition(task_phase: int) -> np.ndarray:
    """Encode the current semantic task phase as a five-way one-hot condition."""

    task_phase = validate_task_phase(task_phase)
    condition = np.zeros(TASK_PHASE_COUNT, dtype=np.float32)
    condition[task_phase - 1] = 1.0
    return condition


def task_phase_condition_values(task_phase: int) -> dict[str, float]:
    """Return named phase values consumed by LeRobot's frame builder."""

    return dict(
        zip(
            TASK_PHASE_CONDITION_NAMES,
            task_phase_condition(task_phase).tolist(),
            strict=True,
        )
    )


def drawer_condition_feature() -> dict[str, dict[str, object]]:
    """Return the native LeRobot environment-state dataset feature."""

    return {
        OBS_ENV_STATE: {
            "dtype": "float32",
            "shape": (DRAWER_COUNT,),
            "names": list(DRAWER_CONDITION_NAMES),
        }
    }


def task_phase_condition_feature() -> dict[str, dict[str, object]]:
    """Return the current-phase portion of the environment-state feature."""

    return {
        OBS_ENV_STATE: {
            "dtype": "float32",
            "shape": (TASK_PHASE_COUNT,),
            "names": list(TASK_PHASE_CONDITION_NAMES),
        }
    }


def next_task_phase_feature() -> dict[str, dict[str, object]]:
    """Return the categorical next-phase supervision feature.

    Values are stored as zero-based class indices so they can be passed directly
    to a five-class cross-entropy loss by a future phase-aware policy.
    """

    return {
        NEXT_TASK_PHASE: {
            "dtype": "int64",
            "shape": (1,),
            "names": ["next_task_phase_index"],
        }
    }


def drawer_task(task_template: str, drawer_index: int) -> str:
    """Make the human-readable task identify the same requested drawer."""

    drawer_index = validate_drawer_index(drawer_index)
    if "{drawer_index}" in task_template:
        return task_template.replace("{drawer_index}", str(drawer_index))
    return f"{task_template.rstrip()} Requested drawer: {drawer_index}."
