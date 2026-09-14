"""Backend-neutral discrete workspaces for the drawer task."""

from __future__ import annotations

from enum import IntEnum, StrEnum

DRAWER_COUNT = 9
CNC_ACTION_NAMES = ("cnc_x", "cnc_y", "head_z")


class Workspace(StrEnum):
    STORAGE = "A"
    DRAWERS = "B"


class TaskPhase(IntEnum):
    START_AT_STORAGE = 1
    OPEN_DRAWER = 2
    PICK_CUBE = 3
    PLACE_AND_CLOSE = 4
    RETURN_TO_STORAGE = 5


def drawer_grid_offsets(drawer: int, *, spacing: float) -> tuple[float, float]:
    """Return column and row offsets for a row-major 3x3 drawer grid."""
    if not 1 <= drawer <= DRAWER_COUNT:
        raise ValueError(f"Drawer must be between 1 and {DRAWER_COUNT}.")
    column = (drawer - 1) % 3
    row = (drawer - 1) // 3
    return (-spacing, 0.0, spacing)[column], (spacing, 0.0, -spacing)[row]


def workspace_for_phase(phase: int | TaskPhase) -> Workspace:
    phase = TaskPhase(phase)
    if phase in {
        TaskPhase.START_AT_STORAGE,
        TaskPhase.PICK_CUBE,
        TaskPhase.RETURN_TO_STORAGE,
    }:
        return Workspace.STORAGE
    return Workspace.DRAWERS
